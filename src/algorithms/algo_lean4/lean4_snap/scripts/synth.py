#!/usr/bin/env python3
"""Synthetic social graph: generate / validate / certify.

generate  — invoke the standalone Lean generator (`lake exe generate`) from
            the distilled genspec (or, with --bundle DIR, the bundle's own
            rebuilt generator and genspec); write meta.json with full
            provenance and DECLARED generation gaps (fields_not_modelled).
validate  — re-ingest the synthetic GraphML through the PRODUCTION ingest +
            dbt pipeline into a scratch DuckDB, verify the planted motifs
            are recalled (hub, broker, social butterfly; recall must be
            1.0), check the structural invariants, GOF-check the synthetic
            degree mix against the genspec (TV), check the global closure
            against the released level, and run the Lean deciders on the
            re-exported observations. Exit codes: 0 pass / 2 soft tolerance
            / 1 hard (schema, planted-pattern or Lean disagreement).
certify   — emit GenCert.lean asserting (by decide, kernel-checked) that the
            planted plan structures satisfy the formal motif definitions and
            that sampled rendered edges are canonical (src < dst); compile
            it and record certificate.json.
"""
from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN = ROOT / "lean"
GENSPEC = LEAN / "out" / "genspec.json"
CHECKS_BIN = LEAN / ".lake" / "build" / "bin" / "checks"
DEG_BINS = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
            "deg_33_64", "deg_65p"]

sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec, spec_version  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lean_env() -> dict:
    """PATH with the shared Lean store's elan first (LEAN_HOME policy),
    falling back to a user-local elan."""
    lean_home = Path(os.environ.get("LEAN_HOME", "/opt/data/lean"))
    env = dict(os.environ)
    prefix = []
    if lean_home.is_dir():
        env.setdefault("ELAN_HOME", str(lean_home / "elan"))
        env.setdefault("XDG_CACHE_HOME", str(lean_home / "cache"))
        prefix.append(str(lean_home / "elan" / "bin"))
    prefix.append(str(Path.home() / ".elan" / "bin"))
    env["PATH"] = ":".join(prefix + [env.get("PATH", "")])
    return env


def cmd_generate(args) -> int:
    base = Path(args.bundle).resolve() if args.bundle else ROOT
    lean_dir = base / "lean"
    genspec = lean_dir / "out" / "genspec.json"
    gen_bin = lean_dir / ".lake" / "build" / "bin" / "generate"
    if not genspec.exists():
        hint = ("run `python3 python/distill.py` inside the bundle first"
                if args.bundle else "run `make distill` first")
        print(f"ERROR: {genspec} missing — {hint}", file=sys.stderr)
        return 1
    if not gen_bin.exists():
        hint = ("run `./build_from_source.sh` inside the bundle first"
                if args.bundle else "run `make lean-build` first")
        print(f"ERROR: {gen_bin} missing — {hint}", file=sys.stderr)
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(gen_bin), str(genspec), str(args.nodes), str(out)],
        cwd=str(lean_dir))
    if r.returncode != 0:
        return r.returncode
    spec = json.loads(genspec.read_text())
    meta = {
        "generator": "lake exe generate (Lean; genspec-driven; every draw "
                      "through the verified sampler sampleCellPMF; edges only "
                      "through mkEdge + the edgesCanonical output check)",
        "created_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   __import__("time").gmtime()),
        "genspec_sha256": sha(genspec),
        "dp_release_sha256": spec["source"]["dp_release_sha256"],
        "spec_version": str(spec.get("spec_version", spec_version())),
        "n_nodes": args.nodes,
        "bundle": str(base) if args.bundle else None,
        "released_fact_sources": {
            "degree": "dp_release.json cells",
            "community_sizes": "aux community_sizes",
            "mixing": "aux mixing (rank-pair edge routing)",
            "closure": "aux closure (triadic-closure pass probabilities)",
            "richness": "aux richness",
            "category": "aux category",
            "circles": "aux circles",
            **({"affiliation_sizes": "aux affil_sizes (member-weighted "
                                     "education/work key-group size bins)",
                "affiliation_densities": "artifact affil_cohesion_stats "
                                         "(suppression-thresholded integer "
                                         "percent; no formal guarantee)"}
               if "affiliation" in spec else {}),
        },
        "validated_size_window": [VALIDATED_MIN_NODES, VALIDATED_MAX_NODES],
        "size_note": ("fidelity tolerances validated for n in the window; "
                      "smaller graphs still verify planted structures, larger "
                      "ones keep closure in band with a ~0.02 margin "
                      "(rank-labelled mixing rows extrapolate)"),
        "assumptions": {
            # the genspec is authoritative for the declared gaps (it knows
            # whether the affiliation section crossed the gate)
            "fields_not_modelled": spec["assumptions"]["fields_not_modelled"],
            "combination_rules": [
                "edge partner = verified draw from the DP mixing row of the "
                "node's community rank, node uniform inside the partner "
                "community",
                "triangles = closureTrials sampled wedges per node, closed "
                "with the RELEASED per-degree-class probability; stub "
                "budgets pre-shrunk by the expected closure contribution",
                *(["affiliation key-groups = uniform distinct node samples "
                   "inside one community, sized by the RELEASED "
                   "member-weighted histogram, deficit-wired to the "
                   "RELEASED density percent (shared instance id past the "
                   "public pool)"]
                  if "affiliation" in spec else []),
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed FFI, "
                                    "consistent with non-reproducible DP draws)",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/synth_graph.graphml + plan.json + meta.json")
    return 0


# affiliation validation tolerances (public, soft). Cells are judged by how
# many synthetic groups back them: >= MIN_GROUPS -> TOL; SPARSE_MIN..MIN_GROUPS-1
# -> the wider SPARSE tol; fewer -> recorded but listed as unjudged.
# node-count window the fidelity tolerances are validated on (mirrors the Lean
# validatedMinNodes/validatedMaxNodes). Below it the degree-mix and group-count
# checks cannot be met by construction (a planted hub of degree >= 150 touches
# most of a few-hundred-node graph; a dozen groups carry sampling noise), so they
# are recorded as unjudged instead of failing.
VALIDATED_MIN_NODES = 500
VALIDATED_MAX_NODES = 10000
AFFIL_MIN_GROUPS = 20
AFFIL_SPARSE_MIN = 5
AFFIL_DENSITY_TOL = 0.30            # cap of the scaled tolerance below
AFFIL_DENSITY_TOL_FLOOR = 0.15      # floor of the scaled tolerance below
AFFIL_DENSITY_TOL_REL = 0.6         # tol = clamp(REL * released, FLOOR, CAP)
AFFIL_DENSITY_TOL_SPARSE = 0.40
AFFIL_COUNT_TV_TOL = 0.20          # synth group-count mix vs the genspec mix
AFFIL_COVERAGE_MIN_EXPECTED = 3.0  # expected groups above which a bin may not be empty


def density_tol(released: float) -> float:
    """Per-cell density tolerance, proportional to the released density so a
    0.22 miss on a 0.24 target fails while a 0.30 miss on a 0.78 target does not.
    Measured on healthy generator runs: a flat 0.25 flagged 4 of 12 runs, this
    rule 1 of 12, and it still flags the overshoot it was written to catch."""
    return min(AFFIL_DENSITY_TOL, max(AFFIL_DENSITY_TOL_FLOOR,
                                      AFFIL_DENSITY_TOL_REL * released))


def check_affiliation(gs: dict, plan: dict, con, notes: dict, soft: list,
                      judged: bool = True) -> None:
    """Judge the re-ingested synthetic key-groups against the genspec.

    Per category: (1) the group-COUNT mix over the public size bins vs the
    genspec size weights (TV), (2) no bin the spec expects to be populated is
    empty, (3) per-bin mean density vs the released density percent, with the
    tolerance chosen by how many groups back the cell. Every cell's evidence
    goes into notes["affil_cells"]; cells too sparse to judge are listed in
    notes["affil_unjudged"] instead of being silently skipped."""
    block = gs["affiliation"]
    bins = [c["name"] for c in block["size_rows"][0]["cells"]]
    want_d = {(r["given"], c["name"]): c["w"] / block["density_den"]
              for r in block["density_rows"] for c in r["cells"]}
    want_w = {(r["given"], c["name"]): c["w"]
              for r in block["size_rows"] for c in r["cells"]}
    got = {(cat, b): (int(n), int(dpct) / 100) for cat, b, n, dpct in con.execute(
        "select category, size_bin, n_groups, density_pct "
        "from mart_affil_release").fetchall()}
    notes["affil_plan"] = plan.get("affiliation", {})
    cells, unjudged = {}, []

    def flag(msg: str) -> None:
        if judged:
            soft.append(msg)
        else:
            notes.setdefault("affil_unjudged_small_n", []).append(msg)

    for cat in block["categories"]:
        w_cat = sum(want_w.get((cat, b), 0) for b in bins)
        n_cat = sum(got.get((cat, b), (0, 0))[0] for b in bins)
        if w_cat == 0:
            continue
        if n_cat == 0:
            flag(f"affil {cat}: spec expects groups, synthetic graph has none")
            continue
        tv = 0.5 * sum(abs(want_w.get((cat, b), 0) / w_cat
                           - got.get((cat, b), (0, 0))[0] / n_cat) for b in bins)
        notes[f"affil_group_count_tv_{cat}"] = round(tv, 4)
        # sampling allowance: with few groups the observed mix is noisy by
        # construction (expected TV ~ sqrt((k-1)/(2 pi n))), so the tolerance
        # widens as the group count shrinks and stays AFFIL_COUNT_TV_TOL when
        # there are plenty. Three standard errors, not two: nine categories are
        # checked per run, and over 12 runs the worst small category (languages,
        # 83 groups) reached 0.155 with one 0.21 tail draw elsewhere; the member-
        # weight defect this check exists for measures 0.49.
        count_tol = max(AFFIL_COUNT_TV_TOL,
                        3 * math.sqrt((len(bins) - 1) / (2 * math.pi * n_cat)))
        notes[f"affil_group_count_tol_{cat}"] = round(count_tol, 4)
        if tv > count_tol:
            flag(f"affil {cat}: group-count mix TV {tv:.2f} > {count_tol:.2f} "
                 f"vs the genspec size weights")
        for b in bins:
            n, got_p = got.get((cat, b), (0, None))
            expected = want_w.get((cat, b), 0) / w_cat * n_cat
            want_p = want_d.get((cat, b))
            cells[f"{cat}/{b}"] = {
                "n_groups": n, "expected_groups": round(expected, 1),
                "synth_density": got_p, "released_density": want_p}
            if expected >= AFFIL_COVERAGE_MIN_EXPECTED and n == 0:
                flag(f"affil {cat}/{b}: ~{expected:.0f} groups expected, none generated")
            if want_p is None or n == 0:
                continue
            if n < AFFIL_SPARSE_MIN:
                unjudged.append(f"{cat}/{b} (n={n})")
                continue
            tol = (density_tol(want_p) if n >= AFFIL_MIN_GROUPS
                   else AFFIL_DENSITY_TOL_SPARSE)
            if abs(got_p - want_p) > tol:
                flag(f"affil density {cat}/{b}: synth {got_p:.2f} vs "
                     f"released {want_p:.2f} (n={n}, tol {tol})")
    notes["affil_cells"] = cells
    notes["affil_unjudged"] = unjudged
    n_shared, n_edges = con.execute(
        "select sum(case when total_shared > 0 then 1 else 0 end), "
        "count(*) from mart_edge_shared").fetchone()
    notes["affil_shared_edge_frac"] = round((n_shared or 0) / max(1, n_edges), 4)


# closure is a stochastic property of the GENERATOR, so it is judged over
# replicate runs, not one sample: at least MIN_FRACTION of runs inside the
# public band AND the mean at least MIN_MARGIN above the band's lower edge.
REPLICATE_MIN_FRACTION = 0.85
REPLICATE_MIN_MARGIN = 0.02


def closure_verdict(values: list[float], released: float, tol: float) -> dict:
    """Pass/fail on the distribution of per-node mean closure over replicate
    generator runs against the band released +- tol."""
    n = len(values)
    lower, upper = released - tol, released + tol
    mean = sum(values) / n
    n_in = sum(1 for v in values if lower <= v <= upper)
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    margin = mean - lower
    return {
        "n_runs": n, "mean": round(mean, 4), "min": round(min(values), 4),
        "max": round(max(values), 4), "std": round(std, 4),
        "band": [round(lower, 4), round(upper, 4)],
        "n_in_band": n_in, "fraction_in_band": round(n_in / n, 4),
        "margin_over_lower_edge": round(margin, 4),
        "rule": f"fraction_in_band >= {REPLICATE_MIN_FRACTION} and margin >= "
                f"{REPLICATE_MIN_MARGIN}",
        "passed": bool(n_in / n >= REPLICATE_MIN_FRACTION
                       and margin >= REPLICATE_MIN_MARGIN),
    }


# affiliation cohesion is judged over the replicate runs as well: the MEAN of
# the per-run mean |synthetic - released| density over cells with >= MIN_GROUPS
# groups. Healthy generator: 0.046-0.058; the single-window generator it replaced
# measured 0.095-0.119 and a too-tight window 0.24.
REPLICATE_MAX_DENSITY_ERR = 0.08


def density_verdict(errors: list[float]) -> dict:
    mean = sum(errors) / len(errors)
    return {"runs": [round(e, 4) for e in errors], "mean_abs_density_error": round(mean, 4),
            "max_allowed": REPLICATE_MAX_DENSITY_ERR,
            "passed": bool(mean <= REPLICATE_MAX_DENSITY_ERR)}


def affil_density_error(gml: Path, gs: dict) -> float | None:
    """Mean |synthetic - released| group density over (category, bin) cells
    with >= AFFIL_MIN_GROUPS synthetic groups; None if the spec has no
    affiliation block or no cell is judgeable."""
    if "affiliation" not in gs:
        return None
    import networkx as nx
    import realism_audit as ra
    want = {(r["given"], c["name"]): c["w"] / gs["affiliation"]["density_den"]
            for r in gs["affiliation"]["density_rows"] for c in r["cells"]}
    g = nx.read_graphml(gml)
    stats = ra.affil_stats(g, ra.node_keys(g))
    errs = [abs(cell["mean_density"] - want[(cat, b)])
            for cat, bins in stats.items() if cat != "shared_key_edge_frac"
            for b, cell in bins.items()
            if cell["n_groups"] >= AFFIL_MIN_GROUPS and (cat, b) in want]
    return sum(errs) / len(errs) if errs else None


def per_node_closure(gml: Path) -> float:
    """Mean over nodes with degree >= 2 of tri/wedges (the statistic the
    released closure grid matches: per-node equal-weighted)."""
    import networkx as nx
    g = nx.read_graphml(gml)
    c = nx.clustering(g)
    vals = [c[v] for v in g if g.degree(v) >= 2]
    return sum(vals) / len(vals)


def cmd_replicates(args) -> int:
    import shutil
    base = Path(args.bundle).resolve() if args.bundle else ROOT
    genspec = base / "lean" / "out" / "genspec.json"
    gen_bin = base / "lean" / ".lake" / "build" / "bin" / "generate"
    if not genspec.exists() or not gen_bin.exists():
        print("ERROR: genspec or generator missing — run the synth flow first",
              file=sys.stderr)
        return 1
    gs = json.loads(genspec.read_text())
    closed = sum(c["w"] for r in gs["closure"]["rows"] for c in r["cells"]
                 if c["name"] == "closed")
    total = sum(c["w"] for r in gs["closure"]["rows"] for c in r["cells"])
    released = closed / total if total else 0.0
    tol = int(load_spec()["synth_closure_tol_pct"]) / 100.0
    out = Path(args.synth).resolve()
    scratch = out / "replicates"
    values, dens_errs = [], []
    for _ in range(args.runs):
        shutil.rmtree(scratch, ignore_errors=True)
        r = subprocess.run([str(gen_bin), str(genspec), str(args.nodes), str(scratch)],
                           cwd=str(base / "lean"), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"replicates: generator failed: {r.stderr[-300:]}", file=sys.stderr)
            return 1
        values.append(per_node_closure(scratch / "synth_graph.graphml"))
        d = affil_density_error(scratch / "synth_graph.graphml", gs)
        if d is not None:
            dens_errs.append(d)
    shutil.rmtree(scratch, ignore_errors=True)
    verdict = closure_verdict(values, released, tol)
    if dens_errs:
        verdict["affiliation_density"] = density_verdict(dens_errs)
        verdict["closure_passed"] = verdict["passed"]
        verdict["passed"] = bool(verdict["passed"] and verdict["affiliation_density"]["passed"])
    verdict["released_closure"] = round(released, 4)
    verdict["values"] = [round(v, 4) for v in values]
    (out / "replicates.json").write_text(json.dumps(verdict, indent=2))
    tag = "PASS" if verdict["passed"] else "soft tolerance"
    dens = verdict.get("affiliation_density")
    print(f"replicates: {tag} — closure mean {verdict['mean']} (min {verdict['min']}, "
          f"{verdict['n_in_band']}/{verdict['n_runs']} in band {verdict['band']}, "
          f"margin {verdict['margin_over_lower_edge']})"
          + (f"; affiliation density mean |err| {dens['mean_abs_density_error']} "
             f"(max {dens['max_allowed']})" if dens else ""))
    return 0 if verdict["passed"] else 2


def cmd_validate(args) -> int:
    synth = Path(args.synth).resolve()
    gml = synth / "synth_graph.graphml"
    plan_p = synth / "plan.json"
    if not gml.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    if not GENSPEC.exists():
        print("ERROR: run `make distill` first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    scratch_dir = synth / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    scratch_db = scratch_dir / "enclave.duckdb"
    if scratch_db.exists():
        scratch_db.unlink()
    hard, soft, notes = [], [], {}
    # re-ingest through the PRODUCTION ingest script
    r = subprocess.run(
        [sys.executable, str(ROOT / "python" / "ingest_graphml.py"),
         "--graphml", str(gml), "--out", str(scratch_dir / "ingest")],
        capture_output=True, text=True)
    if r.returncode != 0:
        hard.append(f"re-ingestion failed:\n{r.stderr[-400:]}")
    else:
        # PRODUCTION dbt models against the scratch db
        env = dict(os.environ,
                   GRAPH_DB=str(scratch_db),
                   GRAPH_INGEST_DIR=str(scratch_dir / "ingest"),
                   DBT_PROFILES_DIR=".")
        r = subprocess.run(["dbt", "build", "--exclude", "ml_graph_scores"],
                           cwd=str(ROOT / "dbt"), env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            tail = "\n".join(r.stdout.splitlines()[-15:])
            hard.append(f"dbt re-ingestion failed:\n{tail}")
    if not hard:
        import duckdb
        con = duckdb.connect(str(scratch_db), read_only=True)
        planted = plan["planted"]
        # planted motifs recalled by the production feat_nodes flags
        for motif, flag in (("hub", "is_hub"), ("broker", "is_broker"),
                            ("butterfly", "is_butterfly")):
            node = int(planted[motif]["node"])
            got = con.execute(
                f"select {flag} from feat_nodes where node_id = ?",
                [node]).fetchone()
            ok = bool(got and got[0])
            notes[f"planted_{motif}_recall"] = 1.0 if ok else 0.0
            if not ok:
                hard.append(f"planted {motif} (node {node}) not recalled")
        # structural: canonical edges, no self-loops (q_data_integrity ran in
        # the dbt build; re-assert the core ones here)
        n_bad = con.execute(
            "select count(*) from stg_edges where src >= dst").fetchone()[0]
        if n_bad:
            hard.append(f"{n_bad} non-canonical edges")
        # GOF (soft): synthetic degree mix vs the released genspec weights (TV)
        gs = json.loads(GENSPEC.read_text())

        def tv(want: dict[str, int], got: dict[str, int], grid: list[str]) -> float:
            tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
            return 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                             for c in grid)

        want_deg = {c["name"]: c["w"] for c in gs["degree"]["cells"]}
        got_deg = dict(con.execute(
            "select degree_class, count(*) from feat_nodes group by 1").fetchall())
        notes["degree_tv_vs_genspec"] = round(tv(want_deg, got_deg, DEG_BINS), 4)
        n_nodes = int(plan["n_nodes"])
        in_window = VALIDATED_MIN_NODES <= n_nodes <= VALIDATED_MAX_NODES
        notes["n_nodes"] = n_nodes
        notes["validated_size_window"] = [VALIDATED_MIN_NODES, VALIDATED_MAX_NODES]
        small = n_nodes < VALIDATED_MIN_NODES
        if not in_window:
            notes["size_note"] = (
                f"n={n_nodes} is outside the validated window "
                f"{VALIDATED_MIN_NODES}..{VALIDATED_MAX_NODES}")
        if notes["degree_tv_vs_genspec"] > 0.25:
            msg = (f"degree_tv_vs_genspec = "
                   f"{notes['degree_tv_vs_genspec']:.3f} > 0.25")
            if small:
                notes.setdefault("unjudged_small_n", []).append(msg)
            else:
                soft.append(msg)
        # per-node mean closure vs the released level (soft, public tolerance).
        # The released closure grid is per-node EQUAL-WEIGHTED (each node's
        # clipped wedge vector counts once), so the matching synth statistic is
        # the mean per-node closure rate, not edge-weighted transitivity.
        rows = con.execute(
            "select tri_count, wedge_count from feat_nodes where wedge_count > 0"
        ).fetchall()
        synth_closure = (sum(t / w for t, w in rows) / len(rows)) if rows else 0.0
        rel_closed = sum(c["w"] for r in gs["closure"]["rows"]
                         for c in r["cells"] if c["name"] == "closed")
        rel_total = sum(c["w"] for r in gs["closure"]["rows"] for c in r["cells"])
        rel_closure = rel_closed / rel_total if rel_total else 0.0
        tol = int(load_spec()["synth_closure_tol_pct"]) / 100.0
        notes["synth_global_closure"] = round(synth_closure, 4)
        notes["released_closure"] = round(rel_closure, 4)
        if abs(synth_closure - rel_closure) > tol:
            soft.append(f"|synth closure {synth_closure:.3f} - released "
                        f"{rel_closure:.3f}| > {tol}")
        if "affiliation" in gs:
            check_affiliation(gs, plan, con, notes, soft, judged=not small)
        con.close()
        # Lean deciders on the re-ingested observations
        leandir = synth / "leancheck"
        (leandir / "out").mkdir(parents=True, exist_ok=True)
        from export_lean_data import export as export_lean
        export_lean(scratch_db, leandir / "data")
        rc = subprocess.run([str(CHECKS_BIN)], cwd=str(leandir),
                            capture_output=True, text=True)
        if rc.returncode != 0:
            hard.append(f"lean checks failed: {rc.stderr[-400:] or rc.stdout[-400:]}")
        else:
            res = json.loads((leandir / "out" / "lean_results.json").read_text())
            n_dis = res["agreement"]["n_disagree"]
            notes["lean_disagreements"] = n_dis
            if n_dis:
                hard.append(f"SQL<->Lean disagreement on synth: {n_dis} "
                            f"(samples: {res['disagreement_samples'][:3]})")
    verdict = {"hard_failures": hard, "soft_failures": soft, "notes": notes,
               "graphml_sha256": sha(gml)}
    (synth / "validation.json").write_text(json.dumps(verdict, indent=2))
    if hard:
        print("validate: HARD FAIL\n - " + "\n - ".join(hard), file=sys.stderr)
        return 1
    if soft:
        print("validate: soft tolerance\n - " + "\n - ".join(soft))
        return 2
    print(f"validate: PASS ({json.dumps(notes)})")
    return 0


def cmd_certify(args) -> int:
    synth = Path(args.synth).resolve()
    plan = json.loads((synth / "plan.json").read_text())
    planted = plan["planted"]
    hub_deg = int(planted["hub"]["degree"])
    br_deg = int(planted["broker"]["degree"])
    br_tri = int(planted["broker"]["triangles"])
    bf_circ = int(planted["butterfly"]["circles"])
    # sample K rendered edges for canonicality facts
    edges = []
    pat = re.compile(r'<edge source="(\d+)" target="(\d+)"')
    with open(synth / "synth_graph.graphml") as f:
        for line in f:
            m = pat.search(line)
            if m:
                edges.append((int(m.group(1)), int(m.group(2))))
                if len(edges) >= int(args.n):
                    break
    pairs = ", ".join(f"({a}, {b})" for a, b in edges)
    cert = f"""/- AUTO-GENERATED by scripts/synth.py certify — kernel-checked certificate
that the emitted plan's structures satisfy the formal spec. -/
import EnclaveGraph.Motifs

open EnclaveGraph

theorem planted_is_hub : IsHub {hub_deg} := by decide

theorem planted_is_broker : IsBroker {br_deg} {br_tri} := by decide

theorem planted_is_butterfly : IsButterfly {bf_circ} := by decide

/-- sampled rendered edges (src, dst) -/
def sampledEdges : List (Nat × Nat) := [{pairs}]

theorem sampled_edges_canonical :
    sampledEdges.all (fun e => decide (EdgeOK e.1 e.2)) = true := by decide
"""
    (LEAN / "GenCert.lean").write_text(cert)
    r = subprocess.run(["lake", "env", "lean", "GenCert.lean"], cwd=str(LEAN),
                       capture_output=True, text=True, env=_lean_env())
    ok = r.returncode == 0
    (synth / "certificate.json").write_text(json.dumps({
        "proved": ok,
        "kernel_only": True,
        "theorems": ["planted_is_hub", "planted_is_broker",
                     "planted_is_butterfly", "sampled_edges_canonical"],
        "n_edges_checked": len(edges),
        "axioms": "decide-based (propext only); no sorry, no native_decide",
        "stderr": r.stderr[-400:] if not ok else "",
    }, indent=2))
    print(f"certify: {'PROVED' if ok else 'FAILED'} -> {synth}/certificate.json")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--nodes", type=int, default=4039)
    g.add_argument("--out", default="synth")
    g.add_argument("--bundle", default=None,
                   help="use DIR/lean/.lake/build/bin/generate and "
                        "DIR/lean/out/genspec.json instead of the repo's")
    v = sub.add_parser("validate")
    v.add_argument("--synth", default="synth")
    rp = sub.add_parser("replicates")
    rp.add_argument("--nodes", type=int, default=4039)
    rp.add_argument("--synth", default="synth")
    rp.add_argument("--runs", type=int, default=8)
    rp.add_argument("--bundle", default=None)
    c = sub.add_parser("certify")
    c.add_argument("--synth", default="synth")
    c.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    return {"generate": cmd_generate, "validate": cmd_validate,
            "replicates": cmd_replicates, "certify": cmd_certify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

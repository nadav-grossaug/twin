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
        },
        "assumptions": {
            "fields_not_modelled": [
                "degree-community dependence",
                "feature identity structure beyond category level "
                "(instance ids are uniform in a fixed public pool)",
                "circle rosters (only membership counts are modelled)",
                "inter-ego overlap (each synthetic node belongs to one "
                "pseudo-ego scope)",
                "degree assortativity",
            ],
            "combination_rules": [
                "edge partner = verified draw from the DP mixing row of the "
                "node's community rank, node uniform inside the partner "
                "community",
                "triangles = closureTrials sampled wedges per node, closed "
                "with the RELEASED per-degree-class probability; stub "
                "budgets pre-shrunk by the expected closure contribution",
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed FFI, "
                                    "consistent with non-reproducible DP draws)",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/synth_graph.graphml + plan.json + meta.json")
    return 0


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
        if notes["degree_tv_vs_genspec"] > 0.25:
            soft.append(f"degree_tv_vs_genspec = "
                        f"{notes['degree_tv_vs_genspec']:.3f} > 0.25")
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
    c = sub.add_parser("certify")
    c.add_argument("--synth", default="synth")
    c.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    return {"generate": cmd_generate, "validate": cmd_validate,
            "certify": cmd_certify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

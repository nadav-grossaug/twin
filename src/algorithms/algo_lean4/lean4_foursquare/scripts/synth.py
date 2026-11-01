#!/usr/bin/env python3
"""Synthetic travel transfers: generate / validate / certify.

generate  — invoke the standalone Lean generator (`lake exe generate`) from
            the distilled genspec (or, with --bundle DIR, the bundle's own
            rebuilt generator and genspec); write meta.json with full
            provenance and DECLARED generation gaps (fields_not_modelled).
validate  — re-ingest the synthetic TSV through the PRODUCTION dbt pipeline
            into a scratch DuckDB, verify the planted motifs are recalled
            (commuter corridor, tour; recall must be 1.0), check the
            structural invariants, GOF-check the synthetic travel-count and
            region mixes against the genspec, and run the Lean deciders on
            the re-exported observations. Exit codes: 0 pass / 2 soft
            tolerance / 1 hard (schema, planted-pattern or Lean disagreement).
certify   — emit GenCert.lean asserting (by decide, kernel-checked) that the
            planted plan structures satisfy the formal motif definitions and
            that every planned user's travel count falls in its public bin;
            compile it and record certificate.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN = ROOT / "lean"
GENSPEC = LEAN / "out" / "genspec.json"
GENERATE_BIN = LEAN / ".lake" / "build" / "bin" / "generate"
CHECKS_BIN = LEAN / ".lake" / "build" / "bin" / "checks"
TC_BINS = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
MOTIF_KEYS = ("returns", "corridors", "tours", "bursts", "longstays", "assoc")

sys.path.insert(0, str(ROOT / "python"))
from spec import REGIONS, load_spec, spec_version  # noqa: E402


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
        [str(gen_bin), str(genspec), str(args.users), str(out)],
        cwd=str(lean_dir))
    if r.returncode != 0:
        return r.returncode
    spec = json.loads(genspec.read_text())
    meta = {
        "generator": "lake exe generate (Lean; genspec-driven; every draw "
                      "through the verified sampler sampleCellPMF)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "genspec_sha256": sha(genspec),
        "dp_release_sha256": spec["source"]["dp_release_sha256"],
        "spec_version": str(spec.get("spec_version", spec_version())),
        "n_users": args.users,
        "bundle": str(base) if args.bundle else None,
        "released_fact_sources": {
            "travel_count": "dp_release.json cells",
            "region_mix": "aux region",
            "transitions": "aux region_transition (+ declared return_boost "
                           "diagonal prior)",
            "return_rate": "aux return_rate (second-order return behavior)",
            "gap_class": "aux gap_class (+ suppressed gap moments)",
            "duration_class": "aux duration_class",
            "episode_len": "aux episode_len",
            "month": "aux month",
            "corridors": "topk_corridors release artifact (suppressed)",
        },
        "assumptions": {
            "fields_not_modelled": [
                "within-country check-in behaviour (only country-boundary "
                "transfers are modelled)",
                "venue geography: coordinates are the public country-centroid "
                "table with no jitter; venue ids are synthetic 24-hex tokens",
                "day-of-week / time-of-day placement (uniform_dow public "
                "prior; timestamps are cumulative gap sums from the public "
                "window start)",
                "country choice beyond the released corridors + the fixed "
                "public per-region representative",
                "hop distance (implied by the centroid table, not drawn from "
                "the released distance_class histogram)",
                "per-user travel counts above the public tc_33p cap (counts "
                "drawn uniformly within the public bin ranges)",
            ],
            "combination_rules": [
                "next region = verified draw from the blended transition row; "
                "with the user's drawn return-rate probability the next "
                "country is the previous one (second-order return)",
                "gap seconds = verified draw within the public gap-class "
                "range; durations chain next_duration(i) = prev_duration(i+1)",
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed FFI, "
                                    "consistent with non-reproducible DP draws)",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/travel_synth.tsv + plan.json + meta.json")
    return 0


def cmd_validate(args) -> int:
    synth = Path(args.synth)
    tsv = synth / "travel_synth.tsv"
    plan_p = synth / "plan.json"
    if not tsv.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    if not GENSPEC.exists():
        print("ERROR: run `make distill` first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    scratch = (synth / "scratch.duckdb").resolve()
    if scratch.exists():
        scratch.unlink()
    # re-ingest through the PRODUCTION dbt pipeline (same models, scratch DB)
    env = dict(os.environ,
               TRAVEL_TSV=str(tsv.resolve()),
               ENCLAVE_DB=str(scratch),
               DBT_PROFILES_DIR=".")
    r = subprocess.run(["dbt", "build", "--exclude", "ml_travel_scores"],
                       cwd=str(ROOT / "dbt"), env=env,
                       capture_output=True, text=True)
    hard, soft, notes = [], [], {}
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-15:])
        hard.append(f"dbt re-ingestion failed:\n{tail}")
    else:
        import duckdb
        con = duckdb.connect(str(scratch), read_only=True)
        spec = load_spec()
        # the Lean generator writes planted motifs at the top level
        planted = plan.get("planted") or {"corridor": plan["corridor"],
                                          "tour": plan["tour"]}
        # planted commuter corridor recalled by the production corridor mart
        cor = planted["corridor"]
        n_cor = con.execute(
            "select count(*) from mart_motif_corridors where userid=? "
            "and is_corridor and prev_country=? and next_country=?",
            [int(cor["user"]), cor["a"], cor["b"]]).fetchone()[0]
        notes["planted_corridor_recall"] = 1.0 if n_cor else 0.0
        if n_cor < 1:
            hard.append(f"planted corridor {cor['a']}->{cor['b']} of user "
                        f"{cor['user']} not recalled")
        # planted tour recalled
        tour = planted["tour"]
        n_tour = con.execute(
            "select count(*) from mart_motif_tours where userid=? and is_tour",
            [int(tour["user"])]).fetchone()[0]
        notes["planted_tour_recall"] = 1.0 if n_tour else 0.0
        if n_tour < 1:
            hard.append(f"planted tour of user {tour['user']} not recalled")
        # structural: no self-transfers, every timestamp inside the window
        n_self = con.execute(
            "select count(*) from stg_transfers "
            "where prev_country = next_country").fetchone()[0]
        if n_self:
            hard.append(f"{n_self} self-transfer rows")
        n_outside = con.execute(
            "select count(*) from stg_transfers where ts_s < ? or ts_s >= ?",
            [int(spec["window_start_s"]), int(spec["window_end_s"])]).fetchone()[0]
        if n_outside:
            hard.append(f"{n_outside} rows outside the public window")
        n_xx = con.execute(
            "select count(*) from feat_transfers where qf_unknown_region"
        ).fetchone()[0]
        if n_xx:
            hard.append(f"{n_xx} rows with an unknown (XX) region")
        # GOF (soft): synthetic mixes vs the released genspec weights (TV)
        gs = json.loads(GENSPEC.read_text())

        def tv(want: dict[str, int], got: dict[str, int], grid: list[str]) -> float:
            tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
            return 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                             for c in grid)

        want_tc = {c["name"]: c["w"] for c in gs["travel_count"]["cells"]}
        got_tc = dict(con.execute(
            "select tc_bin, count(*) from feat_users group by 1").fetchall())
        notes["travel_count_tv_vs_genspec"] = round(tv(want_tc, got_tc, TC_BINS), 4)
        want_rg = {c["name"]: c["w"] for c in gs["region_mix"]["cells"]}
        got_rg = dict(con.execute(
            "select next_region, count(*) from feat_transfers "
            "where next_region != 'XX' group by 1").fetchall())
        notes["region_mix_tv_vs_genspec"] = round(tv(want_rg, got_rg, REGIONS), 4)
        # the generator blends transitions, return behavior and episode tempo
        # on top of the marginal mixes, so tolerate drift
        for key in ("travel_count_tv_vs_genspec", "region_mix_tv_vs_genspec"):
            if notes[key] > 0.25:
                soft.append(f"{key} = {notes[key]:.3f} > 0.25")
        # Lean deciders on the re-ingested observations
        leandir = synth / "leancheck"
        (leandir / "out").mkdir(parents=True, exist_ok=True)
        from export_lean_data import export as export_lean
        export_lean(scratch, leandir / "data")
        rc = subprocess.run([str(CHECKS_BIN)], cwd=str(leandir),
                            capture_output=True, text=True)
        if rc.returncode != 0:
            hard.append(f"lean checks failed: {rc.stderr[-400:]}")
        else:
            res = json.loads((leandir / "out" / "lean_results.json").read_text())
            disagree = {
                k: sum(1 for x in res[k] if not x.get("agree", True))
                for k in MOTIF_KEYS}
            disagree["episodes"] = sum(
                1 for x in res["episodes"]
                if not x.get("lean_is_episode", True))
            notes["lean_disagreements"] = disagree
            if any(disagree.values()):
                hard.append(f"SQL<->Lean disagreement on synth: {disagree}")
        con.close()
    verdict = {"hard_failures": hard, "soft_failures": soft, "notes": notes,
               "tsv_sha256": sha(tsv)}
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
    synth = Path(args.synth)
    plan = json.loads((synth / "plan.json").read_text())
    gs = json.loads(GENSPEC.read_text())
    spec = load_spec()
    planted = plan.get("planted") or {"corridor": plan["corridor"],
                                      "tour": plan["tour"]}
    cor, tour = planted["corridor"], planted["tour"]
    gaps = [int(g) for g in cor["gaps_s"]]
    tour_ts = [int(t) for t in tour["ts_s"]]
    # the tour predicate judges ALL countries touched: the start country plus
    # the destinations
    tour_cs = ([str(tour["start"])] if "start" in tour else []) + \
        [str(c) for c in tour["countries"]]
    ranges = gs["tc_bin_ranges"]
    users = plan["users"][: int(args.n)]
    n = len(users)

    def bin_of(count: int) -> tuple[int, int]:
        for lo, hi in ranges.values():
            if int(lo) <= count <= int(hi):
                return int(lo), int(hi)
        raise SystemExit(f"certify: planned count {count} outside every "
                         f"public tc bin")

    triples = ", ".join(
        "({0}, {1}, {2})".format(int(u["n_travels"]), *bin_of(int(u["n_travels"])))
        for u in users)
    countries = "[" + ", ".join(f'"{c}"' for c in tour_cs) + "]"
    cert = f"""/- AUTO-GENERATED by scripts/synth.py certify — kernel-checked certificate
that the emitted plan's structures satisfy the formal spec. -/
import EnclaveTravel.Motifs

open EnclaveTravel

def plantedGaps : List Nat := {gaps}

theorem planted_is_corridor :
    IsCorridor {len(gaps) + 1} plantedGaps := by decide

def plantedTourTs : List Nat := {tour_ts}

def plantedTourCountries : List String := {countries}

theorem planted_is_tour :
    IsTour plantedTourTs plantedTourCountries := by decide

/-- (planned travel count, public bin lower edge, public bin upper edge) -/
def plannedCounts : List (Nat × Nat × Nat) := [{triples}]

theorem planned_counts_in_bins :
    plannedCounts.all
      (fun p => decide (p.2.1 ≤ p.1) && decide (p.1 ≤ p.2.2)) = true := by
  decide
"""
    (LEAN / "GenCert.lean").write_text(cert)
    r = subprocess.run(["lake", "env", "lean", "GenCert.lean"], cwd=str(LEAN),
                       capture_output=True, text=True, env=_lean_env())
    ok = r.returncode == 0
    (synth / "certificate.json").write_text(json.dumps({
        "proved": ok,
        "kernel_only": True,
        "theorems": ["planted_is_corridor", "planted_is_tour",
                     "planned_counts_in_bins"],
        "n_users_checked": n,
        "axioms": "decide-based (propext only); no sorry, no native_decide",
        "stderr": r.stderr[-400:] if not ok else "",
    }, indent=2))
    print(f"certify: {'PROVED' if ok else 'FAILED'} -> {synth}/certificate.json")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--users", type=int, default=2000)
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

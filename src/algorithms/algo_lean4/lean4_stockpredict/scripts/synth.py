#!/usr/bin/env python3
"""Synthetic stock series: generate / validate / certify.

generate  — invoke the standalone Lean generator (`lake exe generate`) from
            the distilled genspec; write meta.json with full provenance and
            DECLARED generation gaps (fields_not_modelled).
validate  — re-ingest the synthetic TSV through the PRODUCTION dbt pipeline
            into a scratch DuckDB, verify the planted motifs are recalled
            (shock day, down run, drawdown), run the Lean deciders on the
            re-ingested observations, and GOF-check the synthetic class mix
            against the genspec. Exit codes: 0 pass / 2 soft tolerance /
            1 hard (schema, planted-pattern or Lean disagreement).
certify   — emit GenCert.lean asserting (by decide, kernel-checked) that the
            planted plan structures satisfy the formal motif definitions and
            that every planned day's class matches classOfRet; compile it and
            record certificate.json.
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
CLASSES = ["crash", "down", "flat", "up", "surge"]

sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec, spec_version  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def class_of(bp: int, flat: int, big: int) -> str:
    if bp <= -big:
        return "crash"
    if bp < -flat:
        return "down"
    if bp <= flat:
        return "flat"
    if bp < big:
        return "up"
    return "surge"


def cmd_generate(args) -> int:
    if not GENSPEC.exists():
        print("ERROR: run `make distill` first", file=sys.stderr)
        return 1
    if not GENERATE_BIN.exists():
        print("ERROR: run `make lean-build` first", file=sys.stderr)
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(GENERATE_BIN), str(GENSPEC), str(args.tickers), str(args.days),
         str(out)], cwd=str(LEAN))
    if r.returncode != 0:
        return r.returncode
    spec = json.loads(GENSPEC.read_text())
    meta = {
        "generator": "lake exe generate (Lean; genspec-driven; every draw "
                      "through the verified sampler sampleCellPMF)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "genspec_sha256": sha(GENSPEC),
        "dp_release_sha256": spec["source"]["dp_release_sha256"],
        "spec_version": spec_version(),
        "n_tickers": args.tickers,
        "n_days": args.days,
        "released_fact_sources": {
            "classes": "dp_release.json cells",
            "transitions": "aux class_transition (+ declared clustering prior)",
            "dow": "aux dow_class",
            "vol_transitions": "aux vol_transition",
            "mag_bins": "aux mag_bins",
            "comove_lambda_pct": "aux comove (post-processed, 2s-1 rule)",
        },
        "assumptions": {
            "fields_not_modelled": [
                "exact cross-ticker correlation matrix (only the DP co-movement "
                "level drives the market coupling)",
                "market holidays (every weekday from the anchor is a trading day)",
                "absolute price levels (all series start at the public base 100.0)",
                "trading volume (not present in the source data)",
            ],
            "combination_rules": [
                "class row = (transitions+1) * (dow+1) elementwise (declared)",
                "magnitude = DP bins ∩ class bins, public vol-boost table",
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed FFI, "
                                    "consistent with non-reproducible DP draws)",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/stock_synth.tsv + plan.json + meta.json")
    return 0


def cmd_validate(args) -> int:
    synth = Path(args.synth)
    tsv = synth / "stock_synth.tsv"
    plan_p = synth / "plan.json"
    if not tsv.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    scratch = (synth / "scratch.duckdb").resolve()
    if scratch.exists():
        scratch.unlink()
    # re-ingest through the PRODUCTION dbt pipeline (same models, scratch DB)
    env = dict(os.environ,
               STOCK_TSV=str(tsv.resolve()),
               ENCLAVE_DB=str(scratch),
               DBT_PROFILES_DIR=".")
    r = subprocess.run(["dbt", "build", "--exclude", "ml_series_scores"],
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
        planted = plan["planted"][0]
        tick = planted["ticker"]
        shock_idx = int(planted["shock_idx"])
        # planted shock day recalled by the production shock mart
        n_shock = con.execute(
            "select count(*) from mart_motif_shock where ticker=? and is_shock "
            "and day_idx=?", [tick, shock_idx + 0]).fetchone()[0]
        # day_idx in feat_returns counts from 0 at the first price row; the
        # planted index is the return index, which equals day_idx
        if n_shock < 1:
            hard.append(f"planted shock at idx {shock_idx} of {tick} not recalled")
        notes["planted_shock_recall"] = 1.0 if n_shock else 0.0
        # planted down run recalled
        run_len = con.execute(
            "select max(len_days) from mart_motif_runs where ticker=? and is_run "
            "and sgn=-1", [tick]).fetchone()[0] or 0
        if run_len < int(planted["run_days"]):
            hard.append(f"planted down run not recalled (max len {run_len})")
        notes["planted_run_len"] = int(run_len)
        # planted drawdown recalled
        has_dd = con.execute(
            "select has_drawdown from mart_motif_drawdown where ticker=?",
            [tick]).fetchone()[0]
        if not has_dd:
            hard.append("planted drawdown not recalled")
        notes["planted_drawdown"] = bool(has_dd)
        # structural: first price 100, positive prices, weekday calendar
        bad = con.execute(
            "select count(*) from stg_prices where price <= 0 or qf_weekend"
        ).fetchone()[0]
        if bad:
            hard.append(f"{bad} non-positive or weekend rows")
        first_bad = con.execute(
            "select count(*) from (select ticker, arg_min(price, d) p0 "
            "from stg_prices group by ticker) where abs(p0 - 100.0) > 1e-6"
        ).fetchone()[0]
        if first_bad:
            hard.append(f"{first_bad} tickers do not start at 100.0")
        # GOF (soft): synthetic class mix vs genspec class weights (TV)
        gs = json.loads(GENSPEC.read_text())
        want = {c["name"]: c["w"] for c in gs["classes"]["cells"]}
        got = dict(con.execute(
            "select ret_class, count(*) from feat_returns "
            "where ret_class is not null group by 1").fetchall())
        tw, tg = sum(want.values()), sum(got.values())
        tv = 0.5 * sum(abs(want.get(c, 0) / max(tw, 1) - got.get(c, 0) / max(tg, 1))
                       for c in CLASSES)
        notes["class_mix_tv_vs_genspec"] = round(tv, 4)
        # the generator blends transitions, dow seasonality, vol states and the
        # market coupling on top of the marginal mix, so tolerate drift
        if tv > 0.25:
            soft.append(f"class-mix TV vs genspec = {tv:.3f} > 0.25")
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
                for k in ("runs", "shocks", "drawdowns", "vrecoveries",
                          "volbursts", "assoc")}
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
    spec = load_spec()
    flat, big = int(spec["ret_flat_bp"]), int(spec["ret_big_bp"])
    planted = plan["planted"][0]
    tick = next(t for t in plan["tickers"] if t["ticker"] == planted["ticker"])
    rets = tick["rets_bp"]
    classes = tick["classes"]
    start = int(planted["start_idx"]) - 1   # rets index 0 = day 1
    seg = rets[start:start + int(planted["run_days"])]
    n = min(int(args.n), len(rets))
    pairs = ", ".join(f"({r}, {CLASSES.index(c)})"
                      for r, c in list(zip(rets, classes))[:n])
    cert = f"""/- AUTO-GENERATED by scripts/synth.py certify — kernel-checked certificate
that the emitted plan's structures satisfy the formal spec. -/
import EnclaveSeries.Motifs

open EnclaveSeries

def plantedSeg : List Int := {seg}

theorem planted_is_run :
    IsRun {int(spec["run_min_days"])} {flat} plantedSeg := by decide

theorem planted_has_shock :
    HasShock {int(spec["shock_bp"])} plantedSeg := by decide

def plannedDays : List (Int × Nat) := [{pairs}]

theorem planned_classes_valid :
    plannedDays.all (fun p => classOfRet {flat} {big} p.1 == p.2) = true := by
  decide
"""
    (LEAN / "GenCert.lean").write_text(cert)
    r = subprocess.run(["lake", "env", "lean", "GenCert.lean"], cwd=str(LEAN),
                       capture_output=True, text=True,
                       env=dict(os.environ, PATH=f"{Path.home()}/.elan/bin:"
                                + os.environ.get("PATH", "")))
    ok = r.returncode == 0
    (synth / "certificate.json").write_text(json.dumps({
        "proved": ok,
        "kernel_only": True,
        "theorems": ["planted_is_run", "planted_has_shock",
                     "planned_classes_valid"],
        "n_days_checked": n,
        "axioms": "decide-based (propext only); no sorry, no native_decide",
        "stderr": r.stderr[-400:] if not ok else "",
    }, indent=2))
    print(f"certify: {'PROVED' if ok else 'FAILED'} -> {synth}/certificate.json")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--tickers", type=int, default=8)
    g.add_argument("--days", type=int, default=168)
    g.add_argument("--out", default="synth")
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

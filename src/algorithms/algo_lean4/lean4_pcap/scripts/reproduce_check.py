#!/usr/bin/env python3
"""Reproducibility check: same pinned inputs + configuration -> same results.

Phase 1  rebuild-frozen: rebuild the dbt warehouse into a scratch DuckDB with
         the ACTIVE model artifact. All non-ml tables must match the real DB
         exactly (row count + content hash); ml_flow_scores float columns must
         match within ABS_TOL and pred_* string columns exactly.
Phase 2  retrain-determinism: train the minimodels twice on CPU (seeded,
         deterministic algorithms) into two scratch artifact roots. Both runs
         must produce the SAME candidate version id (a content hash of
         data + hparams + code: a mismatch means environment leaked into the
         hparams), metrics within ABS_TOL, and state_dict tensors within
         ABS_TOL. Whether the CPU candidate equals the registry-active version
         is recorded but NOT asserted (the active artifact may be MPS-trained,
         and the device is part of the hparams).

Declared exception: the DP outputs under lean/out are non-reproducible by
design (fresh verified randomness per release) and are not compared.

Output: artifacts/repro/report.json; exit nonzero on failure. Scratch DuckDB
files and scratch artifact roots are deleted afterwards (they hold host-level
data derived from the capture).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import artifacts  # noqa: E402
from run_manifest import table_fingerprints  # noqa: E402

ABS_TOL = 1e-6
CLEAN_DB = ROOT / "artifacts" / "enclave.duckdb"
REPRO_DIR = ROOT / "artifacts" / "repro"
WORK = REPRO_DIR / "work"
ML_TABLE = "ml_flow_scores"


def run(cmd: list[str], cwd: Path, env: dict) -> None:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n"
                         f"{r.stdout[-3000:]}\n{r.stderr[-2000:]}")


def rebuild_frozen(scratch_db: Path) -> None:
    """dbt build into a scratch DB using the active (frozen) artifact."""
    env = {**os.environ,
           "ENCLAVE_DB": str(scratch_db),
           "FLOW_ARTIFACT_ROOT": str(ROOT / "models_out"),
           "DBT_PROFILES_DIR": "."}
    run(["dbt", "build", "--exclude", ML_TABLE], ROOT / "dbt", env)
    run(["dbt", "build", "--select", ML_TABLE], ROOT / "dbt", env)


def compare_ml_scores(db_a: Path, db_b: Path) -> dict:
    """Float columns by max-abs-diff; string columns exactly (NULLs equal)."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_a}' AS a (READ_ONLY)")
    con.execute(f"ATTACH '{db_b}' AS b (READ_ONLY)")
    cols = con.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_catalog = 'a' and table_name = ? order by ordinal_position",
        [ML_TABLE]).fetchall()
    n_a = con.execute(f"select count(*) from a.main.{ML_TABLE}").fetchone()[0]
    n_b = con.execute(f"select count(*) from b.main.{ML_TABLE}").fetchone()[0]
    n_join = con.execute(
        f"select count(*) from a.main.{ML_TABLE} x join b.main.{ML_TABLE} y "
        "using (flow_id)").fetchone()[0]
    out = {"rows_a": n_a, "rows_b": n_b, "rows_joined": n_join,
           "float_max_abs_diff": {}, "null_pattern_mismatches": {},
           "string_mismatches": {}}
    join = (f"from a.main.{ML_TABLE} x join b.main.{ML_TABLE} y "
            "on x.flow_id = y.flow_id")
    for c, d in cols:
        if c == "flow_id":
            continue
        if d in ("DOUBLE", "FLOAT", "REAL"):
            diff, nulls = con.execute(
                f'select max(abs(x."{c}" - y."{c}")), '
                f'sum(((x."{c}" is null) <> (y."{c}" is null))::int) {join}'
            ).fetchone()
            out["float_max_abs_diff"][c] = float(diff or 0.0)
            out["null_pattern_mismatches"][c] = int(nulls or 0)
        else:
            bad = con.execute(
                f'select count(*) {join} '
                f'where x."{c}" is distinct from y."{c}"').fetchone()[0]
            out["string_mismatches"][c] = int(bad)
    out["passed"] = (n_a == n_b == n_join
                     and all(v <= ABS_TOL for v in out["float_max_abs_diff"].values())
                     and all(v == 0 for v in out["null_pattern_mismatches"].values())
                     and all(v == 0 for v in out["string_mismatches"].values()))
    con.close()
    return out


def phase1() -> dict:
    scratch_db = WORK / "rebuild.duckdb"
    scratch_db.unlink(missing_ok=True)
    rebuild_frozen(scratch_db)
    with duckdb.connect(str(CLEAN_DB), read_only=True) as c:
        fa = table_fingerprints(c)
    with duckdb.connect(str(scratch_db), read_only=True) as c:
        fb = table_fingerprints(c)
    exact = {t: fa[t] == fb.get(t) for t in fa if t != ML_TABLE}
    missing = sorted(set(fa) - set(fb)) + sorted(set(fb) - set(fa))
    ml = compare_ml_scores(CLEAN_DB, scratch_db)
    return {
        "non_ml_tables_identical": exact,
        "table_set_mismatch": missing,
        "ml_flow_scores": ml,
        "passed": all(exact.values()) and not missing and ml["passed"],
    }


def retrain(artifact_root: Path) -> tuple[str, Path, dict]:
    """One deterministic CPU training run; returns (version, vdir, meta)."""
    env = {**os.environ,
           "FLOW_DEVICE": "cpu",
           "FLOW_ARTIFACT_ROOT": str(artifact_root)}
    env.pop("FLOW_RETRAIN", None)
    run([sys.executable, "python/train_minimodels.py"], ROOT, env)
    reg = json.loads((artifact_root / "registry.json").read_text())
    version = reg["minimodels"]["active"]
    vdir = artifact_root / "minimodels" / version
    return version, vdir, json.loads((vdir / "meta.json").read_text())


def numeric_leaves(obj, prefix: str = "") -> dict:
    """Flatten numeric leaves of a nested dict (bools and strings skipped)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def state_dict_max_diff(fa: Path, fb: Path) -> float:
    import torch
    sa = torch.load(fa, map_location="cpu", weights_only=True)
    sb = torch.load(fb, map_location="cpu", weights_only=True)
    if sorted(sa) != sorted(sb):
        return float("inf")
    return max((sa[k].double() - sb[k].double()).abs().max().item()
               for k in sa) if sa else 0.0


def phase2() -> dict:
    va, dir_a, meta_a = retrain(WORK / "retrain_a")
    vb, dir_b, meta_b = retrain(WORK / "retrain_b")
    metric_diffs = {}
    for model in ("autoencoder", "classifier", "next_service"):
        la = numeric_leaves(meta_a[model]["metrics"])
        lb = numeric_leaves(meta_b[model]["metrics"])
        for k in sorted(set(la) | set(lb)):
            metric_diffs[f"{model}.{k}"] = (
                abs(la[k] - lb[k]) if k in la and k in lb else float("inf"))
    weight_diffs = {f: state_dict_max_diff(dir_a / f, dir_b / f)
                    for f in ("autoencoder.pt", "classifier.pt", "nextservice.pt")}
    try:
        active = artifacts.active_version("minimodels")
    except LookupError:
        active = None
    return {
        "candidate_version_a": va,
        "candidate_version_b": vb,
        "same_candidate_version": va == vb,
        "max_metric_abs_diff": max(metric_diffs.values(), default=0.0),
        "metric_abs_diffs": metric_diffs,
        "state_dict_max_abs_diff": weight_diffs,
        # recorded, not asserted: the active artifact may be MPS-trained and
        # the device is part of the hparams that the version id hashes
        "registry_active_version": active,
        "candidate_equals_registry_active": va == active,
        "passed": (va == vb
                   and all(v <= ABS_TOL for v in metric_diffs.values())
                   and all(v <= ABS_TOL for v in weight_diffs.values())),
    }


def cleanup() -> None:
    """Scratch DBs and scratch artifact roots hold host-level data."""
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    report = {
        "abs_tol": ABS_TOL,
        "declared_exceptions": {
            "lean_dp_outputs": "lean/out DP releases are non-reproducible by "
                               "design (fresh verified randomness per release)",
        },
    }
    try:
        report["phase1_rebuild_frozen"] = phase1()
        report["phase2_retrain_determinism"] = phase2()
        report["passed"] = (report["phase1_rebuild_frozen"]["passed"]
                            and report["phase2_retrain_determinism"]["passed"])
    finally:
        cleanup()
        REPRO_DIR.mkdir(parents=True, exist_ok=True)
        (REPRO_DIR / "report.json").write_text(
            json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "passed": report.get("passed", False),
        "phase1": report.get("phase1_rebuild_frozen", {}).get("passed"),
        "phase2": report.get("phase2_retrain_determinism", {}).get("passed"),
        "ml_float_max_abs_diff":
            report.get("phase1_rebuild_frozen", {}).get("ml_flow_scores", {})
                  .get("float_max_abs_diff"),
        "retrain_same_version":
            report.get("phase2_retrain_determinism", {}).get("same_candidate_version"),
        "candidate_equals_registry_active":
            report.get("phase2_retrain_determinism", {}).get("candidate_equals_registry_active"),
    }, indent=1))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())

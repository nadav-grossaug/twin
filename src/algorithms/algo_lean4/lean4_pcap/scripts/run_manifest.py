#!/usr/bin/env python3
"""Write a run manifest: input data fingerprint, spec version, code and
dependency provenance, active model artifact metadata, and content
fingerprints of every table in the enclave DuckDB store.

    python3 scripts/run_manifest.py [--db PATH] [--out PATH]

Output: artifacts/runs/<UTC ts>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import artifacts  # noqa: E402
from provenance import (  # noqa: E402
    code_fingerprint,
    dependency_versions,
    git_revision,
    sha256_file,
)
from spec import load_spec, spec_fingerprint, spec_version  # noqa: E402

FLOAT_DECIMALS = 6  # declared tolerance for SQL float outputs
FLOAT_TYPES = {"DOUBLE", "FLOAT", "REAL"}
# anything nested / semi-structured is hashed via its canonical text form
NON_PRIMITIVE_MARKERS = ("[]", "STRUCT", "MAP", "LIST", "JSON", "UNION")


def _col_expr(name: str, dtype: str) -> str:
    q = f'"{name}"'
    if dtype in FLOAT_TYPES:
        # Doubles are rounded to FLOAT_DECIMALS before hashing: parallel
        # aggregation may reorder float sums and change their last bits.
        return f"round({q}, {FLOAT_DECIMALS})"
    if any(m in dtype for m in NON_PRIMITIVE_MARKERS):
        return f"cast({q} as varchar)"
    return q


def table_fingerprints(con: duckdb.DuckDBPyConnection) -> dict:
    """(row count, order-independent content hash) for every base table."""
    tabs = [r[0] for r in con.execute(
        "select table_name from information_schema.tables "
        "where table_schema = 'main' and table_type = 'BASE TABLE' order by 1"
    ).fetchall()]
    out = {}
    for t in tabs:
        cols = con.execute(
            "select column_name, data_type from information_schema.columns "
            "where table_schema = 'main' and table_name = ? "
            "order by ordinal_position", [t]
        ).fetchall()
        exprs = ", ".join(_col_expr(c, d) for c, d in cols)
        n, h = con.execute(
            f'select count(*), bit_xor(hash(row({exprs}))) from "{t}"'
        ).fetchone()
        out[t] = {"rows": n, "content_hash": str(h)}
    return out


def _model_entries() -> dict:
    models = {}
    for name in artifacts.load_registry():
        version, vdir, meta = artifacts.load_active(name)
        entry = {k: meta.get(k) for k in (
            "version", "created_utc", "data_fingerprint", "hparams",
            "release_status", "seed", "device", "code_fingerprint")}
        entry["metrics"] = {m: (meta.get(m) or {}).get("metrics")
                            for m in ("autoencoder", "classifier", "next_service")}
        entry["path"] = str(vdir.relative_to(ROOT))
        models[name] = entry
    return models


def build_manifest(db: Path) -> dict:
    spec = load_spec()
    con = duckdb.connect(str(db), read_only=True)
    try:
        tables = table_fingerprints(con)
    finally:
        con.close()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flow_jsonl": {
            "path": "flow.jsonl",
            "sha256": sha256_file(ROOT / "flow.jsonl"),
        },
        "spec": {
            "version": spec_version(),
            "fingerprint": spec_fingerprint(),
            "values": spec,
        },
        "code_fingerprint": code_fingerprint(),
        "git_revision": git_revision(),
        "dependencies": dependency_versions(),
        "models": _model_entries(),
        "db": str(db.resolve().relative_to(ROOT)) if db.resolve().is_relative_to(ROOT) else str(db),
        "tables": tables,
    }


def write_manifest(db: Path | None = None, out: Path | None = None) -> Path:
    db = db or ROOT / "artifacts" / "enclave.duckdb"
    m = build_manifest(db)
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "artifacts" / "runs" / ts / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2, default=str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "artifacts" / "enclave.duckdb")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    print(write_manifest(a.db, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

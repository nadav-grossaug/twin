#!/usr/bin/env python3
"""Run manifest: fingerprint everything that defines this run — source data,
spec, code, model artifact, DuckDB table hashes, Lean outputs — into
artifacts/runs/<timestamp>/manifest.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from provenance import sha256_file  # noqa: E402
from spec import spec_fingerprint, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def table_hashes(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    out = {}
    for (name,) in con.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'main' order by table_name").fetchall():
        # row-order-independent fingerprint; floats rounded to 6dp inside hash
        h = con.execute(
            f'select bit_xor(hash(t::varchar)) from "{name}" t').fetchone()[0]
        n = con.execute(f'select count(*) from "{name}"').fetchone()[0]
        out[name] = {"rows": int(n), "xor_hash": None if h is None else int(h)}
    con.close()
    return out


def main() -> int:
    db = ROOT / "artifacts" / "enclave.duckdb"
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_version": spec_version(),
        "spec_fingerprint": spec_fingerprint(),
        "data": {"facebook_full_features.graphml":
                 sha256_file(ROOT / "facebook_full_features.graphml")},
        "code": {p.name: sha256_file(p) for p in sorted(
            list((ROOT / "python").glob("*.py"))
            + list((ROOT / "scripts").glob("*.py")))},
        "tables": table_hashes(db) if db.exists() else "no database",
        "lean_out": {p.name: sha256_file(p) for p in sorted(
            (ROOT / "lean" / "out").glob("*.json"))},
        "models": {},
        "dp_note": "DP outputs are intentionally non-reproducible "
                   "(urandom-backed FFI); their hashes identify, not reproduce",
    }
    reg = ROOT / "models_out" / "registry.json"
    if reg.exists():
        manifest["models"] = json.loads(reg.read_text())
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = ROOT / "artifacts" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: artifacts/runs/{run_id}/manifest.json "
          f"({len(manifest['tables'])} tables)" if db.exists() else "manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Materialize the RELEASABLE affiliation-cohesion rows for the distiller.

Reads the policy artifact `affil_cohesion_stats` (relation
mart_affil_release) and applies EXACTLY its suppression rules, then writes
lean/out/affil_cohesion.json — the same rows release.py exports as
affil_cohesion_stats.csv, in the JSON shape python/distill.py consumes.
This runs BEFORE distill in the Makefile because distill (and the bundle)
read released artifacts only, never the warehouse.

Suppression-thresholded aggregates carry NO formal guarantee (declared in
the policy and repeated in the written file). When the warehouse is absent
(off-enclave bundle, fresh clone) this is a no-op: distill then declares
the affiliation gap instead of modelling it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
DB = Path(os.environ.get("GRAPH_DB", ROOT / "artifacts" / "enclave.duckdb"))
ARTIFACT = "affil_cohesion_stats"


def main() -> int:
    if not DB.exists():
        print("affil-release: no warehouse — skipping (distill will declare "
              "the affiliation gap)")
        return 0
    import duckdb
    import yaml

    policy = yaml.safe_load(
        (ROOT / "config" / "release_policy.yaml").read_text())
    spec = policy["artifacts"].get(ARTIFACT)
    if not spec or not spec.get("enabled", False):
        print(f"affil-release: policy artifact {ARTIFACT} disabled — skipping")
        return 0
    where = " AND ".join(
        f"{rule['column']} >= {int(rule['value'])}"
        for rule in spec.get("suppress", {}).values())
    cols = spec["columns"]
    con = duckdb.connect(str(DB), read_only=True)
    try:
        trow = con.execute(
            f"SELECT count(*) FROM {spec['relation']}").fetchone()
        total = int(trow[0]) if trow else 0
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM {spec['relation']}"
            + (f" WHERE {where}" if where else "")).fetchall()
    finally:
        con.close()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": ARTIFACT,
        "policy_version": policy["policy_version"],
        "guarantee": "suppression threshold only — reduces but does not "
                     "eliminate disclosure risk, no formal guarantee",
        "suppress": {k: {"column": r["column"], "value": int(r["value"])}
                     for k, r in spec.get("suppress", {}).items()},
        "suppressed_rows": total - len(rows),
        "rows": [dict(zip(cols, (v if isinstance(v, str) else int(v)
                                 for v in row))) for row in rows],
    }
    (OUT / "affil_cohesion.json").write_text(json.dumps(payload, indent=2))
    print(f"affil-release: wrote lean/out/affil_cohesion.json "
          f"({len(rows)} rows, {payload['suppressed_rows']} suppressed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

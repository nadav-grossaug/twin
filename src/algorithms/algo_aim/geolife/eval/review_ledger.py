"""EVAL: independent review of the privacy ledger of each committed artifact.

    PYTHONPATH=src python geolife/eval/review_ledger.py [--artifacts geolife/transfer/artifacts]
        [--variant outings_v2] [--out geolife/eval/results/attacks/ledger_review.md]

Re-derives everything the ledger states instead of trusting the built-in validator alone:
per-entry rho = sens^2/(2 sigma^2), the rho sum, the zCDP->(eps, delta) conversion, the stated
sensitivity against the schema's clipping bound (schema_v2.l2_bound / max_keys), entry<->table
agreement (names, n_keys, sigma/sens/rho of each table's meta.json), the caps block against the
schema constants, and runs load_artifact (hashes) + validate_no_raw_leakage (leak guard,
provenance allowlist). Findings that need judgement (caveat completeness) are in
geolife/eval/reports/ledger-review.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from geolife.inside.dp import zcdp_to_epsilon
from geolife.transfer import schema_v2 as S2
from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage


def review_one(d: Path) -> tuple[list[str], dict]:
    issues: list[str] = []
    art = load_artifact(d)                      # verifies every SHA-256, rejects unlisted files
    validate_no_raw_leakage(art)                # leak guard + provenance allowlist + ledger checks
    led = art.ledger
    ent = {e["name"]: e for e in led.entries}
    ms = {m.name: m for m in art.measurements}
    if set(ent) != set(ms):
        issues.append(f"entries != tables: {sorted(set(ent) ^ set(ms))}")
    for n, m in sorted(ms.items()):
        e = ent.get(n)
        if e is None:
            continue
        for f, v in (("sigma", m.sigma), ("l2_sensitivity", m.l2_sensitivity), ("rho", m.rho)):
            if not math.isclose(e[f], v, rel_tol=1e-12):
                issues.append(f"{n}: entry {f} {e[f]} != table meta {v}")
        if e.get("n_keys") != len(m.keys):
            issues.append(f"{n}: entry n_keys {e.get('n_keys')} != {len(m.keys)}")
        want_rho = m.l2_sensitivity ** 2 / (2 * m.sigma ** 2)
        if not math.isclose(m.rho, want_rho, rel_tol=1e-9):
            issues.append(f"{n}: rho {m.rho} != sens^2/(2 sigma^2) = {want_rho}")
        bound = S2.l2_bound(n) if n in S2.KIND else None
        if bound is not None and not math.isclose(m.l2_sensitivity, bound, rel_tol=1e-12):
            issues.append(f"{n}: stated sensitivity {m.l2_sensitivity} != schema bound {bound}")
    s = sum(e["rho"] for e in led.entries)
    if not math.isclose(s, led.rho_total, rel_tol=1e-9):
        issues.append(f"rho sum {s} != rho_total {led.rho_total}")
    eps = zcdp_to_epsilon(led.rho_total, led.delta)
    if not math.isclose(eps, led.epsilon, rel_tol=1e-9):
        issues.append(f"epsilon {led.epsilon} != conversion {eps}")
    want_caps = {"max_outings_per_user": S2.MAX_OUTINGS, "max_stop_cells_per_user": S2.MAX_STOP_CELLS,
                 "max_trav_cells_per_user": S2.MAX_TRAV_CELLS,
                 **{f"max_keys_{n}": float(S2.max_keys_of(n)) for n in S2.PLACE_TIERS + S2.TRAV_TIERS}}
    for k, v in want_caps.items():
        if float(led.caps.get(k, float("nan"))) != float(v):
            issues.append(f"caps[{k}] = {led.caps.get(k)} != schema {v}")
    info = {"dir": d.name, "n_tables": len(ms), "rho_total": led.rho_total,
            "epsilon": led.epsilon, "delta": led.delta, "n_caveats": len(led.caveats),
            "adjacency": led.adjacency, "files": len(art.manifest.files)}
    return issues, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="geolife/transfer/artifacts")
    ap.add_argument("--variant", default="outings_v2")
    ap.add_argument("--out", default=None, help="write a markdown table here as well")
    a = ap.parse_args(argv)
    dirs = sorted(p for p in Path(a.artifacts).iterdir()
                  if p.is_dir() and p.name.startswith(a.variant) and (p / "manifest.json").is_file())
    rows, bad = [], 0
    for d in dirs:
        issues, info = review_one(d)
        rows.append((info, issues))
        state = "OK" if not issues else f"{len(issues)} ISSUE(S)"
        print(f"{d.name}: {state}; {info['n_tables']} tables, rho {info['rho_total']:.6f}, "
              f"eps {info['epsilon']:.3f} at delta {info['delta']}, {info['n_caveats']} caveats, "
              f"{info['files']} files verified")
        for i in issues:
            bad += 1
            print(f"  - {i}")
    if a.out:
        L = ["| artifact | tables | rho_total | epsilon | delta | caveats | files | issues |",
             "|---|---|---|---|---|---|---|---|"]
        for info, issues in rows:
            L.append(f"| {info['dir']} | {info['n_tables']} | {info['rho_total']:.6f} | "
                     f"{info['epsilon']:.3f} | {info['delta']} | {info['n_caveats']} | "
                     f"{info['files']} | {'; '.join(issues) or 'none'} |")
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(L) + "\n", encoding="utf-8")
    index = Path(a.artifacts) / "index.json"
    if index.is_file():
        for row in json.loads(index.read_text()):
            info = next((i for i, _ in rows if i["dir"] == row.get("dir")), None)
            if info and not math.isclose(float(row["epsilon"]), info["epsilon"], rel_tol=1e-9):
                print(f"index.json: epsilon {row['epsilon']} != ledger {info['epsilon']} for {row['dir']}")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

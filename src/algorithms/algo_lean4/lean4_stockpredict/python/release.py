"""Policy-enforced export of releasable aggregates (default deny).

    FLOW_ALLOW_DEV_RELEASE=1 python3 python/release.py

Each export goes to release/<export_id>/ with a manifest of file hashes,
attestation claims, an exact-fraction epsilon ledger and suppressed-row
counts. Everything the policy does not list (and enable) stays inside the
enclave. The DP files themselves are produced by the Lean executables
(`make lean`); this module only copies them, verbatim, after checking the
budget.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import duckdb
import yaml

from enclave import (
    PROJECT_ROOT,
    RELEASE_ROOT,
    AttestationError,
    AttestationProvider,
    DevNoEnclave,
    EgressGuard,
    Verifier,
    new_nonce,
)
from provenance import sha256_file
from safe_logging import get_logger

POLICY_PATH = PROJECT_ROOT / "config" / "release_policy.yaml"
DB_PATH = PROJECT_ROOT / "artifacts" / "enclave.duckdb"
LEAN_OUT = PROJECT_ROOT / "lean" / "out"
LEAN_FILES = ("dp_release.json", "samples.json", "lean_results.json")

log = get_logger("release")


def load_policy(path: Path = POLICY_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def attest(policy: dict, provider: AttestationProvider, verifier: Verifier | None) -> dict:
    nonce = new_nonce()
    ev = provider.evidence(nonce)
    if ev.platform == "none-dev":
        if os.environ.get(policy["allow_dev_platform_env"]) != "1":
            raise AttestationError(
                f"no enclave attestation; set {policy['allow_dev_platform_env']}=1 "
                "to export from a dev host (export is then marked attested: false)")
        return {"platform": ev.platform, "measurement": ev.measurement, "attested": False}
    if verifier is None:
        raise AttestationError("attestation required but no verifier configured")
    claims = verifier.verify(ev, nonce)
    return {"platform": claims.platform, "measurement": claims.measurement, "attested": True}


def _frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def epsilon_ledger(policy: dict) -> dict:
    """Exact-fraction DP budget ledger; raises PermissionError when exceeded."""
    total, per_release = Fraction(0), {}
    for name, spec in policy.get("dp_releases", {}).items():
        eps = Fraction(spec["epsilon_per_cell"]) * int(spec["n_cells"])
        # contribution bound: proportional L1 (clip_l1_per_user) or uniform
        # per-cell (clip_per_user_per_cell) — exactly one must be declared
        clip_key = ("clip_l1_per_user" if "clip_l1_per_user" in spec
                    else "clip_per_user_per_cell")
        per_release[name] = {
            "epsilon_per_cell": _frac(Fraction(spec["epsilon_per_cell"])),
            "n_cells": int(spec["n_cells"]),
            clip_key: int(spec[clip_key]),
            "epsilon": _frac(eps),
        }
        total += eps
    cap = Fraction(policy["dp_total_epsilon_max"])
    if total > cap:
        raise PermissionError(f"DP budget exceeded: {total} > {cap}")
    return {
        "privacy_unit": "one ticker (add/remove-one neighbour; its whole "
                        "series is one record)",
        "per_release": per_release,
        "epsilon_total": _frac(total),
        "epsilon_max": _frac(cap),
    }


def lean_claims_payload(results: dict, include: list[str]) -> dict:
    """Theorem claims + aggregate agreement counts. Never per-ticker rows:
    those (runs/shocks/drawdowns/... arrays) identify portfolio members."""

    def agreeing(rows: list[dict], key: str = "agree") -> int:
        return sum(1 for r in rows if r.get(key))

    payload = {
        "claims": results["claims"],
        "agreement_counts": {
            **{k: {"n": len(results[k]), "agree": agreeing(results[k])}
               for k in ("runs", "shocks", "drawdowns", "vrecoveries",
                         "volbursts", "assoc")},
            "sessions": {"n": len(results["sessions"]),
                         "agree": agreeing(results["sessions"], "lean_is_timely_session")},
        },
    }
    return {k: v for k, v in payload.items() if k in include}


def export_relation(con, spec: dict, path: Path) -> tuple[int, int]:
    """Allowlisted columns + suppression thresholds; returns (rows, suppressed)."""
    where = " AND ".join(
        f"{rule['column']} >= {int(rule['value'])}" for rule in spec.get("suppress", {}).values())
    total = con.execute(f"SELECT count(*) FROM {spec['relation']}").fetchone()[0]
    cols = ", ".join(spec["columns"])
    q = f"SELECT {cols} FROM {spec['relation']}" + (f" WHERE {where}" if where else "")
    rows = con.execute(q).fetchall()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(spec["columns"])
        w.writerows(rows)
    return len(rows), total - len(rows)


def export(provider: AttestationProvider | None = None, verifier: Verifier | None = None,
           db_path: Path = DB_PATH, release_root: Path = RELEASE_ROOT) -> Path:
    policy = load_policy()
    if policy["require_attestation"]:
        claims = attest(policy, provider or DevNoEnclave(), verifier)
    else:
        claims = {"attested": False}
    ledger = epsilon_ledger(policy)
    missing = [n for n in LEAN_FILES if not (LEAN_OUT / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing Lean outputs {missing} under {LEAN_OUT} — run `make lean` first")

    release_root = Path(release_root)
    guard = EgressGuard(release_root)
    export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = guard.check(release_root / export_id)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "export_id": export_id,
        "policy_version": policy["policy_version"],
        "attestation": claims,
        "dp_epsilon": ledger,
        "files": {},
        "privacy_note": (
            "dp_release.json / samples.json: per-ticker pure epsilon-DP, produced "
            "by the Lean executables (SampCert discrete Laplace, proved); "
            "lean_claims.json: theorem statuses and aggregate agreement counts only; "
            "seasonality_stats.csv: suppression threshold only — reduces but does "
            "not eliminate disclosure risk, no formal guarantee"),
    }
    log.info("export started: attested=%s" % claims["attested"])

    # DP files: copied verbatim from the Lean executables' output.
    for name, spec in policy.get("dp_releases", {}).items():
        for src in spec["sources"]:
            srcp = PROJECT_ROOT / src
            dst = guard.check(out / Path(src).name)
            shutil.copyfile(srcp, dst)
            manifest["files"][dst.name] = {"sha256": sha256_file(dst), "dp": True, "release": name}
        log.info("exported dp release %s: cells=%s" % (name, spec["n_cells"]))

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for name, spec in policy["artifacts"].items():
            if not spec.get("enabled", False):
                continue
            if "relation" in spec:
                dst = guard.check(out / f"{name}.csv")
                n_rows, suppressed = export_relation(con, spec, dst)
                manifest["files"][dst.name] = {"sha256": sha256_file(dst), "dp": False,
                                               "rows": n_rows, "suppressed_rows": suppressed}
                log.info("exported %s: rows=%d suppressed=%d" % (name, n_rows, suppressed))
            else:
                results = json.loads((PROJECT_ROOT / spec["source"]).read_text())
                payload = lean_claims_payload(results, spec.get("include", ["claims", "agreement_counts"]))
                dst = guard.check(out / f"{name}.json")
                dst.write_text(json.dumps(payload, indent=2))
                manifest["files"][dst.name] = {"sha256": sha256_file(dst), "dp": False,
                                               "rows": len(payload.get("claims", []))}
                log.info("exported %s: claims=%d" % (name, len(payload.get("claims", []))))
    finally:
        con.close()

    mpath = guard.check(out / "manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    log.info("export finished: files=%d" % len(manifest["files"]))
    return out


def is_allowed(relation: str, policy: dict | None = None) -> bool:
    """Default deny: only an enabled artifact's relation may leave."""
    policy = policy or load_policy()
    return any(spec.get("relation") == relation and spec.get("enabled", False)
               for spec in policy["artifacts"].values())


if __name__ == "__main__":
    print(export())

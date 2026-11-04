"""Policy-enforced export of releasable aggregates (default deny).

    BOOKING_ALLOW_DEV_RELEASE=1 python3 python/release.py

Each export goes to release/<export_id>/ with a manifest of file hashes,
attestation claims, an exact-fraction epsilon ledger and suppressed-row
counts. Everything the policy does not list (and enable) stays inside the
enclave. The DP files themselves are produced by the Lean executables
(`make lean`); this module only copies them, verbatim, after checking that
the budget the Lean release says it spent matches the policy ledger —
compared with fractions.Fraction, never floats.
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
LIFETIME_LEDGER_PATH = PROJECT_ROOT / "config" / "epsilon_ledger.jsonl"
DATA_CSV = Path(os.environ.get("BOOKING_CSV", str(PROJECT_ROOT / "combined_set.csv")))

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
        "privacy_unit": "one user (add/remove-one neighbour; their whole "
                        "reservation history is one record)",
        "per_release": per_release,
        "epsilon_total": _frac(total),
        "epsilon_max": _frac(cap),
        "note": "cap 14/1 set by explicit user choice (fidelity over "
                "privacy, as in lean4_snap); the bound is weak "
                "(epsilon ~ 10.08) — do not call the release private",
    }


def _budget_fraction(budget: dict, kind: str) -> Fraction:
    """Read an exact fraction out of dp_release.json's budget block; accepts
    "spent": "1008/100" or the split "spent_num"/"spent_den" spelling."""
    if kind in budget:
        v = budget[kind]
        if isinstance(v, dict):
            return Fraction(int(v["num"]), int(v["den"]))
        return Fraction(str(v))
    if f"{kind}_num" in budget and f"{kind}_den" in budget:
        return Fraction(int(budget[f"{kind}_num"]), int(budget[f"{kind}_den"]))
    raise KeyError(f"dp_release.json budget lacks an exact '{kind}' fraction")


def check_release_budget(ledger: dict, rel_path: Path) -> dict:
    """Cross-check: the budget the Lean release says it SPENT must equal the
    policy ledger exactly (Fraction comparison, never floats), and its total
    must not exceed the policy cap."""
    rel = json.loads(rel_path.read_text())
    budget = rel.get("budget")
    if not isinstance(budget, dict):
        raise PermissionError("dp_release.json carries no budget block — refusing export")
    spent = _budget_fraction(budget, "spent")
    total = _budget_fraction(budget, "total")
    want_spent = Fraction(ledger["epsilon_total"])
    cap = Fraction(ledger["epsilon_max"])
    if spent != want_spent:
        raise PermissionError(
            f"Lean release spent {_frac(spent)} but the policy ledger says "
            f"{ledger['epsilon_total']} — refusing export")
    if spent > total or total > cap:
        raise PermissionError(
            f"Lean release budget {_frac(spent)}/{_frac(total)} exceeds the "
            f"policy cap {ledger['epsilon_max']} — refusing export")
    return {"spent": _frac(spent), "total": _frac(total), "matches_policy": True}


# ---------------------------------------------------------------------------
# LIFETIME epsilon ledger (append-only, keyed by the source-data sha).
# Every export composes with every prior export of the SAME data (pure DP
# adds up); the per-export cap alone would let N exports spend N x eps.
# Each entry: {export_id, ts_utc, data_sha256, spec_version, policy_version,
#              epsilon_spent, lifetime_total, attested}.
# ---------------------------------------------------------------------------

def lifetime_entries(path: Path = LIFETIME_LEDGER_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check_lifetime_budget(policy: dict, data_sha: str, spent: Fraction,
                          path: Path = LIFETIME_LEDGER_PATH) -> dict:
    """Refuse the export if the CUMULATIVE spend over all prior exports of
    the same data would exceed the policy's declared lifetime cap. A policy
    without dp_lifetime_epsilon_max does not gate (backward compatible)."""
    prior = sum((Fraction(e["epsilon_spent"]) for e in lifetime_entries(path)
                 if e.get("data_sha256") == data_sha), Fraction(0))
    after = prior + spent
    base = {"prior": _frac(prior), "after_this_export": _frac(after),
            "n_prior_exports": len(lifetime_entries(path))}
    if "dp_lifetime_epsilon_max" not in policy:
        return {**base, "cap": None}
    cap = Fraction(str(policy["dp_lifetime_epsilon_max"]))
    if after > cap:
        raise PermissionError(
            f"LIFETIME budget exceeded: prior {_frac(prior)} + this export "
            f"{_frac(spent)} = {_frac(after)} > dp_lifetime_epsilon_max "
            f"{policy['dp_lifetime_epsilon_max']} — refusing export")
    return {**base, "cap": policy["dp_lifetime_epsilon_max"]}


def append_lifetime_entry(entry: dict, path: Path = LIFETIME_LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def lean_claims_payload(results: dict, include: list[str]) -> dict:
    """Theorem claims + aggregate agreement counts. Never per-user/per-trip
    rows: those (returns/corridors/tours/longstays/trips arrays) identify
    users."""

    def agreeing(rows: list[dict], key: str = "agree") -> int:
        return sum(1 for r in rows if r.get(key))

    payload = {
        "claims": results["claims"],
        "agreement_counts": {
            **{k: {"n": len(results[k]), "agree": agreeing(results[k])}
               for k in ("returns", "corridors", "tours",
                         "longstays", "assoc", "trips")},
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
    ledger["lean_release_budget"] = check_release_budget(
        ledger, LEAN_OUT / "dp_release.json")

    # lifetime accounting: this export composes with every prior export of
    # the SAME data (pure DP); the entry is appended only after the export
    # fully succeeds, so a refused/failed export spends nothing.
    rel_json = json.loads((LEAN_OUT / "dp_release.json").read_text())
    rel_budget = rel_json["budget"]
    rel_spec = str(rel_json.get("spec_version", ""))
    spent_frac = _budget_fraction(rel_budget, "spent")
    data_sha = sha256_file(DATA_CSV)
    lifetime = check_lifetime_budget(policy, data_sha, spent_frac)
    ledger["lifetime"] = lifetime

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
            "dp_release.json / samples.json: per-user pure epsilon-DP, produced "
            "by the Lean executables (SampCert discrete Laplace, proved); "
            "lean_claims.json: theorem statuses and aggregate agreement counts only; "
            "gap_moment_stats.csv / topk_cities.csv / topk_city_transitions.csv: "
            "suppression threshold only — reduces but does not eliminate "
            "disclosure risk, no formal guarantee"),
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
    append_lifetime_entry({
        "export_id": export_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": data_sha,
        "spec_version": rel_spec,
        "policy_version": policy["policy_version"],
        "epsilon_spent": _frac(spent_frac),
        "lifetime_total": lifetime["after_this_export"],
        "attested": bool(claims.get("attested")),
    })
    log.info("export finished: files=%d, lifetime epsilon now %s of %s"
             % (len(manifest["files"]), lifetime["after_this_export"],
                lifetime["cap"]))
    return out


def is_allowed(relation: str, policy: dict | None = None) -> bool:
    """Default deny: only an enabled artifact's relation may leave."""
    policy = policy or load_policy()
    return any(spec.get("relation") == relation and spec.get("enabled", False)
               for spec in policy["artifacts"].values())


if __name__ == "__main__":
    print(export())

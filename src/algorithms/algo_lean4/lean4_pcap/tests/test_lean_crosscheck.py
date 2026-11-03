"""Lean ↔ SQL agreement on the current snapshot, and formal-layer hygiene.

Scope (stated honestly): these tests show that on THIS data snapshot the SQL
detectors agree with the decidable Lean definitions; they are executable
checks, not proofs of the SQL in general.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_beacon_agreement(lean_results):
    rows = lean_results["beacons"]
    assert rows, "no beacon checks exported"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"SQL/Lean beacon disagreement: {bad}"


def test_fanout_agreement(lean_results):
    rows = lean_results["fanout"]
    assert rows and all(r["agree"] for r in rows)


def test_chi2_exact_agreement(lean_results):
    rows = lean_results.get("assoc", [])
    assert rows, "no chi-square checks exported"
    bad = [r for r in rows if not r["agree"]]
    assert not bad, f"exact-ℚ chi-square diverges from SQL: {bad}"


def test_sessions_are_timely(lean_results):
    rows = lean_results.get("sessions", [])
    assert rows and all(r["lean_is_timely_session"] for r in rows)


def test_claims_ledger_shape(lean_results):
    claims = lean_results["claims"]
    statuses = {c["status"] for c in claims}
    assert statuses <= {"proved", "assumed"}
    assert sum(1 for c in claims if c["status"] == "proved") >= 12
    assert sum(1 for c in claims if c["status"] == "assumed") >= 3


def test_no_sorry_in_lean_sources():
    lean_dir = ROOT / "lean"
    offenders = []
    for p in list(lean_dir.glob("*.lean")) + list((lean_dir / "EnclaveFlows").glob("*.lean")):
        if p.name == "GenCert.lean":  # generated certificate, checked separately
            continue
        text = p.read_text()
        if "sorry" in text or "admit" in text:
            offenders.append(p.name)
    assert not offenders, f"formal layer contains sorry/admit: {offenders}"

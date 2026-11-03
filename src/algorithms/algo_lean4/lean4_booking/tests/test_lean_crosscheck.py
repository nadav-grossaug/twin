"""SQL <-> Lean agreement: every exported motif/assoc/trip observation must
be decided identically by the dbt SQL and the formal Lean definitions."""
import pytest


@pytest.fixture(scope="module")
def results(lean_results):
    return lean_results


KEYS = ["returns", "corridors", "tours", "longstays", "assoc"]


@pytest.mark.parametrize("key", KEYS)
def test_agreement(results, key):
    rows = results[key]
    assert rows, f"no {key} observations exported"
    bad = [r for r in rows if not r.get("agree", False)]
    assert not bad, f"{len(bad)}/{len(rows)} {key} disagreements: {bad[:3]}"


def test_trips_all_agree(results):
    """Well-formedness/contiguity verdicts: the Lean IsWellFormedTrip /
    IsContiguousTrip deciders must agree with the SQL flags on every
    exported trip (agreement means matching verdicts, not that a trip is
    contiguous — ~0.9% of real trips have a gap)."""
    rows = results["trips"]
    assert rows
    assert all(r["agree"] for r in rows), [
        r for r in rows if not r["agree"]][:3]


def test_claims_ledger_shape(results):
    claims = {c["name"]: c["status"] for c in results["claims"]}
    # the mechanism, sampler and generator contracts must be proved
    for name in ("privClippedCell_PureDP", "histMech_DP", "sampleCellPMF_law"):
        assert claims.get(name) == "proved", name
    # at least one generator validity claim is proved
    gen = [n for n, s in claims.items()
           if s == "proved" and ("mk" in n or "sorted" in n.lower())]
    assert gen, "no proved generator-contract claim in the ledger"
    # the trust boundary is explicit
    for name in ("sql_counts_correct", "entropy_source", "io_wrappers"):
        assert claims.get(name) == "assumed", name

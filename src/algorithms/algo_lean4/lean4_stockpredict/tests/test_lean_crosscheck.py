"""SQL <-> Lean agreement: every exported motif/assoc observation must be
decided identically by the dbt SQL and the formal Lean definitions."""
import json

import pytest


@pytest.fixture(scope="module")
def results(lean_results):
    return lean_results


KEYS = ["runs", "shocks", "drawdowns", "vrecoveries", "volbursts", "assoc"]


@pytest.mark.parametrize("key", KEYS)
def test_agreement(results, key):
    rows = results[key]
    assert rows, f"no {key} observations exported"
    bad = [r for r in rows if not r.get("agree", False)]
    assert not bad, f"{len(bad)}/{len(rows)} {key} disagreements: {bad[:3]}"


def test_sessions_all_timely(results):
    rows = results["sessions"]
    assert rows
    assert all(r["lean_is_timely_session"] for r in rows)


def test_claims_ledger_shape(results):
    claims = {c["name"]: c["status"] for c in results["claims"]}
    # the mechanism, sampler and generator contracts must be proved
    for name in ("privClippedCell_PureDP", "histMech_DP", "sampleCellPMF_law",
                 "mkDay_valid", "tradingDayNum_strictMono"):
        assert claims.get(name) == "proved", name
    # the trust boundary is explicit
    for name in ("sql_counts_correct", "entropy_source", "io_wrappers"):
        assert claims.get(name) == "assumed", name

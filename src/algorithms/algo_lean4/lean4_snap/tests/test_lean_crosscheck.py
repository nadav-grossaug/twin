"""SQL <-> Lean agreement: every exported per-node observation must be
decided identically by the dbt SQL and the formal Lean definitions, and the
exact-ℚ chi-square recomputation must match SQL's floats."""
import pytest


@pytest.fixture(scope="module")
def results(lean_results):
    return lean_results


def test_no_disagreements(results):
    agg = results["agreement"]
    assert agg["n_checked"] > 0, "no checks ran"
    assert agg["n_disagree"] == 0, (
        f"{agg['n_disagree']}/{agg['n_checked']} disagreements; samples: "
        f"{results.get('disagreement_samples', [])[:3]}")


def test_full_node_coverage(results):
    # 6 motif predicates + 5 class maps for every one of the 4039 nodes
    assert results["motif_checks"] == results["class_checks"] / 5 * 6
    assert results["motif_checks"] >= 6 * 4000


def test_assoc_grids_agree(results):
    rows = results["assoc"]
    assert rows, "no chi-square grids exported"
    bad = [r for r in rows if not r.get("agree", False)]
    assert not bad, f"{len(bad)}/{len(rows)} chi-square disagreements: {bad[:3]}"


def test_claims_ledger_shape(results):
    claims = {c["name"]: c["status"] for c in results["claims"]}
    # the mechanism, sampler and generator contracts must be proved
    for name in ("privClippedCell_PureDP", "histMech_DP", "sampleCellPMF_law",
                 "edgesCanonical_sound", "mkEdge_ok"):
        assert claims.get(name) == "proved", name
    # the trust boundary is explicit — including the graph-specific
    # row-level-DP caveat
    for name in ("sql_counts_correct", "entropy_source", "io_wrappers",
                 "row_level_privacy_unit"):
        assert claims.get(name) == "assumed", name

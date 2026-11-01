"""Release gate: default deny, exact-fraction ledger, policy-filtered export."""
import json
from fractions import Fraction

import pytest
import yaml

from release import epsilon_ledger, is_allowed, load_policy


@pytest.fixture(scope="module")
def policy():
    return load_policy()


def test_ledger_within_budget(policy):
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) <= Fraction(ledger["epsilon_max"])


def test_ledger_exactly_saturated(policy):
    # a new query must displace an old one at review time
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) == Fraction("4/1")


def test_over_budget_refused(policy):
    bad = json.loads(json.dumps(policy))
    bad["dp_releases"]["class_histogram"]["epsilon_per_cell"] = "2/1"
    with pytest.raises(PermissionError):
        epsilon_ledger(bad)


@pytest.mark.parametrize("relation", [
    "stg_prices", "feat_returns", "ml_series_scores", "mart_series_stats",
    "mart_correlation", "mart_motif_runs", "mart_seq_sessions",
    "mart_assoc_cells",
])
def test_default_deny(policy, relation):
    assert not is_allowed(relation, policy)
    assert relation in policy["denied"], f"{relation} missing from the audit list"


def test_allowed_artifact(policy):
    assert is_allowed("mart_seasonality", policy)


def test_model_weights_denied(policy):
    assert policy["model_weights"] == "deny_unless_reviewed"


def test_dp_release_grids_match_lean_output(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make lean-release`")
    rel = json.loads(p.read_text())
    assert len(rel["cells"]) == policy["dp_releases"]["class_histogram"]["n_cells"]
    by_name = {h["name"]: h for h in rel["aux"]}
    for pol_name, aux_name in [("class_transition_histogram", "class_transition"),
                               ("dow_class_histogram", "dow_class"),
                               ("vol_transition_histogram", "vol_transition"),
                               ("mag_bins_histogram", "mag_bins"),
                               ("comove_histogram", "comove")]:
        spec = policy["dp_releases"][pol_name]
        aux = by_name[aux_name]
        assert len(aux["cells"]) == spec["n_cells"], pol_name
        assert (aux["eps_per_cell"]["num"], aux["eps_per_cell"]["den"]) == (
            Fraction(spec["epsilon_per_cell"]).numerator,
            Fraction(spec["epsilon_per_cell"]).denominator), pol_name

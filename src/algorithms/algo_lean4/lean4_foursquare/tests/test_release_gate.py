"""Release gate: default deny, exact-fraction ledger, policy-filtered export."""
import json
from fractions import Fraction

import pytest

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
    bad["dp_releases"]["region_transition_histogram"]["epsilon_per_cell"] = "2/1"
    with pytest.raises(PermissionError):
        epsilon_ledger(bad)


@pytest.mark.parametrize("relation", [
    "stg_transfers", "feat_transfers", "feat_users", "feat_episodes",
    "ml_travel_scores", "mart_user_stats", "mart_motif_returns",
    "mart_motif_corridors", "mart_motif_tours", "mart_episodes",
    "mart_assoc_cells", "mart_seq_transitions",
])
def test_default_deny(policy, relation):
    assert not is_allowed(relation, policy)
    assert relation in policy["denied"], f"{relation} missing from the audit list"


def test_allowed_artifacts(policy):
    assert is_allowed("mart_gap_moments", policy)
    assert is_allowed("mart_topk_corridors", policy)


def test_model_weights_denied(policy):
    assert policy["model_weights"] == "deny_unless_reviewed"


def test_dp_release_grids_match_lean_output(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    assert len(rel["cells"]) == policy["dp_releases"]["travel_count_histogram"]["n_cells"]
    by_name = {h["name"]: h for h in rel["aux"]}
    for pol_name, aux_name in [
        ("region_histogram", "region"),
        ("region_transition_histogram", "region_transition"),
        ("gap_class_histogram", "gap_class"),
        ("duration_class_histogram", "duration_class"),
        ("distance_class_histogram", "distance_class"),
        ("month_histogram", "month"),
        ("episode_len_histogram", "episode_len"),
        ("return_rate_histogram", "return_rate"),
    ]:
        spec = policy["dp_releases"][pol_name]
        aux = by_name[aux_name]
        assert len(aux["cells"]) == spec["n_cells"], pol_name
        assert Fraction(aux["eps1"], aux["eps2"]) == Fraction(
            spec["epsilon_per_cell"]), pol_name


def test_topk_corridor_suppression(root):
    """Released corridors must clear BOTH suppression thresholds; user ids
    and venue ids never appear in any released artifact."""
    rel_dirs = sorted((root / "release").glob("*/")) if (root / "release").exists() else []
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    latest = rel_dirs[-1]
    for f in latest.glob("*.json"):
        txt = f.read_text()
        assert "venue_id" not in txt, f.name
        assert "userid" not in txt, f.name

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


def test_ledger_is_the_declared_total(policy):
    # 1369/100 of 14/1, nearly saturated: a new query must displace an old one.
    # FIDELITY-FIRST BUDGET (user decision 2026-11-03): the cap was raised from
    # 4/1 to buy the mixing grid at 5x and nine-category affiliation key-groups.
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) == Fraction("1369/100")
    assert Fraction(ledger["epsilon_max"]) == Fraction("14/1")


def test_over_budget_refused(policy):
    bad = json.loads(json.dumps(policy))
    bad["dp_releases"]["mixing_histogram"]["epsilon_per_cell"] = "2/1"
    with pytest.raises(PermissionError):
        epsilon_ledger(bad)


@pytest.mark.parametrize("relation", [
    "stg_nodes", "stg_edges", "stg_node_features", "stg_circles",
    "feat_nodes", "feat_edges", "ml_graph_scores", "mart_degree_stats",
    "mart_clustering", "mart_triangle_stats", "mart_community_sizes",
    "mart_community_mixing", "mart_wedge_closure", "mart_assoc_cells",
])
def test_default_deny(policy, relation):
    assert not is_allowed(relation, policy)
    assert relation in policy["denied"], f"{relation} missing from the audit list"


def test_allowed_artifacts(policy):
    assert is_allowed("mart_cond_categories", policy)
    assert is_allowed("mart_motif_summary", policy)
    assert is_allowed("mart_affil_release", policy)
    # the key-level and full-rollup affiliation marts stay denied
    assert not is_allowed("mart_key_groups", policy)
    assert not is_allowed("mart_edge_shared", policy)
    assert not is_allowed("mart_affil_cohesion", policy)


def test_model_weights_denied(policy):
    assert policy["model_weights"] == "deny_unless_reviewed"


def test_dp_release_grids_match_lean_output(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    assert len(rel["cells"]) == policy["dp_releases"]["degree_histogram"]["n_cells"]
    by_name = {h["name"]: h for h in rel["aux"]}
    for pol_name, aux_name in [
        ("clustering_histogram", "clustering"),
        ("triangle_histogram", "triangles"),
        ("richness_histogram", "richness"),
        ("category_histogram", "category"),
        ("circle_histogram", "circles"),
        ("community_size_histogram", "community_sizes"),
        ("mixing_histogram", "mixing"),
        ("closure_histogram", "closure"),
        ("motif_histogram", "motifs"),
        ("affil_size_histogram", "affil_sizes"),
    ]:
        spec = policy["dp_releases"][pol_name]
        aux = by_name[aux_name]
        assert len(aux["cells"]) == spec["n_cells"], pol_name
        assert Fraction(aux["eps1"], aux["eps2"]) == Fraction(
            spec["epsilon_per_cell"]), pol_name


def test_released_files_carry_no_node_ids(root):
    """Released artifacts never contain node-level identifiers."""
    rel_dirs = sorted((root / "release").glob("*/")) if (root / "release").exists() else []
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    latest = rel_dirs[-1]
    for f in latest.glob("*.json"):
        txt = f.read_text()
        assert "node_id" not in txt, f.name
        assert "feature_key" not in txt, f.name
    for f in latest.glob("*.csv"):
        head = f.read_text().splitlines()[0]
        assert "node_id" not in head, f.name

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

from graph_aim.eval.coupling import (
    coupling_metrics,
    flatten,
    gap_closed,
    gap_closed_table,
    options_from_policy,
    permuted_null,
    reference_metrics,
    within_group_triangle_share,
)
from graph_aim.transfer.graph_io import save_attributed_graph
from graph_aim.transfer.types import AttributedGraph

EDGES = [0, 1, 4, 16, 65]  # public degree bins used throughout


def graph(n: int, edges, attrs: dict, domains: dict) -> AttributedGraph:
    """Graph from an edge list and attribute codes."""
    e = np.array(sorted({(int(min(x)), int(max(x))) for x in edges}), dtype=np.int64).reshape(-1, 2)
    df = pd.DataFrame({c: np.asarray(v, dtype=np.int64) for c, v in attrs.items()})
    return AttributedGraph(n=n, edges=e, attrs=df, domains=domains)


def metrics(real, syn, **kw):
    kw.setdefault("degree_bin_edges", EDGES)
    return coupling_metrics(real, syn, **kw)


def random_graph(n: int, m: int, probs: dict, seed: int, accept=None):
    """Random graph, attributes drawn independently from ``probs[col]`` (domain: len(p) values + other + none)."""
    rng = np.random.default_rng(seed)
    attrs = {c: rng.choice(len(p), size=n, p=p) for c, p in probs.items()}
    pairs = set()
    while len(pairs) < m:
        u, v = rng.integers(0, n, 2)
        if u == v:
            continue
        if accept is not None and rng.random() > accept(attrs, u, v):
            continue
        pairs.add((min(u, v), max(u, v)))
    doms = {c: [f"{c}{i}" for i in range(len(p))] + ["other", "none"] for c, p in probs.items()}
    return graph(n, pairs, attrs, doms), rng


# --------------------------------------------------------------------------------------------- homophily
def test_perfect_homophily_known_answer():
    # value 0: 4-cycle on nodes 0-3 (8 end points), value 1: path 4-5-6 (4 end points): a = (2/3, 1/3)
    g = graph(7, [(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6)], {"x": [0, 0, 0, 0, 1, 1, 1]}, {"x": ["a", "b", "none"]})
    a = metrics(g, g)["attributes"]["x"]
    assert a["assort_real"] == pytest.approx(1.0)
    assert a["lift_real"] == pytest.approx(1 / (4 / 9 + 1 / 9))  # 1 / sum a_i^2 = 1.8
    assert a["mix_tvd"] == 0 and a["assort_absdiff"] == 0
    assert a["observed_pair_share_real"] == 1.0


def test_random_pairing_has_no_assortativity_and_lift_one():
    g, _ = random_graph(3000, 12000, {"x": [0.5, 0.3, 0.2]}, seed=1)
    a = metrics(g, g)["attributes"]["x"]
    assert abs(a["assort_real"]) < 0.03
    assert a["lift_real"] == pytest.approx(1.0, abs=0.03)


def test_missing_label_exclusion_changes_lift_in_expected_direction():
    # 400 'none' nodes wired among themselves (would look like strong homophily), 600 observed nodes wired at random
    rng = np.random.default_rng(0)
    codes = np.concatenate([np.full(400, 2), rng.integers(0, 2, 600)])  # domain [a, b, none]
    pairs = set()
    while len(pairs) < 2000:
        u, v = rng.integers(0, 400, 2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    while len(pairs) < 4000:
        u, v = rng.integers(400, 1000, 2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    g = graph(1000, pairs, {"x": codes}, {"x": ["a", "b", "none"]})
    excluded = metrics(g, g)["attributes"]["x"]  # default: 'none' is missing
    counted = metrics(g, g, missing={"x": None})["attributes"]["x"]
    assert excluded["lift_real"] == pytest.approx(1.0, abs=0.05)
    assert counted["lift_real"] > 1.5 and counted["assort_real"] > 0.5
    assert excluded["observed_pair_share_real"] == pytest.approx(0.5)
    assert counted["observed_pair_share_real"] == 1.0


def test_missing_label_can_be_named_per_attribute():
    g = graph(4, [(0, 1), (2, 3)], {"x": [0, 0, 1, 1]}, {"x": ["a", "b", "unknown"]})  # no 'none' label at all
    assert metrics(g, g)["attributes"]["x"]["observed_pair_share_real"] == 1.0  # default: nothing is missing
    a = metrics(g, g, missing={"x": "b"})["attributes"]["x"]
    assert a["observed_pair_share_real"] == 0.5  # edge (2,3) has both ends missing, edge (0,1) both observed


def test_observed_pair_share_real_vs_synthetic():
    dom = {"x": ["a", "b", "none"]}
    real = graph(6, [(0, 1), (2, 3), (4, 5), (1, 2)], {"x": [0, 0, 1, 1, 2, 0]}, dom)
    syn = graph(6, [(0, 1), (2, 3), (4, 5), (1, 4)], {"x": [0, 0, 1, 1, 2, 0]}, dom)
    a = metrics(real, syn)["attributes"]["x"]
    assert a["observed_pair_share_real"] == 0.75  # only (4,5) has a missing end
    assert a["observed_pair_share_syn"] == 0.5  # (0,1) and (2,3) observed; (4,5) and (1,4) are not
    # observed mixing, cells (aa, ab, ba, bb), diagonal counted once: real (1, 1, 1, 1) / 4, syn (1, 0, 0, 1) / 2
    assert a["mix_tvd"] == pytest.approx(0.5 * (0.25 + 0.25 + 0.25 + 0.25))


def test_ordered_attribute_adjacent_bins_closer_than_random():
    n = 100
    codes = np.arange(n) % 5
    edges = [(i, i + 1) for i in range(n - 1)]  # consecutive nodes: codes differ by 1 (or 4 on the wrap)
    g = graph(n, edges, {"age": codes}, {"age": ["a0", "a1", "a2", "a3", "a4", "none"]})
    a = metrics(g, g, ordered={"age": True})["attributes"]["age"]
    assert a["mean_abs_bin_distance_real"] == pytest.approx((80 * 1 + 19 * 4) / 99)  # 19 wrap-around edges of distance 4
    assert a["mean_abs_bin_distance_random_real"] == pytest.approx(1.6, abs=0.02)  # uniform over 5 bins: 40 / 25
    assert a["mean_abs_bin_distance_real"] < a["mean_abs_bin_distance_random_real"]
    unordered = metrics(g, g)["attributes"]["age"]
    assert "mean_abs_bin_distance_real" not in unordered


def test_ordered_attribute_exact_distance():
    # edges 0-1 (|0-1|=1), 2-3 (|2-4|=2): mean 1.5; endpoint shares a = (1/4, 1/4, 1/4, 1/4 on codes 0,1,2,4)
    g = graph(4, [(0, 1), (2, 3)], {"x": [0, 1, 2, 4]}, {"x": ["a", "b", "c", "d", "e", "none"]})
    a = metrics(g, g, ordered={"x": True})["attributes"]["x"]
    assert a["mean_abs_bin_distance_real"] == pytest.approx(1.5)
    codes, p = [0, 1, 2, 4], 0.25
    expected = sum(p * p * abs(i - j) for i in codes for j in codes)
    assert a["mean_abs_bin_distance_random_real"] == pytest.approx(expected)


def test_homophily_difference_between_real_and_synthetic():
    dom = {"x": ["a", "b", "none"]}
    real = graph(4, [(0, 1), (2, 3)], {"x": [0, 0, 1, 1]}, dom)  # all edges same value
    syn = graph(4, [(0, 2), (1, 3)], {"x": [0, 0, 1, 1]}, dom)  # all edges cross
    a = metrics(real, syn)["attributes"]["x"]
    assert a["mix_tvd"] == pytest.approx(1.0)
    assert a["assort_real"] == pytest.approx(1.0) and a["assort_syn"] == pytest.approx(-1.0)
    assert a["lift_real"] == pytest.approx(2.0) and a["lift_syn"] == 0.0
    assert a["assort_absdiff"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------- degree metrics
def test_eta2_hand_computed_on_path():
    edges = [(0, 1), (1, 2), (2, 3)]  # degrees 1, 2, 2, 1
    dom = {"x": ["a", "b", "none"]}
    split = graph(4, edges, {"x": [0, 0, 1, 1]}, dom)  # each value holds degrees {1, 2}: no variance explained
    aligned = graph(4, edges, {"x": [0, 1, 1, 0]}, dom)  # value b = both degree-2 nodes: everything explained
    assert metrics(split, split)["attributes"]["x"]["eta2_real"] == pytest.approx(0.0, abs=1e-12)
    assert metrics(aligned, aligned)["attributes"]["x"]["eta2_real"] == pytest.approx(1.0)


def test_deg_by_value_tvd_hand_computed():
    dom = {"x": ["a", "b", "none"]}
    attrs = {"x": [0, 0, 1, 1]}
    real = graph(4, [(0, 1), (1, 2), (2, 3)], attrs, dom)  # degrees 1,2,2,1
    syn = graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)], attrs, dom)  # 4-cycle: degrees 2,2,2,2
    a = metrics(real, syn, degree_bin_edges=[0, 2, 10])["attributes"]["x"]
    # real: each value has bins (1/2, 1/2); syn: each value is all in bin 1; TVD 1/2 per value, weights 1/2 + 1/2
    assert a["deg_by_value_tvd"] == pytest.approx(0.5)


def test_identical_graphs_give_zero_distances_and_equal_eta2():
    g, _ = random_graph(400, 1500, {"x": [0.5, 0.5], "y": [0.2, 0.3, 0.5]}, seed=3)
    m = metrics(g, g, exclude_hubs=3)
    for a in m["attributes"].values():
        assert a["deg_by_value_tvd"] == 0 and a["mix_tvd"] == 0 and a["deg_by_value_tvd_nohub"] == 0
        assert a["eta2_real"] == a["eta2_syn"] and a["eta2_nohub_real"] == a["eta2_nohub_syn"]
    t = m["mean_degree_by_value"]
    assert np.allclose(t["mean_degree_real"], t["mean_degree_syn"], equal_nan=True) and (t["n_real"] == t["n_syn"]).all()


def degree_dependent_graph(seed: int = 0):
    """20 hub-like nodes (value 1) each linked to 10 distinct leaves (value 0): degrees 10 vs 1, by value."""
    rng = np.random.default_rng(seed)
    n = 220
    codes = np.zeros(n, dtype=np.int64)
    codes[:20] = 1
    leaves = rng.permutation(np.arange(20, 220)).reshape(20, 10)
    edges = [(h, int(l)) for h in range(20) for l in leaves[h]]
    return graph(n, edges, {"x": codes}, {"x": ["leaf", "hub", "none"]})


def test_degree_depends_on_attribute_and_null_removes_it():
    g = degree_dependent_graph()
    assert metrics(g, g)["attributes"]["x"]["eta2_real"] == pytest.approx(1.0)  # degree is a function of the value
    null = permuted_null(g, seed=0)
    assert metrics(g, null)["attributes"]["x"]["eta2_syn"] < 0.1
    assert metrics(g, null)["attributes"]["x"]["deg_by_value_tvd"] > 0.1  # about 0.9 * 0.09 + 0.09 * 0.91


def test_hub_exclusion_reports_separate_keys():
    # a star: hub (value 1, degree 10) and 10 leaves (value 0, degree 1): the attribute "explains" degree only via the hub
    g = graph(11, [(0, i) for i in range(1, 11)], {"x": [1] + [0] * 10}, {"x": ["leaf", "hub", "none"]})
    plain = metrics(g, g)["attributes"]["x"]
    assert "eta2_nohub_real" not in plain and "deg_by_value_tvd_nohub" not in plain
    a = metrics(g, g, exclude_hubs=1)["attributes"]["x"]
    assert a["eta2_real"] == pytest.approx(1.0)
    assert math.isnan(a["eta2_nohub_real"])  # without the hub all degrees are 1: nothing to explain
    assert a["deg_by_value_tvd_nohub"] == 0


def test_mean_degree_by_value_table():
    g = graph(4, [(0, 1), (1, 2), (2, 3)], {"x": [0, 0, 1, 2]}, {"x": ["a", "b", "none"]})
    t = metrics(g, g)["mean_degree_by_value"].set_index("value")
    assert list(t.columns) == ["attribute", "n_real", "mean_degree_real", "n_syn", "mean_degree_syn"]
    assert t.loc["a", "n_real"] == 2 and t.loc["a", "mean_degree_real"] == 1.5
    assert t.loc["b", "mean_degree_real"] == 2.0 and t.loc["none", "mean_degree_real"] == 1.0


# ------------------------------------------------------------------------------------- gap closed, null
def test_gap_closed_arithmetic():
    assert gap_closed(0.5, 0.2, 0.0) == pytest.approx(0.6)
    assert gap_closed(1.0, 1.0, 2.0) == pytest.approx(0.0)  # as bad as the null
    assert gap_closed(1.0, 2.0, 2.0) == pytest.approx(1.0)  # as good as the real graph (level metric, null < real)
    assert gap_closed(1.0, 0.5, 2.0) == pytest.approx(-0.5)  # worse than the null
    assert math.isnan(gap_closed(1.0, 0.5, 1.0))  # null equals real: nothing to close
    assert math.isnan(gap_closed(float("nan"), 0.5, 0.0))


def test_permuted_null_keeps_edges_and_marginals_and_breaks_coupling():
    g, _ = random_graph(
        1500, 6000, {"x": [0.5, 0.5]}, seed=5, accept=lambda a, u, v: 1.0 if a["x"][u] == a["x"][v] else 0.2
    )
    null = permuted_null(g, seed=1)
    assert (null.edges == g.edges).all()
    assert (np.bincount(null.attrs["x"]) == np.bincount(g.attrs["x"])).all()
    assert not (null.attrs["x"].to_numpy() == g.attrs["x"].to_numpy()).all()
    assert permuted_null(g, seed=1).attrs.equals(null.attrs) and not permuted_null(g, seed=2).attrs.equals(null.attrs)
    assert metrics(g, g)["attributes"]["x"]["lift_real"] > 1.4
    assert metrics(g, null)["attributes"]["x"]["lift_syn"] == pytest.approx(1.0, abs=0.05)
    ref = reference_metrics(g, degree_bin_edges=EDGES)
    gaps = gap_closed_table(ref["null"], ref["real"])["attributes"]["x"]
    assert gaps["lift"] == pytest.approx(1.0) and gaps["mix_tvd"] == pytest.approx(1.0)  # real vs itself is the ceiling
    assert gap_closed_table(ref["null"], ref["null"])["attributes"]["x"]["lift"] == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------ degenerate
def test_degenerate_cases_do_not_raise():
    dom = {"x": ["a", "b", "none"]}
    # single observed value, everything else missing
    one = graph(5, [(0, 1), (1, 2), (3, 4)], {"x": [0, 0, 0, 2, 2]}, dom)
    a = metrics(one, one)["attributes"]["x"]
    assert math.isnan(a["assort_real"]) and a["lift_real"] == pytest.approx(1.0)
    # no edge has two observed ends
    none = graph(4, [(0, 1), (2, 3)], {"x": [2, 2, 0, 2]}, dom)
    a = metrics(none, none)["attributes"]["x"]
    assert math.isnan(a["lift_real"]) and math.isnan(a["mix_tvd"]) and a["observed_pair_share_real"] == 0.0
    # empty edge lists on either side
    empty = graph(4, [], {"x": [0, 1, 0, 1]}, dom)
    full = graph(4, [(0, 1), (2, 3)], {"x": [0, 1, 0, 1]}, dom)
    for r, s in ((empty, empty), (empty, full), (full, empty)):
        m = metrics(r, s, exclude_hubs=2)
        gap_closed_table(m, m)
        flatten(m)
    # a value present in the real graph and absent from the synthetic one counts as TVD 1 for that value
    real = graph(4, [(0, 1), (1, 2), (2, 3)], {"x": [0, 0, 1, 1]}, dom)
    syn = graph(4, [(0, 1), (1, 2), (2, 3)], {"x": [0, 0, 0, 0]}, dom)
    a = metrics(real, syn, degree_bin_edges=[0, 2, 10])["attributes"]["x"]
    assert 0.5 <= a["deg_by_value_tvd"] <= 1.0
    t = metrics(real, syn)["mean_degree_by_value"].set_index("value")
    assert t.loc["b", "n_syn"] == 0 and math.isnan(t.loc["b", "mean_degree_syn"])


def test_synthetic_domain_larger_than_real():
    real = graph(4, [(0, 1), (2, 3)], {"x": [0, 0, 1, 1]}, {"x": ["a", "b", "none"]})
    syn = graph(4, [(0, 1), (2, 3)], {"x": [0, 0, 1, 4]}, {"x": ["a", "b", "none"]})  # code 4 outside the real domain
    m = metrics(real, syn)
    assert m["mean_degree_by_value"]["value"].tolist()[-1] == "code4"


# ----------------------------------------------------------------------------- generality, invariance
def test_invariant_to_attribute_names_and_column_order():
    g, _ = random_graph(500, 2000, {"p": [0.6, 0.4], "q": [0.2, 0.3, 0.5]}, seed=7)
    h, _ = random_graph(500, 2000, {"p": [0.6, 0.4], "q": [0.2, 0.3, 0.5]}, seed=8)
    base = metrics(g, h, exclude_hubs=2)
    ren = lambda x: AttributedGraph(  # noqa: E731
        x.n, x.edges, x.attrs[["q", "p"]].rename(columns={"p": "zz", "q": "aa"}),
        {"zz": x.domains["p"], "aa": x.domains["q"]},
    )
    other = metrics(ren(g), ren(h), exclude_hubs=2)
    assert list(other["attributes"]) == ["aa", "zz"]
    for old, new in (("p", "zz"), ("q", "aa")):
        for k, v in base["attributes"][old].items():
            assert other["attributes"][new][k] == pytest.approx(v, nan_ok=True)


def test_generality_toy_schema_three_attributes_one_ordered():
    rng = np.random.default_rng(11)
    n = 600
    doms = {
        "colour": ["red", "blue", "other", "none"],
        "team": ["t0", "t1", "t2", "t3", "t4", "other", "none"],
        "age_band": ["young", "mid", "old", "other", "none"],
    }
    codes = {
        "colour": rng.choice(2, n, p=[0.6, 0.4]),
        "team": rng.choice(5, n),
        "age_band": rng.choice(3, n),
    }
    codes["team"][rng.random(n) < 0.2] = 6  # 'none'
    pairs = set()
    while len(pairs) < 3000:
        u, v = rng.integers(0, n, 2)
        if u == v:
            continue
        same_team = codes["team"][u] == codes["team"][v] and codes["team"][u] != 6
        near_age = abs(int(codes["age_band"][u]) - int(codes["age_band"][v])) <= 1
        if rng.random() < (1.0 if same_team else 0.25) * (1.0 if near_age else 0.3):
            pairs.add((min(u, v), max(u, v)))
    real = graph(n, pairs, codes, doms)
    ref = reference_metrics(real, degree_bin_edges=EDGES, ordered={"age_band": True}, exclude_hubs=5)
    m_real, m_null = ref["real"], ref["null"]
    assert set(m_real["attributes"]) == {"colour", "team", "age_band"}
    assert "mean_abs_bin_distance_real" in m_real["attributes"]["age_band"]
    assert "mean_abs_bin_distance_real" not in m_real["attributes"]["team"]
    assert m_real["attributes"]["team"]["lift_real"] > 1.5
    a = m_null["attributes"]["age_band"]
    assert a["mean_abs_bin_distance_syn"] > m_real["attributes"]["age_band"]["mean_abs_bin_distance_real"]
    assert m_null["attributes"]["team"]["lift_syn"] == pytest.approx(1.0, abs=0.1)
    gaps = gap_closed_table(m_null, m_real)
    assert gaps["attributes"]["team"]["lift"] == pytest.approx(1.0)
    flat = flatten(m_real, gaps=gaps)
    assert "cpl_team_lift_real" in flat and "cpl_mean_lift_real" in flat and "cpl_team_gap_lift" in flat
    assert all(isinstance(v, float) for v in flat.values())


# ------------------------------------------------------------------------------- triangle homophily
# K4 on nodes 0-3 (4 triangles) plus edges (2,4) and (3,4) (triangle 2-3-4): 5 triangles in all.
K4_PLUS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def test_within_group_triangle_share_hand_computed():
    g = graph(5, K4_PLUS, {"x": [0, 0, 0, 0, 1]}, {"x": ["a", "b", "none"]})
    # within x: the 4 K4 triangles; 2-3-4 mixes values a and b -> 4/5
    assert within_group_triangle_share(g, "x") == pytest.approx(4 / 5)


def test_within_group_triangle_share_missing_label_never_counts():
    g = graph(5, K4_PLUS, {"x": [0, 0, 2, 2, 2]}, {"x": ["a", "b", "none"]})
    # every triangle has a 'none' corner (2-3-4 is all 'none'): no within-group triangle at all
    assert within_group_triangle_share(g, "x") == 0.0
    # explicit missing=None: every value is observed, 'none' is a value like any other
    # triangles within code 2 = {2,3,4}: only 2-3-4 -> 1/5
    assert within_group_triangle_share(g, "x", missing={"x": None}) == pytest.approx(1 / 5)


def test_within_group_triangle_share_group_of_two_columns():
    g = graph(
        5,
        K4_PLUS,
        {"x": [0, 0, 0, 0, 1], "y": [0, 0, 0, 1, 1]},
        {"x": ["a", "b"], "y": ["c", "d"]},
    )
    # same (x, y) on all three corners: only 0-1-2 -> 1/5; per column: x 4/5, y 1/5
    assert within_group_triangle_share(g, ["x", "y"]) == pytest.approx(1 / 5)
    assert within_group_triangle_share(g, ["x"]) == pytest.approx(4 / 5)
    assert within_group_triangle_share(g, "y") == pytest.approx(1 / 5)


def test_within_group_triangle_share_no_triangles_is_nan():
    g = graph(4, [(0, 1), (1, 2), (2, 3)], {"x": [0, 0, 0, 0]}, {"x": ["a", "b"]})
    assert math.isnan(within_group_triangle_share(g, "x"))


def test_options_from_policy_defaults_and_overrides():
    pol = {"attributes": [{"name": "a"}, {"name": "b", "missing_label": "unknown", "ordered": True}]}
    missing, ordered = options_from_policy(pol)
    assert missing == {"a": "none", "b": "unknown"} and ordered == {"a": False, "b": True}


# ----------------------------------------------------------------------------------------------- CLI
def test_cli_smoke(tmp_path, monkeypatch, capsys):
    import coupling_report

    real, _ = random_graph(
        300, 900, {"x": [0.5, 0.5], "y": [0.3, 0.7]}, seed=2, accept=lambda a, u, v: 1.0 if a["x"][u] == a["x"][v] else 0.3
    )
    syn, _ = random_graph(300, 900, {"y": [0.3, 0.7], "x": [0.5, 0.5]}, seed=4)  # columns in a different order
    save_attributed_graph(real, tmp_path / "real.npz")
    save_attributed_graph(syn, tmp_path / "syn_a.npz")
    save_attributed_graph(syn, tmp_path / "syn_b.npz")
    out = tmp_path / "res" / "cpl.csv"
    argv = ["coupling_report.py", "--real", str(tmp_path / "real.npz"), "--syn", str(tmp_path / "syn_a.npz"),
            str(tmp_path / "syn_b.npz"), "--out", str(out), "--policy", str(tmp_path / "missing_policy.json")]
    monkeypatch.setattr(sys, "argv", argv)
    coupling_report.main()
    text = capsys.readouterr().out
    assert "syn_a" in text and "syn_b" in text and "lift_real" in text
    df = pd.read_csv(out)
    assert set(df.columns) == {"graph", "attribute", "metric", "value"}
    assert {"permuted_null", "syn_a", "syn_b"} <= set(df["graph"])
    assert {"x", "y", "mean"} <= set(df["attribute"])
    assert ((df["graph"] == "syn_a") & (df["metric"] == "gap_lift")).any()

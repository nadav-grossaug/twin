from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from graph_aim.eval.composition import (
    N_SHARE_BINS,
    composition_metrics,
    dominant_share,
    flatten,
    gap_closed_table,
    group_codes,
    independence_prediction,
    neigh_simpson_mean_by_bin,
    neigh_simpson_per_node,
    neigh_simpson_sums,
    neighbour_group_counts,
    permuted_null_by_degree,
    two_hop_same_rate,
)
from graph_aim.eval.coupling import permuted_null
from graph_aim.eval.evaluate import adjacency, degrees
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import AttributedGraph

BINS = [0, 2, 10]  # tiny graphs: bin 0 = degree < 2, bin 1 = degree >= 2


def graph(n: int, edges, attrs: dict, domains: dict | None = None) -> AttributedGraph:
    e = np.array(sorted({(int(min(x)), int(max(x))) for x in edges}), dtype=np.int64).reshape(-1, 2)
    df = pd.DataFrame({c: np.asarray(v, dtype=np.int64) for c, v in attrs.items()})
    if domains is None:
        domains = {c: [f"{c}{i}" for i in range(int(df[c].max()) + 1)] + ["other", "none"] for c in df.columns}
    return AttributedGraph(n=n, edges=e, attrs=df, domains=domains)


def star(codes=(9, 0, 0, 1, 2)) -> AttributedGraph:
    """Centre 0 joined to leaves 1..4; the centre's own code never enters its statistic."""
    return graph(5, [(0, i) for i in range(1, 5)], {"x": list(codes)}, {"x": [f"v{i}" for i in range(10)]})


def metrics(real, syn, groups=(("x",),), bin_edges=BINS, min_degree=1):
    return composition_metrics(real, syn, groups=[list(g) for g in groups], bin_edges=bin_edges, min_degree=min_degree)


def naive_simpson(g: AttributedGraph, col: str, min_degree: int) -> np.ndarray:
    """Plan 4.1 definition, node by node, in plain Python."""
    nbrs = [[] for _ in range(g.n)]
    for u, v in g.edges.tolist():
        nbrs[u].append(v)
        nbrs[v].append(u)
    codes = g.attrs[col].tolist()
    out = np.zeros(g.n)
    for u in range(g.n):
        d = len(nbrs[u])
        if d < min_degree:
            continue
        counts: dict[int, int] = {}
        for w in nbrs[u]:
            counts[codes[w]] = counts.get(codes[w], 0) + 1
        out[u] = sum((c / d) ** 2 for c in counts.values())
    return out


# ----------------------------------------------------------------------------------------- hand-computed
def test_simpson_hand_computed_on_star():
    g = star()
    adj, deg = adjacency(g), degrees(g)
    codes = g.attrs["x"].to_numpy()
    s = neigh_simpson_per_node(adj, codes, 10, deg, 1)
    # centre: neighbours (0, 0, 1, 2): (2/4)^2 + (1/4)^2 + (1/4)^2 = 0.375; leaves: one neighbour, Simpson 1
    assert s[0] == pytest.approx(0.375) and (s[1:] == 1.0).all()
    sums = neigh_simpson_sums(adj, codes, 10, deg, BINS, 1)
    assert sums == pytest.approx([4.0, 0.375])
    means, counts = neigh_simpson_mean_by_bin(adj, codes, 10, deg, BINS, 1)
    assert means == pytest.approx([1.0, 0.375]) and counts.tolist() == [4, 1]
    m = metrics(g, g)["groups"]["x"]
    assert m["simpson_by_degree"]["mean_real"] == pytest.approx((4 + 0.375) / 5)
    assert m["simpson_by_degree"]["absdiff"].tolist() == [0.0, 0.0]


def test_two_hop_hand_computed():
    g = star()
    counts = neighbour_group_counts(adjacency(g), g.attrs["x"].to_numpy(), 10)
    # 2-paths through the centre: C(4, 2) = 6, of which the pair of code-0 leaves: 1
    assert two_hop_same_rate(counts, degrees(g)) == pytest.approx(1 / 6)
    # edge ends: centre 4 of 8 (code 9), codes 0: 2, 1: 1, 2: 1 -> (1/2)^2 + (1/4)^2 + 2 (1/8)^2
    assert independence_prediction(g.attrs["x"].to_numpy(), degrees(g), 10) == pytest.approx(0.25 + 0.0625 + 2 / 64)
    path = graph(4, [(0, 1), (1, 2), (2, 3)], {"x": [0, 1, 0, 1]})  # 2-paths 0-1-2 (same) and 1-2-3 (same)
    m = metrics(path, path)["groups"]["x"]["two_hop_same_rate"]
    assert m["real"] == 1.0 and m["syn"] == 1.0 and m["absdiff"] == 0.0


def test_dominant_share_hand_computed():
    g = star()
    counts = neighbour_group_counts(adjacency(g), g.attrs["x"].to_numpy(), 10)
    assert dominant_share(counts, degrees(g), 1).tolist() == [0.5, 1.0, 1.0, 1.0, 1.0]
    assert dominant_share(counts, degrees(g), 2).tolist() == [0.5]
    h = metrics(g, g)["groups"]["x"]["dominant_share_hist"]
    expected = np.zeros(N_SHARE_BINS)
    expected[5], expected[9] = 0.2, 0.8  # 0.5 falls in [0.5, 0.6); 1.0 in the closed last bin
    assert h["real"].tolist() == expected.tolist() and h["tvd"] == 0.0
    syn = star((9, 0, 1, 2, 3))  # centre dominant share 1/4 -> bin 2; leaves unchanged
    h = metrics(g, syn)["groups"]["x"]["dominant_share_hist"]
    assert h["syn"][2] == pytest.approx(0.2) and h["tvd"] == pytest.approx(0.2)


def test_perfectly_concentrated_graph_gives_one():
    # two disjoint 4-cliques, one value each
    clique = lambda nodes: [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]
    g = graph(8, clique([0, 1, 2, 3]) + clique([4, 5, 6, 7]), {"x": [0] * 4 + [1] * 4})
    m = metrics(g, g, bin_edges=[0, 3, 10], min_degree=3)["groups"]["x"]
    assert m["simpson_by_degree"]["mean_real"] == 1.0
    assert np.isnan(m["simpson_by_degree"]["real"][0]) and m["simpson_by_degree"]["real"][1] == 1.0
    assert m["two_hop_same_rate"]["real"] == 1.0
    assert m["dominant_share_hist"]["real"][-1] == 1.0
    assert m["scalars"]["two_hop_indep_real"] == pytest.approx(0.5)


def test_matches_naive_definition_on_random_graph():
    rng = np.random.default_rng(3)
    n = 300
    pairs = {(min(u, v), max(u, v)) for u, v in rng.integers(0, n, (1500, 2)) if u != v}
    g = graph(n, pairs, {"x": rng.integers(0, 4, n), "y": rng.integers(0, 3, n)})
    adj, deg = adjacency(g), degrees(g)
    for dmin in (1, 3, 8):
        s = neigh_simpson_per_node(adj, g.attrs["x"].to_numpy(), 6, deg, dmin)
        assert s == pytest.approx(naive_simpson(g, "x", dmin))
        bins = [0, 4, 9, 20]
        b = degree_bin(deg, bins)
        sums = neigh_simpson_sums(adj, g.attrs["x"].to_numpy(), 6, deg, bins, dmin)
        assert sums == pytest.approx([s[b == k].sum() for k in range(3)])
    # group of two columns: row-major product code, same as the inside measurement
    from graph_aim.inside.measure import group_codes as inside_group_codes

    codes = group_codes(g, ["x", "y"], [6, 5])
    assert codes.tolist() == inside_group_codes(g.attrs, g.domains, ["x", "y"])[0].tolist()
    assert codes.tolist() == (g.attrs["x"].to_numpy() * 5 + g.attrs["y"].to_numpy()).tolist()


# ------------------------------------------------------------------------------------------------ null
def test_permuted_null_near_independence_prediction():
    rng = np.random.default_rng(0)
    n, m = 3000, 15000
    pairs = set()
    while len(pairs) < m:
        u, v = rng.integers(0, n, 2)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    codes = np.concatenate([np.repeat(np.arange(3), n // 3)])  # blocks of equal codes, nothing to do with the edges
    g = graph(n, pairs, {"x": codes})
    m = metrics(g, permuted_null(g, seed=1), bin_edges=[0, 5, 50], min_degree=5)["groups"]["x"]
    assert m["two_hop_same_rate"]["syn"] == pytest.approx(m["two_hop_same_rate"]["indep_syn"], abs=0.01)
    assert m["two_hop_same_rate"]["indep_syn"] == pytest.approx(1 / 3, abs=0.01)
    # i.i.d. attributes: E[s_u] = q + (1 - q) / d_u with q = sum a_g^2, so the mean sits a little above q
    q = m["two_hop_same_rate"]["indep_syn"]
    assert q < m["simpson_by_degree"]["mean_syn"] < q + 0.2


def test_min_degree_exclusion():
    g = star()
    adj, deg = adjacency(g), degrees(g)
    codes = g.attrs["x"].to_numpy()
    # leaves (degree 1) excluded: their bin has no node, the centre keeps its value
    assert neigh_simpson_sums(adj, codes, 10, deg, BINS, 2) == pytest.approx([0.0, 0.375])
    means, counts = neigh_simpson_mean_by_bin(adj, codes, 10, deg, BINS, 2)
    assert np.isnan(means[0]) and means[1] == pytest.approx(0.375) and counts.tolist() == [0, 1]
    m = metrics(g, g, min_degree=2)["groups"]["x"]
    assert m["simpson_by_degree"]["mean_real"] == pytest.approx(0.375)
    assert m["dominant_share_hist"]["real"][5] == 1.0
    # min_degree above every degree: no node, NaN means, zero histogram, NaN TVD
    m = metrics(g, g, min_degree=5)["groups"]["x"]
    assert np.isnan(m["simpson_by_degree"]["mean_real"]) and np.isnan(m["dominant_share_hist"]["tvd"])
    assert m["dominant_share_hist"]["real"].sum() == 0
    # two-hop rate ignores min_degree (every 2-path counts)
    assert m["two_hop_same_rate"]["real"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        metrics(g, g, min_degree=0)


def test_invariant_to_attribute_names_and_column_order():
    rng = np.random.default_rng(5)
    n = 200
    pairs = {(min(u, v), max(u, v)) for u, v in rng.integers(0, n, (800, 2)) if u != v}
    a = graph(n, pairs, {"p": rng.integers(0, 3, n), "q": rng.integers(0, 2, n)})
    syn = graph(n, {(min(u, v), max(u, v)) for u, v in rng.integers(0, n, (800, 2)) if u != v}, {"p": rng.integers(0, 3, n), "q": rng.integers(0, 2, n)})
    renamed = lambda g: AttributedGraph(g.n, g.edges, g.attrs[["q", "p"]].rename(columns={"p": "zz", "q": "aa"}), {"zz": g.domains["p"], "aa": g.domains["q"]})
    m1 = metrics(a, syn, groups=(("p",), ("p", "q")), bin_edges=[0, 3, 20], min_degree=3)
    m2 = metrics(renamed(a), renamed(syn), groups=(("zz",), ("zz", "aa")), bin_edges=[0, 3, 20], min_degree=3)
    assert list(m1["groups"]) == ["p", "p+q"] and list(m2["groups"]) == ["zz", "zz+aa"]
    for l1, l2 in zip(m1["groups"], m2["groups"]):
        assert m1["groups"][l1]["scalars"] == pytest.approx(m2["groups"][l2]["scalars"], nan_ok=True)
    assert m1["mean"] == pytest.approx(m2["mean"], nan_ok=True)
    # a group may be given as a bare column name
    assert metrics(a, syn, groups=("p",), bin_edges=[0, 3, 20], min_degree=3)["groups"]["p"]["scalars"] == pytest.approx(m1["groups"]["p"]["scalars"], nan_ok=True)
    with pytest.raises(ValueError):
        metrics(a, syn, groups=(("nope",),))


def test_synthetic_domain_larger_than_real_and_empty_groups():
    dom = {"x": ["a", "b", "none"]}
    real = graph(5, [(0, i) for i in range(1, 5)], {"x": [2, 0, 0, 0, 1]}, dom)
    syn = graph(5, [(0, i) for i in range(1, 5)], {"x": [2, 5, 5, 6, 6]}, dom)  # codes beyond the real labels
    m = metrics(real, syn)["groups"]["x"]
    assert m["simpson_by_degree"]["real"][1] == pytest.approx((3 / 4) ** 2 + (1 / 4) ** 2)
    assert m["simpson_by_degree"]["syn"][1] == pytest.approx(2 * (2 / 4) ** 2)
    assert m["n_groups"] == 7
    empty = composition_metrics(real, syn, groups=[], bin_edges=BINS, min_degree=1)
    assert empty["groups"] == {} and empty["mean"] == {} and flatten(empty) == {}


# ------------------------------------------------------------------------------------------- gap closed
def test_gap_closed_real_is_one_and_null_is_zero():
    rng = np.random.default_rng(11)
    n = 600
    # strongly concentrated: three blocks wired mostly inside
    block = np.repeat(np.arange(3), n // 3)
    pairs = set()
    while len(pairs) < 3000:
        u, v = rng.integers(0, n, 2)
        if u != v and (block[u] == block[v] or rng.random() < 0.05):
            pairs.add((min(u, v), max(u, v)))
    g = graph(n, pairs, {"x": block})
    kw = dict(groups=[["x"]], bin_edges=[0, 3, 30], min_degree=3)
    null = composition_metrics(g, permuted_null(g, 0), **kw)
    for m, expected in ((composition_metrics(g, g, **kw), 1.0), (null, 0.0)):
        gaps = gap_closed_table(null, m)["groups"]["x"]
        assert set(gaps) == {"simpson_mean", "two_hop_same_rate", "dominant_share_tvd"}
        assert all(v == pytest.approx(expected) for v in gaps.values())
    # a second null draw is near 0 (noise around the floor), never near 1
    other = composition_metrics(g, permuted_null(g, 1), **kw)
    assert abs(gap_closed_table(null, other)["mean"]["simpson_mean"]) < 0.1


def test_permuted_null_by_degree_keeps_degree_coupling():
    rng = np.random.default_rng(2)
    n = 500
    pairs = {(min(u, v), max(u, v)) for u, v in rng.integers(0, n, (3000, 2)) if u != v}
    g = graph(n, pairs, {"x": rng.integers(0, 3, n), "y": rng.integers(0, 2, n)})
    deg = degrees(g)
    g.attrs["x"] = (deg > np.median(deg)).astype(np.int64)  # attribute decided by the degree
    bins = [0, 5, 10, 20, 100]
    null = permuted_null_by_degree(g, bins, seed=0)
    assert np.array_equal(null.edges, g.edges) and list(null.attrs.columns) == ["x", "y"]
    assert not null.attrs.equals(g.attrs)  # something moved
    b = degree_bin(deg, bins)
    for c in ("x", "y"):
        for k in np.unique(b):
            assert np.array_equal(np.sort(null.attrs[c].to_numpy()[b == k]), np.sort(g.attrs[c].to_numpy()[b == k]))
    assert permuted_null_by_degree(g, bins, seed=0).attrs.equals(null.attrs)


# ---------------------------------------------------------------------------------------------- flatten
def test_flatten_column_naming():
    real = star((0, 0, 0, 1, 1))
    syn = star((0, 1, 0, 1, 0))
    g2 = lambda g: AttributedGraph(g.n, g.edges, g.attrs.assign(b=[0, 1, 0, 1, 0]), {**g.domains, "b": ["p", "q"]})
    m = metrics(g2(real), g2(syn), groups=(("x",), ("x", "b")))
    null = metrics(g2(real), permuted_null(g2(real)), groups=(("x",), ("x", "b")))
    row = flatten(m, gaps=gap_closed_table(null, m))
    assert all(k.startswith("cmp_") for k in row) and all(isinstance(v, float) for v in row.values())
    assert {
        "cmp_x_simpson_mean_real",
        "cmp_x_simpson_mean_syn",
        "cmp_x_simpson_mean_absdiff",
        "cmp_x_simpson_bin1_syn",
        "cmp_x_two_hop_same_rate_syn",
        "cmp_x_two_hop_indep_real",
        "cmp_x_dominant_share_tvd",
        "cmp_x_dominant_share_bin9_real",
        "cmp_x+b_simpson_mean_syn",
        "cmp_mean_simpson_mean_absdiff",
        "cmp_x_gap_simpson_mean",
        "cmp_x+b_gap_dominant_share_tvd",
        "cmp_mean_gap_two_hop_same_rate",
    } <= set(row)
    assert "cmp_x_simpson_bin2_syn" not in row  # two bins only
    assert row["cmp_x_simpson_mean_real"] == m["groups"]["x"]["scalars"]["simpson_mean_real"]
    other = flatten(m, gaps=gap_closed_table(null, m), gap_name="gapdeg")
    assert "cmp_x_gapdeg_simpson_mean" in other and "cmp_x_gap_simpson_mean" not in other
    assert flatten(m, prefix="z_")["z_x_simpson_mean_syn"] == row["cmp_x_simpson_mean_syn"]


def test_helper_input_validation():
    g = star()
    adj = adjacency(g)
    with pytest.raises(ValueError):
        neighbour_group_counts(adj, g.attrs["x"].to_numpy(), 5)  # code 9 out of range
    with pytest.raises(ValueError):
        neighbour_group_counts(adj, np.zeros(3, dtype=np.int64), 10)
    with pytest.raises(ValueError):
        neigh_simpson_sums(adj, g.attrs["x"].to_numpy(), 10, degrees(g), [1, 5], 1)  # bins must start at 0
    assert math.isnan(two_hop_same_rate(np.zeros((3, 2)), np.zeros(3)))
    assert math.isnan(independence_prediction(np.zeros(3, dtype=np.int64), np.zeros(3), 2))
    assert neighbour_group_counts(adjacency(AttributedGraph(0, np.zeros((0, 2), np.int64), pd.DataFrame({"x": []}), {"x": []})), np.zeros(0), 2).shape == (0, 2)

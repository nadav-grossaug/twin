"""community_metrics / walk_profile / walk_metrics on tiny graphs with known answers (graph_aim_walks.eval)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.eval.metrics import community_metrics, default_spec, walk_metrics, walk_profile
from graph_aim_walks.transfer.walkstats import WalkSpec

DOMAINS = {"school": ["s1", "s2", "other", "none"], "hometown": ["h1", "h2", "none"], "gender": ["g1", "g2"]}


def from_nx(G: nx.Graph, attrs: dict | None = None) -> AttributedGraph:
    n = G.number_of_nodes()
    edges = np.sort(np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2), axis=1)
    rng = np.random.default_rng(0)
    attrs = attrs or {c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()}
    return AttributedGraph(n, edges, pd.DataFrame(attrs), dict(DOMAINS))


def two_cliques(k: int = 8) -> AttributedGraph:
    """Two disconnected k-cliques; school and hometown = the clique (s1/h1 or s2/h2), gender alternates."""
    G = nx.disjoint_union(nx.complete_graph(k), nx.complete_graph(k))
    side = np.repeat([0, 1], k)
    return from_nx(G, {"school": side, "hometown": side, "gender": np.arange(2 * k) % 2})


SMALL = dict(n_walks=4000, length=64, held_out_windows=(64,), held_out_lags=(24, 32))


def test_community_metrics_of_two_cliques_and_a_ring():
    m = community_metrics(two_cliques())
    assert m["modularity_louvain"] == pytest.approx(0.5) and m["nw_conductance"] == 0.0
    assert m["spectral_gap"] == pytest.approx(0.0, abs=1e-9)  # two components: lambda_2 = 1
    ring = community_metrics(from_nx(nx.cycle_graph(30)))
    assert ring["spectral_gap"] == pytest.approx(1 - np.cos(2 * np.pi / 30), abs=1e-9)
    assert {"modularity_lpa", "n_parts", "nodes_in_large_parts_share", "nw_median_part_size"} <= set(ring)


def test_default_spec_keeps_the_plan_groups_present_in_the_columns():
    spec = default_spec(two_cliques())
    assert spec.groups == (("school",), ("hometown",))  # location is absent
    assert spec.share_attrs == ("school", "hometown", "gender")
    assert spec.dominance_pairs == (("hometown", "school"),)


def test_walks_inside_attribute_pure_cliques():
    g = two_cliques()
    p = walk_profile(g, **SMALL)
    # a walk never leaves its clique: the group never changes, at any lag, matched or held out
    for lab in ("school", "hometown"):
        assert all(v == pytest.approx(1.0) for v in p["same"][lab].values())
        assert set(p["same"][lab]) == {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}
    # the share of s1 (and of s2) in a walk is 0 or 1 by clique: variance 1/4 each; other / none never occur;
    # mean over the 4 values = 2 * 0.25 / 4
    for w in (8, 16, 32, 64):
        assert p["share_var"]["school"][w] == pytest.approx(2 * 0.25 / 4, abs=0.01)
    # K_8 return probability: (1/n)(1 + (n-1)(-1/(n-1))^t), n = 8
    n = 8
    exact = (1 + (n - 1) * (-1 / (n - 1)) ** np.arange(1, 6)) / n
    assert np.allclose(p["return"][:5], exact, atol=0.015)
    assert p["coverage"][-1] == pytest.approx(8.0, abs=0.01)  # a 64-step walk has (almost always) seen its clique
    assert p["coverage"].max() <= 8.0
    assert len(p["return"]) == 64 and len(p["spectrum"]) <= 16


def test_a_single_edge_alternates_and_covers_two_nodes():
    g = from_nx(nx.path_graph(2), {"school": [0, 1], "hometown": [0, 0], "gender": [0, 1]})
    p = walk_profile(g, WalkSpec(length=4, lags=(1, 2), windows=(2, 4), groups=(("school",),), share_attrs=("school",)),
                     n_walks=50, length=8, held_out_windows=(8,), held_out_lags=(3,))
    assert list(p["return"]) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert list(p["coverage"]) == [2] * 8
    assert p["same"]["school"] == {1: 0.0, 2: 1.0, 3: 0.0}
    assert all(v == 0.0 for v in p["share_var"]["school"].values())  # every window holds both nodes: shares 1/2


def test_walk_metrics_of_a_graph_against_itself_are_zero_gaps_and_unit_ratios():
    g = from_nx(nx.powerlaw_cluster_graph(120, 3, 0.4, seed=1))
    real = walk_profile(g, **SMALL)
    m = walk_metrics(g, g, real_profile=real)
    for k, v in m.items():
        if k.startswith("wlk_share_var_ratio"):
            assert v == pytest.approx(1.0), k
        else:
            assert v == pytest.approx(0.0, abs=1e-12), k
    assert {"wlk_share_var_ratio_w8", "wlk_share_var_ratio_w64", "wlk_share_var_ratio_school_w64",
            "wlk_attr_lag_tvd_school", "wlk_attr_lag_tvd_mean", "wlk_coverage_abs_heldout"} <= set(m)


def test_walk_metrics_see_weaker_regions_in_a_mixed_graph():
    real = two_cliques(10)
    G = nx.disjoint_union(nx.complete_graph(10), nx.complete_graph(10))
    G.add_edges_from([(i, 10 + i) for i in range(10)])  # many bridges: walks cross freely
    side = np.repeat([0, 1], 10)
    syn = from_nx(G, {"school": side, "hometown": side, "gender": np.arange(20) % 2})
    m = walk_metrics(real, syn, **{"n_walks": 3000})
    assert m["wlk_share_var_ratio_w32"] < 0.5 and m["wlk_share_var_ratio_w128"] < m["wlk_share_var_ratio_w8"]
    assert m["wlk_attr_lag_tvd_school"] > 0.2 and m["wlk_attr_same_abs_heldout"] > 0.2

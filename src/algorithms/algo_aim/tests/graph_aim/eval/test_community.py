"""Community metrics on tiny graphs with known answers (graph_aim.eval.community)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_aim.eval import community
from graph_aim.eval.evaluate import adjacency, degrees
from graph_aim.transfer.types import AttributedGraph

DOMAINS = {"a": ["x", "y", "other", "none"]}


def from_nx(G: nx.Graph) -> AttributedGraph:
    n = G.number_of_nodes()
    edges = np.sort(np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2), axis=1)
    attrs = pd.DataFrame({"a": np.arange(n) % 2})
    return AttributedGraph(n, edges, attrs, dict(DOMAINS))


def two_cliques(k: int = 8) -> AttributedGraph:
    return from_nx(nx.disjoint_union(nx.complete_graph(k), nx.complete_graph(k)))


def test_two_disconnected_cliques_are_two_closed_parts_of_modularity_one_half():
    m = community.partition_metrics(two_cliques(8))
    assert m["n_parts"] == 2 and m["n_parts_lpa"] == 2
    assert m["modularity_louvain"] == pytest.approx(0.5) and m["modularity_lpa"] == pytest.approx(0.5)
    assert m["nw_conductance"] == 0.0
    assert m["nw_median_part_size"] == 8
    assert m["nodes_in_large_parts_share"] == 0.0  # parts of 8 < 100


def test_large_parts_share_counts_nodes_in_parts_of_at_least_100():
    G = nx.disjoint_union(nx.complete_graph(100), nx.complete_graph(20))
    m = community.partition_metrics(from_nx(G))
    assert m["nodes_in_large_parts_share"] == pytest.approx(100 / 120)
    assert m["nw_median_part_size"] == 100  # the median node sits in the big part


def test_conductance_of_half_a_ring_is_two_cut_edges_over_its_volume():
    g = from_nx(nx.cycle_graph(20))
    half = [set(range(10)), set(range(10, 20))]
    c = community.conductances(adjacency(g), degrees(g), half)
    assert np.allclose(c, 2 / 20)  # cut 2, volume 20 on each side
    m = community.partition_metrics(g)
    assert 0 < m["nw_conductance"] < 1 and m["modularity_louvain"] > 0.5  # a ring splits into arcs


def test_conductance_is_zero_for_an_isolated_node_and_for_a_part_holding_everything():
    G = nx.complete_graph(4)
    G.add_node(4)
    g = from_nx(G)
    c = community.conductances(adjacency(g), degrees(g), [{0, 1, 2, 3}, {4}])
    assert list(c) == [0.0, 0.0]


def test_node_weighted_sizes_and_median():
    parts = [set(range(3)), {3}, set(range(4, 10))]
    assert sorted(community.node_part_sizes(parts)) == [1] + [3] * 3 + [6] * 6
    assert community.weighted_median(np.array([3, 1, 6]), np.array([3, 1, 6])) == 6
    assert community.weighted_median(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 1.0


def test_graph_without_edges_has_nan_modularity():
    g = AttributedGraph(3, np.zeros((0, 2), dtype=np.int64), pd.DataFrame({"a": [0, 1, 0]}), dict(DOMAINS))
    m = community.partition_metrics(g)
    assert np.isnan(m["modularity_louvain"]) and m["n_parts"] == 3 and m["nw_conductance"] == 0.0

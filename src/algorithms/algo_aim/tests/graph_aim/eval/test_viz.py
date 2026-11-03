"""Smoke and behaviour tests for graph_aim.eval.viz on small planted graphs (no real data, offline)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from graph_aim.eval import viz  # noqa: E402
from graph_aim.eval.evaluate import adjacency, degrees, local_clustering  # noqa: E402
from graph_aim.transfer.types import AttributedGraph  # noqa: E402

DOMAINS = {"a": ["a_x", "a_y", "a_z", "other", "none"], "b": ["b_p", "b_q", "other", "none"], "c": ["c_u", "other", "none"]}


def make_graph(seed: int, n: int = 260) -> AttributedGraph:
    g = nx.powerlaw_cluster_graph(n, 4, 0.3, seed=seed)
    edges = np.sort(np.array(list(g.edges()), dtype=np.int64), axis=1)
    rng = np.random.default_rng(seed)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n, edges, attrs, {c: list(d) for c, d in DOMAINS.items()})


@pytest.fixture(scope="module")
def graphs():
    return make_graph(1), {"synthetic A": make_graph(2), "synthetic B": make_graph(3)}


@pytest.mark.parametrize("name", [
    "degree_distribution", "clustering_by_degree", "degree_mixing", "neighbor_degree", "paths_and_cores",
    "attribute_marginals", "attribute_association",
])
def test_distribution_plots_return_figures(graphs, name):
    real, syns = graphs
    fig = getattr(viz, name)(real, syns)
    assert isinstance(fig, plt.Figure) and len(fig.axes) >= 1
    plt.close(fig)


def test_homophily_annotates_every_cell(graphs):
    real, syns = graphs
    fig = viz.homophily(real, syns, "a")
    texts = [t for ax in fig.axes for t in ax.texts]
    assert len(texts) == (1 + len(syns)) * len(DOMAINS["a"]) ** 2
    plt.close(fig)


def test_example_plots_return_figures(graphs):
    real, syns = graphs
    for fig in (viz.ego_examples(real, syns, "a", targets=(5, 12)), viz.bfs_sample(real, syns, "a", size=30, start_degree=8)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_coupling_plots_return_figures(graphs):
    real, syns = graphs
    for fig in (viz.degree_by_attribute(real, syns, ["a", "b"]), viz.homophily_lift(real, syns),
                viz.degree_attribute_coupling(real, syns)):
        assert isinstance(fig, plt.Figure) and len(fig.axes) >= 1
        plt.close(fig)


def test_degree_by_attribute_has_one_panel_per_attribute_with_value_labels(graphs):
    real, syns = graphs
    fig = viz.degree_by_attribute(real, syns, ["a", "b"])
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["x", "y", "z", "other", "none"]
    plt.close(fig)


def test_homophily_lift_names_every_attribute_and_draws_the_chance_line(graphs):
    real, syns = graphs
    fig = viz.homophily_lift(real, syns)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == list(DOMAINS)
    assert any((line.get_ydata() == np.array([1.0, 1.0])).all() for line in ax.lines)
    plt.close(fig)


def test_representative_node_has_the_nearest_degree_and_median_clustering():
    g = make_graph(5, n=400)
    adj, d = adjacency(g), degrees(g)
    clu = local_clustering(adj, d)
    v = viz._representative(d, clu, 9)
    nearest = d[d > 0][np.argmin(np.abs(d[d > 0] - 9))]
    assert d[v] == nearest
    same = np.sort(clu[d == nearest])
    assert clu[v] == same[len(same) // 2]


def test_category_colours_grey_out_other_and_none_and_keep_value_order():
    cols = viz._category_colours(["s_1", "s_2", "other", "none"])
    assert cols[0] != cols[1] and cols[2] == viz.NEUTRAL[0] and cols[3] == viz.NEUTRAL[1]
    assert viz._category_colours(["s_1", "s_2", "s_3", "other", "none"])[:2] == cols[:2]  # colour follows the value


def test_real_graph_is_always_ink_and_first(graphs):
    real, syns = graphs
    series = list(viz._series(real, syns))
    assert series[0][0] == "real" and series[0][2] == viz.REAL_COLOUR
    assert [s[2] for s in series[1:]] == viz.METHOD_COLOURS[: len(syns)]


def test_community_structure_has_four_panels_and_counts_parts(graphs):
    real, syns = graphs
    fig = viz.community_structure(real, syns)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 4
    assert sum("parts" in t.get_text() for t in fig.axes[3].texts) == 1 + len(syns)
    plt.close(fig)

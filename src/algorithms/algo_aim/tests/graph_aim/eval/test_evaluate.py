"""Tests for graph_aim.eval.evaluate on small networkx-generated attributed graphs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graph_aim.eval import evaluate as ev  # noqa: E402
from graph_aim.transfer.types import AttributedGraph  # noqa: E402


def make_graph(n: int = 300, seed: int = 0, clustered: bool = True) -> AttributedGraph:
    g = nx.powerlaw_cluster_graph(n, 5, 0.7 if clustered else 0.0, seed=seed)
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    a = (np.arange(n) * 3 // n).astype(np.int64)
    b = rng.integers(0, 4, n)
    attrs = pd.DataFrame({"a": a, "b": b})
    domains = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}
    return AttributedGraph(n=n, edges=edges, attrs=attrs, domains=domains)


def test_structure_metrics_match_networkx():
    g = make_graph()
    m = ev.structure_metrics(g)
    nxg = nx.Graph()
    nxg.add_nodes_from(range(g.n))
    nxg.add_edges_from(g.edges.tolist())
    assert np.isclose(m["transitivity"], nx.transitivity(nxg))
    assert np.isclose(m["avg_clustering"], nx.average_clustering(nxg))
    assert m["triangles"] == sum(nx.triangles(nxg).values()) / 3
    assert m["degeneracy"] == max(nx.core_number(nxg).values())
    assert np.isclose(m["assortativity"], nx.degree_assortativity_coefficient(nxg))
    assert m["largest_component_share"] == 1.0
    assert np.isclose(m["eigenvalues"][0], max(np.linalg.eigvalsh(nx.to_numpy_array(nxg))))
    k, p = m["degree_ccdf"]
    assert p[0] == 1.0 and np.all(np.diff(p) <= 0)


def test_structure_metrics_deterministic():
    g = make_graph()
    assert ev.structure_metrics(g)["mean_path_length"] == ev.structure_metrics(g)["mean_path_length"]


def test_compare_with_self_is_zero():
    g = make_graph()
    c = ev.compare(g, g, homophily_attrs=["a"])
    assert all(v == 0 for v in c.values()), c


def test_attribute_metrics_detects_shift():
    g = make_graph()
    s = AttributedGraph(g.n, g.edges, g.attrs.copy(), g.domains)
    s.attrs["b"] = 0
    am = ev.attribute_metrics(g, s, ["a"])
    assert am["tvd_1way"]["a"] == 0 and am["tvd_1way"]["b"] > 0.5
    assert am["tvd_2way_max"] > 0 and am["homophily_error"] == 0


def test_homophily_matrix_counts():
    g = make_graph()
    h = ev.homophily_matrix(g, ["a"])
    assert np.allclose(h, h.T)
    off = h.sum() - np.trace(h)
    assert h.shape == (4, 4) and np.trace(h) + off / 2 == len(g.edges)


def test_compare_flags_unclustered_and_table_and_plots():
    real = make_graph(seed=1)
    er = make_graph(seed=2, clustered=False)
    table = ev.comparison_table(real, {"clustered": make_graph(seed=3), "plain": er}, homophily_attrs=["a"])
    assert table.loc["clustering_abs", "plain"] > table.loc["clustering_abs", "clustered"]
    f1 = ev.plot_degree_ccdf(real, {"plain": er})
    f2 = ev.plot_clustering_by_degree(real, {"plain": er})
    assert hasattr(f1, "savefig") and hasattr(f2, "savefig")
    plt.close("all")


def test_empty_graph_does_not_crash():
    g = AttributedGraph(5, np.zeros((0, 2), np.int64), pd.DataFrame({"a": [0] * 5}), {"a": ["x", "other", "none"]})
    m = ev.structure_metrics(g)
    assert m["triangles"] == 0 and m["degeneracy"] == 0

"""Tests for graph_aim.eval.baselines."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_aim.eval import baselines as bl
from graph_aim.eval import evaluate as ev
from graph_aim.transfer.types import AttributedGraph


@pytest.fixture(scope="module")
def real() -> AttributedGraph:
    n = 400
    g = nx.powerlaw_cluster_graph(n, 5, 0.8, seed=0)
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=np.int64)
    rng = np.random.default_rng(0)
    attrs = pd.DataFrame({"a": rng.choice(3, n, p=[0.6, 0.3, 0.1]), "b": rng.integers(0, 4, n)})
    return AttributedGraph(n, edges, attrs, {"a": ["x", "y", "other"], "b": ["p", "q", "other", "none"]})


@pytest.mark.parametrize("name", list(bl.BASELINES))
def test_valid_graph_and_determinism(real, name):
    f = bl.BASELINES[name]
    s1, s2 = f(real, 1), f(real, 1)
    assert s1.n == real.n and np.array_equal(s1.edges, s2.edges)
    assert s1.edges.shape[1] == 2 and (s1.edges[:, 0] < s1.edges[:, 1]).all()
    assert len({tuple(e) for e in s1.edges.tolist()}) == len(s1.edges)
    assert abs(len(s1.edges) - len(real.edges)) <= 0.05 * len(real.edges)
    assert list(s1.attrs.columns) == list(real.attrs.columns)
    assert (s1.attrs.to_numpy() >= 0).all()


def test_attribute_marginals_preserved_independent(real):
    s = bl.chung_lu(real, 0)
    am = ev.attribute_metrics(real, s)
    assert am["tvd_1way_max"] < 0.1


def test_er_worse_than_bter_on_clustering(real):
    er, bt = bl.erdos_renyi(real, 0), bl.bter(real, 0)
    c_er = ev.compare(real, er)["clustering_abs"]
    c_bt = ev.compare(real, bt)["clustering_abs"]
    assert c_er > c_bt


def test_chung_lu_closer_degrees_than_er(real):
    assert ev.compare(real, bl.chung_lu(real, 0))["degree_ks"] < ev.compare(real, bl.erdos_renyi(real, 0))["degree_ks"]

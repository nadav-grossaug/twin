"""Tests for the link-prediction utility on small planted graphs."""

import numpy as np
import pandas as pd

from graph_aim.transfer.types import AttributedGraph
from graph_aim.eval.utility import LinkPredictionTask, _split_edges, link_prediction_utility


def _graph(n, edges, labels) -> AttributedGraph:
    e = np.array(sorted({(min(a, b), max(a, b)) for a, b in edges if a != b}), dtype=np.int64)
    attrs = pd.DataFrame({"school": labels, "other": np.zeros(n, dtype=np.int64)})
    return AttributedGraph(n, e, attrs, {"school": ["a", "b", "c", "d", "other", "none"], "other": ["x", "y", "other", "none"]})


def planted(n=400, k=8, p_in=0.2, p_out=0.002, seed=0) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % k
    iu = np.triu_indices(n, 1)
    same = labels[iu[0]] == labels[iu[1]]
    keep = rng.random(len(same)) < np.where(same, p_in, p_out)
    return _graph(n, zip(iu[0][keep], iu[1][keep]), labels)


def er(n, m, seed) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, n, size=(m, 2))
    return _graph(n, map(tuple, pairs), rng.integers(0, 4, size=n))


def test_real_as_synthetic_is_near_trtr():
    g = planted()
    r = link_prediction_utility(g, g, seed=0)
    assert r["auc_trtr"] > 0.8
    assert abs(r["auc_tstr"] - r["auc_trtr"]) < 0.05
    assert 0.9 < r["auc_ratio"] < 1.1


def test_random_graph_is_clearly_worse():
    g = planted()
    syn = er(g.n, len(g.edges), seed=1)
    good = link_prediction_utility(g, g, seed=0)
    bad = link_prediction_utility(g, syn, seed=0)
    assert bad["auc_tstr"] < good["auc_tstr"] - 0.1
    assert bad["auc_trtr"] == good["auc_trtr"]


def test_deterministic():
    g = planted(seed=2)
    syn = er(g.n, len(g.edges), seed=3)
    assert link_prediction_utility(g, syn, seed=5) == link_prediction_utility(g, syn, seed=5)


def test_no_leakage_between_train_and_test_edges():
    g = planted(seed=4)
    task = LinkPredictionTask(g, seed=0)
    key = lambda e: set(map(tuple, np.asarray(e).tolist()))  # noqa: E731
    train, test = key(task.train_graph.edges), key(task.test_pos)
    assert not train & test
    assert len(train) + len(test) == len(g.edges)
    assert key(_split_edges(g, 0)[1]) == test
    # negatives are never real edges
    assert not key(task.test_neg) & key(g.edges)


def test_empty_synthetic_graph_gives_chance():
    g = planted(seed=5)
    syn = _graph(g.n, [], np.zeros(g.n, dtype=np.int64))
    syn = AttributedGraph(syn.n, np.zeros((0, 2), dtype=np.int64), syn.attrs, syn.domains)
    assert link_prediction_utility(g, syn)["auc_tstr"] == 0.5

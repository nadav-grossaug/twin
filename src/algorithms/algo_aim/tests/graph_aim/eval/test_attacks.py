"""Tests for graph_aim.eval.attacks (empirical diagnostics only)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_aim.eval import attacks as at
from graph_aim.transfer.types import AttributedGraph


def _graph(n: int, seed: int, clustered: bool) -> AttributedGraph:
    if clustered:
        g = nx.powerlaw_cluster_graph(n, 5, 0.8, seed=seed)
    else:
        g = nx.gnm_random_graph(n, 5 * n, seed=seed)
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    attrs = pd.DataFrame({c: rng.integers(0, 5, n) for c in "abc"})
    dom = {c: ["v0", "v1", "v2", "other", "none"] for c in "abc"}
    return AttributedGraph(n, edges, attrs, dom)


def _permuted_copy(g: AttributedGraph, seed: int) -> AttributedGraph:
    perm = np.random.default_rng(seed).permutation(g.n)  # old id i -> new id perm[i]
    e = np.sort(perm[g.edges], axis=1)
    vals = np.zeros_like(g.attrs.to_numpy())
    vals[perm] = g.attrs.to_numpy()
    return AttributedGraph(g.n, e, pd.DataFrame(vals, columns=g.attrs.columns), g.domains)


@pytest.fixture(scope="module")
def real():
    return _graph(600, 0, True)


def test_exact_copy_is_flagged(real):
    copy = _permuted_copy(real, 5)
    r = at.run_attacks(real, copy)
    cn = r["closest_node"]
    assert cn["synthetic"]["median"] == 0 and cn["synthetic"]["share_below_1e-06"] == 1.0
    assert cn["holdout"]["median"] > 0 and cn["too_close"]
    sl = r["signature_linkage"]
    assert sl["rate_syn"] == 1.0 and sl["linkage_flag"]
    assert r["degree_signature"]["rate_syn"] == 1.0
    assert "not the differential privacy" in r["label"]


def test_fresh_er_graph_not_close(real):
    er = _graph(600, 7, False)
    r = at.run_attacks(real, er)
    assert not r["closest_node"]["too_close"]
    assert r["closest_node"]["synthetic"]["median"] > 0
    assert r["closest_node"]["ratio_median"] > 0.5
    assert not r["signature_linkage"]["linkage_flag"]


def test_deterministic(real):
    er = _graph(600, 7, False)
    assert at.run_attacks(real, er, 3) == at.run_attacks(real, er, 3)


def test_induced_subgraph_relabels(real):
    nodes = np.arange(0, 600, 2)
    sub = at._induced(real, nodes)
    assert sub.n == 300 and sub.edges.max() < 300 and (sub.edges[:, 0] < sub.edges[:, 1]).all()

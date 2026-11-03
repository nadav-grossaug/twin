"""Walk sampler and featurizer for the real-vs-fake classifier (docs/realfake/snap.md)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from graph_aim.eval.discriminator import GraphFeaturizer, GraphSample, WalkSampler
from graph_aim.transfer.types import AttributedGraph
from realfake.types import Sample

DOMAINS = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}


def _graph(n: int, edges: np.ndarray, seed: int = 0) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2), attrs, dict(DOMAINS))


def er_graph(n: int = 400, p: float = 0.03, seed: int = 0) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    return _graph(n, np.stack([iu[0][keep], iu[1][keep]], axis=1), seed)


def nx_graph(g: nx.Graph, seed: int = 0) -> AttributedGraph:
    e = np.array([(min(u, v), max(u, v)) for u, v in g.edges()], dtype=np.int64).reshape(-1, 2)
    return _graph(g.number_of_nodes(), e[np.lexsort((e[:, 1], e[:, 0]))], seed)


def _check_payload(s, source: AttributedGraph, k_max: int) -> None:
    p: GraphSample = s.payload
    assert 1 <= p.k <= min(k_max, source.n)
    assert len(p.attrs) == p.k and len(p.parent_deg) == p.k
    assert s.meta["size"] == float(p.k)
    e = p.edges
    assert e.dtype == np.int64 and e.ndim == 2 and e.shape[1] == 2
    if len(e):
        assert (e[:, 0] < e[:, 1]).all()
        assert e.max() < p.k and e.min() >= 0
        assert len(np.unique(e[:, 0] * p.k + e[:, 1])) == len(e)
    # visit order starts at the start node, which is the group id
    start = int(s.group_id)
    assert (p.attrs.iloc[0].to_numpy() == source.attrs.iloc[start].to_numpy()).all()
    deg = np.bincount(np.asarray(source.edges).ravel(), minlength=source.n)
    assert p.parent_deg[0] == deg[start]
    # within-sample degree can never exceed the parent degree
    wdeg = np.bincount(e.ravel(), minlength=p.k)
    assert (wdeg <= p.parent_deg).all()


def test_sampler_determinism_and_validity():
    g = er_graph()
    sampler = WalkSampler()
    s1 = sampler(g, np.random.default_rng(7))
    s2 = WalkSampler()(g, np.random.default_rng(7))
    assert s1.group_id == s2.group_id and s1.meta == s2.meta
    assert np.array_equal(s1.payload.edges, s2.payload.edges)
    assert np.array_equal(s1.payload.parent_deg, s2.payload.parent_deg)
    assert s1.payload.attrs.equals(s2.payload.attrs)

    rng = np.random.default_rng(1)
    for _ in range(25):
        s = sampler(g, rng)
        assert 10 <= s.payload.k <= 200
        _check_payload(s, g, 200)


def test_sampler_small_graph_and_isolated_nodes():
    # a graph smaller than k_max: k caps at n
    ring = nx.cycle_graph(30)
    g = nx_graph(ring)
    rng = np.random.default_rng(3)
    sampler = WalkSampler(k_min=10, k_max=200)
    for _ in range(10):
        s = sampler(g, rng)
        assert s.payload.k <= 30
        _check_payload(s, g, 200)

    # all nodes isolated: the sample is the single start node
    iso = _graph(12, np.zeros((0, 2), dtype=np.int64))
    for _ in range(5):
        s = sampler(iso, rng)
        assert s.payload.k == 1 and len(s.payload.edges) == 0
        assert s.payload.parent_deg[0] == 0
        _check_payload(s, iso, 200)

    # mixed: isolated nodes next to a clique; every start works
    cl = np.array([(u, v) for u in range(5) for v in range(u + 1, 5)], dtype=np.int64)
    mixed = _graph(9, cl)  # nodes 5..8 isolated
    for _ in range(30):
        s = sampler(mixed, rng)
        _check_payload(s, mixed, 200)


def _one_sample(seed: int = 5):
    g = er_graph(n=120, p=0.08, seed=2)
    return WalkSampler()(g, np.random.default_rng(seed))


def test_featurizer_invariant_to_column_order():
    feat = GraphFeaturizer(DOMAINS, homophily_attrs=["a"])
    s = _one_sample()
    v1 = feat(s)
    p = s.payload
    swapped = GraphSample(p.edges, p.attrs[list(reversed(list(p.attrs.columns)))], p.parent_deg, p.k)
    v2 = feat(Sample(payload=swapped, group_id=s.group_id, meta=s.meta))
    assert np.array_equal(v1, v2)
    assert len(v1) == len(feat.feature_names)
    assert sorted(f for names in feat.feature_groups.values() for f in names) == sorted(feat.feature_names)


def test_featurizer_invariant_to_node_relabelling():
    feat = GraphFeaturizer(DOMAINS, homophily_attrs=["a", "b"])
    s = _one_sample(seed=9)
    p = s.payload
    v1 = feat(s)
    rng = np.random.default_rng(0)
    perm = rng.permutation(p.k)  # old id i -> new id perm[i]
    inv = np.argsort(perm)
    e = perm[p.edges]
    e = np.stack([e.min(axis=1), e.max(axis=1)], axis=1)
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    relabelled = GraphSample(e, p.attrs.iloc[inv].reset_index(drop=True), p.parent_deg[inv], p.k)
    v2 = feat(Sample(payload=relabelled, group_id=s.group_id, meta=s.meta))
    assert np.allclose(v1, v2)


def test_featurizer_no_nan_on_edge_cases():
    feat = GraphFeaturizer(DOMAINS, homophily_attrs=["a"])
    one_node = GraphSample(
        np.zeros((0, 2), dtype=np.int64),
        pd.DataFrame({"a": [0], "b": [1]}),
        np.array([0], dtype=np.int64),
        1,
    )
    no_edges = GraphSample(
        np.zeros((0, 2), dtype=np.int64),
        pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, 3]}),
        np.array([4, 2, 1], dtype=np.int64),
        3,
    )
    for payload in (one_node, no_edges):
        v = feat(Sample(payload=payload, group_id=0))
        assert v.shape == (len(feat.feature_names),)
        assert np.isfinite(v).all()


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def test_structure_features_separate_er_from_clustered():
    """ER vs caveman samples must be separable on a single structure feature (AUC > 0.9)."""
    caveman = nx_graph(nx.connected_caveman_graph(25, 12), seed=4)  # n=300, m=1650
    er = er_graph(n=300, p=1650 / (300 * 299 / 2), seed=5)
    sampler = WalkSampler(k_min=10, k_max=100)
    feat = GraphFeaturizer(DOMAINS, homophily_attrs=None)
    i = feat.feature_names.index("structure:transitivity")
    rng = np.random.default_rng(13)
    cave_t = np.array([feat(sampler(caveman, rng))[i] for _ in range(60)])
    er_t = np.array([feat(sampler(er, rng))[i] for _ in range(60)])
    assert _rank_auc(cave_t, er_t) > 0.9

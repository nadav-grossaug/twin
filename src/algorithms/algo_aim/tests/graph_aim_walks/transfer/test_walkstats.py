"""The walk process and statistics shared by both sides (graph_aim_walks.transfer.walkstats)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_aim_walks.transfer import walkstats as W
from tests.graph_aim_walks.conftest import toy_graph


# ------------------------------------------------------------------------------------------ hand-computed
# 6 nodes: triangle 0-1-2, edge 3-4, node 5 isolated.
TINY_EDGES = np.array([[0, 1], [0, 2], [1, 2], [3, 4]])
TINY_DOMAINS = {"a": ["x", "y"], "b": ["p", "q", "r"]}
TINY_ATTRS = pd.DataFrame({"a": [0, 0, 1, 1, 1, 0], "b": [0, 1, 2, 1, 1, 1]})
TINY_SPEC = W.WalkSpec(length=4, lags=(1, 2), windows=(2, 4), share_bins=2, dominance_bins=2,
                       degree_bin_edges=(0, 2, 3), groups=(("a",),), share_attrs=("a",), dominance_pairs=(("a", "b"),))
# walk 0 from node 0: 0 -> 1 -> 0 -> 2 -> 1; walk 1 from 3: 3 -> 4 -> 3 -> 4 -> 3; walk 2 from the isolated node 5
TINY_STARTS = np.array([0, 3, 5])
TINY_U = np.array([[0.0, 0.0, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5], [0.3, 0.7, 0.1, 0.99]])


def tiny_paths():
    indptr, indices = W.csr(TINY_EDGES, 6)
    return W.simulate(indptr, indices, TINY_STARTS, TINY_U), np.diff(indptr)


def test_csr_sorted_neighbours():
    indptr, indices = W.csr(np.array([[2, 0], [0, 1], [1, 2], [3, 4]]), 6)
    assert indptr.tolist() == [0, 2, 4, 6, 7, 8, 8]
    assert indices.tolist() == [1, 2, 0, 2, 0, 1, 4, 3]


def test_hand_computed_paths_and_isolated_node_stays_put():
    paths, _ = tiny_paths()
    assert paths.tolist() == [[0, 1, 0, 2, 1], [3, 4, 3, 4, 3], [5, 5, 5, 5, 5]]


def test_hand_computed_tables():
    paths, deg = tiny_paths()
    c = W.walk_counts(paths, deg, TINY_ATTRS, TINY_DOMAINS, TINY_SPEC)
    assert set(c) == {"return", "coverage", "degree_lag", "attr_lag:a", "share:a", "dominance:a|b"}
    np.testing.assert_array_equal(c["return"], [1, 3, 1, 2])
    np.testing.assert_array_equal(c["coverage"], [5, 5, 6, 6])
    # degree bins: nodes 0-2 (degree 2) in bin 1, nodes 3-5 (degree 1, 1, 0) in bin 0
    np.testing.assert_array_equal(c["degree_lag"], [[[8, 0], [0, 4]], [[6, 0], [0, 3]]])
    np.testing.assert_array_equal(c["attr_lag:a"], [[[6, 1], [1, 4]], [[5, 1], [0, 3]]])
    # share of a: window 2 -> {0,1}: (1, 0); {3,4}: (0, 1); {5}: (1, 0); window 4 -> {0,1,2}: (2/3, 1/3), rest same
    np.testing.assert_array_equal(c["share:a"], [[[1, 2], [2, 1]], [[1, 2], [2, 1]]])
    # dominance a|b: window 2 all three walks (1, 1); window 4 walk 0 has b shares 1/3 each -> (1, 0)
    np.testing.assert_array_equal(c["dominance:a|b"], [[[0, 0], [0, 3]], [[0, 0], [1, 2]]])

    t = W.normalise(c, 3)
    np.testing.assert_allclose(t["return"], np.array([1, 3, 1, 2]) / 3)
    np.testing.assert_allclose(t["coverage"], np.array([5, 5, 6, 6]) / 3)
    np.testing.assert_allclose(t["attr_lag:a"][1], np.array([[5, 1], [0, 3]]) / 9)
    np.testing.assert_allclose(t["share:a"][0], np.array([[1, 2], [2, 1]]) / 3)


def test_share_one_lands_in_the_last_bin():
    spec = W.WalkSpec(length=3, lags=(1,), windows=(3,), share_bins=10, dominance_bins=5, degree_bin_edges=(0, 2),
                      share_attrs=("a",), dominance_pairs=(("a", "a"),))
    attrs = pd.DataFrame({"a": [1, 1, 0]})
    paths = np.array([[0, 1, 0, 1]])  # both nodes have value 1: share of value 1 is exactly 1.0
    c = W.walk_counts(paths, np.array([1, 1, 0]), attrs, {"a": ["u", "v"]}, spec)
    assert c["share:a"][0, 1, 9] == 1 and c["share:a"][0, 0, 0] == 1
    assert c["dominance:a|a"][0, 4, 4] == 1


def test_draw_pool_skips_isolated_nodes():
    starts, U = W.draw_pool(np.array([2, 0, 1, 0, 3]), 500, 7, np.random.default_rng(0))
    assert set(starts.tolist()) <= {0, 2, 4} and U.shape == (500, 7) and (U >= 0).all() and (U < 1).all()


# ------------------------------------------------------------------------------------------ toy graph
@pytest.fixture(scope="module")
def toy():
    g = toy_graph()
    spec = W.WalkSpec(length=12, lags=(1, 2, 4, 8), windows=(4, 12), share_bins=5, dominance_bins=3,
                      degree_bin_edges=(0, 2, 6, 16), groups=(("a",), ("a", "b")), share_attrs=("a", "b"),
                      dominance_pairs=(("a", "b"),))
    indptr, indices = W.csr(g.edges, g.n)
    deg = np.diff(indptr)
    starts, U = W.draw_pool(deg, 2000, spec.length, np.random.default_rng(3))
    paths = W.simulate(indptr, indices, starts, U)
    return g, spec, indptr, indices, deg, starts, U, paths


def test_counts_are_additive_over_walk_subsets(toy):
    g, spec, _, _, deg, _, _, paths = toy
    whole = W.walk_counts(paths, deg, g.attrs, g.domains, spec)
    idx = np.random.default_rng(0).permutation(len(paths))
    parts = [W.walk_counts(paths[i], deg, g.attrs, g.domains, spec) for i in np.array_split(idx, 3)]
    for k, v in whole.items():
        np.testing.assert_allclose(sum(p[k] for p in parts), v, err_msg=k)


def test_normalise_sums(toy):
    g, spec, _, _, deg, _, _, paths = toy
    t = W.normalise(W.walk_counts(paths, deg, g.attrs, g.domains, spec), len(paths))
    np.testing.assert_allclose(t["degree_lag"].sum(axis=(1, 2)), 1.0)
    for k, v in t.items():
        if k.startswith("attr_lag:") or k.startswith("dominance:"):
            np.testing.assert_allclose(v.sum(axis=(1, 2)), 1.0, err_msg=k)
        elif k.startswith("share:"):
            np.testing.assert_allclose(v.sum(axis=2), 1.0, err_msg=k)
    assert (t["return"] <= 1).all() and (t["return"] >= 0).all()
    assert (t["coverage"] >= 1).all() and (t["coverage"] <= np.arange(2, spec.length + 2)).all()
    with pytest.raises(ValueError, match="unknown statistic"):
        W.normalise({"nope": np.zeros(3)}, 1)


def test_walk_tables_reproducible(toy):
    g, spec = toy[0], toy[1]
    a = W.walk_tables(g.edges, g.n, g.attrs, g.domains, spec, n_walks=300, seed=5)
    b = W.walk_tables(g.edges, g.n, g.attrs, g.domains, spec, n_walks=300, seed=5)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_resimulate_from_is_identity_on_an_unchanged_graph(toy):
    _, _, indptr, indices, _, _, U, paths = toy
    rng = np.random.default_rng(1)
    walks = rng.choice(len(paths), size=300, replace=False)
    t0 = rng.integers(0, U.shape[1] + 1, size=300)
    np.testing.assert_array_equal(W.resimulate_from(indptr, indices, paths, U, walks, t0), paths[walks])


def _swap(edges: np.ndarray, rng) -> tuple[np.ndarray, set[int]]:
    """One degree-preserving double-edge swap (a-b, c-d) -> (a-d, c-b) that keeps the graph simple."""
    es = {tuple(e) for e in edges.tolist()}
    while True:
        i, j = rng.choice(len(edges), size=2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if len({a, b, c, d}) == 4 and new1 not in es and new2 not in es:
            keep = np.delete(edges, [i, j], axis=0)
            return np.vstack([keep, [new1, new2]]), {int(a), int(b), int(c), int(d)}


def test_resimulate_from_matches_full_resimulation_after_an_edge_swap(toy):
    g, _, indptr, indices, _, starts, U, paths = toy
    rng = np.random.default_rng(2)
    for _ in range(5):
        new_edges, changed = _swap(g.edges, rng)
        ip2, ix2 = W.csr(new_edges, g.n)
        full = W.simulate(ip2, ix2, starts, U)
        hit = np.isin(paths, list(changed))
        walks = np.flatnonzero(hit.any(axis=1))
        t0 = hit[walks].argmax(axis=1)  # first visit of a changed node
        assert len(walks) > 0
        np.testing.assert_array_equal(W.resimulate_from(ip2, ix2, paths, U, walks, t0), full[walks])
        untouched = np.setdiff1d(np.arange(len(paths)), walks)
        np.testing.assert_array_equal(full[untouched], paths[untouched])


def test_transition_spectrum_known_graphs():
    # complete graph K4: walk-matrix eigenvalues 1, -1/3, -1/3, -1/3; an isolated node is dropped
    k4 = np.array([[i, j] for i in range(4) for j in range(i + 1, 4)])
    np.testing.assert_allclose(W.transition_spectrum(k4, 5, k=3), [1, -1 / 3, -1 / 3])
    # two disjoint triangles: eigenvalue 1 twice
    tri2 = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]])
    np.testing.assert_allclose(W.transition_spectrum(tri2, 6, k=4), [1, 1, -0.5, -0.5], atol=1e-12)

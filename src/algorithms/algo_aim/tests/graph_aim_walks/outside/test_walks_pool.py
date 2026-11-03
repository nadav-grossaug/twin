"""The walk pool: fast counts == walkstats.walk_counts, incremental state == a from-scratch recount."""

from __future__ import annotations

import numpy as np
import pytest

from graph_aim_walks.outside import energy as en
from graph_aim_walks.outside import pool as pl
from graph_aim_walks.transfer import walkstats as ws
from tests.graph_aim_walks.conftest import toy_graph

SPEC = ws.WalkSpec(length=12, lags=(1, 2, 5), windows=(4, 12), degree_bin_edges=(0, 2, 6, 16),
                   groups=(("a",), ("a", "b")), share_attrs=("a", "b"), dominance_pairs=(("a", "b"),))


def _setup(sorted_rows: bool, P: int = 3000, seed: int = 0):
    g = toy_graph(n=120, p=0.08, seed=3)
    attrs = {c: g.attrs[c].to_numpy() for c in g.attrs.columns}
    ip, ix = ws.csr(g.edges, g.n)
    deg = np.diff(ip)
    starts, U = ws.draw_pool(deg, P, SPEC.length, np.random.default_rng(seed))
    arr = pl.node_arrays(deg, attrs, g.domains, SPEC)
    return g, attrs, pl.WalkPool(ip, ix, starts, U, arr, SPEC, sorted_rows=sorted_rows)


def _random_swaps(pool: pl.WalkPool, rng, k: int):
    """``k`` valid swaps with distinct edges and distinct new edges, as CSR slots."""
    e = pool.edges()
    n = pool.n
    rows = np.repeat(np.arange(n), pool.deg)
    key = rows * n + pool.indices
    order = np.argsort(key)
    exist = set((e[:, 0] * n + e[:, 1]).tolist())
    used, new, out = set(), set(), []
    for _ in range(50 * k):
        i, j = rng.integers(len(e), size=2)
        if i == j or i in used or j in used:
            continue
        a, b = e[i][::-1] if rng.random() < .5 else e[i]
        c, d = e[j][::-1] if rng.random() < .5 else e[j]
        if len({a, b, c, d}) < 4:
            continue
        k1, k2 = min(a, d) * n + max(a, d), min(c, b) * n + max(c, b)
        if k1 in exist or k2 in exist or k1 in new or k2 in new:
            continue
        used |= {i, j}
        new |= {k1, k2}
        slot = lambda x, y: order[np.searchsorted(key[order], x * n + y)]  # noqa: E731
        out.append((slot(a, b), slot(b, a), slot(c, d), slot(d, c)))
        if len(out) == k:
            break
    return [np.array(v, dtype=np.int64) for v in zip(*out)]


def test_pool_counts_equal_walkstats_walk_counts():
    g, attrs, pool = _setup(False)
    ref = ws.walk_counts(pool.paths, pool.deg, attrs, g.domains, SPEC)
    assert set(ref) | {"coverage_sq"} == set(pool.counts)
    for k, v in ref.items():
        assert np.array_equal(v, pool.counts[k]), k
    # sorted rows: the pool's walks are exactly walkstats.simulate on walkstats.csr
    _, _, sp_ = _setup(True)
    ip, ix = ws.csr(g.edges, g.n)
    assert np.array_equal(sp_.paths, ws.simulate(ip, ix, sp_.starts, sp_.U))


@pytest.mark.parametrize("sorted_rows", [False, True])
def test_incremental_pool_equals_recount_after_random_swaps(sorted_rows):
    g, attrs, pool = _setup(sorted_rows)
    rng = np.random.default_rng(5)
    degree = pool.deg.copy()
    for _ in range(6):
        sa, sb, sc, sd = _random_swaps(pool, rng, 8)
        pool.apply_swaps(sa, sb, sc, sd)
        pool.check()  # paths, slots and counts against a fresh simulation and recount
    e = pool.edges()
    assert len(e) == len(g.edges) and len(np.unique(e, axis=0)) == len(e) and (e[:, 0] < e[:, 1]).all()
    assert np.array_equal(np.bincount(e.ravel(), minlength=g.n), degree)
    if sorted_rows:  # sorted layout == walkstats.csr of the new edge set
        assert np.array_equal(pool.indices, ws.csr(e, g.n)[1])


def test_set_codes_updates_counts_exactly():
    g, attrs, pool = _setup(False)
    rng = np.random.default_rng(2)
    strides = {"group:a": 1, "group:a+b": len(g.domains["b"])}
    for _ in range(5):
        nodes = rng.choice(g.n, size=6, replace=False)
        old = pool.arr["attr:a"][nodes].copy()
        new = rng.integers(len(g.domains["a"]), size=6)
        pool.set_codes("a", nodes, old, new, strides)
        attrs["a"] = pool.arr["attr:a"].copy()
        ref = ws.walk_counts(pool.paths, pool.deg, attrs, g.domains, SPEC)
        for k, v in ref.items():
            assert np.array_equal(v, pool.counts[k]), k
    pool.check()


def test_walk_scores_are_the_linear_energy_term():
    g, attrs, pool = _setup(False)
    rng = np.random.default_rng(0)
    grads = {k: rng.standard_normal(v.shape) for k, v in pool.counts.items() if k != "coverage_sq"}
    s = pl.walk_scores(pool.paths, pool.values(pool.paths), pool.sizes, SPEC, grads)
    assert s.shape == (pool.P,)
    assert np.isclose(s.sum(), sum((grads[k] * pool.counts[k]).sum() for k in grads))


def test_energy_is_zero_against_itself_and_counts_cells():
    g, attrs, pool = _setup(False)
    real = ws.normalise({k: v for k, v in pool.counts.items() if k != "coverage_sq"}, pool.P)
    e = en.walk_energy(pool.counts, pool.P, real, pool.P, 10 * pool.P)
    assert all(v["energy"] == 0.0 for v in e.values())
    assert all(v["cells"] > 0 for v in e.values())
    # a second independent pool is about one noise unit per cell (proportions; lag tables are conservative)
    _, _, other = _setup(False, seed=9)
    e2 = en.walk_energy(other.counts, other.P, real, other.P, pool.P)
    for k in ("return", "share:a", "dominance:a|b"):
        assert 0.3 * e2[k]["cells"] < e2[k]["energy"] < 3.0 * e2[k]["cells"], (k, e2[k])


def test_triangle_change_matches_recount():
    g, attrs, pool = _setup(False)
    rng = np.random.default_rng(1)
    e0 = pool.edges()
    adj0 = en.adjacency(e0, g.n)
    t0 = en.triangles(adj0)
    sa, sb, sc, sd = _random_swaps(pool, rng, 10)
    a, b, c, d = pool.indices[sb], pool.indices[sa], pool.indices[sd], pool.indices[sc]
    pool.swap_edges(sa, sb, sc, sd)
    e1 = pool.edges()
    adj1 = en.adjacency(e1, g.n)
    dt = en.triangle_change(adj0, adj1, np.r_[np.c_[a, b], np.c_[c, d]], np.r_[np.c_[a, d], np.c_[c, b]], g.n)
    assert np.array_equal(t0 + dt, en.triangles(adj1))

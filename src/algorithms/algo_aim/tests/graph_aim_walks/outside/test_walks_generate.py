"""The graph-aim-walks generator on a planted-partition toy whose tables come from its own real graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_aim.outside.generate_v2 import generate_v2_with_info
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.outside.generate import generate_walks
from tests.graph_aim_walks.conftest import TOY_DOMAINS, toy_config

N, BLOCKS = 240, 4


def planted_graph(seed: int = 4) -> AttributedGraph:
    """Four dense blocks, sparse between; attribute ``a`` follows the block (80%), ``b`` half-follows it."""
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(BLOCKS), N // BLOCKS)
    iu = np.triu_indices(N, 1)
    p = np.where(z[iu[0]] == z[iu[1]], 0.16, 0.008)
    keep = rng.random(len(iu[0])) < p
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    a = np.where(rng.random(N) < 0.8, z, rng.integers(0, 4, N))
    b = np.where(rng.random(N) < 0.5, z, rng.integers(0, 4, N))
    attrs = pd.DataFrame({"a": a, "b": b})
    return AttributedGraph(N, edges, attrs, TOY_DOMAINS)


@pytest.fixture(scope="module")
def planted():
    from graph_aim_walks.inside.fit import fit_walks  # tests may import inside code

    g = planted_graph()
    return g, fit_walks(g, toy_config(n_walks=20000), seed=0)


OPTS = dict(pool_size=4000, cycles=8, structure_sweeps=1.0, label_sweeps=1.0, batch=200, label_batch=60,
            spectrum=False)


@pytest.fixture(scope="module")
def run(planted):
    g, art = planted
    init = generate_v2_with_info(art.base, seed=0)
    out, info = generate_walks(art, seed=0, init=init, check=True, **OPTS)  # check: from-scratch recount per cycle
    return init, out, info


def _total(e: dict) -> float:
    return sum(v["energy"] for v in e.values())


def test_degrees_exact_and_graph_simple(run):
    (g0, _), g, _ = run
    e = g.edges
    assert (e[:, 0] < e[:, 1]).all() and len(np.unique(e, axis=0)) == len(e)
    assert np.array_equal(np.bincount(e.ravel(), minlength=g.n), np.bincount(g0.edges.ravel(), minlength=g0.n))


def test_label_chain_keeps_1way_and_attr_degree(run, planted):
    (g0, _), g, _ = run
    art = planted[1]
    edges = next(m.params["bin_edges"] for m in art.base.measurements.values() if m.kind == "attr_degree")
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    s = degree_bin(deg, edges)
    for c in g.attrs.columns:
        x0, x1 = g0.attrs[c].to_numpy(), g.attrs[c].to_numpy()
        assert np.array_equal(np.bincount(x0, minlength=4), np.bincount(x1, minlength=4))
        t0 = np.bincount(s * 4 + x0, minlength=4 * (len(edges) - 1))
        t1 = np.bincount(s * 4 + x1, minlength=4 * (len(edges) - 1))
        assert np.array_equal(t0, t1)
    assert any((g0.attrs[c] != g.attrs[c]).any() for c in g.attrs.columns)  # the labels did move


def test_walk_energy_falls_from_the_v2_init(run):
    _, _, info = run
    first, final = info["cycles"][0]["fresh"], info["final"]  # both on pools the graph was never fitted to
    assert _total(final) < 0.5 * _total(first), (_total(first), _total(final))
    far = [k for k, v in first.items() if v["energy"] > 5 * v["cells"]]  # tables clearly off at the start
    fell = [k for k in far if final[k]["energy"] < first[k]["energy"]]
    assert len(fell) >= 0.75 * len(far), (far, fell)
    s = info["cycles"][0]["structure"]
    assert s["accepted"] > 0 and info["cycles"][0]["label"]["accepted"] > 0


def test_info_reports_per_cycle_tables(run, planted):
    _, _, info = run
    names = set(planted[1].tables) - {"transition_spectrum"}
    assert set(info["final"]) == names
    for c in info["cycles"]:
        assert set(c["fresh"]) == names and "clustering_by_degree" in c["base"]
        assert all(v["cells"] > 0 for v in c["fresh"].values())


def test_deterministic_per_seed(planted):
    _, art = planted
    init = generate_v2_with_info(art.base, seed=1)
    opts = dict(OPTS, cycles=1)
    g1, i1 = generate_walks(art, seed=1, init=init, **opts)
    g2, i2 = generate_walks(art, seed=1, init=init, **opts)
    assert np.array_equal(g1.edges, g2.edges) and g1.attrs.equals(g2.attrs)
    assert i1["final"] == i2["final"]
    g3, _ = generate_walks(art, seed=2, init=init, **opts)
    assert not np.array_equal(g1.edges, g3.edges)

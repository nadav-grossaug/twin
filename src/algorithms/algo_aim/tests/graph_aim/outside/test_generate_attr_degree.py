"""Tests for the attribute-conditional degree assignment (``attr_degree``) in the generator.

Artifacts are built by hand from a constructed degree / attribute sample (no fit), so these tests do not
depend on the inside zone producing ``attr_degree``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from graph_aim.outside import generate as gen
from graph_aim.outside.generate import generate
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import Artifact, Measurement, PrivacyLedger, Schema

EDGES = [0, 3, 6, 12, 10**6]  # 4 bins
DOM = {"kind": ["k0", "k1", "k2"], "zone": ["z0", "z1", "z2"], "lab": ["l0", "l1"]}


def toy_sample(n: int = 1500, seed: int = 0, couple: bool = True):
    """Attributes and degrees; degree depends strongly on ``kind`` and on ``zone == z2``."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    zone = np.where(rng.random(n) < 0.5, kind, rng.integers(0, 3, n))  # correlated with kind
    lab = rng.integers(0, 2, n)
    base = np.array([2, 5, 9])[kind] if couple else np.full(n, 4)
    bump = np.where((zone == 2) & couple, rng.choice([0, 7], size=n, p=[0.3, 0.7]), 0)
    deg = base + bump + rng.integers(0, 2, n)
    attrs = pd.DataFrame({"kind": kind, "zone": zone, "lab": lab}).astype(np.int64)
    return attrs, deg.astype(np.int64)


def table(attrs: pd.DataFrame, deg: np.ndarray, col: str, edges=EDGES) -> np.ndarray:
    t = np.zeros((len(DOM[col]), len(edges) - 1))
    np.add.at(t, (attrs[col].to_numpy(), degree_bin(deg, edges)), 1.0)
    return t


def make_artifact(attrs, deg, released=("kind", "zone", "lab"), noise=0.0, seed=0, n=None) -> Artifact:
    rng = np.random.default_rng(seed)
    ms: dict[str, Measurement] = {}

    def add(name, kind, values, **params):
        v = np.asarray(values, dtype=float)
        v = v + rng.normal(0, noise, v.shape) if noise else v
        ms[name] = Measurement(name, kind, v, "gaussian", noise, 0.0, 0.0, params)

    add("degree_hist", "degree_hist", np.bincount(deg))
    cols = list(DOM)
    for c in cols:
        add(f"attr_1way:{c}", "attr_1way", np.bincount(attrs[c], minlength=len(DOM[c])), attrs=[c])
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            t = np.zeros((len(DOM[a]), len(DOM[b])))
            np.add.at(t, (attrs[a].to_numpy(), attrs[b].to_numpy()), 1.0)
            add(f"attr_2way:{a}|{b}", "attr_2way", t, attrs=[a, b])
    for c in released:
        add(f"attr_degree:{c}", "attr_degree", table(attrs, deg, c), attrs=[c], cap=None, bin_edges=EDGES)
    schema = Schema(len(deg) if n is None else n, {k: list(v) for k, v in DOM.items()}, int(deg.max()), 0, [])
    return Artifact(schema, ms, PrivacyLedger(1e-6, float("inf"), float("inf")))


def run_assign(art: Artifact, n: int, seed: int):
    """Sample attributes and the pool like ``generate`` and run the private assignment."""
    rng = np.random.default_rng(seed)
    attrs = gen._sample_attributes(art, n, rng)
    tabs = gen._attr_degree_tables(art, attrs)
    h = gen._clean(art.measurements["degree_hist"].values)
    exact = h / h.sum() * n
    pool = np.minimum(np.searchsorted(np.cumsum(exact), rng.random() + np.arange(n), side="left"), len(h) - 1)
    deg, info = gen._assign_degrees_by_attributes(pool, attrs, tabs, rng)
    return attrs, pool, deg, info


def cond(t: np.ndarray) -> np.ndarray:
    return t / np.maximum(t.sum(axis=1, keepdims=True), 1.0)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum(axis=1).max())


@pytest.fixture(scope="module")
def sample():
    return toy_sample()


# ------------------------------------------------------------------------------------ assignment
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_table_reproduced_and_pool_preserved(sample, seed):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=("kind",))
    attrs, pool, deg, info = run_assign(art, len(deg0), seed)
    assert np.array_equal(np.sort(deg), np.sort(pool))  # histogram is exactly the pool
    assert info["residual"] < 1e-3
    tgt, ach = info["targets"][0], info["achieved"][0]
    assert np.abs(ach - tgt).max() <= 3.0  # within a few units of the raked target
    # conditional degree-bin distribution per value is the table's (the toy attributes are sampled from
    # the same tables, so the real conditional is the reference)
    real = cond(table(attrs0, deg0, "kind"))
    got = cond(table(attrs, deg, "kind"))
    assert tv(got, real) < 0.06


def test_independent_baseline_would_fail(sample):
    """Sanity: the shuffle fallback does not reproduce the strong dependence the assignment does."""
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=())
    rng = np.random.default_rng(0)
    attrs = gen._sample_attributes(art, len(deg0), rng)
    deg = gen._sample_degrees(art, len(deg0), rng)
    assert tv(cond(table(attrs, deg, "kind")), cond(table(attrs0, deg0, "kind"))) > 0.3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_correlated_attributes_both_reproduced(sample, seed):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=("kind", "zone", "lab"))
    attrs, pool, deg, info = run_assign(art, len(deg0), seed)
    assert np.array_equal(np.sort(deg), np.sort(pool))
    assert info["attrs"] == ["kind", "zone", "lab"]
    for col in ("kind", "zone", "lab"):
        assert tv(cond(table(attrs, deg, col)), cond(table(attrs0, deg0, col))) < 0.1, col
    for tgt, ach in zip(info["targets"], info["achieved"]):
        assert np.abs(ach - tgt).max() <= 3.0  # within a few units of the raked target


def test_attribute_counts_unchanged_by_assignment(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0)
    rng = np.random.default_rng(3)
    attrs = gen._sample_attributes(art, len(deg0), rng)
    before = attrs.copy()
    gen._sample_degrees(art, len(deg0), rng, attrs)
    pd.testing.assert_frame_equal(attrs, before)


def test_inconsistent_tables_give_finite_compromise(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=("kind", "zone"))
    # make the zone table claim the opposite of the kind table
    t = art.measurements["attr_degree:zone"].values
    art.measurements["attr_degree:zone"].values = t[:, ::-1].copy()
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert np.isfinite(info["residual"]) and info["residual"] > 0.05  # cannot satisfy both
    assert np.array_equal(np.sort(deg), np.sort(pool))
    assert all(np.isfinite(a).all() for a in info["achieved"])


def test_noisy_tables_still_valid(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, noise=15.0, seed=4)  # includes negatives
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert np.array_equal(np.sort(deg), np.sort(pool))
    assert np.isfinite(info["residual"])


def test_nan_inf_and_all_zero_tables(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=("kind", "zone"))
    v = art.measurements["attr_degree:kind"].values.copy()
    v[0, 0], v[1, 1] = np.nan, np.inf
    art.measurements["attr_degree:kind"].values = v
    art.measurements["attr_degree:zone"].values = np.zeros_like(art.measurements["attr_degree:zone"].values)
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert np.array_equal(np.sort(deg), np.sort(pool))
    assert np.isfinite(info["residual"])


def test_single_bin_and_subset_and_absent_value(sample):
    attrs0, deg0 = sample
    # B = 1: nothing to couple, still valid
    art = make_artifact(attrs0, deg0, released=())
    t = np.bincount(attrs0["kind"], minlength=3).astype(float)[:, None]
    art.measurements["attr_degree:kind"] = Measurement(
        "attr_degree:kind", "attr_degree", t, "gaussian", 0.0, 0.0, 0.0,
        {"attrs": ["kind"], "cap": None, "bin_edges": [0, 10**6]},
    )
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert np.array_equal(np.sort(deg), np.sort(pool))
    # a value absent from the sample (kind == 2 never occurs)
    a2 = attrs0.copy()
    a2["kind"] = np.where(a2["kind"] == 2, 1, a2["kind"])
    art = make_artifact(a2, deg0, released=("kind",))
    art.measurements["attr_1way:kind"].values = np.array([a2["kind"].eq(0).sum(), a2["kind"].eq(1).sum(), 0.0])
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert (attrs["kind"] != 2).all() and np.array_equal(np.sort(deg), np.sort(pool))


def test_differing_bin_edges_are_skipped(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0, released=("kind", "zone"))
    other = [0, 5, 10**6]
    art.measurements["attr_degree:zone"] = Measurement(
        "attr_degree:zone", "attr_degree", table(attrs0, deg0, "zone", other), "gaussian", 0.0, 0.0, 0.0,
        {"attrs": ["zone"], "cap": None, "bin_edges": other},
    )
    attrs, pool, deg, info = run_assign(art, len(deg0), 0)
    assert info["attrs"] == ["kind"]  # first table's edges win, the other one is ignored


def test_different_n_than_schema(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0)
    for n in (400, 3000):
        attrs, pool, deg, info = run_assign(art, n, 1)
        assert len(deg) == n and np.array_equal(np.sort(deg), np.sort(pool))
        assert tv(cond(table(attrs, deg, "kind")), cond(table(attrs0, deg0, "kind"))) < 0.15


def test_tiny_n():
    attrs0, deg0 = toy_sample(n=200)
    art = make_artifact(attrs0, deg0)
    g = generate(art, n=5, seed=0)
    assert g.n == 5 and (g.edges[:, 0] < g.edges[:, 1]).all()


def test_deterministic_per_seed(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0)
    a = run_assign(art, 600, 5)[2]
    b = run_assign(art, 600, 5)[2]
    c = run_assign(art, 600, 6)[2]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ------------------------------------------------------------------------------------ end to end
def test_generate_end_to_end_valid_and_coupled(sample):
    attrs0, deg0 = sample
    art = make_artifact(attrs0, deg0)
    g = generate(art, seed=2, rewire_rounds=0)
    assert g.n == len(deg0)
    e = g.edges
    assert (e[:, 0] < e[:, 1]).all() and len(np.unique(e[:, 0] * g.n + e[:, 1])) == len(e)
    d = np.bincount(e.ravel(), minlength=g.n)
    got = cond(table(g.attrs, d, "kind"))
    assert tv(got, cond(table(attrs0, deg0, "kind"))) < 0.2
    g2 = generate(art, seed=2, rewire_rounds=0)
    assert np.array_equal(g.edges, g2.edges)


# ---------------------------------------------------------------------------------- fallback
# Digests of generate() on the unmodified generator (before attr_degree existed); without an
# attr_degree measurement the output, including random-number consumption, must stay byte-identical.
FALLBACK_DIGESTS = {
    0: "8096c29c161cb288799eb340b811fcc387e10bcd39af575c309be0bbdd2a3e8f",
    7: "903a4c5e451fce0f1595669f78fa3bbdadd2ba214ef8b66c2525203b48fd74e5",
}


@pytest.mark.parametrize("seed", [0, 7])
def test_no_attr_degree_is_byte_identical_to_old_generator(seed):
    from test_generate import exact_artifact, planted_graph

    art = exact_artifact(planted_graph(n=300, m=5, seed=3))
    assert not [m for m in art.measurements.values() if m.kind == "attr_degree"]
    out = generate(art, seed=seed)
    h = hashlib.sha256(out.edges.tobytes() + out.attrs.to_numpy().astype("int64").tobytes()).hexdigest()
    assert h == FALLBACK_DIGESTS[seed]

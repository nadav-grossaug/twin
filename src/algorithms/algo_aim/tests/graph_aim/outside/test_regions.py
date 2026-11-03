"""Tests for the latent regions of the structure initializer (package E, plan section 7.5).

``generate(..., regions=Z, within_region_frac=q)`` with the defaults must stay byte-identical to the v1.1 sampler;
with regions, phase-one blocks never cross a region and keep the degree invariant; the planted ``Z`` becomes
visible in the output as ``q`` grows; ``generate_v2`` calibrates ``(R, q)`` from the released ``spectrum``
(artifacts fitted with ``graph_aim.inside.fit`` and the v2 config keys: tests may import inside code).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_aim.outside import generate as G
from graph_aim.outside import generate_v2 as V2
from graph_aim.outside.generate import generate
from graph_aim.outside.generate_v2 import calibrate_regions, generate_v2, generate_v2_with_info, region_labels
from graph_aim.transfer.types import AttributedGraph, Measurement

from test_generate import exact_artifact, planted_graph
from test_generate_attr_degree import FALLBACK_DIGESTS

REPO = Path(__file__).resolve().parents[3]
SEED = 3


# ---------------------------------------------------------------------------------------- helpers
def digest(g: AttributedGraph) -> str:
    return hashlib.sha256(g.edges.tobytes() + g.attrs.to_numpy().astype("int64").tobytes()).hexdigest()


def degrees(g: AttributedGraph) -> np.ndarray:
    return np.bincount(g.edges.ravel(), minlength=g.n)


def sample_pool(art, n: int, seed: int):
    """Attributes, degree pool, group codes and v1 blocks exactly as ``generate`` / ``calibrate_regions`` draw them."""
    rng = np.random.default_rng(seed)
    attrs = G._sample_attributes(art, n, rng)
    deg = G._sample_degrees(art, n, rng, attrs)
    grp0, _ = G._group_codes(art, attrs, n)
    blocks = G._phase_one_blocks(deg, grp0, np.random.default_rng([seed, 1]))
    return attrs, deg, grp0, blocks


def modularity_of(z: np.ndarray, edges: np.ndarray) -> float:
    m = len(edges)
    R = int(z.max()) + 1
    vol = np.bincount(z[edges.ravel()], minlength=R).astype(float)
    same = z[edges[:, 0]] == z[edges[:, 1]]
    inner = np.bincount(z[edges[:, 0]][same], minlength=R).astype(float)
    return float((inner / m - (vol / (2 * m)) ** 2).sum())


def cross_fraction(z: np.ndarray, edges: np.ndarray) -> float:
    return float((z[edges[:, 0]] != z[edges[:, 1]]).mean())


def ks_to_pool(g: AttributedGraph, pool: np.ndarray) -> float:
    a, b = np.sort(degrees(g)), np.sort(pool)
    grid = np.arange(max(a.max(), b.max()) + 2)
    return float(np.abs(np.searchsorted(a, grid, side="right") - np.searchsorted(b, grid, side="right")).max() / len(a))


def toy_graph(kind: str, n: int = 600, seed: int = 1) -> AttributedGraph:
    """``er``: Erdos-Renyi-like (no mesoscale structure); ``pp``: four planted partitions of 150 nodes."""
    if kind == "er":
        gx = nx.gnp_random_graph(n, 0.03, seed=seed)
    else:
        gx = nx.planted_partition_graph(4, n // 4, 0.08, 0.004, seed=seed)
    rng = np.random.default_rng(seed)
    domains = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}
    attrs = pd.DataFrame({c: rng.integers(0, 4, n) for c in domains}).astype(np.int64)
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in gx.edges()), dtype=np.int64)
    return AttributedGraph(n, edges, attrs, domains)


def fitted_v2_artifact(g: AttributedGraph, noise: bool = False, seed: int = 0):
    """Fit with the v2 config keys (spectrum share > 0) on a toy graph; noise-free unless asked."""
    from graph_aim.inside.fit import fit  # tests may import inside code

    cfg = json.loads((REPO / "snap" / "inside" / "config" / "graph_aim_v2.json").read_text())
    cfg.update(D=30, max_degree=60, homophily_attrs=None, homophily_groups=None, neigh_simpson_groups=None,
               attr_degree_attrs=None, spectrum_k=16, noise=noise, epsilon_edge=4.0, epsilon_attr=4.0)
    return fit(g, cfg, seed=seed)


# --------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def planted() -> AttributedGraph:
    return planted_graph()  # 1500 nodes, hubs, clustering 0.6-ish


@pytest.fixture(scope="module")
def artifact(planted):
    return exact_artifact(planted)


@pytest.fixture(scope="module")
def pool(artifact, planted):
    return sample_pool(artifact, planted.n, SEED)


@pytest.fixture(scope="module")
def block_z(pool, planted):
    _, _, _, blocks = pool
    return region_labels(blocks, planted.n, 8, np.random.default_rng([SEED, 3]))


# ------------------------------------------------------------------------------------ byte identity
@pytest.mark.parametrize("seed", [0, 7])
def test_defaults_are_byte_identical_to_the_v1_generator(seed):
    art = exact_artifact(planted_graph(n=300, m=5, seed=3))
    assert digest(generate(art, seed=seed, regions=None, within_region_frac=0.0)) == FALLBACK_DIGESTS[seed]
    # regions_R=1 pinned in v2 (no spectrum, no Simpson kind): still the v1 output
    out, info = generate_v2_with_info(art, seed=seed, regions_R=1, within_region_frac=0.0)
    assert digest(out) == FALLBACK_DIGESTS[seed]
    assert info["regions_R"] == 1 and info["within_region_frac"] == 0.0 and info["regions"] is None
    assert info["calibration"]["skipped"] == "pinned" and info["calibration"]["grid"] == []


@pytest.mark.parametrize("seed", [0, 7])
def test_no_spectrum_means_identity(seed):
    art = exact_artifact(planted_graph(n=300, m=5, seed=3))
    assert not [m for m in art.measurements.values() if m.kind == "spectrum"]
    out, info = generate_v2_with_info(art, seed=seed)
    assert digest(out) == FALLBACK_DIGESTS[seed]
    assert (info["regions_R"], info["within_region_frac"]) == (1, 0.0)
    assert info["calibration"]["skipped"] == "no spectrum measurement"


def test_a_single_region_label_array_is_the_identity(artifact):
    """``regions`` of all zeros with ``q = 0``: the blocks are the v1 ones and the output is identical."""
    n = artifact.schema.n
    a = generate(artifact, seed=SEED)
    b = generate(artifact, seed=SEED, regions=np.zeros(n, dtype=np.int64), within_region_frac=0.0)
    assert np.array_equal(a.edges, b.edges) and a.attrs.equals(b.attrs)


# ------------------------------------------------------------------------------------ phase one
def test_blocks_never_cross_regions_and_keep_the_degree_invariant(pool, planted):
    attrs, deg, grp0, _ = pool
    n = planted.n
    z = np.random.default_rng(5).integers(0, 8, n)  # per-node random regions (plan 7.1), the generic case
    blocks = G._phase_one_blocks(deg, grp0, np.random.default_rng([SEED, 1]), regions=z)
    assert blocks
    seen = np.zeros(n, dtype=bool)
    for b in blocks:
        assert len(b) >= 3
        assert len(np.unique(z[b])) == 1  # never crosses a region
        assert (deg[b] >= deg[b[0]]).all()  # every member has degree >= the block's d
        assert len(b) <= deg[b[0]] + 1
        assert not seen[b].any()
        seen[b] = True
    assert (deg[~seen] < 2).sum() + ((deg >= 2) & ~seen).sum() == (~seen).sum()
    assert ((deg >= 2) & ~seen).sum() <= 2 * 8  # at most the <3 leftover per region
    # phase one never gives a node more than its degree
    cl_edges, c_target = G._clustering_target(planted and exact_artifact(planted))
    e1 = G._phase_one(deg, cl_edges, np.cbrt(c_target), grp0, np.random.default_rng([SEED, 1]), regions=z)
    d1 = np.bincount(e1.ravel(), minlength=n)
    assert (d1 <= deg).all()
    assert (z[e1[:, 0]] == z[e1[:, 1]]).all()


def test_block_aligned_labels_reproduce_the_v1_blocks(pool, block_z, planted):
    """A ``Z`` constant on the v1 blocks re-cuts exactly those blocks (so the degree calibration is untouched)."""
    _, deg, grp0, blocks = pool
    with_z = G._phase_one_blocks(deg, grp0, np.random.default_rng([SEED, 1]), regions=block_z)
    assert {frozenset(b.tolist()) for b in blocks} == {frozenset(b.tolist()) for b in with_z}
    for b in with_z:
        assert len(np.unique(block_z[b])) == 1


def test_region_labels_are_balanced_and_cover_every_node(pool, planted):
    _, _, _, blocks = pool
    n = planted.n
    largest = max(len(b) for b in blocks)
    for R in (1, 2, 8, 64):
        z = region_labels(blocks, n, R, np.random.default_rng(0))
        assert z.shape == (n,) and z.min() == 0 and z.max() == R - 1
        sizes = np.bincount(z, minlength=R)
        assert sizes.max() - sizes.min() <= largest
        for b in blocks:
            assert len(np.unique(z[b])) == 1
    assert (region_labels(blocks, n, 1, np.random.default_rng(0)) == 0).all()


# ------------------------------------------------------------------------------------ phase two
def test_degree_pool_is_kept_with_regions(artifact, pool, block_z):
    """Degrees stay the pool to the same tolerance as v1 (hubs can end slightly short in both)."""
    _, deg, _, _ = pool
    base = generate(artifact, seed=SEED)
    ref_ks, ref_over = ks_to_pool(base, deg), int((degrees(base) - deg).max())
    for q in (0.0, 0.5, 0.9):
        g = generate(artifact, seed=SEED, regions=block_z, within_region_frac=q)
        d = degrees(g)
        # phase two's multinomial overshoot, as in v1 (a few units on a handful of nodes, < 1% of the volume)
        assert (d - deg).max() <= ref_over + 6
        assert (d - deg).clip(0).sum() <= 0.02 * deg.sum()
        assert ks_to_pool(g, deg) <= max(ref_ks, 0.005) + 0.01
        assert len(g.edges) >= 0.97 * deg.sum() / 2


def test_modularity_rises_and_cross_region_fraction_falls_with_q(artifact, block_z):
    mods, cross = [], []
    for q in (0.0, 0.5, 0.9):
        g = generate(artifact, seed=SEED, regions=block_z, within_region_frac=q)
        mods.append(modularity_of(block_z, g.edges))
        cross.append(cross_fraction(block_z, g.edges))
    assert mods[0] < mods[1] < mods[2], mods
    assert cross[0] > cross[1] > cross[2], cross
    assert mods[2] > 0.33 and cross[2] < 0.5  # 0.39 / 0.44 on this graph (hubs and sparse bins stay global)
    # a per-node random Z is invisible in the v1 output (modularity near 0); the block-aligned Z is not, because
    # the v1 blocks are communities of their own (modularity 0.2-0.3), and q raises it from there
    base = generate(artifact, seed=SEED)
    z_nodes = np.random.default_rng(9).integers(0, 8, base.n)
    assert abs(modularity_of(z_nodes, base.edges)) < 0.05
    assert modularity_of(block_z, base.edges) < mods[1]


def test_rewiring_keeps_the_within_region_edge_count(artifact, pool, block_z):
    attrs, deg, _, _ = pool
    cl_edges, c_target = G._clustering_target(artifact)
    rho = np.cbrt(c_target)
    kw = dict(regions=block_z, within_region_frac=0.9)
    e0 = G._generate_structure(artifact, deg, attrs, np.random.default_rng([SEED, 1]), cl_edges, rho, 0, **kw)
    e1 = G._generate_structure(artifact, deg, attrs, np.random.default_rng([SEED, 1]), cl_edges, rho, 40, **kw)
    assert len(e0) == len(e1)
    assert not np.array_equal(np.sort(e0, axis=0), np.sort(e1, axis=0))  # the rewiring did something
    assert (block_z[e0[:, 0]] == block_z[e0[:, 1]]).sum() == (block_z[e1[:, 0]] == block_z[e1[:, 1]]).sum()
    assert np.array_equal(np.bincount(e0.ravel(), minlength=len(deg)), np.bincount(e1.ravel(), minlength=len(deg)))


def test_generate_validates_the_region_arguments(artifact):
    n = artifact.schema.n
    with pytest.raises(ValueError):
        generate(artifact, seed=SEED, regions=np.zeros(n - 1, dtype=np.int64))
    with pytest.raises(ValueError):
        generate(artifact, seed=SEED, regions=np.full(n, -1))
    with pytest.raises(ValueError):
        generate(artifact, seed=SEED, regions=np.zeros(n, dtype=np.int64), within_region_frac=1.5)


# ------------------------------------------------------------------------------------ calibration
def test_calibration_picks_the_identity_on_an_er_like_spectrum():
    art = fitted_v2_artifact(toy_graph("er"))
    assert [m.kind for m in art.measurements.values() if m.kind == "spectrum"]
    R, q, z, info = calibrate_regions(art, art.schema.n, seed=0)
    assert R == 1 or q == 0.0, (R, q)
    assert info["skipped"] is None and len(info["grid"]) >= 17
    assert info["chosen"]["score"] <= info["identity_score"]


def test_calibration_picks_regions_on_a_planted_partition_spectrum():
    art = fitted_v2_artifact(toy_graph("pp"))
    R, q, z, info = calibrate_regions(art, art.schema.n, seed=0)
    assert R > 1 and q > 0.0, (R, q)
    assert z is not None and z.shape == (art.schema.n,) and z.max() == R - 1
    assert info["chosen"]["score"] < 0.5 * info["identity_score"]
    assert info["margin_vs_identity"] == float("inf")  # sigma = 0: no noise
    # the refinement step tried neighbours of the coarse best
    rs = {c["R"] for c in info["grid"]}
    assert rs - set(V2.COARSE_R)
    # the chosen candidate is the grid minimum
    best = min(info["grid"], key=lambda c: (c["score"], c["R"], c["q"]))
    assert (best["R"], best["q"]) == (R, q)
    # the structure carries the regions
    g, ginfo = generate_v2_with_info(art, seed=0)
    assert (ginfo["regions_R"], ginfo["within_region_frac"]) == (R, q)
    assert modularity_of(ginfo["regions"], g.edges) > 0.3


def test_pinning_one_parameter_calibrates_the_other():
    art = fitted_v2_artifact(toy_graph("pp"))
    R, q, z, info = calibrate_regions(art, art.schema.n, seed=0, regions_R=4)
    assert R == 4 and all(c["R"] in (1, 4) for c in info["grid"])
    R2, q2, _, info2 = calibrate_regions(art, art.schema.n, seed=0, within_region_frac=0.5)
    assert q2 in (0.0, 0.5) and all(c["q"] in (0.0, 0.5) for c in info2["grid"])
    R3, q3, z3, info3 = calibrate_regions(art, art.schema.n, seed=0, regions_R=4, within_region_frac=0.5)
    assert (R3, q3) == (4, 0.5) and info3["skipped"] == "pinned" and z3 is not None


def test_noisy_spectrum_still_calibrates_and_reports_the_margin():
    art = fitted_v2_artifact(toy_graph("pp"), noise=True)
    spec = [m for m in art.measurements.values() if m.kind == "spectrum"][0]
    assert spec.sigma > 0
    R, q, z, info = calibrate_regions(art, art.schema.n, seed=0)
    assert info["sigma"] == spec.sigma
    assert info["margin_vs_identity"] is None or np.isfinite(info["margin_vs_identity"])
    assert all(np.isfinite(c["score"]) for c in info["grid"])


def test_top_eigenvalues_dense_and_sparse_agree():
    g = toy_graph("pp", n=600)
    dense = V2.top_eigenvalues(g.edges, g.n, 8)
    V2_dense_n = V2._DENSE_N
    try:
        V2._DENSE_N = 10  # force the sparse path
        sparse = V2.top_eigenvalues(g.edges, g.n, 8)
    finally:
        V2._DENSE_N = V2_dense_n
    assert np.allclose(dense, sparse, atol=1e-6)
    assert (np.diff(dense) <= 1e-9).all()
    assert V2.top_eigenvalues(np.zeros((0, 2), dtype=np.int64), 5, 3).shape == (3,)
    assert V2.top_eigenvalues(g.edges, g.n, 1000).shape == (g.n - 1,)


def test_spectrum_measurement_with_bad_values_is_cleaned(artifact):
    art = exact_artifact(planted_graph(n=300, m=5, seed=3))
    vals = np.array([np.nan, 20.0, np.inf, 5.0])
    art.measurements["spectrum"] = Measurement("spectrum", "spectrum", vals, "gaussian", 0.0, 0.0, 0.0,
                                               {"k": 4, "cap": None, "matrix": "adjacency"})
    g, info = generate_v2_with_info(art, seed=0)
    assert np.isfinite(info["calibration"]["chosen"]["score"])
    assert g.n == 300


# ------------------------------------------------------------------------------------ determinism
def test_deterministic_per_seed():
    art = fitted_v2_artifact(toy_graph("pp"))
    a = generate_v2(art, seed=2, max_rounds=5)
    b = generate_v2(art, seed=2, max_rounds=5)
    assert np.array_equal(a.edges, b.edges) and a.attrs.equals(b.attrs)
    c = generate_v2(art, seed=4, max_rounds=5)
    assert not np.array_equal(a.edges, c.edges)


def test_chain_info_is_still_reported(artifact):
    """The label chain runs after the regions; its report keys are kept next to the region keys."""
    g, info = generate_v2_with_info(artifact, seed=SEED, regions_R=4, within_region_frac=0.5, max_rounds=3)
    assert info["skipped"] and info["rounds"] == 0  # no Simpson kind: the chain is a no-op
    assert info["regions_R"] == 4 and info["regions"].shape == (g.n,)

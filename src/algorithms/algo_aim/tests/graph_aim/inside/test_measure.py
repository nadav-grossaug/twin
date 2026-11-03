"""Brute-force checks of the declared L2 sensitivities in ``graph_aim.inside.measure`` (round 2 design)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from graph_aim.inside import measure as M
from graph_aim.transfer.types import AttributedGraph

DOMAINS = {
    "a": ["x", "y", "other", "none"],
    "b": ["p", "q", "none"],
    "c": ["u", "v", "other", "none"],
}
HOMOPHILY = ["a", "b"]
TOL = 1e-9
SQ2 = math.sqrt(2.0)

# Tiny public parameters so that every mechanism is exercised: Dmax = 4 (below the degrees reached by dense
# graphs on 8..10 nodes, so the clamping into the last histogram cell is active), a coarse histogram grid,
# a clustering minimum degree of 3 and an open-ended top joint-degree bin.
DMAX = 4
CLUSTER_EDGES = [0, 2, 3, 5, DMAX + 7]
JOINT_EDGES = [0, 1, 2, 4, DMAX + 7]
VARIANTS = {
    # name: (caps, hist_exact_upto, counts_from_hist)
    "uncapped_joint": ({"clustering_by_degree": 3, "joint_degree": None, "attr_edge_pair": 3, "attr_neigh_simpson": 3}, 2, False),
    "capped_joint": ({"clustering_by_degree": 5, "joint_degree": 3, "attr_edge_pair": 4, "attr_neigh_simpson": 4}, None, False),
    "mixed": ({"clustering_by_degree": 3, "joint_degree": 5, "attr_edge_pair": 2, "attr_neigh_simpson": 2}, 1, True),
    # the shipped design: clustering on the uncapped graph (k <= n - 2), counts taken from degree_hist
    "uncapped_cluster": ({"clustering_by_degree": None, "joint_degree": None, "attr_edge_pair": 3, "attr_neigh_simpson": 5}, 2, True),
}
SIMPSON_GROUPS = [["a"], ["a", "b"]]  # v2 kinds ride along in every variant: full declared bounds under active caps


def small_specs(variant="uncapped_joint", min_degree=3, n_nodes=10):
    caps, upto, cfh = VARIANTS[variant]
    return M.workload(
        DOMAINS,
        3,
        HOMOPHILY,
        caps=caps,
        max_degree=DMAX,
        hist_exact_upto=upto,
        hist_growth=1.5,
        cluster_bin_edges=CLUSTER_EDGES,
        min_degree=min_degree,
        joint_bin_edges=JOINT_EDGES,
        n_nodes=n_nodes,
        counts_from_hist=cfh,
        neigh_simpson_groups=SIMPSON_GROUPS,
        neigh_simpson_bin_edges=CLUSTER_EDGES,
        neigh_simpson_min_degree=min_degree or 2,  # the Simpson kind always needs a minimum degree (a bin edge)
        spectrum_k=3,
    )


def attrs_zero(n):
    return pd.DataFrame({c: np.zeros(n, dtype=np.int64) for c in DOMAINS})


def random_graph(rng, n, p):
    pairs = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    edges = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n=n, edges=edges, attrs=attrs, domains=DOMAINS)


def with_edge_toggled(g, u, v):
    s = {tuple(e) for e in g.edges.tolist()} ^ {(u, v)}
    arr = np.array(sorted(s), dtype=np.int64).reshape(-1, 2)
    return AttributedGraph(g.n, arr, g.attrs, g.domains)


def with_row(g, node, row):
    attrs = g.attrs.copy()
    attrs.loc[node, list(row)] = list(row.values())
    return AttributedGraph(g.n, g.edges, attrs, g.domains)


def vec(spec, g, salt):
    return M.released_vector(spec.kind, M.exact_from_graph(spec, g, salt), spec.params)


def check_graph(g, salt, specs, edge_pairs, attr_changes, worst):
    base = {s.name: vec(s, g, salt) for s in specs}
    for s in specs:
        s_edge, s_attr = M.sensitivity(s)
        for u, v in edge_pairs:
            d = np.linalg.norm(vec(s, with_edge_toggled(g, u, v), salt) - base[s.name])
            assert d <= s_edge + TOL, (s.name, "edge", u, v, d, s_edge)
            if s_edge:
                worst[(s.kind, "edge")] = max(worst.get((s.kind, "edge"), 0.0), d / s_edge)
        for node, row in attr_changes:
            d = np.linalg.norm(vec(s, with_row(g, node, row), salt) - base[s.name])
            assert d <= s_attr + TOL, (s.name, "attr", node, row, d, s_attr)
            if s_attr:
                worst[(s.kind, "attr")] = max(worst.get((s.kind, "attr"), 0.0), d / s_attr)


def all_attr_changes(g, rng, n_random_rows=3):
    out = []
    for node in range(g.n):
        for c, dom in DOMAINS.items():  # every single-column change
            for val in range(len(dom)):
                if val != g.attrs.loc[node, c]:
                    out.append((node, {c: val}))
        for _ in range(n_random_rows):  # whole-row replacements
            out.append((node, {c: int(rng.integers(0, len(d))) for c, d in DOMAINS.items()}))
    return out


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("seed", range(6))
def test_exhaustive_tiny_graphs(seed, variant):
    """Every possible single-edge toggle and every single-column attribute change, n = 8..10."""
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(8, 11))
    salt = int(rng.integers(0, 1000))
    g = random_graph(rng, n, rng.uniform(0.35, 0.9))
    deg = np.bincount(g.edges.ravel(), minlength=n)
    assert deg.max() > DMAX  # degrees above Dmax exist: the clamping into the last histogram cell is exercised
    specs = small_specs(variant, n_nodes=n)
    check_graph(g, salt, specs, list(itertools.combinations(range(n), 2)), all_attr_changes(g, rng), {})


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("seed", range(3))
def test_sampled_larger_graphs_cap_active(seed, variant):
    rng = np.random.default_rng(500 + seed)
    n = 24
    salt = int(rng.integers(0, 1000))
    g = random_graph(rng, n, 0.3)
    specs = small_specs(variant, n_nodes=n)
    for s in specs:
        cap = s.params.get("cap")
        if cap is not None:
            capped = M.cap_edges(g.edges, n, cap, salt)
            assert len(capped) < len(g.edges)  # this measurement's cap really is active
    pairs_all = list(itertools.combinations(range(n), 2))
    idx = rng.choice(len(pairs_all), size=100, replace=False)
    # also toggle edges that survive the caps, the case that moves the most mass
    capped2 = M.cap_edges(g.edges, n, 2, salt)
    pairs = [pairs_all[i] for i in idx] + [tuple(e) for e in capped2[:25].tolist()]
    changes = [(int(rng.integers(0, n)), {c: int(rng.integers(0, len(d))) for c, d in DOMAINS.items()}) for _ in range(60)]
    check_graph(g, salt, specs, pairs, changes, {})


def test_degree_hist_sensitivity_is_attained_and_clamping_cases():
    """Two nodes of equal degree just below a cell boundary both cross it: norm exactly 2 sqrt(2).
    Endpoints already in the clamped last cell contribute nothing; one in the last cell with a crossing
    partner gives sqrt(2)."""
    spec = M.Spec("degree_hist", "degree_hist", {"max_degree": 6, "cap": None, "hist_edges": None})
    assert M.sensitivity(spec) == (pytest.approx(2 * SQ2), 0.0)
    # nodes 0 and 1 both have degree 2; adding {0, 1}: -2 e_2 + 2 e_3
    g = AttributedGraph(6, np.array([(0, 2), (0, 3), (1, 4), (1, 5)]), attrs_zero(6), DOMAINS)
    d = np.linalg.norm(vec(spec, with_edge_toggled(g, 0, 1), 0) - vec(spec, g, 0))
    assert d == pytest.approx(2 * SQ2)
    # Dmax = 2: degrees 2 and 3 are both clamped into the last cell
    spec2 = M.Spec("degree_hist", "degree_hist", {"max_degree": 2, "cap": None, "hist_edges": None})
    g = AttributedGraph(7, np.array([(0, 2), (0, 3), (1, 4), (1, 5), (1, 6)]), attrs_zero(7), DOMAINS)  # deg 2 and 3
    assert np.linalg.norm(vec(spec2, with_edge_toggled(g, 0, 1), 0) - vec(spec2, g, 0)) == pytest.approx(0.0)
    # u in the last cell (deg 2), v crossing into it (1 -> 2): only v moves
    g = AttributedGraph(5, np.array([(0, 2), (0, 3), (1, 4)]), attrs_zero(5), DOMAINS)
    assert np.linalg.norm(vec(spec2, with_edge_toggled(g, 0, 1), 0) - vec(spec2, g, 0)) == pytest.approx(SQ2)
    # the same on a coarse grid: crossing a coarse boundary is what counts
    spec3 = M.Spec("degree_hist", "degree_hist", {"max_degree": 6, "cap": None, "hist_edges": [0, 1, 3, 7]})
    g = AttributedGraph(6, np.array([(0, 2), (0, 3), (1, 4), (1, 5)]), attrs_zero(6), DOMAINS)  # deg 2 -> 3 crosses 3
    assert np.linalg.norm(vec(spec3, with_edge_toggled(g, 0, 1), 0) - vec(spec3, g, 0)) == pytest.approx(2 * SQ2)
    g = AttributedGraph(6, np.array([(0, 2), (1, 4)]), attrs_zero(6), DOMAINS)  # deg 1 -> 2 stays in [1, 3)
    assert np.linalg.norm(vec(spec3, with_edge_toggled(g, 0, 1), 0) - vec(spec3, g, 0)) == pytest.approx(0.0)


def test_degree_hist_total_edge_count_is_post_processing():
    rng = np.random.default_rng(3)
    g = random_graph(rng, 40, 0.2)
    dmax = int(np.bincount(g.edges.ravel(), minlength=40).max())  # no clamping
    spec = M.Spec("degree_hist", "degree_hist", {"max_degree": dmax, "cap": None, "hist_edges": None})
    h = M.exact_from_graph(spec, g, 0)
    assert len(h) == dmax + 1 and h.sum() == 40
    assert (np.arange(dmax + 1) * h).sum() / 2 == len(g.edges)


def test_hist_grid_properties():
    assert M.hist_grid(10, None) is None and M.hist_grid(10, 20) is None
    g = M.hist_grid(1200, 16, 1.5)
    assert g[0] == 0 and g[16] == 16 and g[-1] == 1201 and g == sorted(set(g))
    assert 20 <= len(g) - 1 <= 40
    # cells beyond the exact part grow by at most the growth ratio (rounded up)
    assert all(b <= max(a + 1, int(np.ceil(a * 1.5))) for a, b in zip(g[16:-2], g[17:-1]))


def test_degree_hist_postprocess_coarse_expansion():
    grid = [0, 1, 2, 4, 8, 13]  # Dmax = 12
    spec = M.Spec("degree_hist", "degree_hist", {"max_degree": 12, "cap": None, "hist_edges": grid, "threshold": 1.0, "spread_power": 1.0})
    raw = np.array([10.0, 20.0, 30.0, 15.0, 0.5])
    out = M.postprocess({"degree_hist": raw}, [spec], {"degree_hist": 2.0}, 100)["degree_hist"]
    assert out.shape == (13,) and out.sum() == pytest.approx(100.0) and np.all(out >= 0)
    # the last cell (0.5 < 1 sigma) is the hub tail: pooled posterior mean instead of a hard zero
    tail = M._nonneg_posterior_mean(0.5, 2.0)
    assert tail > 0.5 and out[8:].sum() > 0.0
    scale = 100.0 / (75.0 + tail)
    cells = [out[a:b].sum() for a, b in zip(grid, grid[1:])]
    assert cells == pytest.approx(list(np.array([10.0, 20.0, 30.0, 15.0, tail]) * scale))
    # inside a cell, mass decreases with the degree (spread power 1)
    assert out[4] > out[5] > out[6] > out[7]
    # sigma = 0: nothing is cut or reshaped
    spec0 = M.Spec("degree_hist", "degree_hist", {"max_degree": 12, "cap": None, "hist_edges": None})
    exact = np.arange(13, dtype=float)
    assert np.allclose(M.postprocess({"degree_hist": exact}, [spec0], {"degree_hist": 0.0}, exact.sum())["degree_hist"], exact)


def test_default_bin_count_and_structure_with_larger_D():
    """Real-sized parameters: bins are public, ascending, cover 0..D (optionally an open top bin)."""
    be = M.degree_bins(100, 12)
    assert be[0] == 0 and be[-1] == 101 and be == sorted(set(be))
    assert 10 <= len(be) - 1 <= 13
    top = M.degree_bins(100, 12, last=1201)
    assert top[-1] == 1201 and top[:-1] == be[:-1]


def test_common_neighbour_construction_is_nearly_tight_for_clustering():
    """u, v share D-1 degree-2 neighbours: without a minimum degree the sum of clustering moves by about D-1
    (the bound is not vacuous); with min_degree = 3 the degree-2 common neighbours drop out of column 0."""
    D = 5
    u, v = 0, 1
    ws = list(range(2, 2 + D - 1))
    edges = [(u, w) for w in ws] + [(v, w) for w in ws]
    n = 2 + len(ws)
    g = AttributedGraph(n, np.array(sorted(edges)), attrs_zero(n), DOMAINS)
    spec = M.Spec("c", "clustering_by_degree", {"bin_edges": [0, 2, 3, 20], "clip": 1.0, "cap": D, "min_degree": None})
    d = np.linalg.norm(vec(spec, with_edge_toggled(g, u, v), 7) - vec(spec, g, 7))
    assert d >= D - 1 - 1e-9
    assert d <= M.sensitivity(spec)[0] / 3 + TOL  # single-step bound s1
    spec3 = M.Spec("c", "clustering_by_degree", {"bin_edges": [0, 2, 3, 20], "clip": 1.0, "cap": D, "min_degree": 3})
    d3 = np.linalg.norm(vec(spec3, with_edge_toggled(g, u, v), 7) - vec(spec3, g, 7))
    assert d3 <= M.sensitivity(spec3)[0] / 3 + TOL
    assert d3 < d
    # degree-3 common neighbours (the smallest that count) really do move by 1 / C(3, 2) each
    ws = list(range(2, 2 + D - 1))
    edges = [(u, w) for w in ws] + [(v, w) for w in ws] + [(w, 2 + D - 1 + i) for i, w in enumerate(ws)]
    n = 2 + len(ws) + len(ws)
    g = AttributedGraph(n, np.array(sorted({tuple(sorted(e)) for e in edges})), attrs_zero(n), DOMAINS)
    d33 = np.linalg.norm(vec(spec3, with_edge_toggled(g, u, v), 7) - vec(spec3, g, 7))
    assert d33 >= (D - 1) / 3 - 1e-9 and d33 <= M.sensitivity(spec3)[0] / 3 + TOL


def test_joint_degree_bound_depends_on_top_internal_edge_not_cap():
    """Uncapped: a node of degree e_k - 1 whose neighbours sit in one bin crosses bin edge e_k and moves e_k - 1
    edges (norm about sqrt(2) (e_k - 1)); the bound is s1 = 2 sqrt(2) r + 1 with r = largest e_k - 1."""
    spec = M.Spec("j", "joint_degree", {"bin_edges": [0, 1, 2, 6, 100], "cap": None})  # internal edges 1, 2, 6 -> r = 5
    assert M.sensitivity(spec)[0] == pytest.approx(2 * SQ2 * 5 + 1)
    n = 8
    g = AttributedGraph(n, np.array([(0, w) for w in range(1, 6)]), attrs_zero(n), DOMAINS)  # hub of degree 5
    d = np.linalg.norm(vec(spec, with_edge_toggled(g, 0, 6), 0) - vec(spec, g, 0))  # 5 -> 6 crosses e = 6
    assert SQ2 * 5 - 1e-9 <= d <= M.sensitivity(spec)[0] + TOL
    # a node deep inside the open top bin (degree 50) moves no edge when it gains another one; only +1 in a cell
    n = 60
    g = AttributedGraph(n, np.array([(0, w) for w in range(1, 51)]), attrs_zero(n), DOMAINS)
    assert np.linalg.norm(vec(spec, with_edge_toggled(g, 0, 55), 0) - vec(spec, g, 0)) == pytest.approx(1.0)
    # a cap truncates the set of crossings (d_u + 1 <= cap) and adds the factor 3
    capped = M.Spec("j", "joint_degree", {"bin_edges": [0, 1, 2, 6, 100], "cap": 3})
    assert M.sensitivity(capped)[0] == pytest.approx(3 * (2 * SQ2 * 1 + 1))


def test_sensitivity_table_values():
    D = 100
    spec = lambda kind, **p: M.Spec(kind, kind, p)  # noqa: E731
    assert M.sensitivity(spec("degree_hist", cap=None, max_degree=1200)) == (pytest.approx(2 * SQ2), 0.0)
    assert M.sensitivity(spec("attr_1way", attrs=["a"], cap=None)) == (0.0, pytest.approx(SQ2))
    assert M.sensitivity(spec("attr_2way", attrs=["a", "b"], cap=None)) == (0.0, pytest.approx(SQ2))
    assert M.sensitivity(spec("attr_edge_pair", attrs=["a"], cap=D)) == (3.0, pytest.approx(SQ2 * D))
    # clustering: 3 * (2 sqrt(2 cmax^2 + 2) + (cap - 1) * min(cmax, 1 / C(max(dmin, 2), 2)))
    s = spec("clustering_by_degree", bin_edges=[0, 2, 12, 200], clip=1.0, cap=100, min_degree=12)
    assert M.sensitivity(s)[0] == pytest.approx(3 * (4.0 + 99 / 66))
    s = spec("clustering_by_degree", bin_edges=[0, 2, 200], clip=1.0, cap=100, min_degree=None)  # old worst case: degree-2 w
    assert M.sensitivity(s)[0] == pytest.approx(3 * (4.0 + 99))
    # counts_from_hist: only column 0 is released, endpoint norm sqrt(2) cmax instead of sqrt(2 cmax^2 + 2)
    s = spec("clustering_by_degree", bin_edges=[0, 2, 12, 200], clip=1.0, cap=100, min_degree=12, counts_from_hist=True)
    assert M.sensitivity(s)[0] == pytest.approx(3 * (2 * SQ2 + 99 / 66))
    # uncapped graph: one edge changes (no factor 3) and the common-neighbour count is bounded by n - 2
    s = spec("clustering_by_degree", bin_edges=[0, 2, 12, 200], clip=1.0, cap=None, min_degree=12, counts_from_hist=True, n_nodes=100)
    assert M.sensitivity(s)[0] == pytest.approx(2 * SQ2 + 98 / 66)
    s = spec("joint_degree", bin_edges=[0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 1201], cap=None)
    assert M.sensitivity(s) == (pytest.approx(2 * SQ2 * 39 + 1), 0.0)
    for bad in (
        spec("degree_hist", cap=100, max_degree=1200),
        spec("clustering_by_degree", bin_edges=[0, 2, 10], cap=None),
        spec("attr_edge_pair", attrs=["a"], cap=None),
    ):
        with pytest.raises(ValueError):
            M.sensitivity(bad)


def test_workload_records_per_measurement_caps_and_public_params():
    specs = M.workload(
        DOMAINS, 100, HOMOPHILY, caps={"clustering_by_degree": 40, "joint_degree": None}, max_degree=500,
        hist_exact_upto=16, cluster_bin_edges=[0, 2, 6, 501], min_degree=6, joint_bin_edges=[0, 1, 5, 501],
    )
    by = {s.name: s for s in specs}
    assert by["degree_hist"].params["cap"] is None and by["degree_hist"].params["max_degree"] == 500
    assert by["clustering_by_degree"].params["cap"] == 40
    assert by["joint_degree"].params["cap"] is None
    assert by["attr_edge_pair"].params["cap"] == 100  # missing kind falls back to D
    assert all(s.params["cap"] is None for s in specs if s.kind.startswith("attr_") and s.kind != "attr_edge_pair")
    with pytest.raises(ValueError):
        M.workload(DOMAINS, 100, HOMOPHILY, cluster_bin_edges=[0, 2, 6, 501], min_degree=5)  # not a bin edge


def test_clustering_min_degree_semantics():
    """Column 1 counts every node; column 0 only nodes with degree >= min_degree; postprocess imputes the rest."""
    rng = np.random.default_rng(4)
    g = random_graph(rng, 30, 0.3)
    base = {"bin_edges": [0, 2, 4, 6, 40], "clip": 1.0, "cap": 40}
    full = M.exact_from_graph(M.Spec("c", "clustering_by_degree", {**base, "min_degree": None}), g, 1)
    cut = M.exact_from_graph(M.Spec("c", "clustering_by_degree", {**base, "min_degree": 4}), g, 1)
    assert np.array_equal(full[:, 1], cut[:, 1]) and cut[:2, 0].sum() == 0
    assert np.allclose(full[2:, 0], cut[2:, 0])
    spec = M.Spec("c", "clustering_by_degree", {**base, "min_degree": 4})
    out = M.postprocess({"c": cut}, [spec], {"c": 0.0}, g.n)["c"]
    cbar = cut[2:4, 0].sum() / cut[2:4, 1].sum()
    assert np.allclose(out[:2, 0], out[:2, 1] * cbar) and np.allclose(out[2:], cut[2:])


def test_counts_from_hist_never_releases_exact_counts():
    """With counts_from_hist the exact count column is zeroed before it could be released and is rebuilt from
    the processed degree histogram (free post-processing); column 1 then sums to n."""
    rng = np.random.default_rng(6)
    g = random_graph(rng, 30, 0.3)
    specs = small_specs("uncapped_cluster", n_nodes=g.n)
    cl = next(s for s in specs if s.kind == "clustering_by_degree")
    dh = next(s for s in specs if s.kind == "degree_hist")
    assert cl.params["counts_from_hist"] and M.released_vector(cl.kind, np.ones((4, 2)), cl.params).shape == (4,)
    exact = M.exact_from_graph(cl, g, 1)
    noisy = M.add_noise(cl, exact, 5.0, np.random.default_rng(0))
    assert np.all(noisy[:, 1] == 0.0) and not np.allclose(noisy[:, 0], exact[:, 0])
    raw = {cl.name: noisy, dh.name: M.add_noise(dh, M.exact_from_graph(dh, g, 1), 1.0, np.random.default_rng(1))}
    out = M.postprocess(raw, specs[:2], {cl.name: 5.0, dh.name: 1.0}, g.n)
    hist = out[dh.name]
    assert out[cl.name][:, 1].sum() == pytest.approx(g.n)
    be = cl.params["bin_edges"]
    for b in range(len(be) - 1):
        assert out[cl.name][b, 1] == pytest.approx(hist[be[b] : be[b + 1]].sum())
    assert np.all(out[cl.name][:, 0] <= out[cl.name][:, 1] + 1e-9)


def test_postprocess_nonneg_symmetric_consistent():
    rng = np.random.default_rng(0)
    g = random_graph(rng, 30, 0.2)
    salt = 3
    for variant in VARIANTS:
        specs = small_specs(variant)
        sig = {s.name: 5.0 for s in specs}
        raw = {}
        for s in specs:
            raw[s.name] = M.add_noise(s, M.exact_from_graph(s, g, salt), sig[s.name], rng)
        out = M.postprocess(raw, specs, sig, g.n)
        for s in specs:
            v = out[s.name]
            if s.kind == "spectrum":
                assert np.all(np.diff(v) <= 0)  # eigenvalues may be negative; only the order is enforced
            else:
                assert np.all(v >= 0)
            if s.kind in M.SYMMETRIC_KINDS:
                assert np.allclose(v, v.T)
            if s.kind in ("attr_1way", "attr_2way", "degree_hist"):
                assert v.sum() == pytest.approx(g.n)
            if s.kind == "degree_hist":
                assert v.shape == (DMAX + 1,)  # expanded to Dmax + 1 even for coarse grids
            else:
                assert v.shape == M.exact_from_graph(s, g, salt).shape


def test_noise_free_postprocess_is_identity():
    rng = np.random.default_rng(1)
    g = random_graph(rng, 30, 0.2)
    salt = 3
    specs = small_specs("capped_joint", min_degree=None)  # exact releases: full histogram, no minimum degree
    raw = {s.name: M.exact_from_graph(s, g, salt) for s in specs}
    out = M.postprocess(raw, specs, {s.name: 0.0 for s in specs}, g.n)
    hist = out["degree_hist"]
    for s in specs:
        if s.kind == "attr_neigh_simpson":
            # column 0 is exact; column 1 is by design the (uncapped) degree-histogram mass, not the capped count
            assert np.allclose(out[s.name][:, 0], raw[s.name][:, 0])
            be = s.params["bin_edges"]
            assert out[s.name][:, 1] == pytest.approx([hist[a:b].sum() for a, b in zip(be, be[1:])])
            continue
        assert np.allclose(out[s.name], raw[s.name])


def _hist_spec(grid, threshold=1.0):
    return M.Spec("degree_hist", "degree_hist", {"max_degree": grid[-1] - 1, "cap": None, "hist_edges": grid,
                                                  "threshold": threshold, "spread_power": 1.0})


def test_degree_hist_tail_pooling_keeps_hub_mass():
    grid = [0, 10, 20, 40, 80, 160, 320, 640, 1000]
    spec = _hist_spec(grid)
    sigma = 3.0
    raw = np.array([400.0, 300.0, 200.0, 90.0, 10.0, 1.0, -2.0, 2.0])  # sigma 3: the last three cells are below one sigma, 10 stays
    out = M.postprocess({"degree_hist": raw}, [spec], {"degree_hist": sigma}, 1000)["degree_hist"]
    assert out.sum() == pytest.approx(1000.0) and np.all(out >= 0)
    tail_cells = [out[a:b].sum() for a, b in zip(grid[5:], grid[6:])]
    # trailing run of sub-threshold cells (1, -2, 2) is pooled: equal mass in each, positive although one raw is < 0
    assert min(tail_cells) > 0 and tail_cells == pytest.approx([tail_cells[0]] * 3)
    pooled = M._nonneg_posterior_mean(1.0, sigma * np.sqrt(3))
    assert sum(tail_cells) == pytest.approx(pooled * 1000.0 / (400 + 300 + 200 + 90 + 10 + pooled))


def test_degree_hist_tail_pooling_recovers_hub_mass_on_average():
    """Five tail cells with 4 nodes each (< 2 sigma, so per-cell thresholding would delete all 20)."""
    rng = np.random.default_rng(0)
    sigma = 3.0
    grid = [0, 10, 20, 40, 80, 160, 320, 640, 1000, 1500, 2000]
    spec = _hist_spec(grid, threshold=2.0)
    true = np.array([500.0, 300.0, 150.0, 40.0, 10.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    n = float(true.sum())
    tails = []
    for _ in range(300):
        out = M._degree_hist_post(true + rng.normal(0, sigma, 10), spec, sigma, n)
        tails.append(out[160:].sum())
    assert abs(np.mean(tails) - 20.0) < 4.0


def test_degree_hist_no_pooling_without_coarse_grid_or_noise():
    spec = M.Spec("degree_hist", "degree_hist", {"max_degree": 5, "cap": None, "hist_edges": None, "threshold": 2.0})
    raw = np.array([0.0, 50.0, 40.0, 30.0, 1.0, 1.0])
    out = M.postprocess({"degree_hist": raw}, [spec], {"degree_hist": 3.0}, 120)["degree_hist"]
    assert out[4:].sum() == 0.0  # legacy behaviour: sub-threshold per-degree cells are cut


def test_nonneg_posterior_mean_properties():
    for sd in (0.5, 3.0):
        xs = np.linspace(-5 * sd, 5 * sd, 21)
        vals = np.array([M._nonneg_posterior_mean(x, sd) for x in xs])
        assert np.all(vals > 0) and np.all(np.diff(vals) > 0)
        assert M._nonneg_posterior_mean(10 * sd, sd) == pytest.approx(10 * sd, rel=1e-6)


def test_degree_hist_postprocess_uses_only_released_values():
    import inspect

    assert list(inspect.signature(M._degree_hist_post).parameters) == ["v", "spec", "sigma", "n"]
    assert list(inspect.signature(M.postprocess).parameters) == ["raw", "specs", "sigmas", "n"]


# --------------------------------------------------------------------------- attr_degree


def ad_spec(attr="a", edges=(0, 1, 2, 4)):
    return M.Spec(f"attr_degree:{attr}", "attr_degree", {"attrs": [attr], "cap": None, "bin_edges": list(edges)})


def random_graph_k(rng, n, p, k):
    """Random graph with one attribute ``a`` of domain size ``k``."""
    dom = {"a": [f"v{i}" for i in range(k)]}
    pairs = [(u, v) for u, v in itertools.combinations(range(n), 2) if rng.random() < p]
    edges = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return AttributedGraph(n, edges, pd.DataFrame({"a": rng.integers(0, k, size=n)}), dom)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("edges", [(0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 4), (0, 3), (0, 1, 5)])
def test_attr_degree_sensitivity_brute_force(seed, k, edges):
    rng = np.random.default_rng(100 * seed + k)
    n = 9
    g = random_graph_k(rng, n, [0.2, 0.4, 0.6, 0.8][seed], k)
    spec = ad_spec("a", edges)
    s_edge, s_attr = M.sensitivity(spec)
    assert s_edge == pytest.approx(2 * SQ2) and s_attr == pytest.approx(SQ2)
    base = M.exact_from_graph(spec, g, 0)
    assert base.shape == (k, len(edges) - 1) and base.sum() == n
    for u, v in itertools.combinations(range(n), 2):  # every edge, added or removed
        d = np.linalg.norm(M.exact_from_graph(spec, with_edge_toggled(g, u, v), 0) - base)
        assert d <= s_edge + TOL
    for node in range(n):  # every node, every other value
        for val in range(k):
            if val == g.attrs.loc[node, "a"]:
                continue
            d = np.linalg.norm(M.exact_from_graph(spec, with_row(g, node, {"a": val}), 0) - base)
            assert d == pytest.approx(SQ2)  # exactly one unit moves between two cells
            assert d <= s_attr + TOL


def test_attr_degree_edge_bound_attained_and_cancellation_case():
    dom = {"a": ["x", "y"]}
    attrs = pd.DataFrame({"a": [0, 0, 1, 1, 1, 1]})
    # nodes 0 and 1 (equal value, degree 1, just below edge 2): the new edge {0, 1} lifts both from bin 1 to 2
    g = AttributedGraph(6, np.array([[0, 4], [1, 5]], dtype=np.int64), attrs, dom)
    spec = ad_spec("a", (0, 1, 2, 5))
    d = M.exact_from_graph(spec, with_edge_toggled(g, 0, 1), 0) - M.exact_from_graph(spec, g, 0)
    assert np.linalg.norm(d) == pytest.approx(2 * SQ2)
    assert d[0, 1] == -2 and d[0, 2] == 2
    # cancellation: node 0 goes 1 -> 2 (bin 1 -> 2) and node 1 goes 2 -> 3 (bin 2 -> 3): middle terms cancel
    g = AttributedGraph(6, np.array([[0, 4], [1, 4], [1, 5]], dtype=np.int64), attrs, dom)
    spec = ad_spec("a", (0, 1, 2, 3, 6))
    d = M.exact_from_graph(spec, with_edge_toggled(g, 0, 1), 0) - M.exact_from_graph(spec, g, 0)
    assert np.linalg.norm(d) == pytest.approx(SQ2)
    assert d[0, 1] == -1 and d[0, 3] == 1 and d[0, 2] == 0


def test_attr_degree_exact_values_hand_computed():
    # path 0-1-2-3 plus isolated node 4; degrees 1,2,2,1,0; attribute codes 0,1,0,1,0 (domain of 3 values)
    dom = {"a": ["x", "y", "z"]}
    g = AttributedGraph(5, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64), pd.DataFrame({"a": [0, 1, 0, 1, 0]}), dom)
    spec = ad_spec("a", (0, 1, 2, 3))  # bins: degree 0 | 1 | >= 2 (last bin open-ended)
    expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=float)
    assert np.array_equal(M.exact_from_graph(spec, g, 0), expected)


def test_attr_degree_cap_must_be_none():
    spec = M.Spec("attr_degree:a", "attr_degree", {"attrs": ["a"], "cap": 5, "bin_edges": [0, 1, 2]})
    with pytest.raises(ValueError, match="uncapped"):
        M.sensitivity(spec)


def test_attr_degree_workload_options():
    base = dict(domains=DOMAINS, D=3, homophily_attrs=HOMOPHILY, max_degree=50)
    assert not any(s.kind == "attr_degree" for s in M.workload(**base))
    specs = M.workload(**base, attr_degree=True)
    ad = [s for s in specs if s.kind == "attr_degree"]
    assert [s.name for s in ad] == [f"attr_degree:{c}" for c in DOMAINS] and specs[-len(ad):] == ad
    default = M.degree_bins(50, 6, 51)
    assert all(s.params == {"attrs": [s.name.split(":")[1]], "cap": None, "bin_edges": default} for s in ad)
    assert len(default) - 1 <= 8
    sub = M.workload(**base, attr_degree=True, attr_degree_attrs=["b"], attr_degree_bin_edges=[0, 3, 10])
    assert [s.name for s in sub if s.kind == "attr_degree"] == ["attr_degree:b"]
    assert sub[-1].params["bin_edges"] == [0, 3, 10]
    with pytest.raises(ValueError, match="not in domains"):
        M.workload(**base, attr_degree=True, attr_degree_attrs=["zzz"])
    with pytest.raises(ValueError):
        M.workload(**base, attr_degree=True, attr_degree_bin_edges=list(range(10)))  # 9 bins
    with pytest.raises(ValueError):
        M.workload(**base, attr_degree=True, attr_degree_bin_edges=[1, 5])  # must start at 0


# --------------------------------------------------------------------------- attr_neigh_simpson (v2)


def ns_spec(attrs=("a",), edges=(0, 2, 3, 5, 20), dmin=3, cap=20):
    return M.Spec(
        f"attr_neigh_simpson:{'+'.join(attrs)}",
        "attr_neigh_simpson",
        {"attrs": list(attrs), "cap": cap, "bin_edges": list(edges), "min_degree": dmin, "counts_from_hist": True},
    )


def simpson_by_hand(g, attrs, edges, dmin):
    """Reference implementation: adjacency sets and a dict per node, no vectorisation."""
    nb = {u: set() for u in range(g.n)}
    for u, v in g.edges.tolist():
        nb[u].add(v)
        nb[v].add(u)
    codes, _ = M.group_codes(g.attrs, g.domains, list(attrs))
    out = np.zeros((len(edges) - 1, 2))
    for u in range(g.n):
        d = len(nb[u])
        b = min(int(np.searchsorted(edges, d, side="right")) - 1, len(edges) - 2)
        out[b, 1] += 1
        if d >= dmin:
            cnt = {}
            for w in nb[u]:
                cnt[codes[w]] = cnt.get(codes[w], 0) + 1
            out[b, 0] += sum((c / d) ** 2 for c in cnt.values())
    return out


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("attrs", [("a",), ("b",), ("a", "b"), ("a", "c")])
@pytest.mark.parametrize("edges, dmin", [((0, 2, 3, 5, 20), 3), ((0, 1, 3, 6, 20), 1), ((0, 2, 3, 5, 20), 2), ((0, 5, 20), 5), ((0, 3, 20), 3)])
def test_neigh_simpson_single_step_bounds_brute_force(seed, attrs, edges, dmin):
    """Every single-edge change on graphs whose max degree stays within the cap (so the cap is a no-op and one
    toggle is one step) moves column 0 by <= 2 sqrt(2) = s1 (the released bound is 3 s1 for the capping); every
    single-row change (and whole-row replacement) by <= 2 Dp / dmin, and in fact by <= 2 d_u / dmin (the proof's
    intermediate bound, d_u the changed node's degree)."""
    rng = np.random.default_rng(1000 * seed + len(attrs))
    n = int(rng.integers(8, 11))
    g = random_graph(rng, n, rng.uniform(0.3, 0.8))
    deg = np.bincount(g.edges.ravel(), minlength=n)
    cap = 20
    assert deg.max() + 1 <= cap  # every toggled graph stays within the cap: the capped graph IS the graph
    spec = ns_spec(attrs, edges, dmin, cap)
    s_edge, s_attr = M.sensitivity(spec)
    assert s_edge == pytest.approx(6 * SQ2) and s_attr == pytest.approx(2 * cap / dmin)
    base_full = M.exact_from_graph(spec, g, 0)
    assert np.allclose(base_full, simpson_by_hand(g, attrs, edges, dmin))
    assert base_full[:, 1].sum() == n and np.all(base_full[:, 0] <= base_full[:, 1] + TOL)
    base = M.released_vector(spec.kind, base_full, spec.params)
    assert base.shape == (len(edges) - 1,)
    for u, v in itertools.combinations(range(n), 2):
        d = np.linalg.norm(vec(spec, with_edge_toggled(g, u, v), 0) - base)
        assert d <= 2 * SQ2 + TOL, (u, v, d)
    for node, row in all_attr_changes(g, rng):
        d = np.linalg.norm(vec(spec, with_row(g, node, row), 0) - base)
        assert d <= s_attr + TOL
        assert d <= 2 * deg[node] / dmin + TOL, (node, row, d)


def test_neigh_simpson_edge_bound_is_attained_and_bin_crossings():
    """Two pure neighbourhoods of degree e_k - 1 = 2, both crossing the bin edge 3 when {u, v} is added and
    staying pure: -e_1 + e_2 per endpoint, 2 sqrt(2) in all (= s1). A crossing of dmin (0 -> s) and a node whose
    bin does not change (one coordinate, <= 1) are the other cases of the proof."""
    dom = {"a": ["x", "y", "other", "none"]}
    attrs = pd.DataFrame({"a": [0, 0, 0, 0, 0, 0]})
    g = AttributedGraph(6, np.array([(0, 2), (0, 3), (1, 4), (1, 5)], dtype=np.int64), attrs, dom)
    spec = M.Spec("s", "attr_neigh_simpson", {"attrs": ["a"], "cap": 10, "bin_edges": [0, 2, 3, 10], "min_degree": 2, "counts_from_hist": True})
    d = vec(spec, with_edge_toggled(g, 0, 1), 0) - vec(spec, g, 0)
    assert np.linalg.norm(d) == pytest.approx(2 * SQ2) and d[1] == -2 and d[2] == 2
    # dmin crossing: node 0 has degree 1 (< dmin = 2, s = 0); adding {0, 1} lifts it to 2 with a pure neighbourhood
    g = AttributedGraph(6, np.array([(0, 2), (1, 4), (1, 5)], dtype=np.int64), attrs, dom)
    spec = M.Spec("s", "attr_neigh_simpson", {"attrs": ["a"], "cap": 10, "bin_edges": [0, 2, 3, 10], "min_degree": 2, "counts_from_hist": True})
    base = vec(spec, g, 0)
    assert base[0] == 0  # the bin below dmin is empty in column 0 even though node 0 lives there
    d = vec(spec, with_edge_toggled(g, 0, 1), 0) - base
    # node 0: 0 -> 1 in bin 1 (+e_1); node 1: degree 2 -> 3 crosses into bin 2 with s = 1 (-e_1 + e_2): net e_2
    assert d.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert np.linalg.norm(d) <= 2 * SQ2 + TOL
    # bin unchanged: node 1 with degree 2 (bin [2, 3) ... open) gains a differently coded neighbour: s 1 -> 5/9
    attrs2 = pd.DataFrame({"a": [1, 0, 0, 0, 0, 0]})
    g = AttributedGraph(6, np.array([(1, 4), (1, 5)], dtype=np.int64), attrs2, dom)
    spec = M.Spec("s", "attr_neigh_simpson", {"attrs": ["a"], "cap": 10, "bin_edges": [0, 2, 10], "min_degree": 2, "counts_from_hist": True})
    d = vec(spec, with_edge_toggled(g, 0, 1), 0) - vec(spec, g, 0)
    assert d[1] == pytest.approx(5 / 9 - 1.0) and d[0] == 0


def test_neigh_simpson_attr_bound_closed_form_and_two_attribute_groups():
    """The row change of u moves every neighbour w (d_w >= dmin) by exactly 2 (n_{w,g'} - n_{w,g} + 1) / d_w^2."""
    dom = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "none"]}
    # star: hub 0 with leaves 1..5; leaves have degree 1, hub degree 5
    edges = np.array([(0, w) for w in range(1, 6)], dtype=np.int64)
    attrs = pd.DataFrame({"a": [0, 0, 0, 1, 1, 2], "b": [0, 0, 0, 0, 1, 1]})
    g = AttributedGraph(6, edges, attrs, dom)
    spec = ns_spec(("a",), (0, 1, 5, 20), dmin=1, cap=20)
    base = M.exact_from_graph(spec, g, 0)
    # hub: n = {x: 2, y: 2, other: 1} over d = 5 -> (4 + 4 + 1) / 25; leaves: pure (hub's code x) -> 1 each
    assert base[2, 0] == pytest.approx(9 / 25) and base[1, 0] == pytest.approx(5.0)
    # leaf 1 changes x -> other: hub's n_x 2 -> 1, n_other 1 -> 2: delta = 2 (1 - 2 + 1) / 25 = 0
    after = M.exact_from_graph(spec, with_row(g, 1, {"a": 2}), 0)
    assert after[2, 0] - base[2, 0] == pytest.approx(0.0)
    # leaf 1 changes x -> y: delta = 2 (2 - 2 + 1) / 25 = 2 / 25
    after = M.exact_from_graph(spec, with_row(g, 1, {"a": 1}), 0)
    assert after[2, 0] - base[2, 0] == pytest.approx(2 / 25)
    # the hub's own change moves every leaf's pure neighbourhood not at all (still pure): s_leaf stays 1
    after = M.exact_from_graph(spec, with_row(g, 0, {"a": 3}), 0)
    assert np.allclose(after, base)
    # a group of two attributes: codes are row-major (a, b) -> a * 3 + b
    spec2 = ns_spec(("a", "b"), (0, 1, 5, 20), dmin=1, cap=20)
    codes, size = M.group_codes(attrs, dom, ["a", "b"])
    assert size == 12 and codes.tolist() == [0, 0, 0, 3, 4, 7]
    base2 = M.exact_from_graph(spec2, g, 0)
    assert base2[2, 0] == pytest.approx((2**2 + 1 + 1 + 1) / 25)  # hub: {0: 2, 3: 1, 4: 1, 7: 1}
    assert np.allclose(base2, simpson_by_hand(g, ("a", "b"), [0, 1, 5, 20], 1))


@pytest.mark.parametrize("seed", range(3))
def test_neigh_simpson_capped_full_bound_sampled(seed):
    """With the cap active (3 capped edges may change per toggle) the released bound 6 sqrt(2) holds; attribute
    changes stay within 2 Dp / dmin."""
    rng = np.random.default_rng(700 + seed)
    n = 24
    salt = int(rng.integers(0, 1000))
    g = random_graph(rng, n, 0.3)
    specs = [ns_spec(("a",), (0, 2, 3, 5, 30), 3, 4), ns_spec(("a", "b"), (0, 2, 3, 5, 30), 2, 3)]
    for s in specs:
        assert len(M.cap_edges(g.edges, n, s.params["cap"], salt)) < len(g.edges)
    pairs_all = list(itertools.combinations(range(n), 2))
    idx = rng.choice(len(pairs_all), size=80, replace=False)
    capped2 = M.cap_edges(g.edges, n, 2, salt)
    pairs = [pairs_all[i] for i in idx] + [tuple(e) for e in capped2[:25].tolist()]
    changes = [(int(rng.integers(0, n)), {c: int(rng.integers(0, len(d))) for c, d in DOMAINS.items()}) for _ in range(40)]
    check_graph(g, salt, specs, pairs, changes, {})


def test_neigh_simpson_sensitivity_requirements():
    assert M.sensitivity(ns_spec(("a",), (0, 2, 3, 5, 20), 3, 150)) == (pytest.approx(6 * SQ2), pytest.approx(2 * 150 / 3))
    assert M.sensitivity(ns_spec(("a",), (0, 16, 35, 60, 110, 1201), 16, 150)) == (pytest.approx(6 * SQ2), pytest.approx(18.75))
    with pytest.raises(ValueError, match="cap"):
        M.sensitivity(ns_spec(cap=None))
    with pytest.raises(ValueError, match="min_degree"):
        M.sensitivity(ns_spec(dmin=None))
    with pytest.raises(ValueError, match="min_degree"):
        M.sensitivity(ns_spec(dmin=0))
    with pytest.raises(ValueError, match="bin edges"):
        M.sensitivity(ns_spec(dmin=4))  # not an edge of (0, 2, 3, 5, 20)
    spec = ns_spec()
    with pytest.raises(ValueError, match="counts_from_hist"):
        M.sensitivity(M.Spec(spec.name, spec.kind, {**spec.params, "counts_from_hist": False}))
    with pytest.raises(ValueError, match="not capped"):
        M.exact_values(spec, np.array([(0, w) for w in range(1, 25)], dtype=np.int64), 25, attrs_zero(25), DOMAINS)


def test_neigh_simpson_workload_options_and_order():
    base = dict(domains=DOMAINS, D=3, homophily_attrs=HOMOPHILY, max_degree=50, caps={"attr_neigh_simpson": 7}, attr_degree=True)
    assert not any(s.kind in ("attr_neigh_simpson", "spectrum") for s in M.workload(**base))
    specs = M.workload(**base, neigh_simpson_groups=[["a"], ["b", "c"]], neigh_simpson_bin_edges=[0, 2, 5, 51], neigh_simpson_min_degree=2, spectrum_k=4)
    kinds = [s.kind for s in specs]
    assert kinds[-3:] == ["attr_neigh_simpson", "attr_neigh_simpson", "spectrum"]  # appended after every v1 kind
    assert kinds.index("attr_degree") < kinds.index("attr_neigh_simpson")
    assert [s.name for s in specs[-3:]] == ["attr_neigh_simpson:a", "attr_neigh_simpson:b+c", "spectrum"]
    assert specs[-3].params == {"attrs": ["a"], "cap": 7, "bin_edges": [0, 2, 5, 51], "min_degree": 2, "counts_from_hist": True}
    assert specs[-2].params["attrs"] == ["b", "c"]
    assert specs[-1].params == {"k": 4, "cap": None, "matrix": "adjacency"}
    # an explicit cap wins over caps[...]; a missing caps entry falls back to D
    assert M.workload(**base, neigh_simpson_groups=[["a"]], neigh_simpson_bin_edges=[0, 2, 51], neigh_simpson_min_degree=2, neigh_simpson_cap=9)[-1].params["cap"] == 9
    nocap = dict(base, caps={})
    assert M.workload(**nocap, neigh_simpson_groups=[["a"]], neigh_simpson_bin_edges=[0, 2, 51], neigh_simpson_min_degree=2)[-1].params["cap"] == 3
    # spectrum_k is clamped to n - 1 when the node count is known
    assert M.workload(**base, n_nodes=5, spectrum_k=32)[-1].params["k"] == 4
    ok = dict(neigh_simpson_bin_edges=[0, 2, 51], neigh_simpson_min_degree=2)
    with pytest.raises(ValueError, match="not in domains"):
        M.workload(**base, neigh_simpson_groups=[["zzz"]], **ok)
    with pytest.raises(ValueError, match="duplicate"):
        M.workload(**base, neigh_simpson_groups=[["a"], ["a"]], **ok)
    with pytest.raises(ValueError, match="empty"):
        M.workload(**base, neigh_simpson_groups=[[]], **ok)
    with pytest.raises(ValueError, match="bin edges"):
        M.workload(**base, neigh_simpson_groups=[["a"]], neigh_simpson_bin_edges=[0, 2, 51], neigh_simpson_min_degree=3)
    with pytest.raises(ValueError, match="min_degree"):
        M.workload(**base, neigh_simpson_groups=[["a"]], neigh_simpson_bin_edges=[0, 2, 51], neigh_simpson_min_degree=None)
    with pytest.raises(ValueError, match="bin_edges"):
        M.workload(**base, neigh_simpson_groups=[["a"]], neigh_simpson_min_degree=2)
    with pytest.raises(ValueError):
        M.workload(**base, neigh_simpson_groups=[["a"]], neigh_simpson_bin_edges=[1, 2, 51], neigh_simpson_min_degree=2)  # must start at 0
    with pytest.raises(ValueError, match="cap"):
        M.workload(**dict(base, caps={"attr_neigh_simpson": None}), neigh_simpson_groups=[["a"]], **ok)
    big = {"u": [f"u{i}" for i in range(9)], "v": [f"v{i}" for i in range(8)]}
    with pytest.raises(ValueError, match="too many attribute groups"):
        M.workload(big, 3, [], caps={"attr_neigh_simpson": 7}, neigh_simpson_groups=[["u", "v"]], **ok)
    with pytest.raises(ValueError, match="spectrum_k"):
        M.workload(**base, spectrum_k=0)


def test_neigh_simpson_release_and_postprocess():
    """Only column 0 is noised and released; postprocess fills column 1 from the processed degree histogram, clips
    column 0 to [0, count] (noisy runs only) and zeroes the bins entirely below min_degree."""
    rng = np.random.default_rng(8)
    g = random_graph(rng, 30, 0.3)
    specs = small_specs("uncapped_cluster", n_nodes=g.n)
    ns = [s for s in specs if s.kind == "attr_neigh_simpson"]
    dh = next(s for s in specs if s.kind == "degree_hist")
    assert len(ns) == 2 and all(M.released_vector(s.kind, np.ones((4, 2)), s.params).shape == (4,) for s in ns)
    exact = {s.name: M.exact_from_graph(s, g, 1) for s in ns}
    noisy = {s.name: M.add_noise(s, exact[s.name], 5.0, np.random.default_rng(0)) for s in ns}
    for s in ns:
        assert np.all(noisy[s.name][:, 1] == 0.0) and not np.allclose(noisy[s.name][:, 0], exact[s.name][:, 0])
    raw = {**noisy, dh.name: M.add_noise(dh, M.exact_from_graph(dh, g, 1), 1.0, np.random.default_rng(1))}
    sig = {**{s.name: 5.0 for s in ns}, dh.name: 1.0}
    out = M.postprocess(raw, [dh, *ns], sig, g.n)
    hist = out[dh.name]
    for s in ns:
        w = out[s.name]
        be = s.params["bin_edges"]
        assert w[:, 1].sum() == pytest.approx(g.n)
        for b in range(len(be) - 1):
            assert w[b, 1] == pytest.approx(hist[be[b] : be[b + 1]].sum())
        assert np.all(w[:, 0] >= 0) and np.all(w[:, 0] <= w[:, 1] + 1e-9)
        below = np.asarray(be[1:]) <= s.params["min_degree"]
        assert below.any() and np.all(w[below, 0] == 0.0)
    # noise-free: column 0 is exactly the statistic (no clip against the uncapped count), below-dmin bins are 0 anyway
    raw0 = {**{s.name: exact[s.name] for s in ns}, dh.name: M.exact_from_graph(dh, g, 1)}
    out0 = M.postprocess(raw0, [dh, *ns], {k: 0.0 for k in raw0}, g.n)
    for s in ns:
        assert np.array_equal(out0[s.name][:, 0], exact[s.name][:, 0])


# --------------------------------------------------------------------------- spectrum (v2)


def sp_spec(k):
    return M.Spec("spectrum", "spectrum", {"k": k, "cap": None, "matrix": "adjacency"})


def dense_top(g, k):
    a = np.zeros((g.n, g.n))
    for u, v in g.edges.tolist():
        a[u, v] = a[v, u] = 1.0
    return np.sort(np.linalg.eigvalsh(a))[::-1][:k]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [1, 2, 4, "n-1", "n"])
def test_spectrum_sensitivity_brute_force(seed, k):
    """Every single-edge toggle moves the top-K eigenvalue vector by <= sqrt(2) in L2 (Mirsky); attribute changes
    move it by exactly 0."""
    rng = np.random.default_rng(300 + seed)
    n = int(rng.integers(8, 11))
    g = random_graph(rng, n, rng.uniform(0.2, 0.8))
    K = {"n-1": n - 1, "n": n}.get(k, k)
    spec = sp_spec(K)
    assert M.sensitivity(spec) == (pytest.approx(SQ2), 0.0)
    base = M.exact_from_graph(spec, g, 0)
    assert base.shape == (K,) and np.all(np.diff(base) <= 1e-12)
    assert np.allclose(base, dense_top(g, K))
    for u, v in itertools.combinations(range(n), 2):
        d = np.linalg.norm(M.exact_from_graph(spec, with_edge_toggled(g, u, v), 0) - base)
        assert d <= SQ2 + TOL, (u, v, d)
    for node, row in all_attr_changes(g, rng, n_random_rows=1):
        assert np.array_equal(M.exact_from_graph(spec, with_row(g, node, row), 0), base)


def test_spectrum_edge_bound_is_approached():
    """Adding an edge to the empty graph on two nodes moves (lambda_1, lambda_2) from (0, 0) to (1, -1): exactly sqrt(2)."""
    g = AttributedGraph(2, np.zeros((0, 2), dtype=np.int64), attrs_zero(2), DOMAINS)
    assert np.array_equal(M.exact_from_graph(sp_spec(2), g, 0), np.zeros(2))
    d = np.linalg.norm(M.exact_from_graph(sp_spec(2), with_edge_toggled(g, 0, 1), 0) - np.zeros(2))
    assert d == pytest.approx(SQ2)
    # K = 1 sees only the +1
    assert np.linalg.norm(M.exact_from_graph(sp_spec(1), with_edge_toggled(g, 0, 1), 0)) == pytest.approx(1.0)


def test_spectrum_sparse_path_matches_dense_and_is_deterministic(monkeypatch):
    rng = np.random.default_rng(9)
    g = random_graph(rng, 80, 0.1)
    dense = M.exact_from_graph(sp_spec(6), g, 0)
    monkeypatch.setattr(M, "SPECTRUM_DENSE_MAX_N", 0)  # force ARPACK
    sparse1 = M.exact_from_graph(sp_spec(6), g, 0)
    sparse2 = M.exact_from_graph(sp_spec(6), g, 0)
    assert np.allclose(sparse1, dense, atol=1e-7) and np.array_equal(sparse1, sparse2)
    assert np.all(np.diff(sparse1) <= 0)
    assert np.allclose(M.exact_from_graph(sp_spec(79), g, 0), dense_top(g, 79), atol=1e-7)  # k = n - 1: dense fallback


def test_spectrum_requirements_and_postprocess():
    with pytest.raises(ValueError, match="uncapped"):
        M.sensitivity(M.Spec("spectrum", "spectrum", {"k": 3, "cap": 5}))
    with pytest.raises(ValueError, match="k"):
        M.sensitivity(M.Spec("spectrum", "spectrum", {"k": 0, "cap": None}))
    g = random_graph(np.random.default_rng(2), 6, 0.5)
    with pytest.raises(ValueError, match="k = 7"):
        M.exact_from_graph(sp_spec(7), g, 0)
    assert np.array_equal(M.exact_from_graph(sp_spec(3), AttributedGraph(6, np.zeros((0, 2), dtype=np.int64), attrs_zero(6), DOMAINS), 0), np.zeros(3))
    spec = sp_spec(4)
    exact = M.exact_from_graph(spec, g, 0)
    assert M.add_noise(spec, exact, 3.0, np.random.default_rng(1)).shape == (4,)
    noisy = exact + np.array([0.0, -4.0, 3.0, -6.0])  # a noise draw that breaks the order and makes an entry negative
    assert not np.all(np.diff(noisy) <= 0) and noisy.min() < 0
    out = M.postprocess({"spectrum": noisy}, [spec], {"spectrum": 3.0}, g.n)["spectrum"]
    assert np.array_equal(out, np.sort(noisy)[::-1])  # sorted descending, nothing clamped
    assert np.array_equal(M.postprocess({"spectrum": exact}, [spec], {"spectrum": 0.0}, g.n)["spectrum"], exact)


def test_attr_degree_postprocess_nonneg_sums_to_n_and_noise_free_exact():
    rng = np.random.default_rng(5)
    g = random_graph_k(rng, 40, 0.15, 3)
    spec = ad_spec("a", (0, 1, 3, 6))
    exact = M.exact_from_graph(spec, g, 0)
    noisy = M.add_noise(spec, exact, 6.0, rng)
    assert noisy.shape == exact.shape and np.any(noisy < 0)
    out = M.postprocess({spec.name: noisy}, [spec], {spec.name: 6.0}, g.n)[spec.name]
    assert np.all(out >= 0) and out.sum() == pytest.approx(g.n) and out.shape == exact.shape
    free = M.postprocess({spec.name: M.add_noise(spec, exact, 0.0, rng)}, [spec], {spec.name: 0.0}, g.n)[spec.name]
    assert np.array_equal(free, exact)

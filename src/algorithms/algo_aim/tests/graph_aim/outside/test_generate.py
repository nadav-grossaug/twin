"""Tests for the generator, using noise-free artifacts built here from a planted graph."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
from scipy import stats

from graph_aim.outside.generate import generate
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement, PrivacyLedger, Schema

DOMAINS = {
    "a": ["c0", "c1", "c2", "other", "none"],
    "b": ["b0", "b1", "other", "none"],
    "c": ["low", "high", "other", "none"],
    "d": ["x", "y", "other", "none"],
}
BIN_EDGES = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 100000]


def planted_graph(n: int = 1500, m: int = 12, seed: int = 1, communities: bool = True) -> AttributedGraph:
    g = nx.powerlaw_cluster_graph(n, m, 0.6, seed=seed)
    rng = np.random.default_rng(seed)
    a = np.full(n, 3)
    if communities:  # greedy modularity is slow at scale; the timing test turns it off
        comms = sorted(nx.community.greedy_modularity_communities(g), key=len, reverse=True)
        for i, com in enumerate(comms[:3]):
            a[list(com)] = i
    else:
        a = rng.integers(0, 4, n)
    b = np.where(rng.random(n) < 0.6, a % 2, rng.integers(0, 4, n))
    deg = np.array([g.degree(i) for i in range(n)])
    c = np.where(deg > np.median(deg), 1, 0)
    d = rng.integers(0, 4, n)
    attrs = pd.DataFrame({"a": a, "b": b, "c": c, "d": d}).astype(np.int64)
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in g.edges()), dtype=np.int64)
    return AttributedGraph(n=n, edges=edges, attrs=attrs, domains={k: list(v) for k, v in DOMAINS.items()})


def exact_artifact(
    g: AttributedGraph, noise: float = 0.0, seed: int = 0, bin_edges: list | None = None
) -> Artifact:
    """Noise-free (or Gaussian-noised, un-postprocessed) measurements following types.Measurement."""
    rng = np.random.default_rng(seed)
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    dmax = int(deg.max())
    adj = sp.coo_matrix((np.ones(len(g.edges)), (g.edges[:, 0], g.edges[:, 1])), shape=(g.n,) * 2)
    adj = (adj + adj.T).tocsr()
    tri = np.asarray((adj @ adj).multiply(adj).sum(axis=1)).ravel() / 2.0
    clus = np.where(deg > 1, tri / np.maximum(deg * (deg - 1) / 2.0, 1.0), 0.0)
    bin_edges = BIN_EDGES if bin_edges is None else bin_edges
    be = np.array(bin_edges, dtype=float)
    nb = len(be) - 1
    dbin = np.clip(np.searchsorted(be, deg, side="right") - 1, 0, nb - 1)

    ms: dict[str, Measurement] = {}

    def add(name, kind, values, **params):
        values = np.asarray(values, dtype=float)
        if noise:
            values = values + rng.normal(0, noise, values.shape)
        ms[name] = Measurement(name, kind, values, "gaussian", noise, 0.0, 0.0, params)

    add("degree_hist", "degree_hist", np.bincount(deg, minlength=dmax + 1))
    cl = np.zeros((nb, 2))
    np.add.at(cl[:, 0], dbin, clus)
    np.add.at(cl[:, 1], dbin, 1.0)
    add("clustering_by_degree", "clustering_by_degree", cl, bin_edges=bin_edges)
    jd = np.zeros((nb, nb))
    bu, bv = dbin[g.edges[:, 0]], dbin[g.edges[:, 1]]
    np.add.at(jd, (bu, bv), 1.0)
    np.add.at(jd, (bv, bu), 1.0)
    jd[np.diag_indices(nb)] /= 2.0  # diagonal = edges inside the bin, counted once
    add("joint_degree", "joint_degree", jd, bin_edges=bin_edges)
    cols = list(DOMAINS)
    sz = {c: len(DOMAINS[c]) for c in cols}
    for c in cols:
        add(f"1way_{c}", "attr_1way", np.bincount(g.attrs[c], minlength=sz[c]), attrs=[c])
    for i, ca in enumerate(cols):
        for cb in cols[i + 1 :]:
            t = np.zeros((sz[ca], sz[cb]))
            np.add.at(t, (g.attrs[ca].to_numpy(), g.attrs[cb].to_numpy()), 1.0)
            add(f"2way_{ca}_{cb}", "attr_2way", t, attrs=[ca, cb])
    gr = g.attrs["a"].to_numpy()
    k = sz["a"]
    ep = np.zeros((k, k))
    gu, gv = gr[g.edges[:, 0]], gr[g.edges[:, 1]]
    np.add.at(ep, (gu, gv), 1.0)
    np.add.at(ep, (gv, gu), 1.0)
    ep[np.diag_indices(k)] /= 2.0
    add("edge_pair", "attr_edge_pair", ep, attrs=["a"])

    schema = Schema(g.n, {k_: list(v) for k_, v in DOMAINS.items()}, dmax, 0, ["a"])
    return Artifact(schema, ms, PrivacyLedger(1e-6, float("inf"), float("inf")))


def group_matrix(g: AttributedGraph) -> np.ndarray:
    gr = g.attrs["a"].to_numpy()
    k = len(DOMAINS["a"])
    m = np.zeros((k, k))
    np.add.at(m, (gr[g.edges[:, 0]], gr[g.edges[:, 1]]), 1.0)
    m = m + m.T
    m[np.diag_indices(k)] /= 2.0
    return np.triu(m) / np.triu(m).sum()


def avg_clustering(g: AttributedGraph) -> float:
    x = nx.Graph()
    x.add_nodes_from(range(g.n))
    x.add_edges_from(map(tuple, g.edges))
    return nx.average_clustering(x)


def degrees(g: AttributedGraph) -> np.ndarray:
    return np.bincount(g.edges.ravel(), minlength=g.n)


@pytest.fixture(scope="module")
def planted():
    return planted_graph()


@pytest.fixture(scope="module")
def clean_artifact(planted):
    return exact_artifact(planted)


@pytest.fixture(scope="module")
def synthetic(clean_artifact):
    return generate(clean_artifact, seed=3)


def check_valid(g: AttributedGraph, n: int, domains: dict) -> None:
    assert g.n == n
    e = g.edges
    assert e.dtype == np.int64 and e.ndim == 2 and e.shape[1] == 2
    assert (e[:, 0] < e[:, 1]).all()
    assert e.min() >= 0 and e.max() < n
    assert len(np.unique(e[:, 0] * n + e[:, 1])) == len(e)
    assert list(g.attrs.index) == list(range(n))
    assert list(g.attrs.columns) == list(domains)
    for c, dom in domains.items():
        assert g.attrs[c].min() >= 0 and g.attrs[c].max() < len(dom)
    assert g.domains == domains


def test_validity(synthetic, planted):
    check_valid(synthetic, planted.n, DOMAINS)


def test_deterministic(clean_artifact):
    a, b = generate(clean_artifact, seed=7), generate(clean_artifact, seed=7)
    assert np.array_equal(a.edges, b.edges)
    assert a.attrs.equals(b.attrs)
    c = generate(clean_artifact, seed=8)
    assert not np.array_equal(a.edges, c.edges)


def test_fidelity_ceiling(planted, synthetic, capsys):
    ks = stats.ks_2samp(degrees(planted), degrees(synthetic)).statistic
    c_p, c_s = avg_clustering(planted), avg_clustering(synthetic)
    mp, ms_ = group_matrix(planted), group_matrix(synthetic)
    l1 = np.abs(mp - ms_).sum()
    m_p, m_s = len(planted.edges), len(synthetic.edges)
    print(f"\nFIDELITY ks={ks:.4f} clus planted={c_p:.4f} gen={c_s:.4f} "
          f"edgepair_L1={l1:.4f} edges planted={m_p} gen={m_s}")
    assert ks < 0.1
    assert abs(c_s - c_p) < 0.15 * c_p + 0.03
    assert l1 < 0.15
    assert abs(m_s - m_p) < 0.1 * m_p


def test_attribute_marginals_and_pairs(planted, synthetic):
    for c in DOMAINS:
        p = np.bincount(planted.attrs[c], minlength=len(DOMAINS[c])) / planted.n
        s = np.bincount(synthetic.attrs[c], minlength=len(DOMAINS[c])) / synthetic.n
        assert 0.5 * np.abs(p - s).sum() < 0.06
    # a tree edge (a, b) should keep its dependence
    def joint(g):
        t = np.zeros((5, 4))
        np.add.at(t, (g.attrs["a"].to_numpy(), g.attrs["b"].to_numpy()), 1.0)
        return t / t.sum()
    assert 0.5 * np.abs(joint(planted) - joint(synthetic)).sum() < 0.08


def test_rescale_n(clean_artifact):
    g = generate(clean_artifact, n=700, seed=1)
    check_valid(g, 700, DOMAINS)
    assert len(g.edges) > 3000  # edge count scales with n (about 12 per node in the fixture)


@pytest.mark.parametrize("noise", [2.0, 20.0])
def test_noisy_measurements_robust(planted, noise):
    art = exact_artifact(planted, noise=noise, seed=5)
    g = generate(art, seed=2)
    check_valid(g, planted.n, DOMAINS)
    assert len(g.edges) > 0
    if noise <= 2.0:
        assert stats.ks_2samp(degrees(planted), degrees(g)).statistic < 0.2


def test_all_zero_and_negative_cells(planted):
    art = exact_artifact(planted)
    for m in art.measurements.values():
        if m.kind in ("attr_2way", "attr_1way", "attr_edge_pair", "joint_degree"):
            m.values = -np.abs(m.values) if m.kind == "attr_1way" else np.zeros_like(m.values)
    art.measurements["clustering_by_degree"].values = -np.ones_like(
        art.measurements["clustering_by_degree"].values
    )
    g = generate(art, seed=1)
    check_valid(g, planted.n, DOMAINS)


def test_missing_optional_measurements(planted):
    art = exact_artifact(planted)
    for name in ["joint_degree", "edge_pair", "clustering_by_degree", "2way_a_b"]:
        art.measurements.pop(name)
    g = generate(art, seed=1)
    check_valid(g, planted.n, DOMAINS)
    assert len(g.edges) > 0


def test_missing_degree_hist_raises(clean_artifact):
    art = Artifact(clean_artifact.schema, {}, clean_artifact.ledger)
    with pytest.raises(ValueError):
        generate(art)


def test_runtime_facebook_scale():
    g = planted_graph(n=4039, m=22, seed=2, communities=False)
    art = exact_artifact(g)
    t0 = time.time()
    s = generate(art, seed=1)
    dt = time.time() - t0
    print(f"\nRUNTIME n=4039 edges real={len(g.edges)} gen={len(s.edges)} seconds={dt:.2f}")
    check_valid(s, 4039, DOMAINS)
    assert dt < 60


# ------------------------------------------------------------------------------ round 2 (R1) tests
def assortativity(g: AttributedGraph) -> float:
    x = nx.Graph()
    x.add_nodes_from(range(g.n))
    x.add_edges_from(map(tuple, g.edges))
    return nx.degree_assortativity_coefficient(x)


def test_assortativity_and_triangles_on_clustered_planted_graph(planted, synthetic, capsys):
    """Planted clustered graph (powerlaw-cluster, communities as attribute): known assortativity."""
    a_p, a_s = assortativity(planted), assortativity(synthetic)
    print(f"\nASSORT planted={a_p:.4f} gen={a_s:.4f}")
    assert abs(a_s - a_p) < 0.1  # the original sampler was off by far more on real data when blocks tied degrees
    tri = lambda g: sum(nx.triangles(_nx(g)).values()) / 3.0  # noqa: E731
    assert abs(tri(synthetic) - tri(planted)) < 0.3 * tri(planted)


def _nx(g: AttributedGraph) -> nx.Graph:
    x = nx.Graph()
    x.add_nodes_from(range(g.n))
    x.add_edges_from(map(tuple, g.edges))
    return x


def test_assortative_planted_graph_is_reproduced():
    """Degree-assortative target: rewire a planted graph to be strongly assortative (known, high)."""
    g0 = planted_graph(n=1200, m=10, seed=4, communities=False)
    x = _nx(g0)
    # greedy assortative rewiring: swap to align degrees (deterministic, seeded)
    rng = np.random.default_rng(0)
    deg = dict(x.degree())
    edges = [tuple(e) for e in x.edges()]
    for _ in range(40000):
        i, j = rng.integers(0, len(edges), 2)
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) < 4:
            continue
        # pair the two higher with each other
        lo = sorted([a, b, c, d], key=lambda v: deg[v])
        new = [(lo[0], lo[1]), (lo[2], lo[3])]
        if any(x.has_edge(*e) for e in new):
            continue
        x.remove_edge(a, b)
        x.remove_edge(c, d)
        x.add_edges_from(new)
        edges[i], edges[j] = new
    e = np.array(sorted((min(u, v), max(u, v)) for u, v in x.edges()), dtype=np.int64)
    g = AttributedGraph(g0.n, e, g0.attrs, g0.domains)
    a_p = assortativity(g)
    assert a_p > 0.2
    syn = generate(exact_artifact(g), seed=0)
    a_s = assortativity(syn)
    print(f"\nASSORT strongly assortative planted={a_p:.4f} gen={a_s:.4f}")
    assert abs(a_s - a_p) < 0.25 * a_p + 0.05


@pytest.mark.parametrize("rewire_rounds", [0, 40])
def test_edges_sorted_unique_no_loops(clean_artifact, planted, rewire_rounds):
    g = generate(clean_artifact, seed=11, rewire_rounds=rewire_rounds)
    e = g.edges
    assert e.dtype == np.int64 and e.ndim == 2 and e.shape[1] == 2
    assert (e[:, 0] < e[:, 1]).all()  # u < v, hence no loops
    assert len(np.unique(e[:, 0] * g.n + e[:, 1])) == len(e)
    assert 0 <= e.min() and e.max() < g.n


def test_rewire_preserves_degrees_and_simplicity(planted, clean_artifact):
    from graph_aim.outside.generate import _rewire

    n = planted.n
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 6, n)
    target = rng.random((6, 6))
    target = target + target.T
    e = planted.edges
    out = _rewire(e, np.ones(len(e), dtype=bool), cls, target, n, rng, rounds=10)
    assert np.array_equal(degrees(AttributedGraph(n, out, planted.attrs, planted.domains)), degrees(planted))
    assert (out[:, 0] < out[:, 1]).all()
    assert len(np.unique(out[:, 0] * n + out[:, 1])) == len(out)
    # fixed edges are never touched
    fixed = np.zeros(len(e), dtype=bool)
    fixed[:10] = True
    out2 = _rewire(e, ~fixed, cls, target, n, np.random.default_rng(1), rounds=10)
    assert np.array_equal(out2[fixed], e[fixed])


def test_round2_contract_ignores_D_and_cap_params(planted):
    """Max degree comes from len(degree_hist.values) - 1; params D / cap are not consulted."""
    base = exact_artifact(planted)
    ref = generate(base, seed=4)
    art = exact_artifact(planted)
    for m in art.measurements.values():
        m.params["cap"] = 7  # per-measurement cap, irrelevant for the generator
        m.params["D"] = 3  # would truncate degrees if (wrongly) used
    art.measurements["degree_hist"].params["max_degree"] = int(degrees(planted).max())
    g = generate(art, seed=4)
    assert np.array_equal(g.edges, ref.edges)
    assert degrees(g).max() > 40  # hubs well above any cap survive


def test_round2_uncapped_degree_hist_with_capped_structure_bins(planted):
    """degree_hist on the uncapped graph, clustering / joint_degree binned only up to a cap."""
    capped_bins = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 61]
    art = exact_artifact(planted, bin_edges=capped_bins)
    for m in art.measurements.values():
        if m.kind in ("clustering_by_degree", "joint_degree"):
            m.params["cap"] = 60
    g = generate(art, seed=2)
    check_valid(g, planted.n, DOMAINS)
    ks = stats.ks_2samp(degrees(planted), degrees(g)).statistic
    assert ks < 0.1
    assert abs(len(g.edges) - len(planted.edges)) < 0.1 * len(planted.edges)


def _coarse_noisy_hist_artifact(planted, sigma, seed):
    """Artifact whose degree_hist went through the real coarse-grid noise + post-processing path."""
    from graph_aim.inside import measure as M

    art = exact_artifact(planted)
    deg = degrees(planted)
    grid = M.hist_grid(int(deg.max()) + 5, 8, 1.5)
    cell = np.searchsorted(np.array(grid), deg, side="right") - 1
    exact = np.bincount(cell, minlength=len(grid) - 1).astype(float)
    raw = exact + np.random.default_rng(seed).normal(0, sigma, len(exact))
    params = {"max_degree": grid[-1] - 1, "cap": None, "hist_edges": grid, "threshold": 2.0, "spread_power": 1.0}
    values = M._degree_hist_post(raw, M.Spec("degree_hist", "degree_hist", params), sigma, planted.n)
    old = art.measurements["degree_hist"]
    art.measurements["degree_hist"] = Measurement(old.name, old.kind, values, "gaussian", sigma, 0.0, 0.0, params)
    return art


def test_systematic_rounding_keeps_expected_hub_count():
    from graph_aim.outside.generate import _sample_degrees

    h = np.zeros(1000)
    h[5] = 995.0
    h[400:500] = 0.05  # 5 expected hubs spread over 100 degrees, each far below one node
    meas = Measurement("degree_hist", "degree_hist", h, "gaussian", 0.0, 0.0, 0.0, {})
    art = Artifact(Schema(1000, DOMAINS, 5, 0, ["a"]), {"degree_hist": meas}, PrivacyLedger(1e-6, 0.0, 0.0))
    for seed in range(8):
        deg = _sample_degrees(art, 1000, np.random.default_rng(seed))
        assert len(deg) == 1000 and abs(int((deg >= 400).sum()) - 5) <= 1


def test_noisy_coarse_hist_tail_is_not_deleted_by_generator(planted):
    """Hub tail of a noisy coarse histogram must survive post-processing + generation (assortativity)."""
    from graph_aim.eval.evaluate import structure_metrics

    rng = np.random.default_rng(7)  # plant six hubs of degree ~600 (they dominate the degree variance)
    extra = [(h, v) for h in range(6) for v in rng.choice(np.arange(6, planted.n), 600, replace=False)]
    e = np.unique(np.vstack([planted.edges, np.array(extra)]), axis=0)
    hubby = AttributedGraph(planted.n, e, planted.attrs, planted.domains)
    real_assort = structure_metrics(hubby)["assortativity"]
    errs, maxdeg = [], []
    for seed in range(3):
        g = generate(_coarse_noisy_hist_artifact(hubby, sigma=6.0, seed=seed), seed=seed)
        check_valid(g, planted.n, DOMAINS)
        errs.append(abs(structure_metrics(g)["assortativity"] - real_assort))
        maxdeg.append(degrees(g).max())
    assert np.mean(errs) < 0.25
    assert np.mean(maxdeg) > 0.5 * degrees(hubby).max()

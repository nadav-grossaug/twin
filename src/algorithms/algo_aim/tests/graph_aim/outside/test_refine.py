"""Tests for the label chain (``graph_aim.outside.refine``) and the v2 generator (plan section 5.4).

The ``attr_neigh_simpson`` measurements are built BY HAND here (the exact statistic of the planted graph,
computed with ``neigh_simpson_sums``), following the contract of plan section 4.1; nothing from the inside zone
is needed to produce them. Attribute names are the generic ones of ``test_generate``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from graph_aim.outside import refine as R
from graph_aim.outside.generate import generate
from graph_aim.outside.generate_v2 import generate_v2, generate_v2_with_info
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement

from test_generate import DOMAINS, exact_artifact, planted_graph
from test_generate_attr_degree import FALLBACK_DIGESTS

REPO = Path(__file__).resolve().parents[3]
EDGES = [0, 4, 8, 16, 32, 10**6]  # strata and Simpson bins; dmin is an edge
DMIN = 4
KIND = "attr_neigh_simpson"


# ---------------------------------------------------------------------------------------- helpers
def degrees(g: AttributedGraph) -> np.ndarray:
    return np.bincount(g.edges.ravel(), minlength=g.n)


def adjacency(g: AttributedGraph) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(len(g.edges)), (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n))
    return (a + a.T).tocsr()


def codes(g: AttributedGraph, cols: list[str]) -> tuple[np.ndarray, int]:
    c = np.zeros(g.n, dtype=np.int64)
    for col in cols:
        c = c * len(g.domains[col]) + g.attrs[col].to_numpy()
    return c, int(np.prod([len(g.domains[col]) for col in cols]))


def simpson_sums(g: AttributedGraph, cols: list[str], edges=EDGES, dmin: int = DMIN) -> np.ndarray:
    c, size = codes(g, cols)
    return R.neigh_simpson_sums(adjacency(g), c, size, degrees(g), edges, dmin)


def simpson_measurement(g: AttributedGraph, cols: list[str], sigma: float = 0.0, edges=EDGES, dmin: int = DMIN,
                        seed: int = 0) -> Measurement:
    """Hand-built ``attr_neigh_simpson`` instance: column 0 = exact per-bin sums (+ optional noise), column 1 = counts."""
    sums = simpson_sums(g, cols, edges, dmin)
    if sigma:
        sums = sums + np.random.default_rng(seed).normal(0, sigma, sums.shape)
    cnt = np.bincount(degree_bin(degrees(g), edges), minlength=len(edges) - 1)
    name = f"{KIND}:" + "+".join(cols)
    return Measurement(name, KIND, np.stack([sums, cnt], axis=1).astype(float), "gaussian", sigma, 0.0, 0.0,
                       {"attrs": list(cols), "cap": 10**6, "bin_edges": list(edges), "min_degree": dmin,
                        "counts_from_hist": True})


def attr_degree_table(g: AttributedGraph, col: str, edges=EDGES) -> np.ndarray:
    t = np.zeros((len(g.domains[col]), len(edges) - 1))
    np.add.at(t, (g.attrs[col].to_numpy(), degree_bin(degrees(g), edges)), 1.0)
    return t


def add_attr_degree(art: Artifact, g: AttributedGraph, edges=EDGES) -> None:
    for col in g.domains:
        art.measurements[f"attr_degree:{col}"] = Measurement(
            f"attr_degree:{col}", "attr_degree", attr_degree_table(g, col, edges), "gaussian", 0.0, 0.0, 0.0,
            {"attrs": [col], "cap": None, "bin_edges": list(edges)})


def v2_artifact(g: AttributedGraph, groups=(["a"], ["b"], ["a", "b"]), strata: bool = True, sigma: float = 0.0) -> Artifact:
    art = exact_artifact(g)
    if strata:
        add_attr_degree(art, g)
    for cols in groups:
        m = simpson_measurement(g, list(cols), sigma)
        art.measurements[m.name] = m
    return art


def two_way(g: AttributedGraph, a: str, b: str) -> np.ndarray:
    t = np.zeros((len(g.domains[a]), len(g.domains[b])))
    np.add.at(t, (g.attrs[a].to_numpy(), g.attrs[b].to_numpy()), 1.0)
    return t


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


# --------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def planted() -> AttributedGraph:
    return planted_graph()  # 1500 nodes, three communities carried by attribute "a" (and 60% of "b")


@pytest.fixture(scope="module")
def artifact(planted) -> Artifact:
    return v2_artifact(planted)


@pytest.fixture(scope="module")
def base(artifact) -> AttributedGraph:
    return generate(artifact, seed=3)


@pytest.fixture(scope="module")
def refined(artifact):
    return generate_v2_with_info(artifact, seed=3)


# ------------------------------------------------------------------------ the pure Simpson helper
def test_neigh_simpson_sums_matches_definition():
    """Plan 4.1 on a tiny hand graph: s_u = [d_u >= dmin] sum_g (n_ug / d_u)^2, binned by degree."""
    # node 0: neighbours 1,2,3 (codes 1,1,0); node 1: neighbours 0,2 (codes 0,1); node 2: 0,1,3,4; node 3: 0,2; node 4: 2
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [2, 4]])
    g = AttributedGraph(5, edges, pd.DataFrame({"x": [0, 1, 1, 0, 1]}), {"x": ["p", "q"]})
    c = g.attrs["x"].to_numpy()
    deg = degrees(g)
    bins = [0, 2, 3, 10]
    got = R.neigh_simpson_sums(adjacency(g), c, 2, deg, bins, 2)
    exp = np.zeros(3)
    for u in range(5):
        nb = [v for e in edges for v in e if u in e and v != u]
        if deg[u] >= 2:
            s = sum((np.sum(c[nb] == k) / deg[u]) ** 2 for k in range(2))
            exp[degree_bin(np.array([deg[u]]), bins)[0]] += s
    assert np.allclose(got, exp)
    assert got[0] == 0.0  # the degree-1 node is excluded by dmin = 2
    # perfectly concentrated neighbourhoods give exactly the node count
    allsame = R.neigh_simpson_sums(adjacency(g), np.zeros(5, dtype=np.int64), 2, deg, bins, 1)
    assert np.allclose(allsame, np.bincount(degree_bin(deg, bins), minlength=3))


# ---------------------------------------------------------------------------- the main planted test
def test_simpson_sums_reach_targets_from_the_iid_level(planted, base, refined, capsys):
    out, info = refined
    assert not info["skipped"] and info["warnings"] == []
    assert info["moving_attrs"] == ["a", "b"]
    d = degrees(out)
    for cols in (["a"], ["b"], ["a", "b"]):
        name = f"{KIND}:" + "+".join(cols)
        t = info["terms"][name]
        use = np.array(t["bins_used"])
        use = use[np.array(t["counts"])[use] > 0]  # populated bins only (the planted graph has no degree < 12)
        target = np.array(t["target"])[use]
        counts = np.array(t["counts"])[use]
        c0, size = codes(base, cols)
        before = R.neigh_simpson_sums(adjacency(base), c0, size, d, EDGES, DMIN)[use]
        after = simpson_sums(out, cols)[use]
        assert np.allclose(after, np.array(t["after"])[use])  # the report is the statistic of the output
        gap_before = np.abs(before - target) / counts
        gap_after = np.abs(after - target) / counts
        print(f"\nSIMPSON {name}: target={np.round(target, 1)} before={np.round(before, 1)} after={np.round(after, 1)}")
        assert gap_after.max() < 0.02  # per-bin mean within 0.02 of the released mean (acceptance criterion 1)
        if cols != ["b"]:  # "a" carries the communities: the i.i.d. start is far off, the chain closes the gap
            assert gap_before.max() > 0.05
            assert gap_after.sum() < 0.2 * gap_before.sum()
    assert info["energy_after"] < 0.1 * info["energy_before"]
    assert info["rounds"] >= 1 and len(info["accepted"]) == info["rounds"] == len(info["proposed"])
    assert sum(info["accepted"]) > 0


def test_marginals_attr_degree_and_untouched_attribute_are_exact(base, refined):
    out, _ = refined
    for col in DOMAINS:
        assert np.array_equal(np.bincount(out.attrs[col], minlength=9), np.bincount(base.attrs[col], minlength=9)), col
        assert np.array_equal(attr_degree_table(out, col), attr_degree_table(base, col)), col
    # "c" and "d" appear in no attr_neigh_simpson / attr_edge_pair instance: never moved
    assert out.attrs["c"].equals(base.attrs["c"]) and out.attrs["d"].equals(base.attrs["d"])
    assert (out.attrs["a"] != base.attrs["a"]).sum() > 50  # the chain did move the labels it controls
    assert list(out.attrs.columns) == list(base.attrs.columns) and list(out.attrs.index) == list(range(out.n))
    assert out.attrs.dtypes.equals(base.attrs.dtypes)


def test_two_way_tables_stay_within_tolerance(planted, base, refined):
    out, info = refined
    cols = list(DOMAINS)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            assert tvd(two_way(out, a, b), two_way(planted, a, b)) < 0.05, (a, b)
            assert tvd(two_way(out, a, b), two_way(base, a, b)) < 0.05, (a, b)
    assert all(t["cells"] > 0 and np.isfinite(t["energy_after"]) for t in info["terms"].values())


def test_edges_identical_to_generate(base, refined):
    out, _ = refined
    assert np.array_equal(out.edges, base.edges) and out.n == base.n and out.domains == base.domains


def test_deterministic_per_seed(artifact):
    a = generate_v2(artifact, seed=3, max_rounds=20)
    b = generate_v2(artifact, seed=3, max_rounds=20)
    assert a.attrs.equals(b.attrs) and np.array_equal(a.edges, b.edges)
    c = generate_v2(artifact, seed=4, max_rounds=20)
    assert not a.attrs.equals(c.attrs)


# ----------------------------------------------------------------------------------- fallback
@pytest.mark.parametrize("seed", [0, 7])
def test_without_the_kind_v2_is_byte_identical_to_v1(seed):
    art = exact_artifact(planted_graph(n=300, m=5, seed=3))
    assert not [m for m in art.measurements.values() if m.kind == KIND]
    out, info = generate_v2_with_info(art, seed=seed)
    assert info["skipped"] and info["rounds"] == 0
    h = hashlib.sha256(out.edges.tobytes() + out.attrs.to_numpy().astype("int64").tobytes()).hexdigest()
    assert h == FALLBACK_DIGESTS[seed]
    ref = generate(art, seed=seed)
    assert ref.attrs.equals(out.attrs) and np.array_equal(ref.edges, out.edges)


def test_refine_labels_without_the_kind_returns_the_input_object(base, planted):
    art = exact_artifact(planted)
    g, info = R.refine_labels(base, art, seed=0)
    assert g is base and info["skipped"] and info["reason"] == "no attr_neigh_simpson measurement"


# ------------------------------------------------------------------------ inconsistent / malformed
def test_inconsistent_targets_terminate_and_report_residuals(planted):
    art = v2_artifact(planted, groups=(["a"],))
    m = art.measurements[f"{KIND}:a"]
    m.values[:, 0] = m.values[:, 1]  # mean 1.0: every neighbourhood pure, unreachable on this structure
    m2 = simpson_measurement(planted, ["a", "b"])
    m2.values[:, 0] = 0.0  # mean 0: below the minimum possible 1 / G
    art.measurements[m2.name] = m2
    out, info = generate_v2_with_info(art, seed=1, max_rounds=40)
    assert not info["skipped"] and info["rounds"] <= 40
    assert info["stop"] in ("max_rounds", "no_accepted_move") and not info["satisfied"]
    for name in (m.name, m2.name):
        t = info["terms"][name]
        assert not t["satisfied"] and np.all(np.isfinite(t["residual"]))
        assert t["max_abs_residual_mean"] > 0.05  # the residual is reported, not hidden
    assert np.isfinite(info["energy_after"]) and info["energy_after"] <= info["energy_before"] + 1e-6
    for col in DOMAINS:  # the invariants hold whatever the targets say
        assert np.array_equal(attr_degree_table(out, col), attr_degree_table(generate(art, seed=1), col))


def test_unexpected_shapes_are_skipped_with_a_warning(planted):
    art = v2_artifact(planted, groups=(["a"],))
    bad = simpson_measurement(planted, ["b"])
    bad.values = bad.values[:, :1].copy()  # (B, 1) instead of (B, 2)
    art.measurements[bad.name] = bad
    bad2 = simpson_measurement(planted, ["d"])
    bad2.params["bin_edges"] = [3, 8, 10**6]  # does not start at 0
    art.measurements[bad2.name] = bad2
    bad3 = simpson_measurement(planted, ["c"])
    bad3.params["attrs"] = ["nope"]
    art.measurements[bad3.name] = bad3
    out, info = generate_v2_with_info(art, seed=2, max_rounds=5)
    assert not info["skipped"] and info["moving_attrs"] == ["a"]
    assert len(info["warnings"]) == 3 and all(any(w.startswith(n) for n in (bad.name, bad2.name, bad3.name)) for w in info["warnings"])
    assert out.attrs["b"].equals(generate(art, seed=2).attrs["b"])
    # only malformed instances: the chain is skipped, nothing crashes
    art2 = exact_artifact(planted)
    art2.measurements[bad.name] = bad
    out2, info2 = generate_v2_with_info(art2, seed=2)
    assert info2["skipped"] and info2["reason"] == "no usable attr_neigh_simpson measurement" and info2["warnings"]
    assert out2.attrs.equals(generate(art2, seed=2).attrs)


# ------------------------------------------------------------------------------------ edge cases
def test_batch_of_one(planted, artifact):
    out, info = generate_v2_with_info(artifact, seed=1, batch=1, max_rounds=6)
    assert not info["skipped"] and info["rounds"] >= 1
    assert all(p <= len(info["moving_attrs"]) for p in info["proposed"])  # one proposal per attribute tried
    assert all(a <= 1 for a in info["accepted"])
    base = generate(artifact, seed=1)
    for col in DOMAINS:
        assert np.array_equal(attr_degree_table(out, col), attr_degree_table(base, col))


def test_single_stratum_without_attr_degree(planted):
    art = v2_artifact(planted, groups=(["a"],), strata=False)
    assert not [m for m in art.measurements.values() if m.kind == "attr_degree"]
    out, info = generate_v2_with_info(art, seed=2, max_rounds=60)
    assert info["strata"] == {"bin_edges": None, "count": 1}
    base = generate(art, seed=2)
    assert np.array_equal(out.edges, base.edges)
    for col in DOMAINS:
        assert np.array_equal(np.bincount(out.attrs[col], minlength=9), np.bincount(base.attrs[col], minlength=9))
    t = info["terms"][f"{KIND}:a"]
    assert t["energy_after"] < 0.2 * t["energy_before"]


def test_attribute_in_no_instance_is_untouched(planted):
    art = v2_artifact(planted, groups=(["b"],))
    art.measurements.pop("edge_pair")  # now only "b" moves
    out, info = generate_v2_with_info(art, seed=5, max_rounds=30)
    assert info["moving_attrs"] == ["b"]
    base = generate(art, seed=5)
    for col in ("a", "c", "d"):
        assert out.attrs[col].equals(base.attrs[col]), col
    assert (out.attrs["b"] != base.attrs["b"]).any()


@pytest.mark.parametrize("n", [700, 2200])
def test_n_differs_from_schema(planted, artifact, n):
    out, info = generate_v2_with_info(artifact, n=n, seed=1, max_rounds=100)
    assert out.n == n and len(out.attrs) == n and out.edges.max() < n
    base = generate(artifact, n=n, seed=1)
    assert np.array_equal(out.edges, base.edges)
    t = info["terms"][f"{KIND}:a"]
    counts = np.bincount(degree_bin(degrees(out), EDGES), minlength=len(EDGES) - 1)
    assert np.array_equal(np.array(t["counts"]), counts)  # targets scale with the synthetic bin counts
    m = artifact.measurements[f"{KIND}:a"]
    mean = m.values[:, 0] / np.maximum(m.values[:, 1], 1)
    use = np.array(t["bins_used"])
    assert np.allclose(np.array(t["target"])[use], (mean * counts)[use])
    assert t["energy_after"] < 0.3 * t["energy_before"]
    for col in DOMAINS:
        assert np.array_equal(attr_degree_table(out, col), attr_degree_table(base, col))


def test_noisy_targets_and_sigma_stop_rule(planted):
    """With sigma > 0 the energy is in noise units, so a large sigma makes the terms satisfiable; the chain
    stops early once every term is within its cell count. With a small sigma (8) on this planted graph the
    Simpson term of the community attribute and the edge-pair term cannot both be met on the v1 structure:
    the chain ends in a compromise and reports the loser, with the energy never rising."""
    art = v2_artifact(planted, groups=(["a"], ["b"]), sigma=30.0)
    for m in art.measurements.values():
        if m.kind in ("attr_2way", "attr_edge_pair"):
            m.sigma = 30.0
    out, info = generate_v2_with_info(art, seed=0)
    assert info["stop"] == "satisfied" and info["satisfied"] and info["rounds"] < 200
    assert all(t["satisfied"] for t in info["terms"].values())

    art = v2_artifact(planted, groups=(["a"], ["b"]), sigma=8.0)
    for m in art.measurements.values():
        if m.kind in ("attr_2way", "attr_edge_pair"):
            m.sigma = 8.0
    out, info = generate_v2_with_info(art, seed=0)
    e = np.array(info["energy"])
    assert np.all(np.diff(e) <= 1e-6)  # the prefix search keeps the exact energy non-increasing
    assert info["terms"][f"{KIND}:b"]["satisfied"] and info["terms"]["edge_pair"]["satisfied"]
    assert info["terms"][f"{KIND}:a"]["max_abs_residual_mean"] < 0.05  # the loser is still within 2-3 sigma


def test_temperature_and_weights_options_run(artifact):
    out, info = generate_v2_with_info(artifact, seed=1, max_rounds=5, temperature=0.5, weights={f"{KIND}:a": 3.0})
    assert info["rounds"] >= 1 and info["terms"][f"{KIND}:a"]["weight"] == 3.0
    assert np.isfinite(info["energy_after"])


# -------------------------------------------------------------------------- change statistics
def test_change_statistics_match_exact_recomputation(planted):
    """Per-proposal deltas: exact for edge-pair and 2-way terms; the Simpson closed form is exact when the two
    endpoints share no neighbour (and only linearised otherwise)."""
    art = v2_artifact(planted_graph(n=400, m=6, seed=2), groups=(["a"], ["a", "b"]))
    syn = generate(art, seed=1)
    chain = R._Chain(syn, art, max_rounds=1, batch=None, damping=1.0, temperature=0.0, weights=None)
    chain.exact()
    rng = np.random.default_rng(0)
    checked = {"attr_edge_pair": 0, "attr_2way": 0, "attr_neigh_simpson": 0}
    for _ in range(200):
        a = chain.moving[rng.integers(len(chain.moving))]
        x = chain.x[a]
        u, v = (int(i) for i in rng.integers(chain.n, size=2))
        if u == v or x[u] == x[v]:
            continue
        nu = set(chain.indices[chain.indptr[u]:chain.indptr[u + 1]])
        nv = set(chain.indices[chain.indptr[v]:chain.indptr[v + 1]])
        if v in nu:
            continue
        shared = bool(nu & nv)
        uu, vv = np.array([u]), np.array([v])
        d = x[vv] - x[uu]
        nbr_u, pid_u = R._neighbours(chain.indptr, chain.indices, uu)
        nbr_v, pid_v = R._neighbours(chain.indptr, chain.indices, vv)
        pred = {}
        for t in chain.pairs + chain.simpson:
            s = t.strides.get(a, 0)
            if s == 0:
                continue
            gu, gv = t.g[uu], t.g[vv]
            gnu, gnv = gu + d * s, gv - d * s
            if isinstance(t, R._EdgePairTerm):
                pred[t.name] = float(t.delta(uu, vv, gu, gnu, gv, gnv)[0])
            elif not shared:
                pred[t.name] = float(t.delta(1, gu, gnu, gv, gnv, nbr_u, pid_u, nbr_v, pid_v)[0][0])
        for t in chain.twoway:
            if a not in (t.a, t.b):
                continue
            other = t.b if a == t.a else t.a
            pred[t.name] = float(t.delta(a, x[uu], x[vv], chain.x[other][uu], chain.x[other][vv])[0][0])
        before = {t.name: t.energy for t in chain.terms}
        # apply the swap for real on a fresh chain and recompute everything from scratch
        other_chain = R._Chain(syn, art, max_rounds=1, batch=None, damping=1.0, temperature=0.0, weights=None)
        other_chain.x[a][u], other_chain.x[a][v] = x[v], x[u]
        other_chain.simpson, other_chain.pairs, other_chain.twoway, _ = R._build_terms(
            syn, art, other_chain.x, other_chain.deg, len(syn.edges), {}, [])
        for t in other_chain.simpson + other_chain.pairs:
            t.init_counts(other_chain.indptr, other_chain.indices, other_chain.n)
        other_chain.exact()
        after = {t.name: t.energy for t in other_chain.terms}
        for name, p in pred.items():
            kind = next(t.kind for t in chain.terms if t.name == name)
            assert abs((after[name] - before[name]) - p) < 1e-6, (name, kind)
            checked[kind] += 1
    assert all(c > 10 for c in checked.values()), checked


# ----------------------------------------------------------------------------- stage-2 script
def _toy_graph() -> AttributedGraph:
    rng = np.random.default_rng(11)
    n = 150
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < 0.06
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    domains = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in domains.items()})
    return AttributedGraph(n, edges, attrs, domains)


def _fitted_toy_artifact():
    from graph_aim.inside.fit import fit  # tests may import inside code; the script under test may not

    cfg = json.loads((REPO / "snap" / "inside" / "config" / "graph_aim_v1.json").read_text())
    cfg.update(D=8, homophily_attrs=None, homophily_groups=None, noise=True, epsilon_edge=4.0, epsilon_attr=4.0)
    return fit(_toy_graph(), cfg, seed=0)


def _run_stage2(tmp_path: Path, art: Artifact, method: str, tag: str) -> tuple[AttributedGraph, list]:
    from graph_aim.transfer.artifact import save
    from graph_aim.transfer.graph_io import load_attributed_graph

    artifacts = tmp_path / f"artifacts_{tag}"
    save(art, artifacts / "toy_seed0")
    (artifacts / "index.json").write_text(json.dumps([{"variant": "toy", "epsilon": 4.0, "seed": 0, "dir": "toy_seed0"}]))
    out = tmp_path / f"synthetic_{tag}"
    proc = subprocess.run(
        [sys.executable, str(REPO / "snap" / "outside" / "generate_synthetic.py"),
         "--artifacts", str(artifacts), "--out", str(out), "--method", method],
        cwd=tmp_path, env={"PYTHONPATH": str(REPO / "src"), "PATH": "", "SYSTEMROOT": ""},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "toy_seed0.npz"]
    return load_attributed_graph(out / "toy_seed0.npz"), json.loads((out / "index.json").read_text())


def test_stage2_method_flag_without_the_kind_is_a_no_op(tmp_path):
    art = _fitted_toy_artifact()
    g1, idx1 = _run_stage2(tmp_path, art, "v1", "v1")
    g2, idx2 = _run_stage2(tmp_path, art, "v2", "v2")
    assert np.array_equal(g1.edges, g2.edges) and g1.attrs.equals(g2.attrs)
    assert idx1[0]["method"] == "v1" and idx2[0]["method"] == "v2"
    assert g1.n == 150 and len(g1.edges) > 0


def test_stage2_runs_v2_from_a_saved_artifact_with_the_kind(tmp_path):
    """Boundary-style run of ``--method v2`` on an artifact carrying ``attr_neigh_simpson``."""
    from graph_aim.transfer.artifact import validate

    toy = _toy_graph()
    art = _fitted_toy_artifact()
    m = simpson_measurement(toy, ["a"], sigma=1.0, edges=[0, 4, 8, 10**6], dmin=4)
    m.params["cap"] = 8
    art.measurements[m.name] = m
    art.ledger.entries.append({"name": m.name, "kind": m.kind, "sigma": 1.0, "rho_edge": 0.0, "rho_attr": 0.0})
    problems = validate(art)
    assert not any("unknown kind" in p for p in problems), problems
    g, idx = _run_stage2(tmp_path, art, "v2", "v2kind")
    assert g.n == 150 and len(g.edges) > 0 and idx[0]["method"] == "v2"
    assert list(g.attrs.columns) == ["a", "b"]
    ref = generate(art, seed=0)
    assert np.array_equal(g.edges, ref.edges)
    assert np.array_equal(np.bincount(g.attrs["a"], minlength=4), np.bincount(ref.attrs["a"], minlength=4))

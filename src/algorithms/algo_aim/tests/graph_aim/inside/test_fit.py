"""Tests for ``fit`` on a small synthetic graph."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graph_aim.inside import measure as M
from graph_aim.inside import privacy as P
from graph_aim.inside.capping import cap_edges
from graph_aim.inside.fit import fit, plan
from graph_aim.transfer.types import AttributedGraph

CONFIG_PATH = Path(__file__).resolve().parents[3] / "snap" / "inside" / "config" / "graph_aim_v1.json"
DOMAINS = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "none"], "c": ["u", "other", "none"]}
SQ2 = math.sqrt(2.0)


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(7)
    n = 120
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < 0.08
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n, edges, attrs, DOMAINS)


@pytest.fixture()
def config():
    """The shipped config scaled down to the toy graph (same structure, small public parameters)."""
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["D"] = 8
    cfg["max_degree"] = 30
    cfg["caps"] = {"clustering_by_degree": 8, "joint_degree": None, "attr_edge_pair": 6}
    cfg["hist"] = {"exact_upto": 8, "growth": 1.5, "threshold": 1.0, "spread_power": 1.0}
    cfg["cluster_bin_edges"] = [0, 2, 4, 8, 31]
    cfg["cluster_min_degree"] = 4
    cfg["joint_bin_edges"] = [0, 1, 3, 6, 31]
    cfg["homophily_attrs"] = None  # the toy graph has no "school" column
    cfg["homophily_groups"] = None  # ditto: the shipped groups name SNAP attributes
    cfg["attr_degree_bin_edges"] = None  # the shipped 4-bin edges go to 1201; use auto bins at max_degree 30
    return cfg


def test_config_shape():
    cfg = json.loads(CONFIG_PATH.read_text())
    assert cfg["D"] == 100 and cfg["delta"] == 1e-6
    assert set(cfg["shares"]) == set(M.KINDS)
    assert cfg["max_degree"] > cfg["D"] and set(cfg["caps"]) == {"clustering_by_degree", "joint_degree", "attr_edge_pair"}
    assert cfg["cluster_min_degree"] in cfg["cluster_bin_edges"]
    assert cfg["cluster_bin_edges"][-1] == cfg["joint_bin_edges"][-1] == cfg["max_degree"] + 1


def test_private_fit_ledger_and_budget(graph, config):
    art = fit(graph, config, seed=1)
    led = art.ledger
    assert led.rho_edge == pytest.approx(sum(e["rho_edge"] for e in led.entries))
    assert led.rho_attr == pytest.approx(sum(e["rho_attr"] for e in led.entries))
    assert {e["name"] for e in led.entries} == set(art.measurements)
    # the whole budget is spent (water-filling hands slack back) and never exceeded
    assert 3.999 < art.provenance["epsilon_edge"] <= 4.0 + 1e-9
    assert 3.999 < art.provenance["epsilon_attr"] <= 4.0 + 1e-9
    assert any("top-K" in c for c in led.caveats)
    assert any("looking at" in c and "shares" in c for c in led.caveats)
    assert art.schema.degree_cap == 8 and art.schema.n == graph.n
    assert art.schema.homophily_attrs == ["a", "b"]
    for m in art.measurements.values():
        assert np.all(m.values >= 0) and m.sigma > 0
        assert "cap" in m.params
    assert art.measurements["degree_hist"].values.shape == (31,)
    assert art.measurements["degree_hist"].params["max_degree"] == 30
    # the clustering node counts are post-processing of the released degree histogram, not a separate release
    cl, hist = art.measurements["clustering_by_degree"], art.measurements["degree_hist"].values
    assert cl.params["counts_from_hist"] is True
    be = cl.params["bin_edges"]
    for b in range(len(be) - 1):
        assert cl.values[b, 1] == pytest.approx(hist[be[b] : be[b + 1]].sum())


def test_per_measurement_caps_recorded(graph, config):
    ms = fit(graph, config, seed=1).measurements
    assert ms["degree_hist"].params["cap"] is None
    assert ms["clustering_by_degree"].params["cap"] == 8
    assert ms["joint_degree"].params["cap"] is None
    assert ms["attr_edge_pair"].params["cap"] == 6
    assert ms["attr_1way:a"].params["cap"] is None
    # sensitivities follow the per-measurement caps
    assert ms["attr_edge_pair"].sensitivity_attr == pytest.approx(SQ2 * 6)
    assert ms["degree_hist"].sensitivity_edge == pytest.approx(2 * SQ2)
    assert ms["joint_degree"].sensitivity_edge == pytest.approx(2 * SQ2 * 5 + 1)  # largest internal edge 6, uncapped


def test_non_private_is_exact(graph, config):
    config["noise"] = False
    art = fit(graph, config, seed=1)
    assert art.ledger.rho_edge == math.inf
    deg_all = np.bincount(graph.edges.ravel(), minlength=graph.n)
    dh = art.measurements["degree_hist"]
    assert np.array_equal(dh.values, np.bincount(np.minimum(deg_all, 30), minlength=31))  # uncapped, full per-degree
    assert dh.values.sum() == graph.n
    assert (np.arange(31) * dh.values).sum() / 2 == len(graph.edges)  # edge count recoverable
    one = art.measurements["attr_1way:a"]
    assert np.array_equal(one.values, np.bincount(graph.attrs["a"], minlength=4))
    assert one.sigma == 0.0
    capped6 = cap_edges(graph.edges, graph.n, 6, config["salt"])
    ep = art.measurements["attr_edge_pair"]
    assert ep.values.shape == (12, 12) and ep.values[np.triu_indices(12)].sum() == len(capped6)
    jd = art.measurements["joint_degree"]  # uncapped
    assert np.allclose(jd.values, jd.values.T) and jd.values[np.triu_indices(len(jd.values))].sum() == len(graph.edges)
    cl = art.measurements["clustering_by_degree"]
    assert cl.values.shape[1] == 2 and len(cl.params["bin_edges"]) == len(cl.values) + 1
    assert cl.params["min_degree"] is None  # exact release: nothing is dropped
    assert cl.values[:, 1].sum() == graph.n


def test_no_cap_reference_mode(graph, config):
    """D at or above max_degree: per-kind caps are ignored and the caps are no-ops (reference artifact)."""
    config["D"] = 5000
    config["noise"] = False
    art = fit(graph, config, seed=1)
    for name in ("clustering_by_degree", "attr_edge_pair"):
        assert art.measurements[name].params["cap"] == 5000
    assert art.measurements["attr_edge_pair"].values[np.triu_indices(12)].sum() == len(graph.edges)
    deg = np.bincount(graph.edges.ravel(), minlength=graph.n)
    assert np.array_equal(art.measurements["degree_hist"].values, np.bincount(deg, minlength=31))


def test_same_seed_reproduces_and_different_seed_differs(graph, config):
    a, b, c = fit(graph, config, 5), fit(graph, config, 5), fit(graph, config, 6)
    for k in a.measurements:
        assert np.array_equal(a.measurements[k].values, b.measurements[k].values)
    assert any(not np.array_equal(a.measurements[k].values, c.measurements[k].values) for k in a.measurements)
    assert a.ledger.entries == b.ledger.entries


def test_epsilon_monotone_sigma(graph, config):
    config["epsilon_edge"] = config["epsilon_attr"] = 1.0
    lo = fit(graph, config, 0)
    config["epsilon_edge"] = config["epsilon_attr"] = 8.0
    hi = fit(graph, config, 0)
    assert lo.measurements["degree_hist"].sigma > hi.measurements["degree_hist"].sigma


# --------------------------------------------------------------- improvement over the round-1 configuration


def _old_sensitivities(D=100):
    """Round-1 proved bounds (cap D on everything, degree_hist on the capped graph)."""
    r8 = 2 * SQ2
    return {
        "degree_hist": (3 * r8, 0.0),
        "clustering_by_degree": (3 * math.hypot(r8 + (D - 1), r8), 0.0),
        "joint_degree": (3 * (r8 * (D - 1) + 1), 0.0),
        "attr_1way": (0.0, SQ2),
        "attr_2way": (0.0, SQ2),
        "attr_edge_pair": (3.0, SQ2 * D),
    }


def test_shipped_config_lowers_noise_versus_round_1():
    """Analytic: at the same (eps, delta) per adjacency the shipped configuration needs much less noise on the
    structural measurements than the round-1 configuration (same budget, proved sensitivities)."""
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["shares"] = {k: v for k, v in cfg["shares"].items() if k != "attr_degree"}  # compare the round-1 workload only
    cfg["homophily_groups"] = None  # round-1 workload: the single group of homophily_attrs (["school"])
    domains = {"school": ["s1", "s2", "s3", "s4", "other", "none"], "gender": ["m", "f", "other", "none"]}
    attrs = pd.DataFrame({c: np.zeros(10, dtype=np.int64) for c in domains})
    g = AttributedGraph(10, np.zeros((0, 2), dtype=np.int64), attrs, domains)
    specs, _ = plan(g, cfg)
    new_sens = {s.name: M.sensitivity(s) for s in specs}
    total = {adj: P.eps_to_rho(4.0, 1e-6) for adj in ("edge", "attr")}
    new_sig = P.calibrate([(s.name, s.kind, *new_sens[s.name]) for s in specs], total, cfg["shares"])
    old = _old_sensitivities()
    old_shares = {"degree_hist": 0.25, "clustering_by_degree": 0.15, "joint_degree": 0.2, "attr_1way": 0.1, "attr_2way": 0.3, "attr_edge_pair": 0.25}
    old_sig = P.calibrate([(s.name, s.kind, *old[s.kind]) for s in specs], total, old_shares)
    # proved sensitivities
    assert old["clustering_by_degree"][0] / new_sens["clustering_by_degree"][0] > 10
    assert old["joint_degree"][0] / new_sens["joint_degree"][0] > 7
    assert old["degree_hist"][0] / new_sens["degree_hist"][0] == pytest.approx(3.0)
    # noise standard deviation on the same budget (signals are of the same size or larger in the new design)
    assert old_sig["degree_hist"] / new_sig["degree_hist"] > 2.5
    assert old_sig["clustering_by_degree"] / new_sig["clustering_by_degree"] > 10
    assert old_sig["joint_degree"] / new_sig["joint_degree"] > 4
    # the attribute-pair edge measurement keeps a (larger) cap for fidelity: allowed to cost a bit more noise
    assert new_sig["attr_edge_pair"] / old_sig["attr_edge_pair"] < 1.6
    # both adjacencies are fully spent, none exceeded
    for adj, idx in (("edge", 0), ("attr", 1)):
        spent = sum(P.zcdp_cost(new_sens[s.name][idx], new_sig[s.name]) for s in specs)
        assert spent == pytest.approx(total[adj], rel=1e-6)


def test_fitted_artifact_provenance_is_allowlisted(graph, config):
    from graph_aim.transfer.artifact import PROVENANCE_ALLOWED, validate

    for noise in (True, False):
        art = fit(graph, dict(config, noise=noise), seed=3)
        assert set(art.provenance) <= PROVENANCE_ALLOWED
        assert [e for e in validate(art) if e.startswith("provenance")] == []


# --------------------------------------------------------------- attr_degree


def test_attr_degree_generated_per_attribute(graph, config):
    art = fit(graph, config, seed=1)
    names = [k for k in art.measurements if k.startswith("attr_degree:")]
    assert names == [f"attr_degree:{c}" for c in DOMAINS]
    edges = M.degree_bins(30, 6, 31)
    entries = {e["name"]: e for e in art.ledger.entries}
    for c in DOMAINS:
        m = art.measurements[f"attr_degree:{c}"]
        assert m.values.shape == (len(DOMAINS[c]), len(edges) - 1) and np.all(m.values >= 0)
        assert m.values.sum() == pytest.approx(graph.n)
        assert m.params == {"attrs": [c], "cap": None, "bin_edges": edges}
        assert (m.sensitivity_edge, m.sensitivity_attr) == (pytest.approx(2 * SQ2), pytest.approx(SQ2))
        assert entries[m.name]["kind"] == "attr_degree" and entries[m.name]["rho_edge"] > 0 and entries[m.name]["rho_attr"] > 0
    # the budget is still respected: both ledgers within the configured total, and fully spent
    rho = P.eps_to_rho(4.0, 1e-6)
    assert art.ledger.rho_edge <= rho * (1 + 1e-9) and art.ledger.rho_attr <= rho * (1 + 1e-9)
    assert art.ledger.rho_edge == pytest.approx(rho, rel=1e-3)
    assert any("attribute-degree" in c for c in art.ledger.caveats)


def test_attr_degree_config_options(graph, config):
    config["attr_degree_bin_edges"] = [0, 2, 5]
    config["attr_degree_attrs"] = ["b"]
    ms = fit(graph, config, seed=1).measurements
    assert [k for k in ms if k.startswith("attr_degree:")] == ["attr_degree:b"]
    assert ms["attr_degree:b"].params["bin_edges"] == [0, 2, 5] and ms["attr_degree:b"].values.shape == (3, 2)
    config["attr_degree_attrs"] = ["nope"]
    with pytest.raises(ValueError, match="not in domains"):
        fit(graph, config, seed=1)


@pytest.mark.parametrize("share", ["absent", 0.0])
def test_attr_degree_absent_without_share(graph, config, share):
    if share == "absent":
        del config["shares"]["attr_degree"]
    else:
        config["shares"]["attr_degree"] = share
    art = fit(graph, config, seed=1)
    assert not any(k.startswith("attr_degree") for k in art.measurements)
    assert not any(s.kind == "attr_degree" for s in plan(graph, config)[0])


def test_attr_degree_missing_share_for_generated_kind_raises():
    specs = [("attr_degree:a", "attr_degree", 2 * SQ2, SQ2)]
    with pytest.raises(ValueError, match="no budget share"):
        P.calibrate(specs, {"edge": 1.0, "attr": 1.0}, {"degree_hist": 1.0})


# --------------------------------------------------------------- homophily groups (several attr_edge_pair)


def test_homophily_groups_specs_ledger_and_split(graph, config):
    config["homophily_groups"] = [["a"], ["b"], ["a", "c"]]
    art = fit(graph, config, seed=1)
    names = [k for k, m in art.measurements.items() if m.kind == "attr_edge_pair"]
    assert names == ["attr_edge_pair:a", "attr_edge_pair:b", "attr_edge_pair:a+c"]
    assert "attr_edge_pair" not in art.measurements  # the bare name is the legacy single-group spelling
    assert art.schema.homophily_attrs == ["a"]  # the primary group, kept for compatibility
    sizes = {"attr_edge_pair:a": 4, "attr_edge_pair:b": 3, "attr_edge_pair:a+c": 12}
    for name in names:
        m = art.measurements[name]
        assert m.values.shape == (sizes[name], sizes[name]) and np.allclose(m.values, m.values.T)
        assert m.params["cap"] == 6  # the kind's cap, shared by every instance
        assert (m.sensitivity_edge, m.sensitivity_attr) == (3.0, pytest.approx(SQ2 * 6))
    # one ledger entry per instance, and the kind's share is split equally: identical sensitivities
    # give identical sigma and rho across the instances
    entries = {e["name"]: e for e in art.ledger.entries if e["kind"] == "attr_edge_pair"}
    assert set(entries) == set(names)
    sig = {n: art.measurements[n].sigma for n in names}
    assert len({round(s, 12) for s in sig.values()}) == 1
    assert len({round(entries[n]["rho_attr"], 15) for n in names}) == 1
    # the split makes each instance noisier than a lone one on the same budget
    config_one = dict(config, homophily_groups=[["a"]])
    one = fit(graph, config_one, seed=1)
    assert list(m.name for m in one.measurements.values() if m.kind == "attr_edge_pair") == ["attr_edge_pair:a"]
    assert sig["attr_edge_pair:a"] > one.measurements["attr_edge_pair:a"].sigma
    # both ledgers still spend within the configured total
    rho = P.eps_to_rho(4.0, 1e-6)
    assert art.ledger.rho_edge <= rho * (1 + 1e-9) and art.ledger.rho_attr <= rho * (1 + 1e-9)
    from graph_aim.transfer.artifact import validate

    # the toy domains lack an "other" label (schema-level nit); the instances themselves are valid
    assert [e for e in validate(art) if "attr_edge_pair" in e] == []
    assert any("homophily" in c and "max_homophily_groups" in c for c in art.ledger.caveats)


def test_homophily_alias_unchanged_and_groups_take_precedence(graph, config):
    """homophily_attrs stays the single-group alias with the legacy bare name; homophily_groups wins."""
    alias = plan(graph, dict(config, homophily_attrs=["b"]))
    assert [s.name for s in alias[0] if s.kind == "attr_edge_pair"] == ["attr_edge_pair"]
    assert alias[1] == ["b"]
    both = plan(graph, dict(config, homophily_attrs=["b"], homophily_groups=[["c"]]))
    assert [s.name for s in both[0] if s.kind == "attr_edge_pair"] == ["attr_edge_pair:c"]
    assert both[1] == ["c"]
    none = plan(graph, dict(config, homophily_groups=[]))
    assert [s for s in none[0] if s.kind == "attr_edge_pair"] == [] and none[1] == []


def test_homophily_groups_limits(graph, config):
    config["homophily_groups"] = [["a"], ["b"], ["c"], ["a", "b"]]
    with pytest.raises(ValueError, match="max_homophily_groups"):
        plan(graph, config)
    config["max_homophily_groups"] = 4  # the limit is a config key
    assert sum(s.kind == "attr_edge_pair" for s in plan(graph, config)[0]) == 4
    with pytest.raises(ValueError, match="duplicate"):
        plan(graph, dict(config, homophily_groups=[["a"], ["a"]]))
    with pytest.raises(ValueError, match="empty"):
        plan(graph, dict(config, homophily_groups=[[]]))
    with pytest.raises(ValueError, match="not in domains"):
        plan(graph, dict(config, homophily_groups=[["nope"]]))


def test_homophily_group_cell_limit(config):
    domains = {"u": [f"u{i}" for i in range(9)], "v": [f"v{i}" for i in range(8)]}
    attrs = pd.DataFrame({c: np.zeros(20, dtype=np.int64) for c in domains})
    g = AttributedGraph(20, np.zeros((0, 2), dtype=np.int64), attrs, domains)
    config["homophily_groups"] = [["u", "v"]]  # 72 cells > MAX_GROUPS = 64
    with pytest.raises(ValueError, match="too many attribute groups"):
        plan(g, config)
    config["homophily_groups"] = [["u"], ["v"]]  # each group alone is fine
    assert sum(s.kind == "attr_edge_pair" for s in plan(g, config)[0]) == 2


# --------------------------------------------------------------- v1 pin (composition round, v2, is additive)


def _values_digest(art) -> str:
    h = hashlib.sha256()
    for m in art.measurements.values():  # measurements are kept in workload order
        h.update(np.ascontiguousarray(m.values, dtype=np.float64).tobytes())
    return h.hexdigest()


# Taken on the unchanged code before the v2 round (2026-11-03, commit 658420a): the shipped v1 config
# (noise on, eps_edge = eps_attr = 4, seed 0) on the boundary test's toy graph. Every measurement's values,
# concatenated in workload order. The v2 kinds are appended after every v1 kind and emitted only when the config
# gives them a share, so this digest must never change.
V1_PIN_VALUES_SHA256 = "dcbb55112db80b0ef5e89a3da76371b36bfd36b473f7a4dd5801543d587127bb"
V1_PIN_META_SHA256 = "b40d43d2772bb5c7b0944678a07f3590f64a208f52221a0d3ccd5f34d1ad7079"


def toy_graph() -> AttributedGraph:
    """Copy of ``tests/graph_aim/test_boundary.py::toy_graph`` (that module is not importable from here)."""
    rng = np.random.default_rng(11)
    n = 150
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < 0.06
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    domains = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in domains.items()})
    return AttributedGraph(n, edges, attrs, domains)


def test_v1_config_is_byte_identical_to_pinned_digest():
    """The shipped v1 config fitted on the deterministic toy graph gives exactly the pinned values and metadata."""
    cfg = json.loads(CONFIG_PATH.read_text())
    # the shipped homophily groups name SNAP attributes the toy graph lacks: the single-group alias is dropped too
    cfg.update(homophily_attrs=None, homophily_groups=None, noise=True, epsilon_edge=4.0, epsilon_attr=4.0)
    art = fit(toy_graph(), cfg, seed=0)
    assert list(art.measurements) == [
        "degree_hist", "clustering_by_degree", "joint_degree", "attr_1way:a", "attr_1way:b", "attr_2way:a|b",
        "attr_edge_pair", "attr_degree:a", "attr_degree:b",
    ]
    assert _values_digest(art) == V1_PIN_VALUES_SHA256
    meta = [[n, m.kind, m.sigma, m.sensitivity_edge, m.sensitivity_attr, list(m.values.shape)] for n, m in art.measurements.items()]
    assert hashlib.sha256(json.dumps(meta).encode()).hexdigest() == V1_PIN_META_SHA256


def test_attr_degree_non_private_is_exact(graph, config):
    config["noise"] = False
    ms = fit(graph, config, seed=1).measurements
    deg = np.bincount(graph.edges.ravel(), minlength=graph.n)
    for c in DOMAINS:
        m = ms[f"attr_degree:{c}"]
        assert m.sigma == 0.0
        e = m.params["bin_edges"]
        expected = np.zeros(m.values.shape)
        for v, d in zip(graph.attrs[c], deg):
            expected[v, min(np.searchsorted(e, d, side="right") - 1, len(e) - 2)] += 1
        assert np.array_equal(m.values, expected)

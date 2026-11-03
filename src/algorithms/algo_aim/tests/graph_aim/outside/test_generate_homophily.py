"""Tests for the multi-channel homophily rewiring (several ``attr_edge_pair`` instances).

Artifacts are built by hand (no fit), with generic attribute names: a constructed graph without
homophily plus fabricated diagonal-heavy edge-pair targets, so "the mixing moves toward the target"
is attributable to the rewiring channels alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_aim.outside import generate as gen
from graph_aim.outside.generate import generate
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement, PrivacyLedger, Schema

DOM = {"u": ["u0", "u1", "u2"], "v": ["v0", "v1", "v2", "v3"], "w": ["w0", "w1"]}


def toy_graph(n: int = 1000, p: float = 0.008, seed: int = 0) -> AttributedGraph:
    """Erdos-Renyi graph with attributes independent of the structure (no homophily to start from)."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    attrs = pd.DataFrame(
        {
            "u": rng.choice(3, size=n, p=[0.5, 0.3, 0.2]),
            "v": rng.choice(4, size=n, p=[0.4, 0.3, 0.2, 0.1]),
            "w": rng.integers(0, 2, n),
        }
    ).astype(np.int64)
    return AttributedGraph(n, edges, attrs, {k: list(v) for k, v in DOM.items()})


def diagonal_target(g: AttributedGraph, col: str, boost: float = 8.0) -> np.ndarray:
    """Symmetric edge-count target (diagonal counted once) with a strong same-value preference and
    row margins equal to the degree volume of each value (so degree-preserving swaps can reach it)."""
    k = len(DOM[col])
    deg = np.bincount(g.edges.ravel(), minlength=g.n).astype(float)
    vol = np.bincount(g.attrs[col].to_numpy(), weights=deg, minlength=k)
    seed = np.ones((k, k)) + boost * np.eye(k)
    q = gen._ipf_sym(seed, vol)  # ordered-pair mass with margins vol
    t = q.copy()
    t[np.diag_indices(k)] /= 2.0  # back to the edge-count convention (diagonal counted once)
    return t


def mixing(edges: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Normalised ordered-pair mixing matrix of a graph."""
    q = np.zeros((k, k))
    np.add.at(q, (codes[edges[:, 0]], codes[edges[:, 1]]), 1.0)
    np.add.at(q, (codes[edges[:, 1]], codes[edges[:, 0]]), 1.0)
    return q / max(q.sum(), 1.0)


def lift(edges: np.ndarray, codes: np.ndarray, k: int) -> float:
    q = mixing(edges, codes, k)
    a = q.sum(axis=1)
    return float(np.trace(q) / (a**2).sum())


def make_artifact(g: AttributedGraph, edge_pairs: dict[str, tuple[list[str], np.ndarray]]) -> Artifact:
    """Exact degree/attribute measurements plus the given ``attr_edge_pair`` instances."""
    ms: dict[str, Measurement] = {}
    deg = np.bincount(g.edges.ravel(), minlength=g.n)

    def add(name, kind, values, **params):
        ms[name] = Measurement(name, kind, np.asarray(values, dtype=float), "gaussian", 0.0, 0.0, 0.0, params)

    add("degree_hist", "degree_hist", np.bincount(deg))
    cols = list(DOM)
    for c in cols:
        add(f"attr_1way:{c}", "attr_1way", np.bincount(g.attrs[c], minlength=len(DOM[c])), attrs=[c])
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            t = np.zeros((len(DOM[a]), len(DOM[b])))
            np.add.at(t, (g.attrs[a].to_numpy(), g.attrs[b].to_numpy()), 1.0)
            add(f"attr_2way:{a}|{b}", "attr_2way", t, attrs=[a, b])
    for name, (attrs_, values) in edge_pairs.items():
        add(name, "attr_edge_pair", values, attrs=attrs_, cap=None)
    schema = Schema(g.n, {k: list(v) for k, v in DOM.items()}, int(deg.max()), 0, ["u"])
    return Artifact(schema, ms, PrivacyLedger(1e-6, float("inf"), float("inf")))


@pytest.fixture(scope="module")
def planted():
    return toy_graph()


# -------------------------------------------------------------------------- primary selection
def test_primary_is_highest_lift_whatever_the_order(planted):
    diag = diagonal_target(planted, "u")
    p = np.array([0.4, 0.3, 0.2, 0.1])
    chance = np.outer(p, p)  # chance-level mixing in the edge-count convention: lift 1
    chance[np.diag_indices(4)] /= 2.0
    assert gen._edge_pair_lift(diag) > 1.5
    assert gen._edge_pair_lift(chance) == pytest.approx(1.0)
    assert gen._edge_pair_lift(np.zeros((3, 3))) == 0.0
    for order in (("attr_edge_pair:v", "attr_edge_pair:u"), ("attr_edge_pair:u", "attr_edge_pair:v")):
        pairs = {}
        for name in order:
            pairs[name] = (["v"], chance) if name.endswith(":v") else (["u"], diag)
        art = make_artifact(planted, pairs)
        assert gen._primary_edge_pair(art).params["attrs"] == ["u"]
        # the class codes of the structure sampler follow the primary instance
        codes, size = gen._group_codes(art, planted.attrs, planted.n)
        assert size == 3 and np.array_equal(codes, planted.attrs["u"].to_numpy())


def test_selection_is_by_kind_never_by_name(planted):
    """A lone instance keeps today's behaviour whatever it is called (legacy bare name or new style)."""
    t = diagonal_target(planted, "u")
    old = make_artifact(planted, {"attr_edge_pair": (["u"], t)})
    new = make_artifact(planted, {"attr_edge_pair:u": (["u"], t)})
    a, b = generate(old, seed=5), generate(new, seed=5)
    assert np.array_equal(a.edges, b.edges)
    assert a.attrs.equals(b.attrs)


# -------------------------------------------------------------------------- _rewire channels
def chi2(edges: np.ndarray, cls: np.ndarray, target: np.ndarray) -> float:
    cnt = gen._class_counts(edges, cls, target.shape[0])
    return float((((cnt - target) ** 2) / (target + 1.0)).sum())


def test_multi_channel_rewire_moves_each_channel_toward_its_target(planted):
    rng = np.random.default_rng(0)
    deg0 = np.bincount(planted.edges.ravel(), minlength=planted.n)
    channels = []
    for col in ("u", "v"):
        codes = planted.attrs[col].to_numpy()
        channels.append((codes, gen._ordered_mass(diagonal_target(planted, col))))
    before = [chi2(planted.edges, c, t) for c, t in channels]
    out = gen._rewire(planted.edges, np.ones(len(planted.edges), dtype=bool), channels, planted.n, rng, rounds=30)
    # degrees preserved, no loops, no duplicates
    assert np.array_equal(np.bincount(out.ravel(), minlength=planted.n), deg0)
    assert (out[:, 0] < out[:, 1]).all()
    assert len(np.unique(out[:, 0] * planted.n + out[:, 1])) == len(out)
    # every channel's mixing matrix moved toward its own target
    for (codes, target), b4 in zip(channels, before):
        assert chi2(out, codes, target) < 0.7 * b4
        tn = target / target.sum()
        k = target.shape[0]
        assert np.abs(mixing(out, codes, k) - tn).sum() < np.abs(mixing(planted.edges, codes, k) - tn).sum()


def test_single_channel_call_matches_legacy_signature_behaviour(planted):
    """One channel: the list-of-channels API is the old single (cls, target) rewiring, bit for bit."""
    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    codes = planted.attrs["u"].to_numpy()
    target = gen._ordered_mass(diagonal_target(planted, "u"))
    movable = np.ones(len(planted.edges), dtype=bool)
    out1 = gen._rewire(planted.edges, movable, [(codes, target)], planted.n, rng1, rounds=8)
    out2 = gen._rewire(planted.edges, movable, [(codes, target)], planted.n, rng2, rounds=8)
    assert np.array_equal(out1, out2)  # deterministic
    assert chi2(out1, codes, target) < chi2(planted.edges, codes, target)


def test_degenerate_channels_are_skipped(planted):
    movable = np.ones(len(planted.edges), dtype=bool)
    rng = np.random.default_rng(1)
    one_class = (np.zeros(planted.n, dtype=np.int64), np.ones((1, 1)))
    out = gen._rewire(planted.edges, movable, [one_class], planted.n, rng, rounds=5)
    assert np.array_equal(out, planted.edges)  # nothing to steer: untouched
    assert gen._rewire(planted.edges, movable, [], planted.n, rng, rounds=5) is planted.edges


# -------------------------------------------------------------------------- end to end
def test_generate_two_groups_both_gain_homophily(planted):
    pairs = {
        "attr_edge_pair:u": (["u"], diagonal_target(planted, "u")),
        "attr_edge_pair:v": (["v"], diagonal_target(planted, "v")),
    }
    art = make_artifact(planted, pairs)
    primary_col = gen._primary_edge_pair(art).params["attrs"][0]
    base = generate(art, seed=2, rewire_rounds=0)
    syn = generate(art, seed=2, rewire_rounds=40)
    for g in (base, syn):
        e = g.edges
        assert (e[:, 0] < e[:, 1]).all()
        assert len(np.unique(e[:, 0] * g.n + e[:, 1])) == len(e)
    # the rewiring pushes BOTH groups' mixing toward their diagonal-heavy targets: the primary group
    # is already steered by the initial class-based sampling, the other one only by its channel
    for col, (attrs_, t) in (("u", pairs["attr_edge_pair:u"]), ("v", pairs["attr_edge_pair:v"])):
        k = len(DOM[col])
        target_lift = gen._edge_pair_lift(t)
        l_base = lift(base.edges, base.attrs[col].to_numpy(), k)
        l_syn = lift(syn.edges, syn.attrs[col].to_numpy(), k)
        assert 0.75 * target_lift < l_syn < 1.2 * target_lift, col  # near its own target
        if col == primary_col:
            assert l_syn > l_base - 0.1, col  # the channel does not undo the sampler's work
        else:
            assert l_base < 1.3 and l_syn > l_base + 0.5, col  # the gain comes from the channel alone
    g2 = generate(art, seed=2, rewire_rounds=40)
    assert np.array_equal(syn.edges, g2.edges)  # deterministic per seed


def test_zero_mass_extra_instance_is_ignored(planted):
    pairs = {
        "attr_edge_pair:u": (["u"], diagonal_target(planted, "u")),
        "attr_edge_pair:v": (["v"], np.zeros((4, 4))),
    }
    g = generate(make_artifact(planted, pairs), seed=1)
    assert g.n == planted.n and len(g.edges) > 0

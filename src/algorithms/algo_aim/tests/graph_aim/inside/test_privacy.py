"""Tests for the zCDP accountant, calibration and ledger."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graph_aim.inside import privacy as P
from graph_aim.transfer.types import Measurement


def test_rho_eps_roundtrip():
    for eps in (0.5, 1, 2, 4, 8):
        rho = P.eps_to_rho(eps, 1e-6)
        assert P.rho_to_eps(rho, 1e-6) == pytest.approx(eps)
    assert P.rho_to_eps(math.inf, 1e-6) == math.inf
    assert P.rho_to_eps(0.0, 1e-6) == 0.0


def test_cost_and_sigma_inverse():
    assert P.zcdp_cost(2.0, 4.0) == pytest.approx(4 / 32)
    assert P.sigma_for(2.0, P.zcdp_cost(2.0, 4.0)) == pytest.approx(4.0)
    assert P.zcdp_cost(0.0, 0.0) == 0.0
    assert P.zcdp_cost(1.0, 0.0) == math.inf
    assert P.sigma_for(1.0, math.inf) == 0.0


def test_noise_seeded_and_scale():
    a = P.gaussian_noise(np.random.default_rng(3), 100000, 2.0)
    b = P.gaussian_noise(np.random.default_rng(3), 100000, 2.0)
    assert np.array_equal(a, b)
    assert a.std() == pytest.approx(2.0, rel=0.02)
    assert not P.gaussian_noise(np.random.default_rng(0), (3, 2), 0.0).any()


SPECS = [
    ("h", "degree_hist", 8.0, 0.0),
    ("j", "joint_degree", 100.0, 0.0),
    ("a1", "attr_1way", 0.0, 1.4),
    ("a2", "attr_1way", 0.0, 1.4),
    ("e", "attr_edge_pair", 3.0, 140.0),
]
SHARES = {"degree_hist": 0.5, "joint_degree": 0.5, "attr_1way": 0.5, "attr_edge_pair": 0.5}


def test_calibrate_spends_within_totals():
    total = {"edge": 0.3, "attr": 0.2}
    sig = P.calibrate(SPECS, total, SHARES)
    spent = {"edge": 0.0, "attr": 0.0}
    for name, _, se, sa in SPECS:
        spent["edge"] += P.zcdp_cost(se, sig[name])
        spent["attr"] += P.zcdp_cost(sa, sig[name])
    assert spent["edge"] <= total["edge"] + 1e-12
    assert spent["attr"] <= total["attr"] + 1e-12
    # water-filling: attr_edge_pair is limited by the attr ledger (large attr sensitivity) and leaves most of
    # its edge-ledger allocation unspent; that slack is handed to h and j, so the edge ledger is used up
    assert spent["edge"] == pytest.approx(total["edge"], rel=1e-6)
    assert spent["attr"] == pytest.approx(total["attr"], rel=1e-6)
    assert P.zcdp_cost(3.0, sig["e"]) < 0.01
    # h and j have equal shares and the slack is split by shares, so they still spend equal rho
    assert P.zcdp_cost(8.0, sig["h"]) == pytest.approx(P.zcdp_cost(100.0, sig["j"]))
    assert P.zcdp_cost(8.0, sig["h"]) > 0.1  # more than the naive 1/3 share


def test_calibrate_waterfilling_never_overspends_and_beats_naive():
    """Random sensitivities: totals respected, and every sigma is <= the naive (no redistribution) value."""
    rng = np.random.default_rng(0)
    kinds = ["k1", "k2", "k3", "k4", "k5"]
    for _ in range(30):
        specs = [(f"m{i}", kinds[i % 5], float(rng.uniform(0, 5) * (rng.random() < 0.7)), float(rng.uniform(0, 5) * (rng.random() < 0.7))) for i in range(9)]
        specs = [sp if (sp[2] > 0 or sp[3] > 0) else (sp[0], sp[1], 1.0, 0.0) for sp in specs]
        shares = {k: float(rng.uniform(0.1, 1)) for k in kinds}
        total = {"edge": float(rng.uniform(0.05, 1)), "attr": float(rng.uniform(0.05, 1))}
        sig = P.calibrate(specs, total, shares)
        for adj, idx in (("edge", 2), ("attr", 3)):
            spent = sum(P.zcdp_cost(sp[idx], sig[sp[0]]) for sp in specs)
            assert spent <= total[adj] * (1 + 1e-9)
        # naive allocation: each adjacency's rho split by normalised kind shares, equally within the kind
        for sp in specs:
            naive = 0.0
            for adj, idx in (("edge", 2), ("attr", 3)):
                if sp[idx] > 0:
                    ks = {s[1] for s in specs if s[idx] > 0}
                    members = [s for s in specs if s[1] == sp[1] and s[idx] > 0]
                    rho = total[adj] * shares[sp[1]] / sum(shares[k] for k in ks) / len(members)
                    naive = max(naive, P.sigma_for(sp[idx], rho))
            assert sig[sp[0]] <= naive * (1 + 1e-9)


def test_calibrate_splits_kind_share_equally_among_instances():
    """Several attr_edge_pair instances share the kind's one budget share (package D)."""
    two = [
        ("e1", "attr_edge_pair", 3.0, 140.0),
        ("e2", "attr_edge_pair", 3.0, 140.0),
        ("a1", "attr_1way", 0.0, 1.4),
        ("h", "degree_hist", 8.0, 0.0),
    ]
    total = {"edge": 0.3, "attr": 0.2}
    shares = {"attr_edge_pair": 0.5, "attr_1way": 0.5, "degree_hist": 0.5}
    sig = P.calibrate(two, total, shares)
    assert sig["e1"] == sig["e2"]  # identical sensitivities, equal split: identical sigma
    for adj, idx in (("edge", 2), ("attr", 3)):
        spent = sum(P.zcdp_cost(sp[idx], sig[sp[0]]) for sp in two)
        assert spent <= total[adj] * (1 + 1e-9)
    # with the kind's rho halved per instance, each is noisier than a lone instance on the same budget
    one = P.calibrate([sp for sp in two if sp[0] != "e2"], total, shares)
    assert sig["e1"] > one["e1"]
    assert sig["e1"] == pytest.approx(one["e1"] * math.sqrt(2.0), rel=1e-6)  # sigma ~ 1/sqrt(rho)


def test_calibrate_no_noise_and_missing_share():
    assert set(P.calibrate(SPECS, {"edge": 1, "attr": 1}, SHARES, noise=False).values()) == {0.0}
    with pytest.raises(ValueError):
        P.calibrate(SPECS, {"edge": 1, "attr": 1}, {"degree_hist": 1.0})


def _m(name, kind, sig, se, sa):
    return Measurement(name, kind, np.zeros(1), "gaussian", sig, se, sa)


def test_ledger_totals_and_caveats():
    ms = [_m("h", "degree_hist", 4.0, 8.0, 0.0), _m("a", "attr_1way", 2.0, 0.0, 1.4)]
    led = P.build_ledger(ms, 1e-6)
    assert led.rho_edge == pytest.approx(sum(e["rho_edge"] for e in led.entries))
    assert led.rho_attr == pytest.approx(sum(e["rho_attr"] for e in led.entries))
    assert led.rho_edge == pytest.approx(64 / 32) and led.rho_attr == pytest.approx(1.96 / 8)
    text = " ".join(led.caveats).lower()
    assert "top-k" in text and "degree cap" in text and "node count n is public" in text


def test_ledger_non_private():
    led = P.build_ledger([_m("h", "degree_hist", 0.0, 8.0, 0.0)], 1e-6, noise=False)
    assert led.rho_edge == math.inf and led.rho_attr == 0.0
    assert P.ledger_epsilons(led)["epsilon_edge"] == math.inf

"""Weighted rho split + hierarchical place tier (synthetic records, no real data)."""
import math

import h3
import numpy as np
import pytest

from geolife.transfer import schema as S
from geolife.outside.fit import fit_behaviour
from geolife.inside.measure import (HierarchySpec, build_behaviour_measurements, rho_weight,
                                      weighted_rho_split)
from geolife.transfer.types import Stay, Trip, UserRecord

LAT, LON = 39.98, 116.30
H = 3600


def _cells():
    c8 = h3.latlng_to_cell(LAT, LON, 8)
    ring = sorted(h3.grid_disk(c8, 6))
    return c8, ring


def _records(n=120, seed=0):
    rng = np.random.default_rng(seed)
    c8, ring = _cells()
    work = f"h3:{ring[0]}"          # the same work cell for everybody -> coarse overlap
    recs = []
    for u in range(n):
        home = f"h3:{ring[1 + (u % 5)]}"
        t = 1_200_000_000 - (1_200_000_000 % 86400) + 5 * 86400    # a weekday 00:00 UTC
        stays = [Stay(home, "unknown", home[3:], t, t + 8 * H),               # night -> home
                 Stay(work, "unknown", work[3:], t + 10 * H, t + 16 * H)]
        trips = [Trip(home, work, t + 8 * H, t + 8 * H + 600, "car", (), 3000.0)]
        recs.append(UserRecord(f"u{u}", stays, trips))
    return recs


def _vocab():
    c8, ring = _cells()
    return [(f"h3:{c}", "unknown") for c in ring]


def test_weights_sum_to_target():
    ws = [rho_weight(1.0, 1.0, 1), rho_weight(math.sqrt(8), 1.0, 8514, 3.0),
          rho_weight(3 * math.sqrt(24), 3.0, 72), rho_weight(70.7, 50.0, 2)]
    for tot in (0.1, 0.5395, 3.0):
        sh = weighted_rho_split(tot, ws)
        assert sum(sh) == pytest.approx(tot, rel=1e-12)
        assert all(x >= 0.4 * tot / len(ws) for x in sh)   # equal-split floor


def test_rho_weight_uses_only_public_quantities_and_orders_sensibly():
    # more keys per user and a larger vocabulary need more rho; priority scales linearly
    assert rho_weight(math.sqrt(8), 1, 1000) > rho_weight(math.sqrt(2), 1, 1000)
    assert rho_weight(math.sqrt(8), 1, 1000) > rho_weight(math.sqrt(8), 1, 10)
    assert rho_weight(math.sqrt(8), 1, 1000, 3.0) == pytest.approx(3 * rho_weight(math.sqrt(8), 1, 1000))
    # the clip cancels: a clipped transition table weighs like a presence table with the same key budget
    assert rho_weight(50 * math.sqrt(2), 50, 2) == pytest.approx(rho_weight(math.sqrt(2), 1, 2))


@pytest.mark.parametrize("split", ["weighted", "equal"])
def test_ledger_total_matches_target_flat(split):
    rng = np.random.default_rng(1)
    ms, led = build_behaviour_measurements(_records(), _vocab(), 0.5, rng, split=split)
    assert led.rho_total == pytest.approx(0.5, rel=1e-6)
    assert S.PLACE_ROLE_USERS in {m.name for m in ms}


def test_weighted_differs_from_equal_and_cheap_tables_pay_less():
    recs, vocab = _records(), _vocab()
    mw, _ = build_behaviour_measurements(recs, vocab, 0.5, np.random.default_rng(1))
    me, _ = build_behaviour_measurements(recs, vocab, 0.5, np.random.default_rng(1), split="equal")
    rw = {m.name: m.rho for m in mw}
    re = {m.name: m.rho for m in me}
    assert rw[S.N_USERS] < re[S.N_USERS]
    assert rw[S.ARRIVAL_HOUR] > re[S.ARRIVAL_HOUR]


def test_hierarchy_release_refines_only_under_kept_parents_and_accounts_everything():
    recs, vocab = _records(), _vocab()
    hs = HierarchySpec(res=(6, 7, 8), tau=2.0)
    ms, led = build_behaviour_measurements(recs, vocab, 2.0, np.random.default_rng(2), hierarchy=hs)
    byname = {m.name: m for m in ms}
    for nm in (S.PLACE_R6, S.PLACE_R7, S.PLACE_ROLE_USERS, S.WORK_HIST):
        assert nm in byname
    assert {e["name"] for e in led.entries} == set(byname)
    assert led.rho_total <= 2.0 * (1 + 1e-6)
    # r6 is released over every (cell, role) in the public vocabulary
    cells6 = {h3.cell_to_parent(c[3:], 6) for c, _ in vocab}
    assert len(byname[S.PLACE_R6].keys) == len(cells6) * 3
    # every r7 key lies under an r6 parent whose NOISY value passed the threshold
    r6 = byname[S.PLACE_R6]
    kept6 = {(k[0], k[2]) for k, v in zip(r6.keys, r6.values) if v >= hs.tau * r6.sigma}
    for k in byname[S.PLACE_R7].keys:
        par = "h3:" + h3.cell_to_parent(k[0][3:], 6)
        assert (par, k[2]) in kept6
    r7 = byname[S.PLACE_R7]
    kept7 = {(k[0], k[2]) for k, v in zip(r7.keys, r7.values) if v >= hs.tau * r7.sigma}
    for k in byname[S.PLACE_ROLE_USERS].keys:
        par = "h3:" + h3.cell_to_parent(k[0][3:], 7)
        assert (par, k[2]) in kept7
    # shared-anchor histogram: sensitivity sqrt(2), 8 bins
    wh = byname[S.WORK_HIST]
    assert wh.l2_sensitivity == pytest.approx(math.sqrt(2)) and len(wh.keys) == 8


def test_fit_consumes_hierarchy_and_falls_back():
    recs, vocab = _records(), _vocab()
    ms, led = build_behaviour_measurements(recs, vocab, 20.0, np.random.default_rng(3),
                                           hierarchy=HierarchySpec())
    p = fit_behaviour(ms, led)
    assert "coarse" in p and p["coarse"]["tier"] == S.PLACE_R6
    assert max(p["coarse"]["popularity"]["work"]) > 50          # everybody works in one cell
    assert abs(sum(p["work_users_per_cell"]["p"]) - 1.0) < 1e-9
    assert set(p["place_tiers"]) == {S.PLACE_R6, S.PLACE_R7, S.PLACE_ROLE_USERS}
    # drop the refined tiers: still fits (coarse only)
    coarse_only = [m for m in ms if m.name not in (S.PLACE_R7, S.PLACE_ROLE_USERS)]
    led2 = led
    led2.entries = [e for e in led.entries if e["name"] in {m.name for m in coarse_only}]
    p2 = fit_behaviour(coarse_only, led2)
    assert p2["coarse"]["tier"] == S.PLACE_R6
    assert p2["hist_users_per_place"]["work"] == {}
    assert p2["zipf_s"] > 0

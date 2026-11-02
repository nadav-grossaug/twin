"""Generator v2: determinism, planted statistics vs params, structure of traces and truth (toy grid city),
and an OvertureProvider Amsterdam run when its cache exists locally (skipped otherwise)."""
import os
import time
from pathlib import Path

import numpy as np
import pytest

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline_v2 import generate_v2, planted_stats

GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 120, "sample_interval_s": 20,
       "compute_cells": False}


@pytest.fixture(scope="module")
def prov():
    return GridCityProvider(n=40, spacing=250, n_places=6000, seed=1)


@pytest.fixture(scope="module")
def gen60(prov):
    return generate_v2(fake_params_v2(), prov, 60, 1, gen=GEN)


def test_planted_statistics_match_params(gen60):
    stats = planted_stats(gen60)
    bad = {k: v for k, v in stats.items() if k != "info" and not v["ok"]}
    assert not bad, bad


def test_determinism(prov):
    gen = {**GEN, "max_outings": 20}
    a = generate_v2(fake_params_v2(), prov, 8, 5, gen=gen)
    b = generate_v2(fake_params_v2(), prov, 8, 5, gen=gen)
    c = generate_v2(fake_params_v2(), prov, 8, 6, gen=gen)
    assert a.result.traces == b.result.traces
    assert a.result.truth.visits == b.result.truth.visits
    assert a.result.truth.shared_places == b.result.truth.shared_places
    assert a.result.traces != c.result.traces


def test_outings_are_separated_and_visits_ordered(gen60):
    for uid, outs in gen60.result.outings.items():
        for o, nxt in zip(outs, outs[1:]):
            assert nxt.t_start - o.t_end >= 7200, (uid, o.t_end, nxt.t_start)
        for o in outs:
            kinds = [v.kind for v in o.visits]
            assert kinds[0] == "start" and kinds[-1] == "end" and set(kinds[1:-1]) <= {"stop"}
            ts = [v.t_arrive for v in o.visits]
            assert ts == sorted(ts) and o.t_end >= o.t_start
            assert all(v.dwell_s >= 600 for v in o.visits if v.kind == "stop")


def test_traces_are_bursty_and_inside_outings(gen60):
    res = gen60.result
    n_with_pts = 0
    for uid, pts in res.traces.items():
        if not pts:
            continue
        n_with_pts += 1
        t = np.array([p.t for p in pts])
        assert np.all(np.diff(t) > 0)
        outs = res.outings[uid]
        starts = np.array([o.t_start for o in outs]) - 120
        ends = np.array([o.t_end for o in outs]) + 180
        ix = np.searchsorted(starts, t, side="right") - 1
        assert np.all(ix >= 0) and np.all(t <= ends[ix])        # logging only while an outing is under way
        # long silences between outings: no overnight stays rendered
        assert np.max(np.diff(t)) > 3600
    assert n_with_pts >= 50


def test_sampling_regime_default(prov):
    g = {**GEN, "sample_interval_s": None, "max_outings": 15}
    r = generate_v2(fake_params_v2(), prov, 12, 2, gen=g)
    med = [np.median(np.diff([p.t for p in pts])) for pts in r.result.traces.values() if len(pts) > 50]
    assert set(med) <= {1.0, 2.0, 3.0, 4.0, 5.0, 13.0} or max(med) <= 14
    assert len(set(med)) >= 2          # per-user sampling periods differ


def test_truth_consistency(gen60):
    tr, res = gen60.result.truth, gen60.result
    assert tr.groups and sum(len(v) for v in tr.groups.values()) > 20
    for pid, us in tr.shared_places.items():
        assert len(us) >= 2
        assert all(any(v.place_id == pid for v in tr.visits[u]) for u in us)
    for sid, us in tr.shared_segments.items():
        assert len(us) >= 2
        assert all(any(sid in o.segments for o in res.outings[u]) for u in us)
    for uid, favs in tr.favourite_places.items():
        c = {}
        for v in tr.visits[uid]:
            c[v.place_id] = c.get(v.place_id, 0) + 1
        assert all(c[p] >= 3 for p in favs)
    prov_ids = {p.id for p in gen60.teleport.mapping and []} or set(gen60.teleport.mapping.values())
    assert all(v.place_id in prov_ids for vs in tr.visits.values() for v in vs)
    # revisits produce favourites for most users even though no home/work role exists
    assert np.mean([len(f) > 0 for u, f in tr.favourite_places.items() if len(tr.visits[u]) > 40]) > 0.8


def test_cotravel_follows_same_route(gen60):
    tr, res = gen60.result.truth, gen60.result
    assert tr.co_traveled_detail
    for lead, il, fol, ifol in tr.co_traveled_detail:
        a, b = res.outings[lead][il], res.outings[fol][ifol]
        assert a.t_start == b.t_start and a.segments == b.segments
        assert [v.place_id for v in a.visits] == [v.place_id for v in b.visits]
    assert {(a, b, i) for a, b, i in tr.co_traveled} == {(a, b, i) for a, i, b, _ in tr.co_traveled_detail}
    for lead, il, fol, ifol in tr.co_traveled_detail[:3]:
        assert any(set(tr.shared_segments).intersection(res.outings[lead][il].segments) for _ in [0]) or \
            not res.outings[lead][il].segments


def test_local_time_convention(gen60):
    """Outing starts peak in the local morning/evening (UTC+8), not in UTC hours."""
    t = np.array([o.t_start for os_ in gen60.result.outings.values() for o in os_])
    local_h = ((t + 8 * 3600) % 86400) // 3600
    share = np.bincount(local_h.astype(int), minlength=24) / len(t)
    assert share[[7, 8, 9]].sum() > share[[2, 3, 4]].sum() * 3
    assert share[[17, 18]].sum() > 0.08


def test_gen_overrides_change_sensor_regime(prov):
    base = {**GEN, "max_outings": 10}
    a = generate_v2(fake_params_v2(), prov, 6, 3, gen=base)
    b = generate_v2(fake_params_v2(), prov, 6, 3, gen={**base, "sample_interval_s": 60})
    na = sum(len(p) for p in a.result.traces.values())
    nb = sum(len(p) for p in b.result.traces.values())
    assert nb < na / 2
    assert a.result.truth.visits == b.result.truth.visits       # same planted world, different sensor


_LEGACY_AMS = Path("C:/Users/NaomiD/Documents/Projects/hackaton-2026/.claude/worktrees/agent-a63109956b0e67a7c/geolife/cache")
AMS_CACHE = next((c for c in (Path("geolife/outside/cache"), _LEGACY_AMS) if (c / "places_amsterdam.parquet").exists()),
                 Path("geolife/outside/cache"))
AMS_BBOX = (4.85, 52.33, 4.95, 52.41)


@pytest.mark.skipif(not (AMS_CACHE / "places_amsterdam.parquet").exists(), reason="Amsterdam cache not available")
def test_overture_amsterdam_quick_run():
    from geolife.outside.map.provider import OvertureProvider
    t0 = time.perf_counter()
    prov = OvertureProvider(AMS_BBOX, cache_dir=AMS_CACHE, max_poi=15000)
    t_prov = time.perf_counter() - t0
    r = generate_v2(fake_params_v2(), prov, 12, 3,
                    gen={**fake_generation_v2(), "max_outings": 40, "sample_interval_s": 10})
    print("provider_s", round(t_prov, 1), "timings", r.timings, r.result.meta)
    assert r.result.meta["n_outings"] > 100 and r.result.meta["n_points"] > 10000
    ids = {p.id for p in prov.places()}
    assert all(v.place_id in ids for vs in r.result.truth.visits.values() for v in vs)
    assert r.result.truth.shared_places
    assert any(o.cells for os_ in r.result.outings.values() for o in os_)
    st = planted_stats(r)
    for k in ("stop_revisit_rate", "start_tod_daytype_tv", "stops_per_outing_mean"):
        assert st[k]["ok"], (k, st[k])

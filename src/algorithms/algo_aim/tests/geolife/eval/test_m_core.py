"""Method M on small synthetic draws (fast; no real data)."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from geolife.eval.m import transfer as T
from geolife.eval.m.core import MapView, MConfig, UserPlace, communities, estimate_night_start, relation_graph, run_m
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.transfer.types import GpsPoint

N, SEED = 40, 5
CFG = MConfig(match_trips=False)


@lru_cache(maxsize=None)
def _setup():
    prov = T.make_provider()
    g = T.generate_data(N, SEED, {"n_days": 6}, prov)
    mv = T.map_view(prov, with_graph=False)
    xy = {p.id: (p.x, p.y) for p in prov.places()}
    return prov, g, mv, xy, run_m(g.result.traces, mv, CFG)


def test_recovers_main_places_and_roles():
    _, g, mv, xy, res = _setup()
    ev = T.evaluate(g, res, xy)
    mp = ev["main_places"]
    assert mp["precision"] > 0.9 and mp["recall"] > 0.85
    assert mp["home_acc"] > 0.85 and mp["work_acc"] > 0.7


def test_shared_places_and_relations():
    _, g, mv, xy, res = _setup()
    ev = T.evaluate(g, res, xy)
    assert ev["shared_places_k2"]["precision"] > 0.8 and ev["shared_places_k2"]["recall"] > 0.8
    rel = ev["relations"]
    assert rel["precision"] > 0.9 and rel["recall"] > 0.7
    # shared-anchor edges are backed by an anchor place used by both users
    for a, b, d in res.graph.edges(data=True):
        assert d["places"] and all({a, b} <= res.anchor_users[p] for p in d["places"])
    assert all(set().union(*res.communities) >= {u} for u in res.users)


def test_invariant_to_longitude_shift_and_time_zone():
    """M uses only map-relative features: shifting the whole world (traces + map) in longitude and
    the clock by a few hours must not change the discovered places."""
    prov, g, mv, xy, res = _setup()
    tr = g.result.traces
    sub = {u: tr[u] for u in sorted(tr)[:12]}
    base = run_m(sub, mv, CFG)
    dl, dt = 9.0, 5 * 3600
    shifted = {u: [GpsPoint(p.t + dt, p.lat, p.lon + dl) for p in ps] for u, ps in sub.items()}
    pt = mv.places
    mv2 = MapView(PlaceTable(pt.ids, pt.categories, pt.lat, pt.lon + dl, pt.popularity))
    out = run_m(shifted, mv2, CFG)
    assert abs((out.night_start_h - base.night_start_h) % 24 - 5) < 1e-9
    for u in sub:
        assert {p.place_id for p in base.main_places(u)} == {p.place_id for p in out.main_places(u)}
        assert [p.role for p in base.users[u]] == [p.role for p in out.users[u]]


def test_deterministic():
    _, g, mv, xy, res = _setup()
    sub = {u: g.result.traces[u] for u in sorted(g.result.traces)[:15]}
    a, b = run_m(sub, mv, CFG), run_m(sub, mv, CFG)
    assert sorted(map(sorted, a.communities)) == sorted(map(sorted, b.communities))
    assert {u: [p.place_id for p in v] for u, v in a.users.items()} == {u: [p.place_id for p in v] for u, v in b.users.items()}


def test_crossings_with_road_graph():
    prov = T.make_provider()
    g = T.generate_data(15, 9, {"n_days": 4}, prov)
    mv = T.map_view(prov, with_graph=True)
    res = run_m(g.result.traces, mv, MConfig(match_trips=True))
    ev = T.evaluate(g, res, {p.id: (p.x, p.y) for p in prov.places()})
    cr = ev["crossings_k2"]
    assert cr["n_pred"] > 0 and cr["precision"] > 0.8


def test_noise_degrades_gracefully():
    prov, _, mv, xy, _ = _setup()
    g = T.generate_data(N, SEED, {"n_days": 6, **T.REAL_LIKE_GEN}, prov)
    res = run_m(g.result.traces, mv, CFG)
    ev = T.evaluate(g, res, xy)
    assert ev["main_places"]["precision"] > 0.9 and ev["main_places"]["recall"] > 0.7
    assert ev["relations"]["precision"] > 0.85


def test_night_window_from_activity_only():
    # movement only between 08:00-18:00 UTC -> quietest 6 h window sits inside 18..08
    pts = []
    for d in range(3):
        for h in range(8, 18):
            for k in range(30):
                t = d * 86400 + h * 3600 + k * 60
                pts.append(GpsPoint(t, 39.9 + 0.001 * k, 116.4))
    ns = estimate_night_start({"u": pts}, 6.0)
    window = {(ns + i) % 24 for i in range(6)}
    assert not window & set(range(8, 18))


def test_relation_graph_weights_and_communities():
    mk = lambda pid, role: UserPlace(pid, "x", 3, 3, 1e4, 0.5, 3, 1e4, 9.0, role=role, main=True)
    users = {u: [] for u in "abcdef"}
    anchors = {"P": {"a", "b", "c"}, "Q": {"d", "e"}, "R": {"a", "f"}, "S": {"b"}}
    G = relation_graph(users, anchors, MConfig())
    assert set(map(frozenset, G.edges)) == {frozenset(e) for e in ["ab", "ac", "bc", "de", "af"]}
    assert G["d"]["e"]["weight"] > G["a"]["b"]["weight"]      # rarer anchor -> stronger tie
    comms = communities(G, MConfig())
    assert {frozenset(c) for c in comms if len(c) > 1} >= {frozenset("de")}


def test_eval_helpers():
    r = T._prf({1, 2, 3}, {2, 3, 4})
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    lab = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert T._ari(lab, {"a": 5, "b": 5, "c": 7, "d": 7}) == pytest.approx(1.0)
    assert T._ari(lab, {"a": 0, "b": 1, "c": 0, "d": 1}) < 0.1
    assert T.compare_distributions({"main_places_hist": {1: 2}, "users_per_place_hist": {1: 1},
                                    "degree_hist": {0: 1}, "role_share": {"home": 1.0}},
                                   {"main_places_hist": {1: 2}, "users_per_place_hist": {1: 1},
                                    "degree_hist": {0: 1}, "role_share": {"home": 1.0}})["tv_degree"] == 0


def test_runs_without_place_table_h3_fallback():
    """The 'real data' path: no places table, no road graph -> H3-cell place ids, label-free stats."""
    _, g, _, _, _ = _setup()
    sub = {u: g.result.traces[u] for u in sorted(g.result.traces)[:20]}
    res = run_m(sub, MapView(PlaceTable.empty()), CFG)
    st = T.distribution_stats(res)
    assert st["users_with_places"] >= 15
    assert all(p.place_id.startswith("h3:") for ps in res.users.values() for p in ps)
    assert st["main_places_per_user"]["mean"] > 1.5

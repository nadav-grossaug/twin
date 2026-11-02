"""M v2 building blocks on hand-made inputs (no generator, runs in a second)."""
import numpy as np

from geolife.eval.m.core import MapView
from geolife.eval.m.outings import (M2Config, _bh, _channel, analyze, communities, place_merge_map, relation_graph, tv_hist)
from geolife.eval.m.outings_transfer import split_user
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.transfer.types import Outing, UserOutings, Visit

T0 = 1_704_067_200 - 8 * 3600      # Monday 00:00 local


def _o(day, hour, places, kinds=None):
    t = T0 + day * 86400 + hour * 3600
    kinds = kinds or ["stop"] * len(places)
    vis = tuple(Visit(p, "shop", "c", t + 600 * i, 900, k) for i, (p, k) in enumerate(zip(places, kinds)))
    return Outing(t, t + 3600, "walk", 2000.0, vis, cells=("x",))


def test_place_merge_is_data_independent_and_shift_invariant():
    lat = np.array([39.9, 39.9, 39.9, 39.95])
    lon = np.array([116.4, 116.4004, 116.4008, 116.5])       # ~34 m apart, then far away
    pt = PlaceTable(["a", "b", "c", "d"], ["shop"] * 4, lat, lon, [1.0, 3.0, 2.0, 1.0])
    m = place_merge_map(MapView(pt), 60.0)
    assert m == {"a": "b", "c": "b"}          # most popular place leads; far place stays alone
    pt2 = PlaceTable(["a", "b", "c", "d"], ["shop"] * 4, lat, lon + 20.0, [1.0, 3.0, 2.0, 1.0])
    assert place_merge_map(MapView(pt2), 60.0) == m
    assert place_merge_map(MapView(pt), 0.0) == {}


def test_user_places_rank_and_time_profile():
    uo = UserOutings("u", [_o(0, 8, ["A"]), _o(1, 9, ["A", "B"]), _o(5, 15, ["A"]), _o(6, 15, ["B"])])
    res = analyze([uo], M2Config())
    ups = res.users["u"]
    assert [p.place_id for p in ups] == ["A", "B"]            # A: 3 outings, B: 2
    assert ups[0].n_outings == 3 and ups[0].n_days == 3 and ups[0].poi
    prof = ups[0].profile_dict()
    assert prof["weekday"]["morning"] == 2 and prof["weekend"]["afternoon"] == 1
    assert res.pois("u")[1].place_id == "B"


def test_poi_needs_two_outings_and_two_days():
    uo = UserOutings("u", [_o(0, 8, ["A"]), _o(0, 20, ["A"]), _o(1, 8, ["B"])])    # A twice but on one day
    res = analyze([uo], M2Config())
    assert not any(p.poi for p in res.users["u"])


def test_common_places_count_distinct_users():
    u = [UserOutings(f"u{i}", [_o(0, 8, ["H", f"own{i}"]), _o(1, 8, ["H"])]) for i in range(4)]
    res = analyze(u, M2Config())
    assert res.common_at(4) == {"H"}
    assert set(res.common_places) == {"H"}
    assert res.common_at(5) == set()


def test_endpoint_only_visit_needs_two_outings_for_common():
    a = UserOutings("a", [_o(0, 8, ["X"], ["start"]), _o(1, 8, ["Y"])])
    b = UserOutings("b", [_o(0, 8, ["X"], ["start"]), _o(1, 8, ["Z"])])
    assert analyze([a, b], M2Config()).common_at(2) == set()
    c = UserOutings("a", [_o(0, 8, ["X"], ["start"]), _o(1, 8, ["X"], ["end"])])
    d = UserOutings("b", [_o(0, 8, ["X"], ["start"]), _o(1, 8, ["X"], ["end"])])
    assert analyze([c, d], M2Config()).common_at(2) == {"X"}


def test_null_model_ignores_shared_hubs_and_links_rare_overlap():
    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(30)]
    hubs = {f"hub{i}" for i in range(6)}
    sets = {u: set(hubs) | {f"own_{u}_{j}" for j in range(8)} for u in users}
    for u in users[:5]:                                        # a clique sharing 5 rare places
        sets[u] |= {f"rare{j}" for j in range(5)}
    # also give everybody random mid-popularity places
    mid = [f"mid{i}" for i in range(40)]
    for u in users:
        sets[u] |= set(rng.choice(mid, 5, replace=False))
    prof = {u: {} for u in users}
    G = relation_graph(users, sets, prof, None, M2Config(rel_alpha=0.05))
    edges = {tuple(sorted(e)) for e in G.edges}
    clique = {tuple(sorted((a, b))) for a in users[:5] for b in users[:5] if a < b}
    assert len(edges & clique) >= 8                           # most of the 10 clique pairs
    assert len(edges - clique) <= 3                            # hubs alone do not create relations
    comms = communities(G, M2Config())
    assert any(set(users[:5]) <= c for c in comms)


def test_no_shared_items_no_edges_and_pvalues_in_unit_interval():
    sets = {f"u{i}": {f"p{i}_{j}" for j in range(5)} for i in range(10)}
    G = relation_graph(list(sets), sets, {}, None, M2Config())
    assert G.number_of_edges() == 0
    shared, iu, pv = _channel({"a": {"x", "y"}, "b": {"x", "y"}, "c": {"x"}, "d": {"q"}})
    assert all(0 <= p <= 1 for p in pv.values()) and ("a", "b") in shared


def test_bh_mask():
    p = np.array([0.001, 0.2, 0.01, 0.9])
    assert _bh(p, 4, 0.05).tolist() == [True, False, True, False]
    assert _bh(np.zeros(0), 4, 0.05).size == 0


def test_time_bonus_only_changes_weights():
    users = [f"u{i}" for i in range(12)]
    sets = {u: {f"own_{u}_{j}" for j in range(6)} for u in users}
    items = [f"s{j}" for j in range(12)]
    sets["u0"] |= set(items)
    sets["u1"] |= set(items)
    same = np.eye(8)[0] * 3
    diff = np.eye(8)[5] * 3
    pa = {u: {i: same for i in items} for u in ("u0", "u1")}
    pb = {"u0": {i: same for i in items}, "u1": {i: diff for i in items}}
    cfg = M2Config(rel_alpha=0.2, time_bonus=1.0)
    g1 = relation_graph(users, sets, pa, None, cfg)
    g2 = relation_graph(users, sets, pb, None, cfg)
    assert set(g1.edges) == set(g2.edges) == {("u0", "u1")}
    assert g1["u0"]["u1"]["weight"] > g2["u0"]["u1"]["weight"]


def test_split_user_partitions_outings():
    uo = UserOutings("u", [_o(d, 8, ["A"]) for d in range(10)])
    for how in ("time", "random"):
        a, b = split_user(uo, how, np.random.default_rng(1))
        assert len(a.outings) == 5 and len(b.outings) == 5
        assert {o.t_start for o in a.outings} | {o.t_start for o in b.outings} == {o.t_start for o in uo.outings}
    a, b = split_user(uo, "time", np.random.default_rng(1))
    assert max(o.t_start for o in a.outings) < min(o.t_start for o in b.outings)


def test_tv_hist():
    assert tv_hist({1: 5, 2: 5}, {1: 5, 2: 5}) == 0.0
    assert abs(tv_hist({1: 10}, {2: 10}) - 1.0) < 1e-9

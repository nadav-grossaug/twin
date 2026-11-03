"""M v2 end to end on a small toy-city generate_v2 draw: planted-truth recovery floors, coordinate invariance."""
import numpy as np
import pytest

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline_v2 import generate_v2
from geolife.eval.m.outings import M2Config, MapView, distribution_stats, run_m2
from geolife.eval.m.outings_transfer import evaluate, grid_graph, places_table, row, split_half
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.transfer.types import GpsPoint

GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 40, "sample_interval_s": 15}


@pytest.fixture(scope="module")
def world():
    prov = GridCityProvider(n=24, spacing=250, n_places=2500, seed=2)
    p = fake_params_v2()
    p["users"]["outings_per_user"]["mu"] = float(np.log(35.0))
    p["users"]["outings_per_user"]["sigma"] = 0.4
    g = generate_v2(p, prov, 36, 3, gen=GEN)
    mv = MapView(places_table(prov), grid_graph(prov))
    res = run_m2(g.result.traces, mv, M2Config())
    return prov, g, mv, res


def test_recovery_floors(world):
    _, g, _, res = world
    ev = evaluate(g, res)
    assert res.stats["n_users"] >= 20
    assert ev["poi"]["p"] > 0.7 and ev["poi"]["r"] > 0.65
    assert ev["common_places_k2"]["p"] > 0.7 and ev["common_places_k2"]["r"] > 0.5
    assert ev["common_segments_k2"]["p"] > 0.85 and ev["common_segments_k2"]["r"] > 0.8
    r = ev["relations"]
    assert r["precision_group_or_cotravel"] >= 0.7 or r["n_edges"] == 0


def test_output_structure(world):
    _, _, _, res = world
    assert res.trav_kind == "segment"
    for u, ups in res.users.items():
        scores = [p.score for p in ups]
        assert scores == sorted(scores, reverse=True)
        for p in ups:
            assert p.profile.shape == (8,) and p.profile.sum() == p.n_visits
    assert set(res.graph.nodes) == {u.user for u in res.outings}
    assert all(len(u) >= 2 for u in res.common_places.values())
    assert all(len(u) >= 2 for u in res.common_segments.values())
    d = distribution_stats(res)
    assert d["outings_per_user"]["median"] >= 10 and 0 <= d["stop_revisit_rate"]["mean"] <= 1


def test_cell_proxy_when_no_graph(world):
    prov, g, _, _ = world
    res = run_m2(g.result.traces, MapView(places_table(prov), None), M2Config())
    assert res.trav_kind == "cell" and res.common_segments


def test_translation_invariance(world):
    """Shifting every trace and the map by a constant longitude must not change what M finds (ids only)."""
    prov, g, mv, res = world
    dl = 7.0
    tr = {u: [GpsPoint(p.t, p.lat, p.lon + dl) for p in pts] for u, pts in list(g.result.traces.items())[:14]}
    pt = mv.places
    mv2 = MapView(PlaceTable(pt.ids, pt.categories, pt.lat, pt.lon + dl, pt.popularity), None)
    mv1 = MapView(mv.places, None)
    a = run_m2({u: g.result.traces[u] for u in tr}, mv1, M2Config())
    b = run_m2(tr, mv2, M2Config())

    def known(r):
        return {u: [p.place_id for p in ups if not p.place_id.startswith("h3:")] for u, ups in r.users.items()}
    assert known(a) == known(b)
    assert {tuple(sorted(e)) for e in a.graph.edges} == {tuple(sorted(e)) for e in b.graph.edges}


def test_split_half_runs_and_beats_other_users(world):
    _, _, _, res = world
    s = split_half(res.outings, M2Config(), True, "random", merge=res.merge)
    assert s["poi_jaccard_median"] >= s["poi_jaccard_other_user"]
    assert s["B_visits_in_A_poi"] > s["B_visits_in_A_poi_other_user"]
    t = split_half(res.outings, M2Config(), True, "time", merge=res.merge)
    assert 0 <= t["common_places_k2_jaccard"] <= 1


def test_row_helper(world):
    _, g, _, res = world
    r = row(evaluate(g, res))
    assert 0 <= r["poi_p"] <= 1 and r["users"] == res.stats["n_users"]

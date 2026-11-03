"""Tokenization tests: stays, snapping, trips, matcher extras, tokens (no network, no real data)."""
import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from geolife.inside.data import LabelSpan
from geolife.transfer.map.graph import build_graph
from geolife.transfer.map.match import Matcher, evaluate_match
from geolife.eval.match_tune import evaluate, synth_path
from geolife.transfer.tokenize.stays import PlaceTable, detect_stays, merge_adjacent, snap_stay, thin
from geolife.transfer.tokenize.tokens import TokenizeConfig, tokenize_all, tokenize_user
from geolife.transfer.tokenize.trips import label_mode, segment_trips, speed_mode
from geolife.transfer.types import GpsPoint

LAT, LON = 39.98, 116.30
KX = 111195.0 * math.cos(math.radians(LAT))
KY = 111195.0
BOX = (39.9, 40.1, 116.2, 116.4)


def pt(t, east_m=0.0, north_m=0.0):
    return GpsPoint(int(t), LAT + north_m / KY, LON + east_m / KX)


def day(stay_a_s=1800, move_s=600, stay_b_s=1800, speed=10.0, t0=1_200_000_000):
    """stay at origin (jitter), straight move east at `speed` m/s, stay at destination."""
    pts, t = [], t0
    rng = np.random.default_rng(0)
    for _ in range(stay_a_s // 5):
        pts.append(pt(t, rng.normal(0, 5), rng.normal(0, 5)))
        t += 5
    x = 0.0
    for _ in range(move_s // 5):
        x += speed * 5
        pts.append(pt(t, x))
        t += 5
    for _ in range(stay_b_s // 5):
        pts.append(pt(t, x + rng.normal(0, 5), rng.normal(0, 5)))
        t += 5
    return pts


# ---------------------------------------------------------------- stays


def test_stay_detection_basic_and_config():
    pts = day()
    raws = detect_stays(pts)
    assert len(raws) == 2
    assert raws[0].t_end - raws[0].t_start >= 1200
    assert abs(raws[1].lon - (LON + 6000 / KX)) < 0.0005
    assert detect_stays(pts, min_dur_s=3600) == []        # configurable duration
    assert len(detect_stays(pts, dist_m=5000, min_dur_s=1200)) >= 1


def test_no_stay_when_moving():
    pts = [pt(i * 5, i * 50.0) for i in range(300)]  # 10 m/s
    assert detect_stays(pts) == []


def test_gap_ends_stay_candidate():
    pts = [pt(i * 5) for i in range(100)] + [pt(5000 + i * 5) for i in range(100)]
    assert len(detect_stays(pts, max_gap_s=None)) == 1
    assert detect_stays(pts, min_dur_s=1200, max_gap_s=600) == []


def test_thin_keeps_ends():
    pts = [pt(i) for i in range(100)]
    out = thin(pts, 10)
    assert out[0] is pts[0] and out[-1] is pts[-1] and len(out) <= 12


def test_snap_to_place_or_h3():
    places = PlaceTable(["p1", "p2"], ["cafe", "school"], [LAT, LAT + 0.01], [LON, LON], [5, 1])
    raws = detect_stays(day())
    s0 = snap_stay(raws[0], places, res=8, radius_m=100)
    assert (s0.place_id, s0.category) == ("p1", "cafe")
    s1 = snap_stay(raws[1], places, res=8, radius_m=100)
    assert s1.place_id == f"h3:{s1.cell}" and s1.category == "unknown"
    assert snap_stay(raws[0], PlaceTable.empty()).place_id.startswith("h3:")


def test_places_parquet_roundtrip(tmp_path):
    p = tmp_path / "places.parquet"
    pd.DataFrame({"id": ["a"], "category": ["shop"], "lat": [LAT], "lon": [LON],
                  "popularity": [3.0]}).to_parquet(p)
    t = PlaceTable.from_parquet(p)
    assert t.nearest(LAT, LON, 50) == ("a", "shop")
    assert t.nearest(LAT + 0.01, LON, 50) is None


def test_merge_adjacent():
    from geolife.transfer.types import Stay
    a = Stay("x", "c", "cell", 0, 100)
    b = Stay("x", "c", "cell", 500, 900)
    c = Stay("x", "c", "cell", 10_000, 11_000)
    assert len(merge_adjacent([c, b, a], 1800)) == 2


# ---------------------------------------------------------------- trips


def test_label_and_speed_modes():
    labels = [LabelSpan(0, 100, "taxi"), LabelSpan(100, 130, "walk")]
    assert label_mode(labels, 0, 110) == "car"
    assert label_mode(labels, 500, 600) is None
    assert speed_mode([pt(i * 5, i * 5 * 1.2) for i in range(60)]) == "walk"
    assert speed_mode([pt(i * 5, i * 5 * 4.0) for i in range(60)]) == "bike"
    assert speed_mode([pt(i * 5, i * 5 * 12.0) for i in range(60)]) == "car"
    assert speed_mode([pt(i * 5, i * 5 * 40.0) for i in range(60)]) == "train"


def test_trip_between_stays_with_mode_and_length():
    pts = day(speed=10.0, move_s=600)
    raws = detect_stays(pts)
    places = PlaceTable.empty()
    stays = [snap_stay(r, places) for r in raws]
    trips, circ = segment_trips(pts, raws, stays)
    assert len(trips) == 1
    t = trips[0]
    assert t.origin == stays[0].place_id and t.dest == stays[1].place_id
    assert t.mode == "car" and 5000 < t.length_m < 7500
    assert list(circ.values())[0] == pytest.approx(1.0, abs=0.15)
    trips_l, _ = segment_trips(pts, raws, stays, labels=[LabelSpan(0, 2**40, "bike")])
    assert trips_l[0].mode == "bike"


def test_trip_skipped_over_gap():
    pts = day()
    # remove the middle of the move to create a >10 min gap
    gap = [p for p in pts if not (pts[400].t < p.t < pts[400].t + 900)]
    raws = detect_stays(gap)
    stays = [snap_stay(r, PlaceTable.empty()) for r in raws]
    trips, _ = segment_trips(gap, raws, stays, max_gap_s=600)
    assert trips == []


# ---------------------------------------------------------------- tokens


def test_tokenize_users_pseudonyms_and_records():
    cfg = TokenizeConfig(bbox=BOX)
    trajs = {"000": [day(), day(t0=1_200_100_000)], "001": [day(t0=1_200_200_000)], "002": []}
    recs, circ = tokenize_all(trajs, PlaceTable.empty(), cfg, seed=3)
    assert len(recs) == 2                       # user without stays dropped
    names = {r.user for r in recs}
    assert all(n.startswith("u") and len(n) == 9 for n in names)
    assert not names & {"000", "001", "002"}
    recs2, _ = tokenize_all(trajs, PlaceTable.empty(), cfg, seed=4)
    assert {r.user for r in recs2} != names     # seed changes the pseudonyms
    r = next(r for r in recs if len(r.trips) == 2)
    assert len(r.stays) >= 2 and r.stays[0].category == "unknown"
    assert all(k[0].startswith("h3:") for k in circ)


def test_tokenize_filters_outside_bbox():
    far = [GpsPoint(p.t, p.lat + 5, p.lon) for p in day()]
    rec, _ = tokenize_user("u", [far], PlaceTable.empty(), TokenizeConfig(bbox=BOX))
    assert rec.stays == [] and rec.trips == []


# ---------------------------------------------------------------- matcher


D = 0.001


def _grid(n=5, one_way_row=None, footway_row=None):
    nodes = {(i, j): f"c{i}{j}" for i in range(n) for j in range(n)}
    con = pd.DataFrame([{"id": c, "geometry": Point(LON + i * D, LAT + j * D).wkb}
                        for (i, j), c in nodes.items()])
    segs = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b in nodes:
                    line = LineString([(LON + a[0] * D, LAT + a[1] * D), (LON + b[0] * D, LAT + b[1] * D)])
                    segs.append({"id": f"s{a[0]}{a[1]}-{b[0]}{b[1]}", "geometry": line.wkb,
                                 "class": "residential",
                                 "connectors": [{"connector_id": nodes[a], "at": 0.0},
                                                {"connector_id": nodes[b], "at": 1.0}]})
    return pd.DataFrame(segs), con


def _line_pts(x0, x1, y, n=30, noise=0.00003, seed=0):
    rng = np.random.default_rng(seed)
    return [(LON + x + rng.normal(0, noise), LAT + y + rng.normal(0, noise))
            for x in np.linspace(x0, x1, n)]


def test_matcher_class_reuse_and_length():
    segs, con = _grid()
    G = build_graph(segs, con)
    m = Matcher(G)
    ids, length = m.match(_line_pts(0.0001, 3 * D, 0), return_length=True)
    assert ids == ["s00-10", "s10-20", "s20-30"]
    assert length == pytest.approx(3 * D * KX, rel=0.05)
    assert m.match(_line_pts(0.0001, 2 * D, 2 * D, seed=1)) == ["s02-12", "s12-22"]


def test_parallel_footway_collapsed_for_cars_kept_for_walkers():
    segs, con = _grid(3)
    # add a footway 12 m north of the first row, same connectors-free (own nodes)
    off = 12 / KY
    fw_nodes = pd.DataFrame([{"id": f"f{i}", "geometry": Point(LON + i * D, LAT + off).wkb} for i in range(3)])
    fw = pd.DataFrame([{"id": f"fw{i}", "geometry": LineString([(LON + i * D, LAT + off), (LON + (i + 1) * D, LAT + off)]).wkb,
                        "class": "footway", "connectors": [{"connector_id": f"f{i}", "at": 0.0},
                                                            {"connector_id": f"f{i+1}", "at": 1.0}]}
                       for i in range(2)])
    # connect footway ends to the road grid so the graph is one component
    link = pd.DataFrame([{"id": f"lk{i}", "geometry": LineString([(LON + i * 2 * D, LAT), (LON + i * 2 * D, LAT + off)]).wkb,
                          "class": "footway", "connectors": [{"connector_id": f"c{i*2}0", "at": 0.0},
                                                             {"connector_id": f"f{i*2}", "at": 1.0}]} for i in range(2)])
    G = build_graph(pd.concat([segs, fw, link], ignore_index=True), pd.concat([con, fw_nodes], ignore_index=True))
    # walker on the footway row (y = +12 m): footway segments allowed
    pts = _line_pts(0.0001, 2 * D - 0.0001, off, noise=0.00001)
    assert any(s.startswith("fw") for s in Matcher(G).match(pts, "walk"))
    # a car 6 m from the footway line is matched onto the road, never the footway
    car = _line_pts(0.0001, 2 * D - 0.0001, off / 2, noise=0.00001)
    assert not any(s.startswith(("fw", "lk")) for s in Matcher(G).match(car, "car"))


def test_one_way_awareness():
    segs, con = _grid(3)
    G = build_graph(segs, con)
    # make the middle edge c10-c20 one way from c20 to c10 (westbound only)
    G["c10"]["c20"]["oneway_from"] = "c20"
    m = Matcher(G)
    assert m.directed
    east = m.match(_line_pts(0.0001, 2 * D, 0, n=40))
    west = m.match(_line_pts(2 * D, 0.0001, 0, n=40, seed=2))
    assert "s10-20" in west
    assert "s10-20" not in east  # eastbound traffic must detour (or avoid the edge)


def test_flicker_cleanup():
    from geolife.transfer.map.match import _clean
    p = [("A", 100, 1), ("B", 5, 2), ("A", 90, 3), ("C", 80, 4)]
    assert [x[0] for x in _clean(p, 15.0)] == ["A", "C"]
    p = [("A", 100, 1), ("B", 5, 2), ("C", 80, 3)]
    assert [x[0] for x in _clean(p, 15.0)] == ["A", "C"]


def test_subsample_and_no_candidates():
    segs, con = _grid()
    G = build_graph(segs, con)
    m = Matcher(G, subsample_m=30)
    pts = _line_pts(0.0001, D, 0, n=80, noise=0.0)
    assert len(m._subsample(pts)) < len(pts) / 2
    assert m.match([(LON + 5, LAT + 5)]) == []


def test_evaluate_match_rates():
    segs, con = _grid()
    G = build_graph(segs, con)
    r = evaluate_match(G, ["s00-10", "s10-20"], ["s00-10", "s20-30"])
    assert r["recovered"] == 1 and r["extra"] == 1 and r["recovered_rate"] == 0.5
    assert 0 < r["true_len_covered"] < 1


def test_synthetic_noisy_path_recovery_on_grid():
    segs, con = _grid(6)
    G = build_graph(segs, con)
    res = evaluate(G, n_paths=3, noise_m=6.0, seed=1, min_len_m=300.0)
    assert res["recovered_rate"] > 0.8 and res["extra_rate"] < 1.0
    truth, pts = synth_path(G, __import__("random").Random(0), np.random.default_rng(0), min_len_m=300)
    assert len(truth) >= 2 and len(pts) > 5

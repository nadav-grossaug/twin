import numpy as np
import pandas as pd

from geolife.inside.data import LabelSpan
from geolife.transfer.tokenize.outings import (OutingsConfig, PlaceIndex, build_outings, label_codes,
                                      local_bins, to_frames)
from geolife.transfer.types import GpsPoint

T0 = 1_224_730_384          # 2008-10-23 03:53 UTC, a Thursday (11:53 local)
M = 1 / 111195.0            # one metre in degrees of latitude


def walk(t0, lat0, lon0, n, dt=5, speed=1.4):
    return [GpsPoint(t0 + k * dt, lat0 + k * dt * speed * M, lon0) for k in range(n)]


def stay(t0, lat0, lon0, n, dt=30):
    return [GpsPoint(t0 + k * dt, lat0, lon0) for k in range(n)]


def user_trace(n_out=12, days=6):
    """n_out outings over `days` local days: walk ~2 km, stop 20 min, walk on."""
    pts = []
    for k in range(n_out):
        t = T0 + (k % days) * 86400 + (k // days) * 4 * 3600
        a = walk(t, 39.90, 116.30, 300)
        end = a[-1]
        b = stay(end.t + 30, end.lat, end.lon, 40)
        c = walk(b[-1].t + 30, end.lat, end.lon, 100)
        pts += a + b + c
    return pts


PLACES = pd.DataFrame({"id": ["p1"], "category": ["food"], "lat": [39.90 + 300 * 5 * 1.4 * M],
                       "lon": [116.30], "popularity": [1.0]})


def test_local_bins_boundaries():
    fri0 = 1224806400 - (1224806400 % 86400)          # a Friday 00:00 UTC (2008-10-24)
    fri22 = fri0 + 22 * 3600                          # = Saturday 06:00 local
    assert local_bins(fri22) == ("morning", "weekend")
    assert local_bins(fri22 - 1) == ("night", "weekend")
    assert local_bins(fri22 - 8 * 3600) == ("evening", "weekday")     
    assert local_bins(fri22 + 17 * 3600 + 59 * 60) == ("evening", "weekend")
    assert local_bins(fri22 + 18 * 3600) == ("night", "weekend")        # Sunday 00:00 local
    assert local_bins(fri22 + 42 * 3600) == ("night", "weekday")        # Monday 00:00 local


def test_build_outings_basic():
    res = build_outings({"alice": user_trace()}, PLACES)
    assert len(res) == 1 and res[0].user.startswith("u") and "alice" not in res[0].user
    outs = res[0].outings
    assert len(outs) == 12
    o = outs[0]
    assert o.t_end > o.t_start and o.length_m > 2500
    kinds = [v.kind for v in o.visits]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("stop") == 1
    stop = [v for v in o.visits if v.kind == "stop"][0]
    assert stop.place_id == "p1" and stop.category == "food" and stop.dwell_s >= 1100
    assert o.visits[0].place_id.startswith("h3:") and o.visits[0].category == "unknown"
    assert o.mode == "walk" and len(o.cells) > 5 and len(set(o.cells)) == len(o.cells)
    assert o.segments == ()


def test_gap_splits_and_merges():
    pts = user_trace(12, 6)
    t_last = pts[-1].t
    assert len(build_outings({"a": pts + walk(t_last + 3600, 39.9, 116.3, 100)}, PLACES)[0].outings) == 12
    more = walk(t_last + 3 * 3600, 39.9, 116.3, 300)
    assert len(build_outings({"a": pts + more}, PLACES)[0].outings) == 13
    tiny = stay(t_last + 3 * 3600, 39.8, 116.2, 5)
    assert len(build_outings({"a": pts + tiny}, PLACES)[0].outings) == 12   # not a real outing


def test_min_activity_rule():
    assert build_outings({"a": user_trace(9, 6)}, PLACES) == []          # < 10 outings
    assert build_outings({"a": user_trace(12, 4)}, PLACES) == []         # < 5 days
    assert len(build_outings({"a": user_trace(12, 5)}, PLACES)) == 1


def test_nested_files_equal_flat():
    pts = user_trace()
    files = [pts[i:i + 500] for i in range(0, len(pts), 500)]
    a = build_outings({"x": pts}, PLACES)[0].outings
    b = build_outings({"x": files}, PLACES)[0].outings
    assert [(o.t_start, o.t_end) for o in a] == [(o.t_start, o.t_end) for o in b]


def test_labels_override_speed():
    pts = user_trace()
    labs = [LabelSpan(pts[0].t - 10, pts[-1].t + 10, "taxi")]
    res = build_outings({"a": pts}, PLACES, labels={"a": labs})
    assert all(o.mode == "car" for o in res[0].outings)


def test_label_codes():
    c = label_codes(np.array([5, 15, 25, 35]), [LabelSpan(10, 20, "bus"), LabelSpan(30, 40, "boat")])
    assert list(c) == [0, 3, 0, 6]


def test_speed_mode_car():
    pts = []
    for k in range(12):
        pts += walk(T0 + k * 86400, 39.8, 116.3, 200, dt=5, speed=10)
    res = build_outings({"a": pts}, None, OutingsConfig())
    assert res and all(o.mode == "car" for o in res[0].outings)


def test_snap_radius_and_empty_places():
    idx = PlaceIndex(PLACES)
    dlon = 1 / (111195.0 * np.cos(np.radians(39.9)))
    assert idx.snap([PLACES.lat[0]], [PLACES.lon[0] + 50 * dlon], 75.0)[0] == 0
    assert idx.snap([PLACES.lat[0]], [PLACES.lon[0] + 150 * dlon], 75.0)[0] == -1
    assert PlaceIndex(None).snap([39.9], [116.3], 75.0)[0] == -1


def test_frames():
    res = build_outings({"a": user_trace()}, PLACES)
    of, vf = to_frames(res)
    assert len(of) == 12 and set(vf.kind) == {"start", "stop", "end"}
    assert (of.n_visits == 3).all()

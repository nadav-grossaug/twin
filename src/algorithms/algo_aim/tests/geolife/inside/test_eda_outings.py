import numpy as np
import pandas as pd

from geolife.inside import eda_outings as E


def _pts():
    # user a: file 0 moving 0..99 s at 1 s, then file 1 starting 100 s later (merge at gap 1h),
    # then file 2 three hours later. user b: one file.
    rows = []
    for k in range(100):
        rows.append(("a", 0, 1000 + k, 39.9 + k * 1e-5, 116.3))
    for k in range(100):
        rows.append(("a", 1, 1200 + k, 39.9 + 1e-3 + k * 1e-5, 116.3))
    for k in range(50):
        rows.append(("a", 2, 20000 + k, 39.95, 116.3 + k * 1e-5))
    for k in range(60):
        rows.append(("b", 0, 5000 + k, 39.9, 116.35))
    return pd.DataFrame(rows, columns=["uid", "traj", "t", "lat", "lon"])


def test_outing_ids_and_table():
    df = E.clean_points(_pts())
    d, _ = E.step_dist_dt(df.u.values, df.t.values, df.lat.values, df.lon.values)
    oid = E.outing_ids(df.u.values, df.t.values, 3600)
    o = E.outing_table(df, oid, d)
    assert len(o) == 3  # a: files 0+1 merge, file 2 separate; b separate
    assert list(o.n) == [200, 50, 60]
    assert list(o.nfiles) == [2, 1, 1]
    assert 150 < o.len_m[0] < 400  # ~1.1 m per step, 199 steps
    oid_small = E.outing_ids(df.u.values, df.t.values, 60)
    assert oid_small.max() + 1 == 4


def test_local_time_is_utc_plus_8():
    t = np.array([0, 16 * 3600])  # 1970-01-01 00:00 UTC (Thursday) -> 08:00 local; 16:00 UTC -> next day 00:00
    h, dow = E.local_hour_dow(t)
    assert list(h) == [8, 0]
    assert list(dow) == [3, 4]  # Thursday, Friday


def test_detect_stays_finds_dwell_and_resets_at_outing_break():
    n = 100
    t = np.arange(n) * 30.0
    x = np.r_[np.linspace(0, 3000, 20), np.full(60, 3000.0) + np.random.default_rng(0).normal(0, 5, 60), np.linspace(3000, 6000, 20)]
    y = np.zeros(n)
    seg = np.zeros(n, int)
    i0, i1, mx, my, rms = E.detect_stays(x, y, t, seg, 100, 600)
    assert len(i0) == 1 and 15 <= i0[0] <= 25 and i1[0] >= 75
    assert abs(mx[0] - 3000) < 200 and rms[0] < 30
    seg2 = np.r_[np.zeros(50, int), np.ones(50, int)]  # split the dwell in two halves of 15 min
    i0b, *_ = E.detect_stays(x, y, t, seg2, 100, 1200)
    assert len(i0b) == 0


def test_covisit_graph_rarity_and_shared():
    # users 0,1 share two rare cells; user 2 shares only a popular cell
    u = [0, 0, 1, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    c = [1, 2, 1, 2, 1, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    g = E.covisit_graph(np.array(u), np.array(c), max_pop=5, min_shared=2)
    assert list(zip(g.a, g.b)) == [(0, 1)]
    deg = E.degrees(g, 3)
    assert list(deg) == [1, 1, 0]


def test_incidence_and_mode_and_jsd():
    inc = E.incidence(np.array([5, 5, 7, 5]), np.array([0, 0, 0, 1]))
    assert len(inc) == 3
    mode, share = E.mode_per_outing(np.array([0, 0, 1, 1]), np.array([2, 2, -1, -1]), 2, 3)
    assert list(mode) == [2, -1] and share[0] == 1.0
    assert E.js_divergence([1, 1], [1, 1]) == 0
    assert abs(E.js_divergence([1, 0], [0, 1]) - 1.0) < 1e-9


def test_h3_cells_hierarchy():
    r = E.h3_cells(np.array([39.9, 39.9001]), np.array([116.3, 116.3]), (7, 9))
    assert r[7].shape == (2,) and r[7][0] == r[7][1]

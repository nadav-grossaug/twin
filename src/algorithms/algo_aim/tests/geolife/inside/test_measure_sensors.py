"""Sensor measurement stage (PLAN step 1): per-user views, caps, sensitivity, release and ledger."""
import math

import numpy as np
import pytest

from geolife.inside.measure_sensors import SensorView, _cap, _user_rng, make_sensor_view
from geolife.inside.measure_v2 import VocabV2, build_measurements_v2, pmf_table
from geolife.transfer import schema_v2 as S2
from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users

LAT0, LON0 = 39.95, 116.35
M_PER_DEG_LAT = 111_194.9


def synth_points(seed=0, period=2.0, n_runs=3, run_s=1800, gap_every=10, jitter_m=0.8,
                 speed_ms=1.3, stop_s=900):
    """Synthetic raw trace: walking legs at `speed_ms`, one jittered stop per run, a 120 s gap in
    each run, runs separated by > 2 h."""
    rng = np.random.default_rng(seed)
    t, lat, lon = [], [], []
    t0 = 1_230_000_000
    for r in range(n_runs):
        tt = t0 + r * (run_s + 4 * 3600)
        x = y = 0.0
        k = 0
        while tt < t0 + r * (run_s + 4 * 3600) + run_s:
            in_stop = (run_s * 0.4) <= (tt - (t0 + r * (run_s + 4 * 3600))) < (run_s * 0.4 + stop_s)
            if not in_stop:
                x += speed_ms * period
            jx, jy = rng.normal(0, jitter_m, 2)
            t.append(int(tt))
            lat.append(LAT0 + (y + jy) / M_PER_DEG_LAT)
            lon.append(LON0 + (x + jx) / (M_PER_DEG_LAT * math.cos(math.radians(LAT0))))
            k += 1
            tt += period + (120 if k % (gap_every * 20) == 0 else 0)
    return np.array(t), np.array(lat), np.array(lon)


@pytest.fixture(scope="module")
def view():
    t, lat, lon = synth_points()
    return make_sensor_view("p0001", t, lat, lon, [], seed=0)


def test_view_has_every_sensor_table(view):
    assert set(view.pmf) == set(S2.SENSOR_NAMES)
    for n in (S2.SENSOR_INTERVAL, S2.SENSOR_GAP_LEN, S2.SENSOR_SPEED,
              S2.SENSOR_STOP_JITTER, S2.SENSOR_PTS_OUTING):
        assert sum(view.pmf[n].values()) > 0, n
    assert S2.SENSOR_STEP not in view.pmf          # dropped in v2d (D27)


def test_interval_and_median_land_in_the_2s_bin(view):
    c = view.pmf[S2.SENSOR_INTERVAL]
    top = max(c, key=c.get)
    assert float(top[0]) <= 2.0 < float(top[1])
    assert sum(view.pmf[S2.SENSOR_INTERVAL_USER].values()) == 1
    (med_bin,) = view.pmf[S2.SENSOR_INTERVAL_USER]
    assert float(med_bin[0]) <= 2.0 < float(med_bin[1])


def test_gaps_counted_and_rate_onehot(view):
    assert sum(view.pmf[S2.SENSOR_GAP_LEN].values()) >= 3     # one planted 120 s gap per run
    assert all(float(k[0]) >= 60 for k in view.pmf[S2.SENSOR_GAP_LEN])
    assert sum(view.pmf[S2.SENSOR_GAP_RATE].values()) == 1


def test_speed_mode_is_heuristic_walk(view):
    by_mode = {}
    for k, n in view.pmf[S2.SENSOR_SPEED].items():
        by_mode[k[0]] = by_mode.get(k[0], 0) + n
    assert max(by_mode, key=by_mode.get) == "walk"
    # weighted median speed of the walk rows is near the planted 1.3 m/s
    rows = sorted((float(k[1]), float(min(float(k[2]), 60.0)), n)
                  for k, n in view.pmf[S2.SENSOR_SPEED].items() if k[0] == "walk")
    tot = sum(n for _, _, n in rows)
    acc = 0
    for lo, hi, n in rows:
        acc += n
        if acc >= tot / 2:
            assert 0.6 <= (lo + hi) / 2 <= 2.5
            break


def test_points_per_outing_counts_runs(view):
    assert sum(view.pmf[S2.SENSOR_PTS_OUTING].values()) == 3


def test_stop_jitter_is_small(view):
    # weighted MEDIAN: the bulk of in-stop points sits at the planted ~0.8 m jitter; the stay rule
    # also sweeps up walk-in/walk-out points (up to its 200 m radius), which form a small tail.
    rows = sorted(((float(k[0]), n) for k, n in view.pmf[S2.SENSOR_STOP_JITTER].items()))
    tot = sum(n for _, n in rows)
    acc = 0
    for lo, n in rows:
        acc += n
        if acc >= tot / 2:
            assert lo < 10.0
            break


def test_deterministic_given_user_and_seed():
    t, lat, lon = synth_points()
    a = make_sensor_view("pX", t, lat, lon, [], seed=3)
    b = make_sensor_view("pX", t, lat, lon, [], seed=3)
    assert a.pmf == b.pmf


def test_cap_is_deterministic_and_bounded():
    arr = np.arange(1000.0)
    r1 = _cap(_user_rng(0, "u"), arr, 100)
    r2 = _cap(_user_rng(0, "u"), arr, 100)
    assert len(r1) == 100 and (r1 == r2).all() and (np.diff(r1) > 0).all()
    assert (_cap(_user_rng(0, "u"), arr[:50], 100) == arr[:50]).all()


def test_pmf_vector_l2_bound_holds(view):
    for name in S2.SENSOR_NAMES:
        tab = pmf_table([view], name, S2.sensor_vocab(name))
        assert tab.l2_sensitivity == 1.0
        assert tab.observed_max_norm <= 1.0 + 1e-9, name


def test_release_with_sensor_views_splits_rho_80_20():
    users = make_users(10, seed=4, pool=40)
    vocab = VocabV2.from_bbox(BBOX, CATS)
    t, lat, lon = synth_points()
    views = [make_sensor_view(u.user, t, lat, lon, [], seed=0) for u in users[:4]]
    rho = 0.5
    ms, led = build_measurements_v2(users, vocab, rho, np.random.default_rng(0),
                                    sensor_views=views)
    names = {m.name for m in ms}
    assert names == set(S2.ALL_NAMES) | set(S2.SENSOR_NAMES)
    rho_sensor = sum(m.rho for m in ms if m.name in S2.SENSOR_NAMES)
    assert rho_sensor == pytest.approx(0.2 * rho, rel=1e-6)
    assert led.rho_total == pytest.approx(sum(m.rho for m in ms))
    assert led.caps["max_sensor_steps_per_user"] == S2.MAX_SENSOR_STEPS
    text = " ".join(led.caveats).lower()
    for must in ("capped sample", "fixed public bin edges", "80%", "20%", "d20", "hand-set"):
        assert must in text, must
    for m in ms:
        if m.name in S2.SENSOR_NAMES:
            assert m.l2_sensitivity == 1.0


def test_empty_and_out_of_bbox_user_gives_empty_view():
    v = make_sensor_view("pZ", np.array([], np.int64), np.array([]), np.array([]), [], seed=0)
    assert all(sum(c.values()) == 0 for c in v.pmf.values())
    tab = pmf_table([v], S2.SENSOR_INTERVAL, S2.sensor_vocab(S2.SENSOR_INTERVAL))
    assert tab.observed_max_norm == 0.0


def test_release_without_sensor_views_unchanged():
    users = make_users(8, seed=5, pool=30)
    vocab = VocabV2.from_bbox(BBOX, CATS)
    ms, led = build_measurements_v2(users, vocab, 0.5, np.random.default_rng(0))
    assert {m.name for m in ms} == set(S2.ALL_NAMES)
    assert led.rho_total == pytest.approx(0.5, rel=1e-6)
    assert "not measured" in " ".join(led.caveats).lower()


# ---- sensor_dt_runlen (v2e, PLAN step 9): the dt lock texture


def test_dt_runlen_runs_broken_by_value_gap_and_outing_boundary():
    # dt sequence: ten 2s steps | five 3s steps | a 122s gap | seven 2s steps | outing break | four 5s
    dts = [2] * 10 + [3] * 5 + [122] + [2] * 7 + [8000] + [5] * 4
    t = np.cumsum(np.r_[1_000_000_000, dts]).astype(np.int64)
    lat = np.full(len(t), LAT0)
    lon = np.full(len(t), LON0)
    v = make_sensor_view("pR", t, lat, lon, [], seed=0)
    from geolife.inside.measure_v2 import _bucket
    exp = {}
    for ln in (10, 5, 7, 4):   # the gap and the outing boundary both break runs; neither is counted
        b = _bucket(float(ln), S2.RUNLEN_EDGES)
        exp[b] = exp.get(b, 0) + 1
    assert dict(v.pmf[S2.SENSOR_DT_RUNLEN]) == exp
    # per-user regime one-hot: the mean run length (6.5 here) lands in exactly one bin
    oh = dict(v.pmf[S2.SENSOR_DT_RUNLEN_USER])
    assert oh == {_bucket(6.5, S2.RUNLEN_EDGES): 1}


def test_dt_runlen_on_synth_view(view):
    c = view.pmf[S2.SENSOR_DT_RUNLEN]
    total = sum(c.values())
    assert 6 <= total <= 25              # a few locked runs per outing-like run, broken by the gaps
    top = max(c, key=c.get)
    assert float(top[0]) >= 90           # the planted stream locks at 2 s for ~200 steps at a time
    # vocabulary and release mechanics are covered by the SENSOR_NAMES-wide tests above

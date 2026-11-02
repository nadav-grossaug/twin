"""Sensor fit (outside/sensors.py) and the inverse-CDF rendering draws (PLAN step 1)."""
import math

import numpy as np
import pytest

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline_v2 import generate_v2
from geolife.outside.sensors import HistSampler, fit_sensors
from geolife.transfer import schema_v2 as S2
from geolife.transfer.types import Measurement


def _measurement(name, weights, sigma=1e-6):
    """Noiseless sensor Measurement with the given mass per vocabulary key."""
    keys = S2.sensor_vocab(name)
    vals = [float(weights.get(k, 0.0)) for k in keys]
    return Measurement(name=name, keys=keys, values=vals, sigma=sigma, l2_sensitivity=1.0,
                       rho=1.0 / (2 * sigma ** 2))


def _bin_containing(edges, x):
    ks = S2.bin_keys(edges)
    i = S2.bin_of(x, edges)
    return ks[i]


# ------------------------------------------------------------------ HistSampler

def test_hist_sampler_matches_pmf():
    edges = [0.0, 1.0, 2.0, 4.0]
    pmf = [0.5, 0.3, 0.2]
    s = HistSampler(edges, pmf)
    x = s.draw(np.random.default_rng(0), 200_000)
    assert ((x >= 0) & (x <= 4)).all()
    got = [float(((x >= edges[i]) & (x < edges[i + 1])).mean()) for i in range(3)]
    assert got == pytest.approx(pmf, abs=5e-3)
    assert s.quantile(0.5) == pytest.approx(s.median())
    assert s.median() == pytest.approx(1.0, abs=1e-9)        # cdf hits 0.5 exactly at edge 1.0


def test_hist_sampler_truncates_open_top_bin():
    s = HistSampler([0.0, 10.0, math.inf], [0.0, 1.0])
    x = s.draw(np.random.default_rng(1), 1000)
    assert ((x >= 10.0) & (x <= 20.0)).all()                 # TOP_MULT = 2


def test_hist_sampler_degenerate_pmf_is_uniform():
    s = HistSampler([0.0, 1.0, 2.0], [0.0, 0.0])
    x = s.draw(np.random.default_rng(2), 1000)
    assert 0.4 < (x < 1.0).mean() < 0.6


# ------------------------------------------------------------------ fit_sensors

@pytest.fixture(scope="module")
def clean_measurements():
    ms = {}
    w = {_bin_containing(S2.INTERVAL_S_EDGES, 2.0): 90.0,
         _bin_containing(S2.INTERVAL_S_EDGES, 5.0): 10.0}
    ms[S2.SENSOR_INTERVAL] = _measurement(S2.SENSOR_INTERVAL, w)
    ms[S2.SENSOR_INTERVAL_USER] = _measurement(S2.SENSOR_INTERVAL_USER, w)
    ms[S2.SENSOR_GAP_LEN] = _measurement(
        S2.SENSOR_GAP_LEN, {_bin_containing(S2.GAP_S_EDGES, 200.0): 100.0})
    ms[S2.SENSOR_GAP_RATE] = _measurement(
        S2.SENSOR_GAP_RATE, {_bin_containing(S2.GAP_RATE_EDGES, 4.0): 100.0})
    ms[S2.SENSOR_STOP_JITTER] = _measurement(
        S2.SENSOR_STOP_JITTER, {_bin_containing(S2.JITTER_M_EDGES, 5.0): 70.0,
                                _bin_containing(S2.JITTER_M_EDGES, 30.0): 30.0})
    sp = {("walk", *_bin_containing(S2.SPEED_MS_EDGES, 1.2)): 60.0,
          ("car", *_bin_containing(S2.SPEED_MS_EDGES, 8.0)): 40.0}
    ms[S2.SENSOR_SPEED] = _measurement(S2.SENSOR_SPEED, sp)
    st = {("walk", *_bin_containing(S2.STEP_M_EDGES, 3.0)): 60.0,
          ("car", *_bin_containing(S2.STEP_M_EDGES, 80.0)): 40.0}
    ms[S2.SENSOR_STEP] = _measurement(S2.SENSOR_STEP, st)
    ms[S2.SENSOR_HEADING] = _measurement(
        S2.SENSOR_HEADING, {_bin_containing(S2.HEADING_DEG_EDGES, 7.0): 100.0})
    ms[S2.SENSOR_PTS_OUTING] = _measurement(
        S2.SENSOR_PTS_OUTING, {_bin_containing(S2.PTS_PER_OUTING_EDGES, 500.0): 100.0})
    return ms


def test_fit_sensors_clean(clean_measurements):
    sens = fit_sensors(clean_measurements)
    assert sens["available"]
    for key in ("interval", "interval_user", "gap_len", "gap_rate", "stop_jitter",
                "speed_by_mode", "step_by_mode", "heading", "points_per_outing"):
        assert key in sens["hists"], key
    assert not any(v.startswith("mostly noise") for v in sens["fallbacks"].values())
    # per-mode speed: walk fitted near the planted bin, silent modes fall back
    assert set(sens["speed_ms_by_mode"]) == {"walk", "car"}
    assert 0.5 <= sens["speed_ms_by_mode"]["walk"] <= 2.0
    assert 4.5 <= sens["speed_ms_by_mode"]["car"] <= 15.0
    assert any(k.startswith("sensor_speed:") for k in sens["fallbacks"])   # e.g. train
    # en-route noise is never derived from the tables (stop jitter conflates indoor drift)
    assert "sensor_noise_m" not in sens and "sensor_noise_m" in sens["fallbacks"]
    assert sens["gap_rate_per_hr_mean"] == pytest.approx(4.0, abs=1.5)
    import json
    json.dumps(sens)                                          # JSON-safe (no inf)


def test_fit_sensors_noise_falls_back(clean_measurements):
    noisy = {n: Measurement(n, m.keys, list(np.random.default_rng(0).normal(0, 50, len(m.keys))),
                            50.0, 1.0, 2e-4) for n, m in clean_measurements.items()}
    sens = fit_sensors(noisy)
    assert not sens["available"] or len(sens["hists"]) < 3
    assert any("mostly noise" in v for v in sens["fallbacks"].values())


def test_fit_sensors_rejects_implausible_mode_speed(clean_measurements):
    ms = dict(clean_measurements)
    sp = {("walk", *_bin_containing(S2.SPEED_MS_EDGES, 12.0)): 100.0}    # 12 m/s "walk" = noise
    ms[S2.SENSOR_SPEED] = _measurement(S2.SENSOR_SPEED, sp)
    sens = fit_sensors(ms)
    assert "walk" not in sens.get("speed_ms_by_mode", {})
    assert "plausibility window" in sens["fallbacks"]["sensor_speed:walk"]


def test_fit_behaviour_v2_carries_sensor_params(clean_measurements):
    from geolife.outside.fit_v2 import fit_behaviour_v2
    from geolife.inside.measure_v2 import VocabV2, build_measurements_v2
    from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users
    users = make_users(10, seed=7, pool=40)
    ms, led = build_measurements_v2(users, VocabV2.from_bbox(BBOX, CATS), 5.0,
                                    np.random.default_rng(0))
    p = fit_behaviour_v2(ms + list(clean_measurements.values()))
    assert p["sensor"]["available"]
    assert "interval_user" in p["sensor"]["hists"]
    p2 = fit_behaviour_v2(ms)
    assert "sensor" not in p2 and "sensor" in p2["diagnostics"]["fallbacks"]


# ------------------------------------------------------------------ renderer draws

GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 40, "compute_cells": False}


def _params_with_sensor(clean):
    p = fake_params_v2()
    p["sensor"] = fit_sensors(clean)
    return p


@pytest.fixture(scope="module")
def prov():
    return GridCityProvider(n=30, spacing=250, n_places=3000, seed=1)


def _dts(traces):
    out = []
    for pts in traces.values():
        t = np.array([p.t for p in pts])
        dt = np.diff(t)
        out.append(dt[(dt > 0) & (dt <= 60)])
    return np.concatenate(out) if out else np.array([])


def test_render_draws_interval_from_histogram(clean_measurements, prov):
    g = generate_v2(_params_with_sensor(clean_measurements), prov, 12, 3, gen=GEN)
    dt = _dts(g.result.traces)
    assert len(dt) > 1000
    # planted interval hist: 90% in the bin around 2 s, 10% around 5 s -> median ~2, p95 <= 10
    assert 1.0 <= np.median(dt) <= 3.5
    assert np.quantile(dt, 0.95) <= 10.0


def test_render_interval_heterogeneity_across_users(clean_measurements, prov):
    # a 50/50 per-user interval histogram (1-2 s vs 13-16 s users) must yield both kinds of user
    ms = dict(clean_measurements)
    ms[S2.SENSOR_INTERVAL_USER] = _measurement(
        S2.SENSOR_INTERVAL_USER, {_bin_containing(S2.INTERVAL_S_EDGES, 2.0): 50.0,
                                  _bin_containing(S2.INTERVAL_S_EDGES, 14.0): 50.0})
    g = generate_v2(_params_with_sensor(ms), prov, 12, 3, gen=GEN)
    meds = [float(np.median(np.diff([p.t for p in pts])))
            for pts in g.result.traces.values() if len(pts) > 50]
    assert any(m <= 4 for m in meds) and any(m >= 8 for m in meds), meds


def test_render_hists_can_be_disabled(clean_measurements, prov):
    gen = {**GEN, "use_sensor_hists": False, "sample_interval_s": 20}
    g = generate_v2(_params_with_sensor(clean_measurements), prov, 6, 3, gen=gen)
    dt = _dts(g.result.traces)
    assert 15.0 <= np.median(dt) <= 25.0                      # the explicit knob wins


def test_pipeline_injects_measured_speeds(clean_measurements, prov):
    # the artifact generation dict of learner v2 carries no explicit speed/noise knobs, so the
    # measured values are injected; fake_generation_v2 sets them explicitly, so strip them here
    gen = {k: v for k, v in GEN.items() if k not in ("speed_ms_by_mode", "sensor_noise_m")}
    g = generate_v2(_params_with_sensor(clean_measurements), prov, 6, 3, gen=gen)
    sp = g.population.gen["speed_ms_by_mode"]
    sens = g.params["sensor"]["speed_ms_by_mode"]
    assert sp["walk"] == sens["walk"] and sp["car"] == sens["car"]
    from geolife.outside.population_v2 import DEFAULT_GEN_V2
    assert sp["train"] == DEFAULT_GEN_V2["speed_ms_by_mode"]["train"]   # unmeasured mode keeps the default
    # en-route noise keeps the hand-set default; an explicit caller override still wins
    g2 = generate_v2(_params_with_sensor(clean_measurements), prov, 4, 3,
                     gen={**GEN, "sensor_noise_m": 2.5})
    assert g2.population.gen["sensor_noise_m"] == 2.5

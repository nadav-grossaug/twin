"""compare.py / viz.py: declarative real-vs-synthetic quantities, distances and panels.

Everything runs on small generated fixtures (no real data, offline)."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from geolife.eval import compare as C
from geolife.eval import viz as V
from geolife.transfer.tokenize.outings import OutingsConfig

CFG = OutingsConfig(min_outings=2, min_days=1, min_dur_s=60, min_len_m=100, stop_dwell_s=120)
DAY = 86400
T0 = 1_200_000_000  # 2008-01-10ish, like GeoLife


def _outing_points(rng, t0, lat0, lon0, speed, n=150, dt=4):
    """A walk north at `speed` m/s with a little noise, then a 3-minute stop."""
    t = t0 + np.arange(n) * dt
    lat = lat0 + np.arange(n) * speed * dt / 111_320.0 + rng.normal(0, 3e-6, n)
    lon = lon0 + rng.normal(0, 3e-6, n)
    ts = t[-1] + dt + np.arange(45) * dt
    return (np.r_[t, ts], np.r_[lat, np.full(45, lat[-1]) + rng.normal(0, 5e-6, 45)],
            np.r_[lon, np.full(45, lon[-1]) + rng.normal(0, 5e-6, 45)])


def make_bundle(label="real", n_users=3, seed=1, speed=1.4, n_outings=4) -> C.TraceBundle:
    rng = np.random.default_rng(seed)
    arrays = {}
    for u in range(n_users):
        ts, las, los = [], [], []
        for k in range(n_outings):
            t, la, lo = _outing_points(rng, T0 + u * 11 + k * (DAY // 2 + 3 * 3600),
                                       39.90 + 0.01 * u, 116.30 + 0.01 * k, speed)
            ts.append(t), las.append(la), los.append(lo)
        t, la, lo = np.concatenate(ts), np.concatenate(las), np.concatenate(los)
        arrays[f"u{u}"] = (t.astype(np.int64), la, lo, np.zeros(len(t), np.int64))
    return C.tokenize_bundle(C.TraceBundle(label, arrays), None, CFG)


@pytest.fixture(scope="module")
def real():
    return make_bundle("real", seed=1)


@pytest.fixture(scope="module")
def same():
    return make_bundle("syn-same", seed=1)


@pytest.fixture(scope="module")
def fast():
    return make_bundle("syn-fast", seed=1, speed=4.0)


def test_tokenize_bundle_builds_outings(real):
    assert len(real.outings) == 3
    assert all(len(u.outings) == 4 for u in real.outings)


def test_identical_data_has_zero_distances(real, same):
    df = C.summary_table(real, same)
    assert set(df.quantity) == {q.name for q in C.QUANTITIES}
    num = df[(df.kind == "numeric") & (df.n_real > 0)]
    assert len(num) >= 10
    assert (num.ks.fillna(0) == 0).all()
    assert (num.wasserstein.fillna(0) == 0).all()
    assert (num.rel_err_mean.fillna(0) < 1e-12).all()
    cat = df[(df.kind == "categorical") & (df.n_real > 0)]
    assert (cat.tv.fillna(0) == 0).all()


def test_speed_shift_is_detected(real, fast):
    df = C.summary_table(real, fast).set_index("quantity")
    assert df.loc["speed", "ks"] > 0.3
    assert df.loc["speed", "syn_mean"] > df.loc["speed", "real_mean"]
    assert df.loc["step_len", "wasserstein"] > 1.0  # same dt, longer steps
    # unaffected quantities stay close
    assert df.loc["outings_per_user", "rel_err_mean"] < 1e-12


def test_moments_and_quantiles():
    m = C.moments(np.arange(101, dtype=float))
    assert m["mean"] == 50 and m["p50"] == 50 and m["p1"] == 1 and m["p99"] == 99
    assert abs(m["skew"]) < 1e-12
    empty = C.moments(np.array([]))
    assert np.isnan(empty["mean"]) and np.isnan(empty["p95"])


def test_adding_a_quantity_is_one_line(real, same):
    qs = [*C.QUANTITIES, C.Quantity("total_points", "user",
                                    lambda b: np.array([sum(len(a[0]) for a in b.arrays.values())]))]
    df = C.summary_table(real, same, qs)
    assert "total_points" in set(df.quantity)
    assert df.set_index("quantity").loc["total_points", "real_mean"] > 0


def test_categorical_tv_bounds(real, fast):
    df = C.summary_table(real, fast)
    cat = df[df.kind == "categorical"]
    ok = cat.tv.dropna()
    assert ((ok >= 0) & (ok <= 1)).all()


def test_split_bundle_halves_outings(real):
    a, b = C.split_bundle(real, "random", seed=0)
    na = {u.user: len(u.outings) for u in a.outings}
    nb = {u.user: len(u.outings) for u in b.outings}
    orig = {u.user: len(u.outings) for u in real.outings}
    assert all(na[u] + nb[u] == orig[u] for u in orig)
    assert not a.arrays and not b.arrays  # outing-level only


def test_bundle_from_synthetic_reads_traces(tmp_path):
    t = np.arange(0, 400, 4, dtype=np.int64) + T0
    df = pd.DataFrame({"uid": ["a"] * len(t), "t": t,
                       "lat": np.linspace(39.9, 39.91, len(t)), "lon": np.full(len(t), 116.3)})
    (tmp_path / "v_seed0").mkdir()
    df.to_parquet(tmp_path / "v_seed0" / "traces.parquet", index=False)
    b = C.bundle_from_synthetic(tmp_path / "v_seed0", cfg=CFG)
    assert set(b.arrays) == {"a"}
    assert len(b.steps()["a"]["dt"]) > 0


def test_distances_on_disjoint_samples():
    d = C.distances(np.zeros(100), np.ones(100))
    assert d["ks"] == 1.0 and d["tv"] == 1.0 and abs(d["wasserstein"] - 1.0) < 1e-12


def test_viz_panels_and_save_all(tmp_path, real, fast):
    syns = {"syn": fast}
    for name in ("density_maps", "marginal_histograms", "example_trajectories", "hour_profiles",
                 "timebin_shares", "structure_panels"):
        fig = getattr(V, name)(real, syns)
        assert fig is not None
        matplotlib.pyplot.close(fig)
    written = V.save_all(real, fast, tmp_path / "viz")
    for name in V.PANELS:
        assert written[name].is_file() and written[name].stat().st_size > 0
    assert written["summary_csv"].is_file() and written["summary_md"].is_file()
    md = written["summary_md"].read_text(encoding="utf-8")
    assert "| quantity |" in md and "speed" in md


def test_shared_frame_detection(real, fast):
    assert V.shared_frame(real, fast)
    far = make_bundle("far", seed=2)
    for uid, (t, la, lo, sg) in far.arrays.items():
        far.arrays[uid] = (t, la + 10.0, lo + 10.0, sg)
    far._cache.clear()
    assert not V.shared_frame(real, far)

"""Window sampler and trace featurizer for the real-vs-fake classifier (docs/realfake/geolife.md).

Everything runs on tiny in-process random-walk traces; no real data, offline and fast. The two
invariance tests are the leak guards the design hangs on: translating all coordinates or shifting
all timestamps by whole weeks must leave the feature vector unchanged (no absolute lon/lat, no
absolute dates reach the classifier).
"""

from __future__ import annotations

import numpy as np

from geolife.eval.discriminator import (
    TraceFeaturizer,
    TraceSample,
    WindowSampler,
    _refine_seg,
)
from geolife.transfer.geo import from_xy
from realfake.dataset import build_dataset
from realfake.split import group_split
from realfake.types import Sample

T0 = 1_600_000_000  # arbitrary epoch base; tests never depend on it


def toy_arrays(n_users: int = 6, n_pts: int = 400, dt_s: int = 2, speed: float = 1.5,
               seed: int = 0, gap: bool = True, trajs: bool = True) -> dict:
    """Random-walk traces near Beijing with a planted trajectory boundary and a > 2 h gap."""
    rng = np.random.default_rng(seed)
    out = {}
    for u in range(n_users):
        t = np.arange(n_pts, dtype=np.int64) * dt_s + T0 + u * 10_000_000
        if gap:
            t[n_pts // 2:] += 9000  # > 2 h: splits the stream in the middle
        ang = np.cumsum(rng.normal(0.0, 0.2, n_pts))
        x = np.cumsum(np.cos(ang)) * speed * dt_s
        y = np.cumsum(np.sin(ang)) * speed * dt_s
        lat, lon = from_xy(x, y)
        seg = np.zeros(n_pts, np.int64)
        if trajs:
            seg[n_pts // 4:] = 1  # a trajectory (file) boundary
        out[f"u{u:03d}"] = (t, 39.95 + lat, 116.35 + lon, _refine_seg(t, seg))
    return out


def _window(n: int = 60, dt_s: int = 2, speed: float = 1.5, seed: int = 3, t0: int = T0,
            dlat: float = 0.0, dlon: float = 0.0) -> Sample:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * dt_s + t0
    ang = np.cumsum(rng.normal(0.0, 0.3, n))
    lat, lon = from_xy(np.cumsum(np.cos(ang)) * speed * dt_s, np.cumsum(np.sin(ang)) * speed * dt_s)
    return Sample(payload=TraceSample(t=t, lat=39.95 + lat + dlat, lon=116.35 + lon + dlon, k=n),
                  group_id="u", meta={"size": float(n)})


# ------------------------------------------------------------------------------------------ sampler

def _runs_of(seg: np.ndarray) -> list[tuple[int, int]]:
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(seg)]
    return list(zip(starts.tolist(), ends.tolist()))


def test_sampler_determinism_and_validity():
    arrays = toy_arrays()
    sampler = WindowSampler(k_min=10, k_max=50)
    s1 = sampler(arrays, np.random.default_rng(7))
    s2 = WindowSampler(k_min=10, k_max=50)(arrays, np.random.default_rng(7))
    assert s1.group_id == s2.group_id and s1.meta == s2.meta
    for f in ("t", "lat", "lon"):
        assert np.array_equal(getattr(s1.payload, f), getattr(s2.payload, f))

    rng = np.random.default_rng(1)
    for _ in range(40):
        s = sampler(arrays, rng)
        p = s.payload
        assert 10 <= p.k <= 50 and len(p.t) == p.k
        assert (np.diff(p.t) > 0).all()
        assert np.diff(p.t).max() <= 7200
        # the window is a contiguous slice of its user's arrays, inside one run
        t_full, _, _, seg = arrays[s.group_id]
        a = int(np.searchsorted(t_full, p.t[0]))
        assert np.array_equal(t_full[a : a + p.k], p.t)
        assert any(r0 <= a and a + p.k <= r1 for r0, r1 in _runs_of(seg))
        assert s.meta["size"] == float(p.k)


def test_sampler_uniform_over_users():
    arrays = {**toy_arrays(n_users=1, n_pts=4000, seed=0),
              "heavy": toy_arrays(n_users=1, n_pts=200, seed=1)["u000"],
              "small": toy_arrays(n_users=1, n_pts=200, seed=2)["u000"]}
    sampler = WindowSampler(k_min=10, k_max=40)
    rng = np.random.default_rng(5)
    counts = {u: 0 for u in arrays}
    n = 1500
    for _ in range(n):
        counts[sampler(arrays, rng).group_id] += 1
    for u, c in counts.items():  # a 20x heavier user must not dominate: uniform over users
        assert abs(c / n - 1 / 3) < 0.08, (u, c / n)


def test_user_disjoint_split_end_to_end():
    feat = TraceFeaturizer(include_map=False)
    sampler = WindowSampler(k_min=10, k_max=40)
    ds = build_dataset(toy_arrays(seed=0), [("seed0", toy_arrays(seed=9))], sampler, feat, n=60, seed=0)
    train, test = group_split(ds.groups, test_frac=0.25, seed=0, y=ds.y)
    assert not (set(ds.groups[train]) & set(ds.groups[test]))
    for mask in (train, test):  # both sides keep both labels
        assert {0, 1} <= set(ds.y[mask].tolist())


# --------------------------------------------------------------------------------------- featurizer

def test_featurizer_shape_groups_and_degenerate_windows():
    feat = TraceFeaturizer(include_map=False)
    v = feat(_window())
    assert v.shape == (len(feat.feature_names),)
    assert sorted(f for names in feat.feature_groups.values() for f in names) == sorted(feat.feature_names)
    assert np.isfinite(v).all()

    two = Sample(payload=TraceSample(t=np.array([T0, T0 + 2], np.int64),
                                     lat=np.array([39.95, 39.95]), lon=np.array([116.35, 116.35]), k=2),
                 group_id="u", meta={"size": 2.0})
    parked = Sample(payload=TraceSample(t=np.arange(50, dtype=np.int64) * 2 + T0,
                                        lat=np.full(50, 39.95), lon=np.full(50, 116.35), k=50),
                    group_id="u", meta={"size": 50.0})
    for s in (two, parked):
        w = feat(s)
        assert w.shape == (len(feat.feature_names),) and np.isfinite(w).all()


def test_featurizer_translation_invariance():
    """No absolute lon/lat reaches the vector (map group off): translating the trace changes nothing."""
    feat = TraceFeaturizer(include_map=False)
    v1 = feat(_window(seed=11))
    v2 = feat(_window(seed=11, dlat=0.13, dlon=0.21))
    assert np.allclose(v1, v2)


def test_featurizer_date_invariance():
    """No absolute date reaches the vector: whole-week shifts of t change nothing; a 6 h shift moves
    only the time group."""
    feat = TraceFeaturizer(include_map=False)
    v = feat(_window(seed=12))
    assert np.array_equal(v, feat(_window(seed=12, t0=T0 + 7 * 86400)))          # one week
    assert np.array_equal(v, feat(_window(seed=12, t0=T0 + 52 * 7 * 86400)))     # ~one year
    v6 = feat(_window(seed=12, t0=T0 + 6 * 3600))
    names = feat.feature_names
    time_cols = [i for i, f in enumerate(names) if f.startswith("time:")]
    other = [i for i, f in enumerate(names) if not f.startswith("time:")]
    assert np.array_equal(v[other], v6[other])
    assert not np.allclose(v[time_cols], v6[time_cols])


def test_map_group_toggle_and_response():
    import h3
    feat_off = TraceFeaturizer(include_map=False)
    assert not any(f.startswith("map:") for f in feat_off.feature_names)

    la, lo = 39.95, 116.35
    road = frozenset([h3.latlng_to_cell(la, lo, 9)])
    feat = TraceFeaturizer(places_latlon=np.array([[la, lo]]), road_cells=road, include_map=True)
    assert any(f.startswith("map:") for f in feat.feature_names)

    def still(lat, lon):
        return Sample(payload=TraceSample(t=np.arange(40, dtype=np.int64) * 2 + T0,
                                          lat=np.full(40, lat), lon=np.full(40, lon), k=40),
                      group_id="u", meta={"size": 40.0})

    on = feat(still(la, lo))
    off = feat(still(39.99, 116.50))
    i = feat.feature_names.index("map:on_road_cell_share")
    j = feat.feature_names.index("map:near_place_100m_share")
    assert on[i] == 1.0 and off[i] == 0.0
    assert on[j] == 1.0 and off[j] == 0.0
    assert np.isfinite(on).all() and np.isfinite(off).all()


def test_thinned_kinematics_separate_noise_scales():
    """2 s zig-zag noise moves the raw turn statistics a lot and the 30 s-thinned ones barely."""
    n, dt_s, vx = 240, 2, 5.0
    t = np.arange(n, dtype=np.int64) * dt_s + T0
    x = np.arange(n, dtype=float) * vx * dt_s
    y0 = np.zeros(n)
    y1 = np.where(np.arange(n) % 2 == 0, 8.0, -8.0)  # +-8 m perpendicular zig-zag

    def sample(y):
        lat, lon = from_xy(x, y)
        return Sample(payload=TraceSample(t=t, lat=39.95 + lat, lon=116.35 + lon, k=n),
                      group_id="u", meta={"size": float(n)})

    feat = TraceFeaturizer(include_map=False)
    smooth, noisy = feat(sample(y0)), feat(sample(y1))
    i_raw = feat.feature_names.index("kinematics:raw_turn_abs_mean")
    i_thin = feat.feature_names.index("kinematics:thin30_turn_abs_mean")
    assert noisy[i_raw] - smooth[i_raw] > 50.0
    assert abs(noisy[i_thin] - smooth[i_thin]) < 15.0


def test_feature_subset_drops_speed_distance():
    from fnmatch import fnmatchcase

    from geolife.eval.discriminator import SPEED_DISTANCE_FEATURES, FeatureSubset
    base = TraceFeaturizer(include_map=False)
    sub = FeatureSubset(base, SPEED_DISTANCE_FEATURES)
    # no speed / distance feature survives; heading, shape ratios, sampling and time do
    assert not any(fnmatchcase(f, p) for f in sub.feature_names for p in SPEED_DISTANCE_FEATURES)
    assert "kinematics:raw_turn_cos_mean" in sub.feature_names
    assert "shape:tortuosity" in sub.feature_names and "sampling:dt_mode_share" in sub.feature_names
    assert sorted(sum(sub.feature_groups.values(), [])) == sorted(sub.feature_names)
    assert set(sub.dropped) | set(sub.feature_names) == set(base.feature_names)
    # the subset vector is exactly the base vector's kept columns
    s = _window()
    full, part = base(s), sub(s)
    assert len(part) == len(sub.feature_names)
    np.testing.assert_array_equal(part, full[[base.feature_names.index(f) for f in sub.feature_names]])

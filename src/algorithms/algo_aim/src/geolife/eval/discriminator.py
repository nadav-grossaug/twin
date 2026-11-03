"""GeoLife side of the real-vs-fake classifier: window sampler and trace featurizer
(``docs/realfake/geolife.md``).

One sample is a short contiguous window of one user's GPS trace: ``k`` consecutive points
(``k`` drawn per sample log-uniformly from ``[k_min, k_max]``) that never cross a trajectory
boundary or a gap above 2 h. The user is drawn uniformly first, then the window uniformly over the
user's valid window starts, so per-user point volume cannot leak into the labels. The payload is
the window's raw ``(t, lat, lon)`` arrays; the group id is the user, so the group-aware split puts
whole users on one side of train/test.

What is deliberately dropped (``docs/realfake/geolife.md`` section 2): user ids and trajectory ids
(not in the payload), absolute dates (real GeoLife is 2007-2012, synthetic epochs are 2024; the
featurizer reads only local time-of-day and day-of-week, both invariant to whole-week shifts of
``t``), absolute lon/lat (all geometry is computed from ``to_xy`` differences or centred
coordinates, so it is invariant to translating the whole trace), and per-user volume (uniform
over users + group split). The optional ``map`` feature group is the single exception on
coordinates: it compares points against the PUBLIC places table and road graph; it is a separate
group precisely so its contribution is visible and removable (``include_map=False``).

Features come in six groups for grouped permutation importance: ``sampling``, ``kinematics``
(duplicated at raw and 30 s-thinned scale: raw ~2 s steps are GPS-noise dominated, see PROCESS.md
D20a), ``shape``, ``stops``, ``time`` and optionally ``map``. Undefined statistics (too few steps,
zero path, no stays) are 0.0, never NaN, so the vector length is always ``len(feature_names)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd

from geolife.transfer.geo import detect_stays, to_xy
from realfake.types import Sample

#: lon/lat bbox (min_lon, min_lat, max_lon, max_lat) of the Beijing core frame (= compare.BEIJING_LONLAT).
BEIJING_LONLAT = (116.15, 39.75, 116.60, 40.10)
GAP_S = 7200          # above this a run (and the tokenizer's outing) ends; windows never cross it
LOCAL_TZ_H = 8        # local time for the time features (UTC+8, public setting)

_STEP_MOVE_M = 5.0    # heading statistics use steps of at least this length (as compare.steps())
_RAW_MAX_DT_S = 60    # raw kinematics use steps at most this long (as compare.STEP_DT_S)
_THIN_MAX_DT_S = 240  # thinned kinematics allow the thinned ~30 s steps plus short gaps
_SLOW_MS = 0.3        # below this a thinned step counts as "not moving" (stops group)
_QS = (25, 50, 75, 90)


# ---------------------------------------------------------------------------------------- loaders

def _refine_seg(t: np.ndarray, seg: np.ndarray, gap_s: int = GAP_S) -> np.ndarray:
    """Split runs additionally at gaps above ``gap_s`` and at non-increasing timestamps, so a run
    is strictly increasing in ``t`` with every within-run step at most ``gap_s`` long."""
    if len(t) == 0:
        return seg.astype(np.int64)
    dt = np.diff(t)
    new = np.r_[True, (seg[1:] != seg[:-1]) | (dt > gap_s) | (dt <= 0)]
    return np.cumsum(new).astype(np.int64) - 1


def load_real_arrays(path, bbox_lonlat=BEIJING_LONLAT, max_users: int | None = None,
                     ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Real points.parquet (uid, traj, t, lat, lon) -> ``{uid: (t, lat, lon, seg)}``, restricted to
    ``bbox_lonlat``. ``seg`` starts from the trajectory (file) id and is refined by the same > 2 h
    gap rule used for the synthetic side, so the window rule is identical on both labels."""
    df = pd.read_parquet(path, columns=["uid", "traj", "t", "lat", "lon"])
    if bbox_lonlat is not None:
        x0, y0, x1, y1 = bbox_lonlat
        df = df[(df.lon >= x0) & (df.lon <= x1) & (df.lat >= y0) & (df.lat <= y1)]
    df = df.sort_values(["uid", "traj", "t"], kind="stable")
    arrays: dict[str, tuple] = {}
    for uid, g in df.groupby("uid", sort=True):
        t = g.t.to_numpy(np.int64)
        if len(t) < 2:
            continue
        seg = pd.factorize(g.traj.to_numpy())[0].astype(np.int64)
        arrays[str(uid)] = (t, g.lat.to_numpy(float), g.lon.to_numpy(float), _refine_seg(t, seg))
        if max_users and len(arrays) >= max_users:
            break
    del df
    return arrays


def load_synthetic_arrays(variant_dir, max_users: int | None = None,
                          ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Stage-2 ``traces.parquet`` (uid, t, lat, lon) -> ``{uid: (t, lat, lon, seg)}``; one stream
    per user, runs split by the same > 2 h gap rule (as ``compare.bundle_from_synthetic``)."""
    df = pd.read_parquet(Path(variant_dir) / "traces.parquet", columns=["uid", "t", "lat", "lon"])
    df = df.sort_values(["uid", "t"], kind="stable")
    arrays: dict[str, tuple] = {}
    for uid, g in df.groupby("uid", sort=True):
        t = g.t.to_numpy(np.int64)
        if len(t) < 2:
            continue
        seg = np.zeros(len(t), dtype=np.int64)
        arrays[str(uid)] = (t, g.lat.to_numpy(float), g.lon.to_numpy(float), _refine_seg(t, seg))
        if max_users and len(arrays) >= max_users:
            break
    del df
    return arrays


# ---------------------------------------------------------------------------------------- sampler

@dataclass
class TraceSample:
    """Payload of one window sample: the window's raw arrays, nothing of the user or the calendar
    position beyond ``t`` itself (the featurizer reads only time-of-day / day-of-week from it)."""

    t: np.ndarray    # int64 (k,), strictly increasing, within one run
    lat: np.ndarray  # float64 (k,)
    lon: np.ndarray  # float64 (k,)
    k: int


class WindowSampler:
    """``Sampler`` for trace arrays: one contiguous ``k``-point window per call.

    ``k`` is drawn log-uniformly from ``[k_min, min(k_max, longest run of the drawn user)]``. The
    user is drawn uniformly over the users whose longest run holds at least ``k_min`` points; the
    window start is then uniform over the user's valid starts (runs shorter than ``k`` contribute
    none), so windows never cross a run boundary. Deterministic given the ``rng`` state. A source
    (``{uid: (t, lat, lon, seg)}``) is prepared once and cached by object identity, with a strong
    reference so ids cannot be reused; a plain call works without any preparation.
    """

    def __init__(self, k_min: int = 30, k_max: int = 600):
        if not 2 <= k_min <= k_max:
            raise ValueError(f"need 2 <= k_min <= k_max, got [{k_min}, {k_max}]")
        self.k_min, self.k_max = int(k_min), int(k_max)
        self._cache: dict[int, tuple[dict, tuple[list[str], dict, dict]]] = {}

    def prepare(self, source: dict) -> tuple[list[str], dict[str, np.ndarray], dict[str, int]]:
        """``(eligible uids, {uid: (n_runs, 2) start/end rows}, {uid: longest run})``, cached."""
        hit = self._cache.get(id(source))
        if hit is not None and hit[0] is source:
            return hit[1]
        uids: list[str] = []
        runs: dict[str, np.ndarray] = {}
        longest: dict[str, int] = {}
        for uid in sorted(source):
            seg = source[uid][3]
            if len(seg) == 0:
                continue
            cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, len(seg)]
            lmax = int((ends - starts).max())
            if lmax < self.k_min:
                continue
            uids.append(uid)
            runs[uid] = np.stack([starts, ends], axis=1)
            longest[uid] = lmax
        if not uids:
            raise ValueError(f"no user has a run of at least k_min={self.k_min} points")
        entry = (uids, runs, longest)
        self._cache[id(source)] = (source, entry)
        return entry

    def __call__(self, source: dict, rng: np.random.Generator) -> Sample:
        uids, runs, longest = self.prepare(source)
        uid = uids[int(rng.integers(len(uids)))]
        k_hi = min(self.k_max, longest[uid])
        k_lo = min(self.k_min, k_hi)
        k = int(np.clip(np.rint(np.exp(rng.uniform(np.log(k_lo), np.log(k_hi)))), k_lo, k_hi))

        r = runs[uid]
        lens = r[:, 1] - r[:, 0]
        starts_per_run = np.maximum(lens - k + 1, 0)
        pos = int(rng.integers(starts_per_run.sum()))
        run_i = int(np.searchsorted(np.cumsum(starts_per_run), pos, side="right"))
        a = int(r[run_i, 0] + pos - int(np.r_[0, np.cumsum(starts_per_run)][run_i]))

        t, lat, lon, _ = source[uid]
        w = slice(a, a + k)
        payload = TraceSample(t=t[w].copy(), lat=lat[w].copy(), lon=lon[w].copy(), k=k)
        dur_min = float(t[a + k - 1] - t[a]) / 60.0
        return Sample(payload=payload, group_id=uid, meta={"size": float(k), "dur_min": dur_min})


# ------------------------------------------------------------------------------------- featurizer

def _q(v: np.ndarray, qs) -> list[float]:
    if len(v) == 0:
        return [0.0] * len(qs)
    return [float(np.percentile(v, q)) for q in qs]


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return ((a + 180.0) % 360.0) - 180.0


def _thin_index(t: np.ndarray, thin_s: int) -> np.ndarray:
    """First point of each ``thin_s`` bucket (the ``compare.stay_jitter`` thinning rule)."""
    key = t // thin_s
    return np.flatnonzero(np.r_[True, key[1:] != key[:-1]])


def _kin_block(t: np.ndarray, x: np.ndarray, y: np.ndarray, max_dt_s: float) -> list[float]:
    """The 13 kinematics numbers of one scale: steps of at most ``max_dt_s`` between consecutive
    points; headings only between consecutive moving (>= 5 m) steps."""
    if len(t) < 2:
        return [0.0] * 13
    dt = np.diff(t).astype(float)
    dx, dy = np.diff(x), np.diff(y)
    d = np.hypot(dx, dy)
    ok = dt <= max_dt_s  # dt > 0 within a run by construction
    vals = _q(d[ok], (25, 50, 90))
    speed = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
    sp = speed[ok]
    vals += _q(sp, _QS)
    vals.append(float(np.log1p(sp).std()) if len(sp) else 0.0)
    pair = ok[1:] & ok[:-1]
    acc = np.abs(np.diff(speed)) / np.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-9)
    vals += _q(acc[pair], (50, 90))
    moving = ok & (d >= _STEP_MOVE_M)
    mpair = moving[1:] & moving[:-1]
    ang = np.degrees(np.arctan2(dy, dx))
    dh = _wrap_deg(ang[1:] - ang[:-1])[mpair]
    vals.append(float(np.abs(dh).mean()) if len(dh) else 0.0)
    vals += _q(np.abs(dh), (90,))
    vals.append(float(np.cos(np.radians(dh)).mean()) if len(dh) else 0.0)
    return vals


class TraceFeaturizer:
    """``Featurizer`` for trace windows; every feature is listed in ``docs/realfake/geolife.md``.

    All geometry is computed from ``to_xy`` differences or centred coordinates (``to_xy`` is a
    fixed-origin linear projection, so translating the whole trace leaves the vector unchanged);
    the time group reads only local time-of-day and day-of-week, so shifting ``t`` by whole weeks
    leaves the vector unchanged. The ``map`` group is the only reader of absolute coordinates
    (public places table + public road-graph cells) and is omitted entirely when
    ``include_map=False``. ``road_cells`` are res-9 H3 cells covering the public road graph; the
    1-ring dilated set is built once here.
    """

    def __init__(self, places_latlon: np.ndarray | None = None, road_cells: frozenset | None = None,
                 include_map: bool = True, tz_offset_h: int = LOCAL_TZ_H, thin_s: int = 30):
        self.include_map = bool(include_map)
        self.tz_offset_h = int(tz_offset_h)
        self.thin_s = int(thin_s)
        self._tree = None
        self._road = None
        self._road_dilated = None
        if self.include_map and places_latlon is not None and len(places_latlon):
            from scipy.spatial import cKDTree
            p = np.asarray(places_latlon, dtype=float)
            px, py = to_xy(p[:, 0], p[:, 1])
            self._tree = cKDTree(np.stack([px, py], axis=1))
        if self.include_map and road_cells:
            import h3
            self._road = frozenset(road_cells)
            dil = set(self._road)
            for c in self._road:
                dil.update(h3.grid_disk(c, 1))
            self._road_dilated = frozenset(dil)

        kin = ["step_q25", "step_q50", "step_q90", "speed_q25", "speed_q50", "speed_q75",
               "speed_q90", "speed_log_std", "accel_abs_q50", "accel_abs_q90",
               "turn_abs_mean", "turn_abs_q90", "turn_cos_mean"]
        self.feature_groups: dict[str, list[str]] = {
            "sampling": (
                [f"sampling:dt_q{q}" for q in (10, 25, 50, 75, 90)]
                + ["sampling:dt_log_mean", "sampling:dt_log_std", "sampling:dt_mode_share",
                   "sampling:dt_le2_share", "sampling:dt_gt60_share", "sampling:dt_gt300_share"]
            ),
            "kinematics": [f"kinematics:raw_{f}" for f in kin] + [f"kinematics:thin{self.thin_s}_{f}" for f in kin],
            "shape": ["shape:tortuosity", "shape:rgyr_over_path", "shape:pc1_extent_over_path",
                      "shape:pc2_extent_over_path", "shape:anisotropy", "shape:reversals_per_km",
                      "shape:net_over_rgyr"],
            "stops": ["stops:slow_share", "stops:slow_time_share", "stops:stay_count_per_h",
                      "stops:stay_time_share", "stops:jitter_rms_log"],
            "time": (
                [f"time:h{3 * i:02d}_{3 * (i + 1):02d}" for i in range(8)]
                + ["time:weekend_share", "time:hour_sin", "time:hour_cos"]
            ),
        }
        if self.include_map:
            self.feature_groups["map"] = [
                "map:place_dist_q10", "map:place_dist_q50", "map:place_dist_q90",
                "map:near_place_100m_share", "map:near_place_250m_share",
                "map:on_road_cell_share", "map:on_road_dilated_share",
            ]
        self.feature_names: list[str] = [f for names in self.feature_groups.values() for f in names]

    def __call__(self, sample: Sample) -> np.ndarray:
        p: TraceSample = sample.payload
        t = np.asarray(p.t, dtype=np.int64)
        x, y = to_xy(np.asarray(p.lat, dtype=float), np.asarray(p.lon, dtype=float))
        dt = np.diff(t).astype(float)

        # sampling
        vals = _q(dt, (10, 25, 50, 75, 90))
        if len(dt):
            ld = np.log1p(dt)
            vals += [float(ld.mean()), float(ld.std())]
            r = np.rint(dt).astype(np.int64)
            vals.append(float(np.bincount(r - r.min()).max() / len(r)))
            vals += [float((dt <= 2).mean()), float((dt > 60).mean()), float((dt > 300).mean())]
        else:
            vals += [0.0] * 6

        # kinematics: raw and thinned sub-blocks
        vals += _kin_block(t, x, y, _RAW_MAX_DT_S)
        ti = _thin_index(t, self.thin_s)
        vals += _kin_block(t[ti], x[ti], y[ti], _THIN_MAX_DT_S)

        # shape (centred coordinates; rotation-invariant via covariance eigenvalues)
        d_all = np.hypot(np.diff(x), np.diff(y))
        path = float(d_all.sum())
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(x) else 0.0
        xc, yc = x - x.mean(), y - y.mean()
        rgyr = float(np.sqrt((xc ** 2 + yc ** 2).mean()))
        cov = np.cov(np.stack([xc, yc])) if len(x) > 1 else np.zeros((2, 2))
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))  # ascending
        pc1, pc2 = float(np.sqrt(ev[1])), float(np.sqrt(ev[0]))
        dxm, dym = np.diff(x), np.diff(y)
        moving = (dt <= _RAW_MAX_DT_S) & (d_all >= _STEP_MOVE_M)
        mpair = moving[1:] & moving[:-1]
        ang = np.degrees(np.arctan2(dym, dxm))
        dh = np.abs(_wrap_deg(ang[1:] - ang[:-1])[mpair])
        vals += [
            net / path if path > 0 else 0.0,
            rgyr / path if path > 0 else 0.0,
            pc1 / path if path > 0 else 0.0,
            pc2 / path if path > 0 else 0.0,
            pc2 / pc1 if pc1 > 0 else 0.0,
            float((dh > 150.0).sum()) / (path / 1000.0) if path > 0 else 0.0,
            net / rgyr if rgyr > 0 else 0.0,
        ]

        # stops (thinned points; single-run window -> seg of zeros)
        tt, xt, yt = t[ti], x[ti], y[ti]
        dur = float(t[-1] - t[0])
        if len(tt) >= 2:
            dtt = np.diff(tt).astype(float)
            dts = np.hypot(np.diff(xt), np.diff(yt))
            sp = np.divide(dts, dtt, out=np.zeros_like(dts), where=dtt > 0)
            slow = sp < _SLOW_MS
            vals += [float(slow.mean()), float(dtt[slow].sum() / dtt.sum()) if dtt.sum() > 0 else 0.0]
            i0, i1, _, _, rms = detect_stays(xt, yt, tt.astype(float), np.zeros(len(tt), np.int64), 200.0, 600.0)
            vals += [
                float(len(i0)) / (dur / 3600.0) if dur > 0 else 0.0,
                float((tt[i1] - tt[i0]).sum() / dur) if dur > 0 and len(i0) else 0.0,
                float(np.log1p(np.mean(rms))) if len(rms) else 0.0,
            ]
        else:
            vals += [0.0] * 5

        # time (local time-of-day and day-of-week only; invariant to whole-week shifts of t)
        tz = self.tz_offset_h * 3600
        h = ((t + tz) % 86400) / 3600.0
        b = np.minimum((h // 3.0).astype(np.int64), 7)
        vals += (np.bincount(b, minlength=8) / len(t)).tolist()
        day = ((t + tz) // 86400 + 3) % 7  # epoch day 0 = Thursday
        vals.append(float((day >= 5).mean()))
        vals += [float(np.sin(2 * np.pi * h / 24.0).mean()), float(np.cos(2 * np.pi * h / 24.0).mean())]

        # map (optional; thinned points against the public places table and road cells)
        if self.include_map:
            if self._tree is not None and len(ti):
                dist, _ = self._tree.query(np.stack([xt, yt], axis=1))
                vals += [float(np.log1p(np.percentile(dist, q))) for q in (10, 50, 90)]
                vals += [float((dist <= 100.0).mean()), float((dist <= 250.0).mean())]
            else:
                vals += [0.0] * 5
            if self._road is not None and len(ti):
                import h3
                cells = [h3.latlng_to_cell(float(a), float(b_), 9) for a, b_ in zip(p.lat[ti], p.lon[ti])]
                vals.append(float(np.mean([c in self._road for c in cells])))
                vals.append(float(np.mean([c in self._road_dilated for c in cells])))
            else:
                vals += [0.0] * 2

        out = np.asarray(vals, dtype=np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------------------- feature subsets
# Features that measure speed or distance (absolute metres, m/s, or a distance-based threshold):
# out of scope for a run where the modelled sensor regime does not cover speed and distance.
# Kept on purpose: heading/turn angles, scale-free shape ratios (tortuosity, extent/path,
# anisotropy), stay counts and dwell shares, sampling, local time, on-road shares.
SPEED_DISTANCE_FEATURES = (
    "kinematics:*_step_*", "kinematics:*_speed_*", "kinematics:*_accel_*",
    "shape:reversals_per_km",
    "stops:slow_share", "stops:slow_time_share", "stops:jitter_rms_log",
    "map:place_dist_*", "map:near_place_*",
)


class FeatureSubset:
    """``Featurizer`` that keeps the columns of ``base`` not matched by any ``exclude`` pattern
    (fnmatch on the ``group:name`` feature names). Groups left empty are dropped, so grouped
    importance only reports groups that still carry features. The base featurizer is unchanged."""

    def __init__(self, base, exclude=()):
        self.base = base
        self.exclude = tuple(exclude)
        keep = [i for i, f in enumerate(base.feature_names)
                if not any(fnmatchcase(f, p) for p in self.exclude)]
        if not keep:
            raise ValueError("exclude patterns remove every feature")
        self._idx = np.asarray(keep, dtype=np.int64)
        self.feature_names: list[str] = [base.feature_names[i] for i in keep]
        kept = set(self.feature_names)
        self.feature_groups: dict[str, list[str]] = {}
        for g, names in base.feature_groups.items():
            sel = [f for f in names if f in kept]
            if sel:
                self.feature_groups[g] = sel
        self.dropped: list[str] = [f for f in base.feature_names if f not in kept]

    def __call__(self, sample: Sample) -> np.ndarray:
        return self.base(sample)[self._idx]

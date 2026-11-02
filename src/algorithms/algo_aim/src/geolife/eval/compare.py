"""Real-vs-synthetic moments and distance metrics over a declarative quantity list (eval zone).

A `TraceBundle` holds one dataset (real or synthetic) as per-user point arrays plus the tokenized
`UserOutings` (same tokenizer for both sides, so the comparison is apples to apples). Every statistic
is a `Quantity`: a name, a group, and an extractor that maps a bundle to samples. `QUANTITIES` is the
declarative list - adding a statistic is one line. `summary_table` tabulates moments (mean, std,
skewness, kurtosis, quantiles) and distances (KS, Wasserstein, total variation, relative error of
moments) for every quantity; `write_summary` lands it in `geolife/eval/results/` so progress can be
tracked run to run.

Point-level quantities are computed on steps between consecutive points of one trajectory (same file
for the real data), deterministically strided to at most `max_steps_per_user` samples per user so a
25M-point dataset stays cheap. Times are UTC epoch seconds; hours are local (UTC+8).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from geolife.transfer.geo import detect_stays, to_xy
from geolife.transfer.tokenize.outings import MODES, OutingsConfig, PlaceIndex, user_outings_from_arrays
from geolife.transfer.types import LOCAL_UTC_OFFSET_H, TOD_BINS, UserOutings

#: lon/lat bbox (min_lon, min_lat, max_lon, max_lat) of the Beijing core frame (= tokens.BEIJING_CORE reordered).
BEIJING_LONLAT = (116.15, 39.75, 116.60, 40.10)
GAP_S = 7200          # above this a new outing starts (public tokenizer setting)
STEP_DT_S = 60        # a "step" (sampling interval, step length): consecutive points at most this far apart
SPEED_DT_S = 120      # speeds use steps up to this long
LOCAL_S = LOCAL_UTC_OFFSET_H * 3600


# --------------------------------------------------------------------------------------- bundles

@dataclass
class TraceBundle:
    """One dataset: per-user point arrays (t, lat, lon, seg) with `seg` the trajectory/file id per
    point (steps never cross a seg boundary), plus tokenized outings."""
    label: str
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    outings: list[UserOutings] = field(default_factory=list)
    max_steps_per_user: int = 20000
    _cache: dict = field(default_factory=dict, repr=False)

    # -- derived, cached ------------------------------------------------------------------
    def steps(self) -> dict[str, dict[str, np.ndarray]]:
        """Per user: t0 (step start), dt (s), d (m) of within-trajectory steps, and dh (deg) heading
        changes between consecutive valid steps; strided to max_steps_per_user."""
        if "steps" not in self._cache:
            out = {}
            for uid, (t, lat, lon, seg) in self.arrays.items():
                if len(t) < 2:
                    continue
                x, y = to_xy(lat, lon)
                dt = np.diff(t).astype(float)
                ok = (seg[1:] == seg[:-1]) & (dt > 0)
                dx, dy = np.diff(x), np.diff(y)
                d = np.hypot(dx, dy)
                ang = np.degrees(np.arctan2(dy, dx))
                moving = ok & (dt <= STEP_DT_S) & (d >= 5.0)
                pair = moving[1:] & moving[:-1]
                dh = np.abs(((ang[1:] - ang[:-1] + 180.0) % 360.0) - 180.0)[pair]
                k = max(1, int(np.ceil(ok.sum() / self.max_steps_per_user)))
                idx = np.flatnonzero(ok)[::k]
                out[uid] = {"t0": t[:-1][idx], "dt": dt[idx], "d": d[idx], "dh": dh[::k]}
            self._cache["steps"] = out
        return self._cache["steps"]

    def stay_jitter(self) -> np.ndarray:
        """RMS radius (m) of each detected stay (200 m / 10 min rule on 30 s-thinned points)."""
        if "jitter" not in self._cache:
            rms_all = []
            for uid, (t, lat, lon, seg) in self.arrays.items():
                if len(t) < 3:
                    continue
                key = seg.astype(np.int64) * (1 << 36) + t // 30
                th = np.r_[True, key[1:] != key[:-1]]
                x, y = to_xy(lat[th], lon[th])
                r = detect_stays(x, y, t[th].astype(float), seg[th], 200.0, 600.0)[4]
                if len(r):
                    rms_all.append(np.asarray(r, float))
            self._cache["jitter"] = np.concatenate(rms_all) if rms_all else np.array([])
        return self._cache["jitter"]

    def m2(self):
        """Co-visit analysis of the tokenized outings (M v2 `analyze`, no road graph), cached."""
        if "m2" not in self._cache:
            from geolife.eval.m.outings import M2Config, analyze
            self._cache["m2"] = analyze(self.outings, M2Config(), has_graph=False) if self.outings else None
        return self._cache["m2"]

    def centroid(self) -> tuple[float, float]:
        """(lat, lon) mean over users of each user's median position (so heavy users do not dominate)."""
        if "centroid" not in self._cache:
            las = [float(np.median(a[1])) for a in self.arrays.values() if len(a[0])]
            los = [float(np.median(a[2])) for a in self.arrays.values() if len(a[0])]
            self._cache["centroid"] = (float(np.mean(las)), float(np.mean(los))) if las else (float("nan"),) * 2
        return self._cache["centroid"]


def tokenize_bundle(b: TraceBundle, places: pd.DataFrame | None, cfg: OutingsConfig | None = None) -> TraceBundle:
    """Tokenize every user's points into outings (user ids kept so quantities can join arrays)."""
    cfg = cfg or OutingsConfig()
    pidx = PlaceIndex(places)
    b.outings = []
    for uid in sorted(b.arrays):
        t, lat, lon, _ = b.arrays[uid]
        uo = user_outings_from_arrays(uid, t, lat, lon, pidx, cfg)
        if uo is not None:
            b.outings.append(uo)
    return b


def bundle_from_points_parquet(path, label: str = "real", bbox_lonlat=BEIJING_LONLAT,
                               places: pd.DataFrame | None = None, cfg: OutingsConfig | None = None,
                               max_users: int | None = None) -> TraceBundle:
    """Real points.parquet (uid, traj, t, lat, lon) -> TraceBundle, restricted to `bbox_lonlat`."""
    df = pd.read_parquet(path, columns=["uid", "traj", "t", "lat", "lon"])
    if bbox_lonlat is not None:
        x0, y0, x1, y1 = bbox_lonlat
        df = df[(df.lon >= x0) & (df.lon <= x1) & (df.lat >= y0) & (df.lat <= y1)]
    df = df.sort_values(["uid", "traj", "t"], kind="stable")
    arrays = {}
    for uid, g in df.groupby("uid", sort=True):
        seg = pd.factorize(g.traj.to_numpy())[0].astype(np.int64)
        arrays[str(uid)] = (g.t.to_numpy(np.int64), g.lat.to_numpy(float), g.lon.to_numpy(float), seg)
        if max_users and len(arrays) >= max_users:
            break
    return tokenize_bundle(TraceBundle(label, arrays), places, cfg)


def bundle_from_synthetic(syn_dir, label: str | None = None, places: pd.DataFrame | None = None,
                          cfg: OutingsConfig | None = None) -> TraceBundle:
    """Stage-2 output directory -> TraceBundle (traces.parquet; one stream per user, gaps split steps
    through the same `GAP_S` rule the tokenizer uses)."""
    d = Path(syn_dir)
    df = pd.read_parquet(d / "traces.parquet").sort_values(["uid", "t"], kind="stable")
    arrays = {}
    for uid, g in df.groupby("uid", sort=True):
        t = g.t.to_numpy(np.int64)
        seg = np.cumsum(np.r_[0, (np.diff(t) > GAP_S).astype(np.int64)])
        arrays[str(uid)] = (t, g.lat.to_numpy(float), g.lon.to_numpy(float), seg)
    return tokenize_bundle(TraceBundle(label or d.name, arrays), places, cfg)


def split_bundle(b: TraceBundle, how: str = "random", seed: int = 0) -> tuple[TraceBundle, TraceBundle]:
    """Split every user's outings in half (outing-level quantities only: the halves carry no arrays)."""
    from geolife.eval.m.outings_transfer import split_user
    rng = np.random.default_rng(seed)
    pairs = [split_user(u, how, rng) for u in b.outings if len(u.outings) >= 2]
    a = TraceBundle(f"{b.label} (A)", {}, [p[0] for p in pairs], b.max_steps_per_user)
    c = TraceBundle(f"{b.label} (B)", {}, [p[1] for p in pairs], b.max_steps_per_user)
    return a, c


# --------------------------------------------------------------------------------------- extractors

def _steps_concat(b: TraceBundle, f: Callable[[dict], np.ndarray]) -> np.ndarray:
    parts = [f(s) for s in b.steps().values()]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.array([])


def q_sampling_interval(b):
    return _steps_concat(b, lambda s: s["dt"][s["dt"] <= STEP_DT_S])


def q_gap_len(b):
    return _steps_concat(b, lambda s: s["dt"][(s["dt"] > STEP_DT_S) & (s["dt"] <= GAP_S)])


def q_step_len(b):
    return _steps_concat(b, lambda s: s["d"][s["dt"] <= STEP_DT_S])


def q_speed(b):
    return _steps_concat(b, lambda s: (s["d"] / s["dt"])[s["dt"] <= SPEED_DT_S])


def q_heading_change(b):
    return _steps_concat(b, lambda s: s["dh"])


def q_stop_jitter(b):
    return b.stay_jitter()


def q_speed_by_mode(mode: str) -> Callable[[TraceBundle], np.ndarray]:
    def f(b: TraceBundle) -> np.ndarray:
        st, parts = b.steps(), []
        for uo in b.outings:
            s = st.get(uo.user)
            ivs = sorted((o.t_start, o.t_end) for o in uo.outings if o.mode == mode)
            if s is None or not ivs:
                continue
            t0, t1 = np.array([a for a, _ in ivs]), np.array([e for _, e in ivs])
            j = np.searchsorted(t0, s["t0"], side="right") - 1
            ok = (j >= 0) & (s["t0"] <= t1[np.maximum(j, 0)]) & (s["dt"] <= SPEED_DT_S)
            parts.append((s["d"] / s["dt"])[ok])
        return np.concatenate(parts) if parts else np.array([])
    return f


def q_outing_length(b):
    return np.array([o.length_m / 1000 for u in b.outings for o in u.outings])


def q_outing_duration(b):
    return np.array([(o.t_end - o.t_start) / 60 for u in b.outings for o in u.outings])


def q_stops_per_outing(b):
    return np.array([sum(v.kind == "stop" for v in o.visits) for u in b.outings for o in u.outings])


def q_stop_dwell(b):
    return np.array([v.dwell_s / 60 for u in b.outings for o in u.outings for v in o.visits if v.kind == "stop"])


def q_points_per_user(b):
    return np.array([len(a[0]) for a in b.arrays.values()])


def q_outings_per_user(b):
    return np.array([len(u.outings) for u in b.outings])


def q_active_days(b):
    return np.array([len({(o.t_start + LOCAL_S) // 86400 for o in u.outings}) for u in b.outings])


def q_distinct_places(b):
    return np.array([len({v.place_id for o in u.outings for v in o.visits}) for u in b.outings])


def q_revisit_rate(b):
    out = []
    for u in b.outings:
        seen: set = set()
        tot = r = 0
        for o in u.outings:
            for v in o.visits:
                if v.kind == "stop":
                    tot += 1
                    r += v.place_id in seen
            seen.update(v.place_id for v in o.visits)
        if tot:
            out.append(r / tot)
    return np.array(out)


def q_users_per_place(b):
    c = Counter()
    for u in b.outings:
        for p in {v.place_id for o in u.outings for v in o.visits}:
            c[p] += 1
    return np.array(sorted(c.values()), float)


def q_users_per_cell(b):
    c = Counter()
    for u in b.outings:
        for cell in {s for o in u.outings for s in o.cells}:
            c[cell] += 1
    return np.array(sorted(c.values()), float)


def q_covisit_degree(b):
    res = b.m2()
    return np.array([d for _, d in res.graph.degree()], float) if res is not None else np.array([])


# -- time (categorical: compared by total variation over the bins) ---------------------------

def _hour(ts):
    return ((np.asarray(ts, np.int64) + LOCAL_S) % 86400) // 3600


def _weekend(ts):
    return (((np.asarray(ts, np.int64) + LOCAL_S) // 86400 + 3) % 7) >= 5


def q_start_hour(daytype: str) -> Callable[[TraceBundle], np.ndarray]:
    def f(b):
        ts = np.array([o.t_start for u in b.outings for o in u.outings], np.int64)
        return _hour(ts[_weekend(ts) == (daytype == "weekend")]) if len(ts) else np.array([])
    return f


def q_stop_hour(daytype: str) -> Callable[[TraceBundle], np.ndarray]:
    def f(b):
        ts = np.array([v.t_arrive for u in b.outings for o in u.outings for v in o.visits
                       if v.kind == "stop"], np.int64)
        return _hour(ts[_weekend(ts) == (daytype == "weekend")]) if len(ts) else np.array([])
    return f


def q_timebin(b):
    ts = np.array([o.t_start for u in b.outings for o in u.outings], np.int64)
    if not len(ts):
        return np.array([])
    tod = np.asarray(TOD_BINS)[_hour(ts) // 6]
    day = np.where(_weekend(ts), "weekend", "weekday")
    return np.array([f"{d}:{t}" for d, t in zip(day, tod)])


# --------------------------------------------------------------------------------------- quantities

@dataclass(frozen=True)
class Quantity:
    name: str
    group: str                                   # point | outing | user | structure | time
    extract: Callable[[TraceBundle], np.ndarray]
    kind: str = "numeric"                        # numeric | categorical
    unit: str = ""
    log: bool = False                            # heavy tail: plot/bin on a log axis


QUANTITIES: list[Quantity] = [
    Quantity("sampling_interval", "point", q_sampling_interval, unit="s", log=True),
    Quantity("gap_len", "point", q_gap_len, unit="s", log=True),
    Quantity("step_len", "point", q_step_len, unit="m", log=True),
    Quantity("speed", "point", q_speed, unit="m/s", log=True),
    *[Quantity(f"speed_{m}", "point", q_speed_by_mode(m), unit="m/s", log=True) for m in MODES],
    Quantity("heading_change", "point", q_heading_change, unit="deg"),
    Quantity("stop_jitter", "point", q_stop_jitter, unit="m", log=True),
    Quantity("outing_length", "outing", q_outing_length, unit="km", log=True),
    Quantity("outing_duration", "outing", q_outing_duration, unit="min", log=True),
    Quantity("stops_per_outing", "outing", q_stops_per_outing),
    Quantity("stop_dwell", "outing", q_stop_dwell, unit="min", log=True),
    Quantity("points_per_user", "user", q_points_per_user, log=True),
    Quantity("outings_per_user", "user", q_outings_per_user, log=True),
    Quantity("active_days", "user", q_active_days, log=True),
    Quantity("distinct_places_per_user", "user", q_distinct_places, log=True),
    Quantity("revisit_rate", "user", q_revisit_rate),
    Quantity("users_per_place", "structure", q_users_per_place, log=True),
    Quantity("users_per_cell", "structure", q_users_per_cell, log=True),
    Quantity("covisit_degree", "structure", q_covisit_degree),
    Quantity("outing_start_hour_weekday", "time", q_start_hour("weekday"), kind="categorical", unit="h"),
    Quantity("outing_start_hour_weekend", "time", q_start_hour("weekend"), kind="categorical", unit="h"),
    Quantity("stop_hour_weekday", "time", q_stop_hour("weekday"), kind="categorical", unit="h"),
    Quantity("stop_hour_weekend", "time", q_stop_hour("weekend"), kind="categorical", unit="h"),
    Quantity("timebin_share", "time", q_timebin, kind="categorical"),
]

PCTS = (1, 5, 25, 50, 75, 95, 99)


def by_name(name: str, quantities: Sequence[Quantity] = QUANTITIES) -> Quantity:
    return next(q for q in quantities if q.name == name)


# --------------------------------------------------------------------------------------- metrics

def moments(x: np.ndarray) -> dict[str, float]:
    """mean, std, skewness, kurtosis (excess) and quantiles p1..p99; NaN when empty."""
    x = np.asarray(x, float)
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "std", "skew", "kurtosis", *[f"p{p}" for p in PCTS])}
    from scipy import stats
    qs = np.percentile(x, PCTS)
    sd = float(x.std())
    return {"mean": float(x.mean()), "std": sd,
            "skew": float(stats.skew(x)) if x.size > 2 and sd > 0 else (0.0 if sd == 0 else float("nan")),
            "kurtosis": float(stats.kurtosis(x)) if x.size > 3 and sd > 0 else (0.0 if sd == 0 else float("nan")),
            **{f"p{p}": float(v) for p, v in zip(PCTS, qs)}}


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    return np.random.default_rng(seed).choice(x, size=n, replace=False)


def tv_binned(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Total variation on a shared binning (quantile edges of the pooled sample)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    return float(0.5 * np.abs(ha / max(ha.sum(), 1) - hb / max(hb.sum(), 1)).sum())


def tv_categorical(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    keys = set(ca) | set(cb)
    return float(0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in keys))


def distances(real: np.ndarray, syn: np.ndarray, kind: str = "numeric",
              max_n: int = 100_000, seed: int = 0) -> dict[str, float]:
    """KS, Wasserstein-1 and total variation between the two samples (subsampled above `max_n`)."""
    if kind == "categorical":
        return {"ks": float("nan"), "ks_p": float("nan"), "wasserstein": float("nan"),
                "tv": tv_categorical(real, syn)}
    if len(real) == 0 or len(syn) == 0:
        return {k: float("nan") for k in ("ks", "ks_p", "wasserstein", "tv")}
    from scipy import stats
    a, b = _subsample(np.asarray(real, float), max_n, seed), _subsample(np.asarray(syn, float), max_n, seed + 1)
    ks = stats.ks_2samp(a, b)
    return {"ks": float(ks.statistic), "ks_p": float(ks.pvalue),
            "wasserstein": float(stats.wasserstein_distance(a, b)), "tv": tv_binned(a, b)}


def rel_errors(mr: Mapping[str, float], ms: Mapping[str, float],
               keys=("mean", "std", "p50", "p95")) -> dict[str, float]:
    """|synthetic - real| / |real| per moment (NaN-safe)."""
    out = {}
    for k in keys:
        r, s = mr.get(k, float("nan")), ms.get(k, float("nan"))
        out[f"rel_err_{k}"] = float(abs(s - r) / abs(r)) if np.isfinite(r) and np.isfinite(s) and r != 0 \
            else float("nan")
    return out


def category_shares(x: np.ndarray) -> dict[str, float]:
    c = Counter(x.tolist())
    return {str(k): v / len(x) for k, v in sorted(c.items())} if len(x) else {}


def compare_quantity(q: Quantity, real: TraceBundle, syn: TraceBundle) -> dict:
    xr, xs = np.asarray(q.extract(real)), np.asarray(q.extract(syn))
    row: dict = {"quantity": q.name, "group": q.group, "kind": q.kind, "unit": q.unit,
                 "n_real": int(xr.size), "n_syn": int(xs.size)}
    if q.kind == "numeric":
        mr, ms = moments(xr), moments(xs)
        row.update({f"real_{k}": v for k, v in mr.items()})
        row.update({f"syn_{k}": v for k, v in ms.items()})
        row.update(distances(xr, xs))
        row.update(rel_errors(mr, ms))
    else:
        row.update(distances(xr, xs, kind="categorical"))
    return row


def summary_table(real: TraceBundle, syn: TraceBundle,
                  quantities: Sequence[Quantity] = QUANTITIES) -> pd.DataFrame:
    """One row per quantity: moments of both sides and the distances between them."""
    return pd.DataFrame([compare_quantity(q, real, syn) for q in quantities])


MD_COLS = ("quantity", "group", "n_real", "n_syn", "real_mean", "syn_mean", "real_p50", "syn_p50",
           "real_p95", "syn_p95", "ks", "wasserstein", "tv", "rel_err_mean", "rel_err_p50")


def to_markdown(df: pd.DataFrame, cols: Sequence[str] = MD_COLS, digits: int = 3) -> str:
    cols = [c for c in cols if c in df.columns]
    def fmt(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else (f"{v:.{digits}g}" if abs(v) < 1e5 else f"{v:.3e}")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_summary(df: pd.DataFrame, out_dir, name: str) -> tuple[Path, Path]:
    """-> (csv path, markdown path) under `out_dir` (e.g. geolife/eval/results/)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcsv, pmd = out / f"{name}.csv", out / f"{name}.md"
    df.to_csv(pcsv, index=False)
    pmd.write_text(to_markdown(df) + "\n", encoding="utf-8")
    return pcsv, pmd

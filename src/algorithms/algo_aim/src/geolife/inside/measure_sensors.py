"""Sensor-noise measurement stage (PLAN step 1): raw GPS points -> per-user sensor views.

The raw points are NOT in `UserOutings`, so this stage has its own reader (next to
`inside/build_outings.py`, reusing `inside/loaders.py`). Per user it computes the per-point
statistics of schema_v2.SENSOR_NAMES as histograms with FIXED PUBLIC bin edges:

  sampling interval (log-ish bins), the user's median interval (one-hot), signal-loss gap length
  and gaps per logged hour, 30 s-thinned step speed by mode (labels.txt vote,
  else the same thinned-speed heuristic as the tokenizer, per outing-like run; raw 1-5 s steps are
  GPS-noise dominated, so speed uses thinned steps), heading change of consecutive moving steps,
  positional jitter at stops (distance of in-stop points from the stop centre, stops detected with
  the public 200 m / 10 min rule on 30 s-thinned points), raw points per outing-like run, and the
  dt-run length (v2e, PLAN step 9: length in steps of maximal runs of equal rounded dt among
  within-run sampling steps - the lock texture the realfake discriminator keyed on, D9).
  (sensor_step - raw step length by mode - was dropped in v2d, D27: per-mode mass below the noise
  floor on every dev seed and never consumed by the renderer.)

Bounded per-user contribution: each statistic is capped at schema_v2.MAX_SENSOR_STEPS samples by a
DETERMINISTIC uniform subsample seeded from the user's pseudonym (a function of that user's data
alone), then L1-normalised into a user pmf (L2 <= 1) or a one-hot (L2 = 1), exactly the vector
families of measure_v2. Release, noise and accounting happen in measure_v2.build_measurements_v2
(one Gaussian measurement and one ledger entry per table; budget split per decision D20).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from geolife.inside.measure_v2 import _bucket, _user_rng
from geolife.transfer import schema_v2 as S2
from geolife.transfer.geo import LabelSpan, detect_stays, step_dist_dt, to_xy
from geolife.transfer.tokenize.outings import OutingsConfig, _clean, label_codes

CFG = OutingsConfig()      # public tokenizer settings (gap 2 h, stop rule, speed heuristic cuts)


@dataclass
class SensorView:
    """One user's sensor-statistic counts (raw, un-normalised; pmf_table normalises per user)."""
    user: str
    pmf: dict[str, Counter]


def _cap(rng: np.random.Generator, arr: np.ndarray, cap: int = S2.MAX_SENSOR_STEPS) -> np.ndarray:
    """Deterministic uniform subsample of one user's own samples (seeded from the pseudonym)."""
    if len(arr) <= cap:
        return arr
    return arr[np.sort(rng.permutation(len(arr))[:cap])]


def _run_modes(rid: np.ndarray, n_runs: int, code: np.ndarray, v: np.ndarray,
               vrid: np.ndarray) -> list[str]:
    """Mode per run: labels.txt vote if >= half the run's points are labelled, else the tokenizer's
    speed heuristic on the median moving speed (same cuts as OutingsConfig)."""
    k = 8
    cnt = np.bincount(rid[code > 0] * k + code[code > 0], minlength=n_runs * k).reshape(n_runs, k)
    tot = np.bincount(rid, minlength=n_runs)
    lab, top = cnt.sum(1), cnt.argmax(1)
    med = np.full(n_runs, np.nan)
    mov = v > 0.5
    if mov.any():
        import pandas as pd
        s = pd.Series(v[mov]).groupby(vrid[mov]).median()
        med[s.index.values.astype(int)] = s.values
    modes = []
    for i in range(n_runs):
        if lab[i] >= CFG.label_min_share * max(tot[i], 1) and 1 <= top[i] <= 5:
            modes.append(("walk", "bike", "bus", "car", "train")[top[i] - 1])
        elif math.isnan(med[i]):
            modes.append("unknown")
        elif med[i] < CFG.speed_walk:
            modes.append("walk")
        elif med[i] < CFG.speed_bike:
            modes.append("bike")
        elif med[i] < CFG.speed_bus:
            modes.append("bus")
        elif med[i] < CFG.speed_car:
            modes.append("car")
        else:
            modes.append("train")
    return modes


def make_sensor_view(user: str, t, lat, lon, labels: Sequence[LabelSpan] = (),
                     seed: int = 0) -> SensorView:
    """One user's raw points (epoch s, deg) -> SensorView. Uses only this user's data."""
    pm: dict[str, Counter] = {n: Counter() for n in S2.SENSOR_NAMES}
    rng = _user_rng(seed, "sensor:" + user)
    t = np.asarray(t, np.int64)
    if len(t) < 2:                     # e.g. all points outside the analysis bbox
        return SensorView(user, pm)
    t, lat, lon = _clean(t, np.asarray(lat, float), np.asarray(lon, float), CFG.max_speed_ms)
    n = len(t)
    if n < 2:
        return SensorView(user, pm)

    # outing-like runs: split where dt > the public outing gap
    dt_all = np.diff(t).astype(float)
    rid = np.cumsum(np.r_[0, (dt_all > CFG.gap_s).astype(np.int64)])
    n_runs = int(rid[-1]) + 1
    same = rid[1:] == rid[:-1]
    d, _ = step_dist_dt(np.zeros(n, np.int8), t, lat, lon)       # metres, length n-1
    dt = dt_all

    # ---- sampling intervals and gaps (within runs)
    iv = dt[same & (dt > 0) & (dt <= S2.SENSOR_GAP_MIN_S)]
    for x in _cap(rng, iv):
        b = _bucket(x, S2.INTERVAL_S_EDGES)
        if b:
            pm[S2.SENSOR_INTERVAL][b] += 1
    if len(iv):
        b = _bucket(float(np.median(iv)), S2.INTERVAL_S_EDGES)
        if b:
            pm[S2.SENSOR_INTERVAL_USER][b] = 1
    gaps = dt[same & (dt > S2.SENSOR_GAP_MIN_S)]
    for x in _cap(rng, gaps):
        b = _bucket(x, S2.GAP_S_EDGES)
        if b:
            pm[S2.SENSOR_GAP_LEN][b] += 1
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    ends = np.r_[np.flatnonzero(~same), n - 1]
    logged_h = float((t[ends] - t[starts]).sum()) / 3600.0
    if logged_h > 0:
        b = _bucket(len(gaps) / logged_h, S2.GAP_RATE_EDGES)
        if b:
            pm[S2.SENSOR_GAP_RATE][b] = 1

    # ---- dt-run lengths (v2e, PLAN step 9): maximal spans of equal ROUNDED dt among within-run
    # sampling steps (dt <= gap-min); a gap or an outing-run boundary breaks a run. Real loggers
    # lock on one interval for long runs; the renderer holds dt per drawn run length.
    samp = same & (dt > 0) & (dt <= S2.SENSOR_GAP_MIN_S)
    if samp.any():
        idx = np.flatnonzero(samp)
        r = np.rint(dt[idx])
        new = np.r_[True, (np.diff(idx) != 1) | (r[1:] != r[:-1])]
        runlens = np.diff(np.r_[np.flatnonzero(new), len(idx)]).astype(float)
        for ln in _cap(rng, runlens):
            b = _bucket(float(ln), S2.RUNLEN_EDGES)
            if b:
                pm[S2.SENSOR_DT_RUNLEN][b] += 1
        # the lock regime is per-user (flicker vs locked loggers): release the user's MEDIAN run
        # length as a one-hot, the same pairing as interval/interval_user
        b = _bucket(float(np.median(runlens)), S2.RUNLEN_EDGES)
        if b:
            pm[S2.SENSOR_DT_RUNLEN_USER][b] = 1

    # ---- speed (30 s-thinned steps: raw 1-5 s steps are GPS-noise dominated) and raw step
    # length, both by run mode (labels.txt vote, else the thinned-speed heuristic)
    x, y = to_xy(lat, lon)
    key = rid * (1 << 36) + t // S2.SENSOR_SPEED_THIN_S
    th = np.r_[True, key[1:] != key[:-1]]
    tx, ty, tt, trid = x[th], y[th], t[th], rid[th]
    tsame = trid[1:] == trid[:-1]
    tdt = np.diff(tt).astype(float)
    tok = tsame & (tdt > 0) & (tdt <= S2.SENSOR_SPEED_THIN_MAX_DT_S)
    tv = np.where(tok, np.hypot(np.diff(tx), np.diff(ty)) / np.where(tdt > 0, tdt, 1.0), np.nan)
    code = label_codes(t, labels)
    modes = _run_modes(rid, n_runs, code, np.where(tok, tv, np.nan), trid[1:])
    for j in _cap(rng, np.flatnonzero(tok)):
        b = _bucket(float(tv[j]), S2.SPEED_MS_EDGES)
        if b:
            pm[S2.SENSOR_SPEED][(modes[trid[j]], *b)] += 1
    # (sensor_step dropped in v2d, D27: raw step lengths are not released; `ok` still gates heading)
    ok = same & (dt > 0) & (dt <= S2.SENSOR_SPEED_MAX_DT_S) & np.isfinite(d)

    # ---- heading change between consecutive moving steps
    dx, dy = np.diff(x), np.diff(y)
    move = ok & (d > S2.SENSOR_MOVE_MIN_M)
    pair = move[:-1] & move[1:]
    if pair.any():
        a1 = np.arctan2(dy[:-1], dx[:-1])
        a2 = np.arctan2(dy[1:], dx[1:])
        turn = np.degrees(np.abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi))[pair]
        for xdeg in _cap(rng, turn):
            b = _bucket(float(xdeg), S2.HEADING_DEG_EDGES)
            if b:
                pm[S2.SENSOR_HEADING][b] += 1

    # ---- positional jitter at stops (public stop rule on the same 30 s-thinned points, per run)
    i0, i1, _mx, _my, _rms = detect_stays(tx, ty, tt.astype(float), trid, CFG.stop_dist_m,
                                          CFG.stop_dwell_s)
    if len(i0):
        # distance from a ROBUST stop centre (per-axis median): the stay window also sweeps up the
        # walk-in / walk-out points, which would drag a mean centre away from the true stop
        rr = np.concatenate([np.hypot(tx[a:b + 1] - np.median(tx[a:b + 1]),
                                      ty[a:b + 1] - np.median(ty[a:b + 1]))
                             for a, b in zip(i0, i1)])
        for r in _cap(rng, rr):
            b = _bucket(float(r), S2.JITTER_M_EDGES)
            if b:
                pm[S2.SENSOR_STOP_JITTER][b] += 1

    # ---- raw points per outing-like run (runs capped like outings)
    counts = (ends - starts + 1).astype(float)
    for c in _cap(rng, counts, S2.MAX_OUTINGS):
        b = _bucket(float(c), S2.PTS_PER_OUTING_EDGES)
        if b:
            pm[S2.SENSOR_PTS_OUTING][b] += 1
    return SensorView(user, pm)


# ------------------------------------------------------------------ reader (inside only)

def sensor_views_from_points(points: str | Path, data_root: str | Path | None = None,
                             seed: int = 0, bbox=None) -> list[SensorView]:
    """points.parquet (uid, t, lat, lon) + GeoLife labels -> pseudonymised SensorViews.
    Same pseudonym seed as learn_v2.users_from_points so the per-user capping seeds line up."""
    from geolife.inside.data import default_data_dir, read_labels
    from geolife.inside.loaders import load_points_arrays
    from geolife.transfer.geo import pseudonyms
    root = Path(data_root) if data_root else default_data_dir()
    pts = load_points_arrays(Path(points))
    names = pseudonyms(list(pts), seed)
    views = []
    for uid in sorted(pts):
        t, lat, lon = pts[uid]
        if bbox is not None:
            lat0, lat1, lon0, lon1 = bbox
            m = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
            t, lat, lon = t[m], lat[m], lon[m]
        lp = root / "Data" / uid / "labels.txt"
        labs = read_labels(lp) if lp.exists() else []
        views.append(make_sensor_view(names[uid], t, lat, lon, labs, seed))
    return views

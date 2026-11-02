"""Raw GPS traces -> `list[UserOutings]` (v2 contract, see docs/geolife/PLAN_OUTINGS.md).

An outing is a maximal run of one user's points with no gap above `gap_s` (files closer than the gap
merge). Outings shorter than 5 min or 500 m are dropped, and so are users without >= 10 real outings
on >= 5 local days. Inside an outing, stops (200 m / 10 min stay rule on 30 s-thinned points) become
visits, plus the first and last point as `start` / `end` visits. Visits snap to the nearest public
place within `snap_radius_m`, else to `h3:<res-8 cell>` with category "unknown".

Times are UTC epoch seconds; local time is UTC+8 (`local_bins`). All heavy work is numpy; the only
Python loops are per outing / per unique rounded position.

Mode: the dominant `labels.txt` mode if >= 50% of the outing's points are labelled, else the speed
heuristic on the median moving speed (steps > 0.5 m/s, 30 s-thinned, dt <= 120 s):
< 2 m/s walk, < 4 bike, < 7 bus, < 15 car, otherwise train (EDA labelled medians: walk 1.1,
bike 3.1, bus 4.7, car 8.5, rail 19 m/s; bus/car/rail overlap, so those are coarse).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from geolife.transfer.geo import LabelSpan, detect_stays, from_xy, pseudonyms, step_dist_dt, to_xy
from geolife.transfer.types import (DAYTYPES, LOCAL_UTC_OFFSET_H, TOD_BINS, GpsPoint, Outing, UserOutings,
                     Visit)

LOCAL_OFFSET_S = LOCAL_UTC_OFFSET_H * 3600
LABEL_MODE = {"walk": "walk", "run": "walk", "bike": "bike", "bus": "bus", "car": "car",
              "taxi": "car", "motorcycle": "car", "subway": "train", "train": "train",
              "railway": "train"}
MODES = ("walk", "bike", "bus", "car", "train")
OUT_CATEGORIES = ("residential", "office", "school", "food", "shop", "transit", "leisure",
                  "health", "other")


@dataclass
class OutingsConfig:
    gap_s: int = 7200
    min_dur_s: int = 300
    min_len_m: float = 500.0
    min_outings: int = 10
    min_days: int = 5
    stop_dist_m: float = 200.0
    stop_dwell_s: int = 600
    stop_thin_s: int = 30
    place_res: int = 8
    trav_res: int = 9
    snap_radius_m: float = 150.0
    max_speed_ms: float = 70.0
    label_min_share: float = 0.5
    speed_walk: float = 2.0
    speed_bike: float = 4.0
    speed_bus: float = 7.0
    speed_car: float = 15.0
    match: bool = False
    matcher: object = field(default=None, repr=False)   # geolife.transfer.map.match.Matcher when match=True
    match_thin_s: int = 5


# --- time ---------------------------------------------------------------------------------


def local_bin_idx(t):
    """Vectorised (tod index 0..3, daytype index 0 weekday / 1 weekend) for UTC epoch seconds."""
    lt = np.asarray(t, dtype=np.int64) + LOCAL_OFFSET_S
    hour = (lt % 86400) // 3600
    dow = ((lt // 86400) + 3) % 7        # 1970-01-01 was a Thursday; Mon=0
    return hour // 6, (dow >= 5).astype(int)


def local_bins(t: int) -> tuple[str, str]:
    """UTC epoch seconds -> (tod_bin, daytype) in local time (UTC+8)."""
    a, b = local_bin_idx(np.int64(t))
    return TOD_BINS[int(a)], DAYTYPES[int(b)]


# --- places -------------------------------------------------------------------------------


class PlaceIndex:
    """Nearest-place lookup (KD-tree in a local metric frame). Empty table -> everything unknown."""

    def __init__(self, places: pd.DataFrame | None, lat0: float = 39.95):
        from scipy.spatial import cKDTree
        self.lat0 = lat0
        self.n = 0 if places is None else len(places)
        if self.n:
            self.ids = places["id"].astype(str).to_numpy()
            self.cats = places["category"].astype(str).to_numpy()
            x, y = to_xy(places["lat"].to_numpy(float), places["lon"].to_numpy(float), lat0)
            self.tree = cKDTree(np.column_stack([x, y]))

    def snap(self, lat, lon, radius_m: float):
        """-> (place index or -1) per position."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if not self.n or lat.size == 0:
            return np.full(lat.shape, -1, np.int64)
        x, y = to_xy(lat, lon, self.lat0)
        d, i = self.tree.query(np.column_stack([x, y]), distance_upper_bound=radius_m)
        return np.where(np.isfinite(d), i, -1).astype(np.int64)


def _h3_strings(lat, lon, res: int, decimals: int = 4):
    """H3 cell id strings for positions; computed once per unique rounded position.
    Returns (unique cell strings array, inverse index per input position)."""
    import h3
    la = np.round(np.asarray(lat, float) * 10 ** decimals).astype(np.int64)
    lo = np.round(np.asarray(lon, float) * 10 ** decimals).astype(np.int64)
    key = (la + 90 * 10 ** decimals) * 10 ** 8 + (lo + 180 * 10 ** decimals)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    ulat = la[first] / 10 ** decimals
    ulon = lo[first] / 10 ** decimals
    cells = np.array([h3.latlng_to_cell(a, b, res) for a, b in zip(ulat.tolist(), ulon.tolist())],
                     dtype=object)
    return cells, inv


# --- per-user core --------------------------------------------------------------------------


def _clean(t, lat, lon, max_speed):
    """Sort by time, drop duplicate timestamps, invalid coordinates and isolated speed spikes."""
    ok = (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    t, lat, lon = t[ok], lat[ok], lon[ok]
    order = np.argsort(t, kind="stable")
    t, lat, lon = t[order], lat[order], lon[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, lat, lon = t[keep], lat[keep], lon[keep]
    if len(t) < 3:
        return t, lat, lon
    d, dt = step_dist_dt(np.zeros(len(t), np.int8), t, lat, lon)
    fast = (d / dt) > max_speed
    spike = np.zeros(len(t), bool)
    spike[1:-1] = fast[:-1] & fast[1:]
    return t[~spike], lat[~spike], lon[~spike]


def label_codes(t: np.ndarray, labels: Sequence[LabelSpan]) -> np.ndarray:
    """Per-point mode code: 0 none, 1.. index into MODES (unlabelled-by-us modes -> len(MODES)+1)."""
    code = np.zeros(len(t), np.int8)
    if not len(labels):
        return code
    spans = sorted(labels, key=lambda s: s.t_start)
    t0 = np.array([s.t_start for s in spans])
    t1 = np.array([s.t_end for s in spans])
    c = np.array([MODES.index(LABEL_MODE[s.mode]) + 1 if s.mode in LABEL_MODE else len(MODES) + 1
                  for s in spans], np.int8)
    k = np.searchsorted(t0, t, side="right") - 1
    ok = (k >= 0) & (t <= t1[np.maximum(k, 0)])
    code[ok] = c[k[ok]]
    return code


def _modes(oid, code, n_out, sx, sy, st, soid, cfg: OutingsConfig):
    """Mode per outing: label vote, else speed heuristic on thinned steps (sx, sy in metres)."""
    k = len(MODES) + 2
    cnt = np.bincount(oid[code > 0] * k + code[code > 0], minlength=n_out * k).reshape(n_out, k)
    tot = np.bincount(oid, minlength=n_out)
    lab = cnt.sum(1)
    top = cnt.argmax(1)
    # speed
    same = soid[1:] == soid[:-1]
    dt = np.diff(st).astype(float)
    d = np.hypot(np.diff(sx), np.diff(sy))
    ok = same & (dt > 0) & (dt <= 120)
    v = np.where(ok, d / np.where(dt > 0, dt, 1), np.nan)
    mov = ok & (v > 0.5)
    med = np.full(n_out, np.nan)
    if mov.any():
        s = pd.Series(v[mov]).groupby(soid[1:][mov]).median()
        med[s.index.values] = s.values
    out = []
    for i in range(n_out):
        if lab[i] >= cfg.label_min_share * max(tot[i], 1) and 1 <= top[i] <= len(MODES):
            out.append(MODES[top[i] - 1])
        elif np.isnan(med[i]):
            out.append("unknown")
        elif med[i] < cfg.speed_walk:
            out.append("walk")
        elif med[i] < cfg.speed_bike:
            out.append("bike")
        elif med[i] < cfg.speed_bus:
            out.append("bus")
        elif med[i] < cfg.speed_car:
            out.append("car")
        else:
            out.append("train")
    return out


def user_outings_from_arrays(user: str, t, lat, lon, pidx: PlaceIndex, cfg: OutingsConfig,
                             labels: Sequence[LabelSpan] = (), stats: dict | None = None
                             ) -> UserOutings | None:
    """One user's raw points (any order, any file structure) -> UserOutings, or None if the user
    does not meet the minimum-activity rule."""
    t = np.asarray(t, np.int64)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    t, lat, lon = _clean(t, lat, lon, cfg.max_speed_ms)
    n = len(t)
    if n < 2:
        return None
    x, y = to_xy(lat, lon)
    d, _ = step_dist_dt(np.zeros(n, np.int8), t, lat, lon)
    step = np.r_[d, 0.0]
    new = np.r_[True, np.diff(t) > cfg.gap_s]
    oid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n] - 1
    # step k joins k and k+1 inside an outing; the last step of each outing is cut
    step[ends] = 0.0
    cs = np.r_[0.0, np.cumsum(step)]
    length = cs[ends + 1] - cs[starts]
    dur = t[ends] - t[starts]
    real = (dur >= cfg.min_dur_s) & (length >= cfg.min_len_m)
    if stats is not None:
        stats["outings_all"] = stats.get("outings_all", 0) + int(len(starts))
    days = (t[starts] + LOCAL_OFFSET_S) // 86400
    if real.sum() < cfg.min_outings or len(np.unique(days[real])) < cfg.min_days:
        return None
    # keep only real outings
    keep_pt = real[oid]
    remap = np.cumsum(real) - 1
    oid = remap[oid[keep_pt]]
    t, lat, lon, x, y = t[keep_pt], lat[keep_pt], lon[keep_pt], x[keep_pt], y[keep_pt]
    code = label_codes(t, labels)
    starts, ends = starts[real], ends[real]
    length, dur = length[real], dur[real]
    n_out = len(starts)
    ns = np.r_[0, np.cumsum(ends - starts + 1)]
    s0, s1 = ns[:-1], ns[1:] - 1                       # first/last point index in the filtered arrays

    # thinned points (stop detection + speed)
    key = oid.astype(np.int64) * (1 << 36) + t // cfg.stop_thin_s
    th = np.r_[True, key[1:] != key[:-1]]
    tx, ty, tt, toid = x[th], y[th], t[th], oid[th]
    tlat, tlon = lat[th], lon[th]
    mode = _modes(oid, code, n_out, tx, ty, tt, toid, cfg)

    i0, i1, mx, my, _ = detect_stays(tx, ty, tt.astype(float), toid, cfg.stop_dist_m,
                                     cfg.stop_dwell_s)
    stop_lat, stop_lon = from_xy(mx, my)
    stop_oid = toid[i0] if len(i0) else np.zeros(0, np.int64)
    stop_t0 = tt[i0] if len(i0) else np.zeros(0, np.int64)
    stop_dw = (tt[i1] - tt[i0]) if len(i0) else np.zeros(0, np.int64)

    # visit table: stops + start + end positions
    v_lat = np.r_[stop_lat, lat[s0], lat[s1]]
    v_lon = np.r_[stop_lon, lon[s0], lon[s1]]
    cells, inv = _h3_strings(v_lat, v_lon, cfg.place_res)
    v_cell = cells[inv]
    pi = pidx.snap(v_lat, v_lon, cfg.snap_radius_m)
    ns_ = len(i0)
    kinds = np.r_[np.zeros(ns_, np.int8), np.ones(n_out, np.int8), np.full(n_out, 2, np.int8)]
    v_oid = np.r_[stop_oid, np.arange(n_out), np.arange(n_out)]
    v_t = np.r_[stop_t0, t[s0], t[s1]]
    v_dw = np.r_[stop_dw, np.zeros(n_out, np.int64), np.zeros(n_out, np.int64)]
    KIND = ("stop", "start", "end")
    visits_by: list[list] = [[] for _ in range(n_out)]
    for j in range(len(v_lat)):
        p = pi[j]
        if p >= 0:
            pid, cat = pidx.ids[p], (pidx.cats[p] if pidx.cats[p] in OUT_CATEGORIES else "other")
        else:
            pid, cat = "h3:" + v_cell[j], "unknown"
        visits_by[v_oid[j]].append((int(v_t[j]), kinds[j], Visit(
            pid, cat, v_cell[j], int(v_t[j]), int(v_dw[j]), KIND[kinds[j]])))

    # traversed res-9 cells, first-seen order
    cc, cinv = _h3_strings(lat, lon, cfg.trav_res)
    ccode = pd.factorize(cc)[0][cinv]
    packed = oid.astype(np.int64) * (len(cc) + 1) + ccode
    _, first = np.unique(packed, return_index=True)
    first.sort()
    cells_by: list[list[str]] = [[] for _ in range(n_out)]
    for f in first:
        cells_by[oid[f]].append(cc[cinv[f]])

    outings = []
    for i in range(n_out):
        vs = tuple(v for _, _, v in sorted(visits_by[i], key=lambda r: (r[0], r[1] != 1, r[1] == 2)))
        segs: tuple[str, ...] = ()
        if cfg.match and cfg.matcher is not None:
            a, b = s0[i], s1[i] + 1
            m = np.r_[True, np.diff(t[a:b]) >= cfg.match_thin_s]
            pts = list(zip(lon[a:b][m].tolist(), lat[a:b][m].tolist()))
            res = cfg.matcher.match(pts, mode[i])
            segs = tuple(str(s) for s in res)
        outings.append(Outing(int(t[s0[i]]), int(t[s1[i]]), mode[i], float(length[i]), vs,
                              tuple(cells_by[i]), segs))
    if stats is not None:
        stats["outings_real"] = stats.get("outings_real", 0) + n_out
        stats["stops"] = stats.get("stops", 0) + ns_
    return UserOutings(user, outings)


# --- public API -----------------------------------------------------------------------------


def _flatten(traces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = list(traces)
    if seq and not isinstance(seq[0], GpsPoint):         # list of trajectories (files)
        seq = [p for tr in seq for p in tr]
    return (np.fromiter((p.t for p in seq), np.int64, len(seq)),
            np.fromiter((p.lat for p in seq), float, len(seq)),
            np.fromiter((p.lon for p in seq), float, len(seq)))


def build_outings(traces_by_user: Mapping[str, Sequence], places: pd.DataFrame | None,
                  cfg: OutingsConfig | None = None,
                  labels: Mapping[str, Sequence[LabelSpan]] | None = None, seed: int = 0,
                  stats: dict | None = None) -> list[UserOutings]:
    """`traces_by_user[uid]` is a list of GpsPoint or a list of trajectories (lists of GpsPoint).
    User ids are replaced by seeded random pseudonyms (data.pseudonyms; the mapping is not kept).
    `places` is the public places table (id, category, lat, lon, popularity) or None.
    Users who fail the minimum-activity rule are dropped."""
    cfg = cfg or OutingsConfig()
    pidx = PlaceIndex(places)
    names = pseudonyms(list(traces_by_user), seed)
    out = []
    for uid in sorted(traces_by_user):
        t, la, lo = _flatten(traces_by_user[uid])
        uo = user_outings_from_arrays(names[uid], t, la, lo, pidx, cfg,
                                      (labels or {}).get(uid, ()), stats)
        if uo is not None:
            out.append(uo)
    return out


# --- flat export ------------------------------------------------------------------------


def to_frames(users: Sequence[UserOutings]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(outings frame, visits frame) for inspection. The outings frame also carries the visit
    table as a JSON string."""
    orows, vrows = [], []
    for u in users:
        for k, o in enumerate(u.outings):
            tod, dt = local_bins(o.t_start)
            orows.append({"user": u.user, "outing_idx": k, "t_start": o.t_start, "t_end": o.t_end,
                          "mode": o.mode, "length_m": o.length_m, "n_visits": len(o.visits),
                          "n_stops": sum(v.kind == "stop" for v in o.visits),
                          "n_cells": len(o.cells), "tod": tod, "daytype": dt,
                          "visits": json.dumps([[v.kind, v.place_id, v.category, v.cell, v.t_arrive,
                                                 v.dwell_s] for v in o.visits])})
            for v in o.visits:
                vrows.append({"user": u.user, "outing_idx": k, "kind": v.kind, "place_id": v.place_id,
                              "category": v.category, "cell": v.cell, "t_arrive": v.t_arrive,
                              "dwell_s": v.dwell_s})
    return pd.DataFrame(orows), pd.DataFrame(vrows)

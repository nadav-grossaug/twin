"""Pure geometry / cleaning helpers shared by inside and outside: haversine, bbox and speed filters, local metric
projection, vectorised stay detection, seeded pseudonyms. Stdlib + numpy only; no file access."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from geolife.transfer.types import GpsPoint

GLOBAL_BBOX = (-90.0, 90.0, -180.0, 180.0)
M_PER_DEG = 111195.0


@dataclass(frozen=True)
class LabelSpan:
    t_start: int
    t_end: int
    mode: str


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def in_bbox(p: GpsPoint, bbox=GLOBAL_BBOX) -> bool:
    lat0, lat1, lon0, lon1 = bbox
    return lat0 <= p.lat <= lat1 and lon0 <= p.lon <= lon1


def filter_bbox(points: list[GpsPoint], bbox=GLOBAL_BBOX) -> list[GpsPoint]:
    return [p for p in points if in_bbox(p, bbox)]


def filter_speed(points: list[GpsPoint], max_speed_ms: float = 70.0) -> list[GpsPoint]:
    """Drop points implying speed above max_speed_ms from the last kept point.

    Points with non-increasing timestamps (duplicates) are dropped.
    """
    out: list[GpsPoint] = []
    for p in points:
        if not out:
            out.append(p)
            continue
        q = out[-1]
        dt = p.t - q.t
        if dt <= 0:
            continue
        if haversine_m(q.lat, q.lon, p.lat, p.lon) / dt <= max_speed_ms:
            out.append(p)
    return out


def clean_trajectory(points, bbox=GLOBAL_BBOX, max_speed_ms: float = 70.0):
    return filter_speed(filter_bbox(points, bbox), max_speed_ms)


def pseudonyms(user_ids: list[str], seed: int) -> dict[str, str]:
    """Seeded random mapping real id -> 'u' + 8 hex chars (unique, input-order independent)."""
    rng = random.Random(seed)
    used: set[str] = set()
    out: dict[str, str] = {}
    for uid in sorted(user_ids):
        while True:
            name = "u%08x" % rng.getrandbits(32)
            if name not in used:
                used.add(name)
                out[uid] = name
                break
    return out


def step_dist_dt(u, t, lat, lon):
    """Distance (m) and dt (s) of consecutive points; NaN where users differ (length n-1)."""
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    d = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dt = np.diff(t).astype(float)
    same = u[1:] == u[:-1]
    d = np.where(same, d, np.nan)
    dt = np.where(same, dt, np.nan)
    return d, dt


def detect_stays(x, y, t, seg, dist_m: float, dwell_s: float):
    """Vectorised Li-et-al stay rule on points sorted within contiguous `seg` groups.

    Anchor i extends while points stay within dist_m of the anchor (same seg); a stay needs
    t[end]-t[i] >= dwell_s; greedy non-overlapping. Returns (i0, i1) inclusive index arrays,
    mean x/y and rms radius (m) around the mean of each stay."""
    n = len(x)
    jend = np.empty(n, np.int64)
    idx = np.arange(n)
    cur = idx + 1
    while idx.size:
        c = np.minimum(cur, n - 1)
        ok = (cur < n) & (seg[c] == seg[idx]) & (np.hypot(x[c] - x[idx], y[c] - y[idx]) <= dist_m)
        jend[idx[~ok]] = cur[~ok]
        idx, cur = idx[ok], cur[ok] + 1
    good = (t[jend - 1] - t) >= dwell_s
    gi = np.flatnonzero(good)
    i0, i1 = [], []
    k = 0
    while True:
        p = np.searchsorted(gi, k)
        if p >= gi.size:
            break
        i = gi[p]
        i0.append(i)
        i1.append(jend[i] - 1)
        k = jend[i]
    i0, i1 = np.array(i0, np.int64), np.array(i1, np.int64)
    if i0.size == 0:
        z = np.zeros(0)
        return i0, i1, z, z, z
    cnt = (i1 - i0 + 1).astype(float)

    def seg_sum(v):
        c = np.r_[0.0, np.cumsum(v)]
        return c[i1 + 1] - c[i0]

    ox, oy = float(np.mean(x)), float(np.mean(y))  # centre first: cumsum of squares loses precision at 1e7 m
    xc, yc = x - ox, y - oy
    mxc, myc = seg_sum(xc) / cnt, seg_sum(yc) / cnt
    var = (seg_sum(xc * xc) / cnt - mxc ** 2) + (seg_sum(yc * yc) / cnt - myc ** 2)
    return i0, i1, mxc + ox, myc + oy, np.sqrt(np.clip(var, 0, None))


def to_xy(lat, lon, lat0: float = 39.95):
    return np.asarray(lon) * M_PER_DEG * np.cos(np.radians(lat0)), np.asarray(lat) * M_PER_DEG


def from_xy(x, y, lat0: float = 39.95):
    return np.asarray(y) / M_PER_DEG, np.asarray(x) / (M_PER_DEG * np.cos(np.radians(lat0)))

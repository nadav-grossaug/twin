"""Stay-point detection and snapping of stays to a public places table / H3 cells."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from geolife.transfer.types import GpsPoint, Stay

M_PER_DEG = 111195.0


@dataclass(frozen=True)
class RawStay:
    """A detected stay before snapping (index range into the (thinned) point list)."""
    lat: float
    lon: float
    t_start: int
    t_end: int
    i0: int
    i1: int  # inclusive


def thin(points: Sequence[GpsPoint], min_dt: int = 10) -> list[GpsPoint]:
    """Keep a point at most every `min_dt` seconds (always keeps first and last)."""
    out: list[GpsPoint] = []
    for p in points:
        if not out or p.t - out[-1].t >= min_dt:
            out.append(p)
    if points and out[-1] is not points[-1] and points[-1].t > out[-1].t:
        out.append(points[-1])
    return out


def detect_stays(points: Sequence[GpsPoint], dist_m: float = 200.0, min_dur_s: int = 1200,
                 max_gap_s: int | None = None) -> list[RawStay]:
    """Distance/time stay rule (Li et al. 2008): anchor i, extend j while within dist_m of the
    anchor; if the elapsed time >= min_dur_s emit the mean position. `max_gap_s` (optional)
    ends the extension at a sampling gap longer than that (signal loss is not a stay)."""
    n = len(points)
    if n == 0:
        return []
    lat0 = points[0].lat
    ky = M_PER_DEG
    kx = ky * math.cos(math.radians(lat0))
    x = np.fromiter((p.lon * kx for p in points), float, n)
    y = np.fromiter((p.lat * ky for p in points), float, n)
    t = np.fromiter((p.t for p in points), float, n)
    out: list[RawStay] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and math.hypot(x[j] - x[i], y[j] - y[i]) <= dist_m:
            if max_gap_s is not None and t[j] - t[j - 1] > max_gap_s:
                break
            j += 1
        if t[j - 1] - t[i] >= min_dur_s:
            out.append(RawStay(float(np.mean([p.lat for p in points[i:j]])),
                               float(np.mean([p.lon for p in points[i:j]])),
                               int(t[i]), int(t[j - 1]), i, j - 1))
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------ snapping


class PlaceTable:
    """Public places (id, category, lat, lon, popularity) with nearest-within-radius lookup."""

    def __init__(self, ids, categories, lats, lons, popularity=None):
        from scipy.spatial import cKDTree
        self.ids = list(ids)
        self.categories = list(categories)
        self.lat = np.asarray(lats, float)
        self.lon = np.asarray(lons, float)
        self.popularity = (np.asarray(popularity, float) if popularity is not None
                           else np.zeros(len(self.ids)))
        self._lat0 = float(self.lat.mean()) if len(self.lat) else 0.0
        self._kx = M_PER_DEG * math.cos(math.radians(self._lat0))
        self._tree = (cKDTree(np.c_[self.lon * self._kx, self.lat * M_PER_DEG])
                      if len(self.ids) else None)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def empty(cls) -> "PlaceTable":
        return cls([], [], [], [])

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PlaceTable":
        import pandas as pd
        df = pd.read_parquet(path)
        return cls(df["id"], df["category"], df["lat"], df["lon"],
                   df["popularity"] if "popularity" in df else None)

    def nearest(self, lat: float, lon: float, radius_m: float):
        """(place_id, category) of the nearest place within radius_m, or None.
        Ties go to higher popularity among places within 10% of the nearest distance."""
        if self._tree is None:
            return None
        pt = (lon * self._kx, lat * M_PER_DEG)
        idx = self._tree.query_ball_point(pt, radius_m)
        if not idx:
            return None
        d = np.hypot(self._tree.data[idx, 0] - pt[0], self._tree.data[idx, 1] - pt[1])
        dmin = d.min()
        near = [k for k, dk in zip(idx, d) if dk <= dmin * 1.1 + 1.0]
        best = max(near, key=lambda k: (self.popularity[k], -k))
        return self.ids[best], self.categories[best]


def h3_cell(lat: float, lon: float, res: int) -> str:
    import h3
    return h3.latlng_to_cell(lat, lon, res)


def snap_stay(raw: RawStay, places: PlaceTable, res: int = 8, radius_m: float = 100.0) -> Stay:
    """Nearest public place within radius, else 'h3:<cell>' with category 'unknown'."""
    cell = h3_cell(raw.lat, raw.lon, res)
    hit = places.nearest(raw.lat, raw.lon, radius_m)
    if hit is None:
        return Stay(f"h3:{cell}", "unknown", cell, raw.t_start, raw.t_end)
    return Stay(hit[0], hit[1], cell, raw.t_start, raw.t_end)


def merge_adjacent(stays: list[Stay], max_gap_s: int = 1800) -> list[Stay]:
    """Merge chronologically consecutive stays at the same place separated by <= max_gap_s."""
    out: list[Stay] = []
    for s in sorted(stays, key=lambda s: s.t_start):
        if out and out[-1].place_id == s.place_id and s.t_start - out[-1].t_end <= max_gap_s:
            p = out[-1]
            out[-1] = Stay(p.place_id, p.category, p.cell, p.t_start, max(p.t_end, s.t_end))
        else:
            out.append(s)
    return out

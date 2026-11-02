"""Trip segmentation between consecutive stays, travel mode, length."""
from __future__ import annotations

import bisect
from typing import Callable, Sequence

import numpy as np

from geolife.transfer.geo import LabelSpan, haversine_m
from geolife.transfer.types import GpsPoint, Stay, Trip
from geolife.transfer.tokenize.stays import RawStay

LABEL_MODE = {"walk": "walk", "bike": "bike", "bus": "bus", "car": "car", "taxi": "car",
              "train": "train", "subway": "train", "railway": "train", "run": "walk",
              "motorcycle": "car"}


def polyline_length_m(points: Sequence[GpsPoint]) -> float:
    return sum(haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(points, points[1:]))


def label_mode(labels: Sequence[LabelSpan], t0: int, t1: int, min_cover: float = 0.5) -> str | None:
    """Mode of the labelled span covering most of [t0, t1] (needs >= min_cover coverage)."""
    if t1 <= t0:
        return None
    cover: dict[str, int] = {}
    for sp in labels:
        ov = min(t1, sp.t_end) - max(t0, sp.t_start)
        if ov > 0:
            m = LABEL_MODE.get(sp.mode, "unknown")
            cover[m] = cover.get(m, 0) + ov
    if not cover:
        return None
    m = max(cover, key=cover.get)
    return m if cover[m] >= min_cover * (t1 - t0) else None


def speed_mode(points: Sequence[GpsPoint]) -> str:
    """Heuristic from the 80th percentile of 30 s-window speeds (m/s):
    <1.8 walk, <5.5 bike, <30 car (bus/taxi/car are not separable from speed), else train."""
    if len(points) < 2:
        return "unknown"
    v = []
    j = 0
    for i in range(len(points)):
        while j < len(points) - 1 and points[j].t - points[i].t < 30:
            j += 1
        dt = points[j].t - points[i].t
        if dt >= 10:
            v.append(haversine_m(points[i].lat, points[i].lon, points[j].lat, points[j].lon) / dt)
    if not v:
        return "unknown"
    s = float(np.percentile(v, 80))
    if s < 1.8:
        return "walk"
    if s < 5.5:
        return "bike"
    if s < 30:
        return "car"
    return "train"


def segment_trips(points: Sequence[GpsPoint], raws: Sequence[RawStay], stays: Sequence[Stay],
                  labels: Sequence[LabelSpan] = (), min_len_m: float = 200.0,
                  max_gap_s: int = 600,
                  matcher: Callable[[Sequence[GpsPoint], str], tuple[tuple[str, ...], float]] | None = None
                  ) -> tuple[list[Trip], dict[tuple, float]]:
    """Trips between consecutive detected stays of one trajectory.

    `raws[k]` and `stays[k]` correspond. A pair becomes a Trip if the points in between are
    continuous (no gap > max_gap_s) and the path is at least min_len_m. Returns (trips,
    circuity) where circuity[(origin, dest, t_start)] = path length / straight-line distance.
    `matcher(points, mode) -> (segment_ids, matched_length_m)` is optional.
    """
    trips: list[Trip] = []
    circ: dict[tuple, float] = {}
    for k in range(len(raws) - 1):
        a, b = raws[k], raws[k + 1]
        seg = list(points[a.i1:b.i0 + 1])
        if len(seg) < 2 or any(q.t - p.t > max_gap_s for p, q in zip(seg, seg[1:])):
            continue
        length = polyline_length_m(seg)
        if length < min_len_m:
            continue
        t0, t1 = seg[0].t, seg[-1].t
        mode = label_mode(labels, t0, t1) or speed_mode(seg)
        segs: tuple[str, ...] = ()
        if matcher is not None:
            segs, mlen = matcher(seg, mode)
            if segs and mlen > 0:
                length = mlen
        o, d = stays[k], stays[k + 1]
        trips.append(Trip(o.place_id, d.place_id, t0, t1, mode, segs, float(length)))
        straight = haversine_m(a.lat, a.lon, b.lat, b.lon)
        if straight >= 100:
            circ[(o.place_id, d.place_id, t0)] = max(1.0, length / straight)
    return trips, circ

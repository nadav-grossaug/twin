"""GeoLife 1.3 loader: PLT/labels parsing, pseudonymisation, sanity filters.

Pure stdlib + contract types. Times are UTC epoch seconds (GeoLife timestamps are GMT).
"""
from __future__ import annotations

import calendar
import time
from pathlib import Path

from geolife.transfer.geo import GLOBAL_BBOX, LabelSpan, haversine_m, pseudonyms
from geolife.transfer.types import GpsPoint

# Generous Beijing-region box used for the "inside Beijing" share.
BEIJING_BBOX = (39.4, 41.1, 115.4, 117.6)  # lat_min, lat_max, lon_min, lon_max
PLT_HEADER_LINES = 6


def _epoch(date: str, clock: str) -> int:
    """'2007-08-04','03:30:32' -> epoch seconds (UTC)."""
    y, mo, d = (int(x) for x in date.split("-"))
    h, mi, s = (int(x) for x in clock.split(":"))
    return calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0))


def parse_plt(lines) -> list[GpsPoint]:
    """Parse the lines of one .plt file; malformed rows are skipped."""
    out: list[GpsPoint] = []
    for i, line in enumerate(lines):
        if i < PLT_HEADER_LINES:
            continue
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        try:
            out.append(GpsPoint(_epoch(p[5], p[6]), float(p[0]), float(p[1])))
        except ValueError:
            continue
    return out


def read_plt(path: Path) -> list[GpsPoint]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_plt(f)


def parse_labels(lines) -> list[LabelSpan]:
    """Parse labels.txt: header, then 'YYYY/MM/DD HH:MM:SS<TAB>same<TAB>mode'."""
    out: list[LabelSpan] = []
    for i, line in enumerate(lines):
        if i == 0 and line.lower().startswith("start"):
            continue
        p = line.rstrip("\r\n").split("\t")
        if len(p) < 3:
            continue
        try:
            t0 = calendar.timegm(time.strptime(p[0].strip(), "%Y/%m/%d %H:%M:%S"))
            t1 = calendar.timegm(time.strptime(p[1].strip(), "%Y/%m/%d %H:%M:%S"))
        except ValueError:
            continue
        out.append(LabelSpan(t0, t1, p[2].strip().lower()))
    return out


def read_labels(path: Path) -> list[LabelSpan]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_labels(f)


def load_geolife(root: Path, seed: int = 0, max_users: int | None = None,
                 with_labels: bool = False):
    """Load `<root>/Data/<uid>/Trajectory/*.plt`.

    Returns {pseudonym: [trajectory (list[GpsPoint]), ...]} (sorted by file name), plus
    {pseudonym: [LabelSpan]} when with_labels. Real user ids never appear in the output.
    """
    data = Path(root) / "Data"
    uids = sorted(d.name for d in data.iterdir() if d.is_dir())
    if max_users is not None:
        uids = uids[:max_users]
    names = pseudonyms(uids, seed)
    trajs: dict[str, list[list[GpsPoint]]] = {}
    labels: dict[str, list[LabelSpan]] = {}
    for uid in uids:
        tdir = data / uid / "Trajectory"
        files = sorted(tdir.glob("*.plt")) if tdir.is_dir() else []
        trajs[names[uid]] = [t for t in (read_plt(f) for f in files) if t]
        if with_labels:
            lp = data / uid / "labels.txt"
            labels[names[uid]] = read_labels(lp) if lp.exists() else []
    return (trajs, labels) if with_labels else trajs


# --- sanity filters (pure) ---------------------------------------------------------------


def stay_points(points: list[GpsPoint], dist_m: float = 200.0, min_dur_s: int = 1200):
    """Classic stay-point rule: anchor i, extend j while within dist_m of the anchor;
    if elapsed >= min_dur_s emit (mean lat, mean lon, t_start, t_end)."""
    out = []
    n, i = len(points), 0
    while i < n:
        j = i + 1
        while j < n and haversine_m(points[i].lat, points[i].lon,
                                    points[j].lat, points[j].lon) <= dist_m:
            j += 1
        if points[j - 1].t - points[i].t >= min_dur_s:
            seg = points[i:j]
            out.append((sum(p.lat for p in seg) / len(seg), sum(p.lon for p in seg) / len(seg),
                        points[i].t, points[j - 1].t))
            i = j
        else:
            i += 1
    return out


def default_data_dir() -> Path:
    """Dataset root: env GEOLIFE_DATA_DIR if set, else <repo>/geolife/inside/data.

    Either the dir containing 'Geolife Trajectories 1.3' or the GeoLife root itself.
    """
    import os
    env = os.environ.get("GEOLIFE_DATA_DIR")
    base = Path(env) if env else Path(__file__).resolve().parents[3] / "geolife" / "inside" / "data"
    sub = base / "Geolife Trajectories 1.3"
    return sub if sub.is_dir() else base

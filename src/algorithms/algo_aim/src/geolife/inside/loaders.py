"""Readers for the real data: points.parquet caches and per-user label files. Inside only."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from geolife.inside.data import read_labels
from geolife.transfer.geo import LabelSpan
from geolife.transfer.tokenize.tokens import BEIJING_CORE
from geolife.transfer.types import GpsPoint


def load_points_parquet(path: str | Path, bbox=BEIJING_CORE, thin_s: int = 10
                        ) -> dict[str, list[list[GpsPoint]]]:
    """Read the flat points.parquet (uid, traj, t, lat, lon) restricted to `bbox`, thinned to one
    point per `thin_s` seconds per trajectory (done in pandas before building objects)."""
    df = pd.read_parquet(path, columns=["uid", "traj", "t", "lat", "lon"])
    lat0, lat1, lon0, lon1 = bbox
    df = df[(df.lat >= lat0) & (df.lat <= lat1) & (df.lon >= lon0) & (df.lon <= lon1)]
    df = df.assign(b=df.t // thin_s).drop_duplicates(["uid", "traj", "b"]).sort_values(
        ["uid", "traj", "t"])
    out: dict[str, list[list[GpsPoint]]] = {}
    for (uid, _), g in df.groupby(["uid", "traj"], sort=True):
        out.setdefault(str(uid), []).append(
            [GpsPoint(int(t), float(la), float(lo)) for t, la, lo in zip(g.t, g.lat, g.lon)])
    return out


def load_labels_for(root: str | Path, uids: Sequence[str]) -> dict[str, list[LabelSpan]]:
    """labels.txt per user from the raw dataset tree (<root>/Data/<uid>/labels.txt)."""
    out = {}
    for uid in uids:
        p = Path(root) / "Data" / uid / "labels.txt"
        if p.exists():
            out[uid] = read_labels(p)
    return out


def load_points_arrays(path: Path) -> dict[str, tuple]:
    """points.parquet -> {uid: (t, lat, lon) arrays} (input of tokenize.outings.user_outings_from_arrays)."""
    df = pd.read_parquet(path, columns=["uid", "t", "lat", "lon"])
    out = {}
    for uid, g in df.groupby("uid", sort=True):
        out[str(uid)] = (g.t.to_numpy(np.int64), g.lat.to_numpy(float), g.lon.to_numpy(float))
    return out

"""Overture Maps loader: bbox -> pandas DataFrame, cached as parquet.

Geometry is kept as WKB bytes (column ``geometry``); no duckdb spatial extension is needed
because Overture parquet carries a ``bbox`` struct usable for predicate pushdown.
No network or duckdb here (shipped code): fetching is geolife/outside/fetch_public_map.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

S3_ROOT = "s3://overturemaps-us-west-2/release"
DEFAULT_RELEASE = "2026-10-21.0"
THEMES = {
    "connector": "transportation",
    "segment": "transportation",
    "place": "places",
    "building": "buildings",
}
BBox = tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _cache_path(cache_dir: Path, release: str, ftype: str, bbox: BBox) -> Path:
    key = hashlib.sha1(repr((release, ftype, bbox)).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"overture_{release}_{ftype}_{key}.parquet"


def load_overture(
    ftype: str,
    bbox: BBox,
    cache_dir: str | Path = "geolife/outside/cache",
    release: str = DEFAULT_RELEASE,
    root: str | None = None,
    fetch=None,
) -> pd.DataFrame:
    """Return Overture features of ``ftype`` fully inside ``bbox`` from the parquet cache.

    On a cache miss ``fetch(theme, ftype, "*", bbox, release, root)`` is called (the duckdb fetcher lives in
    geolife/outside/fetch_public_map.py) and its result cached; without ``fetch`` a miss raises FileNotFoundError.
    """
    if ftype not in THEMES:
        raise ValueError(f"unknown type {ftype!r}")
    cache = _cache_path(Path(cache_dir), release, ftype, bbox)
    if cache.exists():
        return pd.read_parquet(cache)
    if fetch is None:
        raise FileNotFoundError(f"{cache} is not cached; run geolife/outside/fetch_public_map.py (needs network)")
    df = fetch(THEMES[ftype], ftype, "*", bbox, release, root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df

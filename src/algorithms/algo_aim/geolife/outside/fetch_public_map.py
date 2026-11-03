"""PUBLIC-MAP FETCHER (needs network and duckdb; run it on an internet-connected machine).

Downloads Overture Maps data for a bbox (public data, nothing from the real traces) into the parquet cache that
`geolife.outside.map.provider.OvertureProvider` reads, and writes the normalised places table.

    PYTHONPATH=src python geolife/outside/fetch_public_map.py --name beijing 116.25 39.85 116.45 40.05 [--all]
    (bbox order: min_lon min_lat max_lon max_lat)  ->  geolife/outside/cache/places_<name>.parquet

`--all` also fetches buildings and roads. The cache dir is gitignored. This script is NOT part of the shipped stage-2
code: it is the only place that imports duckdb or touches the network, so `src/geolife/{transfer,outside}` stay free
of them. It may import only transfer / outside code (the boundary test checks that).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from geolife.outside.map.overture import DEFAULT_RELEASE, S3_ROOT
from geolife.outside.map.provider import CACHE_DIR, fetch_buildings, fetch_places, fetch_roads, place_table


def duckdb_fetch(theme, kind, cols, bbox, release=DEFAULT_RELEASE, root=None):
    """`fetch` callable for provider.fetch_* / overture.load_overture: Overture parquet -> DataFrame."""
    import duckdb  # only here

    base = root or f"{S3_ROOT}/{release}"
    con = duckdb.connect()
    if base.startswith("s3://"):
        con.execute("LOAD httpfs; SET s3_region='us-west-2'")
    x0, y0, x1, y1 = bbox
    return con.execute(
        f"SELECT {cols} FROM read_parquet('{base}/theme={theme}/type={kind}/*', hive_partitioning=1) "
        "WHERE bbox.xmin > ? AND bbox.xmax < ? AND bbox.ymin > ? AND bbox.ymax < ?", [x0, x1, y0, y1]).df()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--all", action="store_true", help="also fetch buildings + roads into the cache")
    a = ap.parse_args(argv)
    bbox = tuple(a.bbox)
    t = time.time()
    raw = fetch_places(bbox, a.cache_dir, fetch=duckdb_fetch)
    tab = place_table(raw)
    out = Path(a.cache_dir) / f"places_{a.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out, index=False)
    print(f"places: {len(tab)} rows in {time.time() - t:.0f}s -> {out}")
    print(tab["category"].value_counts().to_dict())
    if a.all:
        t = time.time()
        b = fetch_buildings(bbox, a.cache_dir, fetch=duckdb_fetch)
        print(f"buildings: {len(b)} in {time.time() - t:.0f}s")
        t = time.time()
        s, c = fetch_roads(bbox, a.cache_dir, fetch=duckdb_fetch)
        print(f"segments: {len(s)}, connectors: {len(c)} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()

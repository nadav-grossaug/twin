"""MapProvider from a small json-able spec, so a stage-2 run can record the map it used (meta.json) and the eval
stage can rebuild the same map. Public data only.

    {"kind": "grid", "n": 40, "spacing": 250.0, "n_places": 6000, "seed": 1}          toy Manhattan grid
    {"kind": "overture", "bbox": [min_lon, min_lat, max_lon, max_lat], "cache_dir": "geolife/outside/cache",
     "max_poi": 15000, "n_homes": 1500, "seed": 0}                                    Overture cache (read-only)
"""
from __future__ import annotations

from geolife.outside.grid_city import GridCityProvider


def make_provider(spec: dict):
    kind = spec.get("kind")
    if kind == "grid":
        return GridCityProvider(n=int(spec.get("n", 40)), spacing=float(spec.get("spacing", 250.0)),
                                n_places=int(spec.get("n_places", 6000)), seed=int(spec.get("seed", 1)))
    if kind == "overture":
        from geolife.outside.map.provider import CACHE_DIR, OvertureProvider
        return OvertureProvider(tuple(spec["bbox"]), cache_dir=spec.get("cache_dir", CACHE_DIR),
                                max_poi=spec.get("max_poi"), n_homes=int(spec.get("n_homes", 1500)),
                                seed=int(spec.get("seed", 0)))
    raise ValueError(f"unknown map kind {kind!r}")

"""STAGE 2 (outside): artifacts -> synthetic GPS traces + planted truth. Never sees real data.

Needs only Python >= 3.11 with numpy, pandas, scipy (+ h3, networkx, shapely, pyarrow for the shared code), the
`src/geolife/{transfer,outside}` folders and a copied `geolife/transfer/artifacts/`:

    PYTHONPATH=src python geolife/outside/generate_synthetic.py [--only stays_v2] [--artifacts DIR] [--out DIR]
        [--n-users 200] [--seed K] [--map grid|overture] [--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT]
        [--cache-dir geolife/outside/cache] [--gen-json '{"max_outings": 60}']

Every artifact folder `<variant>_seed<k>/` under --artifacts becomes `<out>/<variant>_seed<k>/{traces.parquet,
outings.json, truth.json, meta.json}` (see geolife.outside.synthetic_io) plus `<out>/index.json`. Loading an artifact
verifies its SHA-256 hashes. The target map is public: a toy grid city, or an Overture cache fetched beforehand with
geolife/outside/fetch_public_map.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from geolife.outside.pipeline_v2 import generate_v2
from geolife.outside.providers import make_provider
from geolife.outside.synthetic_io import write_synthetic
from geolife.transfer.artifact import load_artifact

NAME_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="geolife/transfer/artifacts")
    ap.add_argument("--out", default="geolife/outside/synthetic")
    ap.add_argument("--only", default=None, help="only variants starting with this prefix")
    ap.add_argument("--n-users", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None, help="generator seed (default: the artifact's seed number)")
    ap.add_argument("--map", choices=["grid", "overture"], default="grid")
    ap.add_argument("--map-n", type=int, default=40, help="grid: blocks per side")
    ap.add_argument("--map-spacing", type=float, default=250.0)
    ap.add_argument("--map-places", type=int, default=6000)
    ap.add_argument("--map-seed", type=int, default=1)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, help="overture: min_lon min_lat max_lon max_lat")
    ap.add_argument("--cache-dir", default="geolife/outside/cache")
    ap.add_argument("--max-poi", type=int, default=15000)
    ap.add_argument("--gen-json", default=None, help="generator knob overrides, e.g. '{\"max_outings\": 60}'")
    a = ap.parse_args(argv)

    if a.map == "overture":
        if not a.bbox:
            ap.error("--map overture needs --bbox")
        spec = {"kind": "overture", "bbox": list(a.bbox), "cache_dir": a.cache_dir, "max_poi": a.max_poi}
    else:
        spec = {"kind": "grid", "n": a.map_n, "spacing": a.map_spacing, "n_places": a.map_places, "seed": a.map_seed}
    gen = json.loads(a.gen_json) if a.gen_json else None
    provider = make_provider(spec)

    root, out = Path(a.artifacts), Path(a.out)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").is_file())
    if a.only:
        dirs = [d for d in dirs if d.name.startswith(a.only)]
    if not dirs:
        print(f"no artifacts found under {root}", file=sys.stderr)
        return 1
    index = []
    for d in dirs:
        m = NAME_RE.match(d.name)
        variant, k = (m["variant"], int(m["seed"])) if m else (d.name, 0)
        seed = a.seed if a.seed is not None else k
        art = load_artifact(d)                      # verifies every hash
        g = generate_v2(d, provider, a.n_users, seed, gen=gen)
        meta = {"variant": variant, "artifact": d.name, "seed": seed, "n_users": a.n_users, "map": spec,
                "generation": {**(art.generation or {}), **(gen or {})},
                "epsilon": art.ledger.epsilon, "delta": art.ledger.delta,
                "learner": art.manifest.learner}
        write_synthetic(out / d.name, g.result, meta)
        n_pts = sum(len(v) for v in g.result.traces.values())
        index.append({"variant": variant, "seed": seed, "dir": d.name, "n_users": a.n_users, "n_points": n_pts})
        print(f"{d.name}: {a.n_users} users, {n_pts} points -> {out / d.name}", file=sys.stderr)
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

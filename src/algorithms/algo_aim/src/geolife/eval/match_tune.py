"""Evaluate and tune the map matcher on synthetic noisy paths over a (real) road graph.

CLI: python -m geolife.eval.match_tune --cache geolife/outside/cache
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random

import networkx as nx
import numpy as np

from geolife.transfer.map.graph import build_graph
from geolife.transfer.map.match import Matcher, evaluate_match

KY = 111195.0


def largest_component(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def synth_path(G, rng: random.Random, np_rng: np.random.Generator, min_len_m=1500.0,
               spacing_m=25.0, noise_m=8.0, drivable_only=False):
    """Random shortest path between two nodes; returns (truth_segment_ids, noisy [(lon,lat)]).

    Points are laid every `spacing_m` along the true geometry (roughly 10-15 m/s at 1-2 s
    sampling thinned) with isotropic Gaussian noise `noise_m` (m)."""
    from geolife.transfer.map.match import WALK_CLASSES
    H = G
    if drivable_only:
        H = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True)
                             if d.get("class_") not in WALK_CLASSES]).copy()
        H = largest_component(H)
    nodes = list(H.nodes)
    for _ in range(200):
        a, b = rng.sample(nodes, 2)
        try:
            path = nx.shortest_path(H, a, b, weight="length")
        except nx.NetworkXNoPath:
            continue
        length = sum(H[u][v]["length"] for u, v in zip(path, path[1:]))
        if length < min_len_m:
            continue
        coords, truth = [], []
        for u, v in zip(path, path[1:]):
            d = H[u][v]
            c = list(d["geom"].coords)
            # orient geometry from u to v
            nu = (H.nodes[u]["lon"], H.nodes[u]["lat"])
            if (c[0][0] - nu[0]) ** 2 + (c[0][1] - nu[1]) ** 2 > (c[-1][0] - nu[0]) ** 2 + (c[-1][1] - nu[1]) ** 2:
                c = c[::-1]
            coords.extend(c if not coords else c[1:])
            if not truth or truth[-1] != d["segment_id"]:
                truth.append(d["segment_id"])
        lat0 = coords[0][1]
        kx = KY * math.cos(math.radians(lat0))
        xy = np.array([(x * kx, y * KY) for x, y in coords])
        seg = np.hypot(*np.diff(xy, axis=0).T)
        cum = np.r_[0, np.cumsum(seg)]
        s = np.arange(0, cum[-1], spacing_m)
        px, py = np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])
        px += np_rng.normal(0, noise_m, len(s))
        py += np_rng.normal(0, noise_m, len(s))
        return truth, [(float(x / kx), float(y / KY)) for x, y in zip(px, py)]
    raise RuntimeError("no path found")


def evaluate(G, n_paths=12, seed=0, noise_m=8.0, mode=None, drivable_only=False,
             min_len_m=1500.0, **matcher_kw):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    m = Matcher(G, **matcher_kw)
    rows = []
    for _ in range(n_paths):
        truth, pts = synth_path(G, rng, nrng, min_len_m=min_len_m, noise_m=noise_m,
                                 drivable_only=drivable_only)
        rows.append(evaluate_match(G, truth, m.match(pts, mode)))
    keys = ["recovered_rate", "extra_rate", "true_len_covered", "matched_len_correct"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["n_true"] = int(sum(r["n_true"] for r in rows))
    out["n_matched"] = int(sum(r["n_matched"] for r in rows))
    out["recovered"] = int(sum(r["recovered"] for r in rows))
    out["extra"] = int(sum(r["extra"] for r in rows))
    return out


def load_graph(cache_dir: str, road_only=True):
    import glob
    import pandas as pd
    seg = pd.read_parquet(sorted(glob.glob(f"{cache_dir}/overture_*_segment_*.parquet"))[0])
    con = pd.read_parquet(sorted(glob.glob(f"{cache_dir}/overture_*_connector_*.parquet"))[0])
    if road_only and "subtype" in seg:
        seg = seg[seg["subtype"] == "road"]
    return largest_component(build_graph(seg, con))


BASELINE = dict(subsample_m=0, collapse_parallel=False, use_prior=False, postprocess=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="geolife/outside/cache")
    ap.add_argument("--paths", type=int, default=12)
    ap.add_argument("--noise", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    G = load_graph(a.cache)
    res = {"graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}, "runs": {}}
    for label, mode, drv, kw in [
        ("baseline (old matcher), any-road paths", None, False, BASELINE),
        ("improved, any-road paths, mode unknown", None, False, {}),
        ("baseline, drivable paths", "car", True, BASELINE),
        ("improved, drivable paths, mode=car", "car", True, {}),
    ]:
        res["runs"][label] = evaluate(G, a.paths, 0, a.noise, mode, drv, **kw)
        print(label, json.dumps(res["runs"][label]), flush=True)
    grid = {}
    for sigma, beta, sub in itertools.product((8.0, 12.0, 20.0), (10.0, 20.0, 40.0), (15.0, 30.0)):
        r = evaluate(G, max(6, a.paths // 2), 1, a.noise, "car", True, sigma=sigma, beta=beta,
                     subsample_m=sub)
        grid[f"sigma={sigma},beta={beta},sub={sub}"] = r
        print(sigma, beta, sub, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)},
              flush=True)
    res["grid_car_drivable"] = grid
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()

"""Map matching on REAL GPS (eval zone; PLAN step 8 task 3).

The HMM matcher was tuned on synthetic Gaussian-noise paths (`src/geolife/eval/match_tune.py`) and was
never run on real traces: M on real data uses the res-9 cell proxy. This script runs the matcher, exactly
as M v2 configures it (`M2Config`: match_thin_s 5 s, subsample 60 m, postprocess off), on a sample of real
Beijing outings over the cached Overture road graph, and measures

  * matcher diagnostics: share of outings that match at all, segments per outing, matched-path length vs
    the outing's GPS length, seconds per outing;
  * agreement with the res-9 traversed-cell proxy: the matched segments are rasterised to res-9 cells and
    compared with the outing's proxy cells (coverage of the proxy by the matched path, and the share of the
    matched path inside the 1-ring-dilated proxy = "on-trace" precision).

Reads real points (inside data; eval may). Writes an aggregate JSON only (no per-user rows, no coordinates).

Run from the project root:
  PYTHONPATH=src python geolife/eval/match_real.py --points <points.parquet> \
      --cache-dir geolife/outside/cache [--users 8] [--max-outings 60] [--out ...json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from geolife.eval.m.outings import M2Config
from geolife.outside.map.provider import ROAD_EXCLUDE, fetch_roads
from geolife.transfer.map.graph import build_graph
from geolife.transfer.map.match import Matcher
from geolife.transfer.tokenize.outings import PlaceIndex, user_outings_from_arrays

BEIJING = (116.15, 39.75, 116.60, 40.10)


def road_graph(cache_dir, bbox=BEIJING):
    """Road graph as the generator's OvertureProvider builds it (subtype road, pedestrian classes
    excluded, largest connected component)."""
    import networkx as nx
    seg, con = fetch_roads(bbox, cache_dir)
    if "subtype" in seg:
        seg = seg[seg["subtype"].fillna("road") == "road"]
    if "class" in seg:
        seg = seg[~seg["class"].isin(ROAD_EXCLUDE)]
    G = build_graph(seg, con)
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def seg_cells(G, res: int = 9, every_m: float = 30.0) -> dict[str, set]:
    """segment id -> set of res-9 cells sampled along its geometry (planar interpolation is fine at 30 m)."""
    out: dict[str, set] = {}
    for _, _, d in G.edges(data=True):
        s = out.setdefault(str(d["segment_id"]), set())
        c = np.asarray(d["geom"].coords)
        n = max(2, int(d["length"] / every_m) + 1)
        f = np.linspace(0, 1, n)
        lon = np.interp(f, np.linspace(0, 1, len(c)), c[:, 0])
        lat = np.interp(f, np.linspace(0, 1, len(c)), c[:, 1])
        s.update(h3.latlng_to_cell(a, b, res) for a, b in zip(lat.tolist(), lon.tolist()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=os.environ.get("GEOLIFE_POINTS"))
    ap.add_argument("--cache-dir", default="geolife/outside/cache")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--max-outings", type=int, default=60, help="matched outings per user")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="geolife/eval/results/match_real.json")
    a = ap.parse_args(argv)

    t0 = time.time()
    G = road_graph(a.cache_dir)
    sc = seg_cells(G)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(sc)} segments ({time.time() - t0:.0f}s)")

    cfg = M2Config()
    matcher = Matcher(G, subsample_m=cfg.match_subsample_m, postprocess=False)
    tok = cfg.tok

    df = pd.read_parquet(a.points, columns=["uid", "t", "lat", "lon"])
    x0, y0, x1, y1 = BEIJING
    df = df[(df.lon >= x0) & (df.lon <= x1) & (df.lat >= y0) & (df.lat <= y1)]
    counts = df.groupby("uid").size()
    eligible = counts[counts >= 50_000].index.to_numpy()      # users with real Beijing volume
    rng = np.random.default_rng(a.seed)
    sample = rng.choice(eligible, size=min(a.users, len(eligible)), replace=False)

    rows = []
    for uid in sorted(sample.tolist()):
        g = df[df.uid == uid].sort_values("t", kind="stable")
        t = g.t.to_numpy(np.int64)
        uo = user_outings_from_arrays(str(uid), t, g.lat.to_numpy(float), g.lon.to_numpy(float),
                                      PlaceIndex(None), tok)
        if uo is None:
            continue
        lat, lon = g.lat.to_numpy(float), g.lon.to_numpy(float)
        outs = uo.outings
        if len(outs) > a.max_outings:
            idx = rng.choice(len(outs), size=a.max_outings, replace=False)
            outs = [outs[i] for i in sorted(idx)]
        for o in outs:
            i0, i1 = np.searchsorted(t, o.t_start), np.searchsorted(t, o.t_end, side="right")
            tt = t[i0:i1]
            m = np.r_[True, np.diff(tt) >= tok.match_thin_s]
            pts = list(zip(lon[i0:i1][m].tolist(), lat[i0:i1][m].tolist()))
            tm = time.time()
            segs, mlen = matcher._match(pts, o.mode)
            dt_m = time.time() - tm
            proxy = set(o.cells)
            got = set().union(*[sc.get(s, set()) for s in segs]) if segs else set()
            dil = set().union(*[set(h3.grid_disk(c, 1)) for c in proxy]) if proxy else set()
            rows.append({"user": str(uid), "mode": o.mode, "n_pts": len(pts),
                         "gps_len_km": o.length_m / 1000, "match_len_km": mlen / 1000,
                         "n_segs": len(segs), "n_proxy_cells": len(proxy),
                         "proxy_covered": len(got & proxy) / len(proxy) if proxy else float("nan"),
                         "match_on_trace": len(got & dil) / len(got) if got else float("nan"),
                         "secs": dt_m})
        print(f"user done: {len(rows)} outings so far ({time.time() - t0:.0f}s)", flush=True)

    r = pd.DataFrame(rows)
    matched = r[r.n_segs > 0]
    agg = {
        "n_users": int(r.user.nunique()), "n_outings": int(len(r)),
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
        "share_outings_matched": float((r.n_segs > 0).mean()),
        "segments_per_outing": {"mean": float(r.n_segs.mean()), "p50": float(r.n_segs.median()),
                                 "p90": float(r.n_segs.quantile(0.9))},
        "match_len_over_gps_len": {"p25": float((matched.match_len_km / matched.gps_len_km).quantile(0.25)),
                                   "p50": float((matched.match_len_km / matched.gps_len_km).median()),
                                   "p75": float((matched.match_len_km / matched.gps_len_km).quantile(0.75))},
        "proxy_covered": {"mean": float(matched.proxy_covered.mean()),
                          "p50": float(matched.proxy_covered.median())},
        "match_on_trace": {"mean": float(matched.match_on_trace.mean()),
                           "p50": float(matched.match_on_trace.median())},
        "secs_per_outing": {"mean": float(r.secs.mean()), "p90": float(r.secs.quantile(0.9)),
                            "max": float(r.secs.max())},
        "by_mode": {m: {"n": int(len(gq)), "share_matched": float((gq.n_segs > 0).mean()),
                        "proxy_covered": float(gq[gq.n_segs > 0].proxy_covered.mean())
                        if (gq.n_segs > 0).any() else float("nan")}
                    for m, gq in r.groupby("mode")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(agg, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

"""Teleport demo: fake params (outside code only, no real data) -> real Overture provider (central Amsterdam) -> 30 synthetic users -> PNG.

    PYTHONPATH=src python geolife/outside/reports/teleport_demo.py   (needs the geolife/outside/cache fetched by geolife/outside/fetch_public_map.py,
    or network access for the first run)
"""
import cProfile
import pstats
import sys
import time

import numpy as np

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.map.provider import OvertureProvider
from geolife.outside.pipeline_v2 import generate_v2

BBOX = (4.85, 52.33, 4.95, 52.41)   # central Amsterdam
N, SEED = 30, 3


def main(profile: bool = False):
    t = time.time()
    prov = OvertureProvider(BBOX, max_poi=6000, n_homes=1500)
    print(f"provider: {len(prov.places())} places, {len(prov._ids)} road nodes, {time.time() - t:.1f}s")
    t = time.time()
    if profile:
        pr = cProfile.Profile(); pr.enable()
    g = generate_v2(fake_params_v2(), prov, N, SEED, gen={**fake_generation_v2(), "length_scale": 0.3, "max_outings": 60})
    if profile:
        pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
    dt = time.time() - t
    npts = sum(len(v) for v in g.result.traces.values())
    print(f"generate(n={N}): {dt:.1f}s, {npts} GPS points; teleport report {g.teleport.report}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    for e in prov._G.edges(data=True):
        xy = np.asarray(e[2]["geom"].coords)
        x, y = prov._xy(xy[:, 0], xy[:, 1])
        ax.plot(x, y, color="#c8c8c8", lw=0.3)
    cmap = plt.get_cmap("tab20")
    for i, (uid, pts) in enumerate(g.result.traces.items()):
        lat = np.array([p.lat for p in pts]); lon = np.array([p.lon for p in pts])
        x, y = prov._xy(lon, lat)
        ax.plot(x, y, ".", ms=1.2, color=cmap(i % 20), alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(f"{N} synthetic users teleported onto central Amsterdam (Overture roads)")
    fig.savefig("geolife/outside/reports/teleport-demo.png", dpi=110, bbox_inches="tight")
    print("wrote geolife/outside/reports/teleport-demo.png")


if __name__ == "__main__":
    main("--profile" in sys.argv)

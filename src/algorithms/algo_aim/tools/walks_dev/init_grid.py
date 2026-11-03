"""Throwaway: walk energy (fresh 40k pool) of the v2 initializer with pinned regions (R, q)."""
import sys, time
import numpy as np
from graph_aim.outside.generate_v2 import generate_v2_with_info
from graph_aim_walks.outside import energy as en, pool as pl
from graph_aim_walks.transfer import walkstats as ws
from graph_aim_walks.transfer.artifact import load

art = load("snap/transfer/artifacts_walks/graph_aim_walks_seed0")
real = {k: v for k, v in art.tables.items() if k != "transition_spectrum"}
for arg in sys.argv[1:]:
    R, q = arg.split(",")
    t = time.time()
    kw = {} if R == "auto" else dict(regions_R=int(R), within_region_frac=float(q))
    g, info = generate_v2_with_info(art.base, seed=0, **kw)
    attrs = {c: g.attrs[c].to_numpy() for c in g.attrs.columns}
    ip, ix = ws.csr(g.edges, g.n); deg = np.diff(ip)
    st, U = ws.draw_pool(deg, 40000, 32, np.random.default_rng(5))
    P = pl.WalkPool(ip, ix, st, U, pl.node_arrays(deg, attrs, g.domains, art.spec), art.spec)
    e = en.walk_energy(P.counts, P.P, real, 40000, art.n_walks)
    tot = sum(v["energy"] for v in e.values())
    print(R, q, round(tot), {k[:12]: round(v["energy"] / 1000) for k, v in e.items()}, round(time.time() - t), "s", flush=True)

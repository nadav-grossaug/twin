"""Throwaway: predicted (screen) vs exact (validation) change of structure batches on the real artifact."""
import sys
import numpy as np
from pathlib import Path
from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim_walks.outside import generate as G
from graph_aim_walks.outside import pool as pl
from graph_aim_walks.transfer.artifact import load

REPO = Path(__file__).resolve().parents[2]
art = load(REPO / "snap/transfer/artifacts_walks/graph_aim_walks_seed0")
import os
g = load_attributed_graph(os.environ.get("G", REPO / "tools/walks_dev/cache/v2_seed0.npz"))
kw = dict(pool_size=40000, val_pool_size=None, batch=2000, label_batch=500, damping=0.25, retries=0,
          temperature=0.0, local_frac=0.5, weights=None, grad_every=4, base_weight=1.0, base_weights=None, pool_screen=0.0, class_frac=0.0, class_keys=None, share_surrogate=1.0, chunk=150, label_chunk=60)
extra = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])}
NB = int(extra.pop("n", 4)); kw.update({k: (int(v) if k in ("batch","retries") or k=="class_keys" else v) for k, v in extra.items()})
gen = G._Generator(art, g, 0, **kw)
gen._draw_pools(0)
gen._start_structure_phase()
rng = np.random.default_rng(0)
orig_try = gen._try_structure
def wrapped(sel, a, b, c, d, sa, sb, sc, sd, ei, ej, rng_):
    B = gen.B
    eB0 = gen._walk_total(B.counts, B.P); dd0 = gen.walk_energy(B.counts, B.P); eA0 = gen._walk_total(gen.A.counts, gen.A.P)
    base0 = sum(v["energy"] for v in gen.base_energy().values())
    ok = orig_try(sel, a, b, c, d, sa, sb, sc, sd, ei, ej, rng_)
    print(f"  try {len(sel)} ok={ok}", end="")
    if ok:
        print(f" exact dB={gen._walk_total(B.counts, B.P)-eB0:.0f} dA={gen._walk_total(gen.A.counts, gen.A.P)-eA0:.0f}"
              f" dbase={sum(v['energy'] for v in gen.base_energy().values())-base0:.0f}")
        dd1 = gen.walk_energy(B.counts, B.P)
        print("     ", {k[:14]: round(dd1[k]["energy"] - dd0[k]["energy"]) for k in dd0})
    else:
        d1 = gen.walk_energy(gen.last_try["cnt"], B.P)
        print(" base", round(gen.last_try["base"]), "walk", round(gen.last_try["walk"]), {k[:14]: round(d1[k]["energy"] - dd0[k]["energy"]) for k in dd0})
    return ok
gen._try_structure = wrapped
print({k: round(v) for k, v in gen.surrogate_lambda.items()})
for i in range(NB):
    st = {"proposed": 0, "screened": 0, "accepted": 0, "batches_accepted": 0, "batches_rejected": 0}
    gen._structure_batch(rng, st)
    print(st)

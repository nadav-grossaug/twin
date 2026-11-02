"""Per-seed hub-tail mass (nodes with degree >= 275 in the processed histogram) and signed metrics."""

import json
import sys
from pathlib import Path

import numpy as np

from graph_aim.eval.evaluate import structure_metrics
from graph_aim.inside.fit import fit
from graph_aim.outside.generate import generate
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
rm = structure_metrics(real)
cfg = json.loads((INSIDE / "config" / "graph_aim_v1.json").read_text())
print("real", {k: rm[k] for k in ("assortativity", "triangles", "degeneracy", "transitivity")},
      "hubs>=275:", int((np.bincount(real.edges.ravel()) >= 275).sum()))
for eps in [float(x) for x in sys.argv[1:]] or [2.0, 4.0, 8.0]:
    for seed in range(6):
        art = fit(real, dict(cfg, epsilon_edge=eps, epsilon_attr=eps), seed=seed)
        h = art.measurements["degree_hist"].values
        syn = generate(art, seed=seed)
        m = structure_metrics(syn)
        print(f"eps={eps} seed={seed} tail_nodes={h[275:].sum():6.2f} assort={m['assortativity']:.3f} "
              f"tri={m['triangles']:.0f} degen={m['degeneracy']} trans={m['transitivity']:.3f} "
              f"maxdeg={np.bincount(syn.edges.ravel()).max()}", flush=True)

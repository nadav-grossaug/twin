"""Compare exact and noisy degree_hist (tail mass, edges) at one epsilon."""

import json
import sys
from pathlib import Path

import numpy as np

from graph_aim.inside.fit import fit
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
cfg = json.loads((INSIDE / "config" / "graph_aim_v1.json").read_text())
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
ex = fit(real, dict(cfg, noise=False), seed=0).measurements["degree_hist"]
nz = fit(real, dict(cfg, epsilon_edge=eps, epsilon_attr=eps), seed=0).measurements["degree_hist"]
print("sigma", nz.sigma, "hist_edges", nz.params.get("hist_edges"))
d = np.arange(len(ex.values))
for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 150), (150, 300), (300, 600), (600, 1201)]:
    sl = slice(lo, hi)
    print(f"deg [{lo:4d},{hi:4d}) nodes exact {ex.values[sl].sum():8.1f} noisy {nz.values[sl].sum():8.1f}"
          f"  volume exact {(ex.values[sl] * d[sl]).sum():9.0f} noisy {(nz.values[sl] * d[sl]).sum():9.0f}")
print("total volume", (ex.values * d).sum(), (nz.values * d).sum())

"""Degree assortativity of the bulk (edges not touching hubs) for real, exact-artifact and noisy-artifact graphs."""

import json
from pathlib import Path

import numpy as np

from graph_aim.inside.fit import fit
from graph_aim.outside.generate import generate
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
cfg = json.loads((INSIDE / "config" / "graph_aim_v1.json").read_text())


def assort(g, hub=None):
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    e = g.edges
    if hub is not None:
        e = e[(deg[e[:, 0]] < hub) & (deg[e[:, 1]] < hub)]
    a = np.concatenate([deg[e[:, 0]], deg[e[:, 1]]]).astype(float)
    b = np.concatenate([deg[e[:, 1]], deg[e[:, 0]]]).astype(float)
    return np.corrcoef(a, b)[0, 1], len(e)


def line(label, g):
    print(f"{label:20s} all {assort(g)[0]:.3f}  no-hub(<300) {assort(g, 300)[0]:.3f} edges {len(g.edges)} maxdeg {np.bincount(g.edges.ravel()).max()}")


line("real", real)
for label, c in [("exact", dict(cfg, noise=False)), ("eps8", dict(cfg, epsilon_edge=8, epsilon_attr=8))]:
    art = fit(real, c, seed=0)
    for s in range(2):
        line(f"{label} seed{s}", generate(art, seed=s))

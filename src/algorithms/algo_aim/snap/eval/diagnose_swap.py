"""Swap measurement kinds between a noisy and an exact artifact; print assortativity abs error.

Usage (repo root): ``PYTHONPATH=src python snap/eval/diagnose_swap.py [eps]``
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

from graph_aim.eval.evaluate import compare, structure_metrics
from graph_aim.inside.fit import fit
from graph_aim.outside.generate import generate
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
rm = structure_metrics(real)
cfg = json.loads((INSIDE / "config" / "graph_aim_v1.json").read_text())
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
ex = fit(real, dict(cfg, noise=False), seed=0)
nz = fit(real, dict(cfg, epsilon_edge=eps, epsilon_attr=eps), seed=0)
kinds = ["degree_hist", "clustering_by_degree", "joint_degree", "attr_edge_pair", "attr_2way"]


def swap(base, other, ks):
    a = copy.copy(base)
    a.measurements = dict(base.measurements)
    for n, m in other.measurements.items():
        if m.kind in ks:
            a.measurements[n] = m
    return a


rows = [("exact", ex), ("noisy", nz)]
rows += [(f"noisy, exact {k}", swap(nz, ex, [k])) for k in kinds]
rows += [("exact, noisy degree_hist", swap(ex, nz, ["degree_hist"]))]
rows += [("exact, noisy hist+clust", swap(ex, nz, ["degree_hist", "clustering_by_degree"]))]
for label, art in rows:
    r = [compare(real, generate(art, seed=s), cfg["homophily_attrs"], 0, rm)["assortativity_abs"] for s in range(2)]
    print(f"{label:32s} {np.mean(r):.3f}", flush=True)

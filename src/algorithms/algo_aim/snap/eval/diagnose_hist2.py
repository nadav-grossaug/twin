"""Which part of the degree histogram drives generated assortativity? (exact artifact, edited hist)"""

import copy
import json
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
ex = fit(real, dict(cfg, noise=False), seed=0)
nz = fit(real, dict(cfg, epsilon_edge=8, epsilon_attr=8), seed=0)
h_ex = ex.measurements["degree_hist"].values.copy()
h_nz = nz.measurements["degree_hist"].values.copy()


def with_hist(h):
    a = copy.copy(ex)
    a.measurements = dict(ex.measurements)
    m = copy.copy(ex.measurements["degree_hist"])
    m.values = h
    a.measurements["degree_hist"] = m
    return a


def trunc(h, d):
    h = h.copy()
    h[d:] = 0
    return h


def smooth_ex(h, w):
    """Exact hist spread inside coarse cells ~1/d (what a noise-free coarse release would give)."""
    grid = nz.measurements["degree_hist"].params["hist_edges"]
    out = np.zeros_like(h)
    for lo, hi in zip(grid[:-1], grid[1:]):
        d = np.arange(lo, hi)
        wt = 1.0 / np.maximum(d, 1)
        out[lo:hi] = h[lo:hi].sum() * wt / wt.sum()
    return out


cases = {
    "exact hist": h_ex,
    "exact, drop deg>=300": trunc(h_ex, 300),
    "exact, drop deg>=600": trunc(h_ex, 600),
    "exact coarse-smoothed": smooth_ex(h_ex, 1),
    "coarse-smoothed, drop >=300": trunc(smooth_ex(h_ex, 1), 300),
    "noisy hist": h_nz,
    "noisy hist + exact hubs>=300": np.where(np.arange(len(h_nz)) >= 300, h_ex, h_nz),
}
for label, h in cases.items():
    r = [compare(real, generate(with_hist(h), seed=s), cfg["homophily_attrs"], 0, rm)["assortativity_abs"] for s in range(2)]
    print(f"{label:32s} {np.mean(r):.3f}", flush=True)

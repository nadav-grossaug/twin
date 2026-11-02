"""Metric table of the private pipeline (fit + generate) at several epsilons, mean over seeds.

Root cause of the former 0.5 assortativity error (see snap/eval/diagnose_swap.py, diagnose_hist2.py,
diagnose_bulk.py): not joint_degree, but the hub tail of degree_hist being thresholded away.

Usage (repo root): ``PYTHONPATH=src python snap/eval/diagnose_private.py [--seeds 3] [--eps 2 4 8 inf]``

Metrics: see METRICS. 'inf' is the non-private run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_aim.eval.evaluate import compare, structure_metrics
from graph_aim.inside.fit import fit
from graph_aim.outside.generate import generate
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
METRICS = ["assortativity_abs", "triangles_rel", "transitivity_abs", "clustering_abs",
           "degeneracy_rel", "degree_ks", "homophily_error", "attr_tvd_2way_mean"]
# (label, tau or None for "as fitted")
VARIANTS = [("as fitted", None)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps", nargs="+", default=["2", "4", "8", "inf"])
    args = ap.parse_args()
    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    rm = structure_metrics(real)
    cfg = json.loads((INSIDE / "config" / "graph_aim_v1.json").read_text())
    hom = cfg["homophily_attrs"]
    print(f"real assort={rm['assortativity']:.3f}")
    for e in args.eps:
        c = dict(cfg)
        if e == "inf":
            c["noise"] = False
        else:
            c.update(epsilon_edge=float(e), epsilon_attr=float(e))
        arts = [fit(real, c, seed=s) for s in range(args.seeds)]
        print(f"\n== eps={e}  " + "".join(f"{m[:9]:>11s}" for m in METRICS))
        for label, tau in VARIANTS:
            if e == "inf" and tau not in (None, 0.0):
                continue
            res = [compare(real, generate(a, seed=s), hom, seed=0, real_metrics=rm)
                   for s, a in enumerate(arts)]
            print(f"{label:16s}" + "".join(f"{np.mean([r[m] for r in res]):11.3f}" for m in METRICS), flush=True)


if __name__ == "__main__":
    main()

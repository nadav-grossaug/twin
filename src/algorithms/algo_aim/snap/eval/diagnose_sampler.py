"""Diagnose the graph-AIM sampler on the real ego-Facebook graph with a noise-free, uncapped artifact.

Usage (repo root): ``PYTHONPATH=src python snap/eval/diagnose_sampler.py [--seeds 3] [--ablate]``

Builds a non-private artifact with the degree cap switched off (``D`` above the maximum degree), runs
``generate`` and prints a metric table (mean and std over seeds). With ``--ablate`` it also removes
measurements from the artifact (joint_degree, clustering_by_degree, attr_edge_pair) to show which
component drives each error. Baseline rows (non-private BTER) are printed for reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph_aim.eval.baselines import BASELINES
from graph_aim.eval.evaluate import compare, structure_metrics
from graph_aim.inside.fit import fit
from graph_aim.outside.generate import generate
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent
INSIDE = ROOT.parent / "inside"  # real data and config live inside
METRICS = [
    "assortativity_abs",
    "triangles_rel",
    "degeneracy_rel",
    "degree_ks",
    "transitivity_abs",
    "clustering_abs",
    "attr_tvd_2way_mean",
    "homophily_error",
]
TARGETS = {
    "assortativity_abs": 0.15,
    "triangles_rel": 0.3,
    "degeneracy_rel": 0.25,
    "degree_ks": 0.05,
    "transitivity_abs": 0.08,
}


def build_artifact(real, cfg_path: Path, seed: int = 0, cap: int | None = None):
    cfg = json.loads(cfg_path.read_text())
    deg = np.bincount(real.edges.ravel(), minlength=real.n)
    cfg.update(noise=False, D=max(int(deg.max()) + 1, 5000) if cap is None else cap)
    return fit(real, cfg, seed=seed), cfg


def without(art, *kinds):
    from graph_aim.transfer.types import Artifact

    ms = {k: m for k, m in art.measurements.items() if m.kind not in kinds}
    return Artifact(art.schema, ms, art.ledger, art.provenance)


def run(label, real, rm, hom, make_syn, seeds, rows):
    res, secs = [], []
    for s in range(seeds):
        t0 = time.time()
        syn = make_syn(s)
        secs.append(time.time() - t0)
        res.append(compare(real, syn, hom, seed=0, real_metrics=rm))
    rows.append((label, {m: (np.mean([r[m] for r in res]), np.std([r[m] for r in res])) for m in METRICS},
                 float(np.mean(secs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--hist-uncapped", action="store_true",
                    help="round-2 style: degree_hist from the uncapped graph, other measurements capped at --cap")
    ap.add_argument("--cap", type=int, default=None, help="degree cap D (default: no-op cap)")
    ap.add_argument("--config", default=str(INSIDE / "config" / "graph_aim_v1.json"))
    args = ap.parse_args()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    rm = structure_metrics(real)
    art, cfg = build_artifact(real, Path(args.config), cap=args.cap)
    hom = cfg["homophily_attrs"]
    if args.hist_uncapped:
        full, _ = build_artifact(real, Path(args.config))
        art.measurements["degree_hist"] = full.measurements["degree_hist"]
    print(f"real: n={real.n} m={len(real.edges)} assort={rm['assortativity']:.3f} "
          f"trans={rm['transitivity']:.3f} tri={rm['triangles']:.0f} degeneracy={rm['degeneracy']}")

    rows: list = []
    run("graph_aim (full)", real, rm, hom, lambda s: generate(art, seed=s), args.seeds, rows)
    if args.ablate:
        run("full, rewiring off", real, rm, hom, lambda s: generate(art, seed=s, rewire_rounds=0),
            args.seeds, rows)
        variants = {
            "no joint_degree": without(art, "joint_degree"),
            "no clustering (phase 2 only)": without(art, "clustering_by_degree"),
            "no joint, no edge_pair": without(art, "joint_degree", "attr_edge_pair"),
            "no joint, no edge_pair, no clustering (Chung-Lu)": without(
                art, "joint_degree", "attr_edge_pair", "clustering_by_degree"
            ),
        }
        for name, a in variants.items():
            run(name, real, rm, hom, lambda s, a=a: generate(a, seed=s), args.seeds, rows)
    run("BTER baseline (non-private)", real, rm, hom, lambda s: BASELINES["bter"](real, s), args.seeds, rows) \
        if "bter" in BASELINES else None

    short = {m: m.replace("attr_tvd_2way_mean", "tvd2").replace("_abs", "").replace("_rel", "R") for m in METRICS}
    print("\n" + f"{'variant':52s}" + "".join(f"{short[m]:>15s}" for m in METRICS) + "   sec")
    print(f"{'target (<)':52s}" + "".join(f"{TARGETS.get(m, float('nan')):>15.3f}" for m in METRICS))
    for label, d, sec in rows:
        print(f"{label:52s}" + "".join(f"{d[m][0]:>9.3f}±{d[m][1]:.3f}" for m in METRICS) + f"  {sec:5.1f}")


if __name__ == "__main__":
    main()

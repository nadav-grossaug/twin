"""STAGE 3 (eval): real graph + synthetic graphs -> real-vs-fake discriminator. Needs both side by side.

Usage (repo root): ``PYTHONPATH=src python snap/eval/discriminate.py [--only ...] [--n 2000] [--models logreg gbm]``

Reads : snap/inside/data/processed/attributed_graph.npz           (real)
        snap/outside/synthetic/<variant>_seed<k>.npz              (generated outside from the artifacts)
Writes: snap/eval/results/discriminator.csv             one row per (method, epsilon, held-out fake seed, model)
        snap/eval/results/discriminator_summary.csv     mean/std over folds per (method, epsilon, model)
        snap/eval/results/discriminator_importance.csv  permutation importance per feature group (fold 0, strongest model)
        snap/eval/results/discriminator_by_size.csv     AUC by walk-size bucket (fold 0, strongest model)
        (plus json mirrors of the csv tables, written by ``realfake.report``)

A classifier two-sample test (``docs/realfake/PLAN.md``, ``docs/realfake/snap.md``): train a model
to tell random-walk subgraphs of the real graph (label 0) from walks of a synthetic graph (label 1);
AUC 0.5 means indistinguishable at this level, 1.0 trivially detectable. One "method" is a graph_aim
variant with its three generation seeds as fake sources, or one non-private baseline
(``graph_aim.eval.baselines``) with ``--seeds`` generated graphs. Per method one dataset is built
with a shared sampler/featurizer and scored over folds that rotate the held-out fake seed; real rows
are group-split by start node. Controls ride along in every row: the label-permutation AUC on the
same fold and the real-vs-real AUC (also emitted as its own method ``real_vs_real``), which must sit
near 0.5 for any other number to be trusted. Like ``score.py``, the baselines read the real graph:
they are reference points, not part of the inside/outside pipeline. ``--only`` takes variant names,
``baselines`` or a single baseline name.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from graph_aim.eval.baselines import BASELINES
from graph_aim.eval.discriminator import GraphFeaturizer, WalkSampler
from graph_aim.eval.evaluate import align_columns as align
from graph_aim.transfer.graph_io import load_attributed_graph
from realfake.controls import label_permutation, real_vs_real
from realfake.dataset import build_dataset
from realfake.importance import permutation_importance
from realfake.metrics import accuracy, auc, auc_by_bucket, bootstrap_auc_ci
from realfake.models import MODELS
from realfake.report import write_results
from realfake.split import group_split, split_with_held_out_sources

EVAL = Path(__file__).resolve().parent
SNAP = EVAL.parent
INSIDE = SNAP / "inside"  # the real data and the config live inside
RESULTS = EVAL / "results"
SYNTHETIC = SNAP / "outside" / "synthetic"

SIZE_EDGES = (10, 20, 40, 80, 120, 201)  # walk-size buckets for discriminator_by_size.csv


def selected(only: list[str] | None, token: str, group: str | None = None) -> bool:
    return only is None or token in only or (group is not None and group in only)


def run(
    real,
    entries: Iterable[tuple[str, float, dict]],
    homophily_attrs: list[str] | None,
    out: str | Path,
    n: int = 2000,
    models: list[str] | None = None,
    n_boot: int = 1000,
    sampler=None,
    featurizer=None,
    size_edges=SIZE_EDGES,
    log=print,
) -> None:
    """Score every method in ``entries`` (``(method, epsilon, {source_name: graph})``, the graphs
    already column-aligned to ``real``) and write the four result tables to ``out``.

    Per method: one dataset (``n`` real + ``n`` fake walk samples, shared sampler and featurizer,
    seed 0), one fold per fake source rotating the held-out source, every model fitted per fold.
    The real-vs-real control is computed once per model on a real-only dataset (``2 * n`` samples,
    seed 1) and repeated in every row; permutation importance and the AUC-by-size table use fold 0
    and the method's strongest model (highest mean AUC over folds)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = sampler if sampler is not None else WalkSampler()
    featurizer = featurizer if featurizer is not None else GraphFeaturizer(real.domains, homophily_attrs)
    names = list(models) if models else list(MODELS)
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; available: {list(MODELS)}")
    t0 = time.time()

    # real-vs-real control, once per model: one half of the real groups against the other (~0.5)
    ds_real = build_dataset(real, [], sampler, featurizer, n=2 * n, seed=1)
    rvr = {m: real_vs_real(ds_real, m, seed=0) for m in names}
    rows = [
        {"method": "real_vs_real", "epsilon": float("nan"), "seed": 0, "model": m, "auc": rvr[m], "real_vs_real_auc": rvr[m]}
        for m in names
    ]
    log("real_vs_real " + " ".join(f"{m}={rvr[m]:.3f}" for m in names) + f" ({time.time() - t0:.0f}s)", flush=True)

    imp_rows, size_rows = [], []
    for method, eps, fakes in entries:
        src_names = list(fakes)
        ds = build_dataset(real, [(nm, fakes[nm]) for nm in src_names], sampler, featurizer, n=n, seed=0)
        log(f"{method} eps={eps}: dataset built, {len(ds.y)} samples ({time.time() - t0:.0f}s)", flush=True)
        per_model: dict[str, list[float]] = {m: [] for m in names}
        fold0: dict[str, tuple] = {}  # model name -> (fitted model, test mask) of fold 0
        for s, held in enumerate(src_names):
            if len(src_names) > 1:  # train on the other fake seeds, test on the held-out one
                train, test = split_with_held_out_sources(ds, [held], test_frac=0.25, seed=s)
            else:  # a single fake source cannot be held out: plain stratified group split
                train, test = group_split(ds.groups, test_frac=0.25, seed=s, y=ds.y)
            for m in names:
                model = MODELS[m](ds.x[train], ds.y[train], seed=s)
                sc = model.scores(ds.x[test])
                a = auc(sc, ds.y[test])
                lo, hi = bootstrap_auc_ci(sc, ds.y[test], n_boot=n_boot, seed=s)
                rows.append(
                    {
                        "method": method,
                        "epsilon": eps,
                        "seed": s,
                        "model": m,
                        "auc": a,
                        "auc_lo": lo,
                        "auc_hi": hi,
                        "accuracy": accuracy(sc, ds.y[test]),
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "permuted_auc": label_permutation(ds.x[train], ds.y[train], ds.x[test], ds.y[test], m, seed=s),
                        "real_vs_real_auc": rvr[m],
                    }
                )
                per_model[m].append(a)
                if s == 0:
                    fold0[m] = (model, test)
            log(
                f"{method} eps={eps} fold={s} "
                + " ".join(f"{m}={per_model[m][-1]:.3f}" for m in names)
                + f" ({time.time() - t0:.0f}s)",
                flush=True,
            )

        # permutation importance and AUC by walk size: fold 0, strongest model of this method
        best = max(names, key=lambda m: float(np.mean(per_model[m])))
        model0, test0 = fold0[best]
        imp = permutation_importance(model0, ds.x[test0], ds.y[test0], ds.feature_names, ds.feature_groups, seed=0)
        imp_rows += [{"method": method, "epsilon": eps, "model": best, **r} for r in imp.to_dict("records")]
        sizes = np.array([ds.meta[i]["size"] for i in np.flatnonzero(test0)])
        by_size = auc_by_bucket(model0.scores(ds.x[test0]), ds.y[test0], sizes, np.asarray(size_edges, dtype=float))
        size_rows += [{"method": method, "epsilon": eps, "model": best, **r} for r in by_size[["lo", "hi", "n", "auc"]].to_dict("records")]

    write_results(rows, out, "discriminator", summary_keys=["method", "epsilon", "model"])
    write_results(imp_rows, out, "discriminator_importance")
    write_results(size_rows, out, "discriminator_by_size")
    log(f"wrote discriminator tables to {out} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(INSIDE / "config" / "graph_aim_v1.json"))
    ap.add_argument("--synthetic", default=str(SYNTHETIC))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the baselines (synthetic graphs: as generated)")
    ap.add_argument("--only", nargs="+", default=None, help="variants / baselines to score")
    ap.add_argument("--n", type=int, default=2000, help="samples per label per method")
    ap.add_argument("--models", nargs="+", default=None, help=f"models to fit (default: all of {list(MODELS)})")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the AUC CI")
    args = ap.parse_args()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    hom = json.loads(Path(args.config).read_text())["homophily_attrs"]
    syn_dir = Path(args.synthetic)
    index = json.loads((syn_dir / "index.json").read_text())

    def entries():
        by_variant: dict[str, list[dict]] = {}
        for e in index:
            by_variant.setdefault(e["variant"], []).append(e)
        for variant, es in by_variant.items():
            method = "graph_aim_nocap_nonprivate" if variant == "graph_aim_nocap_nonprivate" else "graph_aim"
            if not selected(args.only, variant, "graph_aim" if method == "graph_aim" else None):
                continue
            eps = math.inf if es[0]["epsilon"] is None else float(es[0]["epsilon"])
            fakes = {
                f"seed{e['seed']}": align(load_attributed_graph(syn_dir / f"{e['dir']}.npz"), real)
                for e in sorted(es, key=lambda e: int(e["seed"]))
            }
            yield method, eps, fakes
        for name, fn in BASELINES.items():
            if not selected(args.only, name, "baselines"):
                continue
            yield name, float("nan"), {f"seed{s}": fn(real, s) for s in range(args.seeds)}

    run(real, entries(), hom, args.out, n=args.n, models=args.models, n_boot=args.n_boot)


if __name__ == "__main__":
    main()

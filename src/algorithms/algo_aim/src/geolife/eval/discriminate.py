"""Real-vs-fake trace discriminator (eval zone): the testable core behind ``geolife/eval/discriminate.py``.

A classifier two-sample test (``docs/realfake/PLAN.md``, ``docs/realfake/geolife.md``): train a
model to tell short windows of the real Beijing traces (label 0) from windows of a synthetic
variant (label 1); AUC 0.5 means indistinguishable at this level, 1.0 trivially detectable. One
"method" is a synthetic variant; its generation seeds act as fake sources. With several sources the
folds rotate the held-out source (as ``snap/eval/discriminate.py``); with the single committed
variant on disk every fold falls back to a stratified group split (different split seed per fold),
so the summary still carries a split-stability std. Real and fake rows are group-split by USER, so
no user's windows appear on both sides of train/test. Controls ride along in every row: the
label-permutation AUC on the same fold and the real-vs-real AUC (also emitted as its own method
``real_vs_real``), which must sit near 0.5 for any other number to be trusted.

``road_cells`` is adapted from ``geolife/eval/match_real.py`` (``road_graph`` + ``seg_cells``) -
that file is a stage script off the import path, so the ~20 lines live here too.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geolife.eval.discriminator import (
    BEIJING_LONLAT,
    TraceFeaturizer,
    WindowSampler,
    load_real_arrays,
    load_synthetic_arrays,
)
from realfake.controls import label_permutation, real_vs_real
from realfake.dataset import build_dataset
from realfake.importance import permutation_importance
from realfake.metrics import accuracy, auc, auc_by_bucket, bootstrap_auc_ci
from realfake.models import MODELS
from realfake.report import write_results
from realfake.split import group_split, split_with_held_out_sources

SIZE_EDGES = (30, 60, 120, 240, 480, 601)  # window-size buckets for discriminator_by_size.csv


def road_cells(cache_dir, bbox=BEIJING_LONLAT, res: int = 9, every_m: float = 30.0) -> frozenset:
    """Res-``res`` H3 cells covering the public road graph, built the way the generator and
    ``match_real.py`` build it (subtype road, pedestrian classes excluded, largest component)."""
    import h3
    import networkx as nx

    from geolife.outside.map.provider import ROAD_EXCLUDE, fetch_roads
    from geolife.transfer.map.graph import build_graph

    seg, con = fetch_roads(bbox, cache_dir)
    if "subtype" in seg:
        seg = seg[seg["subtype"].fillna("road") == "road"]
    if "class" in seg:
        seg = seg[~seg["class"].isin(ROAD_EXCLUDE)]
    g = build_graph(seg, con)
    g = g.subgraph(max(nx.connected_components(g), key=len))
    cells: set = set()
    for _, _, d in g.edges(data=True):
        c = np.asarray(d["geom"].coords)
        n = max(2, int(d["length"] / every_m) + 1)
        f = np.linspace(0, 1, n)
        lon = np.interp(f, np.linspace(0, 1, len(c)), c[:, 0])
        lat = np.interp(f, np.linspace(0, 1, len(c)), c[:, 1])
        cells.update(h3.latlng_to_cell(a, b, res) for a, b in zip(lat.tolist(), lon.tolist()))
    return frozenset(cells)


def run(
    real_arrays: dict,
    entries,
    out,
    n: int = 4000,
    models: list[str] | None = None,
    n_boot: int = 1000,
    folds: int = 3,
    sampler=None,
    featurizer=None,
    size_edges=SIZE_EDGES,
    seed: int = 0,
    log=print,
) -> None:
    """Score every method in ``entries`` (``(method, epsilon, {source_name: arrays})``) and write
    the four result tables to ``out``.

    Per method: one dataset (``n`` real + ``n`` fake window samples, shared sampler and featurizer,
    ``seed``); with several fake sources one fold per source rotating the held-out source, with a
    single source ``folds`` stratified group splits (split seed = fold index). The real-vs-real
    control is computed once per model on a real-only dataset (``2 * n`` samples, ``seed + 1``) and
    repeated in every row; permutation importance and the AUC-by-window-size table use fold 0 and
    the method's strongest model (highest mean AUC over folds)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = sampler if sampler is not None else WindowSampler()
    featurizer = featurizer if featurizer is not None else TraceFeaturizer(include_map=False)
    names = list(models) if models else list(MODELS)
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; available: {list(MODELS)}")
    t0 = time.time()

    # real-vs-real control, once per model: one half of the real users against the other (~0.5)
    ds_real = build_dataset(real_arrays, [], sampler, featurizer, n=2 * n, seed=seed + 1)
    rvr = {m: real_vs_real(ds_real, m, seed=0) for m in names}
    rows = [
        {"method": "real_vs_real", "epsilon": float("nan"), "seed": 0, "model": m, "auc": rvr[m], "real_vs_real_auc": rvr[m]}
        for m in names
    ]
    log("real_vs_real " + " ".join(f"{m}={rvr[m]:.3f}" for m in names) + f" ({time.time() - t0:.0f}s)", flush=True)

    imp_rows, size_rows = [], []
    for method, eps, fakes in entries:
        src_names = list(fakes)
        ds = build_dataset(real_arrays, [(nm, fakes[nm]) for nm in src_names], sampler, featurizer, n=n, seed=seed)
        log(f"{method} eps={eps}: dataset built, {len(ds.y)} samples ({time.time() - t0:.0f}s)", flush=True)
        fold_specs = list(enumerate(src_names)) if len(src_names) > 1 else [(s, None) for s in range(folds)]
        per_model: dict[str, list[float]] = {m: [] for m in names}
        fold0: dict[str, tuple] = {}  # model name -> (fitted model, test mask) of fold 0
        for s, held in fold_specs:
            if held is not None:  # train on the other fake sources, test on the held-out one
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

        # permutation importance and AUC by window size: fold 0, strongest model of this method
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="geolife/inside/data/points.parquet")
    ap.add_argument("--synthetic", default="geolife/outside/synthetic", help="stage-2 output root (index.json)")
    ap.add_argument("--places", default="geolife/outside/cache/places_beijing_big.parquet")
    ap.add_argument("--cache-dir", default="geolife/outside/cache", help="Overture cache for the road-cell set")
    ap.add_argument("--out", default="geolife/eval/results/discriminator")
    ap.add_argument("--n", type=int, default=4000, help="samples per label per method")
    ap.add_argument("--models", nargs="+", default=None, help=f"models to fit (default: all of {list(MODELS)})")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the AUC CI")
    ap.add_argument("--folds", type=int, default=3, help="group-split folds when a variant has one fake source")
    ap.add_argument("--seed", type=int, default=0, help="dataset seed")
    ap.add_argument("--no-map", action="store_true", help="drop the map feature group (no places / road graph)")
    ap.add_argument("--max-users", type=int, default=None, help="truncate both sides (smoke runs)")
    ap.add_argument("--only", default=None, help="only variants starting with this prefix")
    a = ap.parse_args(argv)

    t0 = time.time()
    real = load_real_arrays(a.points, max_users=a.max_users)
    print(f"real: {len(real)} users ({time.time() - t0:.0f}s)", flush=True)

    syn_root = Path(a.synthetic)
    index = json.loads((syn_root / "index.json").read_text())
    by_variant: dict[str, list[dict]] = {}
    for e in index:
        if a.only and not str(e["variant"]).startswith(a.only):
            continue
        by_variant.setdefault(e["variant"], []).append(e)
    if not by_variant:
        print(f"no variants selected under {syn_root}")
        return 1

    def entries():
        for variant, es in by_variant.items():
            meta = json.loads((syn_root / es[0]["dir"] / "meta.json").read_text())
            eps = math.inf if meta.get("epsilon") is None else float(meta["epsilon"])
            fakes = {}
            for e in sorted(es, key=lambda e: int(e["seed"])):
                name = f"seed{e['seed']}"
                fakes[name] = load_synthetic_arrays(syn_root / e["dir"], max_users=a.max_users)
                print(f"{variant} {name}: {len(fakes[name])} users ({time.time() - t0:.0f}s)", flush=True)
            yield variant, eps, fakes

    if a.no_map:
        featurizer = TraceFeaturizer(include_map=False)
    else:
        places = pd.read_parquet(a.places, columns=["lat", "lon"]).to_numpy(float)
        rc = road_cells(a.cache_dir)
        print(f"map: {len(places)} places, {len(rc)} road cells ({time.time() - t0:.0f}s)", flush=True)
        featurizer = TraceFeaturizer(places_latlon=places, road_cells=rc, include_map=True)

    run(real, entries(), a.out, n=a.n, models=a.models, n_boot=a.n_boot, folds=a.folds,
        featurizer=featurizer, seed=a.seed)
    return 0

"""STAGE 3 (eval): real graph + synthetic graphs -> metrics. Needs both side by side, so it is neither inside nor outside.

Usage (repo root): ``PYTHONPATH=src python snap/eval/score.py [--seeds 3] [--only ...]``

Reads : snap/inside/data/processed/attributed_graph.npz           (real)
        snap/outside/synthetic/<variant>_seed<k>.npz              (generated outside from the artifacts)
Writes: snap/eval/results/{sweep,sweep_summary,baselines,baselines_summary}.csv, coupling_reference.csv, attacks.json
        (the sweep rows also carry ``cpl_*`` attribute-coupling columns, see ``graph_aim.eval.coupling``)

The non-private baselines (Erdos-Renyi, Chung-Lu, DC-SBM, BTER) are fitted to the real graph here, because they need
it; they are reference points, not part of the inside/outside pipeline. ``--only`` takes variant names, ``baselines`` or a
single baseline name.
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from graph_aim.eval.attacks import run_attacks
from graph_aim.eval.baselines import BASELINES
from graph_aim.eval.coupling import (
    coupling_metrics,
    gap_closed_table,
    options_from_policy,
    permuted_null,
)
from graph_aim.eval.coupling import flatten as flatten_coupling
from graph_aim.eval.evaluate import align_columns as align  # kept as score.align for the notebooks
from graph_aim.eval.evaluate import compare, degrees, structure_metrics
from graph_aim.eval.utility import LinkPredictionTask
from graph_aim.inside.measure import degree_bins
from graph_aim.transfer.graph_io import load_attributed_graph

EVAL = Path(__file__).resolve().parent
SNAP = EVAL.parent
INSIDE = SNAP / "inside"  # the real data and the config live inside
RESULTS = EVAL / "results"
SYNTHETIC = SNAP / "outside" / "synthetic"


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float, bool)):
            out[prefix + k] = float(v)
    return out


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std (over seeds) of every numeric column, one row per group."""
    num = [c for c in df.columns if c not in keys and c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, dropna=False, sort=False)
    agg = g[num].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", g.size())
    return agg.reset_index()



def evaluate_row(real, syn, hom, real_metrics, task) -> dict:
    row = flat(compare(real, syn, hom, seed=0, real_metrics=real_metrics))
    row.update({f"utility_{k}": v for k, v in task.evaluate(syn).items()})
    return row


def coupling_setup(real, policy_path: Path, exclude_hubs: int, seed: int = 0) -> tuple[dict, dict]:
    """Keyword arguments of ``coupling_metrics`` (missing labels and ordered flags from the policy, public degree bins)
    and the metrics of the permuted-attribute null, computed once."""
    missing, ordered = {}, {}
    if policy_path.exists():
        missing, ordered = options_from_policy(json.loads(policy_path.read_text()))
    dmax = int(degrees(real).max()) if real.n else 0
    kw = dict(
        degree_bin_edges=degree_bins(dmax, 6, dmax + 1), missing=missing, ordered=ordered, exclude_hubs=exclude_hubs
    )
    return kw, coupling_metrics(real, permuted_null(real, seed), **kw)


def coupling_row(real, syn, kw: dict, m_null: dict) -> dict:
    """``cpl_`` columns: per attribute and mean over attributes, plus gap closed against the permuted null."""
    m = coupling_metrics(real, syn, **kw)
    return flatten_coupling(m, gaps=gap_closed_table(m_null, m))


def selected(only: list[str] | None, token: str, group: str | None = None) -> bool:
    return only is None or token in only or (group is not None and group in only)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the baselines (synthetic graphs: as generated)")
    ap.add_argument("--config", default=str(INSIDE / "config" / "graph_aim_v1.json"))
    ap.add_argument("--policy", default=str(INSIDE / "attribute_policy.json"), help="optional missing_label / ordered keys")
    ap.add_argument("--exclude-hubs", type=int, default=10, help="hubs dropped for the cpl_*_nohub degree metrics")
    ap.add_argument("--synthetic", default=str(SYNTHETIC))
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--only", nargs="+", default=None, help="variants / baselines to score")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    hom = json.loads(Path(args.config).read_text())["homophily_attrs"]
    real_metrics = structure_metrics(real)
    task = LinkPredictionTask(real, seed=0, homophily_attrs=hom)
    print(f"real TRTR link-prediction AUC {task.auc_trtr:.4f} ({time.time() - t0:.0f}s)", flush=True)
    cpl_kw, cpl_null = coupling_setup(real, Path(args.policy), args.exclude_hubs)

    rows, attacks = [], {}
    syn_dir = Path(args.synthetic)
    for e in json.loads((syn_dir / "index.json").read_text()):
        method = "graph_aim_nocap_nonprivate" if e["variant"] == "graph_aim_nocap_nonprivate" else "graph_aim"
        if not selected(args.only, e["variant"], "graph_aim" if method == "graph_aim" else None):
            continue
        eps = math.inf if e["epsilon"] is None else float(e["epsilon"])
        syn = align(load_attributed_graph(syn_dir / f"{e['dir']}.npz"), real)
        row = {"method": method, "epsilon": eps, "seed": int(e["seed"])}
        row.update(evaluate_row(real, syn, hom, real_metrics, task))
        row.update(coupling_row(real, syn, cpl_kw, cpl_null))
        rows.append(row)
        if e["seed"] == 0:
            attacks[f"{method}_eps{eps}" if method == "graph_aim" else method] = run_attacks(real, syn, seed=0)
        print(f"scored {e['dir']} ({time.time() - t0:.0f}s)", flush=True)

    sweep = pd.DataFrame(rows)
    if len(sweep):
        sweep.to_csv(out / "sweep.csv", index=False)
        summarize(sweep, ["method", "epsilon"]).to_csv(out / "sweep_summary.csv", index=False)

    # floor / ceiling of the coupling metrics (cpl_ columns only): the real graph with permuted attributes (3 seeds
    # at most) and the real graph against itself. The structure baselines carry independent attributes, so they get no
    # cpl_ columns (n/a, not zero).
    if selected(args.only, "coupling_reference", "graph_aim"):
        ref = [
            {"method": "permuted_attributes", "seed": s, **coupling_row(real, permuted_null(real, s), cpl_kw, cpl_null)}
            for s in range(args.seeds)
        ] + [{"method": "real", "seed": 0, **coupling_row(real, real, cpl_kw, cpl_null)}]
        pd.DataFrame(ref).to_csv(out / "coupling_reference.csv", index=False)

    base_rows = []
    for name, fn in BASELINES.items():
        if not selected(args.only, name, "baselines"):
            continue
        for seed in range(args.seeds):
            syn = fn(real, seed)
            row = {"method": name, "seed": seed}
            row.update(evaluate_row(real, syn, hom, real_metrics, task))
            base_rows.append(row)
            if seed == 0:
                attacks[name] = run_attacks(real, syn, seed=0)
            print(f"baseline {name} seed={seed} done ({time.time() - t0:.0f}s)", flush=True)
    if base_rows:
        base = pd.DataFrame(base_rows)
        base.to_csv(out / "baselines.csv", index=False)
        summarize(base, ["method"]).to_csv(out / "baselines_summary.csv", index=False)

    (out / "attacks.json").write_text(json.dumps(attacks, indent=2, default=lambda o: None if o is None else str(o)))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

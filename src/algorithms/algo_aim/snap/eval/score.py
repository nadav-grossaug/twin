"""STAGE 3 (eval): real graph + synthetic graphs -> metrics. Needs both side by side, so it is neither inside nor outside.

Usage (repo root): ``PYTHONPATH=src python snap/eval/score.py [--seeds 3] [--only ...]``

Reads : snap/inside/data/processed/attributed_graph.npz           (real)
        snap/outside/synthetic/<variant>_seed<k>.npz              (generated outside from the artifacts)
Writes: snap/eval/results/{sweep,sweep_summary,baselines,baselines_summary}.csv, coupling_reference.csv, attacks.json
        (the sweep rows also carry ``cpl_*`` attribute-coupling columns, see ``graph_aim.eval.coupling``, and
        ``cmp_*`` neighbourhood-composition columns, see ``graph_aim.eval.composition``)

Every sweep and baseline row also carries ``com_*`` community columns (``graph_aim_walks.eval.metrics.
community_metrics``: Louvain / label-propagation modularity, parts, part sizes, conductance, walk-matrix spectral
gap): ``com_<metric>`` is the scored graph's value and ``com_<metric>_real`` the real graph's (the same in every
row, so a row reads on its own); and ``wlk_*`` walk-statistic gaps to the real graph (``walk_metrics``: return /
coverage error, attribute-lag TVD, share-variance ratio syn / real per window including the held-out windows 64 and
128, spectrum error; ``--walks 0`` skips them).

Method of a synthetic row: the index entry's ``method`` when it names a method (e.g. ``graph_aim_walks``), else
``graph_aim`` / ``graph_aim_nocap_nonprivate`` as before (``graph_aim.eval.evaluate.method_name``: the generator
flags ``v1`` / ``v2`` that v2 index entries carry do not rename rows).

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
from graph_aim.eval.composition import composition_metrics
from graph_aim.eval.composition import flatten as flatten_composition
from graph_aim.eval.composition import gap_closed_table as composition_gap_closed_table
from graph_aim.eval.composition import permuted_null_by_degree
from graph_aim.eval.coupling import (
    coupling_metrics,
    gap_closed_table,
    options_from_policy,
    permuted_null,
)
from graph_aim.eval.coupling import flatten as flatten_coupling
from graph_aim.eval.evaluate import align_columns as align  # kept as score.align for the notebooks
from graph_aim.eval.evaluate import compare, degrees, method_group, method_name, structure_metrics
from graph_aim.eval.utility import LinkPredictionTask
from graph_aim.inside.measure import degree_bins
from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim_walks.eval.metrics import community_metrics, default_spec, walk_metrics, walk_profile

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


def composition_setup(real, config: dict, seed: int = 0) -> tuple[dict, dict]:
    """Keyword arguments of ``composition_metrics`` and the metrics of the two attribute-permuted nulls, computed once.

    Groups: ``neigh_simpson_groups``, else ``homophily_groups``, else ``[homophily_attrs]``, else every attribute
    on its own. Bins: ``neigh_simpson_bin_edges``, else ``cluster_bin_edges``, else public log-spaced bins.
    ``min_degree``: ``neigh_simpson_min_degree``, else ``cluster_min_degree``, else 1. A config lacking the keys
    works (the v1 config, or an empty one). The nulls: ``coupling.permuted_null`` (``gap``) and
    ``permuted_null_by_degree`` on the ``attr_degree_bin_edges`` (fallback: the Simpson bins; ``gapdeg``), which
    keeps the degree x attribute coupling and is the floor for an i.i.d.-per-stratum label assignment.
    """
    groups = config.get("neigh_simpson_groups") or config.get("homophily_groups")
    if not groups:
        hom = config.get("homophily_attrs")
        groups = [list(hom)] if hom else [[c] for c in real.attrs.columns]
    bin_edges = config.get("neigh_simpson_bin_edges") or config.get("cluster_bin_edges")
    if not bin_edges:
        dmax = int(degrees(real).max()) if real.n else 0
        bin_edges = degree_bins(dmax, 6, dmax + 1)
    min_degree = config.get("neigh_simpson_min_degree") or config.get("cluster_min_degree") or 1
    kw = dict(groups=[list(g) for g in groups], bin_edges=list(bin_edges), min_degree=int(min_degree))
    strata = config.get("attr_degree_bin_edges") or bin_edges
    nulls = {
        "gap": composition_metrics(real, permuted_null(real, seed), **kw),
        "gapdeg": composition_metrics(real, permuted_null_by_degree(real, strata, seed), **kw),
    }
    return kw, nulls


def composition_row(real, syn, kw: dict, m_nulls: dict) -> dict:
    """``cmp_`` columns: per group and mean over groups, plus gap closed against each null in ``m_nulls``
    (``{gap token: composition_metrics of real vs null}``)."""
    m = composition_metrics(real, syn, **kw)
    row = flatten_composition(m)
    for token, m_null in m_nulls.items():
        row.update(flatten_composition({"groups": {}, "mean": {}}, gaps=composition_gap_closed_table(m_null, m), gap_name=token))
    return row


def community_row(syn, real_com: dict, seed: int = 0) -> dict:
    """``com_<metric>`` of ``syn`` and ``com_<metric>_real`` (``real_com`` = ``community_metrics(real)``)."""
    m = community_metrics(syn, seed)
    return {**{f"com_{k}": v for k, v in m.items()}, **{f"com_{k}_real": real_com[k] for k in m}}


def walk_setup(real, n_walks: int, seed: int = 0) -> dict | None:
    """The real graph's walk profile (computed once), or ``None`` when ``n_walks`` is 0."""
    return walk_profile(real, default_spec(real), n_walks, seed) if n_walks > 0 else None


def walk_row(real, syn, real_profile: dict | None, seed: int = 0) -> dict:
    """``wlk_*`` gaps of ``syn`` to the real graph (empty when walks are skipped)."""
    if real_profile is None:
        return {}
    return walk_metrics(real, syn, seed=seed, real_profile=real_profile)


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
    ap.add_argument("--walks", type=int, default=50000, help="walks per graph for the wlk_* columns (0 = skip)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    config = json.loads(Path(args.config).read_text())
    hom = config["homophily_attrs"]
    real_metrics = structure_metrics(real)
    task = LinkPredictionTask(real, seed=0, homophily_attrs=hom)
    print(f"real TRTR link-prediction AUC {task.auc_trtr:.4f} ({time.time() - t0:.0f}s)", flush=True)
    cpl_kw, cpl_null = coupling_setup(real, Path(args.policy), args.exclude_hubs)
    cmp_kw, cmp_null = composition_setup(real, config)
    com_real = community_metrics(real)
    wlk_real = walk_setup(real, args.walks)

    rows, attacks = [], {}
    syn_dir = Path(args.synthetic)
    for e in json.loads((syn_dir / "index.json").read_text()):
        method = method_name(e)
        if not selected(args.only, e["variant"], method_group(method)):
            continue
        eps = math.inf if e["epsilon"] is None else float(e["epsilon"])
        syn = align(load_attributed_graph(syn_dir / f"{e['dir']}.npz"), real)
        row = {"method": method, "epsilon": eps, "seed": int(e["seed"])}
        row.update(evaluate_row(real, syn, hom, real_metrics, task))
        row.update(coupling_row(real, syn, cpl_kw, cpl_null))
        row.update(composition_row(real, syn, cmp_kw, cmp_null))
        row.update(community_row(syn, com_real))
        row.update(walk_row(real, syn, wlk_real))
        rows.append(row)
        if e["seed"] == 0:
            attacks[f"{method}_eps{eps}" if method == "graph_aim" else method] = run_attacks(real, syn, seed=0)
        print(f"scored {e['dir']} ({time.time() - t0:.0f}s)", flush=True)

    sweep = pd.DataFrame(rows)
    if len(sweep):
        sweep.to_csv(out / "sweep.csv", index=False)
        summarize(sweep, ["method", "epsilon"]).to_csv(out / "sweep_summary.csv", index=False)

    # floor / ceiling of the coupling and composition metrics (cpl_ / cmp_ columns only): the real graph with permuted
    # attributes (3 seeds at most) and the real graph against itself. The structure baselines carry independent
    # attributes, so they get no cpl_ / cmp_ columns (n/a, not zero).
    if selected(args.only, "coupling_reference", "graph_aim"):
        ref = [
            {
                "method": "permuted_attributes",
                "seed": s,
                **coupling_row(real, permuted_null(real, s), cpl_kw, cpl_null),
                **composition_row(real, permuted_null(real, s), cmp_kw, cmp_null),
            }
            for s in range(args.seeds)
        ] + [
            {
                "method": "real",
                "seed": 0,
                **coupling_row(real, real, cpl_kw, cpl_null),
                **composition_row(real, real, cmp_kw, cmp_null),
            }
        ]
        pd.DataFrame(ref).to_csv(out / "coupling_reference.csv", index=False)

    base_rows = []
    for name, fn in BASELINES.items():
        if not selected(args.only, name, "baselines"):
            continue
        for seed in range(args.seeds):
            syn = fn(real, seed)
            row = {"method": name, "seed": seed}
            row.update(evaluate_row(real, syn, hom, real_metrics, task))
            row.update(community_row(syn, com_real))
            row.update(walk_row(real, syn, wlk_real))
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

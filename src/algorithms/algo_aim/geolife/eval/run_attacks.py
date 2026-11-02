"""EVAL (inside-only): empirical privacy attacks on the GeoLife artifacts and synthetic data.

    PYTHONPATH=src python geolife/eval/run_attacks.py \
        --points <points.parquet> --places <places_beijing_big.parquet> --data <geolife root> \
        [--artifacts geolife/transfer/artifacts] [--synthetic geolife/outside/synthetic/outings_v2_seed0] \
        [--out geolife/eval/results/attacks] [--n-worlds 400] [--no-sensors]

Reads the REAL traces (it is an eval script) and the committed artifacts; writes aggregate tables
only (JSON/CSV/markdown, no per-user records, no coordinates) to --out. The attacks and their
interpretation: src/geolife/eval/attacks.py; the write-up: geolife/eval/reports/privacy-attacks.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geolife.eval import attacks as A
from geolife.inside.learn_v2 import BEIJING_CORE, _clean, users_from_points
from geolife.inside.measure_v2 import VocabV2
from geolife.transfer.artifact import load_artifact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True)
    ap.add_argument("--places", required=True)
    ap.add_argument("--data", default=None, help="GeoLife root holding Data/<uid>/labels.txt")
    ap.add_argument("--artifacts", default="geolife/transfer/artifacts")
    ap.add_argument("--variant", default="outings_v2")
    ap.add_argument("--synthetic", default=None,
                    help="stage-2 output dir generated INTO THE SOURCE CITY (--city beijing) for linkage")
    ap.add_argument("--out", default="geolife/eval/results/attacks")
    ap.add_argument("--n-worlds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sensors", action="store_true", help="skip sensor tables in the MI vectors")
    a = ap.parse_args(argv)

    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in Path(a.artifacts).iterdir()
                  if p.is_dir() and p.name.startswith(a.variant) and (p / "manifest.json").is_file())
    arts = [load_artifact(d) for d in dirs]
    print(f"{len(arts)} artifacts: {[d.name for d in dirs]}", file=sys.stderr)

    cats = sorted(set(pd.read_parquet(a.places, columns=["category"]).category.astype(str)))
    vocab = VocabV2.from_bbox(BEIJING_CORE, cats)
    users = users_from_points(a.points, a.places, a.data, seed=0)
    print(f"{len(users)} real users with outings", file=sys.stderr)
    sensor_views = None
    if not a.no_sensors:
        from geolife.inside.measure_sensors import sensor_views_from_points
        sensor_views = sensor_views_from_points(a.points, a.data, seed=0, bbox=list(BEIJING_CORE))
        print(f"{len(sensor_views)} sensor views", file=sys.stderr)

    res: dict = {"artifacts": [d.name for d in dirs], "n_users": len(users)}
    uv = A.build_user_vectors(users, arts[0], vocab, sensor_views=sensor_views, seed=0)
    res["informed_mi"] = A.informed_mi(uv, rho_total=arts[0].ledger.rho_total)
    res["montecarlo_mi"] = A.montecarlo_mi(uv, n_worlds=a.n_worlds, seed=a.seed)
    res["seed_averaging"] = A.seed_averaging(arts, delta=arts[0].ledger.delta)

    if a.synthetic:
        from geolife.eval.compare import bundle_from_synthetic
        places = pd.read_parquet(a.places)
        syn = bundle_from_synthetic(a.synthetic, places=places).outings
        print(f"{len(syn)} synthetic users tokenized", file=sys.stderr)
        res["nearest_user"] = A.nearest_user_distance(users, syn, seed=a.seed)
        res["top_cell_linkage"] = A.top_cell_linkage(users, syn, seed=a.seed)
        res["signature_linkage"] = A.signature_linkage(users, syn, seed=a.seed)

    (out / "attacks.json").write_text(json.dumps(_clean(res), indent=1) + "\n", encoding="utf-8")
    pd.DataFrame(res["seed_averaging"]["tables"]).to_csv(out / "seed_averaging.csv", index=False)
    (out / "summary.md").write_text(_summary_md(res), encoding="utf-8")
    print(f"written to {out}", file=sys.stderr)
    return 0


def _summary_md(res: dict) -> str:
    L = [f"# Attack results ({A.LABEL})", ""]
    mi, mc = res["informed_mi"], res["montecarlo_mi"]
    L += ["## Membership inference (user level, against the artifact)",
          f"- informed (worst-case, exact): per-user AUC p50 {mi['auc']['p50']:.3f}, "
          f"p90 {mi['auc']['p90']:.3f}, max {mi['auc']['max']:.3f} "
          f"(zCDP bound for a cap-saturating user: {mi['auc']['bound']:.3f})",
          f"- Q (LR separation): p50 {mi['q']['p50']:.3f}, max {mi['q']['max']:.3f}, "
          f"bound 2*rho = {mi['q']['bound_2rho']:.3f}",
          f"- worst user TPR at FPR: " + ", ".join(f"{k}: {v:.3f}" for k, v in mi["tpr_at_fpr_worst_user"].items()),
          f"- Monte Carlo (uninformed matched filter, {mc['n_worlds']} worlds): per-user AUC "
          f"p50 {mc['auc']['p50']:.3f}, p90 {mc['auc']['p90']:.3f}, max {mc['auc']['max']:.3f}",
          f"- tables skipped in the vectors: {mi['tables_skipped'] or 'none'}", ""]
    sa = res["seed_averaging"]
    L += ["## Seed averaging",
          f"- {sa['n_seeds']} seeds; averaging divides the noise std by {sa['noise_reduction_factor']:.2f}",
          f"- honest accounting of releasing all seeds: rho {sa['rho_k_seeds']:.3f} -> "
          f"epsilon {sa['epsilon_k_seeds']:.2f} (single seed: {sa['epsilon_single']:.2f})",
          "- per-table cross-seed std vs ledger sigma and signal gain: `seed_averaging.csv`", ""]
    if "nearest_user" in res:
        nu, tc, sl = res["nearest_user"], res["top_cell_linkage"], res["signature_linkage"]
        L += ["## Linkage / re-identification (against the synthetic data, same frame)",
              f"- nearest-user profile distance: real->syn median {nu['synthetic']['median']:.3f} vs "
              f"real-holdout median {nu['holdout']['median']:.3f} (ratio {nu['ratio_median']:.2f}; "
              f"too_close={nu['too_close']})",
              f"- top-{tc['k']} cell Jaccard: best vs syn mean {tc['best_jaccard_syn']['mean']:.3f} "
              f"(max {tc['best_jaccard_syn']['max']:.3f}); chance (other real users) mean "
              f"{tc['best_jaccard_other_real']['mean']:.3f}; copy reference (own split half) mean "
              f"{tc['self_split_half']['mean']:.3f}",
              f"- re-identification rate (best syn match >= own split half): {tc['reid_rate_syn']:.3f} "
              f"vs chance {tc['reid_rate_chance']:.3f}",
              f"- Hungarian matching mean Jaccard {tc['hungarian_mean_jaccard']:.3f} vs random "
              f"{tc['random_matching_mean_jaccard']:.3f}",
              f"- coarse signature linkage: rate vs syn {sl['rate_syn']:.3f}, holdout "
              f"{sl['rate_holdout']:.3f}, excess {sl['excess']:.3f}", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

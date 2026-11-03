"""Package E gate (plan section 7.5 / section 10 criterion 3): E's sweep vs the v1.1 reference, per metric and epsilon.

Run from the repo root after score.py wrote tools/v2_runs/e/results/sweep.csv:
  /opt/venv/bin/python3 tools/v2_runs/e/gate.py
Writes tools/v2_runs/e/gate.csv: one row per (epsilon, metric) with the v1.1 per-seed min / max / mean, E's per-seed
values and mean, and the verdict: ``within`` (E's mean inside the v1.1 seed spread), ``better`` (outside it on the
good side: lower), ``worse`` (outside on the bad side). Gate metrics must not be ``worse``; path length, degeneracy
and the composition columns are reported, not gated. The ``cmp_*`` columns do not exist in the v1.1 csv (package A
came later): they are listed with E's values only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REF = ROOT / "snap" / "eval" / "results_v1_1" / "sweep.csv"
NEW = HERE / "results" / "sweep.csv"

GATE = ["degree_ks", "triangles_rel", "clustering_abs", "transitivity_abs", "assortativity_abs"]
REPORT = ["path_length_abs", "degeneracy_abs", "homophily_error", "attr_tvd_2way_mean", "eigenvalue_rel",
          "largest_component_abs"]


def main() -> None:
    ref = pd.read_csv(REF)
    new = pd.read_csv(NEW)
    ref = ref[ref.method == "graph_aim"]
    new = new[new.method == "graph_aim"]
    cmp_cols = sorted(c for c in new.columns if c.startswith("cmp_") and c.endswith("gapdeg_simpson_mean"))
    cmp_cols += sorted(c for c in new.columns if c.startswith("cmp_") and c.endswith("_simpson_mean_abs"))
    rows = []
    for eps in sorted(new.epsilon.unique()):
        r = ref[ref.epsilon == eps].sort_values("seed")
        e = new[new.epsilon == eps].sort_values("seed")
        for metric, kind in [(m, "gate") for m in GATE] + [(m, "report") for m in REPORT] + [(m, "composition") for m in cmp_cols]:
            if metric not in e.columns:
                continue
            ev = e[metric].to_numpy()
            row = {"epsilon": eps, "metric": metric, "kind": kind,
                   "E_seed0": ev[0] if len(ev) > 0 else None, "E_seed1": ev[1] if len(ev) > 1 else None,
                   "E_seed2": ev[2] if len(ev) > 2 else None, "E_mean": ev.mean()}
            if metric in r.columns:
                rv = r[metric].to_numpy()
                lo, hi = rv.min(), rv.max()
                row.update(v11_min=lo, v11_max=hi, v11_mean=rv.mean())
                if lo <= ev.mean() <= hi:
                    verdict = "within"
                elif ev.mean() < lo:
                    verdict = "better" if not metric.endswith("gapdeg_simpson_mean") else "worse"
                else:
                    verdict = "worse" if not metric.endswith("gapdeg_simpson_mean") else "better"
                row["verdict"] = verdict
            else:
                row.update(v11_min=None, v11_max=None, v11_mean=None, verdict="n/a")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "gate.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    bad = df[(df.kind == "gate") & (df.verdict == "worse")]
    print("\nGATE:", "PASS" if bad.empty else f"FAIL on {bad[['epsilon', 'metric']].to_dict('records')}")


if __name__ == "__main__":
    main()

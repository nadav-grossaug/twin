"""Throwaway: digest the package-D sweep (X single / Y three / Z two homophily groups).

1. Homophily metrics per attribute and variant (lift, assort, gaps, mix_tvd).
2. B/C degree-coupling gaps vs X.
3. Structure guard vs X and the committed HEAD baseline (pass baseline CSV as argv[1]).
4. Stage-2 per-graph runtimes from the sweep log (argv[2], default /tmp/sweep_d.log).

Run (repo root): python3 tools/sweep_d/digest.py /tmp/baseline_sweep_summary.csv /tmp/sweep_d.log
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "tools" / "sweep_d"
VARIANTS = ["X_single", "Y_three", "Z_two"]
HOM_ATTRS = ["school", "hometown", "location", "birthday", "locale"]
BC_ATTRS = ["school", "birthday", "hometown", "location", "locale", "employer"]


def load(name):
    df = pd.read_csv(D / name / "coupling.csv")
    df["eps"] = df["graph"].str.extract(r"^graph_aim_(eps(?:4|inf))_seed\d$")
    return df


def piv(df, eps):
    sub = df[df["eps"] == eps]
    return sub.pivot_table(index="attribute", columns="metric", values="value", aggfunc="mean")


def main() -> None:
    data = {v: load(v) for v in VARIANTS}

    print("## 1. Homophily (edge x attribute), 3-seed means")
    for eps in ["epsinf", "eps4"]:
        print(f"\n### {eps}\n")
        print("| attribute | variant | lift real | lift syn | assort real | assort syn | gap_lift | gap_assort | mix_tvd |")
        print("|---|---|---|---|---|---|---|---|---|")
        for a in HOM_ATTRS:
            for v in VARIANTS:
                p = piv(data[v], eps)
                r = p.loc[a]
                print(f"| {a} | {v} | {r['lift_real']:.2f} | {r['lift_syn']:.2f} | {r['assort_real']:.3f} "
                      f"| {r['assort_syn']:.3f} | {r.get('gap_lift', float('nan')):.2f} "
                      f"| {r.get('gap_assort', float('nan')):.2f} | {r['mix_tvd']:.3f} |")

    print("\n## 2. B/C gaps (gap_eta2 / gap_deg_by_value_tvd), 3-seed means")
    for eps in ["epsinf", "eps4"]:
        print(f"\n### {eps}\n")
        print("| variant | " + " | ".join(BC_ATTRS) + " |")
        print("|" + "---|" * (len(BC_ATTRS) + 1))
        for v in VARIANTS:
            p = piv(data[v], eps)
            cells = [f"{p.loc[a, 'gap_eta2']:.2f}/{p.loc[a, 'gap_deg_by_value_tvd']:.2f}" for a in BC_ATTRS]
            print(f"| {v} | " + " | ".join(cells) + " |")

    print("\n## 3. Structure guard (mean±std; MOVED-vs-X = beyond max std vs X; base = committed HEAD)")
    metrics = ["degree_ks", "triangles_rel", "clustering_abs", "transitivity_abs",
               "assortativity_abs", "attr_tvd_2way_mean", "homophily_error"]
    base = pd.read_csv(sys.argv[1]).set_index(["method", "epsilon"]) if len(sys.argv) > 1 else None
    summ = {v: pd.read_csv(D / v / "results" / "sweep_summary.csv").set_index(["method", "epsilon"]) for v in VARIANTS}
    for eps in [4.0, float("inf")]:
        print(f"\n### eps={eps}\n")
        print("| metric | HEAD baseline | X | Y | Z | flags |")
        print("|---|---|---|---|---|---|")
        x = summ["X_single"].loc[("graph_aim", eps)]
        b = base.loc[("graph_aim", eps)] if base is not None else None
        for m in metrics:
            cells, flags = [], []
            bs = f"{b[m + '_mean']:.4f}±{b[m + '_std']:.4f}" if b is not None else "n/a"
            for v in VARIANTS:
                r = summ[v].loc[("graph_aim", eps)]
                cells.append(f"{r[m + '_mean']:.4f}±{r[m + '_std']:.4f}")
                if v != "X_single":
                    dx = abs(r[m + "_mean"] - x[m + "_mean"])
                    if dx > max(r[m + "_std"], x[m + "_std"]):
                        flags.append(f"{v[0]} vs X")
                    if b is not None and abs(r[m + "_mean"] - b[m + "_mean"]) > max(r[m + "_std"], b[m + "_std"]):
                        flags.append(f"{v[0]} vs HEAD")
            print(f"| {m} | {bs} | " + " | ".join(cells) + " | " + ("; ".join(flags) or "-") + " |")

    log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/sweep_d.log").read_text()
    print("\n## 4. Stage-2 per-graph runtime (s)")
    for block, v in zip(re.split(r"== \w+: stage 2 ==", log)[1:], VARIANTS):
        ts = [int(x) for x in re.findall(r"generated \S+: \d+ nodes, \d+ edges \((\d+)s\)", block.split("== ")[0])]
        deltas = [b - a for a, b in zip([0] + ts[:-1], ts)]
        print(f"{v}: per-graph {deltas} (total {ts[-1] if ts else '?'}s)")


if __name__ == "__main__":
    main()

"""Throwaway: digest tools/coupling_before.csv + tools/coupling_after.csv into before/after markdown tables,
plus the plan-section-1 headline mean-degree table (birthday / school values), real vs before vs after.

Run (repo root): PYTHONPATH=src:snap/eval python3 tools/coupling_compare.py
"""

import numpy as np
import pandas as pd

from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim.eval.evaluate import align_columns, degrees

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

METRICS = [
    "lift_real", "lift_syn", "eta2_real", "eta2_syn", "deg_by_value_tvd", "mix_tvd",
    "gap_lift", "gap_eta2", "gap_deg_by_value_tvd",
]


def pivot(path):
    df = pd.read_csv(path)
    df = df[df["graph"] != "permuted_null"]
    df["variant"] = df["graph"].str.extract(r"^(graph_aim_eps(?:4|inf))_seed\d$")
    df["seed"] = df["graph"].str.extract(r"seed(\d)$").astype(int)
    out = df.pivot_table(index=["variant", "attribute"], columns="metric", values="value", aggfunc="mean")
    sd = df.pivot_table(index=["variant", "attribute"], columns="metric", values="value", aggfunc="std")
    return out, sd


def null_eta2(path):
    df = pd.read_csv(path)
    df = df[(df["graph"] == "permuted_null") & (df["metric"] == "eta2_syn")]
    return df.set_index("attribute")["value"].to_dict()


def main() -> None:
    before, _ = pivot(TOOLS / "coupling_before.csv")
    after, after_sd = pivot(TOOLS / "coupling_after.csv")
    nulls = null_eta2(TOOLS / "coupling_after.csv")

    for variant in ["graph_aim_epsinf", "graph_aim_eps4"]:
        b, a = before.loc[variant], after.loc[variant]
        print(f"\n### {variant} (3-seed means)\n")
        hdr = ("| attribute | lift real | lift before | lift after | eta2 real | eta2 null | eta2 before | eta2 after "
               "| deg_tvd before | deg_tvd after | mix_tvd before | mix_tvd after "
               "| gap_eta2 after | gap_deg_tvd after | gap_lift after |")
        print(hdr)
        print("|" + "---|" * (hdr.count("|") - 1))
        order = a["eta2_real"].sort_values(ascending=False).index
        for attr in order:
            rb, ra = b.loc[attr], a.loc[attr]
            print(
                f"| {attr} | {ra['lift_real']:.2f} | {rb['lift_syn']:.2f} | {ra['lift_syn']:.2f} "
                f"| {ra['eta2_real']:.3f} | {nulls.get(attr, float('nan')):.4f} | {rb['eta2_syn']:.3f} | {ra['eta2_syn']:.3f} "
                f"| {rb['deg_by_value_tvd']:.3f} | {ra['deg_by_value_tvd']:.3f} "
                f"| {rb['mix_tvd']:.3f} | {ra['mix_tvd']:.3f} "
                f"| {ra['gap_eta2']:.2f} | {ra['gap_deg_by_value_tvd']:.2f} | {ra['gap_lift']:.2f} |"
            )
        print("\nbefore gaps (same attributes): gap_eta2 / gap_deg_tvd / gap_lift")
        for attr in order:
            rb = b.loc[attr]
            print(f"  {attr}: {rb['gap_eta2']:.2f} / {rb['gap_deg_by_value_tvd']:.2f} / {rb['gap_lift']:.2f}")

    # --------------------------------------------------------------- headline mean degrees
    real = load_attributed_graph(ROOT / "snap" / "inside" / "data" / "processed" / "attributed_graph.npz")
    targets = [("birthday", "birthday_5"), ("birthday", "none"),
               ("school", "school_538"), ("school", "school_50"), ("school", "none"), ("school", "other")]

    def mean_deg(g, col, label):
        if label not in g.domains[col]:
            return float("nan")
        code = g.domains[col].index(label)
        d = degrees(g)
        mask = g.attrs[col].to_numpy() == code
        return float(d[mask].mean()) if mask.any() else float("nan")

    rows = []
    for col, label in targets:
        row = {"value": f"{col} `{label}`", "real": mean_deg(real, col, label)}
        for variant in ["graph_aim_epsinf", "graph_aim_eps4"]:
            for kind, folder in [("before", TOOLS / "before_graphs"), ("after", ROOT / "snap" / "outside" / "synthetic")]:
                vals = []
                for s in range(3):
                    g = align_columns(load_attributed_graph(folder / f"{variant}_seed{s}.npz"), real)
                    vals.append(mean_deg(g, col, label))
                row[f"{variant}_{kind}"] = float(np.nanmean(vals))
        rows.append(row)
    t = pd.DataFrame(rows)
    print("\n### Headline mean degree by value (3-seed means for synthetic)\n")
    print(t.to_markdown(index=False, floatfmt=".1f"))


if __name__ == "__main__":
    main()

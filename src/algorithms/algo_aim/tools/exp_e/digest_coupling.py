"""Throwaway (package E): 3-seed mean/std of homophily metrics for the three groups, base vs E."""

import pandas as pd

ATTRS = ["school", "hometown", "location"]
METRICS = ["lift_real", "lift_syn", "gap_lift", "assort_real", "assort_syn", "gap_assort"]


def load(path: str, side: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["eps"] = df["graph"].str.extract(r"^graph_aim_(eps(?:4|inf))_seed\d$")
    df["side"] = side
    return df


df = pd.concat([load("tools/exp_e/coupling_base.csv", "base"), load("tools/exp_e/coupling_e.csv", "E")])
df = df[df["attribute"].isin(ATTRS) & df["metric"].isin(METRICS)]
pd.set_option("display.width", 300)
for eps in ["epsinf", "eps4"]:
    sub = df[df["eps"] == eps]
    p = sub.pivot_table(index=["attribute", "side"], columns="metric", values="value", aggfunc=["mean", "std"])
    p = p.reorder_levels([1, 0], axis=1)[[(m, a) for m in METRICS for a in ["mean", "std"]]]
    print(f"\n=== {eps} (3-seed mean/std) ===")
    print(p.round(3).to_string())

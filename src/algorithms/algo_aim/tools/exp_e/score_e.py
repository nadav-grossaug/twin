"""Throwaway (package E): structure metrics and within-group triangle shares, base vs E sides.

Run (repo root): PYTHONPATH=src:snap/inside python3 tools/exp_e/score_e.py
Writes tools/exp_e/structure_tri.csv and prints 3-seed mean/std per (side, eps).
"""

import json

import pandas as pd

from graph_aim.eval.coupling import options_from_policy, permuted_null, within_group_triangle_share
from graph_aim.eval.evaluate import align_columns, compare, structure_metrics
from graph_aim.transfer.graph_io import load_attributed_graph

REAL = "snap/inside/data/processed/attributed_graph.npz"
GROUPS = ["school", "hometown", "location"]
KEEP = ["degree_ks", "triangles_rel", "clustering_abs", "transitivity_abs", "assortativity_abs"]

policy = json.load(open("snap/inside/attribute_policy.json"))
missing, _ = options_from_policy(policy)

real = load_attributed_graph(REAL)
rm = structure_metrics(real, seed=0)
null = permuted_null(real, seed=0)
print(f"real: n={real.n} m={len(real.edges)} triangles={rm['triangles']:.0f} "
      f"clustering={rm['avg_clustering']:.4f} transitivity={rm['transitivity']:.4f}")
for c in GROUPS:
    print(f"  within-group triangle share {c}: real={within_group_triangle_share(real, c, missing=missing):.4f} "
          f"null={within_group_triangle_share(null, c, missing=missing):.4f}")

rows = []
for side, d in [("base", "tools/exp_e/synthetic_base"), ("E", "tools/exp_e/synthetic_e")]:
    for eps in ["eps4", "epsinf"]:
        for seed in range(3):
            g = align_columns(load_attributed_graph(f"{d}/graph_aim_{eps}_seed{seed}.npz"), real)
            c = compare(real, g, homophily_attrs=["school"], real_metrics=rm)
            row = {"side": side, "eps": eps, "seed": seed, "m": len(g.edges)}
            row.update({k: c[k] for k in KEEP})
            for col in GROUPS:
                row[f"tri_{col}"] = within_group_triangle_share(g, col, missing=missing)
            rows.append(row)
            print(row, flush=True)

df = pd.DataFrame(rows)
df.to_csv("tools/exp_e/structure_tri.csv", index=False)
pd.set_option("display.width", 300)
agg = df.drop(columns=["seed"]).groupby(["side", "eps"]).agg(["mean", "std"])
print(agg.round(4).to_string())

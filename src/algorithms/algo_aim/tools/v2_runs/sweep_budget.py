"""Budget sweep for the attr_neigh_simpson kind (stage 1 only, eval zone: reads the real graph).

For every config variant: fit at eps = 4 (3 noise seeds) and at eps = inf, and compare the released per-bin
Simpson means with the exact ones (RMSE over the informative bins, per instance), plus the sigma of every attribute-
side measurement so the cost to the other kinds is visible. No generation: the question here is how noisy the
targets are, which does not depend on the generator.

Run (repo root): PYTHONPATH=src:snap/inside:snap/eval python3 tools/v2_runs/sweep_budget.py
Writes tools/v2_runs/sweep_budget.csv
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph_aim.inside.fit import fit
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parents[2]
real = load_attributed_graph(ROOT / "snap/inside/data/processed/attributed_graph.npz")
base = json.loads((ROOT / "snap/inside/config/graph_aim_v2.json").read_text())

THREE = [["school"], ["hometown"], ["location"]]
FOUR = base["neigh_simpson_groups"]
SH = [["school"], ["hometown"], ["location"], ["school", "hometown"]]


def variant(name, **kw):
    c = copy.deepcopy(base)
    for k, v in kw.items():
        if k == "cap":
            c["caps"]["attr_neigh_simpson"] = v
        elif k == "share":
            c["shares"]["attr_neigh_simpson"] = v
        elif k == "edge_pair_share":
            c["shares"]["attr_edge_pair"] = v
        else:
            c[k] = v
    return name, c


VARIANTS = [
    variant("base_cap150_s40_g4"),
    variant("cap100", cap=100),
    variant("cap60", cap=60),
    variant("cap40", cap=40),
    variant("share55", share=0.55),
    variant("share55_ep30", share=0.55, edge_pair_share=0.30),
    variant("groups3", neigh_simpson_groups=THREE),
    variant("groups_sh", neigh_simpson_groups=SH),
    variant("bins3", neigh_simpson_bin_edges=[0, 16, 60, 1201]),
    variant("cap60_groups3", cap=60, neigh_simpson_groups=THREE),
    variant("cap60_bins3", cap=60, neigh_simpson_bin_edges=[0, 16, 60, 1201]),
    variant("cap60_share55_ep30", cap=60, share=0.55, edge_pair_share=0.30),
    variant("cap60_bins3_share55_ep30", cap=60, share=0.55, edge_pair_share=0.30, neigh_simpson_bin_edges=[0, 16, 60, 1201]),
]


def means(art):
    out = {}
    for m in art.measurements.values():
        if m.kind != "attr_neigh_simpson":
            continue
        v = np.asarray(m.values, dtype=float)
        lo = np.asarray(m.params["bin_edges"][:-1])
        keep = lo >= int(m.params["min_degree"])
        out[m.name] = v[keep, 0] / np.maximum(v[keep, 1], 1.0)
    return out


rows = []
for name, cfg in VARIANTS:
    exact = fit(real, dict(cfg, noise=False), seed=0)
    ex = means(exact)
    for seed in range(3):
        art = fit(real, dict(cfg, noise=True, epsilon_edge=4.0, epsilon_attr=4.0), seed=seed)
        rel = means(art)
        row = {"variant": name, "seed": seed}
        errs = []
        for k, e in ex.items():
            r = np.clip(rel[k], 0, 1)
            rmse = float(np.sqrt(np.mean((r - e) ** 2)))
            errs.append(rmse)
            row[f"rmse:{k.split(':')[1]}"] = rmse
        row["rmse_mean"] = float(np.mean(errs))
        row["rmse_max"] = float(np.max(errs))
        for m in art.measurements.values():
            if m.kind in ("attr_neigh_simpson", "attr_edge_pair", "attr_2way", "attr_degree"):
                row.setdefault(f"sigma:{m.kind}", m.sigma)
        rows.append(row)
        print(name, seed, f"rmse mean {row['rmse_mean']:.3f} max {row['rmse_max']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "tools/v2_runs/sweep_budget.csv", index=False)
summary = df.groupby("variant", sort=False).mean(numeric_only=True).drop(columns="seed")
pd.set_option("display.width", 250)
print(summary.round(3).to_string())

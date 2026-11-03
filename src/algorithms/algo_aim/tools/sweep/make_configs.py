"""Throwaway: write the lever-sweep config copies (share / bins levers for attr_degree)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
base = json.loads((ROOT / "snap" / "inside" / "config" / "graph_aim_v1.json").read_text())

# 4-bin grid = degree_bins(1200, 4, 1201): the fit's own default convention (log grid, starts at 0,
# last bin open / clamped at Dmax = 1200), just with n_bins=4.
EDGES4 = [0, 1, 11, 113, 1201]

variants = {
    "share35": {"shares": dict(base["shares"], attr_degree=0.35)},
    "bins4": {"attr_degree_bin_edges": EDGES4},
    "share35_bins4": {"shares": dict(base["shares"], attr_degree=0.35), "attr_degree_bin_edges": EDGES4},
}
for name, over in variants.items():
    cfg = dict(base, **over)
    p = ROOT / "tools" / "sweep" / f"{name}.json"
    p.write_text(json.dumps(cfg, indent=2))
    print("wrote", p)

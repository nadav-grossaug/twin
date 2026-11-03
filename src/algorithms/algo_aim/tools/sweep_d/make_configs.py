"""Throwaway: write the package-D decision-point configs (X control, Y three groups, Z two groups).

All start from graph_aim_v1.json with the chosen B/C levers applied:
shares.attr_degree = 0.35, attr_degree_bin_edges = [0, 1, 11, 113, 1201].
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
base = json.loads((ROOT / "snap" / "inside" / "config" / "graph_aim_v1.json").read_text())
base = dict(base, shares=dict(base["shares"], attr_degree=0.35), attr_degree_bin_edges=[0, 1, 11, 113, 1201])

variants = {
    "X_single": {"homophily_groups": None},
    "Y_three": {"homophily_groups": [["school"], ["hometown"], ["location"]]},
    "Z_two": {"homophily_groups": [["school"], ["hometown"]]},
}
for name, over in variants.items():
    p = ROOT / "tools" / "sweep_d" / f"{name}.json"
    p.write_text(json.dumps(dict(base, **over), indent=2))
    print("wrote", p)

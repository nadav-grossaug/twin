"""Throwaway ablation: regenerate synthetic graphs WITHOUT the attr_degree measurements (v1 degree shuffle).

For graph_aim_eps4 and graph_aim_epsinf, seeds 0-2: load each artifact, drop its attr_degree
measurements in memory, generate with the same seed as the pipeline (the artifact's seed), and save
under tools/before_graphs/. attr_edge_pair single-channel steering stays, matching v1.

Run (repo root): PYTHONPATH=src python3 tools/coupling_before.py
"""

import dataclasses
import json
from pathlib import Path

from graph_aim.outside.generate import generate
from graph_aim.transfer.artifact import load
from graph_aim.transfer.graph_io import save_attributed_graph

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "snap" / "transfer" / "artifacts"
OUT = ROOT / "tools" / "before_graphs"

VARIANTS = {"graph_aim_eps4", "graph_aim_epsinf"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((ARTIFACTS / "index.json").read_text()):
        if entry["variant"] not in VARIANTS:
            continue
        art = load(ARTIFACTS / entry["dir"])
        n_ad = sum(1 for m in art.measurements.values() if m.kind == "attr_degree")
        art = dataclasses.replace(
            art,
            measurements={k: m for k, m in art.measurements.items() if m.kind != "attr_degree"},
        )
        g = generate(art, n=None, seed=int(entry["seed"]))
        save_attributed_graph(g, OUT / f"{entry['dir']}.npz")
        print(f"{entry['dir']}: dropped {n_ad} attr_degree tables -> {g.n} nodes, {len(g.edges)} edges", flush=True)


if __name__ == "__main__":
    main()

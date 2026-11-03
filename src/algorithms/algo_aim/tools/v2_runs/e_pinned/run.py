"""Pinned-regions comparison: generate_v2 with regions_R=8, within_region_frac=0.9 (no spectrum calibration)."""
import json, sys, time
from pathlib import Path
from graph_aim.outside.generate_v2 import generate_v2
from graph_aim.transfer.artifact import load
from graph_aim.transfer.graph_io import save_attributed_graph
root = Path(__file__).resolve().parents[3]
src, out = root / "snap/transfer/artifacts_v2", Path(__file__).resolve().parent / "synthetic"
done = []
for e in json.loads((src / "index.json").read_text()):
    if e["variant"] not in ("graph_aim_eps4", "graph_aim_epsinf"):
        continue
    t = time.time()
    g = generate_v2(load(src / e["dir"]), seed=int(e["seed"]), regions_R=8, within_region_frac=0.9)
    save_attributed_graph(g, out / f"{e['dir']}.npz")
    done.append({**e, "edges": int(len(g.edges)), "method": "v2_pinned_R8_q0.9"})
    print(e["dir"], len(g.edges), f"{time.time() - t:.0f}s", flush=True)
(out / "index.json").write_text(json.dumps(done, indent=1))

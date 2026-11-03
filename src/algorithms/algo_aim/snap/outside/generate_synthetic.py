"""STAGE 2 (outside): artifacts -> synthetic graphs. Never sees the real graph.

Usage (repo root): ``PYTHONPATH=src python snap/outside/generate_synthetic.py [--only ...]``
Note the path: only ``src`` is needed; nothing from ``snap/inside`` is importable here.

Reads : snap/transfer/artifacts/<variant>_seed<k>/  (hash-verified on load)
Writes: snap/outside/synthetic/<variant>_seed<k>.npz  (synthetic attributed graphs)

Allowed imports: standard library, numpy/pandas/scipy, ``graph_aim.transfer`` and ``graph_aim.outside``
(enforced by tests/graph_aim/test_boundary.py).
"""

import argparse
import json
import time
from pathlib import Path

from graph_aim.outside.generate import generate
from graph_aim.transfer.artifact import load
from graph_aim.transfer.graph_io import save_attributed_graph

OUTSIDE = Path(__file__).resolve().parent
ARTIFACTS = OUTSIDE.parent / "transfer" / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--out", default=str(OUTSIDE / "synthetic"))
    ap.add_argument("--only", nargs="+", default=None, help="variant names to generate (default: all)")
    args = ap.parse_args()
    src, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    done = []
    for entry in json.loads((src / "index.json").read_text()):
        if args.only and entry["variant"] not in args.only:
            continue
        if not (src / entry["dir"]).is_dir():  # the index lists every variant; copy only the artifacts you need
            print(f"skipped {entry['dir']}: artifact folder not present", flush=True)
            continue
        art = load(src / entry["dir"])
        g = generate(art, seed=int(entry["seed"]))
        save_attributed_graph(g, out / f"{entry['dir']}.npz")
        done.append({**entry, "edges": int(len(g.edges))})
        print(f"generated {entry['dir']}: {g.n} nodes, {len(g.edges)} edges ({time.time() - t0:.0f}s)", flush=True)
    (out / "index.json").write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()

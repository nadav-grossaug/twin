"""STAGE 1 (inside) of graph-aim-walks: real graph -> walks artifact. NOT PRIVATE (docs/snap/walks-plan.md).

Usage (repo root): ``PYTHONPATH=src:snap/inside python snap/inside/fit_walks.py [--seeds 1]``

Reads : snap/inside/data/processed/attributed_graph.npz, snap/inside/config/graph_aim_walks.json (+ its base config)
Writes: snap/transfer/artifacts_walks/graph_aim_walks_seed<k>/  (exact walk tables + exact base artifact)
        snap/transfer/artifacts_walks/index.json                (variant, epsilon = null, seed, dir, method)

The artifact is exact, so one seed is enough; the seed only changes the walk pool (sampling error).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph_aim_walks.inside.fit import fit_walks, load_config
from graph_aim_walks.transfer.artifact import save, validate
from graph_aim_walks.transfer.types import METHOD, SPECTRUM
from prepare import load_attributed_graph

INSIDE = Path(__file__).resolve().parent
ARTIFACTS = INSIDE.parent / "transfer" / "artifacts_walks"
VARIANT = "graph_aim_walks"


def summary(art) -> str:
    r, c, s = art.tables["return"], art.tables["coverage"], art.tables[SPECTRUM]
    ret = ", ".join(f"t={t}: {r[t - 1]:.4f}" for t in (2, 4, 8) if t <= len(r))
    return (f"  return {ret}; coverage at t={len(c)}: {c[-1]:.2f}; "
            f"spectrum top 5: {np.array2string(s[:5], precision=4)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--config", default=str(INSIDE / "config" / "graph_aim_walks.json"))
    ap.add_argument("--out", default=str(ARTIFACTS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    cfg = load_config(args.config)

    index = []
    for seed in range(args.seeds):
        ts = time.time()
        art = fit_walks(real, cfg, seed=seed)
        problems = validate(art)
        assert not problems, problems
        d = f"{VARIANT}_seed{seed}"
        path = out / d
        if not path.exists():
            save(art, path)
        index.append({"variant": VARIANT, "epsilon": None, "seed": seed, "dir": d, "method": METHOD})
        print(f"fitted {d} in {time.time() - ts:.1f}s (total {time.time() - t0:.0f}s)", flush=True)
        print(summary(art), flush=True)

    (out / "index.json").write_bytes(json.dumps(index, indent=1).encode("utf-8"))  # LF on every platform


if __name__ == "__main__":
    main()

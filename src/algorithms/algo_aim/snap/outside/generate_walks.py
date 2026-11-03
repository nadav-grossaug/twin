"""STAGE 2 (outside): walks artifact -> synthetic graphs (graph-aim-walks). Never sees the real graph.

NOT PRIVATE: the artifact holds exact statistics of the real graph (``docs/snap/walks-plan.md``).

Usage (repo root): ``PYTHONPATH=src python snap/outside/generate_walks.py [--artifacts snap/transfer/artifacts_walks]
[--out snap/outside/synthetic_walks] [--seeds 0 1 2] [--cycles 6]``

Reads : <artifacts>/index.json and the artifact folder(s) it lists (hash-verified on load)
Writes: <out>/graph_aim_walks_seed<k>.npz (one graph per GENERATION seed, all from the same exact artifact),
        <out>/graph_aim_walks_seed<k>.info.json (energies per walk table and base term, per cycle and final),
        <out>/index.json (variant, epsilon null, seed, dir, method, edges)

Allowed imports: standard library, numpy/pandas/scipy, ``graph_aim.transfer``, ``graph_aim.outside``,
``graph_aim_walks.transfer`` and ``graph_aim_walks.outside`` (tests/graph_aim_walks/test_boundary.py).
"""

import argparse
import json
import time
from pathlib import Path

OUTSIDE = Path(__file__).resolve().parent
ARTIFACTS = OUTSIDE.parent / "transfer" / "artifacts_walks"
METHOD = "graph_aim_walks"


def _energies(info: dict) -> dict:
    """The small, readable part of the generator's report."""
    keep = ("seed", "pool_size", "val_pool_size", "options", "stop", "seconds", "init_seconds", "final", "final_base",
            "spectrum_sq_gap_init", "spectrum_sq_gap_final", "warnings")
    out = {k: info[k] for k in keep if k in info}
    out["cycles"] = [{k: c[k] for k in ("cycle", "fresh", "base", "structure", "label", "seconds") if k in c}
                     for c in info["cycles"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--out", default=str(OUTSIDE / "synthetic_walks"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="generation seeds")
    ap.add_argument("--cycles", type=int, default=None, help="generator cycles (default: the generator's)")
    ap.add_argument("--pool-size", type=int, default=None)
    args = ap.parse_args()

    from graph_aim.transfer.graph_io import save_attributed_graph
    from graph_aim_walks.outside.generate import generate_walks
    from graph_aim_walks.transfer.artifact import load  # W1's loader (hash-verified)

    src, out = Path(args.artifacts), Path(args.out)
    entries = [e for e in json.loads((src / "index.json").read_text()) if (src / e["dir"]).is_dir()]
    if not entries:
        raise SystemExit(f"no artifact folder listed in {src / 'index.json'} is present")
    entry = entries[0]  # the artifact is exact: one artifact, several generation seeds
    art = load(src / entry["dir"])
    out.mkdir(parents=True, exist_ok=True)
    opts = {}
    if args.cycles is not None:
        opts["cycles"] = args.cycles
    if args.pool_size is not None:
        opts["pool_size"] = args.pool_size
    t0 = time.time()
    done = []
    for seed in args.seeds:
        g, info = generate_walks(art, seed=int(seed), **opts)
        name = f"{METHOD}_seed{seed}"
        save_attributed_graph(g, out / f"{name}.npz")
        (out / f"{name}.info.json").write_text(json.dumps(_energies(info), indent=1))
        done.append({"variant": METHOD, "epsilon": None, "seed": int(seed), "dir": name, "method": METHOD,
                     "artifact": entry["dir"], "edges": int(len(g.edges))})
        print(f"generated {name}: {g.n} nodes, {len(g.edges)} edges ({time.time() - t0:.0f}s)", flush=True)
    (out / "index.json").write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()

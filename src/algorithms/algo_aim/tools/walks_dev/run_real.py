"""Throwaway (eval zone): real-scale run of the W2 generator on the SNAP walks artifact.

Usage (repo root): ``OMP_NUM_THREADS=8 PYTHONPATH=src /opt/venv/bin/python3 tools/walks_dev/run_real.py
[--cycles 6] [--opt key=value ...] [--cached-init] [--profile] [--out tools/walks_dev/cache/run.json]``

Prints the per-table energy on fresh pools: v2 init (cycle 0) vs final, the base terms and the time. With
``--cached-init`` the v2 initial graph is read from / written to ``tools/walks_dev/cache/v2_seed<k>.npz`` (dev
speed only; the shipped path always runs the initializer).
"""

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

from graph_aim.outside.generate_v2 import generate_v2_with_info
from graph_aim.transfer.graph_io import load_attributed_graph, save_attributed_graph
from graph_aim_walks.outside.generate import generate_walks
from graph_aim_walks.transfer.artifact import load

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "tools" / "walks_dev" / "cache"


def parse(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--cached-init", action="store_true")
    ap.add_argument("--init-graph", default=None, help="start from this graph instead of the v2 initializer (dev)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="write the generated graph here (.npz)")
    args = ap.parse_args()
    art = load(REPO / "snap" / "transfer" / "artifacts_walks" / "graph_aim_walks_seed0")
    opts = {k: parse(v) for k, v in (o.split("=", 1) for o in args.opt)}
    init = None
    if args.cached_init:
        p = CACHE / f"v2_seed{args.seed}.npz"
        if not p.exists():
            g, _ = generate_v2_with_info(art.base, seed=args.seed)
            CACHE.mkdir(parents=True, exist_ok=True)
            save_attributed_graph(g, p)
        init = (load_attributed_graph(p), {"cached": True})
    if args.init_graph:
        init = (load_attributed_graph(args.init_graph), {"init_graph": args.init_graph})
    t0 = time.time()
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    g, info = generate_walks(art, seed=args.seed, cycles=args.cycles, init=init, verbose=True, **opts)
    if prof:
        prof.disable()
        pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
    print(f"total {time.time() - t0:.0f}s (init {info['init_seconds']:.0f}s)")
    first = info["cycles"][0]["fresh"]
    final = info["final"]
    print(f"{'table':34s} {'cells':>6s} {'v2 init':>10s} {'final':>10s}")
    for k in first:
        print(f"{k:34s} {first[k]['cells']:6d} {first[k]['energy']:10.1f} {final[k]['energy']:10.1f}")
    print(f"{'WALK TOTAL':34s} {sum(v['cells'] for v in first.values()):6d} "
          f"{sum(v['energy'] for v in first.values()):10.1f} {sum(v['energy'] for v in final.values()):10.1f}")
    b0, b1 = info["cycles"][0]["base"], info["final_base"]
    for k in b0:
        if not k.startswith("attr_2way"):
            print(f"{k:34s} {b0[k]['cells']:6d} {b0[k]['energy']:10.1f} {b1[k]['energy']:10.1f}")
    two0 = sum(v["energy"] for k, v in b0.items() if k.startswith("attr_2way"))
    two1 = sum(v["energy"] for k, v in b1.items() if k.startswith("attr_2way"))
    print(f"{'attr_2way (sum of 28)':34s} {'':6s} {two0:10.1f} {two1:10.1f}")
    print("spectrum sq gap", info.get("spectrum_sq_gap_init"), "->", info.get("spectrum_sq_gap_final"))
    for c in info["cycles"]:
        tot = sum(v["energy"] for v in c["fresh"].values())
        print(c["cycle"], round(tot), c.get("structure"), c.get("label"), round(c.get("seconds", 0)))
    if args.save:
        save_attributed_graph(g, args.save)
    if args.out:
        Path(args.out).write_text(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()

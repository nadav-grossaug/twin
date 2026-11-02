"""STAGE 1 (inside): real graph -> artifacts. The only stage that fits on the real data.

Usage (repo root): ``PYTHONPATH=src:snap/inside python snap/inside/fit_artifacts.py [--seeds 3] [--only ...]``

Reads : snap/inside/data/processed/attributed_graph.npz, snap/inside/config/graph_aim_v1.json
Writes: snap/transfer/artifacts/<variant>_seed<k>/  (the artifacts, the only thing that leaves the inside)
        snap/transfer/artifacts/index.json          (public list of what was produced: variant, nominal epsilon, seed)

Variants: ``graph_aim_eps<e>`` for e in 0.5, 1, 2, 4, 8, inf (inf = no noise, capped like the private runs) and
``graph_aim_nocap_nonprivate`` (no noise, degree cap far above the real max degree). ``--only`` takes variant names.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from graph_aim.inside.fit import fit
from graph_aim.transfer.artifact import save, validate
from prepare import load_attributed_graph

INSIDE = Path(__file__).resolve().parent
ARTIFACTS = INSIDE.parent / "transfer" / "artifacts"
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0, math.inf]
NOCAP = "graph_aim_nocap_nonprivate"
NOCAP_MIN_D = 5000


def eps_token(eps: float) -> str:
    return "inf" if not math.isfinite(eps) else f"{eps:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default=str(INSIDE / "config" / "graph_aim_v1.json"))
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--only", nargs="+", default=None, help="variant names to fit (default: all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    real = load_attributed_graph(INSIDE / "data" / "processed" / "attributed_graph.npz")
    cfg = json.loads(Path(args.config).read_text())

    variants = []  # (name, nominal epsilon, config)
    for eps in EPSILONS:
        private = math.isfinite(eps)
        c = dict(cfg, noise=private)
        if private:
            c.update(epsilon_edge=eps, epsilon_attr=eps)
        variants.append((f"graph_aim_eps{eps_token(eps)}", eps, c))
    variants.append((NOCAP, math.inf, dict(cfg, noise=False, D=max(NOCAP_MIN_D, int(np.bincount(real.edges.ravel(), minlength=real.n).max())))))

    index = []
    for name, eps, c in variants:
        if args.only and name not in args.only:
            continue
        for seed in range(args.seeds):
            art = fit(real, c, seed=seed)
            problems = validate(art)
            assert not problems, problems
            d = f"{name}_seed{seed}"
            path = out / d
            if not path.exists():
                save(art, path)
            index.append({"variant": name, "epsilon": eps if math.isfinite(eps) else None, "seed": seed, "dir": d})
            print(f"fitted {d} ({time.time() - t0:.0f}s)", flush=True)

    index_path = out / "index.json"
    if args.only and index_path.exists():  # keep entries of variants not refitted this time
        keep = [e for e in json.loads(index_path.read_text()) if e["variant"] not in args.only]
        index = keep + index
    index_path.write_bytes(json.dumps(index, indent=1).encode("utf-8"))  # LF on every platform (artifacts are byte-exact)


if __name__ == "__main__":
    main()

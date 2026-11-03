"""Package E: record the (R, q) calibration of ``generate_v2`` per artifact (chosen pair, scores, timing).

Outside-only inputs (artifacts), no real data. Run from the repo root:
  PYTHONPATH=src /opt/venv/bin/python3 tools/v2_runs/e/calibration_report.py [--artifacts DIR] [--only V ...]
Writes tools/v2_runs/e/calibration.csv (one row per artifact) and prints the grid of every artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from graph_aim.outside.generate_v2 import calibrate_regions
from graph_aim.transfer.artifact import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "snap" / "transfer" / "artifacts_v2"))
    ap.add_argument("--only", nargs="+", default=["graph_aim_eps4", "graph_aim_epsinf"])
    ap.add_argument("--out", default=str(HERE / "calibration.csv"))
    args = ap.parse_args()
    src = Path(args.artifacts)
    rows = []
    for entry in json.loads((src / "index.json").read_text()):
        if args.only and entry["variant"] not in args.only:
            continue
        if not (src / entry["dir"]).is_dir():
            continue
        art = load(src / entry["dir"])
        R, q, _, info = calibrate_regions(art, art.schema.n, int(entry["seed"]))
        grid = sorted(info["grid"], key=lambda c: (c["R"], c["q"]))
        rows.append({
            "variant": entry["variant"], "epsilon": entry["epsilon"], "seed": entry["seed"], "R": R, "q": q,
            "score": info["chosen"]["score"], "identity_score": info["identity_score"],
            "margin_vs_identity": info["margin_vs_identity"], "sigma": info["sigma"],
            "candidates": len(grid), "seconds": round(info["seconds"], 1),
            "grid": " ".join(f"{c['R']}/{c['q']:.2f}:{c['score']:.1f}" for c in grid),
        })
        print(f"{entry['dir']}: R={R} q={q} score={info['chosen']['score']:.1f} identity={info['identity_score']:.1f} "
              f"margin={info['margin_vs_identity']} sigma={info['sigma']:.2f} ({info['seconds']:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

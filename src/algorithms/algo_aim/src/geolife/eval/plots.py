"""Artifact plots (transition heatmap, users-per-place histogram). eval zone: needs matplotlib.

    PYTHONPATH=src python -m geolife.eval.plots <artifact dir> <png dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from geolife.transfer.artifact import Artifact, load_artifact


def plot_artifact(art: Artifact, out_dir, transition: str = "transitions",
                  users_per_place: str = "users_per_place") -> list[Path]:
    """Write transition heatmap and users-per-place histogram PNGs (those whose data exist)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by = {m.name: m for m in art.measurements}
    written = []
    if transition in by and by[transition].keys:
        m = by[transition]
        rows = sorted({k[0] for k in m.keys})
        cols = sorted({k[1] for k in m.keys})
        ri, ci = {r: i for i, r in enumerate(rows)}, {c: i for i, c in enumerate(cols)}
        mat = np.zeros((len(rows), len(cols)))
        for k, v in zip(m.keys, m.values):
            mat[ri[k[0]], ci[k[1]]] = v
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), rows, fontsize=6)
        ax.set_xlabel("destination")
        ax.set_ylabel("origin")
        fig.colorbar(im, ax=ax, label="noisy count")
        p = out / "transition_heatmap.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    if users_per_place in by and by[users_per_place].values:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(by[users_per_place].values, bins=30)
        ax.set_xlabel("noisy users per place")
        ax.set_ylabel("places")
        p = out / "users_per_place_hist.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written


if __name__ == "__main__":
    for q in plot_artifact(load_artifact(sys.argv[1]), sys.argv[2]):
        print(f"wrote {q}")

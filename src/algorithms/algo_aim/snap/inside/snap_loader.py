"""Load a Stanford SNAP edge list (https://snap.stanford.edu/data/) into pandas.

SNAP files are whitespace-separated ``src dst`` pairs (optionally extra columns),
``.txt`` or ``.txt.gz``, with ``#`` comment lines at the top that describe the graph, e.g.::

    # Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt
    # Nodes: 7115 Edges: 103689
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def read_header(path: str | Path) -> dict:
    """Parse the ``#`` comment block: raw lines plus directedness and declared sizes."""
    lines: list[str] = []
    with pd.io.common.get_handle(path, "r", compression="infer").handle as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line.lstrip("#").strip())
    text = "\n".join(lines)
    meta: dict = {"lines": lines, "directed": None, "nodes": None, "edges": None}
    if re.search(r"\bundirected\b", text, re.I):
        meta["directed"] = False
    elif re.search(r"\bdirected\b", text, re.I):
        meta["directed"] = True
    if m := re.search(r"Nodes:\s*([\d,]+)", text):
        meta["nodes"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"Edges:\s*([\d,]+)", text):
        meta["edges"] = int(m.group(1).replace(",", ""))
    return meta


def load_edgelist(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return ``(edges, meta)``; ``edges`` has columns ``src``, ``dst`` (+ any extras as ``c2``...)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download a graph from https://snap.stanford.edu/data/ "
            f"and put it in {DATA_DIR} (see data/README.md)."
        )
    meta = read_header(path)
    edges = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c")
    edges.columns = ["src", "dst"] + [f"c{i}" for i in range(2, edges.shape[1])]
    return edges, meta

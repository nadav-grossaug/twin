"""File format for an ``AttributedGraph`` (``.npz``, no pickle).

Shared by both sides of the boundary: the inside stage saves the prepared real graph with it, and the outside stage
saves the synthetic graph it generates with it (which the inside stage later reads back for scoring).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from graph_aim.transfer.types import AttributedGraph


def save_attributed_graph(g: AttributedGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n=np.int64(g.n),
        edges=g.edges,
        attrs=g.attrs.to_numpy(dtype=np.int64),
        columns=np.asarray(list(g.attrs.columns), dtype=str),
        domains_json=np.asarray(json.dumps(g.domains)),
    )


def load_attributed_graph(path: str | Path) -> AttributedGraph:
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        cols = [str(c) for c in z["columns"]]
        attrs = pd.DataFrame(z["attrs"], columns=cols, index=pd.RangeIndex(n))
        domains = json.loads(str(z["domains_json"]))
        return AttributedGraph(n=n, edges=z["edges"].astype(np.int64), attrs=attrs, domains=domains)

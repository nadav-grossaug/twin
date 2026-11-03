"""Shared toy graph and toy config for the graph-aim-walks tests (offline, small)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graph_aim.transfer.types import AttributedGraph

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "snap" / "inside" / "config"
TOY_DOMAINS = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}


def toy_graph(n: int = 150, p: float = 0.06, seed: int = 11) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in TOY_DOMAINS.items()})
    return AttributedGraph(n, edges, attrs, TOY_DOMAINS)


def toy_config(n_walks: int = 3000) -> dict:
    """The shipped walks config with the v2 base config scaled down to the toy graph (attributes a, b)."""
    base = json.loads((CONFIG_DIR / "graph_aim_v2.json").read_text())
    base.update(D=30, max_degree=60, homophily_attrs=None, homophily_groups=None, neigh_simpson_groups=None,
                attr_degree_attrs=None, spectrum_k=16)
    cfg = json.loads((CONFIG_DIR / "graph_aim_walks.json").read_text())
    cfg.update(base_config=base, n_walks=n_walks, spectrum_k=16)
    cfg["spec"] = {"length": 16, "lags": [1, 2, 4, 8], "windows": [4, 8, 16], "share_bins": 10, "dominance_bins": 5,
                   "degree_bin_edges": [0, 2, 6, 16], "groups": [["a"], ["a", "b"]], "share_attrs": "all",
                   "dominance_pairs": [["a", "b"]]}
    return cfg


@pytest.fixture(scope="session")
def graph() -> AttributedGraph:
    return toy_graph()


@pytest.fixture(scope="session")
def config() -> dict:
    return toy_config()


@pytest.fixture(scope="session")
def fitted(graph, config):
    from graph_aim_walks.inside.fit import fit_walks  # tests may import inside code

    return fit_walks(graph, config, seed=0)

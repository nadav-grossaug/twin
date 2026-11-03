"""Smoke test of graph_aim_walks.eval.viz.walk_profiles on small graphs (offline)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from graph_aim.transfer.types import AttributedGraph  # noqa: E402
from graph_aim_walks.eval.viz import walk_profiles  # noqa: E402

DOMAINS = {"school": ["s1", "s2", "other", "none"], "hometown": ["h1", "h2", "none"],
           "location": ["l1", "l2", "none"], "locale": ["x", "y"]}


def graph(seed: int, n: int = 150) -> AttributedGraph:
    G = nx.powerlaw_cluster_graph(n, 3, 0.3, seed=seed)
    edges = np.sort(np.array(list(G.edges()), dtype=np.int64), axis=1)
    rng = np.random.default_rng(seed)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n, edges, attrs, {c: list(d) for c, d in DOMAINS.items()})


def test_walk_profiles_draws_every_panel_and_marks_held_out():
    real, syns = graph(1), {"A": graph(2), "B": graph(3)}
    fig = walk_profiles(real, syns, n_walks=500)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 12
    visible = [ax for ax in fig.axes if ax.axison]
    assert len(visible) == 4 + 4 + 3  # 4 groups (school, hometown, location, hometown+location), 3 group attributes
    marked = sum(any(t.get_text() == "held out" for t in ax.texts) for ax in visible)
    assert marked == len(visible) - 1  # every panel but the spectrum
    # the window panels show the held-out windows 64 and 128 as ticks
    assert [t.get_text() for t in fig.axes[3].get_xticklabels()] == ["8", "16", "32", "64", "128"]
    plt.close(fig)

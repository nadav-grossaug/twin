"""Community structure of a graph: Louvain and label-propagation partitions, part sizes, conductance.

Eval zone. Definitions are those of the v2 diagnostic (``tools/v2_diag/community_strength.py``, numbers in
``docs/snap/v2-diagnostic.md`` section 1), so the values agree with it:

- Louvain: ``networkx.community.louvain_communities(G, seed=seed)`` (seed 0 by default); label propagation:
  ``networkx.community.label_propagation_communities(G)`` (semi-synchronous, deterministic); modularity
  ``networkx.community.modularity``.
- Conductance of a part ``S``: ``cut(S) / min(vol(S), vol(V) - vol(S))`` with ``vol`` the degree sum; 0 when the
  denominator is 0 (an isolated node, or a part holding all the edges).
- Node-weighted statistics weight every part by its size: they describe the part *a random node* belongs to.

``partition_metrics`` gives the scalars. The walk-matrix spectral gap is added by
``graph_aim_walks.eval.metrics.community_metrics`` (it uses the walk contract ``walkstats.transition_spectrum``, and
nothing in ``graph_aim`` imports ``graph_aim_walks``).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp

from graph_aim.eval.evaluate import adjacency, degrees
from graph_aim.transfer.types import AttributedGraph

LARGE_PART = 100  # "large part": at least this many nodes


def nx_graph(g: AttributedGraph) -> nx.Graph:
    """Every node (isolated ones too) and every edge of ``g``."""
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(map(tuple, np.asarray(g.edges, dtype=np.int64).reshape(-1, 2).tolist()))
    return G


def louvain(G: nx.Graph, seed: int = 0) -> list[set]:
    return [set(p) for p in nx.community.louvain_communities(G, seed=seed)]


def label_propagation(G: nx.Graph) -> list[set]:
    return [set(p) for p in nx.community.label_propagation_communities(G)]


def modularity(G: nx.Graph, parts: list[set]) -> float:
    return float(nx.community.modularity(G, parts)) if G.number_of_edges() else float("nan")


def conductances(adj: sp.csr_matrix, deg: np.ndarray, parts: list[set]) -> np.ndarray:
    """Conductance of every part (same order as ``parts``)."""
    vol_total = deg.sum()
    out = []
    for p in parts:
        idx = np.fromiter(p, dtype=np.int64)
        vol = deg[idx].sum()
        internal = adj[idx][:, idx].sum()  # each internal edge twice
        den = min(vol, vol_total - vol)
        out.append((vol - internal) / den if den > 0 else 0.0)
    return np.asarray(out, dtype=float)


def part_sizes(parts: list[set]) -> np.ndarray:
    return np.array([len(p) for p in parts], dtype=np.int64)


def node_part_sizes(parts: list[set]) -> np.ndarray:
    """For every node (over all parts), the size of its part: the node-weighted part-size sample."""
    s = part_sizes(parts)
    return np.repeat(s, s)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value whose cumulative weight reaches half the total."""
    o = np.argsort(values, kind="stable")
    v, w = np.asarray(values, dtype=float)[o], np.asarray(weights, dtype=float)[o]
    if w.sum() <= 0:
        return float("nan")
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def partitions(g: AttributedGraph, seed: int = 0) -> dict:
    """``{"graph", "adj", "deg", "louvain", "label_propagation"}``: the two partitions of ``g`` and what they need."""
    G = nx_graph(g)
    return {"graph": G, "adj": adjacency(g), "deg": degrees(g), "louvain": louvain(G, seed), "label_propagation": label_propagation(G)}


def partition_metrics(g: AttributedGraph, seed: int = 0, parts: dict | None = None) -> dict[str, float]:
    """Scalars of the community structure (Louvain unless named otherwise):

    ``modularity_louvain``, ``modularity_lpa``, ``n_parts``, ``nodes_in_large_parts_share`` (share of nodes in parts
    of >= ``LARGE_PART`` nodes), ``nw_median_part_size`` (node-weighted median part size), ``nw_conductance``
    (node-weighted mean conductance), plus ``n_parts_lpa`` and ``lpa_nodes_in_large_parts_share``."""
    p = parts if parts is not None else partitions(g, seed)
    G, adj, deg, lv, lp = p["graph"], p["adj"], p["deg"], p["louvain"], p["label_propagation"]
    s = part_sizes(lv)
    c = conductances(adj, deg, lv)
    n = max(int(s.sum()), 1)
    s_lp = part_sizes(lp)
    return {
        "modularity_louvain": modularity(G, lv),
        "modularity_lpa": modularity(G, lp),
        "n_parts": float(len(lv)),
        "nodes_in_large_parts_share": float(s[s >= LARGE_PART].sum() / n),
        "nw_median_part_size": weighted_median(s, s),
        "nw_conductance": float((c * s).sum() / n),
        "n_parts_lpa": float(len(lp)),
        "lpa_nodes_in_large_parts_share": float(s_lp[s_lp >= LARGE_PART].sum() / max(int(s_lp.sum()), 1)),
    }

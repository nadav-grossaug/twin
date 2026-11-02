"""Evaluation of synthetic attributed graphs against a real one (plan section 7).

Structure metrics, attribute metrics, scalar distances (``compare``), a comparison table and two
plotting helpers. Everything is numpy/scipy based except the k-core decomposition, which uses
networkx. A 4,000 node / 88k edge graph takes a few seconds.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from scipy.stats import ks_2samp, wasserstein_distance

from graph_aim.transfer.types import AttributedGraph

# Single colour scheme: ink for the real graph, validated categorical slots (light mode) for methods.
REAL_COLOUR = "#0b0b0b"
METHOD_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
SECONDARY_INK = "#52514e"


# ----------------------------------------------------------------------------------------- helpers
def adjacency(g: AttributedGraph) -> sp.csr_matrix:
    """Symmetric 0/1 CSR adjacency matrix."""
    n = g.n
    e = np.asarray(g.edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def degrees(g: AttributedGraph) -> np.ndarray:
    return np.bincount(np.asarray(g.edges, dtype=np.int64).ravel(), minlength=g.n).astype(np.int64)


def triangles_per_node(adj: sp.csr_matrix) -> np.ndarray:
    return np.asarray(adj.multiply(adj @ adj).sum(axis=1)).ravel() / 2.0


def local_clustering(adj: sp.csr_matrix, deg: np.ndarray | None = None) -> np.ndarray:
    """Local clustering coefficient per node (0 for degree < 2)."""
    deg = np.asarray(adj.sum(axis=1)).ravel() if deg is None else deg
    tri = triangles_per_node(adj)
    possible = deg * (deg - 1) / 2.0
    return np.divide(tri, possible, out=np.zeros(len(deg)), where=possible > 0)


def degree_ccdf(deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(k, P(D >= k))`` for every distinct degree value k present."""
    vals, counts = np.unique(deg, return_counts=True)
    ccdf = counts[::-1].cumsum()[::-1] / len(deg)
    return vals, ccdf


def _degree_assortativity(adj: sp.csr_matrix, deg: np.ndarray) -> float:
    coo = sp.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return 0.0
    a = np.concatenate([deg[coo.row], deg[coo.col]]).astype(float)
    b = np.concatenate([deg[coo.col], deg[coo.row]]).astype(float)
    if a.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _path_stats(adj: sp.csr_matrix, n_sources: int, seed: int) -> tuple[float, float]:
    """Mean path length and effective diameter (90th percentile) from sampled BFS sources."""
    n = adj.shape[0]
    if n < 2 or adj.nnz == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    src = rng.choice(n, size=min(n_sources, n), replace=False)
    dist = shortest_path(adj, method="D", directed=False, unweighted=True, indices=src)
    d = dist[np.isfinite(dist) & (dist > 0)]
    if d.size == 0:
        return 0.0, 0.0
    return float(d.mean()), float(np.percentile(d, 90))


def _top_eigenvalues(adj: sp.csr_matrix, k: int) -> np.ndarray:
    """Largest-magnitude ``k`` adjacency eigenvalues, sorted by descending magnitude (zero padded)."""
    n = adj.shape[0]
    out = np.zeros(k)
    if n < 2 or adj.nnz == 0:
        return out
    if n <= 60 or k >= n - 1:
        vals = np.linalg.eigvalsh(adj.toarray())
    else:
        vals = eigsh(adj, k=k, which="LM", return_eigenvectors=False, v0=np.ones(n) / np.sqrt(n))
    vals = vals[np.argsort(-np.abs(vals))][:k]
    out[: len(vals)] = vals
    return out


# ------------------------------------------------------------------------------- structure metrics
def structure_metrics(
    g: AttributedGraph, n_path_sources: int = 100, n_eigs: int = 10, seed: int = 0
) -> dict:
    """Structural summary of a graph. Deterministic for a fixed ``seed`` (path sampling only)."""
    adj = adjacency(g)
    deg = degrees(g)
    n = g.n
    m = len(g.edges)
    tri = triangles_per_node(adj)
    clus = local_clustering(adj, deg)
    wedges = float((deg * (deg - 1) / 2.0).sum())
    ncomp, labels = connected_components(adj, directed=False)
    giant = float(np.bincount(labels).max() / n) if n else 0.0
    mean_path, eff_diam = _path_stats(adj, n_path_sources, seed)

    # clustering by degree: mean local clustering for each distinct degree >= 2
    cbd_k = np.unique(deg[deg >= 2])
    cbd_c = np.array([clus[deg == k].mean() for k in cbd_k]) if len(cbd_k) else np.zeros(0)

    nxg = nx.Graph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(map(tuple, np.asarray(g.edges).tolist()))
    cn = nx.core_number(nxg) if m else {}
    core = np.array([cn.get(i, 0) for i in range(n)], dtype=np.int64)
    kc_vals, kc_counts = np.unique(core, return_counts=True)

    ccdf_k, ccdf_p = degree_ccdf(deg)
    return {
        "n": n,
        "m": m,
        "mean_degree": float(deg.mean()) if n else 0.0,
        "median_degree": float(np.median(deg)) if n else 0.0,
        "max_degree": int(deg.max()) if n else 0,
        "degree_std": float(deg.std()) if n else 0.0,
        "degree_ccdf": (ccdf_k, ccdf_p),
        "transitivity": float(tri.sum() / wedges) if wedges > 0 else 0.0,
        "avg_clustering": float(clus.mean()) if n else 0.0,
        "clustering_by_degree": (cbd_k, cbd_c),
        "triangles": float(tri.sum() / 3.0),
        "n_components": int(ncomp),
        "largest_component_share": giant,
        "mean_path_length": mean_path,
        "effective_diameter": eff_diam,
        "assortativity": _degree_assortativity(adj, deg),
        "degeneracy": int(core.max()) if n else 0,
        "kcore_profile": (kc_vals, kc_counts),
        "eigenvalues": _top_eigenvalues(adj, n_eigs),
    }


# ------------------------------------------------------------------------------- attribute metrics
def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _shared_columns(real: AttributedGraph, syn: AttributedGraph) -> list[str]:
    return [c for c in real.attrs.columns if c in syn.attrs.columns]


def _marginal(g: AttributedGraph, cols: list[str], sizes: list[int]) -> np.ndarray:
    idx = np.zeros(len(g.attrs), dtype=np.int64)
    for c, s in zip(cols, sizes):
        idx = idx * s + g.attrs[c].to_numpy(dtype=np.int64)
    total = int(np.prod(sizes))
    return np.bincount(idx, minlength=total) / max(len(idx), 1)


def _domain_size(real: AttributedGraph, syn: AttributedGraph, col: str) -> int:
    return max(len(real.domains[col]), int(real.attrs[col].max()) + 1, int(syn.attrs[col].max()) + 1)


def homophily_matrix(g: AttributedGraph, attrs: list[str], sizes: list[int] | None = None) -> np.ndarray:
    """``(G, G)`` symmetric edge counts between attribute groups.

    A group is the row-major product of the codes of ``attrs`` (same convention as the
    ``attr_edge_pair`` measurement). Each edge adds 1 to ``M[a, b]`` and ``M[b, a]``; an edge inside
    one group adds 1 to the diagonal entry once.
    """
    if sizes is None:
        sizes = [len(g.domains[c]) for c in attrs]
    grp = np.zeros(g.n, dtype=np.int64)
    for c, s in zip(attrs, sizes):
        grp = grp * s + g.attrs[c].to_numpy(dtype=np.int64)
    ng = int(np.prod(sizes))
    mat = np.zeros((ng, ng))
    e = np.asarray(g.edges, dtype=np.int64).reshape(-1, 2)
    if len(e):
        a, b = grp[e[:, 0]], grp[e[:, 1]]
        np.add.at(mat, (a, b), 1.0)
        off = a != b
        np.add.at(mat, (b[off], a[off]), 1.0)
    return mat


def homophily_error(real_m: np.ndarray, syn_m: np.ndarray) -> float:
    """Total variation distance between the two matrices, each normalised to a distribution."""
    p = real_m / max(real_m.sum(), 1e-12)
    q = syn_m / max(syn_m.sum(), 1e-12)
    return _tvd(p.ravel(), q.ravel())


def attribute_metrics(real: AttributedGraph, syn: AttributedGraph, homophily_attrs: list[str] | None = None) -> dict:
    """1-way and 2-way total variation distances and the homophily matrix error.

    ``homophily_attrs`` defaults to the first attribute column. Distances are in [0, 1].
    """
    cols = _shared_columns(real, syn)
    sizes = {c: _domain_size(real, syn, c) for c in cols}
    one = {c: _tvd(_marginal(real, [c], [sizes[c]]), _marginal(syn, [c], [sizes[c]])) for c in cols}
    two = {
        (a, b): _tvd(
            _marginal(real, [a, b], [sizes[a], sizes[b]]), _marginal(syn, [a, b], [sizes[a], sizes[b]])
        )
        for a, b in combinations(cols, 2)
    }
    h_attrs = [c for c in (homophily_attrs or cols[:1]) if c in cols]
    if h_attrs and real.n and syn.n:
        hs = [sizes[c] for c in h_attrs]
        rm, sm = homophily_matrix(real, h_attrs, hs), homophily_matrix(syn, h_attrs, hs)
        h_err = homophily_error(rm, sm)
    else:
        h_err = float("nan")
    return {
        "tvd_1way": one,
        "tvd_2way": two,
        "tvd_1way_mean": float(np.mean(list(one.values()))) if one else 0.0,
        "tvd_1way_max": float(np.max(list(one.values()))) if one else 0.0,
        "tvd_2way_mean": float(np.mean(list(two.values()))) if two else 0.0,
        "tvd_2way_max": float(np.max(list(two.values()))) if two else 0.0,
        "homophily_attrs": h_attrs,
        "homophily_error": h_err,
    }


# ----------------------------------------------------------------------------------------- compare
def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), 1e-12)


def compare(
    real: AttributedGraph,
    syn: AttributedGraph,
    homophily_attrs: list[str] | None = None,
    seed: int = 0,
    real_metrics: dict | None = None,
) -> dict:
    """Scalar distances between ``real`` and ``syn`` (0 means identical on that statistic).

    Pass ``real_metrics`` (a ``structure_metrics(real)`` result) to avoid recomputing it per method.
    """
    r = real_metrics if real_metrics is not None else structure_metrics(real, seed=seed)
    s = structure_metrics(syn, seed=seed)
    dr, ds = degrees(real), degrees(syn)
    k = min(len(r["eigenvalues"]), len(s["eigenvalues"]))
    er, es = r["eigenvalues"][:k], s["eigenvalues"][:k]
    out = {
        "degree_ks": float(ks_2samp(dr, ds).statistic),
        "degree_emd": float(wasserstein_distance(dr, ds)),
        "clustering_abs": abs(r["avg_clustering"] - s["avg_clustering"]),
        "clustering_rel": _rel(r["avg_clustering"], s["avg_clustering"]),
        "transitivity_abs": abs(r["transitivity"] - s["transitivity"]),
        "transitivity_rel": _rel(r["transitivity"], s["transitivity"]),
        "triangles_abs": abs(r["triangles"] - s["triangles"]),
        "triangles_rel": _rel(r["triangles"], s["triangles"]),
        "path_length_abs": abs(r["mean_path_length"] - s["mean_path_length"]),
        "path_length_rel": _rel(r["mean_path_length"], s["mean_path_length"]),
        "assortativity_abs": abs(r["assortativity"] - s["assortativity"]),
        "degeneracy_abs": abs(r["degeneracy"] - s["degeneracy"]),
        "degeneracy_rel": _rel(r["degeneracy"], s["degeneracy"]),
        "largest_component_abs": abs(r["largest_component_share"] - s["largest_component_share"]),
        "eigenvalue_rel": float(np.linalg.norm(er - es) / max(np.linalg.norm(er), 1e-12)),
    }
    am = attribute_metrics(real, syn, homophily_attrs)
    out.update(
        {
            "attr_tvd_1way_mean": am["tvd_1way_mean"],
            "attr_tvd_1way_max": am["tvd_1way_max"],
            "attr_tvd_2way_mean": am["tvd_2way_mean"],
            "attr_tvd_2way_max": am["tvd_2way_max"],
            "homophily_error": am["homophily_error"],
        }
    )
    return out


def comparison_table(real: AttributedGraph, syns: dict[str, AttributedGraph], **kwargs) -> pd.DataFrame:
    """DataFrame with one row per metric and one column per method in ``syns``."""
    rm = structure_metrics(real, seed=kwargs.get("seed", 0))
    cols = {name: compare(real, g, real_metrics=rm, **kwargs) for name, g in syns.items()}
    return pd.DataFrame(cols)


# ------------------------------------------------------------------------------------------ plots
def _colours(names: list[str]) -> dict[str, str]:
    return {n: METHOD_COLOURS[i % len(METHOD_COLOURS)] for i, n in enumerate(names)}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SECONDARY_INK)
    ax.tick_params(colors=SECONDARY_INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def _overlay(real_xy, syn_xys: dict, xlabel, ylabel, title, loglog, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    else:
        fig = ax.figure
    _style_axes(ax)
    cols = _colours(list(syn_xys))
    ax.plot(*real_xy, color=REAL_COLOUR, linewidth=2.0, label="real")
    for name, (x, y) in syn_xys.items():
        ax.plot(x, y, color=cols[name], linewidth=1.5, label=name)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=REAL_COLOUR, fontsize=10, loc="left")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8)
    return fig


def plot_degree_ccdf(real: AttributedGraph, syns: dict[str, AttributedGraph]):
    """Degree CCDF overlay (log-log). Returns a matplotlib Figure."""
    pos = lambda xy: (xy[0][xy[0] > 0], xy[1][xy[0] > 0])  # noqa: E731 - log axis needs k > 0
    rx = pos(degree_ccdf(degrees(real)))
    sx = {n: pos(degree_ccdf(degrees(g))) for n, g in syns.items()}
    return _overlay(rx, sx, "degree k", "P(D >= k)", "Degree CCDF", True)


def plot_clustering_by_degree(real: AttributedGraph, syns: dict[str, AttributedGraph]):
    """Mean local clustering by degree overlay (log x). Returns a matplotlib Figure."""
    rx = structure_metrics(real, n_path_sources=1, n_eigs=1)["clustering_by_degree"]
    sx = {n: structure_metrics(g, n_path_sources=1, n_eigs=1)["clustering_by_degree"] for n, g in syns.items()}
    fig = _overlay(rx, sx, "degree k", "mean local clustering", "Clustering by degree", False)
    fig.axes[0].set_xscale("log")
    return fig


def align_columns(syn: AttributedGraph, real: AttributedGraph) -> AttributedGraph:
    """Put the synthetic graph's attribute columns in the real graph's order and check the codes mean the same.

    A graph that went through an artifact on disk comes back with alphabetically sorted columns (JSON keys are
    sorted), while the metrics and plots index attributes by position, so alignment is by name and verified.
    """
    cols = list(real.attrs.columns)
    if sorted(cols) != sorted(syn.attrs.columns):
        raise ValueError(f"attribute columns differ: {sorted(cols)} vs {sorted(syn.attrs.columns)}")
    for c in cols:
        if list(syn.domains[c]) != list(real.domains[c]):
            raise ValueError(f"domain labels of {c!r} differ between real and synthetic graph")
    return AttributedGraph(n=syn.n, edges=syn.edges, attrs=syn.attrs[cols], domains={c: list(real.domains[c]) for c in cols})

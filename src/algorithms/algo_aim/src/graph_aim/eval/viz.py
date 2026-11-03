"""Visual comparison of a real attributed graph with synthetic ones (eval zone: needs both side by side).

Every function takes ``real`` (an ``AttributedGraph``) and ``syns`` (``{label: AttributedGraph}``, columns already
aligned to the real graph, see ``evaluate.align_columns``) and returns a matplotlib ``Figure``. Colours: the real
graph is always ink; synthetic graphs take the validated categorical slots in the order given.

Distribution plots: ``degree_distribution``, ``clustering_by_degree``, ``degree_mixing``, ``neighbor_degree``,
``paths_and_cores``, ``attribute_marginals``, ``homophily``, ``attribute_association``.
Attribute coupling (v1.1): ``degree_by_attribute`` (mean degree per value), ``homophily_lift`` (same-value lift per
attribute against chance) and ``degree_attribute_coupling`` (eta^2 of log degree on the attribute), definitions from
``graph_aim.eval.coupling``.
Communities: ``community_structure`` (Louvain / label propagation, part sizes, conductance; ``graph_aim.eval.community``).
Small examples: ``ego_examples`` (same-degree neighbourhoods side by side) and ``bfs_sample`` (a connected 60-node patch).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import breadth_first_order, shortest_path

from graph_aim.eval.community import LARGE_PART, conductances, modularity, node_part_sizes, part_sizes, partitions
from graph_aim.eval.coupling import _eta2, _homophily_stats, _missing_code, _mixing
from graph_aim.eval.evaluate import (
    GRID,
    METHOD_COLOURS,
    REAL_COLOUR,
    SECONDARY_INK,
    SURFACE,
    _degree_assortativity,
    adjacency,
    degree_ccdf,
    degrees,
    homophily_matrix,
    local_clustering,
)
from graph_aim.transfer.types import AttributedGraph

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK, "text.color": REAL_COLOUR,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fb", "#2a78d6", "#123a6e"])
DIVERGING = LinearSegmentedColormap.from_list("blue_orange", ["#2a78d6", "#f1f0ec", "#eb6834"])
NEUTRAL = ["#9a998f", "#cfcdc6"]  # "other", "none" in category colouring

Graphs = dict[str, AttributedGraph]


def _series(real: AttributedGraph, syns: Graphs):
    """``(label, graph, colour, linewidth)`` with the real graph first."""
    yield "real", real, REAL_COLOUR, 2.4
    for i, (name, g) in enumerate(syns.items()):
        yield name, g, METHOD_COLOURS[i % len(METHOD_COLOURS)], 1.8


def _legend(fig, real: AttributedGraph, syns: Graphs, y: float = -0.01) -> None:
    handles = [Line2D([0], [0], color=c, lw=2.4, label=n) for n, _, c, _ in _series(real, syns)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y))


def _log_edges(max_value: int, start: int = 1, n: int = 12) -> np.ndarray:
    return np.unique(np.round(np.geomspace(start, max_value + 1, n)).astype(int))


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


# ----------------------------------------------------------------------------- degree and clustering
def degree_distribution(real: AttributedGraph, syns: Graphs):
    """Degree CCDF (how many nodes have at least k friends) and degree by rank (the hubs), both log-log."""
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, g, c, lw in _series(real, syns):
            d = degrees(g)
            k, ccdf = degree_ccdf(d[d > 0])
            a.loglog(k, ccdf, color=c, lw=lw)
            ds = np.sort(d)[::-1]
            b.loglog(np.arange(1, len(ds) + 1), np.maximum(ds, 1), color=c, lw=lw)
        a.set(xlabel="degree k", ylabel="share of nodes with degree >= k", title="Degree distribution (CCDF)")
        b.set(xlabel="node rank (1 = biggest hub)", ylabel="degree", title="Hubs: degree by rank")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def clustering_by_degree(real: AttributedGraph, syns: Graphs):
    """Mean local clustering per degree bin, and the distribution of local clustering over nodes."""
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
        edges = _log_edges(int(max(degrees(g).max() for g in [real, *syns.values()])), start=2)
        for name, g, c, lw in _series(real, syns):
            d = degrees(g)
            cc = local_clustering(adjacency(g), d)
            ok = d >= 2
            idx = _bin_index(d[ok], edges)
            xs, ys = [], []
            for k in range(len(edges) - 1):
                m = idx == k
                if m.sum() >= 3:
                    xs.append(np.sqrt(edges[k] * edges[k + 1])), ys.append(cc[ok][m].mean())
            a.plot(xs, ys, color=c, lw=lw, marker="o", ms=4, mec=SURFACE, mew=1)
            v = np.sort(cc[ok])
            b.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=lw)
        a.set(xscale="log", ylim=(0, 1), xlabel="degree", ylabel="mean local clustering",
              title="Do friends of friends know each other?")
        b.set(xlabel="local clustering of a node", ylabel="share of nodes (cumulative)", title="Clustering across nodes")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def degree_mixing(real: AttributedGraph, syns: Graphs, n_bins: int = 11):
    """Who connects to whom by degree: P(neighbour's degree bin | own degree bin), one heatmap per graph."""
    graphs = [(n, g) for n, g, _, _ in _series(real, syns)]
    edges = _log_edges(int(max(degrees(g).max() for _, g in graphs)), n=n_bins + 1)
    mats, rs = [], []
    for _, g in graphs:
        d = degrees(g)
        e = np.asarray(g.edges, dtype=np.int64)
        bu, bv = _bin_index(d[e[:, 0]], edges), _bin_index(d[e[:, 1]], edges)
        m = np.zeros((len(edges) - 1,) * 2)
        np.add.at(m, (bu, bv), 1.0)
        np.add.at(m, (bv, bu), 1.0)
        mats.append(m / np.maximum(m.sum(1, keepdims=True), 1.0))
        rs.append(_degree_assortativity(adjacency(g), d))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(graphs), figsize=(3.6 * len(graphs) + 0.8, 3.9), squeeze=False)
        vmax = max(m.max() for m in mats)
        for ax, (name, _), m, r in zip(axes[0], graphs, mats, rs):
            im = ax.imshow(m, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="equal")
            ticks = np.arange(0, len(edges) - 1, 2)
            ax.set_xticks(ticks, [str(edges[t]) for t in ticks])
            ax.set_yticks(ticks, [str(edges[t]) for t in ticks])
            ax.grid(False)
            ax.set(xlabel="neighbour's degree (bin start)", title=f"{name}" + chr(10) + f"assortativity {r:.2f}")
        axes[0, 0].set_ylabel("own degree (bin start)")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="share of the node's friendships")
    return fig


def neighbor_degree(real: AttributedGraph, syns: Graphs):
    """Average degree of a node's friends against its own degree (the 'friendship paradox' curve)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        edges = _log_edges(int(max(degrees(g).max() for g in [real, *syns.values()])), start=1)
        for name, g, c, lw in _series(real, syns):
            adj, d = adjacency(g), degrees(g)
            knn = np.divide(adj @ d, d, out=np.zeros(g.n), where=d > 0)
            idx, ok = _bin_index(d, edges), d > 0
            xs, ys = [], []
            for k in range(len(edges) - 1):
                m = ok & (idx == k)
                if m.sum() >= 3:
                    xs.append(np.sqrt(edges[k] * edges[k + 1])), ys.append(knn[m].mean())
            ax.loglog(xs, ys, color=c, lw=lw, marker="o", ms=4, mec=SURFACE, mew=1)
        ax.set(xlabel="own degree", ylabel="mean degree of neighbours", title="Neighbour degree")
        _legend(fig, real, syns, y=-0.02)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def paths_and_cores(real: AttributedGraph, syns: Graphs, n_sources: int = 100, seed: int = 0):
    """Shortest-path length distribution (sampled sources) and core-number distribution."""
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, g, c, lw in _series(real, syns):
            adj = adjacency(g)
            src = np.random.default_rng(seed).choice(g.n, size=min(n_sources, g.n), replace=False)
            dist = shortest_path(adj, method="D", directed=False, unweighted=True, indices=src)
            d = dist[np.isfinite(dist) & (dist > 0)].astype(int)
            share = np.bincount(d) / max(len(d), 1)
            a.plot(np.arange(len(share)), share, color=c, lw=lw, marker="o", ms=4, mec=SURFACE, mew=1)
            core = np.fromiter(nx.core_number(nx.from_scipy_sparse_array(adj)).values(), dtype=int)
            vals, counts = np.unique(core, return_counts=True)
            b.plot(vals, counts[::-1].cumsum()[::-1] / len(core), color=c, lw=lw)
        a.set(xlabel="shortest path length (hops)", ylabel="share of node pairs", title="How far apart are two people?")
        b.set(xlabel="core number k", ylabel="share of nodes in the k-core or deeper", title="Dense cores")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ------------------------------------------------------------------------------------------ attributes
def _short(col: str, label: str) -> str:
    return label.split("_", 1)[1] if label.startswith(col + "_") else label


def attribute_marginals(real: AttributedGraph, syns: Graphs, ncols: int = 4):
    """Share of nodes per value, for every attribute: the real graph's bar next to each synthetic graph's."""
    cols = list(real.attrs.columns)
    series = list(_series(real, syns))
    nrows = int(np.ceil(len(cols) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.5 * nrows + 0.5), squeeze=False)
        w = 0.8 / len(series)
        for ax, col in zip(axes.ravel(), cols):
            labels = real.domains[col]
            for j, (name, g, c, _) in enumerate(series):
                share = np.bincount(g.attrs[col].to_numpy(dtype=np.int64), minlength=len(labels)) / g.n
                ax.bar(np.arange(len(labels)) + (j - (len(series) - 1) / 2) * w, share, width=w * 0.92, color=c)
            ax.set_xticks(range(len(labels)), [_short(col, x) for x in labels], rotation=45, ha="right", fontsize=7)
            ax.set(title=col)
            ax.grid(axis="x", visible=False)
        for ax in axes.ravel()[len(cols):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("share of nodes")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def homophily(real: AttributedGraph, syns: Graphs, attr: str):
    """P(friend's `attr` group | own group), annotated in percent; the diagonal is 'friends share my group'."""
    graphs = [(n, g) for n, g, _, _ in _series(real, syns)]
    labels = [_short(attr, x) for x in real.domains[attr]]
    mats, same = [], []
    for _, g in graphs:
        m = homophily_matrix(g, [attr], [len(real.domains[attr])])
        np.fill_diagonal(m, np.diag(m) * 2)  # an edge inside a group counts for both of its endpoints
        mats.append(m / np.maximum(m.sum(1, keepdims=True), 1.0))
        e = np.asarray(g.edges, dtype=np.int64)
        a = g.attrs[attr].to_numpy()
        same.append(float((a[e[:, 0]] == a[e[:, 1]]).mean()))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(graphs), figsize=(3.9 * len(graphs) + 0.6, 3.9), squeeze=False)
        for ax, (name, _), m, s in zip(axes[0], graphs, mats, same):
            im = ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=max(x.max() for x in mats), aspect="equal")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if m[i, j] > 0.5 * m.max() else REAL_COLOUR)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.grid(False)
            ax.set(xlabel=f"friend's {attr}", title=f"{name}" + chr(10) + f"same-group edges {100 * s:.0f}%")
        axes[0, 0].set_ylabel(f"own {attr}")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="share of friendships")
    return fig


def _cramers_v(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> float:
    t = np.zeros((ka, kb))
    np.add.at(t, (a, b), 1.0)
    t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    if min(t.shape) < 2:
        return 0.0
    exp = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / t.sum()
    chi2 = ((t - exp) ** 2 / np.maximum(exp, 1e-12)).sum()
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def _association(g: AttributedGraph, cols: list[str], domains: dict) -> np.ndarray:
    k = len(cols)
    m = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            ci, cj = g.attrs[cols[i]].to_numpy(dtype=np.int64), g.attrs[cols[j]].to_numpy(dtype=np.int64)
            m[i, j] = m[j, i] = _cramers_v(ci, cj, len(domains[cols[i]]), len(domains[cols[j]]))
    return m


def attribute_association(real: AttributedGraph, syns: Graphs):
    """How strongly attributes go together (Cramer's V between node attributes): real, synthetic, and the gap."""
    cols = list(real.attrs.columns)
    r = _association(real, cols, real.domains)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(syns), 3, figsize=(11.5, 3.6 * len(syns)), squeeze=False)
        for row, (name, g) in zip(axes, syns.items()):
            s = _association(g, cols, real.domains)
            for ax, m, title, cmap, lim in [(row[0], r, "real", SEQUENTIAL, (0, 0.5)), (row[1], s, name, SEQUENTIAL, (0, 0.5)),
                                            (row[2], s - r, f"{name} minus real", DIVERGING, (-0.3, 0.3))]:
                im = ax.imshow(m, cmap=cmap, vmin=lim[0], vmax=lim[1])
                ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(len(cols)), cols, fontsize=7)
                ax.grid(False)
                ax.set_title(title)
                fig.colorbar(im, ax=ax, shrink=0.75)
        fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------- attribute coupling (v1.1)
def _value_means(g: AttributedGraph, col: str, k: int) -> np.ndarray:
    """Mean degree per attribute value (0 where the value has no node)."""
    codes = g.attrs[col].to_numpy(dtype=np.int64)
    d = degrees(g)
    n = np.bincount(codes, minlength=k)[:k].astype(np.float64)
    s = np.bincount(codes, weights=d, minlength=k)[:k]
    return np.divide(s, n, out=np.zeros(k), where=n > 0)


def _lift(g: AttributedGraph, col: str, missing: dict | None) -> float:
    """Same-value lift of ``col`` (coupling plan section 1): same-observed-value edge share over chance."""
    size = max(len(g.domains[col]), int(g.attrs[col].max()) + 1 if len(g.attrs) else 0)
    mat, ends, _, _ = _mixing(g, col, size, _missing_code(g, col, missing))
    return _homophily_stats(mat, ends, False)["lift"]


def _grouped_bars(ax, series, values: list[np.ndarray]) -> None:
    """One bar group per x position; the real graph's bar first, then each synthetic graph's."""
    w = 0.8 / len(series)
    for j, ((_, _, c, _), v) in enumerate(zip(series, values)):
        ax.bar(np.arange(len(v)) + (j - (len(series) - 1) / 2) * w, v, width=w * 0.92, color=c)
    ax.grid(axis="x", visible=False)


def degree_by_attribute(real: AttributedGraph, syns: Graphs, attrs: list[str]):
    """Mean degree per attribute value: which kinds of people have many friends, one panel per attribute."""
    series = list(_series(real, syns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(attrs), figsize=(3.5 * len(attrs) + 0.4, 3.2), squeeze=False)
        for ax, col in zip(axes[0], attrs):
            labels = real.domains[col]
            _grouped_bars(ax, series, [_value_means(g, col, len(labels)) for _, g, _, _ in series])
            ax.set_xticks(range(len(labels)), [_short(col, x) for x in labels], rotation=45, ha="right", fontsize=7)
            ax.set(title=col)
        axes[0, 0].set_ylabel("mean degree")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def homophily_lift(real: AttributedGraph, syns: Graphs, missing: dict | None = None):
    """Same-value lift per attribute: the share of friendships between people with the same observed value, over the
    chance level of pairing observed endpoints at random. 1 is chance; unobserved ('none') endpoints are excluded.
    ``missing[col]`` overrides the label that means "not observed" (default ``"none"`` where the domain has it)."""
    cols = list(real.attrs.columns)
    series = list(_series(real, syns))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        _grouped_bars(ax, series, [np.array([_lift(g, c, missing) for c in cols]) for _, g, _, _ in series])
        ax.axhline(1.0, color=SECONDARY_INK, linestyle=":", linewidth=1.2)
        ax.text(len(cols) - 0.55, 1.02, "chance", color=SECONDARY_INK, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="same-value lift", title="Do friends share the attribute?")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def degree_attribute_coupling(real: AttributedGraph, syns: Graphs):
    """Correlation ratio eta^2 of ``log(1 + degree)`` on each attribute: how much of "who has many friends" the
    attribute explains (0 = nothing, 1 = everything)."""
    cols = list(real.attrs.columns)
    series = list(_series(real, syns))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        vals = []
        for _, g, _, _ in series:
            d = degrees(g)
            vals.append(np.array([_eta2(g.attrs[c].to_numpy(dtype=np.int64), d,
                                        max(len(g.domains[c]), int(g.attrs[c].max()) + 1)) for c in cols]))
        _grouped_bars(ax, series, vals)
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="eta$^2$ of log(1 + degree)", title="How much does the attribute say about degree?")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ------------------------------------------------------------------------------------------- communities
def community_structure(real: AttributedGraph, syns: Graphs, seed: int = 0):
    """Community structure (definitions in ``graph_aim.eval.community``, as the v2 diagnostic): (a) Louvain and
    label-propagation modularity; (b) the size of the Louvain part each node belongs to (CCDF over nodes, log x);
    (c) the conductance of the part each node belongs to (CDF over nodes, i.e. per part weighted by size);
    (d) the share of nodes in Louvain parts of >= ``LARGE_PART`` nodes, with the number of parts on each bar."""
    series = list(_series(real, syns))
    stats = []
    for _, g, _, _ in series:
        p = partitions(g, seed)
        lv = p["louvain"]
        s = part_sizes(lv)
        stats.append({
            "mod": [modularity(p["graph"], lv), modularity(p["graph"], p["label_propagation"])],
            "node_sizes": np.sort(node_part_sizes(lv)),
            "cond": conductances(p["adj"], p["deg"], lv), "sizes": s,
            "large": float(s[s >= LARGE_PART].sum() / max(int(s.sum()), 1)), "n_parts": len(lv),
        })
    with plt.rc_context(STYLE):
        fig, (a, b, c, d) = plt.subplots(1, 4, figsize=(15, 3.8))
        _grouped_bars(a, series, [np.array(st["mod"]) for st in stats])
        a.set_xticks([0, 1], ["Louvain", "label propagation"])
        a.set(ylabel="modularity", ylim=(0, 1), title="(a) Modularity")
        for (name, _, col, lw), st in zip(series, stats):
            v = st["node_sizes"]
            b.step(v, 1.0 - np.arange(len(v)) / len(v), where="post", color=col, lw=lw)
            o = np.argsort(st["cond"], kind="stable")
            w = np.cumsum(st["sizes"][o]) / max(st["sizes"].sum(), 1)
            c.step(np.concatenate([[0.0], st["cond"][o]]), np.concatenate([[0.0], w]), where="post", color=col, lw=lw)
        b.set(xscale="log", xlabel="size of the node's Louvain part", ylabel="share of nodes in a part >= size",
              title="(b) Part sizes, per node")
        c.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="conductance of the node's Louvain part",
              ylabel="share of nodes (cumulative)", title="(c) How closed are the parts?")
        _grouped_bars(d, series, [np.array([st["large"]]) for st in stats])
        w = 0.8 / len(series)
        for j, st in enumerate(stats):
            d.text((j - (len(series) - 1) / 2) * w, st["large"] + 0.02, f"{st['n_parts']}\nparts", ha="center",
                   va="bottom", fontsize=7, color=SECONDARY_INK)
        d.set_xticks([0], [f"parts of >= {LARGE_PART} nodes"])
        d.set(ylim=(0, 1.15), ylabel="share of nodes", title="(d) Large parts")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# --------------------------------------------------------------------------------------- small examples
def _category_colours(domain: list[str]) -> list[str]:
    """Fixed colour per attribute value; the explicit 'other' and 'none' labels are neutral greys."""
    out, k = [], 0
    for lab in domain:
        if lab == "other":
            out.append(NEUTRAL[0])
        elif lab == "none":
            out.append(NEUTRAL[1])
        else:
            out.append(METHOD_COLOURS[k % len(METHOD_COLOURS)])
            k += 1
    return out


def _representative(deg: np.ndarray, clu: np.ndarray, target: int) -> int:
    """Node whose degree is nearest `target`; among those, the one with the median local clustering."""
    d = deg[deg > 0]
    nearest = d[np.argmin(np.abs(d - target))]
    cand = np.flatnonzero(deg == nearest)
    return int(cand[np.argsort(clu[cand])[len(cand) // 2]])


def _draw(ax, adj, nodes, colours, ego=None, seed=1) -> None:
    sub = adj[nodes][:, nodes]
    g = nx.from_scipy_sparse_array(sub)
    pos = nx.spring_layout(g, seed=seed, k=1.4 / np.sqrt(max(len(nodes), 2)), iterations=120)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#b9b7ae", width=0.5, alpha=0.6)
    size = np.full(len(nodes), 26.0)
    if ego is not None:
        size[ego] = 90.0
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, node_size=size, linewidths=0.4, edgecolors=SURFACE)
    if ego is not None:
        nx.draw_networkx_nodes(g, pos, nodelist=[ego], ax=ax, node_color=[colours[ego]], node_size=110,
                               linewidths=1.6, edgecolors=REAL_COLOUR)
    ax.axis("off")
    ax.grid(False)


def _attr_legend(fig, real: AttributedGraph, attr: str, y: float = -0.01) -> None:
    cols = _category_colours(real.domains[attr])
    handles = [Line2D([0], [0], marker="o", ls="", color=c, label=_short(attr, lab), ms=7)
               for lab, c in zip(real.domains[attr], cols)]
    fig.legend(handles=handles, title=f"colour = {attr}", loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, y))


def ego_examples(real: AttributedGraph, syns: Graphs, attr: str, targets=(12, 30, 70)):
    """One neighbourhood per target degree in every graph, picked by the same rule (typical clustering at that degree).

    The ringed dot is the person, the other dots are friends coloured by `attr`; lines are friendships among them.
    """
    graphs = [(n, g) for n, g, _, _ in _series(real, syns)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(graphs), len(targets), figsize=(3.6 * len(targets), 3.4 * len(graphs) + 0.5), squeeze=False)
        for row, (name, g) in zip(axes, graphs):
            adj, d = adjacency(g), degrees(g)
            clu = local_clustering(adj, d)
            pal = np.asarray(_category_colours(real.domains[attr]))
            codes = g.attrs[attr].to_numpy(dtype=np.int64)
            for ax, t in zip(row, targets):
                v = _representative(d, clu, t)
                nodes = np.concatenate([[v], adj[v].indices])
                _draw(ax, adj, nodes, list(pal[codes[nodes]]), ego=0)
                ax.set_title(f"{name}" + chr(10) + f"degree {d[v]}, clustering {clu[v]:.2f}", fontsize=9, loc="left")
        _attr_legend(fig, real, attr)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def bfs_sample(real: AttributedGraph, syns: Graphs, attr: str, size: int = 60, start_degree: int = 20):
    """A connected patch of `size` nodes (breadth-first from a typical node), coloured by `attr`."""
    graphs = [(n, g) for n, g, _, _ in _series(real, syns)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(graphs), figsize=(4.4 * len(graphs), 4.6), squeeze=False)
        for ax, (name, g) in zip(axes[0], graphs):
            adj, d = adjacency(g), degrees(g)
            v = _representative(d, local_clustering(adj, d), start_degree)
            nodes = breadth_first_order(adj, v, directed=False, return_predecessors=False)[:size]
            pal = np.asarray(_category_colours(real.domains[attr]))
            sub = adj[nodes][:, nodes]
            dens = sub.nnz / 2 / max(len(nodes) * (len(nodes) - 1) / 2, 1)
            _draw(ax, adj, nodes, list(pal[g.attrs[attr].to_numpy(dtype=np.int64)[nodes]]), ego=0)
            ax.set_title(f"{name}: {len(nodes)} nodes, density {dens:.2f}", loc="left")
        _attr_legend(fig, real, attr)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig

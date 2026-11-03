"""Attribute composition metrics: how concentrated are attribute values inside neighbourhoods?

Eval zone (needs the real graph and the synthetic graph side by side). The gap these metrics measure is stated in
``docs/snap/attribute-composition-gap.md``; the definitions are ``docs/snap/composition-plan.md`` section 3 (and 4.1
for the Simpson statistic, which is exactly what the ``attr_neigh_simpson`` measurement releases, so eval and
artifact agree). Generic over the schema: a *group* is a list of attribute columns whose row-major code product is
the group code (the ``attr_edge_pair`` convention); nothing is named in code.

For a node ``u`` of degree ``d_u`` (number of neighbours in the graph at hand) and ``n_{u,g}`` = neighbours with
group code ``g``:

- ``simpson_by_degree``: ``s_u = sum_g (n_{u,g} / d_u)^2`` for ``d_u >= min_degree`` (nodes below contribute 0
  and are excluded from every mean), averaged per public degree bin, real vs synthetic vs absolute difference, plus
  the node-weighted mean over bins (``simpson_mean``).
- ``two_hop_same_rate``: share of 2-paths ``(u, w, v)``, ``u != v``, whose endpoints share the group code
  (``sum_w sum_g C(n_{w,g}, 2) / sum_w C(d_w, 2)``, every node), real vs synthetic, and the independence
  prediction ``sum_g a_g^2`` from the edge-endpoint shares ``a_g`` of each graph.
- ``dominant_share_hist``: histogram (10 bins on [0, 1]) over nodes with ``d_u >= min_degree`` of
  ``max_g n_{u,g} / d_u``, real vs synthetic, and their total variation distance.
- ``gap_closed_table``: the scalar metrics against the permuted-attribute null (``coupling.permuted_null``), the
  ``coupling.gap_closed_table`` convention: 0 = as bad as the null, 1 = as good as the real graph.

The pure helpers ``neigh_simpson_sums`` / ``neigh_simpson_mean_by_bin`` work on a CSR adjacency and a code vector
(numpy / scipy.sparse only) so that the inside measurement and this module can be checked against each other.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from graph_aim.eval.coupling import gap_closed
from graph_aim.eval.evaluate import adjacency, degrees
from graph_aim.transfer.binning import check_bin_edges, degree_bin
from graph_aim.transfer.types import AttributedGraph

# scalar metrics compared as levels (gap measured from the null to the real value) and as distances (real value is 0)
LEVEL_METRICS = ("simpson_mean", "two_hop_same_rate")
DISTANCE_METRICS = ("dominant_share_tvd",)
# a group whose real value is this close to the null has no concentration to reproduce: its gap is NaN
MIN_GAP_DENOMINATOR = {"simpson_mean": 0.01, "two_hop_same_rate": 0.01, "dominant_share_tvd": 0.02}
N_SHARE_BINS = 10
SHARE_EDGES = np.linspace(0.0, 1.0, N_SHARE_BINS + 1)


# ----------------------------------------------------------------------------------------------- helpers
def group_label(attrs) -> str:
    """Column label of a group: the attribute names joined by ``+``."""
    return "+".join(attrs)


def _check_min_degree(min_degree) -> int:
    if int(min_degree) != min_degree or min_degree < 1:
        raise ValueError(f"min_degree must be an integer >= 1, got {min_degree!r}")
    return int(min_degree)


def _domain_size(real: AttributedGraph, syn: AttributedGraph, col: str) -> int:
    """Domain size shared by both graphs (the real labels, or larger codes if either graph has them)."""

    def top(g: AttributedGraph) -> int:
        return int(g.attrs[col].max()) + 1 if len(g.attrs) else 0

    return max(len(real.domains[col]), top(real), top(syn))


def group_codes(g: AttributedGraph, attrs, dims) -> np.ndarray:
    """Row-major product code of the columns ``attrs`` with domain sizes ``dims`` (as ``attr_edge_pair``)."""
    cols = list(attrs)
    if not cols:
        raise ValueError("a group needs at least one attribute")
    if len(g.attrs) == 0:
        return np.zeros(0, dtype=np.int64)
    arrays = tuple(g.attrs[c].to_numpy(dtype=np.int64) for c in cols)
    return np.ravel_multi_index(arrays, tuple(int(d) for d in dims)).astype(np.int64)


def neighbour_group_counts(adjacency_csr, codes, n_groups: int) -> np.ndarray:
    """Dense ``(n, G)`` array ``N[u, g]`` = number of neighbours of ``u`` with group code ``g``.

    ``adjacency_csr`` is a symmetric 0/1 sparse matrix without diagonal (any scipy.sparse format); ``codes`` an
    int vector in ``[0, n_groups)`` with one entry per node.
    """
    adj = sp.csr_matrix(adjacency_csr)
    n = adj.shape[0]
    codes = np.asarray(codes, dtype=np.int64).ravel()
    if adj.shape[1] != n or len(codes) != n:
        raise ValueError(f"adjacency {adj.shape} and codes ({len(codes)},) do not match")
    if n == 0:
        return np.zeros((0, int(n_groups)))
    if codes.min() < 0 or codes.max() >= n_groups:
        raise ValueError(f"codes must lie in [0, {n_groups})")
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, int(n_groups)))
    return np.asarray((adj @ onehot).todense(), dtype=np.float64)


def neigh_simpson_per_node(adjacency_csr, codes, n_groups: int, degree, min_degree) -> np.ndarray:
    """``s_u = [d_u >= min_degree] * sum_g (n_{u,g} / d_u)^2`` per node (plan section 4.1).

    ``degree`` is the number of neighbours of each node in ``adjacency_csr`` (passed in, so the caller decides
    whether it is the capped or the uncapped graph; it must be the row sum of the adjacency).
    """
    min_degree = _check_min_degree(min_degree)
    counts = neighbour_group_counts(adjacency_csr, codes, n_groups)
    d = np.asarray(degree, dtype=np.float64).ravel()
    if len(d) != counts.shape[0]:
        raise ValueError("degree must have one entry per node")
    active = d >= min_degree
    s = np.zeros(len(d))
    if active.any():
        s[active] = ((counts[active] / d[active, None]) ** 2).sum(axis=1)
    return s


def neigh_simpson_sums(adjacency_csr, codes, n_groups: int, degree, bin_edges, min_degree) -> np.ndarray:
    """``(B,)`` sums of ``s_u`` per degree bin (``bin_edges`` as ``transfer.binning``: last bin open-ended).

    Nodes with ``d_u < min_degree`` contribute 0. This is column 0 of the ``attr_neigh_simpson`` statistic.
    """
    edges = check_bin_edges(bin_edges)
    s = neigh_simpson_per_node(adjacency_csr, codes, n_groups, degree, min_degree)
    b = degree_bin(np.asarray(degree), edges)
    return np.bincount(b, weights=s, minlength=len(edges) - 1).astype(np.float64)


def neigh_simpson_mean_by_bin(
    adjacency_csr, codes, n_groups: int, degree, bin_edges, min_degree
) -> tuple[np.ndarray, np.ndarray]:
    """``(means, counts)`` per degree bin: mean ``s_u`` over the nodes of the bin with ``d_u >= min_degree`` (NaN
    for a bin without such nodes) and the number of those nodes."""
    edges = check_bin_edges(bin_edges)
    sums = neigh_simpson_sums(adjacency_csr, codes, n_groups, degree, edges, min_degree)
    d = np.asarray(degree)
    b = degree_bin(d, edges)
    counts = np.bincount(b[d >= min_degree], minlength=len(edges) - 1).astype(np.float64)
    means = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
    return means, counts


def two_hop_same_rate(counts: np.ndarray, degree) -> float:
    """Share of 2-paths whose endpoints share the group code, from the neighbour counts ``N[u, g]``; NaN if the
    graph has no 2-path."""
    d = np.asarray(degree, dtype=np.float64)
    total = float((d * (d - 1.0)).sum() / 2.0)
    if total <= 0:
        return float("nan")
    same = float((counts * (counts - 1.0)).sum() / 2.0)
    return same / total


def independence_prediction(codes, degree, n_groups: int) -> float:
    """``sum_g a_g^2`` for the edge-endpoint shares ``a_g`` (share of edge ends at nodes with code ``g``); NaN for
    an edgeless graph. The two-hop same rate of a graph whose attributes are independent of its structure."""
    ends = np.bincount(np.asarray(codes, dtype=np.int64), weights=np.asarray(degree, dtype=np.float64), minlength=n_groups)
    total = ends.sum()
    if total <= 0:
        return float("nan")
    a = ends / total
    return float(a @ a)


def dominant_share(counts: np.ndarray, degree, min_degree) -> np.ndarray:
    """``max_g n_{u,g} / d_u`` over the nodes with ``d_u >= min_degree`` (in node order)."""
    min_degree = _check_min_degree(min_degree)
    d = np.asarray(degree, dtype=np.float64)
    active = d >= min_degree
    if not active.any():
        return np.zeros(0)
    return counts[active].max(axis=1) / d[active]


def share_histogram(shares: np.ndarray) -> np.ndarray:
    """Shares of ``N_SHARE_BINS`` equal bins on [0, 1] (1.0 falls in the last bin); zeros for no nodes."""
    if len(shares) == 0:
        return np.zeros(N_SHARE_BINS)
    h, _ = np.histogram(np.clip(shares, 0.0, 1.0), bins=SHARE_EDGES)
    return h / len(shares)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


# ------------------------------------------------------------------------------------------------- metrics
class _Side:
    """Everything of one graph that the group metrics need; the adjacency is built once per graph."""

    def __init__(self, g: AttributedGraph):
        self.g = g
        self.adj = adjacency(g)
        self.deg = degrees(g)


def _group_metrics(real: _Side, syn: _Side, attrs: list[str], edges: np.ndarray, min_degree: int) -> dict:
    dims = [_domain_size(real.g, syn.g, c) for c in attrs]
    n_groups = int(np.prod(dims))
    sides = {}
    for name, side in (("real", real), ("syn", syn)):
        codes = group_codes(side.g, attrs, dims)
        counts = neighbour_group_counts(side.adj, codes, n_groups)
        means, n_active = neigh_simpson_mean_by_bin(side.adj, codes, n_groups, side.deg, edges, min_degree)
        total = n_active.sum()
        s = neigh_simpson_per_node(side.adj, codes, n_groups, side.deg, min_degree)
        dom = dominant_share(counts, side.deg, min_degree)
        sides[name] = {
            "bin_means": means,
            "bin_counts": n_active,
            "mean": float(s.sum() / total) if total > 0 else float("nan"),
            "two_hop": two_hop_same_rate(counts, side.deg),
            "indep": independence_prediction(codes, side.deg, n_groups),
            "hist": share_histogram(dom),
            "n_active": int(total),
        }
    r, s = sides["real"], sides["syn"]
    n_bins = len(edges) - 1
    tvd = _tvd(r["hist"], s["hist"]) if r["n_active"] and s["n_active"] else float("nan")
    scalars = {
        "simpson_mean_real": r["mean"],
        "simpson_mean_syn": s["mean"],
        "simpson_mean_absdiff": abs(r["mean"] - s["mean"]),
        "two_hop_same_rate_real": r["two_hop"],
        "two_hop_same_rate_syn": s["two_hop"],
        "two_hop_same_rate_absdiff": abs(r["two_hop"] - s["two_hop"]),
        "two_hop_indep_real": r["indep"],
        "two_hop_indep_syn": s["indep"],
        "dominant_share_tvd": tvd,
    }
    for b in range(n_bins):
        scalars[f"simpson_bin{b}_real"] = float(r["bin_means"][b])
        scalars[f"simpson_bin{b}_syn"] = float(s["bin_means"][b])
        scalars[f"simpson_bin{b}_absdiff"] = abs(float(r["bin_means"][b] - s["bin_means"][b]))
    for b in range(N_SHARE_BINS):
        scalars[f"dominant_share_bin{b}_real"] = float(r["hist"][b])
        scalars[f"dominant_share_bin{b}_syn"] = float(s["hist"][b])
    return {
        "attrs": list(attrs),
        "n_groups": n_groups,
        "simpson_by_degree": {
            "bin_edges": edges.tolist(),
            "real": r["bin_means"],
            "syn": s["bin_means"],
            "absdiff": np.abs(r["bin_means"] - s["bin_means"]),
            "n_real": r["bin_counts"],
            "n_syn": s["bin_counts"],
            "mean_real": r["mean"],
            "mean_syn": s["mean"],
            "mean_absdiff": abs(r["mean"] - s["mean"]),
        },
        "two_hop_same_rate": {
            "real": r["two_hop"],
            "syn": s["two_hop"],
            "absdiff": abs(r["two_hop"] - s["two_hop"]),
            "indep_real": r["indep"],
            "indep_syn": s["indep"],
        },
        "dominant_share_hist": {"edges": SHARE_EDGES.copy(), "real": r["hist"], "syn": s["hist"], "tvd": tvd},
        "scalars": {k: float(v) for k, v in scalars.items()},
    }


def _nanmean_over(dicts: list[dict]) -> dict:
    keys = sorted({k for d in dicts for k in d})
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts if k in d], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def composition_metrics(real: AttributedGraph, syn: AttributedGraph, *, groups, bin_edges, min_degree) -> dict:
    """Neighbourhood composition of attribute groups, real vs synthetic.

    ``groups`` is a list of attribute groups (each a list of column names present in both graphs; the group code
    is their row-major product); ``bin_edges`` the public degree bins (``transfer.binning`` convention);
    ``min_degree >= 1`` the degree below which a node has no neighbourhood statistic. Degrees are each graph's own
    (uncapped) degrees. Returns ``{"groups": {label: {...}}, "mean": {scalar: nanmean over groups},
    "bin_edges": [...], "min_degree": int}`` where each group entry holds the structured ``simpson_by_degree``,
    ``two_hop_same_rate`` and ``dominant_share_hist`` results and a flat ``scalars`` dict (the ``cmp_*`` columns).
    Columns are matched by name; the synthetic domain may be larger than the real one.
    """
    edges = check_bin_edges(bin_edges)
    min_degree = _check_min_degree(min_degree)
    groups = [[g] if isinstance(g, str) else list(g) for g in groups]
    for attrs in groups:
        missing = [c for c in attrs if c not in real.attrs.columns or c not in syn.attrs.columns]
        if missing:
            raise ValueError(f"group {attrs} names columns absent from a graph: {missing}")
    r, s = _Side(real), _Side(syn)
    out = {}
    for attrs in groups:
        out[group_label(attrs)] = _group_metrics(r, s, attrs, edges, min_degree)
    return {
        "groups": out,
        "mean": _nanmean_over([g["scalars"] for g in out.values()]),
        "bin_edges": edges.tolist(),
        "min_degree": min_degree,
    }


# ------------------------------------------------------------------------------------------ gap closed
def permuted_null_by_degree(real: AttributedGraph, bin_edges, seed: int = 0) -> AttributedGraph:
    """The real graph with every attribute column permuted independently *within each degree bin*.

    ``coupling.permuted_null`` also breaks the degree x attribute coupling, which moves the i.i.d. level of the
    Simpson index (the edge-endpoint shares change); this null keeps the ``attr_degree`` tables exact, so it is the
    floor for what a label assignment with i.i.d. rows per degree stratum can reach on the real structure.
    """
    rng = np.random.default_rng(seed)
    bins = degree_bin(degrees(real), bin_edges)
    attrs = real.attrs.copy()
    for c in attrs.columns:
        col = attrs[c].to_numpy().copy()
        for b in np.unique(bins):
            idx = np.flatnonzero(bins == b)
            col[idx] = col[rng.permutation(idx)]
        attrs[c] = col
    return AttributedGraph(n=real.n, edges=real.edges, attrs=attrs, domains={c: list(d) for c, d in real.domains.items()})


def gap_closed_table(m_null: dict, m_syn: dict) -> dict:
    """Gap closed per group and scalar metric, from two ``composition_metrics`` results against the same real graph.

    ``m_null``: real vs ``coupling.permuted_null``; ``m_syn``: real vs the synthetic graph. Level metrics
    (``simpson_mean``, ``two_hop_same_rate``) use the real graph's own value as the ceiling; the distance metric
    (``dominant_share_tvd``) uses 0. Returns ``{"groups": {label: {metric: gap}}, "mean": {metric: nanmean}}``.
    """
    out: dict = {}
    for label, gs in m_syn["groups"].items():
        ms, mn = gs["scalars"], m_null["groups"].get(label, {}).get("scalars", {})
        g = {}
        for k in LEVEL_METRICS:
            if f"{k}_syn" in ms and f"{k}_syn" in mn:
                g[k] = gap_closed(mn[f"{k}_syn"], ms[f"{k}_syn"], ms[f"{k}_real"], MIN_GAP_DENOMINATOR[k])
        for k in DISTANCE_METRICS:
            if k in ms and k in mn:
                g[k] = gap_closed(mn[k], ms[k], 0.0, MIN_GAP_DENOMINATOR[k])
        out[label] = g
    return {"groups": out, "mean": _nanmean_over(list(out.values()))}


def flatten(metrics: dict, prefix: str = "cmp_", gaps: dict | None = None, gap_name: str = "gap") -> dict:
    """Flat ``{column: float}`` for result CSVs: ``cmp_<group>_<metric>`` (group = attribute names joined by ``+``),
    ``cmp_mean_<metric>``, and ``cmp_<group>_gap_<metric>`` / ``cmp_mean_gap_<metric>`` when ``gaps`` (a
    ``gap_closed_table``) is given (``gap_name`` renames the ``gap`` token, e.g. for a second null)."""
    row = {}
    for label, g in metrics["groups"].items():
        row.update({f"{prefix}{label}_{k}": float(v) for k, v in g["scalars"].items()})
    row.update({f"{prefix}mean_{k}": float(v) for k, v in metrics["mean"].items()})
    if gaps is not None:
        for label, g in gaps["groups"].items():
            row.update({f"{prefix}{label}_{gap_name}_{k}": float(v) for k, v in g.items()})
        row.update({f"{prefix}mean_{gap_name}_{k}": float(v) for k, v in gaps["mean"].items()})
    return row

"""Attribute coupling metrics: do the right kinds of nodes connect, and which kinds have many connections?

Eval zone (needs the real graph and the synthetic graph side by side). Generic over the schema: every attribute column
shared by the two graphs is scored, nothing is named in code. Dataset knowledge arrives as arguments: ``missing`` (the
label that means "not observed", per attribute) and ``ordered`` (whether the domain order is a value order).
Definitions are in ``docs/snap/attribute-coupling-plan.md`` section 3.

Edge x attribute (homophily), over edges whose two ends are both observed:
``mix_tvd`` (mixing matrices), ``assort`` (Newman categorical assortativity), ``lift`` (same-value share over the
endpoint-weighted chance level), ``observed_pair_share``, and for ordered attributes ``mean_abs_bin_distance``.
Degree x attribute: ``deg_by_value_tvd``, ``eta2`` (share of the variance of ``log(1 + degree)`` explained by the
attribute) and a long-form ``mean_degree_by_value`` table. Triangle x attribute (package E):
``within_group_triangle_share``. With ``exclude_hubs > 0`` the degree metrics are also reported
without the ``exclude_hubs`` highest-degree nodes of each graph (keys with the suffix ``_nohub``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graph_aim.eval.evaluate import adjacency, degrees, triangles_per_node
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import AttributedGraph

# metrics compared as levels (the gap is measured from the null to the real value) and as distances (real value is 0)
LEVEL_METRICS = ("lift", "assort", "observed_pair_share", "eta2", "eta2_nohub")
DISTANCE_METRICS = ("mix_tvd", "deg_by_value_tvd", "deg_by_value_tvd_nohub")
# In ``gap_closed_table`` an attribute whose real value is this close to the null has no coupling to reproduce: its gap
# is NaN (a ratio of two noise-level numbers would swamp the mean over attributes).
MIN_GAP_DENOMINATOR = {
    "lift": 0.1,
    "assort": 0.05,
    "observed_pair_share": 0.02,
    "eta2": 0.005,
    "eta2_nohub": 0.005,
    "mix_tvd": 0.02,
    "deg_by_value_tvd": 0.02,
    "deg_by_value_tvd_nohub": 0.02,
}


# ----------------------------------------------------------------------------------------------- helpers
def _size(real: AttributedGraph, syn: AttributedGraph, col: str) -> int:
    """Domain size, as ``evaluate._domain_size`` but safe for an empty graph."""

    def top(g: AttributedGraph) -> int:
        return int(g.attrs[col].max()) + 1 if len(g.attrs) else 0

    return max(len(real.domains[col]), top(real), top(syn))


def _missing_code(real: AttributedGraph, col: str, missing: dict | None) -> int | None:
    """Code of the missing label of ``col`` (``None``: every value counts as observed)."""
    label = "none" if missing is None or col not in missing else missing[col]
    if label is None:
        return None
    dom = list(real.domains[col])
    return dom.index(label) if label in dom else None


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _edges(g: AttributedGraph) -> np.ndarray:
    return np.asarray(g.edges, dtype=np.int64).reshape(-1, 2)


def _mixing(g: AttributedGraph, col: str, size: int, miss: int | None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """``(E, ends, m_obs, m)``: Newman matrix of the both-observed edges (each adds 1 to ``E[a,b]`` and ``E[b,a]``),
    observed endpoint counts per value over ALL edges, number of both-observed edges, number of edges."""
    e = _edges(g)
    codes = g.attrs[col].to_numpy(dtype=np.int64)
    mat = np.zeros((size, size))
    if len(e) == 0:
        return mat, np.zeros(size), 0, 0
    a, b = codes[e[:, 0]], codes[e[:, 1]]
    ends = np.bincount(np.concatenate([a, b]), minlength=size).astype(np.float64)
    if miss is not None:
        ends[miss] = 0.0
        keep = (a != miss) & (b != miss)
        a, b = a[keep], b[keep]
    np.add.at(mat, (a, b), 1.0)
    np.add.at(mat, (b, a), 1.0)
    return mat, ends, len(a), len(e)


def _homophily_stats(mat: np.ndarray, ends: np.ndarray, ordered: bool) -> dict:
    """assort / lift (and ordered distances); NaN where undefined.

    ``assort`` is Newman's coefficient of the both-observed edges (their own marginals). ``lift`` is the same-value share
    of those edges over the chance level of pairing two observed endpoints drawn from ``ends`` (endpoint shares over all
    edges, renormalised to the observed values), the definition of plan section 1. The random-pairing expectation of the
    bin distance uses the same endpoint shares.
    """
    nan = float("nan")
    total, n_ends = mat.sum(), ends.sum()
    out = {"assort": nan, "lift": nan}
    if ordered:
        out.update({"mean_abs_bin_distance": nan, "mean_abs_bin_distance_random": nan})
    if total == 0 or n_ends == 0:
        return out
    e = mat / total
    am = e.sum(axis=1)
    a = ends / n_ends
    same, chance_m, chance = float(np.trace(e)), float(am @ am), float(a @ a)
    if 1.0 - chance_m > 1e-12:
        out["assort"] = (same - chance_m) / (1.0 - chance_m)
    out["lift"] = same / chance
    if ordered:
        dist = np.abs(np.subtract.outer(np.arange(len(a)), np.arange(len(a))))
        out["mean_abs_bin_distance"] = float((e * dist).sum())
        out["mean_abs_bin_distance_random"] = float(a @ dist @ a)
    return out


def _keep_mask(deg: np.ndarray, exclude_hubs: int) -> np.ndarray:
    keep = np.ones(len(deg), dtype=bool)
    if exclude_hubs > 0:
        keep[np.argsort(-deg, kind="stable")[:exclude_hubs]] = False
    return keep


def _eta2(codes: np.ndarray, deg: np.ndarray, size: int) -> float:
    """Correlation ratio of ``log(1 + degree)`` on the attribute value; NaN if the degrees do not vary."""
    if len(deg) == 0:
        return float("nan")
    y = np.log1p(deg.astype(np.float64))
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 1e-12:
        return float("nan")
    n = np.bincount(codes, minlength=size).astype(np.float64)
    s = np.bincount(codes, weights=y, minlength=size)
    ok = n > 0
    ssb = float((s[ok] ** 2 / n[ok]).sum() - len(y) * y.mean() ** 2)
    return max(ssb, 0.0) / sst


def _bin_table(codes: np.ndarray, bins: np.ndarray, size: int, n_bins: int) -> np.ndarray:
    return np.bincount(codes * n_bins + bins, minlength=size * n_bins).reshape(size, n_bins).astype(np.float64)


def _deg_by_value_tvd(r_codes, r_bins, s_codes, s_bins, size: int, n_bins: int) -> float:
    """Sum over values of (real share of the value) * TVD(real bin distribution, synthetic bin distribution).

    A value that exists in the real graph but has no node in the synthetic one counts as TVD 1.
    """
    rt, st = _bin_table(r_codes, r_bins, size, n_bins), _bin_table(s_codes, s_bins, size, n_bins)
    rn, sn = rt.sum(axis=1), st.sum(axis=1)
    total = rn.sum()
    if total == 0:
        return float("nan")
    out = 0.0
    for v in np.flatnonzero(rn > 0):
        out += rn[v] / total * (1.0 if sn[v] == 0 else _tvd(rt[v] / rn[v], st[v] / sn[v]))
    return float(out)


def _mean_degree_table(real, syn, col, size, dr, ds) -> list[dict]:
    rc, sc = real.attrs[col].to_numpy(dtype=np.int64), syn.attrs[col].to_numpy(dtype=np.int64)
    rn, sn = np.bincount(rc, minlength=size), np.bincount(sc, minlength=size)
    rs, ss = np.bincount(rc, weights=dr, minlength=size), np.bincount(sc, weights=ds, minlength=size)
    dom = list(real.domains[col])
    rows = []
    for v in range(size):
        rows.append(
            {
                "attribute": col,
                "value": dom[v] if v < len(dom) else f"code{v}",
                "n_real": int(rn[v]),
                "mean_degree_real": float(rs[v] / rn[v]) if rn[v] else float("nan"),
                "n_syn": int(sn[v]),
                "mean_degree_syn": float(ss[v] / sn[v]) if sn[v] else float("nan"),
            }
        )
    return rows


# ------------------------------------------------------------------------------------------------- metrics
def coupling_metrics(
    real: AttributedGraph,
    syn: AttributedGraph,
    *,
    degree_bin_edges,
    missing: dict | None = None,
    ordered: dict | None = None,
    exclude_hubs: int = 0,
) -> dict:
    """Coupling of attributes with edges and with degree, real vs synthetic, for every shared attribute column.

    Returns ``{"attributes": {col: {metric: value}}, "mean": {metric: nanmean over attributes},
    "mean_degree_by_value": DataFrame, "exclude_hubs": int}``. ``missing[col]`` is the label meaning "not observed"
    (default ``"none"`` when the domain has it; ``None`` means every value is observed), ``ordered[col]`` says that the
    domain order is the value order (default False). Columns are matched by name; domains of the synthetic graph may be
    larger than the real ones (the real labels are used where they exist).
    """
    cols = [c for c in real.attrs.columns if c in syn.attrs.columns]
    ordered = ordered or {}
    dr, ds = degrees(real), degrees(syn)
    n_bins = len(np.asarray(degree_bin_edges)) - 1
    rb, sb = degree_bin(dr, degree_bin_edges), degree_bin(ds, degree_bin_edges)
    kr, ks = _keep_mask(dr, exclude_hubs), _keep_mask(ds, exclude_hubs)
    attributes, tables = {}, []
    for col in cols:
        size = _size(real, syn, col)
        miss = _missing_code(real, col, missing)
        is_ordered = bool(ordered.get(col, False))
        rmat, rends, r_obs, r_m = _mixing(real, col, size, miss)
        smat, sends, s_obs, s_m = _mixing(syn, col, size, miss)
        rh, sh = _homophily_stats(rmat, rends, is_ordered), _homophily_stats(smat, sends, is_ordered)
        m: dict = {}
        if r_obs and s_obs:
            # homophily_matrix convention: diagonal counted once
            rd, sd = rmat - np.diag(np.diag(rmat)) / 2, smat - np.diag(np.diag(smat)) / 2
            m["mix_tvd"] = _tvd((rd / rd.sum()).ravel(), (sd / sd.sum()).ravel())
        else:
            m["mix_tvd"] = float("nan")
        for k in ("assort", "lift"):
            m[f"{k}_real"], m[f"{k}_syn"] = rh[k], sh[k]
            m[f"{k}_absdiff"] = abs(rh[k] - sh[k])
        m["observed_pair_share_real"] = r_obs / r_m if r_m else float("nan")
        m["observed_pair_share_syn"] = s_obs / s_m if s_m else float("nan")
        if is_ordered:
            for k in ("mean_abs_bin_distance", "mean_abs_bin_distance_random"):
                m[f"{k}_real"], m[f"{k}_syn"] = rh[k], sh[k]

        rc, sc = real.attrs[col].to_numpy(dtype=np.int64), syn.attrs[col].to_numpy(dtype=np.int64)
        m["deg_by_value_tvd"] = _deg_by_value_tvd(rc, rb, sc, sb, size, n_bins)
        m["eta2_real"], m["eta2_syn"] = _eta2(rc, dr, size), _eta2(sc, ds, size)
        m["eta2_absdiff"] = abs(m["eta2_real"] - m["eta2_syn"])
        if exclude_hubs > 0:
            m["deg_by_value_tvd_nohub"] = _deg_by_value_tvd(rc[kr], rb[kr], sc[ks], sb[ks], size, n_bins)
            m["eta2_nohub_real"], m["eta2_nohub_syn"] = _eta2(rc[kr], dr[kr], size), _eta2(sc[ks], ds[ks], size)
            m["eta2_nohub_absdiff"] = abs(m["eta2_nohub_real"] - m["eta2_nohub_syn"])
        attributes[col] = m
        tables.extend(_mean_degree_table(real, syn, col, size, dr, ds))

    keys = sorted({k for m in attributes.values() for k in m})
    mean = {}
    for k in keys:
        vals = np.array([m[k] for m in attributes.values() if k in m], dtype=float)
        mean[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {
        "attributes": attributes,
        "mean": mean,
        "mean_degree_by_value": pd.DataFrame(
            tables,
            columns=["attribute", "value", "n_real", "mean_degree_real", "n_syn", "mean_degree_syn"],
        ),
        "exclude_hubs": int(exclude_hubs),
    }


# --------------------------------------------------------------------------------- triangle homophily
def within_group_triangle_share(
    graph: AttributedGraph, attrs: str | list[str], *, missing: dict | None = None
) -> float:
    """Share of the graph's triangles whose three corners carry the same observed value of ``attrs``.

    ``attrs`` is one attribute column or a list of columns whose row-major product is the group (the
    ``attr_edge_pair`` convention, plan section 7). The denominator is every triangle of the graph; a
    corner whose value is the missing label of any column (``missing`` as in ``coupling_metrics``:
    per-column label, default ``"none"`` where the domain has it) never counts as "same", so a triangle
    of three unobserved nodes is not a within-group triangle. NaN when the graph has no triangles.
    Deterministic; sparse per-value subgraph triangle counts (a few ms at 4k nodes / 90k edges).
    """
    cols = [attrs] if isinstance(attrs, str) else list(attrs)
    adj = adjacency(graph)
    total = float(triangles_per_node(adj).sum() / 3.0)
    if total <= 0:
        return float("nan")
    grp = np.zeros(graph.n, dtype=np.int64)
    observed = np.ones(graph.n, dtype=bool)
    for c in cols:
        codes = graph.attrs[c].to_numpy(dtype=np.int64)
        size = max(len(graph.domains[c]), int(codes.max()) + 1 if len(codes) else 0)
        grp = grp * size + codes
        miss = _missing_code(graph, c, missing)
        if miss is not None:
            observed &= codes != miss
    within = 0.0
    for v in np.unique(grp[observed]):
        idx = np.flatnonzero(observed & (grp == v))
        if len(idx) < 3:
            continue
        sub = adj[idx][:, idx]
        within += float(triangles_per_node(sub).sum() / 3.0)
    return within / total


# ----------------------------------------------------------------------------------------- reference rows
def permuted_null(real: AttributedGraph, seed: int = 0) -> AttributedGraph:
    """The real graph with every attribute column independently permuted over the nodes (attribute-blind floor)."""
    rng = np.random.default_rng(seed)
    attrs = real.attrs.copy()
    for c in attrs.columns:
        attrs[c] = rng.permutation(attrs[c].to_numpy())
    return AttributedGraph(n=real.n, edges=real.edges, attrs=attrs, domains={c: list(d) for c, d in real.domains.items()})


def reference_metrics(real: AttributedGraph, *, seed: int = 0, **kwargs) -> dict:
    """``{"null": metrics(real vs permuted null), "real": metrics(real vs itself)}``; ``kwargs`` as ``coupling_metrics``."""
    return {
        "null": coupling_metrics(real, permuted_null(real, seed), **kwargs),
        "real": coupling_metrics(real, real, **kwargs),
    }


def gap_closed(m_null: float, m_syn: float, m_real: float, min_denominator: float = 1e-9) -> float:
    """``(m_null - m_syn) / (m_null - m_real)``: 0 is as bad as the null, 1 is as good as the real graph.

    NaN if any input is NaN or the null and the real graph do not differ (nothing to close).
    """
    den = m_null - m_real
    if not np.isfinite([m_null, m_syn, m_real]).all() or abs(den) < min_denominator:
        return float("nan")
    return float((m_null - m_syn) / den)


def gap_closed_table(m_null: dict, m_syn: dict) -> dict:
    """Gap closed per attribute and metric, from two ``coupling_metrics`` results taken against the same real graph.

    ``m_null``: real vs ``permuted_null``; ``m_syn``: real vs the synthetic graph. Level metrics (lift, assort,
    observed_pair_share, eta2) use the real graph's own value as the ceiling; distance metrics (mix_tvd,
    deg_by_value_tvd) use 0. Returns ``{"attributes": {col: {metric: gap}}, "mean": {metric: nanmean}}``.
    """
    out: dict = {}
    for col, ms in m_syn["attributes"].items():
        mn = m_null["attributes"].get(col, {})
        g = {}
        for k in LEVEL_METRICS:
            if f"{k}_syn" in ms and f"{k}_syn" in mn:
                g[k] = gap_closed(mn[f"{k}_syn"], ms[f"{k}_syn"], ms[f"{k}_real"], MIN_GAP_DENOMINATOR[k])
        for k in DISTANCE_METRICS:
            if k in ms and k in mn:
                g[k] = gap_closed(mn[k], ms[k], 0.0, MIN_GAP_DENOMINATOR[k])
        out[col] = g
    keys = sorted({k for g in out.values() for k in g})
    mean = {}
    for k in keys:
        vals = np.array([g[k] for g in out.values() if k in g], dtype=float)
        mean[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"attributes": out, "mean": mean}


def flatten(metrics: dict, prefix: str = "cpl_", gaps: dict | None = None) -> dict:
    """Flat ``{column: float}`` for result CSVs: ``cpl_<attr>_<metric>``, ``cpl_mean_<metric>``, and the same with
    ``gap_`` in front of the metric name when ``gaps`` (a ``gap_closed_table``) is given."""
    row = {}
    for col, m in metrics["attributes"].items():
        row.update({f"{prefix}{col}_{k}": float(v) for k, v in m.items()})
    row.update({f"{prefix}mean_{k}": float(v) for k, v in metrics["mean"].items()})
    if gaps is not None:
        for col, g in gaps["attributes"].items():
            row.update({f"{prefix}{col}_gap_{k}": float(v) for k, v in g.items()})
        row.update({f"{prefix}mean_gap_{k}": float(v) for k, v in gaps["mean"].items()})
    return row


def options_from_policy(policy: dict) -> tuple[dict, dict]:
    """``(missing, ordered)`` dicts from an attribute policy; the per-attribute keys ``missing_label`` and ``ordered``
    are optional (defaults: ``"none"``, ``False``)."""
    attrs = policy.get("attributes", [])
    missing = {a["name"]: a.get("missing_label", "none") for a in attrs}
    ordered = {a["name"]: bool(a.get("ordered", False)) for a in attrs}
    return missing, ordered

"""Energies of graph-aim-walks (outside zone; ``docs/snap/walks-plan.md`` section 4.3).

Walk tables
-----------
Per table, ``E = sum_cells (m_syn - m_real)^2 / var`` with ``m`` the normalised table (``walkstats.normalise``) and
``var`` the sampling variance of the DIFFERENCE of two independent pools of ``n_syn`` and ``n_real`` walks, so the
energy is in noise units and a matched table has energy about its number of informative cells:

- proportions (``return``, ``degree_lag``, ``attr_lag``, ``share``, ``dominance``): the pooled two-sample binomial
  ``p (1 - p) (1 / n_syn + 1 / n_real)``, ``p = (n_syn m_syn + n_real m_real) / (n_syn + n_real)``. ``share`` and
  ``dominance`` put every walk in exactly one cell per (window, value) / window and ``return`` is one Bernoulli per
  walk and step, so this is their exact multinomial variance. A ``*_lag`` cell is the mean over walks of the
  walk's fraction of ``(t, t + k)`` pairs in the cell, whose variance is at most ``m (1 - m) / n`` (a fraction lies
  in [0, 1]); the binomial with ``n`` = number of WALKS is that bound, i.e. a conservative noise unit.
- ``coverage`` (a mean of distinct-node counts): ``s^2 (1 / n_syn + 1 / n_real)`` with ``s^2`` the per-walk variance
  estimated from the synthetic pool (the artifact releases only the mean).

Cells with zero variance (both tables 0, or both 1) carry no information: they add nothing and are not counted.
``n_syn`` is the NOMINAL synthetic pool size of the generator, whatever pool a table was measured on, so energies
measured on pools of different sizes are on one scale. For the chains the variances are frozen and the energy is a
quadratic in the pool's summed counts (the normalisers are constants: ``P``, ``P (T + 1 - k)``), whose gradient
:func:`gradients` gives (used by ``pool.walk_scores`` for the per-proposal first-order screen).

Base terms (exact base artifact)
--------------------------------
``(C - T)^2 / (sigma^2 + T + 1)`` per cell, the convention of ``graph_aim.outside.refine``:
``clustering_by_degree`` (per-bin sums of local clustering, only bins whose lower edge is >= ``min_degree``: the
lower bins' column 0 is imputed by the fit's post-processing), ``joint_degree`` (ordered degree-bin pair counts),
``attr_edge_pair`` (ordered class-pair counts on the graph capped as released) and ``attr_2way``. Uncapped
``clustering_by_degree`` / ``joint_degree`` only (the shipped config); a capped instance is skipped with a warning.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from graph_aim.outside.generate import _by_kind, _clean, _ordered_mass
from graph_aim.transfer.binning import check_bin_edges, degree_bin
from graph_aim.transfer.capping import cap_edges
from graph_aim_walks.transfer import walkstats as ws

_EPS = 1e-12


# ------------------------------------------------------------------------------------------- walk tables
def normaliser(name: str, counts: np.ndarray, n_walks: int) -> np.ndarray:
    """The (broadcastable) denominator that turns summed counts into the released table (``walkstats.normalise``)."""
    v = np.asarray(counts, dtype=float)
    if name in ("return", "coverage"):
        return np.full((1,) * v.ndim, float(max(n_walks, 1)))
    if name == "degree_lag" or name.startswith("attr_lag:") or name.startswith("dominance:"):
        return np.maximum(v.sum(axis=(1, 2), keepdims=True), 1.0)
    if name.startswith("share:"):
        return np.maximum(v.sum(axis=2, keepdims=True), 1.0)
    raise ValueError(f"unknown statistic {name!r}")


def coverage_variance(counts: dict, n_walks: int) -> np.ndarray:
    """Per-walk variance of the distinct-node count after ``t`` steps, from a pool's ``coverage`` sums."""
    P = float(max(n_walks, 1))
    mean = counts["coverage"] / P
    return np.maximum(counts["coverage_sq"] / P - mean**2, 0.0)


def cell_variance(name: str, m_syn: np.ndarray, m_real: np.ndarray, n_syn: float, n_real: float,
                  cov_var: np.ndarray | None = None) -> np.ndarray:
    inv = 1.0 / n_syn + 1.0 / n_real
    if name == "coverage":
        if cov_var is None:
            raise ValueError("coverage needs the per-walk variance")
        return np.asarray(cov_var, dtype=float) * inv
    p = (n_syn * m_syn + n_real * m_real) / (n_syn + n_real)
    return np.clip(p * (1.0 - p), 0.0, None) * inv


def table_energy(name: str, m_syn, m_real, n_syn: float, n_real: float, cov_var=None) -> tuple[float, int]:
    """``(energy, informative cells)`` of one table."""
    m_syn, m_real = np.asarray(m_syn, dtype=float), np.asarray(m_real, dtype=float)
    var = cell_variance(name, m_syn, m_real, n_syn, n_real, cov_var)
    ok = var > _EPS
    e = np.where(ok, (m_syn - m_real) ** 2 / np.where(ok, var, 1.0), 0.0)
    return float(e.sum()), int(ok.sum())


def walk_energy(counts: dict, n_pool: int, real: dict, n_nominal: int, n_real: int,
                weights: dict | None = None, names=None) -> dict[str, dict]:
    """``{table: {"energy", "cells"}}`` of a pool's summed ``counts`` (measured on ``n_pool`` walks) against the
    real tables; variances use the nominal synthetic pool size ``n_nominal``. ``weights`` scale the energies."""
    weights = weights or {}
    syn = ws.normalise({k: v for k, v in counts.items() if k != "coverage_sq"}, n_pool)
    cov_var = coverage_variance(counts, n_pool) if "coverage_sq" in counts else None
    out = {}
    for k in (names or syn):
        if k not in real:
            continue
        e, c = table_energy(k, syn[k], real[k], n_nominal, n_real, cov_var)
        out[k] = {"energy": float(weights.get(k, 1.0)) * e, "cells": c}
    return out


def gradients(counts: dict, n_pool: int, real: dict, n_nominal: int, n_real: int,
              weights: dict | None = None, per: str = "counts") -> dict[str, np.ndarray]:
    """``dE / d counts`` per table at frozen variances (the energy is then quadratic in the counts); with
    ``per="table"`` the gradient in the normalised table ``dE / dm`` instead."""
    weights = weights or {}
    syn = ws.normalise({k: v for k, v in counts.items() if k != "coverage_sq"}, n_pool)
    cov_var = coverage_variance(counts, n_pool) if "coverage_sq" in counts else None
    out = {}
    for k, m in syn.items():
        if k not in real:
            continue
        var = cell_variance(k, m, real[k], n_nominal, n_real, cov_var)
        ok = var > _EPS
        z = normaliser(k, counts[k], n_pool) if per == "counts" else 1.0
        g = np.where(ok, 2.0 * (m - real[k]) / np.where(ok, var, 1.0), 0.0) / z
        out[k] = float(weights.get(k, 1.0)) * g
    return out


# ------------------------------------------------------------------------------------------- helpers
def quad_delta(prop: np.ndarray, cell: np.ndarray, val: np.ndarray, D: np.ndarray, w: np.ndarray, K: int) -> np.ndarray:
    """Exact per-proposal change of ``sum_c w_c (C_c + R_c - T_c)^2`` with ``D = C - T`` and the change ``R`` of
    proposal ``i`` given as entries ``(prop, cell, val)`` (duplicate cells add up)."""
    if len(prop) == 0:
        return np.zeros(K)
    ncell = D.size
    key = prop.astype(np.int64) * ncell + cell.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    R = np.bincount(inv, weights=val, minlength=len(uk))
    c = uk % ncell
    e = w.ravel()[c] * (2.0 * D.ravel()[c] * R + R * R)
    return np.bincount(uk // ncell, weights=e, minlength=K)


def adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    a = sp.coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    a.sort_indices()
    return a


def common_neighbours(adj: sp.csr_matrix, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pair index, w)`` for every common neighbour ``w`` of the pairs ``(x[i], y[i])`` (sparse row product)."""
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    m = adj[x].multiply(adj[y]).tocoo()
    return m.row.astype(np.int64), m.col.astype(np.int64)


def triangles(adj: sp.csr_matrix) -> np.ndarray:
    """Triangles through every node (from scratch)."""
    a = adj.astype(np.int64)
    return np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() // 2


def triangle_change(adj_old: sp.csr_matrix, adj_new: sp.csr_matrix, removed: np.ndarray, added: np.ndarray,
                    n: int) -> np.ndarray:
    """Exact per-node change of the triangle counts when ``removed`` edges go and ``added`` edges come: the
    triangles of the new graph through an added edge minus those of the old graph through a removed edge (each
    triangle counted once, however many of its edges changed)."""

    def through(adj, e):
        if len(e) == 0:
            return np.zeros(n)
        i, w = common_neighbours(adj, e[:, 0], e[:, 1])
        tri = np.sort(np.stack([e[i, 0], e[i, 1], w], axis=1), axis=1)
        tri = np.unique(tri, axis=0)
        return np.bincount(tri.ravel(), minlength=n).astype(float)

    return through(adj_new, np.asarray(added).reshape(-1, 2)) - through(adj_old, np.asarray(removed).reshape(-1, 2))


# ------------------------------------------------------------------------------------------- base terms
class ClusteringTerm:
    """``clustering_by_degree``: per-bin sums of (clipped) local clustering over nodes of degree >= min_degree."""

    kind = "clustering_by_degree"

    def __init__(self, m, degree: np.ndarray, weight: float = 1.0):
        self.name = m.name
        p = m.params
        self.edges = check_bin_edges(p["bin_edges"])
        self.dmin = int(p.get("min_degree") or 0)
        self.clip = float(p.get("clip", 1.0))
        self.nb = len(self.edges) - 1
        d = np.asarray(degree, dtype=float)
        self.bin = degree_bin(np.asarray(degree), self.edges)
        self.pairs = d * (d - 1) / 2.0
        self.inc = (d >= max(self.dmin, 2)) & (self.pairs > 0)
        self.use = self.edges[:-1] >= self.dmin
        v = np.nan_to_num(np.asarray(m.values, dtype=float))
        self.T = np.where(self.use, v[:, 0], 0.0)
        self.w = np.where(self.use, weight / (float(m.sigma or 0.0) ** 2 + np.abs(self.T) + 1.0), 0.0)
        self.cells = int(self.use.sum())
        self.S = np.zeros(self.nb)
        self.energy = 0.0

    def f(self, t: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        c = np.where(self.inc[nodes], t / np.where(self.inc[nodes], self.pairs[nodes], 1.0), 0.0)
        return np.clip(c, 0.0, self.clip)

    def exact(self, tri: np.ndarray) -> float:
        nodes = np.arange(len(tri))
        self.S = np.bincount(self.bin, weights=self.f(tri.astype(float), nodes), minlength=self.nb)
        self.energy = float((self.w * (self.S - self.T) ** 2).sum())
        return self.energy

    def delta(self, prop: np.ndarray, node: np.ndarray, dt: np.ndarray, tri: np.ndarray, K: int) -> np.ndarray:
        """Exact per-proposal energy change from per-(proposal, node) triangle changes (entries may repeat)."""
        if len(prop) == 0:
            return np.zeros(K)
        n = len(tri)
        key = prop * n + node
        uk, inv = np.unique(key, return_inverse=True)
        d = np.bincount(inv, weights=dt, minlength=len(uk))
        nd = uk % n
        df = self.f(tri[nd] + d, nd) - self.f(tri[nd].astype(float), nd)
        return quad_delta(uk // n, self.bin[nd], df, self.S - self.T, self.w, K)


class PairCountTerm:
    """Ordered pair counts of node codes over the edges (``joint_degree``, or ``attr_edge_pair`` measured on
    the uncapped graph against a target shifted by the capped-vs-uncapped offset, in the structure chain)."""

    def __init__(self, name: str, kind: str, codes: np.ndarray, size: int, target: np.ndarray, sigma: float,
                 weight: float = 1.0):
        self.name, self.kind, self.codes, self.size = name, kind, np.asarray(codes, dtype=np.int64), int(size)
        self.T = np.asarray(target, dtype=float)
        self.w = weight / (sigma**2 + np.abs(self.T) + 1.0)
        self.cells = self.size * self.size
        self.C = np.zeros_like(self.T)
        self.energy = 0.0

    def counts(self, edges: np.ndarray) -> np.ndarray:
        G = self.size
        c = np.bincount(self.codes[edges[:, 0]] * G + self.codes[edges[:, 1]], minlength=G * G).reshape(G, G)
        return (c + c.T).astype(float)

    def exact(self, edges: np.ndarray) -> float:
        self.C = self.counts(edges)
        self.energy = float((self.w * (self.C - self.T) ** 2).sum())
        return self.energy

    def swap_delta(self, a, b, c, d) -> np.ndarray:
        """Exact per-proposal change of the swaps ``(a, b), (c, d) -> (a, d), (c, b)``."""
        K, G = len(a), self.size
        q = self.codes
        qa, qb, qc, qd = q[a], q[b], q[c], q[d]
        cells = np.concatenate([qa * G + qb, qb * G + qa, qc * G + qd, qd * G + qc,
                                qa * G + qd, qd * G + qa, qc * G + qb, qb * G + qc])
        val = np.repeat([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], K)
        prop = np.tile(np.arange(K), 8)
        return quad_delta(prop, cells, val, self.C - self.T, self.w, K)

    @property
    def satisfied(self) -> bool:
        return self.energy <= self.cells + 1e-9


class CappedEdgePairTerm:
    """``attr_edge_pair`` on the synthetic graph capped as released (label chain): ordered class-pair counts."""

    kind = "attr_edge_pair"

    def __init__(self, m, cols: list[str], domains: dict, x: dict, edges: np.ndarray, n: int, salt: int,
                 weight: float = 1.0):
        self.name, self.cols = m.name, cols
        self.sizes = [len(domains[c]) for c in cols]
        self.size = int(np.prod(self.sizes))
        self.strides = {}
        for i, c in enumerate(cols):
            self.strides[c] = self.strides.get(c, 0) + int(np.prod(self.sizes[i + 1:], dtype=np.int64))
        self.cap = m.params.get("cap") if isinstance(m.params, dict) else None
        self.n, self.salt = n, salt
        self.q = _ordered_mass(np.asarray(m.values, dtype=float))
        self.sigma = float(m.sigma or 0.0)
        self.weight = weight
        self.cells = self.size * self.size
        self.set_edges(edges)
        self.g = self.codes(x)
        self.energy = 0.0

    def codes(self, x: dict) -> np.ndarray:
        g = np.zeros(self.n, dtype=np.int64)
        for c, s in zip(self.cols, self.sizes):
            g = g * s + np.asarray(x[c], dtype=np.int64)
        return g

    def set_edges(self, edges: np.ndarray) -> None:
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if self.cap is not None:
            e = cap_edges(e, self.n, int(self.cap), self.salt)
        self.e = e
        self.adj = adjacency(e, self.n)
        self.T = self.q * (2.0 * len(e) / max(self.q.sum(), _EPS))
        self.w = self.weight / (self.sigma**2 + self.T + 1.0)

    def counts_for(self, g: np.ndarray) -> np.ndarray:
        G = self.size
        c = np.bincount(g[self.e[:, 0]] * G + g[self.e[:, 1]], minlength=G * G).reshape(G, G)
        return (c + c.T).astype(float)

    def exact(self) -> float:
        self.C = self.counts_for(self.g)
        self.energy = float((self.w * (self.C - self.T) ** 2).sum())
        return self.energy

    def swap_delta(self, attr: str, u: np.ndarray, v: np.ndarray, xu: np.ndarray, xv: np.ndarray) -> np.ndarray:
        """Exact per-proposal change when ``u`` and ``v`` (never adjacent) swap their value of ``attr``."""
        K, G = len(u), self.size
        stride = self.strides.get(attr, 0)
        if stride == 0:
            return np.zeros(K)
        gu, gv = self.g[u], self.g[v]
        gnu, gnv = gu + (xv - xu) * stride, gv + (xu - xv) * stride
        props, cells, vals = [], [], []
        for node, go, gn in ((u, gu, gnu), (v, gv, gnv)):
            lens = np.diff(self.adj.indptr)[node]
            pid = np.repeat(np.arange(K), lens)
            starts = np.repeat(self.adj.indptr[node], lens)
            offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
            gy = self.g[self.adj.indices[starts + offs]]
            for gg, s in ((go, -1.0), (gn, 1.0)):
                props += [pid, pid]
                cells += [gg[pid] * G + gy, gy * G + gg[pid]]
                vals += [np.full(len(pid), s), np.full(len(pid), s)]
        return quad_delta(np.concatenate(props), np.concatenate(cells), np.concatenate(vals),
                          self.C - self.T, self.w, K)


class TwoWayTerm:
    kind = "attr_2way"

    def __init__(self, m, a: str, b: str, ka: int, kb: int, n: int, weight: float = 1.0):
        self.name, self.a, self.b, self.ka, self.kb = m.name, a, b, ka, kb
        t = _clean(np.asarray(m.values, dtype=float)).ravel()
        self.T = t * (n / t.sum()) if t.sum() > 0 else t
        self.w = weight / (float(m.sigma or 0.0) ** 2 + self.T + 1.0)
        self.cells = ka * kb
        self.energy = 0.0

    def exact(self, x: dict) -> float:
        self.C = np.bincount(x[self.a] * self.kb + x[self.b], minlength=self.cells).astype(float)
        self.energy = float((self.w * (self.C - self.T) ** 2).sum())
        return self.energy

    def swap_delta(self, attr: str, x: dict, u, v) -> np.ndarray:
        K, kb = len(u), self.kb
        if attr == self.a and attr != self.b:
            o = x[self.b]
            cu, cv = x[self.a][u] * kb + o[u], x[self.a][v] * kb + o[v]
            nu, nv = x[self.a][v] * kb + o[u], x[self.a][u] * kb + o[v]
        elif attr == self.b and attr != self.a:
            o = x[self.a]
            cu, cv = o[u] * kb + x[self.b][u], o[v] * kb + x[self.b][v]
            nu, nv = o[u] * kb + x[self.b][v], o[v] * kb + x[self.b][u]
        else:
            return np.zeros(K)
        cells = np.concatenate([cu, cv, nu, nv])
        val = np.repeat([-1.0, -1.0, 1.0, 1.0], K)
        return quad_delta(np.tile(np.arange(K), 4), cells, val, self.C - self.T, self.w, K)


def structure_terms(base, degree: np.ndarray, warnings: list) -> tuple[ClusteringTerm | None, PairCountTerm | None]:
    """The ``clustering_by_degree`` and ``joint_degree`` terms of the base artifact (``None`` when absent)."""
    clu = jd = None
    ms = _by_kind(base, "clustering_by_degree")
    if ms:
        if ms[0].params.get("cap") is not None:
            warnings.append(f"{ms[0].name}: capped clustering is not tracked, skipped")
        else:
            clu = ClusteringTerm(ms[0], degree)
    ms = _by_kind(base, "joint_degree")
    if ms:
        m = ms[0]
        if m.params.get("cap") is not None:
            warnings.append(f"{m.name}: capped joint_degree is not tracked, skipped")
        else:
            e = check_bin_edges(m.params["bin_edges"])
            jd = PairCountTerm(m.name, "joint_degree", degree_bin(np.asarray(degree), e), len(e) - 1,
                               _ordered_mass(np.asarray(m.values, dtype=float)), float(m.sigma or 0.0))
    return clu, jd


def label_terms(base, x: dict, domains: dict, edges: np.ndarray, n: int, warnings: list):
    """``attr_edge_pair`` (capped as released) and ``attr_2way`` terms of the base artifact."""
    salt = int(base.schema.cap_salt)
    pairs = []
    for m in _by_kind(base, "attr_edge_pair"):
        cols = m.params.get("attrs") if isinstance(m.params, dict) else None
        if not cols or any(c not in domains or c not in x for c in cols):
            warnings.append(f"{m.name}: attrs unusable, skipped")
            continue
        size = int(np.prod([len(domains[c]) for c in cols]))
        if np.asarray(m.values).shape != (size, size):
            warnings.append(f"{m.name}: shape does not match the domains, skipped")
            continue
        pairs.append(CappedEdgePairTerm(m, list(cols), domains, x, edges, n, salt))
    twoway = []
    for m in _by_kind(base, "attr_2way"):
        cols = m.params.get("attrs") if isinstance(m.params, dict) else None
        if not cols or len(cols) != 2 or any(c not in domains or c not in x for c in cols):
            continue
        a, b = cols
        if np.asarray(m.values).shape != (len(domains[a]), len(domains[b])):
            warnings.append(f"{m.name}: shape does not match the domains, skipped")
            continue
        twoway.append(TwoWayTerm(m, a, b, len(domains[a]), len(domains[b]), n))
    return pairs, twoway

"""Label chain: refine the attribute labels of a synthetic graph against released neighbourhood tables.

The edge set is fixed. A move swaps the value of ONE attribute between two nodes of the same ``attr_degree``
stratum (degree bin), so the attribute marginals (``attr_1way``), the attribute x degree-bin tables
(``attr_degree``) and the edges are preserved exactly. The chain descends an energy that is the sum, over every
released table it can score, of the noise-normalised squared distance between the synthetic statistic and the
released target (``docs/snap/composition-plan.md`` section 5):

- ``attr_edge_pair`` instance: ordered class-pair edge counts ``C`` against the released ordered mass scaled to
  the synthetic ordered edge count ``T``: ``sum_ij (C_ij - T_ij)^2 / (sigma^2 + T_ij + 1)``.
- ``attr_neigh_simpson`` instance: per degree bin ``b`` (lower edge ``>= min_degree``), the sum ``S_b`` of the
  neighbourhood Simpson index ``s_u = sum_g (N[u, g] / d_u)^2`` over synthetic nodes of the bin against the
  released per-bin mean times the synthetic bin count: ``sum_b (S_b - S*_b)^2 / (sigma^2 + 1)``.
- ``attr_2way`` table touching a moving attribute: ``sum (C - T)^2 / (sigma^2 + T + 1)`` with ``T`` scaled to ``n``.

One round proposes a batch of disjoint pairs (a random perfect matching inside each stratum, pairs with equal
values or an edge between them dropped), scores every proposal against the current state with vectorised change
statistics (exact per proposal for the edge-pair and 2-way terms; the closed form
``delta s_w = 2 (N[w, g'] - N[w, g] + 1) / d_w^2`` summed over the neighbours ``w`` of each endpoint for the
Simpson term), accepts the proposals with a negative change, damps the batch to a random subset of at most a
fraction of the proposals, and keeps the prefix of that (randomly ordered) subset whose combined change is
lowest: moves that are good one by one can push the same table cells together, and their interaction is
quadratic in the batch, so the combined change of every prefix is evaluated (exactly for the edge-pair and 2-way
terms from cumulative change matrices, linearised for Simpson) before anything is applied. The chosen moves are
applied, the neighbour-code counts updated and every term recomputed exactly; if the exact energy still rose
(Simpson nonlinearity on shared neighbours) the damping is halved for the next round and restored slowly. The
chain stops when every term is within its own noise (energy ``<=`` number of cells), when no proposal of any
moving attribute lowers the energy, or after ``max_rounds``.

Everything here is outside-zone code: it sees the artifact and the synthetic graph only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graph_aim.outside.generate import _attr_degree_tables, _by_kind, _clean, _ordered_mass
from graph_aim.transfer.binning import check_bin_edges, degree_bin
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement

SIMPSON_KIND = "attr_neigh_simpson"
EDGE_PAIR_KIND = "attr_edge_pair"
TWO_WAY_KIND = "attr_2way"
MAX_DENSE_GROUPS = 4096  # guards the dense (n, G) neighbour-count arrays
_EPS = 1e-9


# ----------------------------------------------------------------------------------- pure helpers
def _neighbour_code_counts(indptr: np.ndarray, indices: np.ndarray, codes: np.ndarray, n: int, n_groups: int) -> np.ndarray:
    """Dense ``(n, G)`` counts ``N[u, g]`` of the neighbours of ``u`` with code ``g`` from a CSR structure."""
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    flat = rows * n_groups + codes[indices]
    return np.bincount(flat, minlength=n * n_groups).reshape(n, n_groups).astype(np.int64)


def _simpson_per_node(counts: np.ndarray, degree: np.ndarray, min_degree: int) -> np.ndarray:
    """``s_u = [d_u >= dmin] sum_g (N[u, g] / d_u)^2`` (0 when the degree is below ``min_degree`` or zero)."""
    d = np.asarray(degree, dtype=float)
    ok = (d >= max(int(min_degree), 1)) & (d > 0)
    sq = (counts.astype(float) ** 2).sum(axis=1)
    return np.where(ok, sq / np.where(ok, d, 1.0) ** 2, 0.0)


def neigh_simpson_sums(adjacency_csr, codes, n_groups: int, degree, bin_edges, min_degree: int) -> np.ndarray:
    """Per-degree-bin sums of the neighbourhood Simpson index (plan section 4.1), ``(B,)``.

    ``adjacency_csr`` is a symmetric 0/1 ``(n, n)`` sparse matrix, ``codes`` the ``(n,)`` group code of every
    node, ``degree`` the ``(n,)`` degree used both for the bin and for the normalisation ``d_u`` (the row sums
    of the adjacency in the usual case; pass the capped degrees with a capped adjacency). Node ``u`` contributes
    ``s_u = sum_g (n_{u,g} / d_u)^2`` to bin ``degree_bin(d_u, bin_edges)`` if ``d_u >= min_degree``, else 0.
    """
    a = sp.csr_matrix(adjacency_csr)
    a.sum_duplicates()
    n = a.shape[0]
    codes = np.asarray(codes, dtype=np.int64)
    deg = np.asarray(degree)
    counts = _neighbour_code_counts(a.indptr, a.indices, codes, n, int(n_groups))
    s = _simpson_per_node(counts, deg, int(min_degree))
    b = degree_bin(deg, bin_edges)
    return np.bincount(b, weights=s, minlength=len(bin_edges) - 1)


def _codes_of(x: dict[str, np.ndarray], domains: dict[str, list[str]], cols: list[str]) -> tuple[np.ndarray, int, dict[str, int]]:
    """Row-major group code over ``cols``, the group count and the stride (multiplier) of every column."""
    sizes = [len(domains[c]) for c in cols]
    n = len(next(iter(x.values())))
    g = np.zeros(n, dtype=np.int64)
    for c, s in zip(cols, sizes):
        g = g * s + x[c]
    strides: dict[str, int] = {}
    for i, c in enumerate(cols):
        strides[c] = strides.get(c, 0) + int(np.prod(sizes[i + 1 :], dtype=np.int64))  # empty product = 1
    return g, int(np.prod(sizes, dtype=np.int64)), strides


def _neighbours(indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated neighbour lists of ``nodes`` and, for each entry, the position of its node in ``nodes``."""
    lens = indptr[nodes + 1] - indptr[nodes]
    total = int(lens.sum())
    starts = np.repeat(indptr[nodes], lens)
    offs = np.repeat(np.cumsum(lens) - lens, lens)
    nbr = indices[starts + np.arange(total, dtype=np.int64) - offs]
    pid = np.repeat(np.arange(len(nodes), dtype=np.int64), lens)
    return nbr, pid


# ----------------------------------------------------------------------------------- energy terms
class _CodeTerm:
    """Common state of the two neighbourhood terms: group codes and dense neighbour-code counts."""

    kind = ""

    def __init__(self, name: str, cols: list[str], size: int, codes: np.ndarray, strides: dict[str, int], weight: float):
        self.name, self.cols, self.size, self.g, self.strides, self.weight = name, cols, size, codes, strides, weight
        self.N: np.ndarray | None = None  # (n, G)
        self.energy = 0.0
        self.cells = 0

    def init_counts(self, indptr: np.ndarray, indices: np.ndarray, n: int) -> None:
        self.N = _neighbour_code_counts(indptr, indices, self.g, n, self.size)

    def apply(self, n: int, sel, u, v, gu, gnu, gv, gnv, nbr_u, pid_u, nbr_v, pid_v) -> None:
        """Move ``u: gu -> gnu`` and ``v: gv -> gnv`` for the accepted proposals ``sel`` in codes and counts.

        ``u, v, gu, ...`` are the full batch arrays; ``nbr_*`` / ``pid_*`` are the neighbour lists of the
        accepted proposals only (``pid`` indexes the full batch).
        """
        old = np.concatenate([nbr_u * self.size + gu[pid_u], nbr_v * self.size + gv[pid_v]])
        new = np.concatenate([nbr_u * self.size + gnu[pid_u], nbr_v * self.size + gnv[pid_v]])
        delta = np.bincount(new, minlength=n * self.size) - np.bincount(old, minlength=n * self.size)
        self.N += delta.reshape(n, self.size)
        self.g[u[sel]] = gnu[sel]
        self.g[v[sel]] = gnv[sel]

    @property
    def satisfied(self) -> bool:
        return self.energy <= self.weight * self.cells + _EPS


class _EdgePairTerm(_CodeTerm):
    kind = EDGE_PAIR_KIND

    def __init__(self, name, cols, size, codes, strides, target: np.ndarray, sigma: float, weight: float):
        super().__init__(name, cols, size, codes, strides, weight)
        self.T = target
        self.w = weight / (sigma**2 + target + 1.0)
        self.cells = size * size
        self.C = np.zeros_like(target)
        self.wd = np.zeros_like(target)  # w * (C - T), refreshed by ``exact``

    def exact(self, e0: np.ndarray, e1: np.ndarray) -> float:
        G = self.size
        c = np.bincount(self.g[e0] * G + self.g[e1], minlength=G * G).reshape(G, G).astype(float)
        self.C = c + c.T
        self.wd = self.w * (self.C - self.T)
        self.energy = float((self.wd * (self.C - self.T)).sum())
        return self.energy

    def delta(self, u, v, gu, gnu, gv, gnv) -> np.ndarray:
        """Exact energy change of every proposal (u: gu -> gnu, v: gv -> gnv) against the current counts.

        The change of the ordered count matrix is ``R + R^T`` with ``R`` carrying rows ``-N[u]`` at ``gu``,
        ``+N[u]`` at ``gnu``, ``-N[v]`` at ``gv``, ``+N[v]`` at ``gnv`` (coinciding rows add up); with symmetric
        ``w`` and ``D = C - T``: ``dE = 4 sum w D R + 2 sum w R^2 + 2 sum_ij w_ij R_ij R_ji``.
        """
        B = len(u)
        codes = np.stack([gu, gnu, gv, gnv], axis=1)  # (B, 4)
        Nu, Nv = self.N[u].astype(float), self.N[v].astype(float)
        vec = np.stack([-Nu, Nu, -Nv, Nv], axis=1)  # (B, 4, G)
        W = self.w[codes]  # (B, 4, G)
        A = np.einsum("bsh,bsh->b", self.wd[codes], vec)
        eq = codes[:, :, None] == codes[:, None, :]
        M = np.einsum("bsh,bsh,bth->bst", W, vec, vec)
        Bq = (M * eq).sum(axis=(1, 2))
        Wst = self.w[codes[:, :, None], codes[:, None, :]]  # (B, 4, 4)
        V = np.take_along_axis(vec, np.broadcast_to(codes[:, None, :], (B, 4, 4)), axis=2)  # vec[b, s, codes[b, t]]
        Cc = (Wst * V * V.transpose(0, 2, 1)).sum(axis=(1, 2))
        return 4.0 * A + 2.0 * Bq + 2.0 * Cc

    def prefix_delta(self, u, v, gu, gnu, gv, gnv) -> np.ndarray:
        """Exact combined energy change of every prefix of the (ordered) proposals ``1..K``."""
        K, G = len(u), self.size
        codes = np.stack([gu, gnu, gv, gnv], axis=1)
        Nu, Nv = self.N[u].astype(float), self.N[v].astype(float)
        vec = np.stack([-Nu, Nu, -Nv, Nv], axis=1)
        Z = np.zeros((K, G, G))
        np.add.at(Z, (np.repeat(np.arange(K), 4), codes.ravel()), vec.reshape(K * 4, G))
        Z = np.cumsum(Z + Z.transpose(0, 2, 1), axis=0)
        return np.einsum("ij,kij->k", 2.0 * self.wd, Z) + np.einsum("ij,kij,kij->k", self.w, Z, Z)


class _SimpsonTerm(_CodeTerm):
    kind = SIMPSON_KIND

    def __init__(self, name, cols, size, codes, strides, deg: np.ndarray, bin_edges: np.ndarray,
                 min_degree: int, target_mean: np.ndarray, sigma: float, weight: float):
        super().__init__(name, cols, size, codes, strides, weight)
        self.deg = deg.astype(float)
        self.dmin = int(min_degree)
        self.bin_edges = bin_edges
        self.nb = len(bin_edges) - 1
        self.sbin = degree_bin(deg, bin_edges)
        self.use = np.asarray(bin_edges[:-1], dtype=float) >= self.dmin
        self.counts = np.bincount(self.sbin, minlength=self.nb).astype(float)
        self.target_mean = np.clip(np.nan_to_num(target_mean, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        self.target = np.where(self.use, self.target_mean * self.counts, 0.0)
        self.scale = weight / (sigma**2 + 1.0)
        self.cells = int(self.use.sum())
        self.S = np.zeros(self.nb)
        self.dmin_mask = self.deg >= self.dmin
        self.inv_d2 = np.where(self.dmin_mask, 1.0 / np.maximum(self.deg, 1.0) ** 2, 0.0)

    def exact(self) -> float:
        s = _simpson_per_node(self.N, self.deg, self.dmin)
        self.S = np.bincount(self.sbin, weights=s, minlength=self.nb)
        self.energy = float(self.scale * (((self.S - self.target) ** 2)[self.use]).sum())
        return self.energy

    def delta(self, B: int, gu, gnu, gv, gnv, nbr_u, pid_u, nbr_v, pid_v) -> tuple[np.ndarray, np.ndarray]:
        """Linearised change: closed-form ``delta s_w`` summed per bin over the neighbours of both endpoints.

        Returns the energy change per proposal and the per-bin change ``dS`` ``(B, nb)`` it was computed from.
        """
        dS = np.zeros(B * self.nb)
        for nbr, pid, go, gn in ((nbr_u, pid_u, gu, gnu), (nbr_v, pid_v, gv, gnv)):
            val = 2.0 * (self.N[nbr, gn[pid]] - self.N[nbr, go[pid]] + 1) * self.inv_d2[nbr]
            dS += np.bincount(pid * self.nb + self.sbin[nbr], weights=val, minlength=B * self.nb)
        dS = dS.reshape(B, self.nb)
        dS[:, ~self.use] = 0.0
        return self.scale * (2.0 * (self.S - self.target)[None, :] * dS + dS**2).sum(axis=1), dS

    def prefix_delta(self, dS: np.ndarray) -> np.ndarray:
        """Combined (linearised) energy change of every prefix of the ordered proposals from their ``dS`` rows."""
        Z = np.cumsum(dS, axis=0)
        return self.scale * (2.0 * (self.S - self.target)[None, :] * Z + Z**2).sum(axis=1)


class _TwoWayTerm:
    kind = TWO_WAY_KIND

    def __init__(self, name: str, a: str, b: str, ka: int, kb: int, target: np.ndarray, sigma: float, weight: float):
        self.name, self.a, self.b, self.ka, self.kb, self.weight = name, a, b, ka, kb, weight
        self.T = target.ravel()
        self.w = weight / (sigma**2 + self.T + 1.0)
        self.cells = ka * kb
        self.C = np.zeros_like(self.T)
        self.wd = np.zeros_like(self.T)
        self.energy = 0.0

    def exact(self, xa: np.ndarray, xb: np.ndarray) -> float:
        self.C = np.bincount(xa * self.kb + xb, minlength=self.cells).astype(float)
        self.wd = self.w * (self.C - self.T)
        self.energy = float((self.wd * (self.C - self.T)).sum())
        return self.energy

    _VALS = np.array([-1.0, 1.0, -1.0, 1.0])

    def delta(self, moving: str, xu_m, xv_m, xu_o, xv_o) -> tuple[np.ndarray, np.ndarray]:
        """Exact change when ``u`` and ``v`` swap the value of ``moving`` (``*_o`` = their other value).

        Returns the energy change per proposal and the four touched cells ``(B, 4)`` (signs ``-+-+``).
        """
        kb = self.kb
        if moving == self.a:
            cells = np.stack([xu_m * kb + xu_o, xv_m * kb + xu_o, xv_m * kb + xv_o, xu_m * kb + xv_o], axis=1)
        else:
            cells = np.stack([xu_o * kb + xu_m, xu_o * kb + xv_m, xv_o * kb + xv_m, xv_o * kb + xu_m], axis=1)
        vals = self._VALS
        A = (self.wd[cells] * vals[None, :]).sum(axis=1)
        eq = cells[:, :, None] == cells[:, None, :]
        Bq = (self.w[cells][:, :, None] * (vals[:, None] * vals[None, :])[None] * eq).sum(axis=(1, 2))
        return 2.0 * A + Bq, cells

    def prefix_delta(self, cells: np.ndarray) -> np.ndarray:
        """Exact combined energy change of every prefix of the ordered proposals from their touched cells."""
        K = len(cells)
        Z = np.zeros((K, self.cells))
        np.add.at(Z, (np.repeat(np.arange(K), 4), cells.ravel()), np.tile(self._VALS, K))
        Z = np.cumsum(Z, axis=0)
        return (2.0 * self.wd[None, :] * Z + self.w[None, :] * Z**2).sum(axis=1)

    @property
    def satisfied(self) -> bool:
        return self.energy <= self.weight * self.cells + _EPS


# ----------------------------------------------------------------------------------- term builders
def _usable_cols(m: Measurement, domains: dict, columns, count: int | None = None) -> list[str] | None:
    cols = m.params.get("attrs") if isinstance(m.params, dict) else None
    if not isinstance(cols, (list, tuple)) or not cols:
        return None
    cols = [str(c) for c in cols]
    if count is not None and len(cols) != count:
        return None
    if any(c not in domains or c not in columns for c in cols):
        return None
    return cols


def _build_terms(graph: AttributedGraph, artifact: Artifact, x: dict[str, np.ndarray], deg: np.ndarray,
                 n_edges: int, weights: dict[str, float], warnings: list[str]):
    domains = artifact.schema.domains
    columns = list(x)
    simpson: list[_SimpsonTerm] = []
    pairs: list[_EdgePairTerm] = []
    for m in _by_kind(artifact, SIMPSON_KIND):
        cols = _usable_cols(m, domains, columns)
        if cols is None:
            warnings.append(f"{m.name}: params['attrs'] unusable, skipped")
            continue
        try:
            edges = check_bin_edges(m.params["bin_edges"])
            dmin = int(m.params["min_degree"])
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"{m.name}: bin_edges / min_degree unusable ({exc}), skipped")
            continue
        v = np.asarray(m.values, dtype=float)
        if v.ndim != 2 or v.shape != (len(edges) - 1, 2):
            warnings.append(f"{m.name}: values shape {v.shape}, expected {(len(edges) - 1, 2)}, skipped")
            continue
        g, size, strides = _codes_of(x, domains, cols)
        if size > MAX_DENSE_GROUPS:
            warnings.append(f"{m.name}: {size} groups exceed the dense limit {MAX_DENSE_GROUPS}, skipped")
            continue
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        mean = v[:, 0] / np.maximum(v[:, 1], 1.0)
        simpson.append(_SimpsonTerm(m.name, cols, size, g, strides, deg, edges, dmin, mean,
                                    float(m.sigma or 0.0), float(weights.get(m.name, 1.0))))
    for m in _by_kind(artifact, EDGE_PAIR_KIND):
        cols = _usable_cols(m, domains, columns)
        if cols is None:
            warnings.append(f"{m.name}: params['attrs'] unusable, skipped")
            continue
        g, size, strides = _codes_of(x, domains, cols)
        v = np.asarray(m.values, dtype=float)
        if v.shape != (size, size) or size > MAX_DENSE_GROUPS:
            warnings.append(f"{m.name}: values shape {v.shape} does not match {size} groups, skipped")
            continue
        q = _ordered_mass(v)
        if q.sum() <= 0:
            warnings.append(f"{m.name}: table has no mass, skipped")
            continue
        target = q * (2.0 * n_edges / q.sum())
        pairs.append(_EdgePairTerm(m.name, cols, size, g, strides, target, float(m.sigma or 0.0),
                                   float(weights.get(m.name, 1.0))))
    moving = sorted({c for t in simpson + pairs for c in t.cols})
    twoway: list[_TwoWayTerm] = []
    for m in _by_kind(artifact, TWO_WAY_KIND):
        cols = _usable_cols(m, domains, columns, 2)
        if cols is None or not (set(cols) & set(moving)):
            continue
        a, b = cols
        ka, kb = len(domains[a]), len(domains[b])
        v = np.asarray(m.values, dtype=float)
        if v.shape != (ka, kb):
            warnings.append(f"{m.name}: values shape {v.shape}, expected {(ka, kb)}, skipped")
            continue
        t = _clean(v)
        t = t * (len(deg) / t.sum()) if t.sum() > 0 else t
        twoway.append(_TwoWayTerm(m.name, a, b, ka, kb, t, float(m.sigma or 0.0), float(weights.get(m.name, 1.0))))
    return simpson, pairs, twoway, moving


# ----------------------------------------------------------------------------------- the chain
class _Chain:
    def __init__(self, graph: AttributedGraph, artifact: Artifact, *, max_rounds: int, batch: int | None,
                 damping: float, temperature: float, weights: dict[str, float] | None):
        self.n = n = int(graph.n)
        edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
        self.e0, self.e1 = edges[:, 0], edges[:, 1]
        self.keys = np.sort(self.e0 * n + self.e1)
        adj = sp.coo_matrix((np.ones(len(edges)), (self.e0, self.e1)), shape=(n, n))
        adj = (adj + adj.T).tocsr()
        adj.sum_duplicates()
        self.indptr, self.indices = adj.indptr.astype(np.int64), adj.indices.astype(np.int64)
        self.deg = np.bincount(edges.ravel(), minlength=n).astype(np.int64)
        self.columns = list(graph.attrs.columns)
        self.x = {c: graph.attrs[c].to_numpy().astype(np.int64).copy() for c in self.columns}
        self.domains = {c: list(v) for c, v in graph.domains.items()}
        self.warnings: list[str] = []
        self.max_rounds = int(max_rounds)
        self.batch = None if batch is None else max(int(batch), 1)
        self.damping0 = float(damping)
        self.temperature = float(temperature)

        self.simpson, self.pairs, self.twoway, self.moving = _build_terms(
            graph, artifact, self.x, self.deg, len(edges), weights or {}, self.warnings
        )
        for t in self.simpson + self.pairs:
            t.init_counts(self.indptr, self.indices, n)

        tables = _attr_degree_tables(artifact, graph.attrs)
        if tables is None:
            self.stratum_edges = None
            self.stratum = np.zeros(n, dtype=np.int64)
        else:
            self.stratum_edges = tables[0]
            self.stratum = degree_bin(self.deg, tables[0])

    # ---- energy
    @property
    def terms(self):
        return self.simpson + self.pairs + self.twoway

    def exact(self) -> float:
        for t in self.pairs:
            t.exact(self.e0, self.e1)
        for t in self.simpson:
            t.exact()
        for t in self.twoway:
            t.exact(self.x[t.a], self.x[t.b])
        return float(sum(t.energy for t in self.terms))

    # ---- proposals
    def _proposals(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """A random matching inside every stratum: disjoint pairs ``(u, v)`` with the same stratum."""
        n = self.n
        perm = rng.permutation(n)
        order = perm[np.argsort(self.stratum[perm], kind="stable")]
        s = self.stratum[order]
        new = np.r_[True, s[1:] != s[:-1]] if n else np.zeros(0, dtype=bool)
        gid = np.cumsum(new) - 1
        starts = np.flatnonzero(new)
        sizes = np.diff(np.r_[starts, n])
        pos = np.arange(n) - starts[gid]
        first = (pos % 2 == 0) & (pos + 1 < sizes[gid])
        i = np.flatnonzero(first)
        u, v = order[i], order[i + 1]
        if self.batch is not None and len(u) > self.batch:
            pick = rng.permutation(len(u))[:self.batch]
            u, v = u[pick], v[pick]
        return u, v

    def _batch(self, a: str, rng: np.random.Generator, damp: float) -> tuple[int, int]:
        """One batch of swaps of attribute ``a``; returns (proposed, accepted)."""
        x = self.x[a]
        u, v = self._proposals(rng)
        keep = x[u] != x[v]
        u, v = u[keep], v[keep]
        if len(self.keys) and len(u):
            key = np.minimum(u, v) * self.n + np.maximum(u, v)
            pos = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
            adjacent = self.keys[pos] == key
            u, v = u[~adjacent], v[~adjacent]
        B = len(u)
        if B == 0:
            return 0, 0
        xu, xv = x[u], x[v]
        d = xv - xu
        nbr_u, pid_u = _neighbours(self.indptr, self.indices, u)
        nbr_v, pid_v = _neighbours(self.indptr, self.indices, v)
        delta = np.zeros(B)
        moved: list[tuple[_CodeTerm, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        simpson_ds: list[tuple[_SimpsonTerm, np.ndarray]] = []
        twoway_cells: list[tuple[_TwoWayTerm, np.ndarray]] = []
        for t in self.pairs:
            stride = t.strides.get(a, 0)
            if stride == 0:
                continue
            gu, gv = t.g[u], t.g[v]
            gnu, gnv = gu + d * stride, gv - d * stride
            delta += t.delta(u, v, gu, gnu, gv, gnv)
            moved.append((t, gu, gnu, gv, gnv))
        for t in self.simpson:
            stride = t.strides.get(a, 0)
            if stride == 0:
                continue
            gu, gv = t.g[u], t.g[v]
            gnu, gnv = gu + d * stride, gv - d * stride
            dE, dS = t.delta(B, gu, gnu, gv, gnv, nbr_u, pid_u, nbr_v, pid_v)
            delta += dE
            moved.append((t, gu, gnu, gv, gnv))
            simpson_ds.append((t, dS))
        for t in self.twoway:
            if a == t.a and a == t.b:
                continue  # a table of an attribute with itself never changes under a swap
            if a == t.a:
                o = self.x[t.b]
            elif a == t.b:
                o = self.x[t.a]
            else:
                continue
            dE, cells = t.delta(a, xu, xv, o[u], o[v])
            delta += dE
            twoway_cells.append((t, cells))
        acc = delta < -_EPS
        if self.temperature > 0:
            with np.errstate(over="ignore"):
                p = np.exp(-np.maximum(delta, 0.0) / self.temperature)
            acc |= rng.random(B) < p
        cand = np.flatnonzero(acc)
        if len(cand) == 0:
            return B, 0
        # damp to a random subset in random order, then keep the prefix whose combined change is lowest:
        # proposals that are good one by one can push the same cells together (their interaction is
        # quadratic in the batch), and the prefix search sees that before anything is applied
        sel = rng.permutation(cand)[: max(int(damp * B), 1)]
        total = np.zeros(len(sel))
        for t, gu, gnu, gv, gnv in moved:
            if isinstance(t, _EdgePairTerm):
                total += t.prefix_delta(u[sel], v[sel], gu[sel], gnu[sel], gv[sel], gnv[sel])
        for t, dS in simpson_ds:
            total += t.prefix_delta(dS[sel])
        for t, cells in twoway_cells:
            total += t.prefix_delta(cells[sel])
        if self.temperature <= 0:
            k = int(np.argmin(total)) + 1
            if total[k - 1] >= -_EPS:
                return B, 0
            sel = sel[:k]
        mask = np.zeros(B, dtype=bool)
        mask[sel] = True
        mu, mv = mask[pid_u], mask[pid_v]
        for t, gu, gnu, gv, gnv in moved:
            t.apply(self.n, sel, u, v, gu, gnu, gv, gnv, nbr_u[mu], pid_u[mu], nbr_v[mv], pid_v[mv])
        x[u[sel]], x[v[sel]] = xv[sel], xu[sel]
        return B, int(len(sel))

    # ---- the run
    def run(self, rng: np.random.Generator) -> dict:
        before = {t.name: t.energy for t in self.terms}
        energy = float(sum(before.values()))
        hist_acc: list[int] = []
        hist_prop: list[int] = []
        hist_energy: list[float] = []
        hist_damp: list[float] = []
        damp = self.damping0
        stop = "max_rounds"
        rounds = 0
        if not self.moving:
            stop = "nothing_to_move"
        for _ in range(self.max_rounds):
            if not self.moving:
                break
            if all(t.satisfied for t in self.terms):
                stop = "satisfied"
                break
            start = int(rng.integers(len(self.moving)))
            proposed = accepted = 0
            for k in range(len(self.moving)):
                a = self.moving[(start + k) % len(self.moving)]
                p, acc = self._batch(a, rng, damp)
                proposed += p
                accepted += acc
                if acc > 0:
                    break
            rounds += 1
            new_energy = self.exact()
            hist_acc.append(accepted)
            hist_prop.append(proposed)
            hist_energy.append(new_energy)
            hist_damp.append(damp)
            if accepted == 0:
                stop = "no_accepted_move"
                break
            if new_energy > energy + _EPS:
                damp = max(damp / 2.0, 1.0 / max(self.n // 2, 1))
            else:
                damp = min(self.damping0, damp * 1.25)
            energy = new_energy
        else:
            if all(t.satisfied for t in self.terms):
                stop = "satisfied"
        after = {t.name: t.energy for t in self.terms}
        terms = {}
        for t in self.terms:
            entry = {
                "kind": t.kind,
                "cells": int(t.cells),
                "weight": float(t.weight),
                "energy_before": float(before[t.name]),
                "energy_after": float(after[t.name]),
                "satisfied": bool(t.satisfied),
            }
            if isinstance(t, _SimpsonTerm):
                entry.update(
                    attrs=list(t.cols),
                    bin_edges=[int(e) for e in t.bin_edges],
                    min_degree=t.dmin,
                    bins_used=[int(b) for b in np.flatnonzero(t.use)],
                    counts=t.counts.tolist(),
                    target=t.target.tolist(),
                    after=t.S.tolist(),
                    residual=(t.S - t.target).tolist(),
                    max_abs_residual_mean=float(
                        np.max(np.abs(t.S - t.target)[t.use] / np.maximum(t.counts[t.use], 1.0)) if t.cells else 0.0
                    ),
                )
            elif isinstance(t, _EdgePairTerm):
                entry.update(attrs=list(t.cols), max_abs_residual=float(np.abs(t.C - t.T).max()))
            else:
                entry.update(attrs=[t.a, t.b], max_abs_residual=float(np.abs(t.C - t.T).max()))
            terms[t.name] = entry
        return {
            "skipped": False,
            "moving_attrs": list(self.moving),
            "strata": {
                "bin_edges": None if self.stratum_edges is None else [int(e) for e in self.stratum_edges],
                "count": int(self.stratum.max()) + 1 if self.n else 0,
            },
            "terms": terms,
            "energy_before": float(sum(before.values())),
            "energy_after": float(sum(after.values())),
            "satisfied": bool(all(t.satisfied for t in self.terms)),
            "rounds": rounds,
            "stop": stop,
            "accepted": hist_acc,
            "proposed": hist_prop,
            "energy": hist_energy,
            "damping": hist_damp,
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------------------- API
def refine_labels(
    graph: AttributedGraph,
    artifact: Artifact,
    seed: int = 0,
    *,
    max_rounds: int = 200,
    batch: int | None = None,
    damping: float = 0.25,
    temperature: float = 0.0,
    weights: dict[str, float] | None = None,
) -> tuple[AttributedGraph, dict]:
    """Run the label chain on ``graph`` against the tables in ``artifact``; returns ``(graph, info)``.

    The chain only runs when the artifact carries at least one usable ``attr_neigh_simpson`` instance;
    otherwise the input graph object is returned untouched (``info["skipped"]``). The output graph shares the
    edge array with the input and gets a new attribute frame. ``batch`` caps the proposals per round (default:
    one matching of every stratum, about ``n // 2`` pairs); ``damping`` is the largest accepted fraction of a
    batch; ``temperature > 0`` switches to Metropolis acceptance; ``weights`` (by measurement name) rescale terms
    for diagnostics. Randomness comes from ``np.random.default_rng([seed, 2])``.
    """
    rng = np.random.default_rng([int(seed), 2])
    chain = _Chain(graph, artifact, max_rounds=max_rounds, batch=batch, damping=damping,
                   temperature=temperature, weights=weights)
    if not chain.simpson:
        has_kind = bool(_by_kind(artifact, SIMPSON_KIND))
        reason = "no usable attr_neigh_simpson measurement" if has_kind else "no attr_neigh_simpson measurement"
        return graph, {"skipped": True, "reason": reason, "warnings": list(chain.warnings), "rounds": 0,
                       "moving_attrs": [], "terms": {}, "accepted": [], "proposed": []}
    chain.exact()
    info = chain.run(rng)
    attrs = pd.DataFrame({c: chain.x[c] for c in chain.columns})
    attrs.index = pd.RangeIndex(graph.n)
    out = AttributedGraph(n=graph.n, edges=graph.edges, attrs=attrs, domains=graph.domains)
    return out, info

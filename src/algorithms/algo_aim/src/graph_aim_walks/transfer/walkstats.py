"""The walk process and the walk statistics: ONE definition, used on both sides of the boundary.

graph-aim-walks (``docs/snap/walks-plan.md``) releases tables of random-walk statistics measured on the real graph
(inside) and generates a synthetic graph whose walks have the same statistics (outside). Both sides must measure in
exactly the same way, so the walk process and every statistic live here, in the transfer zone (numpy only).

Walk process
------------
A pool of ``P`` simple random walks of ``T`` steps. Walk ``p`` starts at ``starts[p]`` (uniform over the nodes of
degree >= 1) and is driven by its own uniforms ``U[p, t]`` in ``[0, 1)``: from node ``x`` of degree ``d`` the next node
is ``neighbours(x)[floor(U * d)]``, the neighbours in ascending node order (CSR of the sorted edge list). A node of
degree 0 is never left. Because the walk is a deterministic function of ``(starts, U)`` and the graph, a generator can
keep ``(starts, U)`` fixed while it changes the graph (common random numbers) and re-simulate only the walks that pass
through a changed node, from their first visit of it on (:func:`resimulate_from`).

Statistics (all are SUMS over walks, so the contribution of a subset of walks can be subtracted and added back;
:func:`normalise` turns the sums into the released tables)
-----------------------------------------------------------------------------------------------------------------
``return``       (T,)            number of walks with ``X_t == X_0``, t = 1..T
``coverage``     (T,)            sum over walks of the number of distinct nodes among ``X_0..X_t``, t = 1..T
``degree_lag``   (L, B, B)       for every lag ``k`` in ``spec.lags`` and every ``t`` with ``t + k <= T``: count of
                                 ``(bin(deg X_t), bin(deg X_{t+k}))`` (``spec.degree_bin_edges``, transfer binning)
``attr_lag:<g>`` (L, G, G)       same with the group code of ``X_t`` and ``X_{t+k}`` (group = row-major product of the
                                 attributes in ``g``, like ``attr_edge_pair``)
``share:<a>``    (W, K, S)       for every window ``w`` in ``spec.windows``: the DISTINCT nodes among ``X_0..X_w``;
                                 the share of each value ``v`` of attribute ``a`` among them, binned into ``S`` equal
                                 bins on [0, 1] (share 1 falls in the last bin); counts of (window, value, bin)
``dominance:<a>|<b>`` (W, D, D)  for every window: the largest value share of ``a`` and of ``b`` among the distinct
                                 nodes, each binned into ``D`` equal bins on [0, 1]; counts of the pair (do regions
                                 cohere in two attributes at once)

The share and dominance statistics are walk-window versions of what the real-vs-fake classifier measures on its
samples (``graph_aim.eval.discriminator``); the classifier's samples (10-200 distinct nodes, a different walk rule)
are larger than the windows here, which keeps them as a held-out check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graph_aim.transfer.binning import check_bin_edges, degree_bin


@dataclass(frozen=True)
class WalkSpec:
    """Public parameters of the walk statistics (the same object is used inside and outside)."""

    length: int = 32
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    windows: tuple[int, ...] = (8, 16, 32)
    share_bins: int = 10
    dominance_bins: int = 5
    degree_bin_edges: tuple[int, ...] = (0, 2, 6, 16, 40, 100, 1201)
    groups: tuple[tuple[str, ...], ...] = field(default_factory=tuple)  # attr_lag groups
    share_attrs: tuple[str, ...] = field(default_factory=tuple)
    dominance_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.length < 1:
            raise ValueError("length must be >= 1")
        if any(k < 1 or k > self.length for k in self.lags):
            raise ValueError("every lag must be in [1, length]")
        if any(w < 1 or w > self.length for w in self.windows):
            raise ValueError("every window must be in [1, length]")
        check_bin_edges(list(self.degree_bin_edges))

    def to_json(self) -> dict:
        return {
            "length": self.length,
            "lags": list(self.lags),
            "windows": list(self.windows),
            "share_bins": self.share_bins,
            "dominance_bins": self.dominance_bins,
            "degree_bin_edges": list(self.degree_bin_edges),
            "groups": [list(g) for g in self.groups],
            "share_attrs": list(self.share_attrs),
            "dominance_pairs": [list(p) for p in self.dominance_pairs],
        }

    @classmethod
    def from_json(cls, d: dict) -> "WalkSpec":
        return cls(
            length=int(d["length"]),
            lags=tuple(int(k) for k in d["lags"]),
            windows=tuple(int(w) for w in d["windows"]),
            share_bins=int(d["share_bins"]),
            dominance_bins=int(d["dominance_bins"]),
            degree_bin_edges=tuple(int(e) for e in d["degree_bin_edges"]),
            groups=tuple(tuple(g) for g in d["groups"]),
            share_attrs=tuple(d["share_attrs"]),
            dominance_pairs=tuple(tuple(p) for p in d["dominance_pairs"]),
        )


# ------------------------------------------------------------------------------------------- graph / codes
def csr(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(indptr, indices)`` of the undirected graph, neighbours of every node in ascending order."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr), dst


def group_codes(attrs, domains: dict, cols) -> tuple[np.ndarray, int]:
    """Row-major product code of ``cols`` (same convention as ``attr_edge_pair``) and the number of codes."""
    dims = [len(domains[c]) for c in cols]
    codes = np.ravel_multi_index(tuple(np.asarray(attrs[c], dtype=np.int64) for c in cols), dims)
    return codes.astype(np.int64), int(np.prod(dims))


# ------------------------------------------------------------------------------------------------ walks
def draw_pool(degree: np.ndarray, n_walks: int, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, U)``: starts uniform over nodes of degree >= 1, ``U`` of shape ``(n_walks, length)``."""
    cand = np.flatnonzero(np.asarray(degree) > 0)
    if len(cand) == 0:
        cand = np.arange(len(degree))
    starts = cand[rng.integers(len(cand), size=n_walks)]
    return starts.astype(np.int64), rng.random((n_walks, length))


def simulate(indptr: np.ndarray, indices: np.ndarray, starts: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Paths ``(P, T + 1)`` of the walks driven by ``(starts, U)``; vectorized over walks."""
    P, T = U.shape
    deg = np.diff(indptr)
    paths = np.empty((P, T + 1), dtype=np.int64)
    cur = np.asarray(starts, dtype=np.int64).copy()
    paths[:, 0] = cur
    for t in range(T):
        d = deg[cur]
        step = np.minimum((U[:, t] * d).astype(np.int64), np.maximum(d - 1, 0))
        nxt = indices[np.minimum(indptr[cur] + step, len(indices) - 1)] if len(indices) else cur
        cur = np.where(d > 0, nxt, cur)
        paths[:, t + 1] = cur
    return paths


def resimulate_from(
    indptr: np.ndarray, indices: np.ndarray, paths: np.ndarray, U: np.ndarray, walks: np.ndarray, t0: np.ndarray
) -> np.ndarray:
    """New paths of ``walks`` after a graph change, keeping each walk's prefix ``X_0..X_{t0}`` (the first visit of
    a changed node) and re-drawing every later step with the walk's own uniforms. Returns ``(len(walks), T + 1)``."""
    walks = np.asarray(walks, dtype=np.int64)
    t0 = np.asarray(t0, dtype=np.int64)
    T = U.shape[1]
    deg = np.diff(indptr)
    out = paths[walks].copy()
    cur = out[np.arange(len(walks)), t0]
    for t in range(int(t0.min()) if len(t0) else T, T):
        active = t >= t0
        d = deg[cur]
        step = np.minimum((U[walks, t] * d).astype(np.int64), np.maximum(d - 1, 0))
        nxt = indices[np.minimum(indptr[cur] + step, len(indices) - 1)] if len(indices) else cur
        nxt = np.where(d > 0, nxt, cur)
        cur = np.where(active, nxt, cur)
        out[:, t + 1] = np.where(active, cur, out[:, t + 1])
    return out


# ------------------------------------------------------------------------------------------- statistics
def _first_occurrence(paths: np.ndarray) -> np.ndarray:
    """``(P, T + 1)`` bool: position ``t`` holds a node not seen at positions ``< t`` of the same walk."""
    P, L = paths.shape
    order = np.argsort(paths, axis=1, kind="stable")
    sp = np.take_along_axis(paths, order, axis=1)
    new = np.ones_like(sp, dtype=bool)
    new[:, 1:] = sp[:, 1:] != sp[:, :-1]
    first = np.zeros_like(new)
    np.put_along_axis(first, order, new, axis=1)
    return first


def _share_bin(share: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((share * bins).astype(np.int64), bins - 1)


def walk_counts(
    paths: np.ndarray,
    degree: np.ndarray,
    attrs,
    domains: dict,
    spec: WalkSpec,
    codes: dict | None = None,
) -> dict[str, np.ndarray]:
    """Summed statistics of the walks in ``paths`` (see the module docstring). ``codes`` may carry precomputed
    ``{("group", g): (codes, G), ("attr", a): (codes, K)}`` to avoid recomputing them on every call."""
    paths = np.asarray(paths, dtype=np.int64)
    P, L = paths.shape
    T = L - 1
    codes = dict(codes or {})
    out: dict[str, np.ndarray] = {}
    out["return"] = (paths[:, 1:] == paths[:, :1]).sum(axis=0).astype(float)
    first = _first_occurrence(paths)
    distinct = np.cumsum(first, axis=1)  # (P, T + 1): distinct nodes among X_0..X_t
    out["coverage"] = distinct[:, 1:].sum(axis=0).astype(float)

    edges_b = list(spec.degree_bin_edges)
    B = len(edges_b) - 1
    db = degree_bin(np.asarray(degree)[paths], edges_b)
    dl = np.zeros((len(spec.lags), B, B))
    for i, k in enumerate(spec.lags):
        np.add.at(dl[i], (db[:, :-k].ravel(), db[:, k:].ravel()), 1.0)
    out["degree_lag"] = dl

    for g in spec.groups:
        key = ("group", tuple(g))
        if key not in codes:
            codes[key] = group_codes(attrs, domains, list(g))
        c, G = codes[key]
        cp = c[paths]
        al = np.zeros((len(spec.lags), G, G))
        for i, k in enumerate(spec.lags):
            np.add.at(al[i], (cp[:, :-k].ravel(), cp[:, k:].ravel()), 1.0)
        out["attr_lag:" + "+".join(g)] = al

    S, D = spec.share_bins, spec.dominance_bins
    need = set(spec.share_attrs) | {a for pr in spec.dominance_pairs for a in pr}
    win_shares: dict[str, list[np.ndarray]] = {}
    for a in sorted(need):
        key = ("attr", a)
        if key not in codes:
            codes[key] = (np.asarray(attrs[a], dtype=np.int64), len(domains[a]))
        c, K = codes[key]
        cp = c[paths]
        per_w = []
        for w in spec.windows:
            m = first[:, : w + 1]
            cnt = np.zeros((P, K))
            rows = np.repeat(np.arange(P), w + 1)[m.ravel()]
            np.add.at(cnt, (rows, cp[:, : w + 1].ravel()[m.ravel()]), 1.0)
            per_w.append(cnt / distinct[:, w][:, None])  # (P, K) shares
        win_shares[a] = per_w
    for a in spec.share_attrs:
        K = len(domains[a])
        sh = np.zeros((len(spec.windows), K, S))
        for i, s in enumerate(win_shares[a]):
            b = _share_bin(s, S)  # (P, K)
            np.add.at(sh[i], (np.broadcast_to(np.arange(K), b.shape).ravel(), b.ravel()), 1.0)
        out["share:" + a] = sh
    for a, b in spec.dominance_pairs:
        dm = np.zeros((len(spec.windows), D, D))
        for i in range(len(spec.windows)):
            ma = _share_bin(win_shares[a][i].max(axis=1), D)
            mb = _share_bin(win_shares[b][i].max(axis=1), D)
            np.add.at(dm[i], (ma, mb), 1.0)
        out[f"dominance:{a}|{b}"] = dm
    return out


def normalise(counts: dict[str, np.ndarray], n_walks: int) -> dict[str, np.ndarray]:
    """Released tables from summed counts: ``return`` and ``coverage`` per walk; ``degree_lag`` and ``attr_lag``
    as joint distributions per lag; ``share`` per (window, value) and ``dominance`` per window as distributions."""
    out = {}
    for k, v in counts.items():
        v = np.asarray(v, dtype=float)
        if k in ("return", "coverage"):
            out[k] = v / max(n_walks, 1)
        elif k == "degree_lag" or k.startswith("attr_lag:"):
            s = v.sum(axis=(1, 2), keepdims=True)
            out[k] = np.divide(v, s, out=np.zeros_like(v), where=s > 0)
        elif k.startswith("share:"):
            s = v.sum(axis=2, keepdims=True)
            out[k] = np.divide(v, s, out=np.zeros_like(v), where=s > 0)
        elif k.startswith("dominance:"):
            s = v.sum(axis=(1, 2), keepdims=True)
            out[k] = np.divide(v, s, out=np.zeros_like(v), where=s > 0)
        else:
            raise ValueError(f"unknown statistic {k!r}")
    return out


def walk_tables(
    edges: np.ndarray, n: int, attrs, domains: dict, spec: WalkSpec, n_walks: int, seed: int
) -> dict[str, np.ndarray]:
    """Reference computation: draw a pool, simulate, count, normalise. Used inside on the real graph and in eval."""
    indptr, indices = csr(edges, n)
    deg = np.diff(indptr)
    starts, U = draw_pool(deg, n_walks, spec.length, np.random.default_rng(seed))
    paths = simulate(indptr, indices, starts, U)
    return normalise(walk_counts(paths, deg, attrs, domains, spec), n_walks)


def transition_spectrum(edges: np.ndarray, n: int, k: int = 32) -> np.ndarray:
    """The ``k`` largest eigenvalues of ``D^{-1/2} A D^{-1/2}`` (same spectrum as the walk matrix ``D^{-1} A``),
    descending. Isolated nodes are dropped. Dense below 1,500 nodes, ARPACK with a fixed start vector above."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = a + a.T
    deg = np.asarray(a.sum(axis=1)).ravel()
    keep = np.flatnonzero(deg > 0)
    a = a[keep][:, keep]
    dinv = sp.diags(1.0 / np.sqrt(deg[keep]))
    m = dinv @ a @ dinv
    nk = m.shape[0]
    k = int(min(k, max(nk - 1, 1)))
    if nk <= 1500 or k >= nk - 1:
        vals = np.linalg.eigvalsh(m.toarray())
    else:
        v0 = np.random.default_rng(0).standard_normal(nk)
        vals = eigsh(m, k=k, which="LA", v0=v0, tol=0, return_eigenvectors=False)
    return np.sort(vals)[::-1][:k]

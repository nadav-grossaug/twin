"""The walk pool on a mutable synthetic graph (outside zone; ``docs/snap/walks-plan.md`` section 4).

A :class:`WalkPool` holds ``P`` walks driven by fixed ``(starts, U)`` (``walkstats.draw_pool``) on a graph whose
degrees never change: the generator only makes degree-preserving double-edge swaps and attribute swaps. The graph is
a CSR whose ``indptr`` is therefore fixed; a swap rewrites four entries of ``indices`` (one per endpoint row).

Two row orders
--------------
- ``sorted_rows=True``: every neighbour list is kept in ascending order, exactly the ``walkstats.csr`` layout, so the
  pool's paths are the paths ``walkstats.simulate`` gives on ``walkstats.csr(edges)``. A swap re-sorts its four
  rows; the entries between the old and the new neighbour shift, so every walk that VISITS one of the four
  endpoints is re-simulated from its first visit (``walkstats.resimulate_from``).
- ``sorted_rows=False`` (the generator's default): a swap writes the new neighbour into the slot of the old one
  and the rows are not re-sorted. A walk changes only if it TRAVERSES one of the four rewritten slots, so a swap
  re-simulates the walks through two edges (about ``4 P T / 2m`` walk-steps, ~30 at SNAP scale with ``P =
  40000``) instead of every walk through four hubs (~3,000). The walk process is the same Markov chain (a uniform
  neighbour per step, driven by ``U``), only the map from ``U`` to the neighbour is permuted, so the statistics
  have the same distribution; the walks are not the ones ``walkstats.simulate`` draws on ``walkstats.csr`` with
  the same ``U`` (the released tables are measured on a fresh pool anyway).

Counts
------
``counts`` holds the summed statistics of ``walkstats.walk_counts`` (additive over walks) for the current paths,
plus ``coverage_sq`` (sum over walks of the squared distinct-node count after ``t`` steps, for the per-walk
variance of ``coverage``). :func:`pool_counts` computes the same sums as ``walkstats.walk_counts`` with
``np.bincount`` (several times faster than ``np.add.at``); the tests check that they are equal.
:meth:`WalkPool.check` recomputes paths and counts from scratch and asserts exact agreement.
"""

from __future__ import annotations

import numpy as np

from graph_aim.transfer.binning import degree_bin
from graph_aim_walks.transfer import walkstats as ws

# ------------------------------------------------------------------------------------- per-path values
def node_arrays(degree: np.ndarray, attrs: dict, domains: dict, spec: ws.WalkSpec) -> dict:
    """Per-node codes used by the statistics: ``"deg"`` (degree bin), ``"group:<g>"`` and ``"attr:<a>"``, plus
    the code counts under ``("size", key)``."""
    arr: dict = {}
    edges_b = list(spec.degree_bin_edges)
    arr["deg"] = degree_bin(np.asarray(degree), edges_b)
    arr[("size", "deg")] = len(edges_b) - 1
    for g in spec.groups:
        c, G = ws.group_codes(attrs, domains, list(g))
        key = "group:" + "+".join(g)
        arr[key], arr[("size", key)] = c, G
    need = set(spec.share_attrs) | {a for pr in spec.dominance_pairs for a in pr}
    for a in sorted(need):
        arr["attr:" + a] = np.asarray(attrs[a], dtype=np.int64)
        arr[("size", "attr:" + a)] = len(domains[a])
    return arr


def path_values(paths: np.ndarray, arr: dict, keys=None) -> dict[str, np.ndarray]:
    """``{key: arr[key][paths]}`` for the (string) keys of ``arr`` (or the given ``keys``)."""
    keys = [k for k in arr if isinstance(k, str)] if keys is None else keys
    return {k: arr[k][paths] for k in keys}


def first_occurrence(paths: np.ndarray) -> np.ndarray:
    """``(Q, T + 1)`` bool: position ``t`` holds a node not seen before in the same walk (as ``walkstats``)."""
    return ws._first_occurrence(paths)


def _window_shares(first: np.ndarray, distinct: np.ndarray, cp: np.ndarray, K: int, w: int) -> np.ndarray:
    """``(Q, K)`` value shares among the distinct nodes of ``X_0..X_w`` (same arithmetic as ``walk_counts``)."""
    Q = len(cp)
    m = first[:, : w + 1]
    rows = np.broadcast_to(np.arange(Q)[:, None], m.shape)[m]
    cnt = np.bincount(rows * K + cp[:, : w + 1][m], minlength=Q * K).reshape(Q, K).astype(float)
    return cnt / distinct[:, w][:, None]


def table_names(spec: ws.WalkSpec) -> list[str]:
    names = ["return", "coverage", "degree_lag"]
    names += ["attr_lag:" + "+".join(g) for g in spec.groups]
    names += ["share:" + a for a in spec.share_attrs]
    names += [f"dominance:{a}|{b}" for a, b in spec.dominance_pairs]
    return names


def tables_touching(spec: ws.WalkSpec, attr: str) -> list[str]:
    """The walk tables whose value can change when the labels of ``attr`` move (paths fixed)."""
    out = ["attr_lag:" + "+".join(g) for g in spec.groups if attr in g]
    out += ["share:" + a for a in spec.share_attrs if a == attr]
    out += [f"dominance:{a}|{b}" for a, b in spec.dominance_pairs if attr in (a, b)]
    return out


def pool_counts(paths: np.ndarray, vals: dict, sizes: dict, spec: ws.WalkSpec, only=None,
                first: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The sums of ``walkstats.walk_counts`` (plus ``coverage_sq``) from per-path values ``vals`` (see
    :func:`path_values`); ``sizes`` maps a value key to its code count. ``only``: restrict to these tables."""
    paths = np.asarray(paths, dtype=np.int64)
    Q, L = paths.shape
    want = set(table_names(spec) + ["coverage_sq"]) if only is None else set(only)
    out: dict[str, np.ndarray] = {}
    if first is None:
        first = first_occurrence(paths)
    distinct = np.cumsum(first, axis=1)
    if "return" in want:
        out["return"] = (paths[:, 1:] == paths[:, :1]).sum(axis=0).astype(float)
    if "coverage" in want:
        out["coverage"] = distinct[:, 1:].sum(axis=0).astype(float)
    if "coverage_sq" in want:
        out["coverage_sq"] = (distinct[:, 1:].astype(float) ** 2).sum(axis=0)

    def lag_table(cp, G):
        al = np.zeros((len(spec.lags), G, G))
        for i, k in enumerate(spec.lags):
            al[i] = np.bincount((cp[:, :-k] * G + cp[:, k:]).ravel(), minlength=G * G).reshape(G, G)
        return al

    if "degree_lag" in want:
        out["degree_lag"] = lag_table(vals["deg"], sizes["deg"])
    for g in spec.groups:
        name = "attr_lag:" + "+".join(g)
        if name in want:
            key = "group:" + "+".join(g)
            out[name] = lag_table(vals[key], sizes[key])
    S, D = spec.share_bins, spec.dominance_bins
    need = {a for a in spec.share_attrs if "share:" + a in want}
    need |= {a for a, b in spec.dominance_pairs if f"dominance:{a}|{b}" in want}
    need |= {b for a, b in spec.dominance_pairs if f"dominance:{a}|{b}" in want}
    shares = {a: [_window_shares(first, distinct, vals["attr:" + a], sizes["attr:" + a], w) for w in spec.windows]
              for a in sorted(need)}
    for a in spec.share_attrs:
        if "share:" + a not in want:
            continue
        K = sizes["attr:" + a]
        sh = np.zeros((len(spec.windows), K, S))
        for i, s in enumerate(shares[a]):
            b = ws._share_bin(s, S)
            sh[i] = np.bincount((np.arange(K)[None, :] * S + b).ravel(), minlength=K * S).reshape(K, S)
        out["share:" + a] = sh
    for a, b in spec.dominance_pairs:
        name = f"dominance:{a}|{b}"
        if name not in want:
            continue
        dm = np.zeros((len(spec.windows), D, D))
        for i in range(len(spec.windows)):
            ma = ws._share_bin(shares[a][i].max(axis=1), D)
            mb = ws._share_bin(shares[b][i].max(axis=1), D)
            dm[i] = np.bincount(ma * D + mb, minlength=D * D).reshape(D, D)
        out[name] = dm
    return out


def walk_scores(paths: np.ndarray, vals: dict, sizes: dict, spec: ws.WalkSpec, grads: dict,
                first: np.ndarray | None = None) -> np.ndarray:
    """Per-walk linear score ``sum_cells grads[table][cell] * count_walk[table][cell]`` over the tables in
    ``grads`` (``(Q,)``): the first-order energy contribution of each walk (see ``energy.gradients``)."""
    paths = np.asarray(paths, dtype=np.int64)
    Q, L = paths.shape
    if first is None:
        first = first_occurrence(paths)
    distinct = np.cumsum(first, axis=1)
    s = np.zeros(Q)
    if "return" in grads:
        s += (paths[:, 1:] == paths[:, :1]) @ grads["return"]
    if "coverage" in grads:
        s += distinct[:, 1:] @ grads["coverage"]

    def lag_score(cp, G, gr):
        out = np.zeros(Q)
        for i, k in enumerate(spec.lags):
            out += gr[i].ravel()[cp[:, :-k] * G + cp[:, k:]].sum(axis=1)
        return out

    if "degree_lag" in grads:
        s += lag_score(vals["deg"], sizes["deg"], grads["degree_lag"])
    for g in spec.groups:
        name = "attr_lag:" + "+".join(g)
        if name in grads:
            key = "group:" + "+".join(g)
            s += lag_score(vals[key], sizes[key], grads[name])
    S, D = spec.share_bins, spec.dominance_bins
    need = {a for a in spec.share_attrs if "share:" + a in grads}
    need |= {x for a, b in spec.dominance_pairs if f"dominance:{a}|{b}" in grads for x in (a, b)}
    shares = {a: [_window_shares(first, distinct, vals["attr:" + a], sizes["attr:" + a], w) for w in spec.windows]
              for a in sorted(need)}
    for a in spec.share_attrs:
        name = "share:" + a
        if name not in grads:
            continue
        K = sizes["attr:" + a]
        for i, sh in enumerate(shares[a]):
            b = ws._share_bin(sh, S)
            s += grads[name][i].ravel()[np.arange(K)[None, :] * S + b].sum(axis=1)
    for a, b in spec.dominance_pairs:
        name = f"dominance:{a}|{b}"
        if name not in grads:
            continue
        for i in range(len(spec.windows)):
            ma = ws._share_bin(shares[a][i].max(axis=1), D)
            mb = ws._share_bin(shares[b][i].max(axis=1), D)
            s += grads[name][i].ravel()[ma * D + mb]
    return s


def add_counts(acc: dict, delta: dict, sign: float = 1.0) -> None:
    for k, v in delta.items():
        acc[k] = acc[k] + sign * v


# ------------------------------------------------------------------------------------- the pool
class WalkPool:
    """Walks on a mutable degree-preserving graph (module docstring).

    ``indptr`` is fixed; ``indices`` is owned (copied) and rewritten by :meth:`swap_edges`. ``arr`` are the per-node
    codes (:func:`node_arrays`); :meth:`set_codes` changes them for the label chain.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, starts: np.ndarray, U: np.ndarray, arr: dict,
                 spec: ws.WalkSpec, *, sorted_rows: bool = False):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.deg = np.diff(self.indptr)
        self.n = len(self.deg)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.U = np.asarray(U, dtype=float)
        self.P, self.T = self.U.shape
        self.spec = spec
        self.sorted_rows = bool(sorted_rows)
        self.arr = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
        self.sizes = {k[1]: v for k, v in self.arr.items() if isinstance(k, tuple) and k[0] == "size"}
        self.paths = ws.simulate(self.indptr, self.indices, self.starts, self.U)
        self.first = first_occurrence(self.paths)  # cached: the label chain re-counts fixed paths
        self.slots = self._slots(self.paths, np.arange(self.P))
        self.counts = self.count(self.paths, first=self.first)

    # ---- helpers
    def _slots(self, paths: np.ndarray, walks: np.ndarray) -> np.ndarray:
        """``(Q, T)`` CSR slot used for each step (``-1`` where the node has degree 0)."""
        cur = paths[:, :-1]
        d = self.deg[cur]
        step = np.minimum((self.U[walks] * d).astype(np.int64), np.maximum(d - 1, 0))
        return np.where(d > 0, self.indptr[cur] + step, -1)

    def values(self, paths: np.ndarray, keys=None) -> dict:
        return path_values(paths, self.arr, keys)

    def count(self, paths: np.ndarray, only=None, first: np.ndarray | None = None) -> dict:
        return pool_counts(paths, self.values(paths), self.sizes, self.spec, only=only, first=first)

    def edges(self) -> np.ndarray:
        """Current edge list ``(m, 2)``, ``u < v``, lexicographically sorted."""
        rows = np.repeat(np.arange(self.n, dtype=np.int64), self.deg)
        keep = rows < self.indices
        e = np.stack([rows[keep], self.indices[keep]], axis=1)
        return e[np.lexsort((e[:, 1], e[:, 0]))]

    # ---- which walks a change touches
    def walks_through_slots(self, slots: np.ndarray, owner: np.ndarray | None = None):
        """Walks that traverse any of ``slots`` and the first step ``t0`` at which they do (``X_t0`` is kept).

        With ``owner`` (an ``(len(slots),)`` label per slot), returns ``(walk, t0, label)`` for every distinct
        ``(walk, label)`` pair, ``t0`` the first traversal of a slot of that label."""
        lab = np.full(len(self.indices) + 1, -1, dtype=np.int64)  # index -1 (degree-0 steps) maps to the last
        lab[np.asarray(slots, dtype=np.int64)] = 0 if owner is None else np.asarray(owner, dtype=np.int64)
        hit = lab[self.slots]
        w, t = np.nonzero(hit >= 0)
        if owner is None:
            first = np.r_[True, w[1:] != w[:-1]] if len(w) else np.zeros(0, dtype=bool)
            return w[first], t[first]
        o = hit[w, t]
        order = np.lexsort((t, o, w))
        w, t, o = w[order], t[order], o[order]
        first = np.r_[True, (w[1:] != w[:-1]) | (o[1:] != o[:-1])] if len(w) else np.zeros(0, dtype=bool)
        return w[first], t[first], o[first]

    def walks_visiting(self, nodes: np.ndarray, before_end: bool = True):
        """Walks that visit any of ``nodes`` and the first position of such a visit (the node -> (walk, first
        visit) index, computed on demand). ``before_end``: ignore visits at the last position (nothing follows)."""
        mark = np.zeros(self.n, dtype=bool)
        mark[np.asarray(nodes, dtype=np.int64)] = True
        hit = mark[self.paths[:, :-1] if before_end else self.paths]
        any_ = hit.any(axis=1)
        w = np.flatnonzero(any_)
        return w, hit[w].argmax(axis=1)

    # ---- graph changes
    def swap_edges(self, slot_a: np.ndarray, slot_b: np.ndarray, slot_c: np.ndarray, slot_d: np.ndarray) -> np.ndarray:
        """Apply double-edge swaps given by CSR slots: edges ``(a, b)`` at ``slot_a`` (row a) / ``slot_b`` (row b)
        and ``(c, d)`` at ``slot_c`` / ``slot_d`` become ``(a, d)`` and ``(c, b)``. Returns the changed rows
        (``sorted_rows``) or the changed slots. Paths and counts are NOT updated (see :meth:`update`)."""
        a = self.indices[slot_b]
        b = self.indices[slot_a]
        c = self.indices[slot_d]
        d = self.indices[slot_c]
        self.indices[slot_a] = d
        self.indices[slot_b] = c
        self.indices[slot_c] = b
        self.indices[slot_d] = a
        if not self.sorted_rows:
            return np.concatenate([slot_a, slot_b, slot_c, slot_d])
        rows = np.unique(np.concatenate([a, b, c, d]))
        self._sort_rows(rows)
        return rows

    def _sort_rows(self, rows: np.ndarray) -> None:
        lens = self.deg[rows]
        pos = np.repeat(self.indptr[rows], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        rid = np.repeat(np.arange(len(rows)), lens)
        vals = self.indices[pos]
        order = np.lexsort((vals, rid))
        self.indices[pos] = vals[order]

    def affected(self, changed: np.ndarray):
        """``(walks, t0)`` touched by the change returned from :meth:`swap_edges`."""
        if self.sorted_rows:
            return self.walks_visiting(changed)
        return self.walks_through_slots(changed)

    def resimulate(self, walks: np.ndarray, t0: np.ndarray) -> np.ndarray:
        return ws.resimulate_from(self.indptr, self.indices, self.paths, self.U, walks, t0)

    def update(self, walks: np.ndarray, new_paths: np.ndarray) -> None:
        """Replace the paths of ``walks`` and update the counts (old contribution out, new in)."""
        walks = np.asarray(walks, dtype=np.int64)
        if len(walks) == 0:
            return
        new_first = first_occurrence(new_paths)
        add_counts(self.counts, self.count(self.paths[walks], first=self.first[walks]), -1.0)
        add_counts(self.counts, self.count(new_paths, first=new_first), 1.0)
        self.paths[walks] = new_paths
        self.first[walks] = new_first
        self.slots[walks] = self._slots(new_paths, walks)

    def apply_swaps(self, slot_a, slot_b, slot_c, slot_d) -> int:
        """Swap edges and bring paths and counts up to date; returns the number of re-simulated walks."""
        changed = self.swap_edges(slot_a, slot_b, slot_c, slot_d)
        walks, t0 = self.affected(changed)
        self.update(walks, self.resimulate(walks, t0))
        return len(walks)

    def set_codes(self, attr: str, nodes: np.ndarray, old_values: np.ndarray, new_values: np.ndarray,
                  strides: dict) -> np.ndarray:
        """Change attribute ``attr`` of ``nodes`` from ``old_values`` to ``new_values`` (paths fixed) and update the
        counts of the touched tables; ``strides[group_key]`` is the row-major stride of ``attr`` in that group
        code. Returns the touched walks."""
        nodes = np.asarray(nodes, dtype=np.int64)
        walks, _ = self.walks_visiting(nodes, before_end=False)
        only = tables_touching(self.spec, attr)
        p, f = self.paths[walks], self.first[walks]
        old = pool_counts(p, self.values(p), self.sizes, self.spec, only=only, first=f)
        diff = np.asarray(new_values, dtype=np.int64) - np.asarray(old_values, dtype=np.int64)
        for gk, stride in strides.items():
            self.arr[gk][nodes] += diff * stride
        if "attr:" + attr in self.arr:
            self.arr["attr:" + attr][nodes] += diff
        new = pool_counts(p, self.values(p), self.sizes, self.spec, only=only, first=f)
        add_counts(self.counts, old, -1.0)
        add_counts(self.counts, new, 1.0)
        return walks

    # ---- debug
    def check(self) -> None:
        """Recompute paths and counts from scratch and assert exact agreement with the incremental state."""
        paths = ws.simulate(self.indptr, self.indices, self.starts, self.U)
        assert np.array_equal(paths, self.paths), "incremental paths differ from a fresh simulation"
        assert np.array_equal(self._slots(paths, np.arange(self.P)), self.slots)
        assert np.array_equal(first_occurrence(paths), self.first)
        fresh = self.count(paths)
        for k, v in fresh.items():
            assert np.array_equal(v, self.counts[k]), f"counts {k} differ"
        if self.sorted_rows:
            for r in range(self.n):
                row = self.indices[self.indptr[r]: self.indptr[r + 1]]
                assert np.all(row[1:] > row[:-1]), "row not sorted"

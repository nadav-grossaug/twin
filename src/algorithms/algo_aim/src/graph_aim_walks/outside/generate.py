"""graph-aim-walks generator (outside zone; ``docs/snap/walks-plan.md`` section 4). NOT PRIVATE: it reads the
exact walk tables and the exact base artifact of a :class:`WalksArtifact`.

``generate_walks(artifact, seed)``:

1. **Initialize** with ``graph_aim.outside.generate_v2.generate_v2_with_info(artifact.base, seed=seed)``.
2. **Cycles.** Each cycle draws two fresh walk pools on the current graph (``pool.WalkPool``, unsorted rows: a
   swap re-simulates only the walks that traverse a rewritten slot): the SELECTION pool ``A`` supplies the energy
   gradient of the screens, the VALIDATION pool ``B`` decides whether screened proposals are applied (exact
   re-simulation / recount of the walk energy plus the exact base terms). Screening and validating on the same
   pool fits the pool's own noise (``tools/walks_dev/NOTES.md``). The energies recorded at the start of a cycle
   (``info["cycles"][c]["fresh"]``) are measured on the freshly drawn ``B``, i.e. on walks the graph was never
   fitted to, and ``info["final"]`` on one more fresh pool after the last cycle.
3. **Structure chain** (``structure_sweeps`` x ``m / 2`` proposals per cycle, in batches of ``batch``):
   degree-preserving double-edge swaps ``(a, b), (c, d) -> (a, d), (c, b)``. Partners: a fraction ``local_frac``
   picks ``d`` two hops from ``a`` (the new edge closes a triangle), a fraction ``class_frac`` picks ``d`` of the
   same class as ``b`` (``class_keys``, default degree bin x the largest ``attr_lag`` group: the swap keeps those
   pair counts), the rest uniform. Screen, per proposal and independently of the others: the exact change of the
   base terms ``clustering_by_degree`` (triangles from sparse row intersections), ``joint_degree`` and
   ``attr_edge_pair`` (uncapped counts against the capped target shifted by the capped-vs-uncapped offset at the
   start of the phase), times their weights (``base_weight`` x ``base_weights[kind]``, default
   :data:`BASE_WEIGHTS`); plus the first-order change of the EXPECTED lag tables
   (``degree_lag``, ``attr_lag``) from their closed-form gradient in the transition matrix
   (``analytic.LagGradient``, no sampling noise); plus, for every ``share`` attribute whose walks are clearly
   more or less concentrated than the real ones, a surrogate (the same-value probability at every lag, scaled by
   the share table's energy over the concentration gap). ``pool_screen > 0`` adds the noisy pool-based first-order
   change of the other tables (every walk through the proposal's slots re-simulated with only that proposal).
4. **Label chain** (``label_sweeps`` matchings per attribute and cycle, batches of ``label_batch``): value swaps of
   one attribute between two non-adjacent nodes of the same ``attr_degree`` stratum (``attr_1way`` and
   ``attr_degree`` exact). Screen: exact base changes (``attr_edge_pair`` capped as released, ``attr_2way``) times
   their weights, the first-order change of the expected ``attr_lag`` tables, and the change of the ``share`` /
   ``dominance`` scores of the pool walks containing the two nodes (paths fixed).
5. **Validation.** The proposals with a negative screen (Metropolis at ``temperature`` > 0) are applied in chunks
   (``chunk`` / ``label_chunk``, most promising first, at most ``damping`` x batch per batch) to ``B``; a chunk is
   kept only if the exact energy (walk energy on ``B`` + weighted base terms) falls; a batch stops after
   ``retries + 1`` rejected chunks in a row.
6. **Stop** when every walk table's fresh energy is within ``2 x`` its informative cells, or after ``cycles``.
   ``info`` carries per-cycle per-table energies (fresh pool), base energies, acceptance counts, timings, the
   final energies on a fresh pool and the initializer's report.

Randomness: ``default_rng([seed, 101])`` for the chains, ``[seed, 102, cycle]`` for the pools; the output is
deterministic per seed. Everything here sees only the artifact.
"""

from __future__ import annotations

import time

import numpy as np

from graph_aim.outside.generate import _attr_degree_tables
from graph_aim.outside.generate_v2 import generate_v2_with_info
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.outside import analytic as an
from graph_aim_walks.outside import energy as en
from graph_aim_walks.outside import pool as pl
from graph_aim_walks.transfer import walkstats as ws

SPECTRUM = "transition_spectrum"
# the share surrogate is used only for attributes whose mean squared share (window 32) differs from the real one
# by at least this much: below it the scale 2 E / gap explodes and points the chain at the wrong target
SURROGATE_MIN_GAP = 0.02
# base terms are constraints the walk chain must keep, but their noise units are tiny next to the walk tables'
# (9 clustering cells against ~12,600 walk cells on SNAP): a collapse of every bin's mean clustering from 0.6 to
# 0.2 costs only ~850 units, which the walk tables outbid. Their weights in the chains' decisions, per kind:
BASE_WEIGHTS = {"clustering_by_degree": 300.0, "joint_degree": 1.0, "attr_edge_pair": 3.0, "attr_2way": 3.0}


def _resim_overlay(pool: pl.WalkPool, walks, t0, prop, slot_owner, slot_val) -> np.ndarray:
    """Paths of ``walks`` re-simulated from ``t0`` on the pool's graph with ONLY proposal ``prop[r]`` applied to
    row ``r``: a step through a slot owned by that proposal takes ``slot_val`` instead of the current entry."""
    T = pool.T
    out = pool.paths[walks].copy()
    if len(walks) == 0:
        return out
    r = np.arange(len(walks))
    cur = out[r, t0]
    last = len(pool.indices) - 1
    for t in range(int(t0.min()), T):
        active = t >= t0
        d = pool.deg[cur]
        step = np.minimum((pool.U[walks, t] * d).astype(np.int64), np.maximum(d - 1, 0))
        slot = np.minimum(pool.indptr[cur] + step, last)
        nxt = np.where(slot_owner[slot] == prop, slot_val[slot], pool.indices[slot])
        nxt = np.where(d > 0, nxt, cur)
        cur = np.where(active, nxt, cur)
        out[:, t + 1] = np.where(active, cur, out[:, t + 1])
    return out


class _Generator:
    def __init__(self, artifact, graph: AttributedGraph, seed: int, *, pool_size: int, val_pool_size: int | None,
                 batch: int, label_batch: int, damping: float, retries: int, temperature: float, local_frac: float,
                 weights: dict | None, grad_every: int, base_weight: float, base_weights: dict | None, pool_screen: float, class_frac: float,
                 class_keys, share_surrogate: float, chunk: int, label_chunk: int):
        self.art = artifact
        self.spec: ws.WalkSpec = artifact.spec
        self.real = {k: np.asarray(v, dtype=float) for k, v in artifact.tables.items() if k != SPECTRUM}
        self.n_real = int(artifact.n_walks)
        self.n = n = int(graph.n)
        self.domains = {c: list(v) for c, v in graph.domains.items()}
        self.columns = list(graph.attrs.columns)
        self.x = {c: graph.attrs[c].to_numpy().astype(np.int64).copy() for c in self.columns}
        e = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
        self.edges = np.stack([e.min(axis=1), e.max(axis=1)], axis=1)
        self.m = len(self.edges)
        self.indptr, self.indices = ws.csr(self.edges, n)
        self.deg = np.diff(self.indptr)
        rows = np.repeat(np.arange(n, dtype=np.int64), self.deg)
        key = rows * n + self.indices  # globally sorted (sorted rows)
        self.eslot = np.stack([np.searchsorted(key, self.edges[:, 0] * n + self.edges[:, 1]),
                               np.searchsorted(key, self.edges[:, 1] * n + self.edges[:, 0])], axis=1)
        self.edge_of_slot = np.empty(2 * self.m, dtype=np.int64)
        self.edge_of_slot[self.eslot[:, 0]] = np.arange(self.m)
        self.edge_of_slot[self.eslot[:, 1]] = np.arange(self.m)
        self.pool_size, self.val_pool_size = int(pool_size), int(val_pool_size or pool_size)
        self.batch, self.label_batch = int(batch), int(label_batch)
        self.damping, self.retries, self.temperature = float(damping), int(retries), float(temperature)
        self.local_frac = float(local_frac)
        self.weights = dict(weights or {})
        self.grad_every = max(int(grad_every), 1)
        self.base_weight = float(base_weight)
        self.base_weights = dict(BASE_WEIGHTS if base_weights is None else base_weights)
        self.class_frac = float(class_frac)
        self.share_surrogate = float(share_surrogate)
        self.chunk, self.label_chunk = max(int(chunk), 1), max(int(label_chunk), 1)
        self.class_keys = list(class_keys) if class_keys is not None else None
        self.pool_screen = float(pool_screen)
        self.seed = int(seed)
        self.warnings: list[str] = []
        self._refresh_graph()
        # base terms
        self.clu, self.jd = en.structure_terms(artifact.base, self.deg, self.warnings)
        self.tri = en.triangles(self.adj).astype(float)
        if self.clu is not None:
            self.clu.exact(self.tri)
        if self.jd is not None:
            self.jd.exact(self.edges)
        self.pairs, self.twoway = en.label_terms(artifact.base, self.x, self.domains, self.edges, n, self.warnings)
        for t in self.pairs:
            t.exact()
        for t in self.twoway:
            t.exact(self.x)
        # label strata and moving attributes
        tables = _attr_degree_tables(artifact.base, graph.attrs)
        self.stratum = np.zeros(n, dtype=np.int64) if tables is None else degree_bin(self.deg, tables[0])
        spec = self.spec
        moving = set(spec.share_attrs) | {a for g in spec.groups for a in g}
        moving |= {a for pr in spec.dominance_pairs for a in pr}
        self.moving = sorted(a for a in moving if a in self.x)
        self.gstrides: dict[str, dict[str, int]] = {a: {} for a in self.moving}
        for g in spec.groups:
            sizes = [len(self.domains[c]) for c in g]
            for i, c in enumerate(g):
                if c in self.gstrides:
                    key_ = "group:" + "+".join(g)
                    self.gstrides[c][key_] = self.gstrides[c].get(key_, 0) + int(np.prod(sizes[i + 1:], dtype=np.int64))
        # lag tables with a closed-form gradient: (table name, node-code key)
        self.lag_tables = [("degree_lag", "deg")] + [("attr_lag:" + "+".join(g), "group:" + "+".join(g))
                                                      for g in spec.groups]
        self.lag_tables = [(t, k) for t, k in self.lag_tables if t in self.real]
        self.p0 = an.start_distribution(self.deg)
        if self.class_keys is None:  # default: the degree bin and the largest attr_lag group
            groups = sorted(self.spec.groups, key=lambda g: -int(np.prod([len(self.domains[c]) for c in g])))
            self.class_keys = ["deg"] + (["group:" + "+".join(groups[0])] if groups else [])

    def _bw(self, kind: str) -> float:
        """Weight of a base term in the chains' decisions (the reported energies are unweighted)."""
        return self.base_weight * float(self.base_weights.get(kind, 1.0))

    # ------------------------------------------------------------------ graph bookkeeping
    def _refresh_graph(self) -> None:
        self.adj = en.adjacency(self.edges, self.n)
        self.keys = np.sort(self.edges[:, 0] * self.n + self.edges[:, 1])

    def _is_edge(self, u, v) -> np.ndarray:
        k = np.minimum(u, v) * self.n + np.maximum(u, v)
        pos = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys) - 1)
        return self.keys[pos] == k

    def graph(self) -> AttributedGraph:
        import pandas as pd

        e = self.edges[np.lexsort((self.edges[:, 1], self.edges[:, 0]))]
        attrs = pd.DataFrame({c: self.x[c] for c in self.columns})
        return AttributedGraph(n=self.n, edges=e, attrs=attrs, domains=self.domains)

    # ------------------------------------------------------------------ pools and energies
    def _pool(self, size: int, rng) -> pl.WalkPool:
        starts, U = ws.draw_pool(self.deg, size, self.spec.length, rng)
        arr = pl.node_arrays(self.deg, self.x, self.domains, self.spec)
        return pl.WalkPool(self.indptr, self.indices, starts, U, arr, self.spec)

    def _draw_pools(self, cycle: int) -> None:
        rng = np.random.default_rng([self.seed, 102, int(cycle)])
        self.A = self._pool(self.pool_size, rng)
        self.B = self._pool(self.val_pool_size, rng)
        self._refresh_grads()

    def walk_energy(self, counts: dict, n_pool: int) -> dict:
        return en.walk_energy(counts, n_pool, self.real, self.pool_size, self.n_real, self.weights)

    def _walk_total(self, counts: dict, n_pool: int) -> float:
        return float(sum(v["energy"] for v in self.walk_energy(counts, n_pool).values()))

    def _refresh_grads(self) -> None:
        """Energy gradients of the selection pool: per-walk scores for the tables screened on the pool, and the
        closed-form lag-table gradient (``analytic.LagGradient``) on the current graph."""
        A = self.A
        g_all = en.gradients(A.counts, A.P, self.real, self.pool_size, self.n_real, self.weights)
        lag = {t for t, _ in self.lag_tables}
        self.grads = {k: v for k, v in g_all.items() if k not in lag}  # pool-screened tables
        W = en.gradients(A.counts, A.P, self.real, self.pool_size, self.n_real, self.weights, per="table")
        P, _ = an.transition(self.edges, self.n)
        self.lg_keys = [key for _, key in self.lag_tables]
        tabs = [(A.arr[key], A.sizes[key], W[t]) for t, key in self.lag_tables]
        self.surrogate_lambda = {}
        if self.share_surrogate > 0:
            # surrogate for the share tables (no closed form): the same-value probability at every lag of each
            # share attribute, pushed up (down) when the pool's walks are less (more) concentrated in that
            # attribute than the real ones, scaled so that closing the concentration gap is worth the table's energy
            syn = ws.normalise({k: v for k, v in A.counts.items() if k != "coverage_sq"}, A.P)
            e = en.walk_energy(A.counts, A.P, self.real, self.pool_size, self.n_real, self.weights)
            centre2 = ((np.arange(self.spec.share_bins) + 0.5) / self.spec.share_bins) ** 2
            L = len(self.spec.lags)
            for a in self.spec.share_attrs:
                name = "share:" + a
                if name not in self.real or "attr:" + a not in A.arr:
                    continue
                gap = float((self.real[name][-1] * centre2).sum() - (syn[name][-1] * centre2).sum())
                if abs(gap) < SURROGATE_MIN_GAP:  # concentration matched: the rest of the gap is shape, not scale
                    continue
                lam = self.share_surrogate * 2.0 * e[name]["energy"] / gap
                self.surrogate_lambda[a] = lam
                K = A.sizes["attr:" + a]
                tabs.append((A.arr["attr:" + a], K, np.broadcast_to(-lam / L * np.eye(K), (L, K, K)).copy()))
                self.lg_keys.append("attr:" + a)
        self.lg = an.LagGradient(P, self.p0, self.spec, tabs,
                                 with_codes=[not k.startswith("attr:") for k in self.lg_keys])
        self.s_old = (pl.walk_scores(A.paths, A.values(A.paths), A.sizes, self.spec, self.grads)
                      if self.pool_screen > 0 else None)

    def _rescore(self, walks: np.ndarray) -> None:
        if len(walks) and self.s_old is not None:
            p = self.A.paths[walks]
            self.s_old[walks] = pl.walk_scores(p, self.A.values(p), self.A.sizes, self.spec, self.grads)

    def base_energy(self) -> dict:
        out = {}
        if self.clu is not None:
            out[self.clu.name] = {"energy": self.clu.energy, "cells": self.clu.cells}
        if self.jd is not None:
            out[self.jd.name] = {"energy": self.jd.energy, "cells": self.jd.cells}
        for t in self.pairs:
            out[t.name] = {"energy": t.energy, "cells": t.cells}
        for t in self.twoway:
            out[t.name] = {"energy": t.energy, "cells": t.cells}
        return out

    # ------------------------------------------------------------------ structure chain
    def _propose_swaps(self, rng, K: int):
        m, n = self.m, self.n
        i = rng.integers(m, size=K)
        fi = rng.random(K) < 0.5
        E, S = self.edges, self.eslot
        a = np.where(fi, E[i, 1], E[i, 0])
        b = np.where(fi, E[i, 0], E[i, 1])
        sa = np.where(fi, S[i, 1], S[i, 0])
        sb = np.where(fi, S[i, 0], S[i, 1])
        # partner edge: uniform, or (local) through a random 2-hop neighbour d of a
        j = rng.integers(m, size=K)
        fj = rng.random(K) < 0.5
        sc = np.where(fj, S[j, 1], S[j, 0])
        sd = np.where(fj, S[j, 0], S[j, 1])
        r = rng.random(K)
        loc = r < self.local_frac
        cls = (r >= self.local_frac) & (r < self.local_frac + self.class_frac)
        if cls.any():
            # class-preserving partner: d has the class of b (so a's and c's neighbour classes are kept)
            cb = self.swap_class[b]
            sdd = self.class_slots[self.class_start[cb] + (rng.random(K) * self.class_count[cb]).astype(np.int64)]
            j_c = self.edge_of_slot[sdd]
            other = np.where(S[j_c, 0] == sdd, S[j_c, 1], S[j_c, 0])
            sd = np.where(cls, sdd, sd)
            sc = np.where(cls, other, sc)
            j = np.where(cls, j_c, j)
        if loc.any():
            ip, ix, deg = self.indptr, self.indices, self.deg
            y = ix[ip[a] + (rng.random(K) * deg[a]).astype(np.int64)]
            sd_l = ip[y] + (rng.random(K) * deg[y]).astype(np.int64)  # slot in row y pointing at d -> edge (y, d)
            d_l = ix[sd_l]
            # the partner edge is (c, d) with c a random neighbour of d: slot in row d
            sdd = ip[d_l] + (rng.random(K) * deg[d_l]).astype(np.int64)
            j_l = self.edge_of_slot[sdd]
            other = np.where(S[j_l, 0] == sdd, S[j_l, 1], S[j_l, 0])  # slot of the edge in row c
            sd = np.where(loc, sdd, sd)
            sc = np.where(loc, other, sc)
            j = np.where(loc, j_l, j)
        c = self.indices[sd]
        d = self.indices[sc]
        ok = (a != d) & (c != b) & (a != c) & (b != d) & (i != j)
        ok &= ~self._is_edge(a, d) & ~self._is_edge(c, b)
        k1 = np.minimum(a, d) * n + np.maximum(a, d)
        k2 = np.minimum(c, b) * n + np.maximum(c, b)
        ok &= k1 != k2
        for keys in (np.concatenate([k1, k2]), np.concatenate([i, j])):  # new edges and used edges unique
            _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
            dup = cnt[inv] > 1
            ok &= ~(dup[:K] | dup[K:])
        sel = np.flatnonzero(ok)
        return a[sel], b[sel], c[sel], d[sel], sa[sel], sb[sel], sc[sel], sd[sel], i[sel], j[sel]

    def _clustering_screen(self, a, b, c, d) -> np.ndarray:
        K = len(a)
        if self.clu is None or K == 0:
            return np.zeros(K)
        X = np.concatenate([a, c, a, c])
        Y = np.concatenate([b, d, d, b])
        r, w = en.common_neighbours(self.adj, X, Y)
        kind, pid = r // K, r % K
        excl = ((kind == 2) & ((w == b[pid]) | (w == c[pid]))) | ((kind == 3) & ((w == d[pid]) | (w == a[pid])))
        r, w, kind, pid = r[~excl], w[~excl], kind[~excl], pid[~excl]
        sign = np.where(kind < 2, -1.0, 1.0)
        prop = np.concatenate([pid, pid, pid])
        node = np.concatenate([w, X[r], Y[r]])
        return self.clu.delta(prop, node, np.concatenate([sign, sign, sign]), self.tri, K)

    def _base_exact_structure(self, new_edges, removed, added, adj_new):
        dE = 0.0
        out = {}
        if self.clu is not None:
            tri = self.tri + en.triangle_change(self.adj, adj_new, removed, added, self.n)
            S = np.bincount(self.clu.bin, weights=self.clu.f(tri, np.arange(self.n)), minlength=self.clu.nb)
            e = float((self.clu.w * (S - self.clu.T) ** 2).sum())
            dE += self._bw("clustering_by_degree") * (e - self.clu.energy)
            out["clu"] = (tri, S, e)
        for t in ([self.jd] if self.jd is not None else []) + self.struct_pairs:
            C = t.counts(new_edges)
            e = float((t.w * (C - t.T) ** 2).sum())
            dE += self._bw(t.kind) * (e - t.energy)
            out[t.name] = (C, e)
        return dE, out

    def _structure_batch(self, rng, stats: dict) -> None:
        a, b, c, d, sa, sb, sc, sd, ei, ej = self._propose_swaps(rng, self.batch)
        K = len(a)
        stats["proposed"] += K
        if K == 0:
            return
        delta = self._bw("clustering_by_degree") * self._clustering_screen(a, b, c, d)
        for t in ([self.jd] if self.jd is not None else []) + self.struct_pairs:
            delta += self._bw(t.kind) * t.swap_delta(a, b, c, d)
        # lag tables: exact first-order change of their expectation (no sampling noise)
        delta += self.lg.swap_delta(a, b, c, d, self.deg)
        if self.pool_screen > 0 and self.grads:
            # the other tables: re-simulate the pool walks through the proposal's slots, one proposal at a time
            A = self.A
            slots = np.concatenate([sa, sb, sc, sd])
            owner = np.tile(np.arange(K), 4)
            w, t0, o = A.walks_through_slots(slots, owner)
            slot_owner = np.full(len(A.indices), -1, dtype=np.int64)
            slot_val = np.zeros(len(A.indices), dtype=np.int64)
            slot_owner[slots] = owner
            slot_val[slots] = np.concatenate([d, c, b, a])
            newp = _resim_overlay(A, w, t0, o, slot_owner, slot_val)
            s_new = pl.walk_scores(newp, A.values(newp), A.sizes, self.spec, self.grads)
            delta += self.pool_screen * np.bincount(o, weights=s_new - self.s_old[w], minlength=K)
        acc = delta < 0
        if self.temperature > 0:
            with np.errstate(over="ignore"):
                acc |= rng.random(K) < np.exp(-np.maximum(delta, 0.0) / self.temperature)
        cand = np.flatnonzero(acc)
        stats["screened"] += len(cand)
        if len(cand) == 0:
            return
        self._validate(cand, delta, K, stats, rng,
                       lambda sel: self._try_structure(sel, a, b, c, d, sa, sb, sc, sd, ei, ej, rng), self.chunk)

    def _validate(self, cand, delta, K, stats, rng, try_fn, chunk: int) -> None:
        """Validate the screened proposals on pool B in chunks of ``chunk``, most promising first; at most
        ``damping * K`` proposals are applied per batch, and the batch stops after ``retries + 1`` rejected
        chunks in a row."""
        if self.temperature > 0:
            cand = rng.permutation(cand)
        else:
            cand = cand[np.argsort(delta[cand], kind="stable")]
        cand = cand[: max(int(self.damping * K), 1)]
        misses = 0
        for s in range(0, len(cand), chunk):
            sel = cand[s: s + chunk]
            if try_fn(sel):
                stats["accepted"] += len(sel)
                stats["batches_accepted"] += 1
                misses = 0
            else:
                stats["batches_rejected"] += 1
                misses += 1
                if misses > self.retries:
                    return

    def _try_structure(self, sel, a, b, c, d, sa, sb, sc, sd, ei, ej, rng) -> bool:
        a, b, c, d, sa, sb, sc, sd, ei, ej = (v[sel] for v in (a, b, c, d, sa, sb, sc, sd, ei, ej))
        new_edges = self.edges.copy()
        new_edges[ei] = np.stack([np.minimum(a, d), np.maximum(a, d)], axis=1)
        new_edges[ej] = np.stack([np.minimum(c, b), np.maximum(c, b)], axis=1)
        removed = np.concatenate([np.stack([a, b], 1), np.stack([c, d], 1)])
        added = np.concatenate([np.stack([a, d], 1), np.stack([c, b], 1)])
        adj_new = en.adjacency(new_edges, self.n)
        dE, base_new = self._base_exact_structure(new_edges, removed, added, adj_new)
        B = self.B
        e_old = self._walk_total(B.counts, B.P)
        changed = B.swap_edges(sa, sb, sc, sd)
        wb, t0 = B.affected(changed)
        newp = B.resimulate(wb, t0)
        cnt = {k: v.copy() for k, v in B.counts.items()}
        pl.add_counts(cnt, B.count(B.paths[wb], first=B.first[wb]), -1.0)
        pl.add_counts(cnt, B.count(newp), 1.0)
        dW = self._walk_total(cnt, B.P) - e_old
        self.last_try = {"base": dE, "walk": dW, "cnt": cnt}
        dE += dW
        ok = dE < 0 or (self.temperature > 0 and rng.random() < np.exp(-dE / self.temperature))
        if not ok:
            B.swap_edges(sa, sd, sc, sb)  # undo (see pool.swap_edges: the reverse swap)
            return False
        B.update(wb, newp)
        wa = np.unique(np.concatenate([self.A.walks_through_slots(changed)[0]]))
        self.A.apply_swaps(sa, sb, sc, sd)
        self._rescore(wa)
        # master graph
        ix = self.indices
        ix[sa], ix[sb], ix[sc], ix[sd] = d, c, b, a
        self.edges = new_edges
        self.eslot[ei] = np.where((new_edges[ei, 0] == a)[:, None], np.stack([sa, sd], 1), np.stack([sd, sa], 1))
        self.eslot[ej] = np.where((new_edges[ej, 0] == c)[:, None], np.stack([sc, sb], 1), np.stack([sb, sc], 1))
        self.edge_of_slot[sd] = ei
        self.edge_of_slot[sb] = ej
        self.adj = adj_new
        self.keys = np.sort(new_edges[:, 0] * self.n + new_edges[:, 1])
        if self.clu is not None:
            self.tri, self.clu.S, self.clu.energy = base_new["clu"]
        for t in ([self.jd] if self.jd is not None else []) + self.struct_pairs:
            t.C, t.energy = base_new[t.name]
        return True

    def _start_structure_phase(self) -> None:
        """Class index of the class-preserving proposals, and the edge-pair guards for the structure chain:
        uncapped counts, target = released (capped) target shifted by the current uncapped-minus-capped offset."""
        cls = np.zeros(self.n, dtype=np.int64)
        arr = pl.node_arrays(self.deg, self.x, self.domains, self.spec)
        for key in self.class_keys:
            size = arr[("size", key)]
            cls = cls * size + arr[key]
        self.swap_class = cls
        row_of_slot = np.repeat(np.arange(self.n), self.deg)
        cs = cls[row_of_slot]
        self.class_slots = np.argsort(cs, kind="stable")
        ncls = int(cls.max()) + 1 if self.n else 1
        self.class_count = np.bincount(cs, minlength=ncls)
        self.class_start = np.cumsum(self.class_count) - self.class_count
        self.struct_pairs = []
        for t in self.pairs:
            t.set_edges(self.edges)
            t.exact()
            unc = en.PairCountTerm(t.name, "attr_edge_pair", t.g, t.size, np.zeros_like(t.T), t.sigma)
            C_unc = unc.counts(self.edges)
            unc.T = t.T + (C_unc - t.C)
            unc.w = t.w
            unc.exact(self.edges)
            self.struct_pairs.append(unc)

    # ------------------------------------------------------------------ label chain
    def _label_proposals(self, rng, attr: str):
        n = self.n
        perm = rng.permutation(n)
        order = perm[np.argsort(self.stratum[perm], kind="stable")]
        s = self.stratum[order]
        new = np.r_[True, s[1:] != s[:-1]]
        gid = np.cumsum(new) - 1
        starts = np.flatnonzero(new)
        sizes = np.diff(np.r_[starts, n])
        pos = np.arange(n) - starts[gid]
        idx = np.flatnonzero((pos % 2 == 0) & (pos + 1 < sizes[gid]))
        u, v = order[idx], order[idx + 1]
        x = self.x[attr]
        keep = (x[u] != x[v]) & ~self._is_edge(u, v)
        return u[keep], v[keep]

    def _label_batch(self, rng, attr: str, u, v, stats: dict) -> None:
        K = len(u)
        stats["proposed"] += K
        if K == 0:
            return
        x = self.x[attr]
        xu, xv = x[u], x[v]
        delta = np.zeros(K)
        for t in self.pairs:
            delta += self._bw(t.kind) * t.swap_delta(attr, u, v, xu, xv)
        for t in self.twoway:
            delta += self._bw(t.kind) * t.swap_delta(attr, self.x, u, v)
        for ti, key in enumerate(self.lg_keys):
            # the share surrogates ("attr:<a>") steer the structure only: the label screen scores the share
            # tables on the pool directly (the paths are fixed, so that score has no re-simulation noise)
            stride = 0 if key.startswith("attr:") else self.gstrides[attr].get(key, 0)
            if stride:
                gc = self.A.arr[key]
                gu, gv = gc[u], gc[v]
                delta += self.lg.code_delta(ti, u, gu, gu + (xv - xu) * stride, v, gv, gv + (xu - xv) * stride)
        tables = [k for k in pl.tables_touching(self.spec, attr) if k in self.grads]
        if tables:
            A = self.A
            lab = np.full(self.n, -1, dtype=np.int64)
            lab[u] = np.arange(K)
            lab[v] = np.arange(K)
            hit = lab[A.paths]
            w, tt = np.nonzero(hit >= 0)
            o = hit[w, tt]
            key = np.unique(w * K + o)
            w, o = key // K, key % K
            p = A.paths[w]
            keys = ["attr:" + attr] + list(self.gstrides[attr])
            keys += ["attr:" + b for a_, b in self.spec.dominance_pairs if a_ == attr]
            keys += ["attr:" + a_ for a_, b in self.spec.dominance_pairs if b == attr]
            vals = A.values(p, sorted(set(k for k in keys if k in A.arr)))
            grads = {k: self.grads[k] for k in tables}
            f = A.first[w]
            s_old = pl.walk_scores(p, vals, A.sizes, self.spec, grads, first=f)
            isu, isv = p == u[o][:, None], p == v[o][:, None]
            dv = np.where(isu, (xv - xu)[o][:, None], 0) + np.where(isv, (xu - xv)[o][:, None], 0)
            nv = dict(vals)
            if "attr:" + attr in nv:
                nv["attr:" + attr] = vals["attr:" + attr] + dv
            for gk, stride in self.gstrides[attr].items():
                nv[gk] = vals[gk] + dv * stride
            s_new = pl.walk_scores(p, nv, A.sizes, self.spec, grads, first=f)
            delta += np.bincount(o, weights=s_new - s_old, minlength=K)
        acc = delta < 0
        if self.temperature > 0:
            with np.errstate(over="ignore"):
                acc |= rng.random(K) < np.exp(-np.maximum(delta, 0.0) / self.temperature)
        cand = np.flatnonzero(acc)
        stats["screened"] += len(cand)
        if len(cand) == 0:
            return
        self._validate(cand, delta, K, stats, rng, lambda sel: self._try_label(attr, u[sel], v[sel], rng),
                       self.label_chunk)

    def _try_label(self, attr: str, u, v, rng) -> bool:
        x = self.x[attr]
        xu, xv = x[u].copy(), x[v].copy()
        nodes = np.concatenate([u, v])
        newv = np.concatenate([xv, xu])
        oldv = np.concatenate([xu, xv])
        x_new = dict(self.x)
        x_new[attr] = x.copy()
        x_new[attr][nodes] = newv
        dE = 0.0
        new_state = []
        for t in self.pairs:
            stride = t.strides.get(attr, 0)
            if stride == 0:
                continue
            g = t.g.copy()
            g[nodes] += (newv - oldv) * stride
            C = t.counts_for(g)
            e = float((t.w * (C - t.T) ** 2).sum())
            dE += self._bw(t.kind) * (e - t.energy)
            new_state.append((t, g, C, e))
        two = []
        for t in self.twoway:
            if attr not in (t.a, t.b):
                continue
            C = np.bincount(x_new[t.a] * t.kb + x_new[t.b], minlength=t.cells).astype(float)
            e = float((t.w * (C - t.T) ** 2).sum())
            dE += self._bw(t.kind) * (e - t.energy)
            two.append((t, C, e))
        B = self.B
        e_old = self._walk_total(B.counts, B.P)
        B.set_codes(attr, nodes, oldv, newv, self.gstrides[attr])
        dE += self._walk_total(B.counts, B.P) - e_old
        ok = dE < 0 or (self.temperature > 0 and rng.random() < np.exp(-dE / self.temperature))
        if not ok:
            B.set_codes(attr, nodes, newv, oldv, self.gstrides[attr])
            return False
        wa = self.A.set_codes(attr, nodes, oldv, newv, self.gstrides[attr])
        self._rescore(wa)
        self.x[attr] = x_new[attr]
        for t, g, C, e in new_state:
            t.g, t.C, t.energy = g, C, e
        for t, C, e in two:
            t.C, t.energy = C, e
        return True

    # ------------------------------------------------------------------ the run
    def check(self) -> None:
        """Debug: pools, triangle counts and base counts against from-scratch recomputation."""
        for p in (self.A, self.B):
            p.check()
            assert np.array_equal(p.indices, self.indices)
        assert np.array_equal(self.tri, en.triangles(en.adjacency(self.edges, self.n)).astype(float))
        assert np.array_equal(np.bincount(self.edges.ravel(), minlength=self.n), self.deg)
        ip, ix = ws.csr(self.edges, self.n)
        rows = np.repeat(np.arange(self.n), self.deg)
        assert np.array_equal(np.sort(rows * self.n + self.indices), rows * self.n + ix)
        assert np.array_equal(self.indices[self.eslot[:, 0]], self.edges[:, 1])
        assert np.array_equal(self.indices[self.eslot[:, 1]], self.edges[:, 0])
        for t in self.pairs:
            assert np.array_equal(t.g, t.codes(self.x))


def _round_energies(d: dict) -> dict:
    return {k: {"energy": round(float(v["energy"]), 3), "cells": int(v["cells"])} for k, v in d.items()}


def _spectrum_gap(edges, n, released) -> float | None:
    if released is None:
        return None
    lam = ws.transition_spectrum(edges, n, k=len(released))
    k = min(len(lam), len(released))
    return float(((lam[:k] - np.asarray(released)[:k]) ** 2).sum())


def generate_walks(
    artifact,
    seed: int = 0,
    *,
    pool_size: int = 40000,
    val_pool_size: int | None = None,
    cycles: int = 20,
    structure_sweeps: float = 1.5,
    label_sweeps: float = 1.5,
    temperature: float = 0.0,
    batch: int = 6000,
    label_batch: int = 500,
    damping: float = 0.25,
    retries: int = 2,
    local_frac: float = 0.25,
    weights: dict | None = None,
    grad_every: int = 4,
    base_weight: float = 1.0,
    base_weights: dict | None = None,
    pool_screen: float = 0.0,
    class_frac: float = 0.5,
    class_keys: list[str] | None = None,
    share_surrogate: float = 1.0,
    chunk: int = 150,
    label_chunk: int = 60,
    init: tuple[AttributedGraph, dict] | None = None,
    check: bool = False,
    spectrum: bool = True,
    verbose: bool = False,
) -> tuple[AttributedGraph, dict]:
    """Generate a synthetic graph whose walk statistics match ``artifact.tables`` (module docstring).

    ``pool_size`` is the nominal synthetic pool (it sets the energy's noise unit) and the selection pool;
    ``val_pool_size`` the validation pool (default: the same). ``structure_sweeps`` proposals per cycle in units of
    ``m / 2`` swaps; ``label_sweeps`` matchings per attribute per cycle. ``init`` = a precomputed
    ``(graph, info)`` of ``generate_v2_with_info(artifact.base, seed=seed)`` (skips the initializer).
    ``check`` recomputes everything from scratch after every cycle (debug). ``weights`` scale tables by name.
    """
    t_start = time.time()
    if init is None:
        g0, init_info = generate_v2_with_info(artifact.base, seed=seed)
    else:
        g0, init_info = init
    init_info = {k: v for k, v in init_info.items() if k != "regions"}
    t_init = time.time() - t_start
    gen = _Generator(artifact, g0, seed, pool_size=pool_size, val_pool_size=val_pool_size, batch=batch,
                     label_batch=label_batch, damping=damping, retries=retries, temperature=temperature,
                     local_frac=local_frac, weights=weights, grad_every=grad_every, base_weight=base_weight,
                     base_weights=base_weights,
                     pool_screen=pool_screen, class_frac=class_frac, class_keys=class_keys,
                     share_surrogate=share_surrogate, chunk=chunk, label_chunk=label_chunk)
    rng = np.random.default_rng([int(seed), 101])
    released_spec = artifact.tables.get(SPECTRUM)
    info: dict = {"method": "graph_aim_walks", "seed": int(seed), "pool_size": int(pool_size),
                  "val_pool_size": int(gen.val_pool_size), "options": {
                      "cycles": cycles, "structure_sweeps": structure_sweeps, "label_sweeps": label_sweeps,
                      "temperature": temperature, "batch": batch, "label_batch": label_batch, "damping": damping,
                      "retries": retries, "local_frac": local_frac, "grad_every": grad_every,
                      "base_weight": base_weight, "base_weights": gen.base_weights, "pool_screen": pool_screen, "class_frac": class_frac,
                      "class_keys": gen.class_keys, "share_surrogate": share_surrogate,
                      "chunk": chunk, "label_chunk": label_chunk},
                  "init_seconds": t_init, "init": init_info, "cycles": [], "warnings": gen.warnings}
    if spectrum:
        info["spectrum_sq_gap_init"] = _spectrum_gap(gen.edges, gen.n, released_spec)
    stop = "cycles"
    for cyc in range(int(cycles)):
        t0 = time.time()
        gen._draw_pools(cyc)
        fresh = gen.walk_energy(gen.B.counts, gen.B.P)
        gen._start_structure_phase()
        rec = {"cycle": cyc, "fresh": _round_energies(fresh), "base": _round_energies(gen.base_energy())}
        if all(v["energy"] <= 2.0 * v["cells"] for v in fresh.values()):
            stop = "matched"
            info["cycles"].append(rec)
            break
        st = {"proposed": 0, "screened": 0, "accepted": 0, "batches_accepted": 0, "batches_rejected": 0}
        n_batches = int(np.ceil(structure_sweeps * gen.m / 2 / max(gen.batch, 1)))
        for bi in range(n_batches):
            if bi and bi % gen.grad_every == 0:
                gen._refresh_grads()
            gen._structure_batch(rng, st)
        t_struct = time.time() - t0
        gen._refresh_grads()
        for t in gen.pairs:  # the label chain sees the capped graph of the new structure
            t.set_edges(gen.edges)
            t.exact()
        sl = {"proposed": 0, "screened": 0, "accepted": 0, "batches_accepted": 0, "batches_rejected": 0}
        n_match = int(np.ceil(label_sweeps * len(gen.moving)))
        for k in range(n_match):
            attr = gen.moving[k % len(gen.moving)]
            u, v = gen._label_proposals(rng, attr)
            for s in range(0, len(u), gen.label_batch):
                gen._label_batch(rng, attr, u[s: s + gen.label_batch], v[s: s + gen.label_batch], sl)
            gen._refresh_grads()
        rec.update(structure=st, label=sl, end_selection_pool=_round_energies(gen.walk_energy(gen.A.counts, gen.A.P)),
                   end_validation_pool=_round_energies(gen.walk_energy(gen.B.counts, gen.B.P)),
                   seconds=time.time() - t0)
        rec["seconds_label"] = rec["seconds"] - t_struct
        rec["seconds_structure"] = t_struct
        if check:
            gen.check()
        info["cycles"].append(rec)
        if verbose:
            tot = sum(v["energy"] for v in fresh.values())
            print(f"cycle {cyc}: fresh walk energy {tot:.0f}, struct acc {st['accepted']}/{st['proposed']}, "
                  f"label acc {sl['accepted']}/{sl['proposed']}, {rec['seconds']:.0f}s", flush=True)
    # final energies on a fresh pool the graph was never fitted to
    final_pool = gen._pool(gen.pool_size, np.random.default_rng([int(seed), 102, 10**6]))
    for t in gen.pairs:
        t.set_edges(gen.edges)
        t.exact()
    info["final"] = _round_energies(gen.walk_energy(final_pool.counts, final_pool.P))
    info["final_base"] = _round_energies(gen.base_energy())
    if spectrum:
        info["spectrum_sq_gap_final"] = _spectrum_gap(gen.edges, gen.n, released_spec)
    info["stop"] = stop
    info["seconds"] = time.time() - t_start
    return gen.graph(), info

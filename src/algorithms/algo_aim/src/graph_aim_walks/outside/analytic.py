"""Expected lag tables of the walk process and their exact gradient in the transition matrix (outside zone).

Why: a double-edge swap re-routes the ~30 pool walks that traverse its two edges; scoring the swap by re-drawing
those walks' continuations measures mostly the noise of the continuations (``tools/walks_dev/NOTES.md``). The
``degree_lag`` and ``attr_lag`` tables, however, are linear functionals of the walk's distribution, so their
EXPECTED value and its derivative in the transition matrix ``P = D^-1 A`` have closed forms (no sampling noise).

With ``p_t = p_0 P^t`` (``p_0`` uniform over the nodes of degree >= 1), ``H`` the ``(n, G)`` one-hot matrix of a
node code and ``c_k = 1 / (T + 1 - k)``, the expected normalised lag table is
``m_k[g, h] = c_k sum_{t=0}^{T-k} sum_x p_t(x) [code(x) = g] (P^k H)[x, h]``.
For weights ``W_k`` (the energy gradient ``dE / dm_k``), ``F = sum_k <W_k, m_k>`` has
``dF / dP[x, y] = sum_{s=0}^{T-1} p_s(x) Q_s(y) + sum_{j=0}^{L-1} Lt_j[x] . R_j[y]`` with

- ``q_k(x) = sum_h W_k[code(x), h] (P^k H)[x, h]``, ``Q_s = P Q_{s+1} + sum_{k: s <= T-1-k} c_k q_k``, ``Q_T = 0``
  (the change of ``p_t``);
- ``R_j = P^j H``, ``Lambda_k[x, :] = c_k S_k(x) W_k[code(x), :]`` with ``S_k = sum_{t=0}^{T-k} p_t``,
  ``Lt_j = P^T Lt_{j+1} + Lambda_{j+1} [j + 1 is a lag]``, ``Lt_L = 0`` (the change of ``P^k``).

Both are bilinear, so :class:`LagGradient` stores ``left`` and ``right`` ``(n, D)`` matrices with
``dF / dP[x, y] = left[x] . right[y]``; a swap's first-order change is the sum over its eight changed entries of
``P``. For the label chain, ``Y_k = (P^T)^k Lambda_k`` gives the first-order change of moving one node's code.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from graph_aim_walks.transfer import walkstats as ws


def transition(edges: np.ndarray, n: int) -> tuple[sp.csr_matrix, np.ndarray]:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)
    P = sp.diags(inv) @ a
    # a node of degree 0 is never left: its row is the identity
    iso = np.flatnonzero(deg == 0)
    if len(iso):
        P = P + sp.coo_matrix((np.ones(len(iso)), (iso, iso)), shape=(n, n))
    return P.tocsr(), deg


def start_distribution(deg: np.ndarray) -> np.ndarray:
    cand = np.asarray(deg) > 0
    if not cand.any():
        cand[:] = True
    return cand / cand.sum()


def expected_lag_tables(P: sp.csr_matrix, p0: np.ndarray, spec: ws.WalkSpec, codes: np.ndarray, G: int) -> np.ndarray:
    """Expected normalised ``(L, G, G)`` lag table of ``codes`` (matches ``walkstats.normalise`` in expectation)."""
    T = spec.length
    H = np.zeros((len(codes), G))
    H[np.arange(len(codes)), codes] = 1.0
    p = [p0]
    for _ in range(T):
        p.append(P.T @ p[-1])
    out = np.zeros((len(spec.lags), G, G))
    R = H
    k_prev = 0
    for i, k in enumerate(spec.lags):
        for _ in range(k - k_prev):
            R = P @ R
        k_prev = k
        S = np.sum(p[: T - k + 1], axis=0)
        out[i] = (H * S[:, None]).T @ R / (T + 1 - k)
    return out


class LagGradient:
    """``dF / dP`` of ``F = sum_tables sum_k <W_k, m_k>`` as a bilinear form ``left[x] . right[y]``."""

    def __init__(self, P: sp.csr_matrix, p0: np.ndarray, spec: ws.WalkSpec, tables: list[tuple[np.ndarray, int, np.ndarray]],
                 with_codes: list[bool] | None = None):
        """``tables``: ``(codes (n,), G, W (L, G, G))`` per lag table; ``with_codes[i]``: also prepare the
        code-move derivatives of table ``i`` (:meth:`code_delta`, default all)."""
        with_codes = [True] * len(tables) if with_codes is None else list(with_codes)
        T = spec.length
        n = P.shape[0]
        lags = list(spec.lags)
        Lmax = max(lags)
        PT = P.T.tocsr()
        p = [p0]
        for _ in range(T):
            p.append(PT @ p[-1])
        pm = np.stack(p[:T], axis=1)  # (n, T): p_s, s = 0..T-1
        csum = np.cumsum(np.stack(p, axis=0), axis=0)  # csum[t] = sum_{u<=t} p_u
        c = {k: 1.0 / (T + 1 - k) for k in lags}
        q = {k: np.zeros(n) for k in lags}
        lefts, rights = [pm], []
        self.Y: list[dict] = []
        for ti, (codes, G, W) in enumerate(tables):
            H = np.zeros((n, G))
            H[np.arange(n), codes] = 1.0
            R = [H]
            for _ in range(Lmax):
                R.append(P @ R[-1])
            Lam = {}
            for i, k in enumerate(lags):
                Wk = W[i]
                q[k] += (Wk[codes] * R[k]).sum(axis=1)
                Lam[k] = c[k] * csum[T - k][:, None] * Wk[codes]
            Lt = [None] * (Lmax + 1)
            Lt[Lmax] = np.zeros((n, G))
            for j in range(Lmax - 1, -1, -1):
                Lt[j] = PT @ Lt[j + 1]
                if (j + 1) in Lam:
                    Lt[j] = Lt[j] + Lam[j + 1]
            lefts.append(np.concatenate(Lt[:Lmax], axis=1))
            rights.append(np.concatenate(R[:Lmax], axis=1))
            if not with_codes[ti]:
                self.Y.append(None)
                continue
            # label chain: Y = sum_k (P^T)^k Lambda_k (dF of moving a node's RIGHT-hand code), by Horner
            Y = Lam.get(Lmax, np.zeros((n, G)))
            for j in range(Lmax - 1, -1, -1):
                Y = PT @ Y
                if j in Lam:
                    Y = Y + Lam[j]
            # LEFT-hand code: dF/d(code row) = c_k S_k(x) W_k[g', :] . R_k[x, :]  ->  per-node per-code value
            Z = np.zeros((n, G))
            for i, k in enumerate(lags):
                Z += c[k] * csum[T - k][:, None] * (R[k] @ W[i].T)
            self.Y.append({"Y": Y, "Z": Z})
        Q = [np.zeros(n) for _ in range(T + 1)]
        for s in range(T - 1, -1, -1):
            b = np.zeros(n)
            for k in lags:
                if s <= T - 1 - k:
                    b += c[k] * q[k]
            Q[s] = P @ Q[s + 1] + b
        rights.insert(0, np.stack(Q[:T], axis=1))
        self.left = np.concatenate(lefts, axis=1)
        self.right = np.concatenate(rights, axis=1)

    def entries(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``dF / dP[x_i, y_i]``."""
        return np.einsum("ij,ij->i", self.left[x], self.right[y])

    def swap_delta(self, a, b, c, d, deg: np.ndarray) -> np.ndarray:
        """First-order change of ``F`` for the swaps ``(a, b), (c, d) -> (a, d), (c, b)``."""
        ia, ib, ic, id_ = (1.0 / np.maximum(deg[v], 1) for v in (a, b, c, d))
        return (ia * (self.entries(a, d) - self.entries(a, b)) + id_ * (self.entries(d, a) - self.entries(d, c))
                + ic * (self.entries(c, b) - self.entries(c, d)) + ib * (self.entries(b, c) - self.entries(b, a)))

    def code_delta(self, t: int, u, gu, gnu, v, gv, gnv) -> np.ndarray:
        """First-order change of ``F`` when node ``u``'s code in table ``t`` goes ``gu -> gnu`` and ``v``'s
        ``gv -> gnv`` (the interaction of the two moves is second order and ignored)."""
        Y, Z = self.Y[t]["Y"], self.Y[t]["Z"]
        return (Y[u, gnu] - Y[u, gu] + Z[u, gnu] - Z[u, gu]) + (Y[v, gnv] - Y[v, gv] + Z[v, gnv] - Z[v, gv])

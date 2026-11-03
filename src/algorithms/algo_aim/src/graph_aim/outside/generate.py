"""Generator: sample a synthetic attributed graph from an ``Artifact`` alone (no real data).

What each measurement controls
------------------------------
- ``attr_2way`` (+ ``attr_1way``): the attribute joint. A Chow-Liu tree is built from the mutual
  information of every released attribute pair; conditional tables ``P(child | parent)`` come from
  the same pair tables; nodes are sampled by ancestral sampling. Pairs that were not released are
  treated as independent (MI 0).
- ``degree_hist``: the target degree sequence. Counts are clamped at 0, cells below one noise sigma
  are zeroed (they are indistinguishable from noise), the histogram is rescaled to ``n`` nodes with
  systematic (stratified) rounding, which keeps the expected count of rare hub degrees (largest-
  remainder rounding drops them and shifts degree assortativity by ~0.5), and shuffled over nodes; the
  sum is made even.
- ``attr_degree`` (optional, one table per attribute): replaces the plain shuffle. Each table is raked
  to the sampled attribute-value counts and the degree-bin counts of the pool; a node-level log-linear
  model ``P(bin | attributes) ~ exp(phi_j + sum_k theta_k[a_k, j])`` is fitted so all released tables are
  matched at once (a compromise, with a reported residual, if noisy tables disagree); each node then
  gets a bin under exact bin capacities and a pool degree of that bin. The degree histogram stays
  exactly the pool. Tables whose ``bin_edges`` differ from the first one are ignored. Without any
  ``attr_degree`` the plain shuffle is used (identical to artifacts that predate the measurement).
- ``clustering_by_degree``: phase one of a BTER-style sampler. Nodes of degree >= 2 are sorted by
  degree and cut into consecutive blocks of ``d + 1`` nodes (``d`` = degree of the first node); each
  block is an Erdos-Renyi graph with density ``rho = c ** (1 / 3)``, where ``c`` is the mean local
  clustering of the block's degree bin (``values[b, 0] / max(values[b, 1], 1)``, clipped to [0, 1]).
- ``joint_degree`` and ``attr_edge_pair``: phase two. The remaining ("excess") degree of each node is
  wired with a Chung-Lu sampler whose endpoint classes are steered by a class-pair matrix. A node's
  class is (attribute group, degree bin). An artifact may carry several ``attr_edge_pair`` instances
  (one per homophily group, selected by KIND, never by name); the PRIMARY instance, the one with the
  highest lift computed from the released table itself (``sum_i q_ii / sum_i a_i^2`` on the
  normalised table), defines the group of the class. The seed of the class-pair matrix is the product
  ``edge_pair[g, h] * joint_degree[a, b]`` (group affinity times degree-bin mixing); it is fitted by
  symmetric iterative proportional fitting (IPF) so its margins equal the degree volume the
  synthetic nodes of each class actually have. This is done twice: once against the total target
  degree to get the full target, then, after subtracting what phase one already placed (clamped at
  0, with a small floor), against the excess degree of phase two. Edges are then drawn as a
  multinomial over class pairs, with endpoints inside a class chosen proportional to excess degree.
  The final degree-preserving rewiring is multi-channel: one channel is the class-pair target above,
  and every NON-primary ``attr_edge_pair`` instance adds a channel whose target is its own
  ordered-pair mass scaled to the synthetic total degree volume. The swap cost is the sum over
  channels of the linearised chi-square change, each channel normalised by its target's total mass so
  none dominates; with a single instance (old artifacts) there is one channel and the behaviour is
  unchanged. Channels share the phase-two edges, so targets that fight each other may not all be
  reached; the rewiring stops early when no proposed swap lowers the cost.

Optional latent regions (plan section 7, used by ``generate_v2``): ``generate(..., regions=Z,
within_region_frac=q)`` takes an ``(n,)`` int label per node. Phase-one blocks are cut inside each
region (sort by region, degree, group; a block closes at the region boundary, so every member still
has degree >= the block's ``d``), a fraction ``q`` of the phase-two draw is made inside regions (the
class-pair cell and first endpoint as usual, the partner from its degree bin inside the first endpoint's
region, proportional to excess degree, unless the demand exceeds half that pool), and the rewiring
keeps the number of within-region edges (proposals are paired within regions). The defaults leave the
sampler byte-identical.

Matrix conventions assumed (see the ``Measurement`` docstring, which does not pin them down):
``joint_degree`` and ``attr_edge_pair`` are symmetric, off-diagonal cell ``[a, b] == [b, a]`` is the
number of edges between the two classes, and the diagonal ``[a, a]`` is the number of edges inside
the class (counted once).

Known simplifications
---------------------
- Degree depends on attributes only through the released ``attr_degree`` tables (per bin, one
  attribute at a time, combined by a log-linear model); without them degree is independent of
  attributes and the IPF step has to absorb the mismatch in group degree volume. With them, phase
  two's class-pair step sees the true group degree volumes.
- The group and degree-bin effects on edges are combined as a product (no 3-way interaction).
- Phase-one blocks ignore attributes, so group homophily is only imposed in phase two (compensated
  by the subtraction described above, which is exact only if phase one does not overshoot).
- Only the primary attr_edge_pair group steers the initial class-based sampling (a product over all
  groups would explode the class count); the other groups act through the rewiring channels only.
- Duplicates / self loops from Chung-Lu are removed and re-drawn in a few top-up rounds; very high
  degree nodes can still end slightly below their target degree.
- No clipping of noise beyond clamping at 0 and the sigma threshold on ``degree_hist``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graph_aim.transfer.binning import check_bin_edges, degree_bin
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement

_SMOOTH = 1e-3  # mass mixed into every attribute-pair cell so noisy zeros stay samplable
_REGION_POOL_FRAC = 0.5  # a within-region draw stays local only if its expected demand fits in this share of the pool


# --------------------------------------------------------------------------------------- helpers
def _by_kind(artifact: Artifact, kind: str) -> list[Measurement]:
    return [m for m in artifact.measurements.values() if m.kind == kind]


def _clean(values: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, 0.0, None)


def _normalise(v: np.ndarray) -> np.ndarray:
    """Probability table from clamped counts; uniform if there is no mass."""
    s = v.sum()
    if s <= 0:
        return np.full(v.shape, 1.0 / v.size)
    return v / s


def _symmetrise(m: np.ndarray) -> np.ndarray:
    m = _clean(m)
    return (m + m.T) / 2.0


def _ordered_mass(edge_counts: np.ndarray) -> np.ndarray:
    """Edge-count matrix (diagonal counted once) -> ordered-pair mass (diagonal doubled)."""
    q = _symmetrise(edge_counts).copy()
    q[np.diag_indices_from(q)] *= 2.0
    return q


def _bin_of(deg: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    b = np.searchsorted(bin_edges, deg, side="right") - 1
    return np.clip(b, 0, len(bin_edges) - 2)


def _ipf_sym(seed: np.ndarray, margins: np.ndarray, iters: int = 300) -> np.ndarray:
    """Symmetric IPF: return ``x_i seed_ij x_j`` whose row sums approximate ``margins``."""
    k = len(margins)
    seed = np.asarray(seed, dtype=float).copy()
    active = margins > 0
    if not active.any():
        return np.zeros((k, k))
    seed[~active, :] = 0.0
    seed[:, ~active] = 0.0
    seed[np.ix_(active, active)] += 1e-6 * max(seed.mean(), 1e-12)  # keep the fit feasible
    x = np.where(active, 1.0, 0.0)
    for _ in range(iters):
        s = x * (seed @ x)
        ratio = np.where(s > 0, margins / np.where(s > 0, s, 1.0), 1.0)
        x = x * np.sqrt(ratio)
    return x[:, None] * seed * x[None, :]


# ------------------------------------------------------------------------------------- attributes
def _pair_tables(artifact: Artifact) -> dict[tuple[str, str], np.ndarray]:
    """Joint probability tables keyed by (col_a, col_b) in the orientation they were released."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for m in _by_kind(artifact, "attr_2way"):
        a, b = m.params["attrs"]
        v = _clean(m.values)
        p = _normalise(v)
        p = (1 - _SMOOTH) * p + _SMOOTH / p.size
        out[(a, b)] = p
    return out


def _sample_attributes(artifact: Artifact, n: int, rng: np.random.Generator) -> pd.DataFrame:
    domains = artifact.schema.domains
    cols = list(domains)
    sizes = {c: len(domains[c]) for c in cols}
    pairs = _pair_tables(artifact)

    one_way = {}
    for m in _by_kind(artifact, "attr_1way"):
        one_way[m.params["attrs"][0]] = _normalise(_clean(m.values))

    def joint(a: str, b: str) -> np.ndarray | None:
        if (a, b) in pairs:
            return pairs[(a, b)]
        if (b, a) in pairs:
            return pairs[(b, a)].T
        return None

    def marginal(c: str) -> np.ndarray:
        if c in one_way and one_way[c].shape == (sizes[c],):
            return one_way[c]
        for o in cols:
            j = joint(c, o) if o != c else None
            if j is not None:
                return j.sum(axis=1)
        return np.full(sizes[c], 1.0 / sizes[c])

    # mutual information on every pair (0 when the pair was not released)
    k = len(cols)
    mi = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            p = joint(cols[i], cols[j])
            if p is None:
                continue
            outer = np.outer(p.sum(axis=1), p.sum(axis=0))
            mi[i, j] = mi[j, i] = float(np.sum(p * np.log(p / outer)))

    # Prim maximum spanning tree rooted at column 0 (ties / disconnected: MI 0 edges)
    parent = [-1] * k
    in_tree = [0]
    rest = set(range(1, k))
    while rest:
        best = max(((mi[i, j], i, j) for i in in_tree for j in rest), key=lambda t: t[0])
        _, i, j = best
        parent[j] = i
        in_tree.append(j)
        rest.remove(j)

    data: dict[str, np.ndarray] = {}
    for idx in in_tree:
        c = cols[idx]
        if parent[idx] < 0:
            p = marginal(c)
            data[c] = rng.choice(sizes[c], size=n, p=p / p.sum())
            continue
        pc = cols[parent[idx]]
        j = joint(pc, c)
        if j is None:
            p = marginal(c)
            data[c] = rng.choice(sizes[c], size=n, p=p / p.sum())
            continue
        cond = j / np.maximum(j.sum(axis=1, keepdims=True), 1e-300)
        cdf = np.cumsum(cond, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random(n)
        codes = (u[:, None] > cdf[data[pc]]).sum(axis=1)
        data[c] = np.minimum(codes, sizes[c] - 1)
    df = pd.DataFrame({c: data[c].astype(np.int64) for c in cols})
    df.index = pd.RangeIndex(n)
    return df


# --------------------------------------------------------------------------------------- degrees
def _attr_degree_tables(
    artifact: Artifact, attrs: pd.DataFrame
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]] | None:
    """Usable ``attr_degree`` tables as ``(bin_edges, [(attr, cleaned (K_a, B) counts), ...])``.

    The bin edges of the first usable table are the reference; a table with different edges, a shape that
    does not match its attribute's domain and edges, or an attribute that is not sampled is skipped
    (never re-binned: all tables of one artifact share the public edges). ``None`` if nothing is usable.
    """
    edges: np.ndarray | None = None
    out: list[tuple[str, np.ndarray]] = []
    for m in _by_kind(artifact, "attr_degree"):
        names = list(m.params.get("attrs", []))
        try:
            e = check_bin_edges(m.params["bin_edges"])
        except (KeyError, ValueError):
            continue
        if len(names) != 1 or names[0] not in attrs.columns or names[0] not in artifact.schema.domains:
            continue
        v = np.asarray(m.values, dtype=float)
        if v.shape != (len(artifact.schema.domains[names[0]]), len(e) - 1):
            continue
        if edges is None:
            edges = e
        elif not np.array_equal(e, edges):
            continue
        out.append((names[0], _clean(v)))
    if not out:
        return None
    return edges, out


def _rake(table: np.ndarray, rows: np.ndarray, cols: np.ndarray, iters: int = 200) -> np.ndarray:
    """IPF of a cleaned count table to the given row and column sums (equal totals).

    Rows or columns with zero target are zeroed; a row with no mass in the allowed columns becomes uniform
    over them, so the fit is always feasible.
    """
    t = np.where((rows > 0)[:, None] & (cols > 0)[None, :], table, 0.0)
    empty = (t.sum(axis=1) <= 0) & (rows > 0)
    t[empty] = (cols > 0).astype(float)
    t = t + 1e-9 * max(t.mean(), 1e-12) * ((rows > 0)[:, None] & (cols > 0)[None, :])
    for _ in range(iters):
        rs = t.sum(axis=1)
        t = t * np.where(rs > 0, rows / np.where(rs > 0, rs, 1.0), 0.0)[:, None]
        cs = t.sum(axis=0)
        t = t * np.where(cs > 0, cols / np.where(cs > 0, cs, 1.0), 0.0)[None, :]
    return t


def _assign_degrees_by_attributes(
    pool: np.ndarray,
    attrs: pd.DataFrame,
    tables: tuple[np.ndarray, list[tuple[str, np.ndarray]]],
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Give the degree pool to nodes so degree-bin counts per attribute value follow the tables.

    1. ``c_j``: pool degrees per bin. Each table is raked to the sampled value counts (rows) and ``c_j``
       (columns), giving targets ``R_k[v, j]`` with exactly consistent margins.
    2. Node model ``P[i, j] ~ exp(phi_j + sum_k theta_k[a_k(i), j])`` fitted by damped iterative scaling
       (cyclic over tables, then ``phi``) so the expected tables match ``R_k`` and the bin totals ``c_j``.
       Mutually inconsistent noisy tables give a compromise; ``info["residual"]`` is the largest
       ``|expected - R_k| / max(R_k, 1)`` (also over the bin totals).
    3. Capacity-exact assignment on the table ``n_type * P`` of attribute-code types: scale to exact
       margins, round to integers keeping them (random placement of the leftover units), deal each type's
       nodes to the bins by random permutation, then deal the pool degrees of each bin to its nodes by
       random permutation.

    The output is a permutation of ``pool``. ``info`` has ``residual``, ``iterations``, ``attrs``,
    ``bin_edges``, ``targets`` and ``achieved`` (per-table ``(K_a, B)`` arrays; ``achieved`` is the table
    of the actual assignment).
    """
    bin_edges, tabs = tables
    n = len(pool)
    nb = len(bin_edges) - 1
    pbin = degree_bin(pool, bin_edges)
    c = np.bincount(pbin, minlength=nb).astype(float)
    act = np.flatnonzero(c > 0)
    codes = [attrs[a].to_numpy().astype(np.int64) for a, _ in tabs]
    targets = []
    for (a, t), code in zip(tabs, codes):
        r = np.bincount(code, minlength=t.shape[0]).astype(float)
        targets.append(_rake(t, r, c))

    # --- node-level fit over the active bins
    ca = c[act]
    logc = np.log(ca)
    theta = [np.zeros((t.shape[0], len(act))) for _, t in tabs]
    phi = logc.copy()

    def probs() -> np.ndarray:
        z = phi[None, :] + sum(th[code] for th, code in zip(theta, codes))
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def expected(p: np.ndarray, code: np.ndarray, k: int) -> np.ndarray:
        e = np.zeros((k, len(act)))
        np.add.at(e, code, p)
        return e

    def residual(p: np.ndarray) -> float:
        res = np.abs(p.sum(axis=0) - ca) / np.maximum(ca, 1.0)
        for r_k, code in zip(targets, codes):
            e = expected(p, code, r_k.shape[0])
            res = np.append(res, np.abs(e - r_k[:, act]) / np.maximum(r_k[:, act], 1.0))
        return float(res.max()) if res.size else 0.0

    eps = 1e-12
    it = 0
    p = probs()
    for it in range(1, max_iter + 1):
        for k, (r_k, code) in enumerate(zip(targets, codes)):
            e = expected(p, code, r_k.shape[0])
            theta[k] = np.clip(theta[k] + 0.8 * np.log((r_k[:, act] + eps) / (e + eps)), -30.0, 30.0)
            p = probs()
        phi = phi + 0.8 * np.log((ca + eps) / (p.sum(axis=0) + eps))
        p = probs()
        if residual(p) < tol:
            break
    res = residual(p)

    # --- capacity-exact assignment (columns of p index the active bins)
    # Nodes with the same attribute codes have the same P row, so the assignment is made on the table
    # E[type, bin] = n_type * P[type, bin]: scaled to the exact margins (type sizes, c_j), rounded to
    # integers with those margins kept (floor, then the leftover units are placed one by one at random
    # with probability proportional to the fractional parts, only where both the row and the column still
    # have room), and the nodes of each type are dealt out to the bins by a random permutation.
    # Tested against drawing the bins node by node with a Gumbel top-c_j per bin (bins in random order):
    # that one left errors of ~3% of n on tables with structural zeros because late bins only see the
    # nodes the early bins did not take; the rounding is within one unit per cell of the fitted table.
    key = np.stack(codes, axis=1)
    _, first, tid = np.unique(key, axis=0, return_index=True, return_inverse=True)
    tid = tid.ravel()
    nt = np.bincount(tid).astype(float)
    ex = p[first] * nt[:, None]
    for _ in range(50):  # make the margins exact before rounding
        ex = ex * (nt / np.maximum(ex.sum(axis=1), 1e-300))[:, None]
        ex = ex * (ca / np.maximum(ex.sum(axis=0), 1e-300))[None, :]
    cnt = np.maximum(np.floor(ex - 1e-9), 0.0)
    row_left = nt - cnt.sum(axis=1)
    col_left = ca - cnt.sum(axis=0)
    frac = ex - cnt
    while row_left.sum() > 0.5:
        room = (row_left > 0.5)[:, None] & (col_left > 0.5)[None, :]
        w = np.where(room, frac + 1e-12, 0.0).ravel()
        cell = rng.choice(w.size, p=w / w.sum())
        t_, j_ = divmod(cell, len(act))
        cnt[t_, j_] += 1.0
        frac[t_, j_] = 0.0
        row_left[t_] -= 1.0
        col_left[j_] -= 1.0
    assign = np.empty(n, dtype=np.int64)
    by_type = np.split(np.argsort(tid, kind="stable"), np.cumsum(nt)[:-1].astype(np.int64))
    for t_, members in enumerate(by_type):
        assign[rng.permutation(members)] = np.repeat(act, cnt[t_].astype(np.int64))

    deg = np.empty(n, dtype=pool.dtype)
    for j in act:
        nodes = np.flatnonzero(assign == j)
        deg[nodes] = rng.permutation(pool[pbin == j])
    achieved = [
        np.bincount(code * nb + assign, minlength=t.shape[0] * nb).reshape(t.shape[0], nb).astype(float)
        for (_, t), code in zip(tabs, codes)
    ]
    info = {
        "residual": res,
        "iterations": it,
        "attrs": [a for a, _ in tabs],
        "bin_edges": bin_edges,
        "targets": targets,
        "achieved": achieved,
    }
    return deg, info


def _sample_degrees(
    artifact: Artifact, n: int, rng: np.random.Generator, attrs: pd.DataFrame | None = None
) -> np.ndarray:
    hs = _by_kind(artifact, "degree_hist")
    if not hs:
        raise ValueError("artifact has no degree_hist measurement")
    h = _clean(hs[0].values)
    # fit already thresholds a coarse-grid histogram and spreads it over degrees (params["hist_edges"])
    if hs[0].sigma and hs[0].sigma > 0 and "hist_edges" not in hs[0].params:
        thresholded = np.where(h < hs[0].sigma, 0.0, h)  # cells below one sigma are noise
        h = thresholded if thresholded.sum() > 0 else h
    if h.sum() <= 0:
        raise ValueError("degree_hist has no positive mass after clamping")
    exact = h / h.sum() * n
    # systematic (stratified) rounding: every degree gets floor or ceil of its expected count with
    # probability equal to the fractional part and the total is exactly n. Largest-remainder rounding
    # would silently drop the hub tail, whose expected counts are far below one per degree, and hubs
    # dominate degree assortativity.
    pos = rng.random() + np.arange(n)
    deg = np.minimum(np.searchsorted(np.cumsum(exact), pos, side="left"), len(h) - 1)
    tables = _attr_degree_tables(artifact, attrs) if attrs is not None else None
    if tables is None:
        rng.shuffle(deg)  # no attr_degree released: degree independent of attributes
        deg = np.minimum(deg, max(n - 1, 0))
    else:
        deg = np.minimum(deg, max(n - 1, 0))
        deg, _ = _assign_degrees_by_attributes(deg, attrs, tables, rng)
    if deg.sum() % 2 == 1:
        cand = np.flatnonzero(deg > 0)
        if len(cand):
            deg[rng.choice(cand)] -= 1
    return deg


# ------------------------------------------------------------------------------------ structure
def _clustering_target(artifact: Artifact) -> tuple[np.ndarray, np.ndarray]:
    """(bin_edges, mean local clustering c per bin). Zero everywhere when not released."""
    ms = _by_kind(artifact, "clustering_by_degree")
    if not ms:
        return np.array([0.0, np.inf]), np.zeros(1)
    m = ms[0]
    v = np.nan_to_num(np.asarray(m.values, dtype=float))
    c = np.clip(v[:, 0] / np.maximum(v[:, 1], 1.0), 0.0, 1.0)
    return np.asarray(m.params["bin_edges"], dtype=float), c


def _mean_clustering_by_bin(edges: np.ndarray, n: int, bin_edges: np.ndarray) -> np.ndarray:
    """Mean local clustering of the nodes in each degree bin (NaN for empty bins)."""
    nb = len(bin_edges) - 1
    if len(edges) == 0:
        return np.full(nb, np.nan)
    a = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    a = a + a.T
    deg = np.asarray(a.sum(axis=1)).ravel()
    tri = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2.0
    pairs = deg * (deg - 1) / 2.0
    c = np.where(pairs > 0, tri / np.maximum(pairs, 1.0), 0.0)
    b = _bin_of(deg, bin_edges)
    cnt = np.bincount(b, minlength=nb)
    return np.where(cnt > 0, np.bincount(b, weights=c, minlength=nb) / np.maximum(cnt, 1), np.nan)


def _phase_one_blocks(
    deg: np.ndarray,
    grp: np.ndarray,
    rng: np.random.Generator,
    regions: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Membership of the phase-one blocks (each an int array of nodes, first member = lowest degree).

    Nodes of degree >= 2 are shuffled, then sorted by (region, degree, group); consecutive blocks of
    ``d + 1`` nodes are cut, ``d`` = degree of the block's first node, so every member has degree >= d.
    With ``regions`` (an ``(n,)`` int label per node) a block is closed at the region boundary,
    ``size = min(d + 1, nodes left in the region)``, so no block crosses a region and the invariant holds
    (the last block of a region is simply smaller; a hub whose ``d + 1`` exceeds its region gets the whole
    region). Blocks of fewer than 3 nodes are dropped. ``regions=None`` is the single-region case."""
    nodes = np.flatnonzero(deg >= 2)
    nodes = nodes[rng.permutation(len(nodes))]
    if regions is None:
        nodes = nodes[np.lexsort((grp[nodes], deg[nodes]))]
        ends = np.full(len(nodes), len(nodes), dtype=np.int64)
    else:
        nodes = nodes[np.lexsort((grp[nodes], deg[nodes], regions[nodes]))]
        reg = regions[nodes]
        ends = np.searchsorted(reg, reg, side="right")  # end of the current region in the sorted order
    blocks = []
    pos = 0
    while pos < len(nodes):
        d = int(deg[nodes[pos]])
        size = min(d + 1, int(ends[pos]) - pos)
        if size >= 3:
            blocks.append(nodes[pos : pos + size])
        pos += size
    return blocks


def _phase_one(
    deg: np.ndarray,
    edges_b: np.ndarray,
    rho_b: np.ndarray,
    grp: np.ndarray,
    rng: np.random.Generator,
    regions: np.ndarray | None = None,
) -> np.ndarray:
    """Affinity blocks of similar-degree nodes (ties broken by attribute group); (m1, 2) edges.

    Package E (plan section 7) tried the sort (primary group, then degree) instead: within-group
    triangle share and homophily improved a lot, but blocks at group boundaries mix degrees, so the
    block size (first node's degree + 1) no longer matches its members: degree KS, clustering,
    transitivity and the edge count got worse beyond the seed spread. Dropped (tools/exp_e). The
    ``regions`` option (plan section 7.2) keeps the degree invariant instead: blocks are cut inside
    each region (see ``_phase_one_blocks``)."""
    out = []
    for members in _phase_one_blocks(deg, grp, rng, regions):
        d = int(deg[members[0]])
        size = len(members)
        rho = rho_b[_bin_of(np.array([d]), edges_b)[0]]
        if rho > 0:
            iu, ju = np.triu_indices(size, k=1)
            keep = rng.random(len(iu)) < rho
            a, b = members[iu[keep]], members[ju[keep]]
            out.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def _edge_pair_lift(values: np.ndarray) -> float:
    """Lift of a released ``attr_edge_pair`` table: ``sum_i q_ii / sum_i a_i^2`` on the normalised
    ordered-pair mass ``q`` (``a_i`` = endpoint share of group i). 0.0 for a table without mass."""
    q = _ordered_mass(values)
    total = q.sum()
    if total <= 0:
        return 0.0
    q = q / total
    a = q.sum(axis=1)
    denom = float((a**2).sum())
    return float(np.trace(q) / denom) if denom > 0 else 0.0


def _primary_edge_pair(artifact: Artifact) -> Measurement | None:
    """The ``attr_edge_pair`` instance with the highest lift computed from the released table itself
    (needs no outside knowledge; ties keep artifact order). Selected by KIND, never by name, so one
    legacy instance named exactly ``attr_edge_pair`` behaves as before."""
    ms = _by_kind(artifact, "attr_edge_pair")
    if not ms:
        return None
    return max(ms, key=lambda m: _edge_pair_lift(m.values))


def _attr_codes(attrs: pd.DataFrame, domains: dict[str, list[str]], cols: list[str]) -> tuple[np.ndarray, int]:
    """Row-major group code of each node over ``cols`` and the number of groups."""
    sizes = [len(domains[c]) for c in cols]
    g = np.zeros(len(attrs), dtype=np.int64)
    for c, s in zip(cols, sizes):
        g = g * s + attrs[c].to_numpy()
    return g, int(np.prod(sizes))


def _group_codes(artifact: Artifact, attrs: pd.DataFrame, n: int) -> tuple[np.ndarray, int]:
    m = _primary_edge_pair(artifact)
    if m is None:
        return np.zeros(n, dtype=np.int64), 1
    return _attr_codes(attrs, artifact.schema.domains, list(m.params["attrs"]))


def _edge_key(e: np.ndarray, n: int) -> np.ndarray:
    return e[:, 0] * n + e[:, 1]


def _class_counts(edges: np.ndarray, cls: np.ndarray, k: int) -> np.ndarray:
    """Ordered class-pair edge counts (an edge inside one class adds 2 to the diagonal)."""
    c = np.zeros((k, k))
    if len(edges):
        np.add.at(c, (cls[edges[:, 0]], cls[edges[:, 1]]), 1.0)
        np.add.at(c, (cls[edges[:, 1]], cls[edges[:, 0]]), 1.0)
    return c


def _rewire(
    edges: np.ndarray,
    movable: np.ndarray,
    channels: list[tuple[np.ndarray, np.ndarray]],
    n: int,
    rng: np.random.Generator,
    rounds: int = 40,
    regions: np.ndarray | None = None,
) -> np.ndarray:
    """Degree-preserving double-edge swaps that pull class-pair edge counts toward per-channel targets.

    ``channels`` is a list of ``(cls_k, target_k)``: per-node class codes and the ``G_k x G_k``
    ordered-pair target of the channel (channel 0 is the combined class-pair target, further channels
    come from extra ``attr_edge_pair`` groups). Only edges flagged in ``movable`` (phase two) are
    swapped, so the phase-one blocks, which carry the triangles, are left intact. A swap
    ``(a,b),(c,d) -> (a,d),(c,b)`` is applied when it lowers the summed cost
    ``sum_k m_0/m_k sum_ij (C_ij - T_ij)^2 / (T_ij + 1)`` (``m_k`` = total mass of ``target_k``: each
    channel is normalised by its own mass so none dominates, and the common factor ``m_0`` keeps the
    single-channel cost, and hence the output, exactly what it was before channels existed). Loops and
    duplicate edges are never created; each channel keeps its own ``G_k x G_k`` count matrix. Every
    round proposes a batch of disjoint swaps and the loop stops early when no swap lowers the cost.
    Uses only artifact-derived targets. With ``regions`` (plan section 7, an ``(n,)`` label per node) a
    swap is only applied when it keeps the number of within-region edges, so the region glue drawn in
    phase two survives the rewiring; ``regions=None`` leaves the behaviour exactly as before.
    """
    channels = [(c, t) for c, t in channels if t.shape[0] >= 2]
    mov_idx = np.flatnonzero(movable)
    if len(mov_idx) < 4 or not channels:
        return edges
    edges = edges.copy()
    masses = [float(t.sum()) for _, t in channels]
    ref = masses[0] if masses[0] > 0 else 1.0
    scales = [ref / m if m > 0 else 1.0 for m in masses]  # exactly 1.0 for channel 0
    ws = [1.0 / (t + 1.0) for _, t in channels]
    cnts = [_class_counts(edges, c, t.shape[0]) for c, t in channels]
    keys = np.sort(_edge_key(edges, n))
    if regions is not None:
        # stratum of a movable edge: its region when both ends share one, -1 when it crosses regions. Proposals
        # pair edges of the same stratum (otherwise only about 1/R of random pairs could keep the within-region
        # count); the strata are invariant under the accepted swaps.
        stratum = np.where(regions[edges[mov_idx, 0]] == regions[edges[mov_idx, 1]], regions[edges[mov_idx, 0]], -1)
    for _ in range(rounds):
        if regions is None:
            nb = len(mov_idx) // 2
            perm = rng.permutation(mov_idx)
            i, j = perm[:nb], perm[nb : 2 * nb]
        else:
            order_ = np.lexsort((rng.random(len(mov_idx)), stratum))  # random order inside each stratum
            p_, s_ = mov_idx[order_], stratum[order_]
            half = len(p_) // 2
            same = s_[0 : 2 * half : 2] == s_[1 : 2 * half : 2]
            i, j = p_[0 : 2 * half : 2][same], p_[1 : 2 * half : 2][same]
            nb = len(i)
        a, b = edges[i, 0], edges[i, 1]
        flip = rng.random(nb) < 0.5
        c = np.where(flip, edges[j, 1], edges[j, 0])
        d = np.where(flip, edges[j, 0], edges[j, 1])
        ok = (a != d) & (c != b) & (a != c) & (b != d)
        if regions is not None:  # keep the within-region edge count
            same_old = (regions[a] == regions[b]).astype(np.int64) + (regions[c] == regions[d]).astype(np.int64)
            same_new = (regions[a] == regions[d]).astype(np.int64) + (regions[c] == regions[b]).astype(np.int64)
            ok &= same_old == same_new
        n1 = np.stack([np.minimum(a, d), np.maximum(a, d)], axis=1)
        n2 = np.stack([np.minimum(c, b), np.maximum(c, b)], axis=1)
        k1, k2 = _edge_key(n1, n), _edge_key(n2, n)
        ok &= k1 != k2
        # the same new edge proposed twice in one batch would become a duplicate: drop both
        _, inv, cnts_ = np.unique(np.concatenate([k1, k2]), return_inverse=True, return_counts=True)
        dup = cnts_[inv] > 1
        ok &= ~(dup[:nb] | dup[nb:])
        for kk in (k1, k2):
            pos = np.clip(np.searchsorted(keys, kk), 0, len(keys) - 1)
            ok &= keys[pos] != kk
        ends = [(cls[a], cls[b], cls[c], cls[d]) for cls, _ in channels]
        # linearised cost change over the 8 touched ordered cells per channel (removals -1, additions +1)
        delta = np.zeros(nb)
        for (_, target), w, cnt, scale, (ca, cb, cc, cd) in zip(channels, ws, cnts, scales, ends):
            for (x, y, s) in ((ca, cb, -1.0), (cc, cd, -1.0), (ca, cd, 1.0), (cc, cb, 1.0)):
                for (p, q) in ((x, y), (y, x)):
                    delta += scale * w[p, q] * (2.0 * (cnt[p, q] - target[p, q]) * s + 1.0)
        ok &= delta < -1e-9
        sel = np.flatnonzero(ok)
        if len(sel) == 0:
            break
        # accept at most one swap per class-pair neighbourhood only loosely: damp the batch so the
        # linearisation stays valid when many accepted swaps hit the same cell
        if len(sel) > 0.25 * nb:
            sel = rng.choice(sel, size=max(int(0.25 * nb), 1), replace=False)
        for cnt, (ca, cb, cc, cd) in zip(cnts, ends):
            for (x, y, s) in ((ca, cb, -1.0), (cc, cd, -1.0), (ca, cd, 1.0), (cc, cb, 1.0)):
                np.add.at(cnt, (x[sel], y[sel]), s)
                np.add.at(cnt, (y[sel], x[sel]), s)
        edges[i[sel]] = n1[sel]
        edges[j[sel]] = n2[sel]
        keys = np.sort(_edge_key(edges, n))
    return edges


def _generate_structure(
    artifact: Artifact,
    deg: np.ndarray,
    attrs: pd.DataFrame,
    rng: np.random.Generator,
    cl_edges: np.ndarray,
    rho: np.ndarray,
    rewire_rounds: int = 40,
    regions: np.ndarray | None = None,
    within_region_frac: float = 0.0,
) -> np.ndarray:
    """Phase one (blocks) + phase two (class-pair Chung-Lu, top-up, repair, rewiring); ``(m, 2)`` edges.

    ``regions`` (plan section 7, optional): an ``(n,)`` int label per node. Phase-one blocks never cross a
    region, and a fraction ``within_region_frac`` (``q``) of the phase-two draw is made inside regions
    (class-pair cell and first endpoint as in the class-pair draw, the partner from its degree bin inside
    the first endpoint's region, proportional to excess degree, so expected degrees are preserved as in the
    degree-corrected SBM); the rest is the class-pair draw. Duplicate / top-up / repair steps are
    unchanged; the rewiring keeps the within-region edge count. With ``regions=None`` and ``q = 0`` the
    random stream and the output are exactly those of the original sampler."""
    n = len(deg)
    grp0, _ = _group_codes(artifact, attrs, n)
    e1 = _phase_one(deg, cl_edges, rho, grp0, rng, regions)
    deg1 = np.bincount(e1.ravel(), minlength=n) if len(e1) else np.zeros(n, dtype=np.int64)
    w = np.maximum(deg - deg1, 0).astype(float)
    m2 = int(round(w.sum() / 2))
    if m2 == 0:
        return e1

    # node classes: (attribute group, degree bin)
    grp, n_grp = _group_codes(artifact, attrs, n)
    jd = _by_kind(artifact, "joint_degree")
    if jd:
        jedges = np.asarray(jd[0].params["bin_edges"], dtype=float)
        dbin = _bin_of(deg, jedges)
        n_bin = len(jedges) - 1
        jq = _ordered_mass(jd[0].values)
        if jq.shape != (n_bin, n_bin):
            raise ValueError("joint_degree shape does not match its bin_edges")
    else:
        dbin, n_bin, jq = np.zeros(n, dtype=np.int64), 1, np.ones((1, 1))
    ep = _by_kind(artifact, "attr_edge_pair")
    primary = _primary_edge_pair(artifact)
    if primary is not None:
        gq = _ordered_mass(primary.values)
        if gq.shape != (n_grp, n_grp):
            raise ValueError("attr_edge_pair shape does not match attribute domains")
    else:
        gq = np.ones((1, 1))
    cls = grp * n_bin + dbin
    k = n_grp * n_bin
    seed = np.kron(gq, jq)  # index (g * n_bin + a, h * n_bin + b)

    # full target class-pair mass (ordered), then what remains after phase one
    total_vol = np.bincount(cls, weights=deg.astype(float), minlength=k)
    target = _ipf_sym(seed, total_vol)
    placed = np.zeros((k, k))
    if len(e1):
        np.add.at(placed, (cls[e1[:, 0]], cls[e1[:, 1]]), 1.0)
        np.add.at(placed, (cls[e1[:, 1]], cls[e1[:, 0]]), 1.0)
    remain = np.maximum(target - placed, 0.0) + 0.05 * target
    vol2 = np.bincount(cls, weights=w, minlength=k)
    mass = _ipf_sym(remain, vol2)

    # cell probabilities over unordered class pairs
    iu, ju = np.triu_indices(k)
    cell = mass[iu, ju] * np.where(iu == ju, 0.5, 1.0)
    if cell.sum() <= 0:
        cell = vol2[iu] * vol2[ju]  # degenerate: plain Chung-Lu
    p_cell = cell / cell.sum()

    # nodes sorted by class with cumulative excess-degree weights for in-class sampling
    order = np.argsort(cls, kind="stable")
    cw = np.cumsum(w[order])
    cls_sorted = cls[order]
    start_idx = np.searchsorted(cls_sorted, np.arange(k), side="left")
    end_idx = np.searchsorted(cls_sorted, np.arange(k), side="right")
    cw0 = np.concatenate([[0.0], cw])
    c_lo = cw0[start_idx]
    c_tot = cw0[end_idx] - c_lo

    def draw_classes(count: int) -> np.ndarray:
        cells = np.repeat(np.arange(len(p_cell)), rng.multinomial(count, p_cell))
        ends = []
        for cc in (iu[cells], ju[cells]):
            r = c_lo[cc] + rng.random(len(cc)) * c_tot[cc]
            idx = np.searchsorted(cw, r, side="right")
            idx = np.clip(idx, start_idx[cc], np.maximum(end_idx[cc] - 1, 0))
            ends.append(order[np.minimum(idx, n - 1)])
        a, b = ends
        e = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
        return e[e[:, 0] != e[:, 1]]

    # within-region draw (plan section 7.3): the class-pair cell and the first endpoint are drawn exactly as
    # above (so the joint_degree x attr_edge_pair steering of the first endpoint is kept); the partner is
    # drawn INSIDE the first endpoint's region from the partner class's DEGREE BIN, proportional to excess
    # degree (nodes sorted by (degree bin, region)): the degree mixing of the within-region edges follows
    # joint_degree, their attribute mixing is left to the rewiring and the label chain (pools per
    # (class, region) would be too small to give a node enough distinct partners). A draw stays local only
    # when the node's expected number of edges into the partner's degree bin (its excess degree times the
    # bin's share of its class's phase-two mass) fits in half the bin's node count inside its region: the
    # largest hubs of an ego-network graph, and the edges of mid-degree nodes into the sparse hub bin, are
    # wired globally instead of being lost to duplicates (which would skew the degree mixing).
    q_within = float(within_region_frac) if regions is not None else 0.0
    if q_within > 0:
        reg = np.asarray(regions, dtype=np.int64)
        n_reg = int(reg.max()) + 1 if len(reg) else 0
        key = dbin * n_reg + reg
        cnt2 = np.bincount(key, minlength=n_bin * n_reg).astype(float)  # nodes per (degree bin, region)
        order2 = np.argsort(key, kind="stable")
        cw2 = np.cumsum(w[order2])
        key_sorted = key[order2]
        start2 = np.searchsorted(key_sorted, np.arange(n_bin * n_reg), side="left")
        end2 = np.searchsorted(key_sorted, np.arange(n_bin * n_reg), side="right")
        cw20 = np.concatenate([[0.0], cw2])
        lo2 = cw20[start2]
        tot2 = cw20[end2] - lo2  # excess-degree volume per (degree bin, region)
        bin_share = _symmetrise(mass).reshape(k, n_grp, n_bin).sum(axis=1)  # (class, partner degree bin)
        bin_share = bin_share / np.maximum(bin_share.sum(axis=1, keepdims=True), 1e-300)

    def draw_regions(count: int) -> np.ndarray:
        cells = np.repeat(np.arange(len(p_cell)), rng.multinomial(count, p_cell))
        flip = rng.random(len(cells)) < 0.5  # which endpoint of the cell is pulled into the other's region
        ca = np.where(flip, ju[cells], iu[cells])
        cb = np.where(flip, iu[cells], ju[cells])
        r = c_lo[ca] + rng.random(len(ca)) * c_tot[ca]
        idx = np.searchsorted(cw, r, side="right")
        idx = np.clip(idx, start_idx[ca], np.maximum(end_idx[ca] - 1, 0))
        a = order[np.minimum(idx, n - 1)]
        kk = (cb % n_bin) * n_reg + reg[a]
        demand = w[a] * bin_share[cls[a], cb % n_bin]
        local = (tot2[kk] > 0) & (demand <= _REGION_POOL_FRAC * cnt2[kk])
        u = rng.random(len(cb))
        # partner inside the region (local) or anywhere in its class (global fallback)
        r_loc = lo2[kk] + u * tot2[kk]
        idx_loc = np.clip(np.searchsorted(cw2, r_loc, side="right"), start2[kk], np.maximum(end2[kk] - 1, 0))
        b_loc = order2[np.minimum(idx_loc, n - 1)]
        r_glob = c_lo[cb] + u * c_tot[cb]
        idx_glob = np.clip(np.searchsorted(cw, r_glob, side="right"), start_idx[cb], np.maximum(end_idx[cb] - 1, 0))
        b_glob = order[np.minimum(idx_glob, n - 1)]
        b = np.where(local, b_loc, b_glob)
        e = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
        return e[e[:, 0] != e[:, 1]]

    def draw(count: int) -> np.ndarray:
        if q_within <= 0:
            return draw_classes(count)
        n_in = int(round(q_within * count))
        parts = []
        if count - n_in > 0:
            parts.append(draw_classes(count - n_in))
        if n_in > 0:
            parts.append(draw_regions(n_in))
        return np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)

    have = e1 if len(e1) else np.zeros((0, 2), dtype=np.int64)
    keys = _edge_key(have, n) if len(have) else np.zeros(0, dtype=np.int64)
    need = m2
    for _ in range(8):
        if need <= 0:
            break
        cand = draw(int(need * 1.1) + 1)
        ck = np.unique(_edge_key(cand, n))
        ck = ck[~np.isin(ck, keys)]
        if len(ck) > need:
            ck = rng.choice(ck, size=need, replace=False)
        keys = np.concatenate([keys, ck])
        need -= len(ck)
    keys = np.unique(keys)

    # repair: drop edges between over-target nodes, then pair stubs of under-target nodes
    # (random matching), never exceeding the target edge count
    for _ in range(4):
        u, v = keys // n, keys % n
        cur = np.bincount(np.concatenate([u, v]), minlength=n)
        over = np.maximum(cur - deg, 0).astype(float)
        cand = (over[u] > 0) & (over[v] > 0)  # edges whose both ends are above target
        ccount = np.bincount(np.concatenate([u[cand], v[cand]]), minlength=n).astype(float)
        p_drop = np.minimum(
            over[u] / np.maximum(ccount[u], 1.0), over[v] / np.maximum(ccount[v], 1.0)
        )
        keys = keys[~(cand & (rng.random(len(keys)) < p_drop))]
        cur = np.bincount(np.concatenate([keys // n, keys % n]), minlength=n)
        stubs = np.repeat(np.arange(n), np.maximum(deg - cur, 0))
        budget = int(deg.sum() // 2 - len(keys)) * 2  # never exceed the target edge count
        if len(stubs) < 2 or budget < 2:
            break
        stubs = rng.permutation(stubs)[:budget]
        half = len(stubs) // 2
        a, b = stubs[:half], stubs[half : 2 * half]
        sk = np.minimum(a, b) * n + np.maximum(a, b)
        sk = np.unique(sk[a != b])
        keys = np.union1d(keys, sk)
    out = np.stack([keys // n, keys % n], axis=1).astype(np.int64)
    if rewire_rounds > 0 and (jd or ep):
        movable = ~np.isin(keys, _edge_key(e1, n)) if len(e1) else np.ones(len(keys), dtype=bool)
        # channel 0: the combined class-pair target (as before). Every non-primary attr_edge_pair
        # instance adds a channel over its own group codes; its target is the released ordered-pair
        # mass scaled to the synthetic total degree volume (the convention of channel 0, whose IPF
        # margins are the class degree volumes). Unusable instances (bad params, mismatched shape,
        # no mass, a single group) are skipped: validation flags them, generation stays robust.
        channels = [(cls, target)]
        tot = float(deg.sum())
        for m in ep:
            if m is primary:
                continue
            mcols = list(m.params.get("attrs") or [])
            if not mcols or any(c not in artifact.schema.domains or c not in attrs.columns for c in mcols):
                continue
            gk, size = _attr_codes(attrs, artifact.schema.domains, mcols)
            q = _ordered_mass(m.values)
            if q.shape != (size, size) or size < 2 or q.sum() <= 0:
                continue
            channels.append((gk, q * (tot / q.sum())))
        out = _rewire(out, movable, channels, n, rng, rewire_rounds,
                      regions=np.asarray(regions, dtype=np.int64) if regions is not None else None)
    return out


def _calibrated_structure(
    artifact: Artifact,
    deg: np.ndarray,
    attrs: pd.DataFrame,
    seed: int,
    rounds: int = 4,
    rewire_rounds: int = 40,
    regions: np.ndarray | None = None,
    within_region_frac: float = 0.0,
) -> np.ndarray:
    """Run the sampler, then rescale per-bin block density so phase two's extra triangles do not
    push the clustering by degree above the released target (uses only the artifact).
    ``regions`` / ``within_region_frac`` go to ``_generate_structure`` (plan section 7)."""
    n = len(deg)
    cl_edges, c_target = _clustering_target(artifact)
    rho = np.cbrt(c_target)
    edges = None
    for _ in range(rounds + 1):
        edges = _generate_structure(
            artifact, deg, attrs, np.random.default_rng([seed, 1]), cl_edges, rho, rewire_rounds,
            regions=regions, within_region_frac=within_region_frac,
        )
        if rounds == 0 or not (c_target > 0).any():
            break
        c_gen = _mean_clustering_by_bin(edges, n, cl_edges)
        ok = (c_target > 0) & (c_gen > 0)
        new = rho.copy()
        new[ok] = np.clip(rho[ok] * (c_target[ok] / c_gen[ok]) ** (1.0 / 3.0), 0.0, 1.0)
        if np.allclose(new, rho, rtol=0.02):
            break
        rho = new
        rounds -= 1
    return edges


# ------------------------------------------------------------------------------------------ API
def generate(
    artifact: Artifact,
    n: int | None = None,
    seed: int = 0,
    rewire_rounds: int = 40,
    regions: np.ndarray | None = None,
    within_region_frac: float = 0.0,
) -> AttributedGraph:
    """Sample a synthetic ``AttributedGraph`` using only the measurements in ``artifact``.

    ``n`` defaults to ``artifact.schema.n``; other values rescale all counts proportionally.
    ``rewire_rounds=0`` disables the final class-pair rewiring (used by the diagnostics).
    ``regions`` (optional, plan section 7): an ``(n,)`` int array of latent region labels; phase-one
    blocks then never cross a region and a fraction ``within_region_frac`` of the phase-two draw is made
    inside regions. The defaults (``None``, ``0.0``) leave the sampler, including its random stream,
    exactly as it was.
    """
    n = int(artifact.schema.n if n is None else n)
    if n < 1:
        raise ValueError("n must be positive")
    if regions is not None:
        regions = np.asarray(regions, dtype=np.int64).ravel()
        if regions.shape != (n,) or (len(regions) and regions.min() < 0):
            raise ValueError("regions must be an (n,) array of non-negative int labels")
        if len(regions) == 0 or regions.max() == 0:
            regions = None  # a single region is the identity
    if not 0.0 <= float(within_region_frac) <= 1.0:
        raise ValueError("within_region_frac must be in [0, 1]")
    rng = np.random.default_rng(seed)
    attrs = _sample_attributes(artifact, n, rng)
    deg = _sample_degrees(artifact, n, rng, attrs)
    edges = _calibrated_structure(
        artifact, deg, attrs, seed, rewire_rounds=rewire_rounds,
        regions=regions, within_region_frac=within_region_frac,
    )
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = np.stack([edges.min(axis=1), edges.max(axis=1)], axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    keys = np.unique(edges[:, 0] * n + edges[:, 1])  # sorted, unique
    edges = np.stack([keys // n, keys % n], axis=1).astype(np.int64)
    domains = {c: list(v) for c, v in artifact.schema.domains.items()}
    return AttributedGraph(n=n, edges=edges, attrs=attrs, domains=domains)

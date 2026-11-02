"""Non-private reference baselines (plan section 7).

Every baseline reads the REAL graph directly (degree sequence, edge counts, communities, clustering
profile), so none of them is differentially private. They are reference points for what a model with
full access achieves, not competitors under the same privacy guarantee.

All baselines have the signature ``fit_sample(real, seed) -> AttributedGraph`` and draw the node
attributes independently per column from the real 1-way marginals. They therefore carry no
attribute correlation and no attribute-structure (homophily) signal; any homophily error they show
is the cost of ignoring attributes in the structure model.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from graph_aim.eval.evaluate import degrees, local_clustering, adjacency
from graph_aim.transfer.types import AttributedGraph

_MAX_ROUNDS = 30


def _independent_attrs(real: AttributedGraph, rng: np.random.Generator) -> pd.DataFrame:
    """Each column drawn independently from its real 1-way marginal."""
    cols = {}
    for c in real.attrs.columns:
        size = max(len(real.domains[c]), int(real.attrs[c].max()) + 1)
        p = np.bincount(real.attrs[c].to_numpy(dtype=np.int64), minlength=size).astype(float)
        p /= p.sum()
        cols[c] = rng.choice(size, size=real.n, p=p).astype(np.int64)
    return pd.DataFrame(cols, index=pd.RangeIndex(real.n))


def _keys_to_edges(keys: np.ndarray, n: int) -> np.ndarray:
    keys = np.unique(keys)
    return np.stack([keys // n, keys % n], axis=1).astype(np.int64) if len(keys) else np.zeros((0, 2), np.int64)


def _pair_keys(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Encode pairs as ``min * n + max``, dropping self loops."""
    keep = u != v
    u, v = u[keep], v[keep]
    return np.minimum(u, v).astype(np.int64) * n + np.maximum(u, v)


def _sample_weighted_edges(
    nodes_a: np.ndarray,
    wa: np.ndarray,
    nodes_b: np.ndarray,
    wb: np.ndarray,
    count: int,
    n: int,
    rng: np.random.Generator,
    existing: np.ndarray | None = None,
) -> np.ndarray:
    """``count`` distinct non-loop edge keys, endpoints drawn proportional to weights (fast Chung-Lu).

    Endpoint ``u`` from ``nodes_a`` ~ ``wa`` and ``v`` from ``nodes_b`` ~ ``wb``. Tops up for a bounded
    number of rounds, so the result can fall short if the request is infeasible.
    """
    have = np.zeros(0, dtype=np.int64) if existing is None else np.unique(existing)
    out = np.zeros(0, dtype=np.int64)
    if count <= 0 or wa.sum() <= 0 or wb.sum() <= 0:
        return out
    pa, pb = wa / wa.sum(), wb / wb.sum()
    for _ in range(_MAX_ROUNDS):
        need = count - len(out)
        if need <= 0:
            break
        draw = int(need * 1.2) + 8
        k = _pair_keys(rng.choice(nodes_a, draw, p=pa), rng.choice(nodes_b, draw, p=pb), n)
        out = np.unique(np.concatenate([out, np.setdiff1d(k, have)]))
    if len(out) > count:
        out = rng.choice(out, count, replace=False)
    return out


def _wrap(real: AttributedGraph, keys: np.ndarray, rng: np.random.Generator) -> AttributedGraph:
    return AttributedGraph(
        n=real.n,
        edges=_keys_to_edges(keys, real.n),
        attrs=_independent_attrs(real, rng),
        domains={c: list(d) for c, d in real.domains.items()},
    )


def erdos_renyi(real: AttributedGraph, seed: int) -> AttributedGraph:
    """G(n, m) with the real node and edge counts. Non-private reference; independent attributes."""
    rng = np.random.default_rng(seed)
    n, m = real.n, len(real.edges)
    nodes = np.arange(n)
    ones = np.ones(n)
    keys = _sample_weighted_edges(nodes, ones, nodes, ones, m, n, rng)
    return _wrap(real, keys, rng)


def chung_lu(real: AttributedGraph, seed: int) -> AttributedGraph:
    """Chung-Lu with the real degree sequence as expected degrees. Non-private reference.

    Fast sampler: endpoints drawn proportional to degree, duplicates and loops rejected until the
    real edge count is reached. Independent attributes.
    """
    rng = np.random.default_rng(seed)
    n, m = real.n, len(real.edges)
    nodes, w = np.arange(n), degrees(real).astype(float)
    keys = _sample_weighted_edges(nodes, w, nodes, w, m, n, rng)
    return _wrap(real, keys, rng)


def dc_sbm(real: AttributedGraph, seed: int) -> AttributedGraph:
    """Degree-corrected stochastic block model. Non-private reference (uses real blocks and counts).

    Blocks are Louvain communities of the real graph (networkx, fixed ``seed``). For every block pair
    the real edge count is reproduced, with endpoints chosen proportional to real degree inside each
    block. Independent attributes (blocks do not depend on attributes).
    """
    rng = np.random.default_rng(seed)
    n = real.n
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, np.asarray(real.edges).tolist()))
    comms = nx.community.louvain_communities(g, seed=seed)
    block = np.zeros(n, dtype=np.int64)
    for b, members in enumerate(comms):
        block[list(members)] = b
    nb = len(comms)
    deg = degrees(real).astype(float)
    e = np.asarray(real.edges, dtype=np.int64).reshape(-1, 2)
    br, bs = np.minimum(block[e[:, 0]], block[e[:, 1]]), np.maximum(block[e[:, 0]], block[e[:, 1]])
    counts = np.zeros((nb, nb), dtype=np.int64)
    np.add.at(counts, (br, bs), 1)
    members = [np.flatnonzero(block == b) for b in range(nb)]
    parts = []
    for r in range(nb):
        for s in range(r, nb):
            t = int(counts[r, s])
            if t:
                parts.append(
                    _sample_weighted_edges(members[r], deg[members[r]], members[s], deg[members[s]], t, n, rng)
                )
    keys = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return _wrap(real, keys, rng)


def bter(real: AttributedGraph, seed: int) -> AttributedGraph:
    """Block two-level Erdos-Renyi (Seshadhri, Kolda, Pinar). Non-private reference.

    Phase 1: nodes of degree >= 2 sorted by degree and packed into affinity blocks of size ``d + 1``
    (``d`` = smallest degree in the block); inside a block edges are ER with density
    ``c(d) ** (1/3)`` where ``c(d)`` is the real mean local clustering at degree ``d`` (nearest
    observed degree if ``d`` is missing). Phase 2: Chung-Lu on the excess degree. Collisions with phase
    1 edges are rejected. Independent attributes.
    """
    rng = np.random.default_rng(seed)
    n = real.n
    deg = degrees(real)
    clus = local_clustering(adjacency(real), deg)
    ks = np.unique(deg[deg >= 2])
    c_of_k = np.array([clus[deg == k].mean() for k in ks]) if len(ks) else np.zeros(0)

    def rho_for(d: int) -> float:
        if len(ks) == 0:
            return 0.0
        c = c_of_k[np.argmin(np.abs(ks - d))]
        return float(np.clip(c, 0, 1) ** (1.0 / 3.0))

    order = np.flatnonzero(deg >= 2)
    order = order[np.argsort(deg[order], kind="stable")]
    excess = deg.astype(float).copy()
    phase1 = []
    i = 0
    while i < len(order):
        d = int(deg[order[i]])
        blk = order[i : i + d + 1]
        i += d + 1
        if len(blk) < 2:
            continue
        rho = rho_for(d)
        iu, ju = np.triu_indices(len(blk), k=1)
        keep = rng.random(len(iu)) < rho
        phase1.append(_pair_keys(blk[iu[keep]], blk[ju[keep]], n))
        excess[blk] = np.maximum(deg[blk] - rho * (len(blk) - 1), 0.0)
    p1 = np.unique(np.concatenate(phase1)) if phase1 else np.zeros(0, np.int64)
    m2 = int(round(excess.sum() / 2.0))
    nodes = np.arange(n)
    p2 = _sample_weighted_edges(nodes, excess, nodes, excess, m2, n, rng, existing=p1)
    return _wrap(real, np.concatenate([p1, p2]), rng)


BASELINES = {
    "erdos_renyi": erdos_renyi,
    "chung_lu": chung_lu,
    "dc_sbm": dc_sbm,
    "bter": bter,
}

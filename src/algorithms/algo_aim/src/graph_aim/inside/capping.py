"""Stable degree cap (plan section 3).

Edge ``{u, v}`` is kept iff its hash rank is among the ``D`` smallest over the edges incident to
``u`` AND among the ``D`` smallest over the edges incident to ``v``. The rank key of an edge is
``(h, lo, hi)`` where ``h = edge_hash(lo, hi, salt)`` and ``lo < hi`` are the endpoints; it depends
only on the edge itself (never on the rest of the graph), so the order is a fixed public total order.

Stability. Adding an edge ``e = {u, v}`` inserts one element into the rank order at ``u`` and one at
``v`` and nothing else. At ``u`` the only existing edge whose status can flip is the one that was
at position ``D - 1`` and is pushed to position ``D`` (it loses "within the D smallest at u");
likewise at ``v``. No edge ever *gains* membership. Hence the capped set changes by at most
``e`` itself plus one edge per endpoint: at most 3 edges. Removal is the reverse of an addition.
The capped graph has maximum degree <= D. Attribute changes do not touch the capped graph.
"""

from __future__ import annotations

import numpy as np

_M64 = (1 << 64) - 1


def edge_hash(lo: np.ndarray, hi: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic 64-bit hash (splitmix64 finaliser) of sorted pairs; platform independent."""
    lo = np.asarray(lo).astype(np.uint64)
    hi = np.asarray(hi).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (lo << np.uint64(32)) ^ hi
        x = x + np.uint64((int(salt) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & _M64)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def cap_edges(graph_edges: np.ndarray, n: int, D: int, salt: int) -> np.ndarray:
    """Return the capped edge set as a lexicographically sorted ``(m', 2)`` int64 array (u < v).

    ``graph_edges`` may be in any order and orientation but must be a simple graph on ``0..n-1``.
    """
    e = np.asarray(graph_edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return e.copy()
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    m = len(e)
    h = edge_hash(lo, hi, salt)
    # incidence list: each edge appears once per endpoint
    node = np.concatenate([lo, hi])
    eid = np.concatenate([np.arange(m), np.arange(m)])
    # sort by node, then by the edge key (h, lo, hi)
    order = np.lexsort((hi[eid], lo[eid], h[eid], node))
    node_s = node[order]
    starts = np.r_[0, np.flatnonzero(node_s[1:] != node_s[:-1]) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(node_s)]))
    rank = np.arange(len(node_s)) - group_start
    within = rank < D
    ok = np.zeros(m, dtype=np.int64)
    np.add.at(ok, eid[order], within.astype(np.int64))
    keep = ok == 2
    out = np.stack([lo[keep], hi[keep]], axis=1)
    if len(out) == 0:
        return out
    return out[np.lexsort((out[:, 1], out[:, 0]))]

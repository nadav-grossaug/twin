"""SNAP side of the real-vs-fake classifier: walk sampler and featurizer (``docs/realfake/snap.md``).

One sample is a random-walk subgraph: a start node drawn uniformly, a simple random walk (jumping
back to the start node when stuck) until ``k`` distinct nodes are visited, ``k`` drawn per sample
log-uniformly from ``[k_min, k_max]``. The payload is the induced subgraph on the visited nodes,
their attribute rows and their degrees in the parent graph.

What is deliberately dropped (``docs/realfake/snap.md`` section 2): parent node ids and parent edge
order (nodes are relabelled by visit order and edges stored in canonical sorted ``u < v`` order,
because real SNAP ids carry the ego-network ordering while synthetic ids are exchangeable), the
attribute column order (the featurizer reads columns in sorted-name order), and whole-graph
constants (``n``, ``m`` are not a property of a sample). Everything the method models or fails to
model (degrees, clustering, marginals, homophily) stays in; that is what the classifier detects.

Features come in four groups (``docs/realfake/snap.md`` section 3) for grouped permutation
importance: ``size``, ``degree``, ``structure``, ``attributes``. Undefined statistics (empty edge
set, constant degrees) are 0.0, never NaN, so the vector length is always ``len(feature_names)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from graph_aim.eval.evaluate import (
    _degree_assortativity,
    adjacency,
    local_clustering,
    triangles_per_node,
)
from graph_aim.transfer.types import AttributedGraph
from realfake.types import Sample

_DEG_QUANTILES = (25, 50, 75, 90)


@dataclass
class GraphSample:
    """Payload of one walk sample.

    ``edges`` is ``(m, 2)`` int64 with ``u < v``, unique rows, sorted lexicographically; node ids
    are ``0..k-1`` in visit order, so nothing of the parent graph's labelling or edge order survives
    beyond visit order. ``attrs`` has one row per visited node in visit order (the parent's column
    order, which the featurizer ignores). ``parent_deg`` is each visited node's degree in the
    parent graph; ``k`` the number of visited nodes.
    """

    edges: np.ndarray
    attrs: pd.DataFrame
    parent_deg: np.ndarray
    k: int


# ------------------------------------------------------------------------------------------ sampler
class WalkSampler:
    """``Sampler`` for attributed graphs: one random-walk subgraph per call.

    ``k`` is drawn log-uniformly from ``[k_min, min(k_max, source.n)]``. The walk jumps back to the
    start node when the current node has no neighbours; an isolated start node yields the
    single-node sample. A step budget (``max_steps_per_node * k``) bounds walks trapped in a
    component smaller than ``k``, in which case the sample is smaller than drawn. Deterministic
    given the ``rng`` state. Per-source adjacency is cached (keyed by object identity, with a strong
    reference so ids cannot be reused); a plain call works without any preparation.
    """

    def __init__(self, k_min: int = 10, k_max: int = 200, max_steps_per_node: int = 100):
        if not 1 <= k_min <= k_max:
            raise ValueError(f"need 1 <= k_min <= k_max, got [{k_min}, {k_max}]")
        self.k_min, self.k_max = int(k_min), int(k_max)
        self.max_steps_per_node = int(max_steps_per_node)
        self._cache: dict[int, tuple[AttributedGraph, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def prepare(self, source: AttributedGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR neighbour lookup ``(indptr, indices, degrees)`` of ``source``, cached."""
        hit = self._cache.get(id(source))
        if hit is not None and hit[0] is source:
            return hit[1]
        adj = adjacency(source).tocsr()
        deg = np.bincount(np.asarray(source.edges, dtype=np.int64).ravel(), minlength=source.n)
        entry = (adj.indptr.astype(np.int64), adj.indices.astype(np.int64), deg.astype(np.int64))
        self._cache[id(source)] = (source, entry)
        return entry

    def __call__(self, source: AttributedGraph, rng: np.random.Generator) -> Sample:
        indptr, indices, pdeg = self.prepare(source)
        n = source.n
        k_hi = max(min(self.k_max, n), 1)
        k_lo = min(self.k_min, k_hi)
        k = int(np.clip(np.rint(np.exp(rng.uniform(np.log(k_lo), np.log(k_hi)))), k_lo, k_hi))

        start = int(rng.integers(n))
        seen = np.zeros(n, dtype=bool)
        seen[start] = True
        order = [start]
        cur = start
        budget = self.max_steps_per_node * k
        steps = 0
        while len(order) < k and steps < budget:
            steps += 1
            lo, hi = indptr[cur], indptr[cur + 1]
            if hi == lo:  # stuck: no neighbours
                if cur == start:
                    break  # isolated start node: the sample is that single node
                cur = start
                continue
            cur = int(indices[lo + rng.integers(hi - lo)])
            if not seen[cur]:
                seen[cur] = True
                order.append(cur)

        nodes = np.asarray(order, dtype=np.int64)
        k_act = len(nodes)
        new_id = np.full(n, -1, dtype=np.int64)
        new_id[nodes] = np.arange(k_act)

        # induced subgraph from the neighbour lists of the visited nodes (each undirected edge
        # appears once as u_new < v_new); canonical sort drops any trace of the parent edge order
        nbrs = np.concatenate([indices[indptr[u] : indptr[u + 1]] for u in nodes]) if k_act else np.zeros(0, np.int64)
        u_new = np.repeat(np.arange(k_act), (indptr[nodes + 1] - indptr[nodes])) if k_act else np.zeros(0, np.int64)
        v_new = new_id[nbrs]
        keep = (v_new >= 0) & (u_new < v_new)
        u_new, v_new = u_new[keep], v_new[keep]
        o = np.argsort(u_new * max(k_act, 1) + v_new)
        edges = np.stack([u_new[o], v_new[o]], axis=1).astype(np.int64)

        payload = GraphSample(
            edges=edges,
            attrs=source.attrs.iloc[nodes].reset_index(drop=True),
            parent_deg=pdeg[nodes],
            k=k_act,
        )
        return Sample(payload=payload, group_id=start, meta={"size": float(k_act)})


# --------------------------------------------------------------------------------------- featurizer
class GraphFeaturizer:
    """``Featurizer`` for walk subgraphs, built from public schema information only.

    ``domains`` fixes every attribute column and its full label list, so the vector has the same
    length whichever values appear in a sample; ``homophily_attrs`` names the columns that get a
    same-value edge rate. Columns are handled in sorted-name order, so the vector is invariant to
    the payload's column order; every feature is a symmetric statistic of the subgraph, so it is
    invariant to a consistent relabelling of the payload's node ids.
    """

    def __init__(self, domains: dict[str, list[str]], homophily_attrs: list[str] | None = None):
        self.domains = {c: list(v) for c, v in domains.items()}
        self._cols = sorted(self.domains)
        hom = list(homophily_attrs) if homophily_attrs else []
        missing = [c for c in hom if c not in self.domains]
        if missing:
            raise ValueError(f"homophily attributes not in domains: {missing}")
        self._hom = sorted(set(hom))
        self._none_other = {
            c: np.array([i for i, lab in enumerate(d) if lab in ("none", "other")], dtype=np.int64)
            for c, d in self.domains.items()
        }
        self.feature_groups: dict[str, list[str]] = {
            "size": ["size:k", "size:edges", "size:density"],
            "degree": (
                ["degree:mean", "degree:std", "degree:max"]
                + [f"degree:q{q}" for q in _DEG_QUANTILES]
                + ["degree:parent_log_mean", "degree:parent_log_std", "degree:parent_log_max"]
                + ["degree:ratio_mean"]
            ),
            "structure": [
                "structure:triangles_per_node",
                "structure:clustering_mean",
                "structure:transitivity",
                "structure:assortativity",
            ],
            "attributes": (
                [f"attributes:share:{c}={lab}" for c in self._cols for lab in self.domains[c]]
                + [f"attributes:none_other:{c}" for c in self._cols]
                + [f"attributes:same_edge_rate:{c}" for c in self._hom]
                + [f"attributes:pair_edge_match:{a}|{b}" for a, b in combinations(self._cols, 2)]
            ),
        }
        self.feature_names: list[str] = [f for names in self.feature_groups.values() for f in names]

    def __call__(self, sample: Sample) -> np.ndarray:
        p: GraphSample = sample.payload
        k = int(p.k)
        e = np.asarray(p.edges, dtype=np.int64).reshape(-1, 2)
        m = len(e)
        deg = np.bincount(e.ravel(), minlength=k).astype(np.float64)

        vals = [float(k), float(m), 2.0 * m / (k * (k - 1)) if k > 1 else 0.0]

        vals += [float(deg.mean()), float(deg.std()), float(deg.max())]
        vals += [float(np.percentile(deg, q)) for q in _DEG_QUANTILES]
        lp = np.log1p(p.parent_deg.astype(np.float64))
        vals += [float(lp.mean()), float(lp.std()), float(lp.max())]
        pdeg = p.parent_deg.astype(np.float64)
        ratio = np.divide(deg, pdeg, out=np.zeros(k), where=pdeg > 0)
        vals.append(float(ratio.mean()))

        adj = adjacency(AttributedGraph(k, e, p.attrs, {}))
        tri = triangles_per_node(adj)
        wedges = float((deg * (deg - 1) / 2.0).sum())
        vals += [
            float(tri.sum() / 3.0 / k),
            float(local_clustering(adj, deg).mean()),
            float(tri.sum() / wedges) if wedges > 0 else 0.0,
            _degree_assortativity(adj, deg),
        ]

        codes = {c: p.attrs[c].to_numpy(dtype=np.int64) for c in self._cols}
        shares = {
            c: np.bincount(codes[c], minlength=len(self.domains[c]))[: len(self.domains[c])] / k
            for c in self._cols
        }
        for c in self._cols:
            vals += shares[c].tolist()
        for c in self._cols:
            vals.append(float(shares[c][self._none_other[c]].sum()))
        u, v = e[:, 0], e[:, 1]
        for c in self._hom:
            vals.append(float((codes[c][u] == codes[c][v]).mean()) if m else 0.0)
        for a, b in combinations(self._cols, 2):
            both = (codes[a][u] == codes[a][v]) & (codes[b][u] == codes[b][v])
            vals.append(float(both.mean()) if m else 0.0)

        out = np.asarray(vals, dtype=np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

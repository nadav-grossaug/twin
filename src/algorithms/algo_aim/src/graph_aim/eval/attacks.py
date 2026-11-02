"""EMPIRICAL privacy diagnostics for a synthetic graph. These are NOT the differential privacy guarantee.

The DP statement lives in the artifact's privacy ledger. The checks here only ask: "does this particular
released synthetic graph look like a copy of the real one?" A pass does not prove privacy, and a flag does
not mean the DP accounting is wrong. They are sanity checks that catch gross failures (for example a
generator that memorised the input).

Synthetic node ids are fresh, so edge-id membership inference is meaningless. Instead we compare node
level fingerprints, and always against a real-vs-real HOLDOUT: the real nodes are split into random halves
A and B, features are computed inside each induced subgraph, and the same statistic is computed between A
and B. The holdout is what two independent samples from the same population achieve by chance; a synthetic
graph that is markedly closer than that is suspicious. Degrees measured inside a half-sized induced
subgraph are about half of the full-graph degrees, so degree-like features of the halves are rescaled by
``n / n_half`` to be comparable.

Checks:

1. ``signature_linkage``: signature = (attribute vector, degree). Rate = share of the nodes with a unique
   signature in the real graph that also occur in the synthetic graph. Holdout rate = same with
   "unique inside A, occurs in B".
2. ``closest_node_distance``: for each real node, the distance to the nearest synthetic node in a feature
   space (attributes one-hot, standardised log degree, local clustering, standardised log mean neighbour
   degree). Compared with the distance from A nodes to their nearest B node (the synthetic
   reference set is about twice the holdout reference size, which biases mildly towards flagging).
3. ``degree_signature_check``: as (1) with the degree alone, a weaker adversary who knows only degrees.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from graph_aim.eval.evaluate import adjacency, degrees, local_clustering
from graph_aim.transfer.types import AttributedGraph

LABEL = "EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee"


def _induced(g: AttributedGraph, nodes: np.ndarray) -> AttributedGraph:
    """Induced subgraph on ``nodes`` (relabelled 0..len-1 in the order of ``nodes``)."""
    new = -np.ones(g.n, dtype=np.int64)
    new[nodes] = np.arange(len(nodes))
    e = np.asarray(g.edges, dtype=np.int64).reshape(-1, 2)
    keep = (new[e[:, 0]] >= 0) & (new[e[:, 1]] >= 0)
    sub = new[e[keep]]
    sub = np.sort(sub, axis=1) if len(sub) else sub.reshape(0, 2)
    return AttributedGraph(
        n=len(nodes),
        edges=sub,
        attrs=g.attrs.iloc[nodes].reset_index(drop=True),
        domains=g.domains,
    )


def _split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    h = n // 2
    return np.sort(perm[:h]), np.sort(perm[h : 2 * h])


def _sizes(real: AttributedGraph, syn: AttributedGraph) -> list[int]:
    return [
        max(len(real.domains[c]), int(real.attrs[c].max()) + 1, int(syn.attrs[c].max()) + 1)
        for c in real.attrs.columns
    ]


def node_features(g: AttributedGraph, sizes: list[int], degree_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(one_hot, numeric)``: attribute one-hot block and ``[log1p degree, clustering, log1p mean nbr degree]``."""
    blocks = []
    for c, s in zip(g.attrs.columns, sizes):
        oh = np.zeros((g.n, s))
        oh[np.arange(g.n), g.attrs[c].to_numpy(dtype=np.int64)] = 1.0
        blocks.append(oh)
    one_hot = np.hstack(blocks) if blocks else np.zeros((g.n, 0))
    adj = adjacency(g)
    deg = degrees(g).astype(float)
    nbr = np.divide(adj @ deg, deg, out=np.zeros(g.n), where=deg > 0)
    clus = local_clustering(adj, deg)
    numeric = np.column_stack([np.log1p(deg * degree_scale), clus, np.log1p(nbr * degree_scale)])
    return one_hot, numeric


def _embed(parts: tuple[np.ndarray, np.ndarray], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    one_hot, numeric = parts
    return np.hstack([one_hot, (numeric - mu) / sd])


def _summary(d: np.ndarray, thresholds: tuple[float, ...]) -> dict:
    out = {
        "median": float(np.median(d)),
        "p05": float(np.percentile(d, 5)),
        "mean": float(d.mean()),
    }
    for t in thresholds:
        out[f"share_below_{t:g}"] = float((d < t).mean())
    return out


def closest_node_distance(
    real: AttributedGraph,
    syn: AttributedGraph,
    seed: int = 0,
    thresholds: tuple[float, ...] = (1e-6, 0.25, 0.5),
    ratio_tol: float = 0.5,
) -> dict:
    """Nearest-neighbour distance from real nodes to synthetic nodes versus a real-vs-real holdout.

    Interpretation: ``synthetic`` summarises, for every real node, the distance to the nearest synthetic
    node (the synthetic reference is about twice the holdout reference size, a mild bias towards
    flagging). ``holdout`` summarises the distance from half A
    of the real nodes to the nearest node of the other half B (features inside induced subgraphs, degrees
    rescaled). ``ratio_median = synthetic.median / holdout.median``: near 1 means the synthetic graph is
    as close to the real nodes as an independent real sample (expected for a good generator); well below 1
    means it sits closer than independent data would, as a copy does (exact copy: ratio 0).
    ``too_close`` is a heuristic flag: ``ratio_median < ratio_tol`` or the 5th percentile ratio is below
    ``ratio_tol / 2``. Distances are in feature units (one-hot mismatch contributes sqrt(2) per attribute).
    """
    sizes = _sizes(real, syn)
    a, b = _split_halves(real.n, seed)

    full_real = node_features(real, sizes)
    mu, sd = full_real[1].mean(axis=0), full_real[1].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    # synthetic: queries = all real nodes (full-graph features), reference = all synthetic nodes
    syn_feat = _embed(node_features(syn, sizes), mu, sd)
    d_syn = cKDTree(syn_feat).query(_embed(full_real, mu, sd))[0]

    # holdout: halves A (queries) and B (reference), features inside induced subgraphs
    ga, gb = _induced(real, a), _induced(real, b)
    fa = _embed(node_features(ga, sizes, real.n / ga.n), mu, sd)
    fb = _embed(node_features(gb, sizes, real.n / gb.n), mu, sd)
    d_hold = cKDTree(fb).query(fa)[0]

    s, ho = _summary(d_syn, thresholds), _summary(d_hold, thresholds)
    ratio_med = s["median"] / ho["median"] if ho["median"] > 0 else float("nan")
    ratio_p05 = s["p05"] / ho["p05"] if ho["p05"] > 0 else float("nan")
    too_close = bool(
        (ratio_med < ratio_tol) or (ratio_p05 == ratio_p05 and ratio_p05 < ratio_tol / 2)
    )
    return {
        "label": LABEL,
        "synthetic": s,
        "holdout": ho,
        "ratio_median": float(ratio_med),
        "ratio_p05": float(ratio_p05),
        "too_close": too_close,
    }


def _signature_keys(g: AttributedGraph, use_attrs: bool) -> np.ndarray:
    """Hashable per-node signature as a structured-row id (unique integer per distinct signature)."""
    deg = degrees(g)
    cols = [deg.reshape(-1, 1)]
    if use_attrs:
        cols.append(g.attrs.to_numpy(dtype=np.int64))
    return np.hstack(cols)


def _unique_linked(query_sig: np.ndarray, ref_sig: np.ndarray, within_sig: np.ndarray) -> tuple[int, int]:
    """(# nodes unique in ``within_sig`` and present in ``ref_sig``, # unique nodes)."""
    rows, inv, cnt = np.unique(within_sig, axis=0, return_inverse=True, return_counts=True)
    unique_mask = cnt[inv.ravel()] == 1
    ref_rows = {tuple(r) for r in np.unique(ref_sig, axis=0).tolist()}
    q = query_sig[unique_mask]
    linked = sum(tuple(r) in ref_rows for r in q.tolist())
    return linked, int(unique_mask.sum())


def signature_linkage(
    real: AttributedGraph,
    syn: AttributedGraph,
    seed: int = 0,
    use_attrs: bool = True,
    excess_tol: float = 0.25,
) -> dict:
    """Share of real-unique node signatures that also occur in ``syn``, versus a real-vs-real holdout.

    Signature = (degree, attribute vector), or the degree alone with ``use_attrs=False``. ``rate_syn`` is
    over real nodes whose signature is unique in the full real graph and counts those whose signature
    occurs anywhere in ``syn``. ``rate_holdout`` is the same with "unique inside half A's induced
    subgraph" and "occurs in half B's induced subgraph" (degrees rescaled by ``n / n_half`` and rounded).
    A good generator reproduces degree and attribute distributions, so some matches are expected by
    chance; ``rate_holdout`` is that chance level. ``excess = rate_syn - rate_holdout`` well above zero
    means signatures are reproduced more often than independent data would; ``linkage_flag`` is raised
    when ``excess > excess_tol`` (heuristic). Because the synthetic graph is full size and the holdout
    reference is half size, mild excess is expected for large synthetic graphs; see ``excess_tol``.
    """
    a, b = _split_halves(real.n, seed)
    ga, gb = _induced(real, a), _induced(real, b)

    def scaled(g: AttributedGraph, full_n: int) -> np.ndarray:
        s = _signature_keys(g, use_attrs)
        s[:, 0] = np.rint(s[:, 0] * full_n / g.n).astype(np.int64)
        return s

    real_sig, syn_sig = _signature_keys(real, use_attrs), _signature_keys(syn, use_attrs)
    lk, un = _unique_linked(real_sig, syn_sig, real_sig)
    sa, sb = scaled(ga, real.n), scaled(gb, real.n)
    lk_h, un_h = _unique_linked(sa, sb, sa)
    rate_syn = lk / un if un else float("nan")
    rate_h = lk_h / un_h if un_h else float("nan")
    excess = rate_syn - rate_h
    return {
        "label": LABEL,
        "signature": "degree+attributes" if use_attrs else "degree",
        "n_unique_real": un,
        "rate_syn": float(rate_syn),
        "n_unique_holdout": un_h,
        "rate_holdout": float(rate_h),
        "excess": float(excess),
        "linkage_flag": bool(excess == excess and excess > excess_tol),
    }


def degree_signature_check(real: AttributedGraph, syn: AttributedGraph, seed: int = 0) -> dict:
    """Weaker adversary who knows only node degrees: ``signature_linkage`` with the degree alone."""
    return signature_linkage(real, syn, seed=seed, use_attrs=False)


def run_attacks(real: AttributedGraph, syn: AttributedGraph, seed: int = 0) -> dict:
    """All empirical diagnostics in one dict. Empirical only; says nothing about the DP guarantee."""
    return {
        "label": LABEL,
        "signature_linkage": signature_linkage(real, syn, seed),
        "closest_node": closest_node_distance(real, syn, seed),
        "degree_signature": degree_signature_check(real, syn, seed),
    }

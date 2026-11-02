"""The seven measurement kinds: exact statistics, proved L2 sensitivities, noise and post-processing.

Round 2 design (see docs/snap/graph-aim-plan.md section 11). Every measurement carries its OWN cap in
``params["cap"]`` (an int, or ``None`` = no cap). ``cap_edges(G, n, cap, salt)`` is the stable degree
cap of ``capping`` (max degree <= cap, edge-adjacent inputs give capped graphs that differ by <= 3
edges). Attribute-only measurements do not use a cap (``cap`` is recorded as ``None``).

Adjacency facts used by all derivations
---------------------------------------
* Attribute-row adjacency: the capped graph is unchanged (the cap hash ignores attributes), so purely
  structural statistics have attribute sensitivity 0.
* Edge adjacency, UNCAPPED measurement (``cap is None``): the statistic changes by ONE edge addition or
  removal; a removal is the reverse of an addition, so it suffices to bound one addition (``s1``).
* Edge adjacency, CAPPED measurement: the capped graphs ``G'_a, G'_b`` differ in at most 3 edges (see
  ``capping``). Remove the edges of ``G'_a \\ G'_b`` first, then add those of ``G'_b \\ G'_a``. Every
  intermediate graph is a subgraph of ``G'_b`` or of ``G'_a``, so has max degree <= cap. Each step is
  an addition of an edge ``{u, v}`` to a graph ``H`` such that ``H + {u, v}`` has max degree <= cap
  (a removal is the reverse of such an addition). The statistic changes by at most ``3 * s1``, where
  ``s1`` bounds one such addition (triangle inequality over <= 3 steps).

Released vector. For symmetric kinds (``joint_degree``, ``attr_edge_pair``) noise is added only to
the upper triangle including the diagonal and then mirrored (post-processing), so the sensitivity is
that of the upper-triangle vector. The matrix stores, for a != b, ``M[a, b] = M[b, a] =`` number of
edges with end-points in the two classes, and ``M[a, a] =`` number of edges inside the class.

Derivations (``s1`` = single-step bound)
----------------------------------------
``degree_hist`` (UNCAPPED, public ``max_degree = Dmax``). The released vector counts nodes in cells of
    a public grid ``hist_edges`` (``0 = h_0 < ... < h_K = Dmax + 1``; cell ``j`` holds degrees in
    ``[h_j, h_{j+1})``; degrees above Dmax are clamped into the last cell). With ``hist_edges = None``
    the grid is every single degree 0..Dmax. Write ``cell(d) = cell of min(d, Dmax)``. Adding {u, v}
    changes only d_u -> d_u + 1 and d_v -> d_v + 1. Node u's part of the change is zero if
    ``cell(d_u + 1) == cell(d_u)`` (incl. both clamped into the last cell), else ``-e_a + e_{a'}``
    with ``a' != a``, of norm sqrt(2). The two parts add up: worst case both endpoints cross the same
    boundary, giving ``-2 e_a + 2 e_a'`` of norm 2 sqrt(2); in every other case the norm is <= 2 or
    sqrt(2) (< 2 sqrt(2)). So **edge = 2 sqrt(2)** (no cap, hence no factor 3, and no D factor);
    **attr = 0**. The bound is attained (two nodes of equal degree just below a cell boundary).
    The total edge count is the post-processing ``sum(d * count_d) / 2`` of the released histogram.

``clustering_by_degree`` (capped at ``Dc = params["cap"]``, public ``min_degree = dmin`` and clip
    ``cmax``): per node x let ``f_x = [d_x >= dmin] * clip(t_x / C(d_x, 2), 0, cmax)`` (``C(d, 2)`` =
    d (d - 1) / 2; f_x = 0 when d_x < 2) and let ``phi_x = f_x e_{b(d_x), 0} + e_{b(d_x), 1}``, where
    ``b`` is the public degree bin (last bin is open-ended). The released (B, 2) array is
    ``sum_x phi_x``: column 0 the sum of clipped local clustering over nodes with degree >= dmin,
    column 1 the node count (so bins below dmin carry only counts; ``postprocess`` imputes their
    column 0 from the first informative bins, which is free post-processing).
    Adding {u, v} to H (max deg of H + {u, v} <= Dc, hence d_u, d_v <= Dc - 1 before) with common
    neighbours ``W = N(u) & N(v)``, ``k = |W| <= min(d_u, d_v) <= Dc - 1``: phi changes only for
    u, v and for w in W (d_w is unchanged for w, only t_w grows by 1).
    - u: if its bin is unchanged the change is ``(f' - f) e_{b, 0}`` with ``f, f' in [0, cmax]``, norm
      <= cmax. If its bin changes the change is ``-f e_{b,0} - e_{b,1} + f' e_{b',0} + e_{b',1}`` with
      four distinct coordinates, norm^2 = f^2 + f'^2 + 2 <= 2 cmax^2 + 2. Same for v.
    - w in W: only ``f_w`` changes, and only if d_w >= dmin (else f_w stays 0). Then t_w / C(d_w, 2)
      grows by exactly ``1 / C(d_w, 2) <= 1 / C(max(dmin, 2), 2)`` and clip(.) is 1-Lipschitz, so the
      change is <= ``delta = min(cmax, 1 / C(max(dmin, 2), 2))`` in one coordinate; sum over W has
      norm <= ``k delta`` (triangle inequality).
    s1 = 2 sqrt(2 cmax^2 + 2) + (Dc - 1) delta; **edge = 3 s1**; **attr = 0**.
    ``counts_from_hist`` (used by the shipped config): column 1 is NOT released; ``postprocess`` fills it
    with the degree-histogram mass of each bin (the histogram is released anyway: free post-processing).
    The released vector is column 0 only; an endpoint then moves by <= sqrt(2) cmax in either case (two
    coordinates in [0, cmax] when the bin changes), so ``s1 = 2 sqrt(2) cmax + (Dc - 1) delta``.
    ``cap = None`` (uncapped graph): no factor 3 (one edge changes), and the common-neighbour count is
    only bounded by the public node count, ``k <= n - 2``: ``s1 = 2 end + (n - 2) delta``.
    (The old bound used delta = min(1, cmax) -- the degree-2 common neighbour -- so the cost of a big
    cap was linear; excluding degrees below dmin from column 0 divides it by C(dmin, 2).)

``joint_degree`` (B, B), upper-triangle vector, public bins ``e_0 = 0 < e_1 < ... < e_B`` (the last
    bin is open-ended: degrees >= e_{B-1} all fall in it), cap ``params["cap"]`` is ``None`` or an int.
    Adding {u, v}: the new edge adds +1 in one cell. Every other edge (u, x) (x != v) of u has its
    u-end move from bin b_u to b_u' only if d_u + 1 is an internal bin edge ``e_k`` (1 <= k <= B - 1),
    in which case ``b_u' = b_u + 1`` and the number of such edges is exactly ``d_u = e_k - 1``. With
    m_j = number of those edges whose x lies in bin j (sum m_j = e_k - 1), the u-part is
    P + Q, P = -sum m_j e_{b_u, j}, Q = +sum m_j e_{b_u', j}; cells {b_u, j} are distinct for distinct
    j (likewise {b_u', j}), P and Q overlap in at most the cell {b_u, b_u'} where the entries have
    opposite signs, so <P, Q> <= 0 and ||P + Q||^2 <= ||P||^2 + ||Q||^2 <= 2 (sum m_j)^2: norm
    <= sqrt(2) (e_k - 1). Define ``r = max (e_k - 1)`` over internal edges (restricted to
    ``e_k <= cap`` when a cap is set, since d_u + 1 <= cap). Same for v. Triangle inequality over
    {new edge, u-part, v-part}: ``s1 = 2 sqrt(2) r + 1``. **edge = s1 if cap is None else 3 s1**;
    **attr = 0**. The bound depends on the LARGEST INTERNAL bin edge, not on the cap: an uncapped
    measurement with a coarse top bin is cheaper than a capped one with fine bins.

``attr_1way`` (K,) per attribute (one Measurement per attribute). Edge: 0 (nodes and attributes are
    fixed). A row change moves one unit from one cell to another: **attr = sqrt(2)**, **edge = 0**.

``attr_2way`` (Ka, Kb) per attribute pair. Same: **attr = sqrt(2)**, **edge = 0**.

``attr_edge_pair`` (G, G) upper triangle, group g(x) = row-major code of x's homophily attributes,
    capped at ``Dp = params["cap"]`` (required). Edge: one capped-edge change moves one cell by +-1, so
    s1 = 1; **edge = 3**. Attribute row of u changes group g -> g' (capped graph fixed): each of u's
    d_u <= Dp capped edges (u, x) moves from cell {g, g_x} to {g', g_x}. Identical argument to
    ``joint_degree`` with sum m_j <= Dp: **attr = sqrt(2) Dp**.
    Several INSTANCES may be released, one per configured homophily group (``homophily_groups``),
    named ``attr_edge_pair:<a>+<b>``; the sensitivities above hold per instance unchanged (each is the
    same statistic on its own group), and the kind's budget share is split equally among the instances
    by ``privacy.calibrate``, so every extra group makes all of them noisier.

``attr_degree`` (K_a, B) per attribute (one Measurement per attribute), UNCAPPED, public degree bins
    ``0 = e_0 < ... < e_B`` (last bin open-ended, ``graph_aim.transfer.binning.degree_bin``). Cell
    ``(v, j)`` counts nodes with attribute code v and ``bin(d_x) = j`` (true, uncapped degree).
    Edge adjacency: adding {u, v} changes only d_u -> d_u + 1 and d_v -> d_v + 1 and no attribute row.
    Node u's part of the change is zero if its bin stays the same (incl. the open-ended last bin), else
    ``-e_{(a_u, j)} + e_{(a_u, j+1)}`` (bins only move up by one), of norm sqrt(2). Cases for the sum of the
    two parts: (i) both endpoints cross a boundary with equal attribute value and the same boundary:
    ``-2 e_{(a, j)} + 2 e_{(a, j+1)}``, norm 2 sqrt(2); (ii) different attribute values (or different
    boundaries), all four cells distinct: norm 2; (iii) one endpoint's destination cell equals the other's
    source cell (equal value, adjacent bins j, j + 1): the middle terms cancel, leaving
    ``-e_{(a, j)} + e_{(a, j+2)}``, norm sqrt(2); (iv) only one endpoint crosses: sqrt(2); neither: 0.
    So **edge = 2 sqrt(2)** (no cap, no factor 3, no D factor), attained in case (i) (two nodes of equal
    attribute and equal degree just below a bin edge). Attribute-row adjacency: node u changes a -> a';
    edges and degrees are unchanged, so one unit moves from (a, j) to (a', j), j = bin(d_u):
    **attr = sqrt(2)**. Same sensitivities as ``degree_hist`` (edge) and ``attr_1way`` (attr).

The sensitivities above are upper bounds; the brute-force tests (``tests/graph_aim/test_measure.py``)
check them on every single-edge change and every single-attribute change of small random graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

from graph_aim.inside.capping import cap_edges
from graph_aim.inside.privacy import gaussian_noise
from graph_aim.transfer.binning import check_bin_edges, degree_bin

KINDS = (
    "degree_hist",
    "clustering_by_degree",
    "joint_degree",
    "attr_1way",
    "attr_2way",
    "attr_edge_pair",
    "attr_degree",
)
ATTR_DEGREE_MAX_BINS = 8
SYMMETRIC_KINDS = ("joint_degree", "attr_edge_pair")
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Spec:
    """A planned measurement: name, kind and public parameters (never data-dependent)."""

    name: str
    kind: str
    params: dict = field(default_factory=dict, hash=False, compare=False)


# --------------------------------------------------------------------------- public structure


def degree_bins(D: int, n_bins: int = 12, last: int | None = None) -> list[int]:
    """Deterministic public degree bin edges: 0, then log-spaced up to D + 1 (last bin holds D).

    If ``last`` is given and exceeds D + 1 the final edge is replaced by ``last`` (open-ended top bin).
    """
    pts = np.unique(np.round(np.geomspace(1, D + 1, n_bins)).astype(np.int64))
    edges = sorted({0, *[int(p) for p in pts], D + 1})
    if last is not None and last > edges[-1]:
        edges[-1] = int(last)
    return edges


def hist_grid(max_degree: int, exact_upto: int | None = None, growth: float = 1.5) -> list[int] | None:
    """Public cell edges for the degree histogram: single degrees below ``exact_upto``, then cells whose
    width grows geometrically (ratio ``growth``) up to ``max_degree + 1``. ``None`` = every degree."""
    if exact_upto is None or exact_upto >= max_degree + 1:
        return None
    edges = list(range(0, int(exact_upto) + 1))
    x = edges[-1]
    while x < max_degree + 1:
        x = max(x + 1, int(math.ceil(x * growth)))
        edges.append(min(x, max_degree + 1))
    return edges


def sensitivity(spec: Spec) -> tuple[float, float]:
    """Proved L2 sensitivity ``(edge, attr)`` of a planned measurement (derivations in the module docstring)."""
    kind, p = spec.kind, spec.params
    cap = p.get("cap")
    if kind == "degree_hist":
        if cap is not None:
            raise ValueError("degree_hist is measured on the uncapped graph (cap must be None)")
        return 2.0 * SQRT2, 0.0
    if kind == "clustering_by_degree":
        cmax = float(p.get("clip", 1.0))
        dmin = p.get("min_degree") or 0
        delta = min(cmax, 1.0 / math.comb(max(int(dmin), 2), 2))
        # per endpoint: with the count column released the change has norm <= sqrt(2 cmax^2 + 2); when the
        # counts are derived from degree_hist (counts_from_hist) only column 0 is released: <= sqrt(2) cmax
        end = SQRT2 * cmax if p.get("counts_from_hist") else math.sqrt(2.0 * cmax**2 + 2.0)
        if cap is None:
            # uncapped: the common-neighbour count is only bounded by the public node count, k <= n - 2
            if not p.get("n_nodes"):
                raise ValueError("uncapped clustering_by_degree needs the public node count params['n_nodes']")
            return 2.0 * end + (int(p["n_nodes"]) - 2) * delta, 0.0
        s1 = 2.0 * end + (int(cap) - 1) * delta
        return 3.0 * s1, 0.0
    if kind == "joint_degree":
        internal = [int(e) for e in p["bin_edges"][1:-1]]
        if cap is not None:
            internal = [e for e in internal if e <= int(cap)]
        r = max((e - 1 for e in internal), default=0)
        s1 = 2.0 * SQRT2 * r + 1.0
        return (s1 if cap is None else 3.0 * s1), 0.0
    if kind in ("attr_1way", "attr_2way"):
        return 0.0, SQRT2
    if kind == "attr_degree":
        if cap is not None:
            raise ValueError("attr_degree is measured on the uncapped graph (cap must be None)")
        return 2.0 * SQRT2, SQRT2
    if kind == "attr_edge_pair":
        if cap is None:
            raise ValueError("attr_edge_pair needs a cap (attribute sensitivity is proportional to it)")
        return 3.0, SQRT2 * int(cap)
    raise ValueError(f"unknown kind {kind!r}")


def workload(
    domains: dict[str, list[str]],
    D: int,
    homophily_attrs: list[str],
    n_bins: int = 12,
    cmax: float = 1.0,
    *,
    caps: dict | None = None,
    max_degree: int | None = None,
    hist_exact_upto: int | None = None,
    hist_growth: float = 1.5,
    hist_threshold: float = 1.0,
    hist_spread_power: float = 0.0,
    cluster_bin_edges: list[int] | None = None,
    min_degree: int | None = None,
    joint_bin_edges: list[int] | None = None,
    n_nodes: int | None = None,
    counts_from_hist: bool = False,
    attr_degree: bool = False,
    attr_degree_bin_edges: list[int] | None = None,
    attr_degree_attrs: list[str] | None = None,
    homophily_groups: list[list[str]] | None = None,
) -> list[Spec]:
    """The fixed public workload for a schema.

    ``D`` is the default cap of every measurement that needs one; ``caps`` overrides it per kind
    (``{"clustering_by_degree": 40, "joint_degree": None, ...}``; a missing kind uses ``D``, an explicit
    ``None`` means uncapped). ``max_degree`` is the public Dmax of ``degree_hist`` (default ``D``).
    ``hist_exact_upto``/``hist_growth`` define the coarse histogram grid, ``hist_threshold`` the
    post-processing cut at ``hist_threshold * sigma`` and ``hist_spread_power`` the shape ``d**-power``
    used to spread a coarse cell over its degrees (see ``postprocess``). ``attr_degree=True`` adds one
    ``attr_degree:<attr>`` table per attribute in ``attr_degree_attrs`` (default all), with degree bins
    ``attr_degree_bin_edges`` (default ``degree_bins(dmax, 6, dmax + 1)``, at most 8 bins). All arguments are public.

    ``homophily_groups`` (list of attribute-name lists) releases one ``attr_edge_pair`` instance per
    group, named ``attr_edge_pair:<a>`` / ``attr_edge_pair:<a>+<b>``; ``None`` keeps the legacy single
    group ``homophily_attrs`` with the bare name ``attr_edge_pair``. Group counts and sizes are
    validated by ``fit.plan`` (``max_homophily_groups``, ``MAX_GROUPS``).
    """
    caps = dict(caps or {})

    def cap_of(kind: str):
        if kind in caps:
            return None if caps[kind] is None else int(caps[kind])
        return int(D)

    cols = list(domains)
    dmax = int(max_degree) if max_degree is not None else int(D)
    c_cap, j_cap = cap_of("clustering_by_degree"), cap_of("joint_degree")
    c_edges = list(cluster_bin_edges) if cluster_bin_edges else degree_bins(c_cap or D, n_bins, dmax + 1)
    j_edges = list(joint_bin_edges) if joint_bin_edges else degree_bins(j_cap or D, n_bins, dmax + 1)
    if min_degree is not None and min_degree not in c_edges:
        raise ValueError("min_degree must be one of the clustering bin edges")
    grid = hist_grid(dmax, hist_exact_upto, hist_growth)
    specs = [
        Spec(
            "degree_hist",
            "degree_hist",
            {"D": dmax, "max_degree": dmax, "cap": None, "hist_edges": grid, "threshold": float(hist_threshold), "spread_power": float(hist_spread_power)},
        ),
        Spec(
            "clustering_by_degree",
            "clustering_by_degree",
            {"bin_edges": c_edges, "clip": cmax, "cap": c_cap, "min_degree": min_degree, "n_nodes": n_nodes, "counts_from_hist": bool(counts_from_hist)},
        ),
        Spec("joint_degree", "joint_degree", {"bin_edges": j_edges, "cap": j_cap}),
    ]
    specs += [Spec(f"attr_1way:{c}", "attr_1way", {"attrs": [c], "cap": None}) for c in cols]
    specs += [
        Spec(f"attr_2way:{a}|{b}", "attr_2way", {"attrs": [a, b], "cap": None})
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
    ]
    if homophily_groups is not None:
        specs += [
            Spec(f"attr_edge_pair:{'+'.join(g)}", "attr_edge_pair", {"attrs": list(g), "cap": cap_of("attr_edge_pair")})
            for g in homophily_groups
        ]
    elif homophily_attrs:
        specs.append(Spec("attr_edge_pair", "attr_edge_pair", {"attrs": list(homophily_attrs), "cap": cap_of("attr_edge_pair")}))
    if attr_degree:  # appended last so that the noise draws of the other kinds do not change
        ad_edges = [int(e) for e in check_bin_edges(attr_degree_bin_edges)] if attr_degree_bin_edges else degree_bins(dmax, 6, dmax + 1)
        if len(ad_edges) - 1 > ATTR_DEGREE_MAX_BINS:
            raise ValueError(f"attr_degree_bin_edges: at most {ATTR_DEGREE_MAX_BINS} bins, got {len(ad_edges) - 1}")
        check_bin_edges(ad_edges)
        ad_attrs = cols if attr_degree_attrs is None else list(attr_degree_attrs)
        unknown = [c for c in ad_attrs if c not in domains]
        if unknown:
            raise ValueError(f"attr_degree_attrs not in domains: {unknown}")
        specs += [Spec(f"attr_degree:{c}", "attr_degree", {"attrs": [c], "cap": None, "bin_edges": list(ad_edges)}) for c in ad_attrs]
    return specs


# --------------------------------------------------------------------------- exact statistics


def _degrees(edges: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(edges.ravel(), minlength=n).astype(np.int64)


def _triangles(edges: np.ndarray, n: int) -> np.ndarray:
    if len(edges) == 0:
        return np.zeros(n)
    a = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    return np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2.0


def _symmetric_counts(ca: np.ndarray, cb: np.ndarray, size: int) -> np.ndarray:
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    u = np.bincount(lo * size + hi, minlength=size * size).reshape(size, size).astype(float)
    return u + u.T - np.diag(np.diag(u))


def _bin_of(deg: np.ndarray, bin_edges) -> np.ndarray:
    """Bin index of each degree; degrees at or above the last edge fall in the (open-ended) last bin."""
    be = np.asarray(bin_edges)
    return np.minimum(np.searchsorted(be, deg, side="right") - 1, len(be) - 2)


def group_codes(attrs: pd.DataFrame, domains: dict[str, list[str]], cols: list[str]) -> tuple[np.ndarray, int]:
    dims = [len(domains[c]) for c in cols]
    codes = np.ravel_multi_index(tuple(attrs[c].to_numpy(dtype=np.int64) for c in cols), dims)
    return codes, int(np.prod(dims))


def exact_values(spec: Spec, capped: np.ndarray, n: int, attrs: pd.DataFrame, domains: dict[str, list[str]]) -> np.ndarray:
    """Noise-free statistic of ``spec``. ``capped`` must already be the graph capped at ``spec.params["cap"]``
    (the raw edge array for ``cap is None``)."""
    k, p = spec.kind, spec.params
    cap = p.get("cap")
    deg = _degrees(capped, n)
    if cap is not None and deg.size and deg.max() > cap:
        raise ValueError(f"edges are not capped to degree {cap}")
    if k == "degree_hist":
        dmax = int(p["max_degree"])
        d = np.minimum(deg, dmax)
        grid = p.get("hist_edges")
        if grid is None:
            return np.bincount(d, minlength=dmax + 1).astype(float)
        cell = np.searchsorted(np.asarray(grid), d, side="right") - 1
        return np.bincount(cell, minlength=len(grid) - 1).astype(float)
    if k in ("clustering_by_degree", "joint_degree"):
        be = p["bin_edges"]
        nb = len(be) - 1
        b = _bin_of(deg, be)
        if k == "joint_degree":
            if len(capped) == 0:
                return np.zeros((nb, nb))
            return _symmetric_counts(b[capped[:, 0]], b[capped[:, 1]], nb)
        t = _triangles(capped, n)
        wedges = deg * (deg - 1) / 2.0
        c = np.divide(t, wedges, out=np.zeros(n), where=wedges > 0)
        c = np.clip(c, 0.0, p.get("clip", 1.0))
        if p.get("min_degree"):
            c = np.where(deg >= int(p["min_degree"]), c, 0.0)
        out = np.zeros((nb, 2))
        out[:, 0] = np.bincount(b, weights=c, minlength=nb)
        out[:, 1] = np.bincount(b, minlength=nb)
        return out
    if k == "attr_1way":
        col = p["attrs"][0]
        return np.bincount(attrs[col].to_numpy(dtype=np.int64), minlength=len(domains[col])).astype(float)
    if k == "attr_2way":
        a, b = p["attrs"]
        ka, kb = len(domains[a]), len(domains[b])
        idx = attrs[a].to_numpy(dtype=np.int64) * kb + attrs[b].to_numpy(dtype=np.int64)
        return np.bincount(idx, minlength=ka * kb).reshape(ka, kb).astype(float)
    if k == "attr_degree":
        col = p["attrs"][0]
        ka, nb = len(domains[col]), len(p["bin_edges"]) - 1
        idx = attrs[col].to_numpy(dtype=np.int64) * nb + degree_bin(deg, p["bin_edges"])
        return np.bincount(idx, minlength=ka * nb).reshape(ka, nb).astype(float)
    if k == "attr_edge_pair":
        g, size = group_codes(attrs, domains, p["attrs"])
        if len(capped) == 0:
            return np.zeros((size, size))
        return _symmetric_counts(g[capped[:, 0]], g[capped[:, 1]], size)
    raise ValueError(f"unknown kind {k!r}")


def capped_for(spec: Spec, graph, salt: int, _cache: dict | None = None) -> np.ndarray:
    """The graph's edges capped at the spec's own cap (cached by cap value when a dict is passed)."""
    cap = spec.params.get("cap")
    if cap is None:
        return graph.edges
    if _cache is not None and cap in _cache:
        return _cache[cap]
    out = cap_edges(graph.edges, graph.n, int(cap), salt)
    if _cache is not None:
        _cache[cap] = out
    return out


def exact_from_graph(spec: Spec, graph, salt: int) -> np.ndarray:
    """Apply the spec's cap to ``graph`` and compute the exact statistic (what the mechanism perturbs)."""
    return exact_values(spec, capped_for(spec, graph, salt), graph.n, graph.attrs, graph.domains)


def released_vector(kind: str, values: np.ndarray, params: dict | None = None) -> np.ndarray:
    """The vector that actually receives noise (upper triangle for symmetric kinds; column 0 only for a
    clustering statistic whose counts come from the degree histogram)."""
    if kind in SYMMETRIC_KINDS:
        return values[np.triu_indices(values.shape[0])]
    if kind == "clustering_by_degree" and (params or {}).get("counts_from_hist"):
        return values[:, 0].ravel()
    return values.ravel()


# --------------------------------------------------------------------------- noise


def add_noise(spec: Spec, values: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise on the released vector; symmetric kinds are noised on the upper triangle and mirrored."""
    if spec.kind in SYMMETRIC_KINDS:
        size = values.shape[0]
        iu = np.triu_indices(size)
        out = np.zeros_like(values, dtype=float)
        out[iu] = values[iu] + gaussian_noise(rng, len(iu[0]), sigma)
        low = np.tril_indices(size, -1)
        out[low] = out.T[low]
        return out
    if spec.kind == "clustering_by_degree" and spec.params.get("counts_from_hist"):
        # only column 0 is released; column 1 is NOT touched here and NOT released: postprocess overwrites it
        # from the degree histogram (never let the exact count column leave the mechanism)
        out = np.zeros_like(values, dtype=float)
        out[:, 0] = values[:, 0] + gaussian_noise(rng, values.shape[0], sigma)
        return out
    return values + gaussian_noise(rng, values.shape, sigma)


# --------------------------------------------------------------------------- post-processing


def _clamp_rescale(x: np.ndarray, total: float) -> np.ndarray:
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x * (total / s) if s > 0 else x


_TAIL_POOL_MAX = 8  # at most this many trailing coarse cells are pooled


def _nonneg_posterior_mean(x: float, sd: float) -> float:
    """E[mu | x] for x ~ N(mu, sd^2) under a flat prior on mu >= 0 (mean of a truncated normal).

    Always positive and equal to x when x >> sd; lower mean-squared error than clamping x at 0.
    """
    z = x / sd
    return x + sd * math.exp(norm.logpdf(z) - norm.logcdf(z))


def _degree_hist_post(v: np.ndarray, spec: Spec, sigma: float, n: int) -> np.ndarray:
    """Cut cells below ``threshold * sigma`` (indistinguishable from noise), rescale to the public n, and
    spread each coarse cell uniformly (or ``d ** -spread_power``) over its degrees (length Dmax + 1).

    Tail pooling (coarse grids only): the trailing run of sub-threshold cells, the hub tail, is NOT
    zeroed. Individually each cell is noise-dominated, but hubs (a handful of nodes of degree in the
    hundreds) carry most of the degree variance, so deleting them shifts degree assortativity from about
    0.06 to 0.6 on ego-Facebook. Instead the raw (unclamped, so unbiased) cell values of the run are
    summed, the sum is clamped at 0 and spread evenly over the run's cells; its noise only grows with
    the square root of the run length. Pure post-processing of the released vector.
    """
    p = spec.params
    raw = np.asarray(v, dtype=float)
    v = np.maximum(raw, 0.0)
    tau = float(p.get("threshold", 0.0))
    grid = p.get("hist_edges")
    if sigma > 0 and tau > 0:
        keep = v >= tau * sigma
        cut = np.where(keep, v, 0.0)
        if cut.sum() > 0:
            v = cut
            if grid is not None:
                last = np.flatnonzero(keep)
                first_tail = int(last[-1]) + 1 if len(last) else 0
                k = len(v) - first_tail
                if 0 < k <= _TAIL_POOL_MAX:
                    v = v.copy()
                    v[first_tail:] = _nonneg_posterior_mean(float(raw[first_tail:].sum()), sigma * math.sqrt(k)) / k
    v = _clamp_rescale(v, n)
    if grid is None:
        return v
    grid = np.asarray(grid)
    width = np.diff(grid)
    power = float(p.get("spread_power", 0.0))
    d = np.arange(grid[0], grid[-1])
    w = np.maximum(d, 1).astype(float) ** (-power)  # public shape inside a cell (power 0 = uniform)
    cell = np.repeat(np.arange(len(width)), width)
    norm = np.bincount(cell, weights=w, minlength=len(width))
    return v[cell] * w / norm[cell]


def _impute_low_degree_clustering(w: np.ndarray, bin_edges, min_degree: int, cmax: float) -> None:
    """In place: bins entirely below ``min_degree`` carry no column-0 measurement; fill them with
    count * (mean clustering of the two lowest informative bins). Pure post-processing."""
    first = int(np.searchsorted(np.asarray(bin_edges), min_degree))
    if first <= 0 or first >= len(w):
        return
    ref = slice(first, min(first + 2, len(w)))
    cbar = float(np.clip(w[ref, 0].sum() / max(w[ref, 1].sum(), 1.0), 0.0, cmax))
    w[:first, 0] = w[:first, 1] * cbar


def postprocess(raw: dict[str, np.ndarray], specs: list[Spec], sigmas: dict[str, float], n: int) -> dict[str, np.ndarray]:
    """Clamp to non-negative, mirror symmetric matrices, rescale to the public n, make 1-way agree with 2-way.

    All of this is post-processing of the noisy values and costs no privacy. Consistency step: each
    attribute's 1-way vector is replaced by the inverse-variance weighted mean of its own noisy
    release and the marginals of every noisy 2-way table containing it (skipped for noise-free runs),
    then clamped and rescaled to sum to n. The 2-way and ``attr_degree`` tables are clamped and rescaled to sum to n but
    not otherwise adjusted. ``degree_hist``: see :func:`_degree_hist_post`. ``clustering_by_degree``:
    counts rescaled to n (or, with ``counts_from_hist``, taken from the processed degree histogram), column 0
    clipped to ``[0, cmax * count]``, bins below ``min_degree`` imputed.
    """
    by_name = {s.name: s for s in specs}
    out: dict[str, np.ndarray] = {}
    for name, v in raw.items():
        spec = by_name[name]
        k = spec.kind
        if k == "degree_hist":
            out[name] = _degree_hist_post(v, spec, sigmas.get(name, 0.0), n)
        elif k == "clustering_by_degree":
            continue  # handled below: may need the processed degree_hist
        elif k in SYMMETRIC_KINDS:
            w = np.maximum(v, 0.0)
            out[name] = (w + w.T) / 2.0
        elif k in ("attr_2way", "attr_degree"):
            out[name] = _clamp_rescale(v, n)
        else:
            out[name] = v
    for spec in specs:
        if spec.kind != "clustering_by_degree" or spec.name not in raw:
            continue
        w = raw[spec.name].copy()
        hist = next((out[t.name] for t in specs if t.kind == "degree_hist" and t.name in out), None)
        if spec.params.get("counts_from_hist") and hist is not None:
            # counts per bin = degree-histogram mass in the bin (post-processing of an already paid release)
            cs = np.concatenate([[0.0], np.cumsum(hist)])
            be = np.minimum(np.asarray(spec.params["bin_edges"]), len(hist))
            w[:, 1] = cs[be[1:]] - cs[be[:-1]]
        else:
            w[:, 1] = _clamp_rescale(w[:, 1], n)
        cmax = spec.params.get("clip", 1.0)
        w[:, 0] = np.clip(w[:, 0], 0.0, cmax * w[:, 1])
        if spec.params.get("min_degree"):
            _impute_low_degree_clustering(w, spec.params["bin_edges"], int(spec.params["min_degree"]), cmax)
        out[spec.name] = w
    # 1-way consistency with 2-way
    for s in specs:
        if s.kind != "attr_1way":
            continue
        col = s.params["attrs"][0]
        sig1 = sigmas[s.name]
        ests = [(raw[s.name], sig1**2)]
        for t in specs:
            if t.kind == "attr_2way" and col in t.params["attrs"]:
                tab = raw[t.name]
                if t.params["attrs"][0] == col:
                    ests.append((tab.sum(axis=1), sigmas[t.name] ** 2 * tab.shape[1]))
                else:
                    ests.append((tab.sum(axis=0), sigmas[t.name] ** 2 * tab.shape[0]))
        if all(var > 0 for _, var in ests):
            w = np.array([1.0 / var for _, var in ests])
            est = sum(wi * e for wi, (e, _) in zip(w, ests)) / w.sum()
        else:
            est = raw[s.name]
        out[s.name] = _clamp_rescale(est, n)
    return out

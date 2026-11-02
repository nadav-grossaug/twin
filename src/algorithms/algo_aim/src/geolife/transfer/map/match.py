"""Pure-python HMM map matcher (Newson-Krumm style) over a ``build_graph`` graph.

States are candidate (edge, offset) projections within ``radius`` metres of each GPS point;
emission = Gaussian in distance (+ road-class prior), transition = exponential in
|route distance - great-circle distance|.  O(T * K^2) bounded dijkstra calls.

Improvements over the first version (all optional, defaults on):
  * ``subsample_m``: GPS at 1-2 s is thinned to >= that spacing before matching;
  * ``mode``-dependent road-class prior (cars do not use footways, walkers avoid trunks, ...);
  * ``collapse_parallel``: sidewalk/footway candidates are dropped where a vehicle road lies
    within ``parallel_m`` (unless mode is walk), so parallel footways collapse onto the road;
  * optional one-way awareness: an edge attribute ``oneway_from`` (node id from which travel is
    allowed) is honoured when present (the stock undirected graph has none);
  * post-processing removes intersection stubs and A-B-A flicker from the segment sequence;
  * a finite jump penalty instead of a hard failure when two layers are disconnected.
"""
from __future__ import annotations

import math

import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, Point

from geolife.transfer.map.graph import haversine

WALK_CLASSES = frozenset({"footway", "pedestrian", "steps", "path", "cycleway", "sidewalk",
                          "crosswalk", "bridleway"})
FAST_CLASSES = frozenset({"motorway", "trunk"})
# log-score penalties added to the emission, same units as 0.5*(d/sigma)^2
CLASS_PRIOR: dict[str | None, dict[str, float]] = {
    "car": {**{c: 6.0 for c in WALK_CLASSES}, "steps": 12.0},
    "bus": {**{c: 6.0 for c in WALK_CLASSES}, "steps": 12.0},
    "train": {**{c: 6.0 for c in WALK_CLASSES}, "steps": 12.0},
    "walk": {c: 3.0 for c in FAST_CLASSES},
    "bike": {"steps": 6.0, "motorway": 8.0, "trunk": 4.0},
    None: {"steps": 1.0},
}


def _index(G, kx, ky):
    edges = list(G.edges(data=True))
    lines = [LineString([(x * kx, y * ky) for x, y in d["geom"].coords]) for _, _, d in edges]
    return edges, lines, STRtree(lines)


def _candidates(idx, lon, lat, radius, k, kx, ky, penalty=None, collapse=False, parallel_m=30.0):
    edges, lines, tree = idx
    pt = Point(lon * kx, lat * ky)
    out = []
    for i in tree.query(pt.buffer(radius)):
        line = lines[i]
        dist = line.distance(pt)
        if dist <= radius:
            cls = edges[i][2].get("class_")
            pen = penalty.get(cls, 0.0) if penalty else 0.0
            out.append((dist, edges[i][0], edges[i][1], line.project(pt), line.length, pen, cls))
    if collapse and out:
        road = [c for c in out if c[6] not in WALK_CLASSES]
        if road:
            dmin = min(c[0] for c in road)
            out = [c for c in out if c[6] not in WALK_CLASSES or c[0] > dmin + parallel_m]
    out.sort(key=lambda t: t[0] + 0.0)
    return out[:k]


def _weight(u, v, d):
    f = d.get("oneway_from")
    if f is not None and f != u:
        return None
    return d["length"]


def _route(G, c1, c2, cache, directed):
    """Network distance between two candidates; inf if unreachable (or against one-way)."""
    (_, u1, v1, o1, l1, *_), (_, u2, v2, o2, l2, *_) = c1, c2
    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
        same = (o2 if (u1, v1) == (u2, v2) else l2 - o2)
        if directed:
            f = G[u1][v1].get("oneway_from")
            if f is not None:
                fwd = same >= o1 if f == u1 else same <= o1
                if not fwd:
                    return math.inf
        return abs(same - o1)
    best = math.inf
    w = _weight if directed else "length"
    for n1, d1, ok1 in ((u1, o1, _leave_ok(G, u1, v1, u1)), (v1, l1 - o1, _leave_ok(G, u1, v1, v1))):
        if not ok1:
            continue
        if n1 not in cache:
            cache[n1] = nx.single_source_dijkstra_path_length(G, n1, cutoff=3000, weight=w)
        for n2, d2 in ((u2, o2), (v2, l2 - o2)):
            if directed and not _enter_ok(G, u2, v2, n2):
                continue
            best = min(best, d1 + cache[n1].get(n2, math.inf) + d2)
    return best


def _leave_ok(G, u, v, node):
    """Leaving the candidate edge (u,v) toward `node`: allowed by one-way?"""
    f = G[u][v].get("oneway_from")
    return f is None or node != f


def _enter_ok(G, u, v, node):
    """Reaching the candidate on edge (u,v) from `node`: allowed by one-way?"""
    f = G[u][v].get("oneway_from")
    return f is None or f == node


class Matcher:
    """HMM matcher with a cached spatial index (build once per graph, reuse for many paths)."""

    def __init__(self, G, lat0: float | None = None, radius=50.0, sigma=10.0, beta=40.0, k=5,
                 subsample_m=15.0, collapse_parallel=True, parallel_m=30.0, use_prior=True,
                 postprocess=True, min_piece_m=15.0, jump_m=500.0):
        self.G = G
        if lat0 is None:
            lat0 = sum(d["lat"] for _, d in G.nodes(data=True)) / max(1, G.number_of_nodes())
        self.ky = 111195.0
        self.kx = self.ky * math.cos(math.radians(lat0))
        self.idx = _index(G, self.kx, self.ky)
        self.radius, self.sigma, self.beta, self.k = radius, sigma, beta, k
        self.subsample_m, self.collapse_parallel, self.parallel_m = subsample_m, collapse_parallel, parallel_m
        self.use_prior, self.postprocess = use_prior, postprocess
        self.min_piece_m, self.jump_m = min_piece_m, jump_m
        self.directed = any("oneway_from" in d for _, _, d in G.edges(data=True))
        self._cache: dict = {}

    def _subsample(self, pts):
        if self.subsample_m <= 0 or len(pts) < 3:
            return list(pts)
        out = [pts[0]]
        for p in pts[1:-1]:
            if haversine(*out[-1], *p) >= self.subsample_m:
                out.append(p)
        out.append(pts[-1])
        return out

    def match(self, points, mode: str | None = None, return_length: bool = False):
        """points: [(lon, lat)] in time order. Returns segment ids (or (ids, length_m))."""
        res = self._match(points, mode)
        return res if return_length else res[0]

    def _match(self, points, mode):
        if len(points) < 1:
            return [], 0.0
        pts = self._subsample(points)
        prior = CLASS_PRIOR.get(mode, CLASS_PRIOR[None]) if self.use_prior else None
        collapse = self.collapse_parallel and mode != "walk"
        layers = [_candidates(self.idx, lo, la, self.radius, self.k, self.kx, self.ky, prior,
                              collapse, self.parallel_m) for lo, la in pts]
        keep = [i for i, c in enumerate(layers) if c]
        if not keep:
            return [], 0.0
        layers = [layers[i] for i in keep]
        pts = [pts[i] for i in keep]
        G, sigma, beta = self.G, self.sigma, self.beta
        cache = self._cache
        if len(cache) > 20000:
            cache.clear()
        em = lambda c: 0.5 * (c[0] / sigma) ** 2 + c[5]
        score = [[-em(c) for c in layers[0]]]
        back: list[list[int]] = [[-1] * len(layers[0])]
        for t in range(1, len(layers)):
            gc = haversine(*pts[t - 1], *pts[t])
            row, brow = [], []
            for c2 in layers[t]:
                best, arg = -math.inf, 0
                for j, c1 in enumerate(layers[t - 1]):
                    r = _route(G, c1, c2, cache, self.directed)
                    if math.isinf(r):
                        r = gc + self.jump_m
                    s = score[-1][j] - abs(r - gc) / beta
                    if s > best:
                        best, arg = s, j
                row.append(best - em(c2))
                brow.append(arg)
            score.append(row)
            back.append(brow)
        i = max(range(len(score[-1])), key=score[-1].__getitem__)
        chosen = []
        for t in range(len(layers) - 1, -1, -1):
            chosen.append(layers[t][i])
            i = back[t][i]
        chosen.reverse()
        pieces: list[tuple] = []  # (segment_id, piece_length) with consecutive duplicates collapsed
        for _, u, v, *_ in chosen:
            d = G[u][v]
            if pieces and pieces[-1][0] == d["segment_id"]:
                if pieces[-1][2] != (u, v) and pieces[-1][2] != (v, u):
                    pieces[-1] = (pieces[-1][0], pieces[-1][1] + d["length"], (u, v))
                continue
            pieces.append((d["segment_id"], d["length"], (u, v)))
        if self.postprocess:
            pieces = _clean(pieces, self.min_piece_m)
        return [p[0] for p in pieces], float(sum(p[1] for p in pieces))


def _clean(pieces, min_piece_m):
    """Drop short intersection stubs between two different segments and A-B-A flicker."""
    changed = True
    while changed and len(pieces) > 2:
        changed = False
        for i in range(1, len(pieces) - 1):
            a, b, c = pieces[i - 1], pieces[i], pieces[i + 1]
            if a[0] == c[0] and b[1] < 4 * min_piece_m:  # A-B-A
                pieces = pieces[:i - 1] + [(a[0], a[1] + c[1] + b[1], c[2])] + pieces[i + 2:]
                changed = True
                break
            if b[1] < min_piece_m:  # stub between two different segments
                pieces = pieces[:i] + pieces[i + 1:]
                changed = True
                break
    return pieces


def match(G, points, radius=50.0, sigma=10.0, beta=40.0, k=5, mode=None, **kw):
    """points: [(lon, lat)]. Returns list of segment_ids (consecutive duplicates collapsed),
    or [] if no candidate exists. Back-compatible wrapper around ``Matcher``."""
    if not points:
        return []
    return Matcher(G, points[0][1], radius, sigma, beta, k, **kw).match(points, mode)


def evaluate_match(G, truth, matched, tol_m=15.0):
    """Segment-id and length-based agreement between true and matched segment sequences.

    recovered_rate = |truth & matched| / |truth|; extra_rate = |matched - truth| / |truth|
    (the P0 convention: 19/27 recovered, 33 extra). Length variants: share of the true
    path's length within tol_m of a matched segment, and share of matched length within tol_m of
    a true segment (geometric, so a parallel footway matched instead of the road is not 'wrong').
    """
    from shapely.ops import unary_union
    ts, ms = set(truth), set(matched)
    out = {"n_true": len(ts), "n_matched": len(ms), "recovered": len(ts & ms),
           "extra": len(ms - ts)}
    out["recovered_rate"] = out["recovered"] / max(1, len(ts))
    out["extra_rate"] = out["extra"] / max(1, len(ts))

    lat0 = sum(d["lat"] for _, d in G.nodes(data=True)) / max(1, G.number_of_nodes())
    ky = 111195.0
    kx = ky * math.cos(math.radians(lat0))
    geo: dict = {}
    for _, _, d in G.edges(data=True):
        geo.setdefault(d["segment_id"], []).append(
            LineString([(x * kx, y * ky) for x, y in d["geom"].coords]))

    def union(ids):
        parts = [g for s in ids for g in geo.get(s, [])]
        return unary_union(parts) if parts else None

    tu, mu = union(ts), union(ms)
    if tu is not None and mu is not None and tu.length > 0 and mu.length > 0:
        out["true_len_covered"] = tu.intersection(mu.buffer(tol_m)).length / tu.length
        out["matched_len_correct"] = mu.intersection(tu.buffer(tol_m)).length / mu.length
    else:
        out["true_len_covered"] = out["matched_len_correct"] = 0.0
    return out

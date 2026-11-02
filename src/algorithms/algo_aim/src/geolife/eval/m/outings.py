"""Method M v2: unsupervised, coordinate-agnostic analysis of OUTINGS (no home/work roles).

Input: raw traces (user -> list[GpsPoint]) and a public map (``MapView``: places table + optional road graph).
Output (``M2Result``):

(a) places of interest per user: stop/endpoint clusters snapped to public map places (``h3:<res-8>`` fallback),
    ranked by distinct outings, then distinct local days, each with a visit-time profile over
    ``TOD_BINS x DAYTYPES`` (local time UTC+8, fixed public choice, see docs/geolife/OUTINGS_PARAMS.md);
(b) common places (visited by >= k distinct users) and common road segments (>= k distinct users; HMM-matched
    segment ids when a road graph is given, else res-9 traversed cells as a proxy);
(c) a co-visit user graph: users linked by significantly shared places of interest (and optionally
    segments). Each shared item adds a rarity (IDF) weight, optionally boosted when both users visit it at
    similar times of day. Significance: Poisson tail of the observed number of shared items under a
    degree-preserving null whose only inputs are the place popularities (users per place) and the users'
    set sizes, both computed from the data being analysed (so M runs identically on synthetic and real
    data); Benjamini-Hochberg over all user pairs. Louvain communities on top.

Segmentation into outings/stops/visits is ``tokenize.outings`` (public parameters). Absolute coordinates are
used only there, to snap to the public map and to run the map matcher; everything M reasons about afterwards
is map-relative (place id, category, local time bin, segment/cell ids).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import chi2

from geolife.transfer.tokenize.outings import OutingsConfig, PlaceIndex, local_bin_idx, user_outings_from_arrays
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.transfer.types import DAYTYPES, LOCAL_UTC_OFFSET_H, TOD_BINS, GpsPoint, UserOutings

NPROF = len(TOD_BINS) * len(DAYTYPES)


@dataclass
class MapView:
    """Public map as M sees it: places table + optional road graph (build_graph-style graph with
    edge attrs segment_id, length, geom). (Moved here from the retired v1 eval/m/core.py.)"""
    places: PlaceTable
    graph: nx.Graph | None = None


@dataclass
class M2Config:
    tok: OutingsConfig = field(default_factory=OutingsConfig)
    # map-resolution place merge: places closer than this (public map distances) are one place, because GPS
    # noise at stops (tens of metres) cannot tell neighbouring map places apart. 0 = off.
    merge_m: float = 60.0
    # places of interest (per user)
    poi_min_outings: int = 2         # distinct outings with a visit to the place
    poi_min_days: int = 2            # distinct local days
    use_endpoints: bool = True       # outing start/end positions count as visits (they are destinations)
    # common places / segments
    k_users: int = 2
    common_min_outings: int = 2      # a user "visits" a place after this many distinct outings ...
    common_min_stop_outings: int = 1  # ... or this many outings with a real stop there (endpoints alone are noisy); 0 = ignore
    k_segments: int = 2
    seg_min_outings: int = 1         # a user "uses" a segment/cell after this many distinct outings
    match: bool = True               # HMM-match outings when the map has a road graph
    match_subsample_m: float = 60.0
    # relations
    rel_min_outings: int = 2         # a user's place set for relations = places with >= this many outings
    # Frozen on dev + dev-sparse draws only (2026-11-02, step 8 task 2; sweep: outings_transfer.rule_study).
    # BH over ALL pairs collapses in sparse regimes (held-out 2024-11 run: 2 edges at AUC 0.87) because pairs
    # that share nothing have p = 1 and can never reject, yet still count as tests; the "sharing" family
    # adapts the test count to the pairs actually testable, and min_shared >= 3 restores the precision.
    rel_alpha: float = 0.2           # BH false-discovery level (family set by rel_universe)
    rel_universe: str = "sharing"    # BH family: "all" = all n(n-1)/2 pairs, "sharing" = pairs sharing >= 1 item
    rel_min_shared: int = 3          # minimum number of shared items
    use_trav: bool = False           # also test shared segments/cells (second channel, Fisher-combined)
    trav_weight: float = 0.2         # weight of one shared segment relative to one shared place
    time_bonus: float = 0.5          # shared-place weight x (1 + bonus * cosine of the two time profiles)
    louvain_resolution: float = 1.0
    seed: int = 0


@dataclass
class UserPlace:
    place_id: str
    category: str
    n_outings: int
    n_days: int
    n_visits: int
    n_stop_outings: int
    dwell_s: float
    profile: np.ndarray            # (NPROF,) visit counts; index = daytype * 4 + tod  (DAYTYPES x TOD_BINS)
    poi: bool = False
    score: float = 0.0

    def profile_dict(self) -> dict:
        p = self.profile
        return {dt: {b: float(p[i * 4 + j]) for j, b in enumerate(TOD_BINS)} for i, dt in enumerate(DAYTYPES)}


@dataclass
class M2Result:
    config: M2Config
    outings: list[UserOutings]
    users: dict[str, list[UserPlace]]                 # every visited place, ranked (score desc)
    place_users: dict[str, set[str]]                  # place -> users with >= common_min_outings outings
    place_category: dict[str, str]
    place_profile: dict[str, np.ndarray]              # place -> pooled (NPROF,) time profile
    common_places: dict[str, set[str]]                # place -> users (>= k_users)
    trav_kind: str                                    # "segment" | "cell"
    trav_users: dict[str, set[str]]                   # segment/cell -> users
    common_segments: dict[str, set[str]]              # (>= k_segments)
    graph: nx.Graph
    communities: list[set[str]]
    stats: dict = field(default_factory=dict)
    merge: dict = field(default_factory=dict)         # place id -> leader id (map-resolution merge)

    def pois(self, user: str) -> list[UserPlace]:
        return [p for p in self.users.get(user, []) if p.poi]

    def common_at(self, k: int) -> set[str]:
        return {p for p, u in self.place_users.items() if len(u) >= k}

    def common_trav_at(self, k: int) -> set[str]:
        return {s for s, u in self.trav_users.items() if len(u) >= k}


# --------------------------------------------------------------------------------------- tokenize

def places_frame(mv: MapView) -> pd.DataFrame | None:
    pt = mv.places
    if pt is None or len(pt) == 0:
        return None
    return pd.DataFrame({"id": pt.ids, "category": pt.categories, "lat": pt.lat, "lon": pt.lon})


def _arrays(tr: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = list(tr)
    if seq and not isinstance(seq[0], GpsPoint):
        seq = [p for t in seq for p in t]
    return (np.fromiter((p.t for p in seq), np.int64, len(seq)),
            np.fromiter((p.lat for p in seq), float, len(seq)),
            np.fromiter((p.lon for p in seq), float, len(seq)))


def tokenize_users(traces: Mapping[str, Sequence], mv: MapView, cfg: M2Config,
                   stats: dict | None = None) -> list[UserOutings]:
    """Raw traces -> outings (user ids kept, since M only sees what it is given). Users under the
    minimum-activity rule are dropped."""
    tok = cfg.tok
    if cfg.match and mv.graph is not None:
        from geolife.transfer.map.match import Matcher
        tok = replace(tok, match=True, matcher=Matcher(mv.graph, subsample_m=cfg.match_subsample_m,
                                                       postprocess=False))
    pidx = PlaceIndex(places_frame(mv))
    out = []
    for uid in sorted(traces):
        t, la, lo = _arrays(traces[uid])
        uo = user_outings_from_arrays(uid, t, la, lo, pidx, tok, (), stats)
        if uo is not None:
            out.append(uo)
    return out


# --------------------------------------------------------------------------------------- places

def place_merge_map(mv: MapView, radius_m: float) -> dict[str, str]:
    """Data-independent leader clustering of the public places: visit places in decreasing map popularity; each
    unassigned place becomes a leader and absorbs the unassigned places within radius_m. Only pairwise map
    distances are used (translation invariant); the result depends on the map, not on any trace."""
    pt = mv.places
    if radius_m <= 0 or pt is None or len(pt) == 0:
        return {}
    from scipy.spatial import cKDTree
    kx = 111195.0 * math.cos(math.radians(float(np.mean(pt.lat))))
    xy = np.c_[np.asarray(pt.lon) * kx, np.asarray(pt.lat) * 111195.0]
    tree = cKDTree(xy)
    order = np.lexsort((np.arange(len(pt)), -np.asarray(pt.popularity, float)))
    lead = np.full(len(pt), -1, np.int64)
    for i in order:
        if lead[i] >= 0:
            continue
        nb = tree.query_ball_point(xy[i], radius_m)
        nb = [j for j in nb if lead[j] < 0]
        lead[nb] = i
    return {pt.ids[j]: pt.ids[lead[j]] for j in range(len(pt)) if lead[j] != j}


def _user_places(uo: UserOutings, cfg: M2Config, merge: Mapping[str, str] | None = None) -> list[UserPlace]:
    outs: dict[str, set[int]] = defaultdict(set)
    souts: dict[str, set[int]] = defaultdict(set)
    days: dict[str, set[int]] = defaultdict(set)
    nvis: Counter = Counter()
    dwell: Counter = Counter()
    prof: dict[str, np.ndarray] = {}
    cat: dict[str, str] = {}
    for oi, o in enumerate(uo.outings):
        for v in o.visits:
            if v.kind != "stop" and not cfg.use_endpoints:
                continue
            p = merge.get(v.place_id, v.place_id) if merge else v.place_id
            outs[p].add(oi)
            tod, dty = local_bin_idx(v.t_arrive)
            if v.kind == "stop":
                souts[p].add(oi)
            days[p].add(int((v.t_arrive + LOCAL_UTC_OFFSET_H * 3600) // 86400))
            nvis[p] += 1
            dwell[p] += v.dwell_s
            cat.setdefault(p, v.category)
            prof.setdefault(p, np.zeros(NPROF))[int(dty) * 4 + int(tod)] += 1
    res = []
    for p in outs:
        up = UserPlace(p, cat[p], len(outs[p]), len(days[p]), nvis[p], len(souts[p]), float(dwell[p]), prof[p])
        up.poi = up.n_outings >= cfg.poi_min_outings and up.n_days >= cfg.poi_min_days
        up.score = up.n_outings + 0.01 * up.n_days + 1e-7 * up.dwell_s
        res.append(up)
    res.sort(key=lambda u: (-u.score, u.place_id))
    return res


def _trav_items(uo: UserOutings, kind: str) -> Counter:
    c: Counter = Counter()
    for o in uo.outings:
        for s in set(o.segments if kind == "segment" else o.cells):
            c[s] += 1
    return c


# --------------------------------------------------------------------------------------- relations

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _channel(user_items: Mapping[str, set]) -> tuple[dict, dict, dict]:
    """Pair tests for one channel (places, or segments/cells).
    -> (pair -> shared items, item -> users, pair -> p-value) for every pair sharing >= 1 item.
    Null (popularity only): user a holds item i independently with probability min(1, |A| m_i / S), m_i = users
    holding i, S = sum m_i (a degree-preserving approximation). The number of shared items is then a
    Poisson-binomial variable with p_i = P_a,i P_b,i; it is matched in mean and variance by a binomial
    (n = mu / q, q = sum p^2 / sum p), which, unlike a Poisson tail, does not penalise near-certain overlap on hubs."""
    users = sorted(user_items)
    item_users: dict[str, list[str]] = defaultdict(list)
    for u in users:
        for i in user_items[u]:
            item_users[i].append(u)
    S = float(sum(len(v) for v in item_users.values()))
    shared: dict[tuple, list] = defaultdict(list)
    for i, us in item_users.items():
        if len(us) >= 2:
            for a, b in combinations(us, 2):
                shared[(a, b)].append(i)
    iu = {i: set(u) for i, u in item_users.items()}
    if S == 0 or not shared:
        return dict(shared), iu, {}
    m = np.array([len(v) for v in item_users.values()], float)
    size = np.array([len(user_items[u]) for u in users], float)
    P = np.minimum(1.0, size[:, None] * m[None, :] / S)
    MU = P @ P.T
    Q2 = (P ** 2) @ (P ** 2).T
    uidx = {u: k for k, u in enumerate(users)}
    pv = {}
    for pr, its in shared.items():
        a, b = uidx[pr[0]], uidx[pr[1]]
        x, mu, q = len(its), float(MU[a, b]), float(Q2[a, b] / max(MU[a, b], 1e-300))
        if x <= 0:
            pv[pr] = 1.0
        elif q >= 1 - 1e-9:
            pv[pr] = 1.0 if x <= mu + 1e-9 else 1e-12
        else:
            n_eff = mu / q
            pv[pr] = 1e-12 if x > n_eff else float(betainc(x, n_eff - x + 1, q))
    return dict(shared), iu, pv


def _bh(pvals: np.ndarray, n_tests: int, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg over n_tests hypotheses of which only len(pvals) can be < 1. Returns keep mask."""
    m = len(pvals)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(pvals)
    thr = alpha * (np.arange(1, m + 1)) / max(n_tests, 1)
    ok = pvals[order] <= thr
    keep = np.zeros(m, bool)
    if ok.any():
        k = np.flatnonzero(ok).max()
        keep[order[:k + 1]] = True
    return keep


def relation_graph(users: Sequence[str], poi_sets: Mapping[str, set], profiles: Mapping[str, Mapping[str, np.ndarray]],
                   trav_sets: Mapping[str, set] | None, cfg: M2Config) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(users)
    n = len(users)
    if n < 2:
        return G
    shared_p, iu_p, pv_p = _channel({u: poi_sets.get(u, set()) for u in users})
    shared_t, iu_t, pv_t = (_channel({u: trav_sets.get(u, set()) for u in users})
                            if (cfg.use_trav and trav_sets is not None) else ({}, {}, {}))
    pairs = sorted(set(shared_p) | set(shared_t))
    if not pairs:
        return G
    pv = np.ones(len(pairs))
    w = np.zeros(len(pairs))
    nsh = np.zeros(len(pairs), int)
    for k, pr in enumerate(pairs):
        a, b = pr
        ps = []
        sp = shared_p.get(pr, [])
        if sp:
            ps.append(pv_p[pr])
            for i in sp:
                idf = math.log(n / len(iu_p[i]))
                pa, pb = profiles.get(a, {}).get(i), profiles.get(b, {}).get(i)
                sim = _cos(pa, pb) if pa is not None and pb is not None else 0.0
                w[k] += idf * (1.0 + cfg.time_bonus * sim)
            nsh[k] += len(sp)
        st = shared_t.get(pr, [])
        if st:
            ps.append(pv_t[pr])
            w[k] += cfg.trav_weight * sum(math.log(n / len(iu_t[i])) for i in st)
            nsh[k] += len(st)
        if len(ps) == 2:
            pv[k] = float(chi2.sf(-2 * sum(math.log(max(p, 1e-300)) for p in ps), 4))
        elif ps:
            pv[k] = ps[0]
    G.graph["pvals"] = {pr: float(pv[k]) for k, pr in enumerate(pairs)}      # every pair sharing >= 1 item
    n_tests = n * (n - 1) // 2 if cfg.rel_universe == "all" else len(pairs)
    keep = _bh(pv, n_tests, cfg.rel_alpha) & (nsh >= cfg.rel_min_shared)
    for k in np.flatnonzero(keep):
        a, b = pairs[k]
        G.add_edge(a, b, weight=float(w[k]), p=float(pv[k]), shared=int(nsh[k]))
    return G


def communities(G: nx.Graph, cfg: M2Config) -> list[set[str]]:
    if G.number_of_edges() == 0:
        return [{x} for x in G.nodes]
    return [set(c) for c in nx.community.louvain_communities(
        G, weight="weight", resolution=cfg.louvain_resolution, seed=cfg.seed)]


# --------------------------------------------------------------------------------------- run

def analyze(uos: Sequence[UserOutings], cfg: M2Config | None = None, has_graph: bool = False,
            merge: Mapping[str, str] | None = None) -> M2Result:
    """Everything after tokenization (also used by the split-half study)."""
    cfg = cfg or M2Config()
    kind = "segment" if (has_graph and cfg.match) else "cell"
    users: dict[str, list[UserPlace]] = {}
    place_users: dict[str, set[str]] = defaultdict(set)
    cat: dict[str, str] = {}
    pprof: dict[str, np.ndarray] = {}
    trav_users: dict[str, set[str]] = defaultdict(set)
    poi_sets: dict[str, set] = {}
    prof_by_user: dict[str, dict] = {}
    trav_sets: dict[str, set] = {}
    for uo in uos:
        ups = _user_places(uo, cfg, merge)
        users[uo.user] = ups
        poi_sets[uo.user] = {p.place_id for p in ups if p.n_outings >= cfg.rel_min_outings and p.n_days >= cfg.poi_min_days}
        prof_by_user[uo.user] = {p.place_id: p.profile for p in ups}
        for p in ups:
            cat.setdefault(p.place_id, p.category)
            pprof[p.place_id] = pprof.get(p.place_id, 0) + p.profile
            if p.n_outings >= cfg.common_min_outings or (cfg.common_min_stop_outings > 0 and p.n_stop_outings >= cfg.common_min_stop_outings):
                place_users[p.place_id].add(uo.user)
        tc = _trav_items(uo, kind)
        trav_sets[uo.user] = {s for s, c in tc.items() if c >= cfg.seg_min_outings}
        for s in trav_sets[uo.user]:
            trav_users[s].add(uo.user)
    common = {p: u for p, u in place_users.items() if len(u) >= cfg.k_users}
    common_seg = {s: u for s, u in trav_users.items() if len(u) >= cfg.k_segments}
    G = relation_graph([u.user for u in uos], poi_sets, prof_by_user, trav_sets, cfg)
    comms = communities(G, cfg)
    stats = {"n_users": len(uos), "n_outings": sum(len(u.outings) for u in uos),
             "n_places": len(place_users), "n_common_places": len(common),
             "n_common_trav": len(common_seg), "n_edges": G.number_of_edges()}
    return M2Result(cfg, list(uos), users, dict(place_users), cat, pprof, common, kind, dict(trav_users),
                    common_seg, G, comms, stats)


def run_m2(traces: Mapping[str, Sequence], mv: MapView, cfg: M2Config | None = None) -> M2Result:
    cfg = cfg or M2Config()
    st: dict = {}
    uos = tokenize_users(traces, mv, cfg, st)
    merge = place_merge_map(mv, cfg.merge_m)
    res = analyze(uos, cfg, has_graph=mv.graph is not None, merge=merge)
    res.merge = merge
    res.stats.update({"users_in": len(traces), **st})
    return res


# --------------------------------------------------------------------------------------- statistics

def _summ(x) -> dict:
    x = np.asarray(list(x), float)
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}


def distribution_stats(res: M2Result) -> dict:
    """Label-free statistics of PLAN_OUTINGS.md section 4.2, computed from M's own outputs/tokens only."""
    uos = res.outings
    n_out = [len(u.outings) for u in uos]
    n_stops = [sum(v.kind == "stop" for v in o.visits) for u in uos for o in u.outings]
    length = [o.length_m / 1000 for u in uos for o in u.outings]
    dur = [(o.t_end - o.t_start) / 60 for u in uos for o in u.outings]
    ndist = [len(res.users[u.user]) for u in uos]
    npoi = [len(res.pois(u.user)) for u in uos]
    rev = []
    for u in uos:
        seen: set = set()
        tot = r = 0
        for o in u.outings:
            for v in o.visits:
                if v.kind == "stop":
                    tot += 1
                    r += v.place_id in seen
            seen.update(v.place_id for v in o.visits)
        if tot:
            rev.append(r / tot)
    upp = Counter(len(s) for s in res.place_users.values())
    ups = Counter(len(s) for s in res.trav_users.values())
    deg = [d for _, d in res.graph.degree()]
    days = [len({int((o.t_start + LOCAL_UTC_OFFSET_H * 3600) // 86400) for o in u.outings}) for u in uos]
    span = [(max(o.t_start for o in u.outings) - min(o.t_start for o in u.outings)) / 86400 for u in uos]
    return {"n_users": len(uos), "outings_per_user": _summ(n_out), "stops_per_outing": _summ(n_stops),
            "length_km": _summ(length), "duration_min": _summ(dur), "distinct_places_per_user": _summ(ndist),
            "pois_per_user": _summ(npoi), "stop_revisit_rate": _summ(rev), "active_days": _summ(days),
            "window_days": _summ(span),
            "users_per_place_hist": dict(sorted(upp.items())), "users_per_trav_hist": dict(sorted(ups.items())),
            "share_places_ge2_users": float(sum(v for k, v in upp.items() if k >= 2) / max(1, sum(upp.values()))),
            "degree": _summ(deg), "degree_list": deg,
            "share_users_with_relation": float(np.mean([d > 0 for d in deg])) if deg else float("nan"),
            "n_communities_ge2": int(sum(len(c) >= 2 for c in res.communities)), "trav_kind": res.trav_kind}


def tv_hist(a: Mapping, b: Mapping, cap: int = 10) -> float:
    def norm(h):
        v = np.zeros(cap + 1)
        for k, c in h.items():
            v[min(int(k), cap)] += c
        return v / max(v.sum(), 1)
    return float(0.5 * np.abs(norm(a) - norm(b)).sum())

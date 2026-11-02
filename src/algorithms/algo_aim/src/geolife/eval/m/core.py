"""Method M: unsupervised, coordinate-agnostic discovery of main places, shared places/crossings and
shared-anchor user relations from raw GPS traces + a public map.

Inputs: ``traces`` (user -> list[GpsPoint]) and a ``MapView`` (places table + optional road graph).
Absolute coordinates are only used to *snap* stays onto the public map (place ids / H3 fallback)
and to run the map matcher; every feature that M reasons about afterwards is map-relative:
place id, category, dwell, visit days, hours-of-day and road-segment ids. The only dataset-level
nuisance parameter, the local night window (time zone), is estimated from the activity rhythm of the
traces themselves, not from longitude.

Pipeline: clean (rolling-median) -> thin -> stay detection (tokenize.stays) -> per-user clustering of
stays into visited places -> snap to map place (nearest) -> per-place features (dwell, days, night
share, median stay) -> main places (return frequency + dwell) and home/work-type roles ->
shared places (>= k users) and, if a road graph is given, shared road segments from map-matched trips
-> weighted user graph from shared anchors -> Louvain communities.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from geolife.transfer.tokenize.stays import M_PER_DEG, PlaceTable, RawStay, detect_stays, h3_cell, snap_stay, thin
from geolife.transfer.tokenize.trips import segment_trips
from geolife.transfer.types import GpsPoint


@dataclass
class MConfig:
    # cleaning / stays
    median_win: int = 5
    thin_s: int = 15
    stay_dist_m: float = 120.0
    stay_min_s: int = 900
    max_gap_s: int = 1800
    merge_gap_s: int = 3 * 3600    # same-place stays closer than this are one visit
    # place clustering and snapping
    cluster_m: float = 70.0
    snap_radius_m: float = 150.0
    h3_res: int = 9
    # main places
    min_days: int = 2             # a main place is revisited on >= this many distinct days
    min_dwell_s: int = 3600
    w_dwell: float = 0.5          # score = w_dwell * dwell share + (1 - w_dwell) * day share
    # roles
    night_hours: float = 6.0
    home_min_nights: int = 2
    home_min_night_share: float = 0.5   # of the user's observed nights spent at the place
    work_min_days: int = 2
    work_min_median_s: int = 14400
    work_max_night_share: float = 0.3
    # shared places / crossings / relations
    k_users: int = 2
    k_segments: int = 2
    min_place_visits: int = 2     # a user "uses" a place if it has >= this many stays
    match_trips: bool = True
    match_subsample_m: float = 60.0
    trip_min_len_m: float = 300.0
    min_shared_weight: float = 0.0
    louvain_resolution: float = 1.0
    seed: int = 0


@dataclass
class MapView:
    places: PlaceTable
    graph: nx.Graph | None = None      # build_graph-style graph (edge attrs segment_id, length, geom)


@dataclass
class UserPlace:
    place_id: str
    category: str
    n_visits: int
    n_days: int
    dwell_s: float
    night_frac: float
    night_count: int
    median_stay_s: float
    mean_hour: float
    score: float = 0.0
    nights: frozenset = frozenset()
    role: str = "other"          # home | work | other
    main: bool = False


@dataclass
class MResult:
    config: MConfig
    night_start_h: float
    users: dict[str, list[UserPlace]]
    place_users: dict[str, set[str]]              # place_id -> users that use it (>= min_place_visits stays)
    place_category: dict[str, str]
    shared_places: dict[str, set[str]]            # place_id -> users (>= k_users)
    shared_segments: dict[str, set[str]]          # segment_id -> users (>= k_segments)
    segment_users: dict[str, set[str]]            # every matched segment -> users
    anchor_users: dict[str, set[str]]             # home/work-type place -> users
    graph: nx.Graph
    communities: list[set[str]]
    stats: dict = field(default_factory=dict)

    def main_places(self, user: str) -> list[UserPlace]:
        return [p for p in self.users.get(user, []) if p.main]


# ------------------------------------------------------------------------------ helpers

def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(a) < w:
        return a
    pad = w // 2
    ap = np.concatenate([np.repeat(a[:1], pad), a, np.repeat(a[-1:], pad)])
    return np.median(np.lib.stride_tricks.sliding_window_view(ap, w), axis=1)


def clean_points(points: Sequence[GpsPoint], win: int) -> list[GpsPoint]:
    """Sort, de-duplicate timestamps, rolling-median filter positions (kills isolated outliers)."""
    pts = sorted({p.t: p for p in points}.values(), key=lambda p: p.t)
    if len(pts) < 3 or win <= 1:
        return pts
    lat = _rolling_median(np.array([p.lat for p in pts]), win)
    lon = _rolling_median(np.array([p.lon for p in pts]), win)
    return [GpsPoint(p.t, float(a), float(b)) for p, a, b in zip(pts, lat, lon)]


def estimate_night_start(traces: Mapping[str, Sequence[GpsPoint]], night_hours: float = 6.0,
                         max_users: int = 400) -> float:
    """Start hour (UTC, 0-24) of the quietest circular `night_hours` window of movement.
    Movement = consecutive samples <= 120 s apart that moved >= 60 m (speed independent of coordinates'
    meaning). The window is found on the dataset, so the method does not need a time zone."""
    hist = np.zeros(24)
    for uid in sorted(traces)[:max_users]:
        pts = traces[uid]
        if len(pts) < 2:
            continue
        t = np.array([p.t for p in pts], float)
        lat = np.array([p.lat for p in pts])
        lon = np.array([p.lon for p in pts])
        kx = M_PER_DEG * math.cos(math.radians(float(lat.mean())))
        step = np.hypot(np.diff(lon) * kx, np.diff(lat) * M_PER_DEG)
        dt = np.diff(t)
        mov = (dt > 0) & (dt <= 120) & (step >= 60) & (step / np.maximum(dt, 1) < 70)
        h = ((t[1:][mov] % 86400) // 3600).astype(int)
        hist += np.bincount(h, minlength=24)
    # also count pure presence so empty nights are not mistaken for data holes
    w = int(round(night_hours))
    sums = [sum(hist[(s + i) % 24] for i in range(w)) for s in range(24)]
    best = int(np.argmin(sums))
    # centre ties on a flat minimum
    ties = [s for s in range(24) if sums[s] <= sums[best] * 1.0001 + 1e-9]
    if len(ties) > 1:
        ang = np.angle(np.mean(np.exp(2j * math.pi * np.array(ties) / 24)))
        best = int(round((ang % (2 * math.pi)) / (2 * math.pi) * 24)) % 24
    return float(best)


def _night_overlap(t0: float, t1: float, ns: float, nh: float) -> tuple[float, int]:
    """(seconds overlapping the daily night window, number of distinct nights touched)."""
    tot, nights = 0.0, set()
    day0 = int((t0 - ns * 3600) // 86400) - 1
    day1 = int((t1 - ns * 3600) // 86400) + 1
    for d in range(day0, day1 + 1):
        a = d * 86400 + ns * 3600
        b = a + nh * 3600
        ov = min(t1, b) - max(t0, a)
        if ov > 0:
            tot += ov
            if ov >= 1800:
                nights.add(d)
    return tot, len(nights)


def _cluster_xy(xy: np.ndarray, radius: float) -> np.ndarray:
    """Single-linkage components at `radius` (union-find over a KD-tree)."""
    n = len(xy)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n > 1:
        from scipy.spatial import cKDTree
        for i, j in cKDTree(xy).query_pairs(radius):
            parent[find(i)] = find(j)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


# ------------------------------------------------------------------------------ per user

def _user_places(uid: str, points: Sequence[GpsPoint], mv: MapView, cfg: MConfig, ns: float,
                 matcher) -> tuple[list[UserPlace], list[tuple[str, ...]]]:
    pts = thin(clean_points(points, cfg.median_win), cfg.thin_s)
    if len(pts) < 2:
        return [], []
    raws = detect_stays(pts, cfg.stay_dist_m, cfg.stay_min_s, cfg.max_gap_s)
    if not raws:
        return [], []
    lat0 = float(np.mean([r.lat for r in raws]))
    kx = M_PER_DEG * math.cos(math.radians(lat0))
    xy = np.array([[r.lon * kx, r.lat * M_PER_DEG] for r in raws])
    lab = _cluster_xy(xy, cfg.cluster_m)
    seg_lists: list[tuple[str, ...]] = []
    if matcher is not None:
        snapped = [snap_stay(r, mv.places, cfg.h3_res, cfg.snap_radius_m) for r in raws]
        trips, _ = segment_trips(pts, raws, snapped, (), cfg.trip_min_len_m, cfg.max_gap_s,
                                 matcher)
        seg_lists = [t.segments for t in trips if t.segments]
    # consecutive stays in the same place separated by a short hole (noise or signal gap) are one visit
    ivs: list[list] = []     # [label, t0, t1, sum_lat*w, sum_lon*w, sum_w]
    for r, c in sorted(zip(raws, lab), key=lambda z: z[0].t_start):
        w = max(1.0, r.t_end - r.t_start)
        if ivs and ivs[-1][0] == c and r.t_start - ivs[-1][2] <= cfg.merge_gap_s:
            iv = ivs[-1]
            iv[2] = max(iv[2], r.t_end)
            iv[3] += r.lat * w; iv[4] += r.lon * w; iv[5] += w
        else:
            ivs.append([c, r.t_start, r.t_end, r.lat * w, r.lon * w, w])
    out: list[UserPlace] = []
    for c in sorted(set(lab)):
        mine = [iv for iv in ivs if iv[0] == c]
        durs = np.array([iv[2] - iv[1] for iv in mine], float)
        W = sum(iv[5] for iv in mine)
        lat = sum(iv[3] for iv in mine) / W
        lon = sum(iv[4] for iv in mine) / W
        hit = mv.places.nearest(lat, lon, cfg.snap_radius_m)
        pid, cat = hit if hit else (f"h3:{h3_cell(lat, lon, cfg.h3_res)}", "unknown")
        night = 0.0
        nights: set[int] = set()
        days: set[int] = set()
        hours = []
        for _, t0, t1, *_ in mine:
            night += _night_overlap(t0, t1, ns, cfg.night_hours)[0]
            days.add(int(((t0 + t1) / 2 - ns * 3600) // 86400))
            hours.append((t0 % 86400) / 3600)
            for d in range(int((t0 - ns * 3600) // 86400) - 1, int((t1 - ns * 3600) // 86400) + 2):
                a_ = d * 86400 + ns * 3600
                if min(t1, a_ + cfg.night_hours * 3600) - max(t0, a_) >= 1800:
                    nights.add(d)
        ang = np.angle(np.mean(np.exp(2j * math.pi * np.array(hours) / 24)))
        out.append(UserPlace(pid, cat, len(mine), len(days), float(durs.sum()),
                             night / max(1.0, float(durs.sum())), len(nights),
                             float(np.median(durs)), float((ang % (2 * math.pi)) / (2 * math.pi) * 24),
                             nights=frozenset(nights)))
    # two clusters can snap to the same map place: merge them
    merged: dict[str, UserPlace] = {}
    for p in out:
        q = merged.get(p.place_id)
        if q is None:
            merged[p.place_id] = p
        else:
            tot = q.dwell_s + p.dwell_s
            merged[p.place_id] = UserPlace(
                p.place_id, q.category, q.n_visits + p.n_visits, q.n_days + p.n_days, tot,
                (q.night_frac * q.dwell_s + p.night_frac * p.dwell_s) / max(1.0, tot),
                len(q.nights | p.nights), max(q.median_stay_s, p.median_stay_s), q.mean_hour,
                nights=q.nights | p.nights)
    ups = list(merged.values())
    _rank_and_roles(ups, cfg)
    return ups, seg_lists


def _rank_and_roles(ups: list[UserPlace], cfg: MConfig) -> None:
    if not ups:
        return
    tot_dwell = sum(p.dwell_s for p in ups)
    max_days = max(p.n_days for p in ups)
    for p in ups:
        p.score = cfg.w_dwell * p.dwell_s / max(1.0, tot_dwell) + (1 - cfg.w_dwell) * p.n_days / max(1, max_days)
        p.main = p.n_days >= cfg.min_days and p.dwell_s >= cfg.min_dwell_s
    ups.sort(key=lambda p: -p.score)
    all_nights = set().union(*(p.nights for p in ups))
    homes = [p for p in ups if len(p.nights) >= cfg.home_min_nights
             and len(p.nights) >= cfg.home_min_night_share * len(all_nights)]
    home = max(homes, key=lambda p: (len(p.nights), p.dwell_s)) if homes else None
    if home:
        home.role, home.main = "home", True
    works = [p for p in ups if p is not home and p.n_days >= cfg.work_min_days
             and p.median_stay_s >= cfg.work_min_median_s
             and len(p.nights) <= cfg.work_max_night_share * max(1, len(all_nights))]
    if works:
        w = max(works, key=lambda p: p.dwell_s)
        w.role, w.main = "work", True


# ------------------------------------------------------------------------------ run M

def make_matcher(mv: MapView, cfg: MConfig):
    if mv.graph is None or not cfg.match_trips:
        return None
    from geolife.transfer.map.match import Matcher
    m = Matcher(mv.graph, subsample_m=cfg.match_subsample_m, postprocess=False)

    def f(seg: Sequence[GpsPoint], mode: str):
        ids, ln = m.match([(p.lon, p.lat) for p in seg], mode, return_length=True)
        return tuple(ids), ln
    return f


def run_m(traces: Mapping[str, Sequence[GpsPoint]], mv: MapView, cfg: MConfig | None = None
          ) -> MResult:
    cfg = cfg or MConfig()
    ns = estimate_night_start(traces, cfg.night_hours)
    matcher = make_matcher(mv, cfg)
    users: dict[str, list[UserPlace]] = {}
    seg_users: dict[str, set[str]] = defaultdict(set)
    seg_count: Counter = Counter()
    for uid in sorted(traces):
        ups, segs = _user_places(uid, traces[uid], mv, cfg, ns, matcher)
        users[uid] = ups
        for sl in segs:
            for s in set(sl):
                seg_users[s].add(uid)
                seg_count[s] += 1
    place_users: dict[str, set[str]] = defaultdict(set)
    cat: dict[str, str] = {}
    anchors: dict[str, set[str]] = defaultdict(set)
    for uid, ups in users.items():
        for p in ups:
            cat[p.place_id] = p.category
            if p.n_visits >= cfg.min_place_visits:
                place_users[p.place_id].add(uid)
            if p.role in ("home", "work"):
                anchors[p.place_id].add(uid)
    shared_places = {p: u for p, u in place_users.items() if len(u) >= cfg.k_users}
    shared_segments = {s: u for s, u in seg_users.items() if len(u) >= cfg.k_segments}
    G = relation_graph(users, anchors, cfg)
    comms = communities(G, cfg)
    stats = {"n_users": len(traces), "night_start_h": ns,
             "n_places": len(place_users), "n_shared_places": len(shared_places),
             "n_shared_segments": len(shared_segments)}
    return MResult(cfg, ns, users, dict(place_users), cat, shared_places, shared_segments, dict(seg_users),
                   dict(anchors), G, comms, stats)


def relation_graph(users: Mapping[str, list[UserPlace]], anchors: Mapping[str, set[str]],
                   cfg: MConfig) -> nx.Graph:
    """D2: users sharing a home/work-type place are linked; weight sums an IDF-style term per shared
    anchor (a place used as anchor by few users is stronger evidence than a huge one)."""
    G = nx.Graph()
    G.add_nodes_from(users)
    n = max(2, len(users))
    for pid, us in anchors.items():
        if len(us) < 2:
            continue
        idf = math.log(n / len(us)) + 1.0
        for a, b in combinations(sorted(us), 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += idf
                G[a][b]["places"].append(pid)
            else:
                G.add_edge(a, b, weight=idf, places=[pid])
    drop = [(a, b) for a, b, d in G.edges(data=True) if d["weight"] < cfg.min_shared_weight]
    G.remove_edges_from(drop)
    return G


def communities(G: nx.Graph, cfg: MConfig) -> list[set[str]]:
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes]
    return [set(c) for c in nx.community.louvain_communities(
        G, weight="weight", resolution=cfg.louvain_resolution, seed=cfg.seed)]

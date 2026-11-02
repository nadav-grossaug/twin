"""OvertureProvider: the generate.maps.MapProvider protocol on real Overture data.

Overture release 2026-10-21.0 places schema (checked 2026-11): there is NO ``categories`` column any more. Use
``taxonomy`` = STRUCT(primary, hierarchy[], alternates[]) (hierarchy[0] is the top level, e.g. ``food_and_drink``,
hierarchy[1] the second level, e.g. ``casual_eatery``) and ``basic_category`` (VARCHAR, a mid-level label).
``coarse_category`` maps these onto a small taxonomy:
residential, office, school, food, shop, transit, leisure, health, other.

Popularity proxy for places (Overture has no visit counts):
    weight = confidence * (1 + 0.5 * has_brand) * (1 + 0.3 * ln(1 + n_colocated_pois_within_100m))
(brand = chain outlets are busier; co-located POIs = commercial centre). Homes come from building footprints:
building centroids (bbox centre) sampled without replacement with probability proportional to footprint area
(bbox area in m^2, strongly down-weighted when the class is a known non-residential one); weight = footprint area.
Roads: Overture road segments minus pedestrian/cycle classes, undirected, largest connected component.
Routing: scipy Dijkstra, one single-source run per origin node (LRU cached, undirected symmetry used so that
via-point detours a->v->b cost two cached sources, not one run per candidate).
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from geolife.transfer.maps import MapPlace, Route
from geolife.transfer.map.graph import build_graph
from geolife.outside.map.overture import DEFAULT_RELEASE, BBox

CACHE_DIR = Path("geolife/outside/cache")   # public-map cache (public data; inside may read it too)
CATEGORIES = ("residential", "office", "school", "food", "shop", "transit", "leisure", "health", "other")
ROAD_EXCLUDE = {"footway", "steps", "pedestrian", "cycleway", "path", "bridleway", "track", "sidewalk", "crosswalk"}
NONRES_BUILDING = {"commercial", "industrial", "office", "retail", "school", "university", "hospital", "church",
                   "civic", "transportation", "warehouse", "service", "garage", "parking", "government",
                   "religious", "kindergarten", "college", "stadium", "supermarket", "public", "mosque", "temple",
                   "train_station", "hangar", "shed", "roof", "greenhouse", "barn"}

# top-level taxonomy (hierarchy[0]) -> coarse; keywords below refine first.
_L0 = {
    "food_and_drink": "food", "shopping": "shop", "education": "school", "health_care": "health",
    "arts_and_entertainment": "leisure", "sports_and_recreation": "leisure", "travel_and_transportation": "transit",
    "lifestyle_services": "shop", "services_and_business": "office", "business_to_business": "office",
    "community_and_government": "office", "cultural_and_historic": "leisure", "lodging": "other",
    "geographic_entities": "other", "public_service": "office", "financial_service": "office",
}
_KEYWORDS = (  # substring of any hierarchy element / basic_category, checked in order
    ("residential", "residential"), ("apartment", "residential"), ("housing", "residential"),
    ("school", "school"), ("university", "school"), ("college", "school"), ("kindergarten", "school"),
    ("hospital", "health"), ("clinic", "health"), ("pharmacy", "health"), ("doctor", "health"), ("dentist", "health"),
    ("subway", "transit"), ("metro", "transit"), ("train_station", "transit"), ("bus_station", "transit"),
    ("bus_stop", "transit"), ("transit", "transit"), ("airport", "transit"),
    ("restaurant", "food"), ("cafe", "food"), ("coffee", "food"), ("bar", "food"), ("bakery", "food"),
    ("parking", "other"), ("park", "leisure"), ("gym", "leisure"), ("fitness", "leisure"), ("cinema", "leisure"), ("museum", "leisure"),
    ("supermarket", "shop"), ("retail", "shop"), ("store", "shop"), ("shop", "shop"), ("mall", "shop"),
    ("office", "office"), ("coworking", "office"), ("company", "office"), ("bank", "office"),
)


def coarse_category(hierarchy, basic_category=None) -> str:
    """Map an Overture ``taxonomy.hierarchy`` (+ ``basic_category``) to the coarse generator taxonomy."""
    h = [str(x).lower() for x in (hierarchy if hierarchy is not None else [])]
    if basic_category:
        h.append(str(basic_category).lower())
    for kw, cat in _KEYWORDS:
        if any(kw in x for x in h):
            return cat
    if h and h[0] in _L0:
        return _L0[h[0]]
    return "other"


# --------------------------------------------------------------------------- cached public-map tables
def _cache_file(cache_dir, release, kind, bbox) -> Path:
    key = hashlib.sha1(repr((release, kind, tuple(bbox))).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{kind}_{release}_{key}.parquet"


def _query(kind: str, cols: str, theme: str, bbox: BBox, cache_dir, release: str, root: str | None,
           fetch=None) -> pd.DataFrame:
    """Cached parquet if present; else ``fetch(theme, kind, cols, bbox, release, root)`` (the duckdb fetcher lives in
    geolife/outside/fetch_public_map.py, never in shipped code) and cache its result."""
    f = _cache_file(cache_dir, release, kind, bbox)
    if f.exists():
        return pd.read_parquet(f)
    if fetch is None:
        raise FileNotFoundError(f"{f} is not cached; run geolife/outside/fetch_public_map.py for this bbox "
                                "(public data, needs network) or pass the frames to OvertureProvider")
    df = fetch(theme, kind, cols, bbox, release, root)
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f)
    return df


def fetch_places(bbox: BBox, cache_dir=CACHE_DIR, release=DEFAULT_RELEASE, root=None, fetch=None) -> pd.DataFrame:
    """Raw light places table: id, confidence, has_brand, xmin..ymax, hierarchy, basic_category."""
    cols = ("id, confidence, (brand.wikidata IS NOT NULL OR brand.names.\"primary\" IS NOT NULL) AS has_brand, "
            "bbox.xmin AS xmin, bbox.xmax AS xmax, bbox.ymin AS ymin, bbox.ymax AS ymax, "
            "taxonomy.hierarchy AS hierarchy, basic_category")
    df = _query("place", cols, "places", bbox, cache_dir, release, root, fetch)
    df["hierarchy"] = df["hierarchy"].map(lambda a: list(a) if isinstance(a, (list, tuple, np.ndarray)) else [])
    return df


def fetch_buildings(bbox: BBox, cache_dir=CACHE_DIR, release=DEFAULT_RELEASE, root=None, fetch=None) -> pd.DataFrame:
    cols = 'id, "class" AS cls, bbox.xmin AS xmin, bbox.xmax AS xmax, bbox.ymin AS ymin, bbox.ymax AS ymax'
    return _query("building", cols, "buildings", bbox, cache_dir, release, root, fetch)


def fetch_roads(bbox: BBox, cache_dir=CACHE_DIR, release=DEFAULT_RELEASE, root=None, fetch=None):
    seg = _query("segment", 'id, subtype, "class", connectors, geometry', "transportation", bbox,
                 cache_dir, release, root, fetch)
    con = _query("connector", "id, geometry", "transportation", bbox, cache_dir, release, root, fetch)
    return seg, con


def place_table(places: pd.DataFrame) -> pd.DataFrame:
    """Raw places -> normalised table: id, category, lat, lon, popularity (the fetch_public_map.py output)."""
    if places.empty:
        return pd.DataFrame({"id": [], "category": [], "lat": [], "lon": [], "popularity": []})
    lon = ((places["xmin"] + places["xmax"]) / 2).to_numpy(float)
    lat = ((places["ymin"] + places["ymax"]) / 2).to_numpy(float)
    cat = [coarse_category(h, b) for h, b in zip(places["hierarchy"], places["basic_category"])]
    conf = places["confidence"].fillna(0.5).clip(0.01, 1.0).to_numpy(float)
    brand = places["has_brand"].fillna(False).astype(float).to_numpy()
    xy = np.column_stack([(lon - lon.mean()) * 111_320.0 * math.cos(math.radians(lat.mean())),
                          (lat - lat.mean()) * 111_320.0])
    nco = np.array([len(x) - 1 for x in cKDTree(xy).query_ball_point(xy, 100.0)], float)
    pop = conf * (1 + 0.5 * brand) * (1 + 0.3 * np.log1p(nco))
    return pd.DataFrame({"id": places["id"].astype(str).to_numpy(), "category": cat, "lat": lat, "lon": lon,
                         "popularity": pop})


# --------------------------------------------------------------------------- provider
class OvertureProvider:
    """MapProvider on Overture data for ``bbox`` = (min_lon, min_lat, max_lon, max_lat).

    Pass pre-built frames (``places`` raw or normalised, ``buildings``, ``segments``, ``connectors``) to skip any
    any fetch; otherwise they are read through ``fetch_*`` from the cached parquet under ``cache_dir`` (filled by
    geolife/outside/fetch_public_map.py, or by a ``fetch`` callable passed here).
    """

    def __init__(self, bbox: BBox, *, cache_dir=CACHE_DIR, release=DEFAULT_RELEASE, root=None,
                 places: pd.DataFrame | None = None, buildings: pd.DataFrame | None = None,
                 segments: pd.DataFrame | None = None, connectors: pd.DataFrame | None = None, fetch=None,
                 n_homes: int = 1500, seed: int = 0, max_poi: int | None = None, cache_sources: int = 256):
        self.bbox = tuple(bbox)
        x0, y0, x1, y1 = self.bbox
        self.origin = ((y0 + y1) / 2, (x0 + x1) / 2)
        self._kx = 111_320.0 * math.cos(math.radians(self.origin[0]))
        if places is None:
            places = fetch_places(self.bbox, cache_dir, release, root, fetch)
        if buildings is None:
            buildings = fetch_buildings(self.bbox, cache_dir, release, root, fetch)
        if segments is None or connectors is None:
            segments, connectors = fetch_roads(self.bbox, cache_dir, release, root, fetch)
        self._build_graph(segments, connectors)
        tab = places if "category" in places else place_table(places)
        if max_poi and len(tab) > max_poi:
            tab = tab.nlargest(max_poi, "popularity")
        self.place_df = tab
        homes = self._home_sites(buildings, n_homes, np.random.default_rng(seed))
        recs = [(r.id, r.category, r.popularity, r.lon, r.lat) for r in tab.itertuples()] + homes
        xs, ys = self._xy(np.array([r[3] for r in recs]), np.array([r[4] for r in recs]))
        self._places = [MapPlace(r[0], r[1], float(r[2]), float(x), float(y)) for r, x, y in zip(recs, xs, ys)]
        self._byid = {p.id: p for p in self._places}
        _, nn = self._tree.query(np.column_stack([xs, ys]))
        self._node = {p.id: int(n) for p, n in zip(self._places, nn)}
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_max = cache_sources
        self._edges: dict[tuple[int, int], tuple[str, np.ndarray]] = {}

    # -- coordinates
    def _xy(self, lon, lat):
        return ((np.asarray(lon, float) - self.origin[1]) * self._kx,
                (np.asarray(lat, float) - self.origin[0]) * 111_320.0)

    def to_latlon(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.origin[0] + y / 111_320.0, self.origin[1] + x / self._kx

    # -- sites
    def _home_sites(self, b: pd.DataFrame | None, n: int, rng: np.random.Generator):
        if b is None or len(b) == 0 or n <= 0:
            return []
        lon = ((b["xmin"] + b["xmax"]) / 2).to_numpy(float)
        lat = ((b["ymin"] + b["ymax"]) / 2).to_numpy(float)
        area = ((b["xmax"] - b["xmin"]) * self._kx * (b["ymax"] - b["ymin"]) * 111_320.0).to_numpy(float)
        area = np.clip(area, 10.0, None)
        cls = b["cls"].fillna("").astype(str) if "cls" in b else pd.Series([""] * len(b))
        w = area * np.where(cls.isin(NONRES_BUILDING).to_numpy(), 0.02, 1.0)
        w = w / w.sum()
        k = min(n, len(b))
        idx = rng.choice(len(b), size=k, replace=False, p=w)
        ids = b["id"].astype(str).to_numpy()
        return [(f"b:{ids[i]}", "residential", float(area[i]), float(lon[i]), float(lat[i])) for i in idx]

    # -- graph
    def _build_graph(self, segments: pd.DataFrame, connectors: pd.DataFrame):
        seg = segments
        if "subtype" in seg:
            seg = seg[seg["subtype"].fillna("road") == "road"]
        if "class" in seg:
            seg = seg[~seg["class"].isin(ROAD_EXCLUDE)]
        G = build_graph(seg, connectors)
        G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
        ids = list(G.nodes)
        ix = {n: i for i, n in enumerate(ids)}
        k = len(ids)
        if k == 0:
            raise ValueError("no routable road graph in bbox")
        r, c, w = [], [], []
        for a, b, d in G.edges(data=True):
            r += [ix[a], ix[b]]; c += [ix[b], ix[a]]; w += [d["length"]] * 2
        A = coo_matrix((w, (r, c)), shape=(k, k)).tocsr()
        _, lab = connected_components(A, directed=False)
        keep = lab == np.bincount(lab).argmax()
        self._ids = [n for n, kp in zip(ids, keep) if kp]
        self._G = G
        self._A = A[keep][:, keep].tocsr()
        ll = np.array([[G.nodes[n]["lon"], G.nodes[n]["lat"]] for n in self._ids])
        self.node_xy = np.column_stack(self._xy(ll[:, 0], ll[:, 1]))
        self._tree = cKDTree(self.node_xy)

    # -- protocol
    def places(self) -> Sequence[MapPlace]:
        return self._places

    def _source(self, n: int):
        hit = self._cache.get(n)
        if hit is not None:
            self._cache.move_to_end(n)
            return hit
        d, p = dijkstra(self._A, directed=False, indices=n, return_predecessors=True)
        hit = (d.astype(np.float32), p.astype(np.int32))
        self._cache[n] = hit
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return hit

    def _node_path(self, na: int, nb: int) -> list[int]:
        """Node path na -> nb; reuses a cached source at either end (graph is undirected)."""
        rev = nb in self._cache and na not in self._cache
        src, tgt = (nb, na) if rev else (na, nb)
        _, pred = self._source(src)
        path = [tgt]
        while path[-1] != src:
            nxt = int(pred[path[-1]])
            if nxt < 0:
                raise ValueError("places not connected")
            path.append(nxt)
        return path if rev else path[::-1]   # built tgt..src; want na..nb

    def _edge(self, u: int, v: int):
        """(segment id, metric polyline oriented u -> v); memoised per directed edge."""
        hit = self._edges.get((u, v))
        if hit is None:
            e = self._G[self._ids[u]][self._ids[v]]
            g = np.asarray(e["geom"].coords)
            line = np.column_stack(self._xy(g[:, 0], g[:, 1]))
            if np.linalg.norm(line[0] - self.node_xy[u]) > np.linalg.norm(line[-1] - self.node_xy[u]):
                line = line[::-1]
            hit = self._edges[(u, v)] = (e["segment_id"], line)
        return hit

    def route(self, a: str, b: str) -> Route:
        pa, pb = self._byid[a], self._byid[b]
        path = self._node_path(self._node[a], self._node[b])
        segs: list[str] = []
        pts = [np.array([[pa.x, pa.y]])]
        for u, v in zip(path, path[1:]):
            sid, line = self._edge(u, v)
            if not segs or segs[-1] != sid:
                segs.append(sid)
            pts.append(line)
        pts.append(np.array([[pb.x, pb.y]]))
        poly = np.vstack(pts)
        d = np.diff(poly, axis=0)
        length = float(np.hypot(d[:, 0], d[:, 1]).sum())
        return Route(tuple(segs), length, poly)

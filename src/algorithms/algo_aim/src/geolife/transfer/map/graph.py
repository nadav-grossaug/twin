"""Routable graph from Overture connectors + segments (pure networkx/shapely)."""
from __future__ import annotations

import math

import networkx as nx
from shapely import wkb
from shapely.geometry import LineString, Point

R_EARTH = 6371008.8


def haversine(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def _geom(g):
    return wkb.loads(bytes(g)) if isinstance(g, (bytes, bytearray, memoryview)) else g


def line_length_m(line: LineString) -> float:
    c = list(line.coords)
    return sum(haversine(a[0], a[1], b[0], b[1]) for a, b in zip(c, c[1:]))


def _split_line(line: LineString, a: float, b: float) -> LineString:
    """Sub-line between fractions a<b of the line (by planar length)."""
    L = line.length
    pts = [line.interpolate(a * L)]
    pts += [Point(c) for c in line.coords if a * L < line.project(Point(c)) < b * L]
    pts.append(line.interpolate(b * L))
    return LineString([(p.x, p.y) for p in pts])


def build_graph(segments, connectors) -> nx.Graph:
    """Undirected graph; nodes=connector ids (attrs lon, lat), one edge per segment piece.

    ``segments``: DataFrame with id, geometry (WKB or shapely), connectors (list of
    {connector_id, at}). Pieces between consecutive connectors get ``length`` in metres
    (haversine over the sub-geometry), ``segment_id``, ``class_``. Direction rules are ignored
    (undirected) - fine for matching/planting; revisit for one-ways.
    """
    G = nx.Graph()
    for cid, g in zip(connectors["id"], connectors["geometry"]):
        p = _geom(g)
        G.add_node(cid, lon=p.x, lat=p.y)
    for row in segments.to_dict("records"):
        line = _geom(row["geometry"])
        cons = sorted(
            ((float(c["at"]), c["connector_id"]) for c in row["connectors"] if c["connector_id"] in G),
            key=lambda t: t[0],
        )
        cls = row.get("class")
        for (a, ca), (b, cb) in zip(cons, cons[1:]):
            if ca == cb or b <= a:
                continue
            length = line_length_m(_split_line(line, a, b))
            if G.has_edge(ca, cb) and G[ca][cb]["length"] <= length:
                continue
            G.add_edge(ca, cb, segment_id=row["id"], length=length, class_=cls,
                       geom=_split_line(line, a, b))
    return G


def nearest_node(G: nx.Graph, lon: float, lat: float):
    best, bd = None, float("inf")
    for n, d in G.nodes(data=True):
        dd = haversine(lon, lat, d["lon"], d["lat"])
        if dd < bd:
            best, bd = n, dd
    return best, bd


def shortest_path(G: nx.Graph, p0: tuple[float, float], p1: tuple[float, float]):
    """Snap (lon, lat) points to nearest nodes; return (segment_id_list, length_m, node_path)."""
    n0, _ = nearest_node(G, *p0)
    n1, _ = nearest_node(G, *p1)
    nodes = nx.shortest_path(G, n0, n1, weight="length")
    segs: list = []
    total = 0.0
    for a, b in zip(nodes, nodes[1:]):
        e = G[a][b]
        total += e["length"]
        if not segs or segs[-1] != e["segment_id"]:
            segs.append(e["segment_id"])
    return segs, total, nodes

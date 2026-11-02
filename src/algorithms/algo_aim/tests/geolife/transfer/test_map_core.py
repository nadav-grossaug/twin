import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from geolife.transfer.map.graph import build_graph, shortest_path
from geolife.transfer.map.match import match
from geolife.outside.map.overture import load_overture

LON0, LAT0 = 116.30, 39.98
D = 0.001  # ~85 m lon / ~111 m lat
N = 4


def _grid():
    nodes = {(i, j): f"c{i}{j}" for i in range(N) for j in range(N)}
    con = pd.DataFrame(
        [{"id": cid, "geometry": Point(LON0 + i * D, LAT0 + j * D).wkb} for (i, j), cid in nodes.items()]
    )
    segs = []
    for i in range(N):
        for j in range(N):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i + di, j + dj)
                if b in nodes:
                    # mid-way extra vertex check: two connectors only at ends
                    line = LineString([(LON0 + a[0] * D, LAT0 + a[1] * D), (LON0 + b[0] * D, LAT0 + b[1] * D)])
                    segs.append({
                        "id": f"s{a[0]}{a[1]}-{b[0]}{b[1]}", "geometry": line.wkb, "class": "residential",
                        "connectors": [{"connector_id": nodes[a], "at": 0.0}, {"connector_id": nodes[b], "at": 1.0}],
                    })
    return pd.DataFrame(segs), con


def test_graph_and_shortest_path():
    segs, con = _grid()
    G = build_graph(segs, con)
    assert G.number_of_nodes() == 16 and G.number_of_edges() == 24
    ids, length, nodes = shortest_path(G, (LON0 + 0.00005, LAT0), (LON0 + 3 * D, LAT0 + 0.0001))
    assert ids == ["s00-10", "s10-20", "s20-30"]
    assert abs(length - 3 * D * 111195 * np.cos(np.radians(LAT0))) < 5


def test_split_segment_by_connector_fraction():
    line = LineString([(LON0, LAT0), (LON0 + 0.003, LAT0)])
    con = pd.DataFrame([{"id": k, "geometry": Point(LON0 + x, LAT0).wkb}
                        for k, x in (("a", 0), ("b", 0.001), ("c", 0.003))])
    segs = pd.DataFrame([{"id": "s", "geometry": line.wkb, "class": "x", "connectors": [
        {"connector_id": "c", "at": 1.0}, {"connector_id": "a", "at": 0.0}, {"connector_id": "b", "at": 1 / 3}]}])
    G = build_graph(segs, con)
    assert G.number_of_edges() == 2
    assert abs(G["a"]["b"]["length"] * 2 - G["b"]["c"]["length"]) < 1.0


def test_hmm_matches_noisy_path():
    segs, con = _grid()
    G = build_graph(segs, con)
    # true route along y=LAT0 then up x=LAT0+3D
    truth = ["s00-10", "s10-20", "s20-30", "s30-31", "s31-32"]
    rng = np.random.default_rng(0)
    pts = [(LON0 + x, LAT0) for x in np.linspace(0.0001, 3 * D, 16)]
    pts += [(LON0 + 3 * D, LAT0 + y) for y in np.linspace(0.0001, 2 * D - 0.0001, 10)]
    noisy = [(lo + rng.normal(0, 0.00005), la + rng.normal(0, 0.00004)) for lo, la in pts]  # ~4-5 m
    assert match(G, noisy, radius=40, sigma=8) == truth


def test_overture_cache_roundtrip(tmp_path):
    bbox = (116.28, 39.96, 116.32, 40.0)
    from geolife.outside.map.overture import _cache_path, DEFAULT_RELEASE
    p = _cache_path(tmp_path, DEFAULT_RELEASE, "segment", bbox)
    pd.DataFrame({"id": ["x"]}).to_parquet(p)
    assert load_overture("segment", bbox, cache_dir=tmp_path)["id"].tolist() == ["x"]  # no network

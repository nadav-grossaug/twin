"""OvertureProvider on tiny in-memory fixtures (no network) + a GEOLIFE_NET=1 integration test."""
import os

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from geolife.outside.map.provider import OvertureProvider, coarse_category, place_table

X0, Y0, D = 116.30, 39.95, 0.002   # 3x3 grid of connectors, ~170 m x ~220 m cells


def _fixture():
    cons, segs = [], []
    for i in range(3):
        for j in range(3):
            cons.append({"id": f"c{i}{j}", "geometry": Point(X0 + j * D, Y0 + i * D).wkb})
    for i in range(3):
        for j in range(3):
            for (i2, j2) in ((i, j + 1), (i + 1, j)):
                if i2 < 3 and j2 < 3:
                    line = LineString([(X0 + j * D, Y0 + i * D), (X0 + j2 * D, Y0 + i2 * D)])
                    segs.append({"id": f"s{i}{j}-{i2}{j2}", "subtype": "road", "class": "residential",
                                 "geometry": line.wkb,
                                 "connectors": [{"connector_id": f"c{i}{j}", "at": 0.0},
                                                {"connector_id": f"c{i2}{j2}", "at": 1.0}]})
    # a footway that must be excluded, and an isolated road (separate component)
    segs.append({"id": "foot", "subtype": "road", "class": "footway",
                 "geometry": LineString([(X0, Y0), (X0 + 2 * D, Y0 + 2 * D)]).wkb,
                 "connectors": [{"connector_id": "c00", "at": 0.0}, {"connector_id": "c22", "at": 1.0}]})
    cons += [{"id": "iso1", "geometry": Point(X0 + 1, Y0 + 1).wkb}, {"id": "iso2", "geometry": Point(X0 + 1.01, Y0 + 1).wkb}]
    segs.append({"id": "iso", "subtype": "road", "class": "primary",
                 "geometry": LineString([(X0 + 1, Y0 + 1), (X0 + 1.01, Y0 + 1)]).wkb,
                 "connectors": [{"connector_id": "iso1", "at": 0.0}, {"connector_id": "iso2", "at": 1.0}]})
    places = pd.DataFrame({
        "id": ["p1", "p2", "p3"], "confidence": [0.9, 0.5, None], "has_brand": [True, False, None],
        "xmin": [X0 + 0.0001, X0 + 0.0039, X0 + 0.002], "xmax": [X0 + 0.0001, X0 + 0.0039, X0 + 0.002],
        "ymin": [Y0 + 0.0001, Y0 + 0.0039, Y0 + 0.002], "ymax": [Y0 + 0.0001, Y0 + 0.0039, Y0 + 0.002],
        "hierarchy": [["food_and_drink", "casual_eatery", "fast_food_restaurant"],
                      ["education", "school"], []],
        "basic_category": ["fast_food_restaurant", "school", None]})
    b = pd.DataFrame({"id": ["b1", "b2", "b3"], "cls": ["apartments", None, "commercial"],
                      "xmin": [X0, X0 + 0.001, X0 + 0.003], "xmax": [X0 + 0.0004, X0 + 0.0012, X0 + 0.0034],
                      "ymin": [Y0, Y0 + 0.001, Y0 + 0.003], "ymax": [Y0 + 0.0004, Y0 + 0.0012, Y0 + 0.0034]})
    return places, b, pd.DataFrame(segs), pd.DataFrame(cons)


@pytest.fixture(scope="module")
def prov():
    p, b, s, c = _fixture()
    return OvertureProvider((X0, Y0, X0 + 0.004, Y0 + 0.004), places=p, buildings=b, segments=s, connectors=c,
                            n_homes=3, seed=1)


def test_coarse_category():
    assert coarse_category(["food_and_drink", "casual_eatery"], "fast_food_restaurant") == "food"
    assert coarse_category(["education", "school"]) == "school"
    assert coarse_category(["travel_and_transportation", "public_transit", "subway_station"]) == "transit"
    assert coarse_category(["lodging", "hotel"]) == "other"
    assert coarse_category(None, None) == "other"
    assert coarse_category(["shopping", "market"]) == "shop"


def test_place_table_columns_and_popularity():
    p, *_ = _fixture()
    t = place_table(p)
    assert list(t.columns) == ["id", "category", "lat", "lon", "popularity"]
    assert t.category.tolist() == ["food", "school", "other"]
    assert (t.popularity > 0).all() and t.popularity[0] > t.popularity[1]   # confident branded POI ranks higher


def test_places_and_homes(prov):
    pl = {p.id: p for p in prov.places()}
    assert {"p1", "p2", "p3"} <= set(pl)
    homes = [p for p in pl.values() if p.id.startswith("b:")]
    assert len(homes) == 3 and all(h.category == "residential" and h.weight >= 10 for h in homes)
    assert pl["p1"].category == "food" and pl["p2"].category == "school"


def test_route_polyline_segments_and_symmetry(prov):
    r = prov.route("p1", "p2")
    assert r.length_m > 0 and r.segments and "foot" not in r.segments and "iso" not in r.segments
    assert np.allclose(r.polyline[0], [prov._byid["p1"].x, prov._byid["p1"].y])
    assert np.allclose(r.polyline[-1], [prov._byid["p2"].x, prov._byid["p2"].y])
    seglen = np.linalg.norm(np.diff(r.polyline, axis=0), axis=1)
    assert abs(seglen.sum() - r.length_m) < 1e-6
    # reverse route (served from the cached source) is the same road path reversed
    r2 = prov.route("p2", "p1")
    assert abs(r2.length_m - r.length_m) < 1e-6 and r2.segments == tuple(reversed(r.segments))
    assert np.allclose(r2.polyline[0], r.polyline[-1])
    # Manhattan: p1 and p2 are the opposite corners, so route >= straight line
    assert r.length_m >= np.linalg.norm(r.polyline[0] - r.polyline[-1]) - 1e-6


def test_latlon_roundtrip(prov):
    p = prov._byid["p1"]
    lat, lon = prov.to_latlon([p.x], [p.y])
    assert abs(lat[0] - (Y0 + 0.0001)) < 1e-7 and abs(lon[0] - (X0 + 0.0001)) < 1e-7


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("GEOLIFE_NET") != "1", reason="needs network (set GEOLIFE_NET=1)")
def test_overture_network_small_bbox(tmp_path):
    p = OvertureProvider((116.300, 39.970, 116.310, 39.980), cache_dir=tmp_path, n_homes=50)
    pl = list(p.places())
    assert len(pl) > 50 and any(x.category == "food" for x in pl)
    ids = [x.id for x in pl]
    r = p.route(ids[0], ids[-1])
    assert r.length_m > 0 and r.segments

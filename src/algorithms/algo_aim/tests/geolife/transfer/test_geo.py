from geolife.inside.data import BEIJING_BBOX
from geolife.transfer.geo import filter_bbox, filter_speed, haversine_m, pseudonyms
from geolife.transfer.types import GpsPoint


def test_pseudonyms_seeded_and_unique():
    a = pseudonyms(["000", "001", "002"], 1)
    assert a == pseudonyms(["002", "001", "000"], 1)
    assert a != pseudonyms(["000", "001", "002"], 2)
    assert len(set(a.values())) == 3


def test_filters():
    pts = [GpsPoint(0, 39.9, 116.4), GpsPoint(1, 0.0, 0.0), GpsPoint(10, 39.9001, 116.4),
           GpsPoint(11, 39.95, 116.4), GpsPoint(20, 39.9002, 116.4)]
    assert len(filter_bbox(pts, BEIJING_BBOX)) == 4
    kept = filter_speed(filter_bbox(pts, BEIJING_BBOX))
    assert [p.t for p in kept] == [0, 10, 20]
    assert haversine_m(39.9, 116.4, 39.9, 116.4) == 0

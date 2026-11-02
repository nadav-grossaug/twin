from pathlib import Path

from geolife.inside.data import load_geolife, parse_labels, parse_plt, stay_points
from geolife.transfer.types import GpsPoint

HDR = ["Geolife trajectory", "WGS 84", "Altitude is in Feet", "Reserved 3",
       "0,2,255,My Track,0,0,2,8421376", "0"]


def plt_lines(rows):
    return HDR + [f"{la},{lo},0,13,39298.1,{d},{t}" for la, lo, d, t in rows]


def make_root(tmp_path: Path) -> Path:
    for uid, n in (("000", 2), ("001", 1)):
        td = tmp_path / "Data" / uid / "Trajectory"
        td.mkdir(parents=True)
        for k in range(n):
            rows = [(39.9 + 0.0001 * i, 116.4, "2008-01-01", f"00:00:{i:02d}") for i in range(5)]
            (td / f"2008010{k}000000.plt").write_text("\n".join(plt_lines(rows)))
    (tmp_path / "Data" / "000" / "labels.txt").write_text(
        "Start Time\tEnd Time\tTransportation Mode\n"
        "2008/01/01 00:00:01\t2008/01/01 00:00:03\tBus\n")
    return tmp_path


def test_parse_plt_skips_header_and_bad_rows():
    lines = plt_lines([(39.9, 116.4, "1970-01-01", "00:00:10")]) + ["garbage", "x,y,0,0,0,d,t"]
    assert parse_plt(lines) == [GpsPoint(10, 39.9, 116.4)]


def test_parse_labels():
    sp = parse_labels(["Start Time\tEnd Time\tTransportation Mode",
                       "1970/01/01 00:00:10\t1970/01/01 00:01:10\tTaxi"])
    assert (sp[0].t_start, sp[0].t_end, sp[0].mode) == (10, 70, "taxi")


def test_load_geolife(tmp_path):
    root = make_root(tmp_path)
    tr, lb = load_geolife(root, seed=3, with_labels=True)
    assert sorted(len(v) for v in tr.values()) == [1, 2]
    assert all(k.startswith("u") and k not in ("000", "001") for k in tr)
    assert sum(len(v) for v in lb.values()) == 1
    assert all(isinstance(p, GpsPoint) for v in tr.values() for t in v for p in t)


def test_stay_points():
    pts = [GpsPoint(i * 60, 39.9, 116.4) for i in range(30)]
    pts += [GpsPoint(1800 + i * 10, 39.9 + 0.01 * (i + 1), 116.4) for i in range(5)]
    sp = stay_points(pts)
    assert len(sp) == 1 and sp[0][3] - sp[0][2] >= 1200

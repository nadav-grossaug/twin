"""Per-column tests of the synthetic GeoLife TSV.

Every column of synth/geolife_synth.tsv must either match the source schema
exactly (header, user_id format, altitude constant, raw mode labels), satisfy
its structural invariant (strictly increasing timestamps per trajectory,
datetime inside its label window), respect the public geography contract
(ZERO points in the Beijing exclusion bbox, ALL points in the Tokyo target
bbox — the declared manipulate step), or avoid the synthetic-data lattice
artefacts (no long exactly-collinear runs, no turn-angle atom at ±90°).
Skips gracefully when `make synth` has not been run."""
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import pytest

from spec import load_spec, mode_map

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "synth"

COLS = ["user_id", "sequence_uuid", "trajectory_id", "datetime", "latitude",
        "longitude", "altitude_ft", "label_start", "label_end", "mode"]
DT_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tsv():
    p = SYNTH / "geolife_synth.tsv"
    if not p.exists():
        pytest.skip("no synthetic data — run `make synth`")
    with open(p, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        rows = [tuple(row) for row in r]
    assert rows, "empty synthetic TSV"
    return header, rows


@pytest.fixture(scope="module")
def trajectories(tsv):
    """(user_id, trajectory_id) -> rows in file order."""
    _, rows = tsv
    by_traj = defaultdict(list)
    for r in rows:
        by_traj[(r[0], r[2])].append(r)
    return by_traj


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def _micro(s: str) -> int:
    return int(round(float(s) * 1e6))


# ------------------------------------------------------- schema and format
def test_header_and_schema_exact(tsv):
    header, rows = tsv
    assert header == COLS
    for r in rows:
        assert len(r) == len(COLS), r


def test_user_id_zero_padded(tsv):
    _, rows = tsv
    for uid in {r[0] for r in rows}:
        assert re.fullmatch(r"\d{3,}", uid), uid
        assert len(uid) == 3 or not uid.startswith("0"), uid


def test_altitude_constant(tsv):
    _, rows = tsv
    assert all(r[6] == "-777" for r in rows)


def test_mode_labels_are_raw_seed_labels(tsv):
    _, rows = tsv
    raw_labels = set(mode_map())
    got = {r[9] for r in rows}
    assert got <= raw_labels, got - raw_labels


# ------------------------------------------------------------- timestamps
def test_datetimes_parse(tsv):
    _, rows = tsv
    for r in rows:
        for col in (3, 7, 8):
            assert DT_RE.fullmatch(r[col]), r[col]
    # full strptime on a sample (the regex covers the rest)
    for r in rows[:2000]:
        for col in (3, 7, 8):
            datetime.strptime(r[col], "%Y-%m-%d %H:%M:%S")


def test_timestamps_strictly_increasing_per_trajectory(trajectories):
    # "YYYY-MM-DD HH:MM:SS" sorts lexicographically == chronologically
    for key, rs in trajectories.items():
        ts = [r[3] for r in rs]
        assert all(a < b for a, b in zip(ts, ts[1:])), (
            f"trajectory {key}: timestamps not strictly increasing")


def test_datetime_inside_its_label_window(tsv):
    _, rows = tsv
    for r in rows:
        assert r[7] <= r[3] <= r[8], (
            f"point {r[0]}/{r[2]} at {r[3]} outside label window "
            f"[{r[7]}, {r[8]}]")


def test_points_inside_public_window(tsv, spec):
    _, rows = tsv
    start = datetime.utcfromtimestamp(int(spec["window_start_s"])).strftime(
        "%Y-%m-%d %H:%M:%S")
    end = datetime.utcfromtimestamp(int(spec["window_end_s"])).strftime(
        "%Y-%m-%d %H:%M:%S")
    for r in rows:
        assert start <= r[3] < end, r[3]


# -------------------------------------------------------------- geography
def test_no_point_in_beijing_exclusion_bbox(tsv, spec):
    lat0, lat1 = spec["excl_lat_min_cd"] / 100, spec["excl_lat_max_cd"] / 100
    lon0, lon1 = spec["excl_lon_min_cd"] / 100, spec["excl_lon_max_cd"] / 100
    _, rows = tsv
    offenders = [r for r in rows
                 if lat0 <= float(r[4]) <= lat1 and lon0 <= float(r[5]) <= lon1]
    assert not offenders, (
        f"{len(offenders)} points inside the Beijing exclusion bbox, "
        f"e.g. {offenders[:3]}")


def test_all_points_in_tokyo_target_bbox(tsv, spec):
    lat0, lat1 = spec["tgt_lat_min_cd"] / 100, spec["tgt_lat_max_cd"] / 100
    lon0, lon1 = spec["tgt_lon_min_cd"] / 100, spec["tgt_lon_max_cd"] / 100
    _, rows = tsv
    offenders = [r for r in rows
                 if not (lat0 <= float(r[4]) <= lat1
                         and lon0 <= float(r[5]) <= lon1)]
    assert not offenders, (
        f"{len(offenders)} points outside the Tokyo target bbox, "
        f"e.g. {offenders[:3]}")


def test_neutral_frame_tsv_present(tsv):
    """The pre-manipulation neutral frame ships alongside, same schema."""
    p = SYNTH / "geolife_synth_frame.tsv"
    assert p.exists(), "geolife_synth_frame.tsv missing"
    with open(p, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        n = sum(1 for _ in r)
    assert header == COLS
    assert n == len(tsv[1]), "frame and placed TSVs must have the same points"


# ------------------------------------------------- lattice artefact guards
def _deltas(rs):
    """Nonzero-step displacement deltas (micro-degrees) down a trajectory."""
    pts = [(_micro(r[5]), _micro(r[4])) for r in rs]   # (x=lon, y=lat)
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        if dx or dy:
            out.append((dx, dy))
    return out


def test_no_long_exactly_collinear_runs(trajectories):
    """The generator forces a turn every few steps; GPS noise does the rest —
    a run of > 8 exactly-collinear consecutive steps is a lattice artefact."""
    worst, where = 0, None
    for key, rs in trajectories.items():
        run = 0
        ds = _deltas(rs)
        for (ax, ay), (bx, by) in zip(ds, ds[1:]):
            if ax * by - ay * bx == 0:
                run += 1
                if run > worst:
                    worst, where = run, key
            else:
                run = 0
    assert worst <= 8, f"collinear run of {worst} steps in trajectory {where}"


def test_no_turn_angle_atom_at_90_degrees(trajectories):
    """Grid-walk generators leave a spike at exactly ±90°; the released
    turn-histogram + jitter design must not."""
    n = at90 = 0
    for rs in trajectories.values():
        ds = _deltas(rs)
        for (ax, ay), (bx, by) in zip(ds, ds[1:]):
            ang = math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))
            n += 1
            if abs(abs(ang) - 90.0) <= 0.5:
                at90 += 1
    if n < 1000:
        pytest.skip("too few turns for a distributional check")
    assert at90 / n <= 0.05, f"{at90}/{n} turns within 0.5° of ±90°"


# -------------------------------------------------------------- meta/plan
def test_meta_and_plan_present(tsv):
    import json
    meta = json.loads((SYNTH / "meta.json").read_text())
    plan = json.loads((SYNTH / "plan.json").read_text())
    assert meta["assumptions"]["fields_not_modelled"]
    assert plan["planted"]


# ------------------------------------------------- road-likeness contract
# Synthetic trajectories must be rendered from the released route-shape
# statistics (seg_hist / junction_hist / routes), not straight vectors.
# These are STRUCTURAL (hard) checks mirroring synth.py's hard/soft split:
# distributional drift from the released stats stays a soft concern of
# `scripts/synth.py validate`; here only the road-shape fingerprints are
# enforced. The contract applies to FUTURE synth runs: a synth dir rendered
# from a pre-route genspec (meta.json's genspec_sha256 does not match a
# routes-bearing genspec) skips instead of failing on stale output.

ROUTE_MODE_CLASSES = ("walk", "bus", "bike", "car")


@pytest.fixture(scope="module")
def route_synth(tsv):
    """Skip unless the synth on disk was rendered by the route-shape
    generator — the marker is the "routes" entry the new synth.py writes
    into meta.json's released_fact_sources (stale straight-vector outputs
    lack it and must not fail the build)."""
    meta_p = SYNTH / "meta.json"
    if not meta_p.exists():
        pytest.skip("no synth meta — run `make synth` first")
    meta = json.loads(meta_p.read_text())
    if "routes" not in meta.get("released_fact_sources", {}):
        pytest.skip("synth predates the route-shape generator — "
                    "rerun `make synth`")
    return meta


def _haversine_m(lat0, lon0, lat1, lon1):
    r = 6371000.0
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dp, dl = p1 - p0, math.radians(lon1 - lon0)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


@pytest.fixture(scope="module")
def sequences(tsv):
    """(user_id, sequence_uuid) -> (mode_class, [(lat, lon)] in file order
    — file order is time order within a trajectory)."""
    _, rows = tsv
    raw2cls = mode_map()
    seqs: dict = {}
    for r in rows:
        key = (r[0], r[1])
        if key not in seqs:
            seqs[key] = (raw2cls.get(r[9], "XX"), [])
        seqs[key][1].append((float(r[4]), float(r[5])))
    return seqs


def test_synth_legs_are_not_straight_vectors(route_synth, sequences, spec):
    """Per-mode median straightness (net displacement / haversine path
    length per sequence) must be < 0.98 for every route mode present with
    >= 20 moving sequences — a straight-vector renderer medians ~1.0."""
    min_net = int(spec["route_min_net_m"])
    per_mode = defaultdict(list)
    for (_, _), (cls, pts) in sequences.items():
        if len(pts) < 3:
            continue
        path = sum(_haversine_m(a[0], a[1], b[0], b[1])
                   for a, b in zip(pts, pts[1:]))
        net = _haversine_m(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
        if path <= 0 or net < min_net:
            continue   # circuity/straightness is defined for moving legs only
        per_mode[cls].append(min(1.0, net / path))
    checked = 0
    for cls in ROUTE_MODE_CLASSES:
        vals = per_mode.get(cls, [])
        if len(vals) < 20:
            continue
        checked += 1
        med = median(vals)
        assert med < 0.98, (
            f"{cls}: median straightness {med:.4f} >= 0.98 over "
            f"{len(vals)} sequences — straight-vector fingerprint")
    assert checked, (
        f"no route mode with >= 20 moving sequences to check: "
        f"{ {m: len(v) for m, v in per_mode.items()} }")


def test_synth_has_junction_turns(route_synth, sequences, spec):
    """A 100-sequence sample, resampled to >= route_resample_m between kept
    points, must contain at least one junction turn
    (|turn| >= junction_min_turn_deg) — road-like paths bend at junctions."""
    resample_m = int(spec["route_resample_m"])
    min_turn = int(spec["junction_min_turn_deg"])
    n_seqs = n_junctions = 0
    for (_, _), (_cls, pts) in sequences.items():
        if n_seqs >= 100:
            break
        if len(pts) < 3:
            continue
        n_seqs += 1
        # resample: keep a point once >= resample_m of path has accrued
        kept = [pts[0]]
        acc = 0.0
        for a, b in zip(pts, pts[1:]):
            acc += _haversine_m(a[0], a[1], b[0], b[1])
            if acc >= resample_m:
                kept.append(b)
                acc = 0.0
        if len(kept) < 3:
            continue
        # turn angles between consecutive resampled displacement vectors
        # (local planar frame: fine at the ~city scale of a sequence)
        for p0, p1, p2 in zip(kept, kept[1:], kept[2:]):
            c = math.cos(math.radians(p1[0]))
            v1 = ((p1[1] - p0[1]) * c, p1[0] - p0[0])
            v2 = ((p2[1] - p1[1]) * c, p2[0] - p1[0])
            if v1 == (0.0, 0.0) or v2 == (0.0, 0.0):
                continue
            ang = math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0],
                                          v1[0] * v2[0] + v1[1] * v2[1]))
            if abs(ang) >= min_turn:
                n_junctions += 1
    assert n_seqs, "no synthetic sequences to sample"
    assert n_junctions >= 1, (
        f"no junction turn (|turn| >= {min_turn}° on a {resample_m} m "
        f"resampled stream) in a {n_seqs}-sequence sample — "
        f"straight-vector fingerprint")

"""Synthetic travel data: meta/provenance, planted-pattern recall, per-column
structural and distributional checks on the generated TSV."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pytest

from spec import REGIONS, load_spec, region_of

COLS = ["userid", "dt", "prev_country", "prev_last_venue_id", "prev_last_lat",
        "prev_last_long", "prev_duration", "next_country",
        "next_first_venue_id", "next_first_lat", "next_first_long",
        "next_duration"]
TC_BINS = {"tc_1": (1, 1), "tc_2": (2, 2), "tc_3_4": (3, 4), "tc_5_8": (5, 8),
           "tc_9_16": (9, 16), "tc_17_32": (17, 32), "tc_33p": (33, 10**6)}


@pytest.fixture(scope="module")
def synth(root):
    s = root / "synth"
    if not (s / "travel_synth.tsv").exists():
        pytest.skip("run `make synth` first")
    rows = list(csv.DictReader(open(s / "travel_synth.tsv"), delimiter="\t"))
    plan = json.loads((s / "plan.json").read_text())
    # the Lean generator writes planted motifs at the top level
    plan.setdefault("planted", {"corridor": plan["corridor"],
                                "tour": plan["tour"]})
    meta = json.loads((s / "meta.json").read_text())
    return s, rows, plan, meta


def _ts(r):
    return datetime.strptime(r["dt"], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp()


def test_meta_declares_gaps_and_provenance(synth):
    _, _, _, meta = synth
    gaps = meta["assumptions"]["fields_not_modelled"]
    assert any("geograph" in g or "coordinate" in g for g in gaps)
    assert any("venue" in g for g in gaps)
    assert meta["genspec_sha256"] and meta["dp_release_sha256"]


def test_schema_and_structure(synth):
    """The structural invariants the Lean generator proves by construction,
    re-checked empirically on the rendered TSV."""
    _, rows, _, _ = synth
    assert list(rows[0].keys()) == COLS
    spec = load_spec()
    w0, w1 = int(spec["window_start_s"]), int(spec["window_end_s"])
    by_user = defaultdict(list)
    for r in rows:
        by_user[r["userid"]].append(r)
    for u, rs in by_user.items():
        ts = [_ts(r) for r in rs]
        assert ts == sorted(ts) and len(set(ts)) == len(ts), (
            f"user {u}: timestamps not strictly increasing")
        for r in rs:
            assert r["prev_country"] != r["next_country"], u
            assert int(r["prev_duration"]) >= 0
            assert int(r["next_duration"]) >= 0
            assert len(r["prev_last_venue_id"]) == 24
            assert len(r["next_first_venue_id"]) == 24
            int(r["prev_last_venue_id"], 16)
            int(r["next_first_venue_id"], 16)
            assert -90 <= float(r["prev_last_lat"]) <= 90
            assert -180 <= float(r["prev_last_long"]) <= 180
        assert w0 <= ts[0] and ts[-1] < w1, f"user {u} outside public window"
        # duration chaining: next stay of row i = previous stay of row i+1
        for a, b in zip(rs, rs[1:]):
            assert a["next_duration"] == b["prev_duration"], (
                f"user {u}: duration chain broken")
            assert a["next_country"] == b["prev_country"], (
                f"user {u}: country chain broken")


def test_travel_counts_match_plan_and_bins(synth):
    _, rows, plan, _ = synth
    got = Counter(r["userid"] for r in rows)
    assert len(plan["users"]) == plan["n_users"]
    for u in plan["users"]:
        assert got[str(u["user"])] == u["n_travels"], u["user"]
        # every planned count sits inside SOME public bin (1..tc_33p cap)
        assert any(lo <= u["n_travels"] <= hi for lo, hi in TC_BINS.values()), u


def test_planted_corridor_in_plan(synth):
    _, rows, plan, _ = synth
    c = plan["planted"]["corridor"]
    user_rows = [r for r in rows if r["userid"] == str(c["user"])]
    hits = [r for r in user_rows
            if r["prev_country"] == c["a"] and r["next_country"] == c["b"]]
    assert len(hits) >= c["n_repeats"] >= 5


def test_planted_tour_in_plan(synth):
    _, rows, plan, _ = synth
    t = plan["planted"]["tour"]
    user_rows = sorted((r for r in rows if r["userid"] == str(t["user"])),
                       key=_ts)
    assert len(user_rows) >= 3
    ts = [_ts(r) for r in user_rows[:3]]
    assert ts[2] - ts[0] <= 2 * 7 * 86400
    countries = {user_rows[0]["prev_country"]} | {
        r["next_country"] for r in user_rows[:3]}
    assert len(countries) >= 3


def test_validation_passed(synth):
    s, _, _, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    assert not v["hard_failures"], v["hard_failures"]
    assert v["notes"]["planted_corridor_recall"] == 1.0
    assert v["notes"]["planted_tour_recall"] == 1.0


def test_region_mix_tracks_genspec(synth, root):
    _, rows, _, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    gs = json.loads(gs_p.read_text())
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")
    want = {c["name"]: c["w"] for c in gs["region_mix"]["cells"]}
    got = Counter(region_of(r["next_country"]) for r in rows)
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    tv = 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                   for c in REGIONS)
    # the generator blends transitions and corridors on top of the marginal
    assert tv < 0.35, f"region mix TV {tv:.3f}"


def test_certificate_kernel_checked(synth):
    s, _, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True

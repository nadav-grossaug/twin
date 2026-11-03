"""The validator's checks must FAIL on the defects they exist to catch.
Pure unit tests: an in-memory DuckDB stands in for the re-ingested synthetic
warehouse, so no pipeline artifacts are needed."""
import importlib.util
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("synth_script", ROOT / "scripts" / "synth.py")
assert _spec is not None and _spec.loader is not None
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)

RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
CT_CELLS = synth.CT_CELLS
NIGHTS = synth.NIGHTS_CELLS
GAPS = synth.GAP_CELLS

# genspec weights matching the healthy warehouse below
RC_W = {"rc_1_3": 30, "rc_4": 500, "rc_5": 400, "rc_6": 200,
        "rc_7_8": 150, "rc_9_12": 80, "rc_13p": 40}
CT_W = {c: 10 for c in CT_CELLS}
NIGHTS_W = {"n_1": 600, "n_2": 300, "n_3": 150, "n_4_6": 100, "n_7p": 50}
GAP_W = {"g_0": 1000, "g_1": 50, "g_2_3": 40, "g_4p": 30}


def genspec():
    return {
        "res_count": {"cells": [{"name": b, "w": RC_W[b]} for b in RC_BINS]},
        "country_mix": {"cells": [{"name": c, "w": CT_W[c]} for c in CT_CELLS]},
        "nights": {"cells": [{"name": n, "w": NIGHTS_W[n]} for n in NIGHTS]},
        "gap": {"cells": [{"name": g, "w": GAP_W[g]} for g in GAPS]},
    }


def warehouse(rc, ct, nights, gaps, stg=None, motif_rows=0, motif_tours=0,
              motif_longstays=0):
    """Build the minimal re-ingested warehouse the validator queries."""
    con = duckdb.connect(":memory:")
    con.execute("create table feat_users (rc_bin varchar)")
    for b, n in rc.items():
        for _ in range(n):
            con.execute("insert into feat_users values (?)", [b])
    con.execute("create table feat_reservations (hotel_ct varchar)")
    for c, n in ct.items():
        for _ in range(n):
            con.execute("insert into feat_reservations values (?)", [c])
    con.execute("create table stg_reservations (nights_cell varchar, "
                "qf_bad_dates boolean, qf_overlap boolean)")
    for c, n in nights.items():
        for _ in range(n):
            con.execute("insert into stg_reservations values (?, false, false)", [c])
    con.execute("create table feat_transitions (gap_days integer)")
    for g, n in gaps.items():
        v = {"g_0": 0, "g_1": 1, "g_2_3": 2, "g_4p": 5}[g]
        for _ in range(n):
            con.execute("insert into feat_transitions values (?)", [v])
    con.execute("create table mart_motif_returns (userid bigint)")
    con.execute("create table mart_motif_tours (userid bigint, is_tour boolean)")
    con.execute("create table mart_motif_longstays (userid bigint, "
                "is_longstay boolean)")
    for row in stg or []:
        con.execute("insert into stg_reservations values (?, ?, ?)", row)
    for _ in range(motif_rows):
        con.execute("insert into mart_motif_returns values (1)")
    for _ in range(motif_tours):
        con.execute("insert into mart_motif_tours values (2, true)")
    for _ in range(motif_longstays):
        con.execute("insert into mart_motif_longstays values (3, true)")
    return con


def run(rc=None, ct=None, nights=None, gaps=None, n_rows=10**9,
        judged=True, **kw):
    rc = rc or {b: RC_W[b] * 2 for b in RC_BINS}      # 2x = roughly on-spec
    ct = ct or {c: CT_W[c] * 20 for c in CT_CELLS}
    nights = nights or {n: NIGHTS_W[n] * 2 for n in NIGHTS}
    gaps = gaps or {g: GAP_W[g] * 2 for g in GAPS}
    con = warehouse(rc, ct, nights, gaps, **kw)
    notes, soft = {}, []
    synth.check_mixes(genspec(), con, notes, soft, n_rows=n_rows, judged=judged)
    return con, notes, soft


def test_healthy_synth_passes():
    _, _, soft = run()
    assert soft == [], soft


def test_skewed_res_count_mix_is_caught():
    bad = {b: RC_W[b] * 2 for b in RC_BINS}
    bad["rc_4"], bad["rc_13p"] = 0, 2200     # mass dumped into the tail
    _, notes, soft = run(rc=bad)
    assert any("res_count" in s for s in soft), soft
    assert notes["res_count_tv_vs_genspec"] > 0.25


def test_expected_populated_bin_may_not_be_empty():
    bad = {b: RC_W[b] * 2 for b in RC_BINS}
    bad["rc_4"] = 0                          # spec expects ~568 of 2800
    _, _, soft = run(rc=bad)
    assert any("rc_4" in s and "none generated" in s for s in soft), soft


def test_small_run_gets_a_wider_tolerance():
    """TV ~ 0.36 with every expected cell populated: a defect at full scale
    (fails the flat 0.25) but inside the 3-sigma sampling allowance at
    n = 60 users (expected TV ~ sqrt(6/(2 pi 60)) ~ 0.13, allowance ~ 0.38)."""
    bad = {"rc_1_3": 0, "rc_4": 36, "rc_5": 4, "rc_6": 4, "rc_7_8": 4,
           "rc_9_12": 6, "rc_13p": 6}
    _, notes, soft = run(rc=bad, n_rows=60)
    assert notes["res_count_tv_vs_genspec"] > 0.25, notes
    assert notes["res_count_tv_tol"] > notes["res_count_tv_vs_genspec"], notes
    assert not any("res_count_tv" in s for s in soft), soft


def test_the_same_defect_is_caught_at_full_scale():
    bad = {"rc_1_3": 0, "rc_4": 36, "rc_5": 4, "rc_6": 4, "rc_7_8": 4,
           "rc_9_12": 6, "rc_13p": 6}
    _, _, soft = run(rc=bad)          # n_rows -> the flat 0.25 tolerance
    assert any("res_count_tv" in s for s in soft), soft


def test_unjudged_small_n_records_findings_without_failing():
    bad = {b: RC_W[b] * 2 for b in RC_BINS}
    bad["rc_4"] = 0
    _, notes, soft = run(rc=bad, judged=False)
    assert soft == []
    assert any("rc_4" in u for u in notes["unjudged_small_n"]), notes


def test_gap_mix_defect_is_caught():
    bad = {g: GAP_W[g] * 2 for g in GAPS}
    bad["g_0"], bad["g_4p"] = 100, 2000      # contiguity destroyed
    _, notes, soft = run(gaps=bad)
    assert any("gap" in s for s in soft), soft
    assert notes["gap_tv_vs_genspec"] > 0.25


# --- planted-motif recall: presence in the production marts -----------------
def test_planted_recall_detected_and_missed():
    plan = {"planted": {"return": {"user": 1}, "tour": {"user": 2},
                        "longstay": {"user": 3}}}
    con = warehouse({b: 1 for b in RC_BINS}, {c: 1 for c in CT_CELLS},
                     {n: 1 for n in NIGHTS}, {g: 1 for g in GAPS},
                     motif_rows=1, motif_tours=1, motif_longstays=1)
    notes, hard = {}, []
    synth.check_planted(plan, con, notes, hard)
    assert hard == []
    assert notes["planted_return_recall"] == 1.0
    assert notes["planted_tour_recall"] == 1.0
    assert notes["planted_longstay_recall"] == 1.0
    # and missed plants are hard failures with recall 0.0
    con2 = warehouse({b: 1 for b in RC_BINS}, {c: 1 for c in CT_CELLS},
                     {n: 1 for n in NIGHTS}, {g: 1 for g in GAPS})
    notes2, hard2 = {}, []
    synth.check_planted(plan, con2, notes2, hard2)
    assert len(hard2) == 3
    assert notes2["planted_return_recall"] == 0.0


# --- structure: the flags the generator proves must hold --------------------
def _struct_db(stg):
    con = duckdb.connect(":memory:")
    con.execute("create table stg_reservations (userid bigint, utrip varchar, "
                "checkin date, qf_bad_dates boolean, qf_overlap boolean)")
    for row in stg:
        con.execute("insert into stg_reservations values (?, ?, ?, ?, ?)", row)
    # the contiguity-rate note reads feat_transitions (always present in the
    # re-ingested warehouse)
    con.execute("create table feat_transitions (gap_days integer)")
    con.execute("insert into feat_transitions values (0), (0), (1)")
    return con


SPEC = {"window_start_day": 16770, "window_end_day": 17226}


def test_structure_clean_passes():
    con = _struct_db([
        (1, "1_1", "2016-05-10", False, False),
        (1, "1_1", "2016-05-12", False, False),
        (2, "2_1", "2016-07-01", False, False),
    ])
    notes, hard = {}, []
    synth.check_structure(con, SPEC, notes, hard)
    assert hard == []
    assert notes["multi_trip_users"] == 0  # fixture: single-trip users


def test_bad_dates_and_overlap_are_hard_failures():
    con = _struct_db([
        (1, "1_1", "2016-05-12", True, False),    # checkout <= checkin upstream
        (2, "2_1", "2016-05-10", False, True),    # overlapping presence
    ])
    notes, hard = {}, []
    synth.check_structure(con, SPEC, notes, hard)
    assert len(hard) == 2, hard


def test_window_violation_and_bad_utrip_are_hard_failures():
    # 1.3.0: multi-trip users are EXPECTED (the trips_per_user release);
    # the hard failures are window violations and malformed utrip names
    con = _struct_db([
        (1, "1_1", "2014-05-10", False, False),   # before the public window
        (2, "2_1", "2016-05-10", False, False),
        (2, "2_2", "2016-08-10", False, False),   # second trip: fine now
        (3, "3_x", "2016-05-10", False, False),   # utrip not "<user>_<k>"
    ])
    notes, hard = {}, []
    synth.check_structure(con, SPEC, notes, hard)
    assert any("public window" in h for h in hard), hard
    assert any("'<user>_<k>'" in h for h in hard), hard
    assert not any("more than one" in h for h in hard), hard
    assert notes["multi_trip_users"] == 1


# --- replicates: the band verdict separates healthy from degenerate ---------
RELEASED, TOL = 0.3807, 0.15


def test_band_verdict_rejects_a_hugging_distribution():
    hugging = [0.231, 0.234, 0.237, 0.233, 0.235, 0.232, 0.236, 0.234]
    v = synth.band_verdict(hugging, RELEASED, TOL)
    assert not v["passed"], v
    assert v["margin_over_lower_edge"] < synth.REPLICATE_MIN_MARGIN


def test_band_verdict_accepts_a_healthy_distribution():
    healthy = [0.39, 0.41, 0.38, 0.40, 0.42, 0.37, 0.39, 0.41]
    v = synth.band_verdict(healthy, RELEASED, TOL)
    assert v["passed"], v
    assert v["n_in_band"] == len(healthy)


def test_band_verdict_needs_both_fraction_and_margin():
    # healthy mean but 2 of 8 runs below the band -> fail on fraction
    spread = [0.40, 0.41, 0.42, 0.40, 0.41, 0.42, 0.10, 0.11]
    v = synth.band_verdict(spread, RELEASED, TOL)
    assert v["n_in_band"] == 6 and not v["passed"], v
    # one stray run of eight is tolerated (7/8 = 0.875 >= 0.85)
    one_out = [0.40, 0.41, 0.42, 0.40, 0.41, 0.42, 0.40, 0.10]
    assert synth.band_verdict(one_out, RELEASED, TOL)["passed"]


def test_released_repeat_share_uses_band_midpoints():
    gs = {"return_rate": {"cells": [
        {"name": "rr_0_19", "w": 62}, {"name": "rr_20_39", "w": 0},
        {"name": "rr_40_59", "w": 0}, {"name": "rr_60_79", "w": 0},
        {"name": "rr_80_100", "w": 38}]}}
    # 62% of users in the 0-19 band (midpoint 10%) + 38% in 80-100 (90%)
    assert abs(synth.released_repeat_share(gs) - (0.62 * 10 + 0.38 * 90) / 100) < 1e-9


def test_ct_of_and_gap_cell_are_the_public_grids():
    assert synth.ct_of("Gondal") == "Gondal"
    assert synth.ct_of("Urkesh") == "ct_other"
    assert synth.ct_of("ct_other") == "ct_other"
    assert [synth.gap_cell(g) for g in (0, 1, 2, 3, 4, 33)] == [
        "g_0", "g_1", "g_2_3", "g_2_3", "g_4p", "g_4p"]

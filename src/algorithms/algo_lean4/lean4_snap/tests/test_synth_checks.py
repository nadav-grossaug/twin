"""The validator's affiliation checks must FAIL on the defects they exist to
catch. Pure unit tests: an in-memory DuckDB stands in for the re-ingested
synthetic warehouse, so no pipeline artifacts are needed."""
import importlib.util
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("synth_script", ROOT / "scripts" / "synth.py")
assert _spec is not None and _spec.loader is not None
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)

BINS = ["g_2_4", "g_5_8", "g_9_16", "g_17p"]
# genspec group-count weights (post affil_member_weighted) and released density %
EDU_WEIGHTS = {"g_2_4": 55, "g_5_8": 26, "g_9_16": 20, "g_17p": 68}
EDU_DENSITY = {"g_2_4": 78, "g_5_8": 34, "g_9_16": 25, "g_17p": 17}


def genspec():
    return {"affiliation": {
        "categories": ["education"],
        "group_ranges": {b: [0, 0] for b in BINS},
        "density_den": 100,
        "size_rows": [{"given": "education", "cells": [
            {"name": b, "w": EDU_WEIGHTS[b]} for b in BINS]}],
        "density_rows": [{"given": "education", "cells": [
            {"name": b, "w": EDU_DENSITY[b]} for b in BINS]}],
    }}


def run(groups: dict, densities: dict | None = None):
    """groups: bin -> n synthetic groups; densities: bin -> density percent
    (default: the released percent)."""
    con = duckdb.connect(":memory:")
    con.execute("create table mart_affil_release (category varchar, size_bin "
                "varchar, n_groups bigint, n_member_nodes bigint, density_pct bigint)")
    for b, n in groups.items():
        d = (densities or {}).get(b, EDU_DENSITY[b])
        con.execute("insert into mart_affil_release values ('education', ?, ?, ?, ?)",
                    [b, n, n * 5, d])
    con.execute("create table mart_edge_shared (total_shared bigint)")
    con.execute("insert into mart_edge_shared values (1)")
    notes, soft = {}, []
    synth.check_affiliation(genspec(), {}, con, notes, soft)
    return notes, soft


def test_healthy_synth_passes():
    _, soft = run({"g_2_4": 55, "g_5_8": 27, "g_9_16": 19, "g_17p": 66})
    assert soft == []


def test_old_member_weight_defect_is_caught():
    """The pre-fix generator drew bins by MEMBER weight: education came out as
    4/6/13/174 groups against spec weights 55/26/20/68. The old validator
    skipped those sparse cells and passed; this must now fail."""
    _, soft = run({"g_2_4": 4, "g_5_8": 6, "g_9_16": 13, "g_17p": 174})
    assert any("group-count mix TV" in s for s in soft), soft


def test_empty_expected_bin_is_caught():
    _, soft = run({"g_2_4": 0, "g_5_8": 30, "g_9_16": 25, "g_17p": 70})
    assert any("g_2_4" in s and "none generated" in s for s in soft), soft


def test_no_groups_at_all_is_caught():
    _, soft = run({})
    assert any("has none" in s for s in soft), soft


def test_density_off_in_a_well_populated_cell_is_caught():
    _, soft = run({"g_2_4": 55, "g_5_8": 26, "g_9_16": 20, "g_17p": 68},
                  densities={"g_17p": 17 + 40})
    assert any("density education/g_17p" in s for s in soft), soft


def test_sparse_cell_uses_wider_tolerance():
    """10 groups: a 0.30 miss is within the sparse tolerance (0.40), a 0.50 miss is not."""
    base = {"g_2_4": 55, "g_5_8": 26, "g_9_16": 10, "g_17p": 68}
    _, soft = run(base, densities={"g_9_16": 25 + 30})
    assert not any("g_9_16" in s for s in soft), soft
    _, soft = run(base, densities={"g_9_16": 25 + 50})
    assert any("density education/g_9_16" in s for s in soft), soft


def test_tiny_cells_are_reported_not_silently_skipped():
    notes, soft = run({"g_2_4": 55, "g_5_8": 26, "g_9_16": 3, "g_17p": 68})
    assert any("g_9_16" in u for u in notes["affil_unjudged"]), notes["affil_unjudged"]
    assert notes["affil_cells"]["education/g_9_16"]["n_groups"] == 3
    assert not any("density education/g_9_16" in s for s in soft)


@pytest.mark.parametrize("key", ["n_groups", "expected_groups",
                                 "synth_density", "released_density"])
def test_every_cell_records_its_evidence(key):
    notes, _ = run({"g_2_4": 55, "g_5_8": 26, "g_9_16": 20, "g_17p": 68})
    assert all(key in c for c in notes["affil_cells"].values())


# --- closure is judged over replicate runs, not one sample -----------------
RELEASED, TOL = 0.6209, 0.15  # the band [0.4709, 0.7709]

# per-node closure of 8 consecutive generator runs measured BEFORE the
# closure-biased density fill (affiliation planted, window 8)
BEFORE_FIX = [0.5008, 0.459, 0.4598, 0.4566, 0.4986, 0.468, 0.4859, 0.4732]
# 8 runs after it
AFTER_FIX = [0.4984, 0.5045, 0.5239, 0.5321, 0.5309, 0.5257, 0.5266, 0.5068]


def test_closure_verdict_rejects_the_marginal_distribution():
    """Any single one of these runs could pass or fail by luck; the
    distribution is what is wrong (half the runs below the band, mean on its
    edge)."""
    v = synth.closure_verdict(BEFORE_FIX, RELEASED, TOL)
    assert not v["passed"], v
    assert v["fraction_in_band"] < synth.REPLICATE_MIN_FRACTION
    assert v["margin_over_lower_edge"] < synth.REPLICATE_MIN_MARGIN


def test_closure_verdict_accepts_the_fixed_distribution():
    v = synth.closure_verdict(AFTER_FIX, RELEASED, TOL)
    assert v["passed"], v
    assert v["n_in_band"] == len(AFTER_FIX)


def test_closure_verdict_needs_both_fraction_and_margin():
    # all inside the band but the mean hugs the lower edge -> still a fail
    hugging = [0.472, 0.475, 0.478, 0.474, 0.476, 0.473, 0.477, 0.475]
    assert not synth.closure_verdict(hugging, RELEASED, TOL)["passed"]
    # healthy mean but 2 of 8 runs below the band -> fail on fraction
    spread = [0.55, 0.56, 0.57, 0.55, 0.56, 0.57, 0.40, 0.41]
    v = synth.closure_verdict(spread, RELEASED, TOL)
    assert v["n_in_band"] == 6 and not v["passed"], v
    # one stray run of eight is tolerated (7/8 = 0.875 >= 0.85)
    one_out = [0.55, 0.56, 0.57, 0.55, 0.56, 0.57, 0.55, 0.40]
    assert synth.closure_verdict(one_out, RELEASED, TOL)["passed"]


# --- sample size: tolerances widen for few groups, the defect still fails ---
def test_few_groups_get_a_wider_count_tolerance():
    """At n=200 nodes there are ~12 groups; the same count-mix distance that
    is a defect at 224 groups is just sampling noise here."""
    # 12 groups spread unevenly across the 4 bins: TV ~0.3 vs the spec mix
    notes, soft = run({"g_2_4": 3, "g_5_8": 4, "g_9_16": 1, "g_17p": 4})
    assert not any("group-count mix" in s for s in soft), soft
    assert notes["affil_group_count_tol_education"] > 0.30


def test_wider_tolerance_does_not_hide_the_old_defect():
    """4/6/13/174 groups (the member-weight defect, n=197) must still fail."""
    notes, soft = run({"g_2_4": 4, "g_5_8": 6, "g_9_16": 13, "g_17p": 174})
    assert any("group-count mix" in s for s in soft), soft
    assert notes["affil_group_count_tol_education"] < 0.25


def test_below_the_validated_window_findings_are_unjudged_not_dropped():
    """judged=False keeps the finding (in notes) but raises no soft failure."""
    con = duckdb.connect(":memory:")
    con.execute("create table mart_affil_release (category varchar, size_bin "
                "varchar, n_groups bigint, n_member_nodes bigint, density_pct bigint)")
    for b, n in {"g_2_4": 4, "g_5_8": 6, "g_9_16": 13, "g_17p": 174}.items():
        con.execute("insert into mart_affil_release values ('education', ?, ?, ?, ?)",
                    [b, n, n * 5, EDU_DENSITY[b]])
    con.execute("create table mart_edge_shared (total_shared bigint)")
    con.execute("insert into mart_edge_shared values (1)")
    notes, soft = {}, []
    synth.check_affiliation(genspec(), {}, con, notes, soft, judged=False)
    assert soft == []
    assert any("group-count mix" in m for m in notes["affil_unjudged_small_n"])


def test_validated_window_matches_the_lean_constants():
    src = (ROOT / "lean" / "Generate.lean").read_text()
    import re
    lo = int(re.search(r"def validatedMinNodes : Nat := (\d+)", src).group(1))
    hi = int(re.search(r"def validatedMaxNodes : Nat := (\d+)", src).group(1))
    assert (lo, hi) == (synth.VALIDATED_MIN_NODES, synth.VALIDATED_MAX_NODES)


# --- per-cell density tolerance scales with the released density -----------
def test_density_tol_is_proportional_with_floor_and_cap():
    assert synth.density_tol(0.10) == synth.AFFIL_DENSITY_TOL_FLOOR      # floor
    assert abs(synth.density_tol(0.40) - 0.24) < 1e-9                    # 0.6 * 0.40
    assert synth.density_tol(0.90) == synth.AFFIL_DENSITY_TOL            # cap


def test_a_mid_bin_overshoot_the_flat_tolerance_missed_is_now_caught():
    """work 5-8 came out at 0.46 against a released 0.24 (a 0.22 miss, nearly 2x).
    The old flat 0.25 tolerance passed it; the proportional rule must not."""
    base = {"g_2_4": 55, "g_5_8": 26, "g_9_16": 20, "g_17p": 68}
    _, soft = run(base, densities={"g_5_8": 34 + 22})        # released 0.34, got 0.56
    assert any("density education/g_5_8" in s for s in soft), soft


def test_a_high_density_cell_keeps_a_wider_absolute_tolerance():
    """released 0.78: a 0.20 miss is within the cap (0.30) and not a failure."""
    base = {"g_2_4": 55, "g_5_8": 26, "g_9_16": 20, "g_17p": 68}
    _, soft = run(base, densities={"g_2_4": 78 - 20})
    assert not any("density education/g_2_4" in s for s in soft), soft


# --- replicate-level affiliation density verdict -----------------------------
def test_density_verdict_separates_the_healthy_generator_from_the_old_ones():
    healthy = [0.046, 0.058, 0.052, 0.049]
    old_single_window = [0.095, 0.119, 0.105, 0.115]
    too_tight_window = [0.241, 0.23, 0.25]
    assert synth.density_verdict(healthy)["passed"]
    assert not synth.density_verdict(old_single_window)["passed"]
    assert not synth.density_verdict(too_tight_window)["passed"]

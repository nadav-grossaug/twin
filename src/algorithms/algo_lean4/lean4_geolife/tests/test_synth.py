"""Synthetic GeoLife trajectories: meta/provenance declarations, planted
patterns declared in the plan, round-trip validation and the kernel-checked
certificate. Per-column structural checks live in test_synth_columns.py."""
import hashlib
import json

import pytest


@pytest.fixture(scope="module")
def synth(root):
    s = root / "synth"
    if not (s / "geolife_synth.tsv").exists():
        pytest.skip("run `make synth` first")
    plan = json.loads((s / "plan.json").read_text())
    meta = json.loads((s / "meta.json").read_text())
    return s, plan, meta


def test_meta_declares_gaps_and_provenance(synth):
    _, _, meta = synth
    gaps = meta["assumptions"]["fields_not_modelled"]
    assert isinstance(gaps, list) and gaps, (
        "the generator must DECLARE what it does not model")
    assert meta["genspec_sha256"] and meta["dp_release_sha256"]


def test_meta_declares_manipulation(synth):
    """The Beijing-exclusion / Tokyo-placement step is a declared
    manipulation, not a silent rewrite."""
    _, _, meta = synth
    assert meta.get("manipulation"), "manipulation block missing from meta.json"


def test_meta_sha_matches_live_genspec(synth, root):
    _, _, meta = synth
    gs_p = root / "lean" / "out" / "genspec.json"
    if not gs_p.exists():
        pytest.skip("run `make train` (distill) first")
    if meta["genspec_sha256"] != hashlib.sha256(gs_p.read_bytes()).hexdigest():
        pytest.skip("synth predates the current genspec — rerun `make synth`")
    gs = json.loads(gs_p.read_text())
    assert meta["dp_release_sha256"] == gs["source"]["dp_release_sha256"]


def test_plan_declares_planted_patterns(synth):
    _, plan, _ = synth
    p = plan["planted"]
    assert p["commute_user"] == "001"
    assert p["beacon_user"] == "002"
    assert p["tour_user"] == "002"


def test_plan_population(synth):
    _, plan, _ = synth
    users = plan["users"]
    assert len(users) == plan["n_users"]
    ids = [u["user"] for u in users]
    assert len(set(ids)) == len(ids)
    assert plan["planted"]["commute_user"] in ids
    assert plan["planted"]["tour_user"] in ids


def test_planted_commuter_in_plan(synth):
    _, plan, _ = synth
    users = {u["user"]: u for u in plan["users"]}
    motifs = list(users[plan["planted"]["commute_user"]]["motifs"])
    assert motifs and all(m == "commute" for m in motifs), motifs


def test_planted_tour_in_plan(synth):
    _, plan, _ = synth
    users = {u["user"]: u for u in plan["users"]}
    assert "tour" in list(users[plan["planted"]["tour_user"]]["motifs"])


def test_validation_passed(synth):
    s, _, _ = synth
    p = s / "validation.json"
    if not p.exists():
        pytest.skip("run `make synth` (validate step) first")
    v = json.loads(p.read_text())
    assert not v["hard_failures"], v["hard_failures"]


def test_certificate_kernel_checked(synth):
    s, _, _ = synth
    p = s / "certificate.json"
    if not p.exists():
        pytest.skip("run `make certify` first")
    cert = json.loads(p.read_text())
    assert cert["proved"] is True
    assert cert["kernel_only"] is True


# ------------------------------------- time / volume checks (validate helper)
def _mini_scratch(dt_walk: float, dep: dict):
    import duckdb
    c = duckdb.connect()
    c.execute("create table mart_kinematics(mode_class varchar, median_dt_s double)")
    c.execute("insert into mart_kinematics values ('walk', ?)", [dt_walk])
    c.execute("create table feat_visits(dep_bin varchar, dep_censored boolean)")
    for b, n in dep.items():
        c.execute("insert into feat_visits select ?, false from range(?)", [b, n])
    c.execute("create table stg_points(user_id varchar, datetime timestamp)")
    c.execute("insert into stg_points values ('001', '2008-01-01 09:00:00'), "
              "('002', '2008-01-01 21:00:00')")
    return c


GS = {"departure": {"cells": [{"name": "dep_morning", "w": 80},
                              {"name": "dep_evening", "w": 20}]},
      "kinematics": [{"mode": "walk", "speed_mmps": 1100, "dt_ds": 20}]}


def _run_check(con):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import synth as synth_mod
    notes, soft = {}, []
    synth_mod._check_time_and_volume(con, GS, notes, soft)
    return notes, soft


def test_validate_cadence_guards_volume():
    # released walk cadence 2 s; the pre-fix generator sampled ~30 s
    notes, soft = _run_check(_mini_scratch(30.0, {"dep_morning": 8, "dep_evening": 2}))
    assert any(s.startswith("cadence: walk") for s in soft), soft
    notes, soft = _run_check(_mini_scratch(2.0, {"dep_morning": 8, "dep_evening": 2}))
    assert not soft, soft
    assert notes["volume"]["n_users"] == 2
    assert len(notes["point_hour_share_local"]) == 24


def test_validate_departure_mix_guards_time_of_day():
    # all departures in the evening vs a released 80% morning mix
    notes, soft = _run_check(_mini_scratch(2.0, {"dep_evening": 10}))
    assert notes["departure_mix_tv_vs_release"] == 0.8
    assert any(s.startswith("departure_mix_tv") for s in soft), soft


def test_plan_days_per_user_are_heavy_tailed(synth):
    """Recorded days per (non-planted) user follow the public heavy-tailed
    prior, not a fixed slice: spread across classes, all within bounds."""
    _, plan, _ = synth
    days = [u.get("n_days") for u in plan["users"][2:]]
    if any(d is None for d in days):
        pytest.skip("plan predates per-user day counts — run `make synth`")
    assert all(2 <= d <= 128 for d in days), days
    assert len(set(days)) > len(days) // 4, "day counts look fixed, not drawn"
    if len(days) >= 30:
        days.sort()
        assert days[-len(days) // 10] >= 2.5 * days[len(days) // 2], (
            "p90 should be several times the median (heavy tail; the prior's "
            "expected ratio is ~5–8, 2.5× leaves room for n≈60 sampling)")

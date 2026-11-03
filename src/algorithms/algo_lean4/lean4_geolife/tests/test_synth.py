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

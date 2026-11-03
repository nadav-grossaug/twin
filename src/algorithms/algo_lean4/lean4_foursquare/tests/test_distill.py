"""Distillation: the genspec is pure post-processing of released values —
every weight equals a released value or released + a DECLARED public prior."""
import hashlib
import json

import pytest

from spec import REGIONS

TC_BINS = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
GAPS = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
DURS = ["dur_day", "dur_week", "dur_month", "dur_quarter", "dur_long"]
DISTS = ["dist_short", "dist_mid", "dist_long", "dist_ultra"]
EPS = ["ep_1", "ep_2", "ep_3", "ep_4_5", "ep_6p"]
RRS = ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]


@pytest.fixture(scope="module")
def rel_and_spec(root):
    rel_p = root / "lean" / "out" / "dp_release.json"
    gs_p = root / "lean" / "out" / "genspec.json"
    if not rel_p.exists() or not gs_p.exists():
        pytest.skip("run `make train` first")
    rel = json.loads(rel_p.read_text())
    gs = json.loads(gs_p.read_text())
    if gs["source"]["dp_release_sha256"] != hashlib.sha256(
            rel_p.read_bytes()).hexdigest():
        pytest.skip("genspec predates the current (non-reproducible) release — "
                    "run `make distill` after the lean release")
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel["aux"]}
    return rel, gs, aux


def _cells(gs, key):
    return {c["name"]: c["w"] for c in gs[key]["cells"]}


def test_travel_count_equals_release(rel_and_spec):
    rel, gs, _ = rel_and_spec
    released = {c["cell"]: max(0, int(c["noised_count"])) for c in rel["cells"]}
    got = _cells(gs, "travel_count")
    assert set(got) == set(TC_BINS)
    for name, w in got.items():
        assert w == released[name], name


@pytest.mark.parametrize("key,aux_name,names", [
    ("region_mix", "region", REGIONS),
    ("gap_class", "gap_class", GAPS),
    ("duration_class", "duration_class", DURS),
    ("distance_class", "distance_class", DISTS),
    ("month", "month", [f"m{i:02d}" for i in range(1, 13)]),
    ("hod", "hod", [f"h{i:02d}" for i in range(24)]),
    ("episode_len", "episode_len", EPS),
    ("return_rate", "return_rate", RRS),
])
def test_marginals_equal_release(rel_and_spec, key, aux_name, names):
    _, gs, aux = rel_and_spec
    got = _cells(gs, key)
    assert set(got) == set(names), key
    for name, w in got.items():
        assert w == max(0, aux[aux_name].get(name, 0)), (key, name)


def test_transition_rows_follow_declared_prior(rel_and_spec):
    """Transition weights = clamped released counts, plus EXACTLY the declared
    derived_from_data:false return_boost prior on the diagonal."""
    _, gs, aux = rel_and_spec
    trans = aux["region_transition"]
    priors = {p["name"]: p for p in gs["priors"]}
    assert "return_boost" in priors
    assert priors["return_boost"]["derived_from_data"] is False
    for row in gs["transitions"]["rows"]:
        g = row["given"]
        raw = {n: max(0, trans.get(f"{g}__{n}", 0)) for n in REGIONS}
        boost = max(1, sum(raw.values()))
        for cell in row["cells"]:
            n = cell["name"]
            want = raw[n] + (boost if n == g else 0)
            assert cell["w"] == want, (g, n, cell["w"], want)


def test_corridors_are_released_only(rel_and_spec, root):
    import csv
    _, gs, _ = rel_and_spec
    rel_dirs = sorted((root / "release").glob("*/"))
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    with open(rel_dirs[-1] / "topk_corridors.csv") as f:
        released_pairs = {(r["prev_country"], r["next_country"])
                          for r in csv.DictReader(f)}
    for c in gs["corridors"]:
        assert (c["a"], c["b"]) in released_pairs, (c["a"], c["b"])


def test_countries_are_released_only(rel_and_spec, root):
    """Country weights in the genspec must come from the suppressed
    topk_countries release — value for value, nothing below threshold."""
    import csv
    _, gs, _ = rel_and_spec
    rel_dirs = sorted((root / "release").glob("*/"))
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    with open(rel_dirs[-1] / "topk_countries.csv") as f:
        released = {r["country"]: r for r in csv.DictReader(f)}
    assert gs["countries"], "genspec has no countries"
    for c in gs["countries"]:
        assert c["c"] in released, c["c"]
        assert c["w_in"] == int(released[c["c"]]["n_in"]), c["c"]
        assert c["w_out"] == int(released[c["c"]]["n_out"]), c["c"]
        assert c["region"] == released[c["c"]]["region"], c["c"]


def test_window_and_ranges_are_public_spec(rel_and_spec):
    from spec import load_spec
    _, gs, _ = rel_and_spec
    s = load_spec()
    assert gs["window"]["start_s"] == int(s["window_start_s"])
    assert gs["window"]["end_s"] == int(s["window_end_s"])
    assert gs["gap_ranges_s"]["gap_trip"][1] == int(s["gap_trip_max_s"])


def test_fidelity_report_honest(root):
    p = root / "models_out" / "fidelity.json"
    if not p.exists():
        pytest.skip("run `make distill` first")
    rep = json.loads(p.read_text())
    assert "accepted" in rep
    # Honest gate: the first-order matrix student is NOT expected to beat the
    # second-order return-to-previous baseline (72%); accepted=False is a
    # finding, not a failure — the generator compensates with the declared
    # return_rate mechanism. The report must be CONSISTENT, not flattering.
    assert rep["accepted"] == (
        rep["student_accuracy"] >= rep["return_prev_accuracy"] - 0.02), rep

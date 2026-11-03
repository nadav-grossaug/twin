"""Distillation: the genspec is pure post-processing of released values —
every weight equals a released value or released + a DECLARED public prior."""
import hashlib
import json

import pytest

RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
            "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan",
            "Rolisica", "Kasnia", "ct_other"]
NIGHTS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
GAPS = ["g_0", "g_1", "g_2_3", "g_4p"]
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


def test_res_count_equals_release(rel_and_spec):
    rel, gs, _ = rel_and_spec
    released = {c["cell"]: max(0, int(c["noised_count"])) for c in rel["cells"]}
    got = _cells(gs, "res_count")
    assert set(got) == set(RC_BINS)
    for name, w in got.items():
        assert w == released[name], name


@pytest.mark.parametrize("key,aux_name,names", [
    ("country_mix", "country", CT_CELLS),
    ("nights", "nights", NIGHTS),
    ("gap", "gap", GAPS),
    ("month_window", "month_window", [f"mw{i:02d}" for i in range(15)]),
    ("repeat_struct", "repeat_struct",
     ["r_ext1", "r_ext2", "r_ext3p", "r_ret1", "r_ret2p", "r_both"]),
    ("trips_per_user", "trips_per_user", ["tp_1", "tp_2", "tp_3p"]),
    ("device_persist", "device_persist", ["dv_1", "dv_2", "dv_3"]),
    ("affiliate_persist", "affiliate_persist",
     ["ap_1", "ap_2", "ap_3", "ap_4p"]),
    ("device", "device", ["desktop", "mobile", "tablet"]),
    ("affiliate", "affiliate",
     ["a9924", "a359", "a384", "a9452", "a4541", "a8436", "a10332",
      "a3631", "aff_other"]),
    ("booker", "booker",
     ["Gondal", "The Devilfire Empire", "Elbonia", "Tcherkistan",
      "Bartovia"]),
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
    trans = aux["country_transition"]
    priors = {p["name"]: p for p in gs["priors"]}
    assert "return_boost" in priors
    assert priors["return_boost"]["derived_from_data"] is False
    for row in gs["transitions"]["rows"]:
        g = row["given"]
        raw = {n: max(0, trans.get(f"{g}__{n}", 0)) for n in CT_CELLS}
        boost = max(1, sum(raw.values()))
        for cell in row["cells"]:
            n = cell["name"]
            want = raw[n] + (boost if n == g else 0)
            assert cell["w"] == want, (g, n, cell["w"], want)


def test_cities_are_released_only(rel_and_spec, root):
    """City weights in the genspec must come from the suppressed
    topk_cities release — value for value, nothing below threshold."""
    import csv
    _, gs, _ = rel_and_spec
    rel_dirs = sorted((root / "release").glob("*/"))
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    with open(rel_dirs[-1] / "topk_cities.csv") as f:
        released = {r["city_id"]: r for r in csv.DictReader(f)}
    assert gs["cities"], "genspec has no cities"
    for c in gs["cities"]:
        assert str(c["city_id"]) in released, c["city_id"]
        assert c["w"] == int(released[str(c["city_id"])]["n_res"]), c["city_id"]
        assert c["country"] == released[str(c["city_id"])]["hotel_country"], c
        assert c["ct"] == released[str(c["city_id"])]["hotel_ct"], c


def test_corridors_are_released_only(rel_and_spec, root):
    import csv
    _, gs, _ = rel_and_spec
    rel_dirs = sorted((root / "release").glob("*/"))
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    with open(rel_dirs[-1] / "topk_city_transitions.csv") as f:
        released_pairs = {(r["prev_city"], r["next_city"])
                          for r in csv.DictReader(f)}
    for c in gs["corridors"]:
        assert (str(c["a_city"]), str(c["b_city"])) in released_pairs, c


def test_window_and_ranges_are_public_spec(rel_and_spec):
    from spec import load_spec
    _, gs, _ = rel_and_spec
    s = load_spec()
    assert gs["window"]["start_day"] == int(s["window_start_day"])
    assert gs["window"]["end_day"] == int(s["window_end_day"])
    assert gs["gap_ranges_d"]["g_2_3"] == [int(s["gap_e2"]), int(s["gap_e4"]) - 1]
    assert gs["nights_ranges"]["n_4_6"] == [int(s["nights_e4"]),
                                           int(s["nights_e7"]) - 1]
    assert gs["rc_bin_ranges"]["rc_13p"][0] == int(s["rc_e13"])


def test_single_trip_and_uniform_dow_priors_declared(rel_and_spec):
    """The generator's declared public priors must all be present and
    derived_from_data:false — no silent fabrication."""
    _, gs, _ = rel_and_spec
    priors = {p["name"]: p for p in gs["priors"]}
    for name in ("return_boost", "uniform_dow", "multi_trip",
               "within_bin_decay", "channel_persistence"):
        assert name in priors, name
        assert priors[name]["derived_from_data"] is False, name


def test_fidelity_report_honest(root):
    p = root / "models_out" / "fidelity.json"
    if not p.exists():
        pytest.skip("run `make distill` first")
    rep = json.loads(p.read_text())
    assert "accepted" in rep
    # Honest gate: the first-order matrix student is NOT expected to beat the
    # second-order return-to-previous baseline; accepted=False is a finding,
    # not a failure — the generator compensates with the declared
    # return_rate mechanism. The report must be CONSISTENT, not flattering.
    assert rep["accepted"] == (
        rep["student_accuracy"] >= rep["return_prev_accuracy"] - 0.02), rep

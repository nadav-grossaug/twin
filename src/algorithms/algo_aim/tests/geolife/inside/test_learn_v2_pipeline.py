"""End-to-end: fixtures -> measurements -> ledger -> fit -> artifact round trip."""
import json

import numpy as np
import pytest

from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage
from geolife.inside.dp import zcdp_to_epsilon
from geolife.transfer import schema_v2 as S2
from geolife.outside.fit_v2 import DEFAULTS, fit_behaviour_v2
from geolife.inside.measure_v2 import VocabV2, build_measurements_v2
from geolife.inside.learn_v2 import run
from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users


@pytest.fixture(scope="module")
def data():
    return make_users(40, seed=2), VocabV2.from_bbox(BBOX, CATS)


def test_budget_ledger_and_names(data):
    users, vocab = data
    ms, led = build_measurements_v2(users, vocab, 0.5, np.random.default_rng(0))
    assert {m.name for m in ms} == set(S2.ALL_NAMES)
    assert led.rho_total <= 0.5 * (1 + 1e-6) and led.rho_total > 0.45
    assert led.epsilon == pytest.approx(zcdp_to_epsilon(led.rho_total, led.delta))
    text = " ".join(led.caveats).lower()
    for must in ("informed by real data", "bin edges", "floating-point"):
        assert must in text
    for m in ms:
        assert m.l2_sensitivity == pytest.approx(S2.l2_bound(m.name))


def test_hierarchy_vocab_depends_only_on_noisy_parents(data):
    users, vocab = data
    ms, _ = build_measurements_v2(users, vocab, 0.5, np.random.default_rng(0))
    by = {m.name: m for m in ms}
    r7 = by["place_users_r7"]
    parents = {k[0] for k, v in zip(by["place_users_r6"].keys, by["place_users_r6"].values)
               if v >= 2.0 * by["place_users_r6"].sigma}
    import h3
    assert {h3.cell_to_parent(k[0], 6) for k in r7.keys} <= parents


def test_fit_params_shape_and_fallbacks(data):
    users, vocab = data
    ms, led = build_measurements_v2(users, vocab, 5.0, np.random.default_rng(1))
    p = fit_behaviour_v2(ms, led)
    assert p["version"] == 2
    assert set(p) >= {"places", "users", "groups", "outing", "time", "route", "diagnostics"}
    assert set(p["places"]) >= {"zipf_s", "pool_size", "cat_mix", "users_per_place_hist"}
    assert sum(p["outing"]["stops_pmf"].values()) == pytest.approx(1.0)
    for d in S2.DAYTYPES:
        assert sum(p["outing"]["start_pmf"][d].values()) == pytest.approx(1.0)
    assert p["users"]["outings_per_user"]["sigma"] > 0
    json.dumps(p)                                   # JSON-safe


def test_pure_noise_falls_back_to_defaults(data):
    users, vocab = data
    ms, led = build_measurements_v2(users, vocab, 1e-6, np.random.default_rng(1))
    p = fit_behaviour_v2(ms, led)
    fb = p["diagnostics"]["fallbacks"]
    assert S2.OUTING_DURATION in fb and "zipf_s" in fb
    assert p["outing"]["duration_min"] == DEFAULTS["outing"]["duration_min"]
    assert p["places"]["zipf_s"] == DEFAULTS["places"]["zipf_s"]


def test_signal_recovered_at_large_budget(data):
    users, vocab = data
    ms, led = build_measurements_v2(users, vocab, 200.0, np.random.default_rng(1))
    p = fit_behaviour_v2(ms, led)
    assert not any(k.startswith("outing_") or k.startswith("user_") for k in p["diagnostics"]["fallbacks"])
    assert p["users"]["outings_per_user"]["mu"] > np.log(15)       # fixture median ~30


def test_pav_monotone():
    from geolife.outside.fit_v2 import _pav_decreasing
    y = _pav_decreasing(np.array([5.0, 1.0, 3.0, 0.0, 2.0]))
    assert (np.diff(y) <= 1e-12).all() and y.sum() == pytest.approx(11.0)


def test_cli_run_roundtrip(data, tmp_path):
    users, _ = data
    res = run(users, tmp_path / "art", epsilon=6.0, bbox=BBOX, categories=CATS, verbose=False)
    art = load_artifact(tmp_path / "art")
    validate_no_raw_leakage(art)
    assert art.behaviour["params_v2"]["version"] == 2
    assert len(res["snr"]) == len(S2.ALL_NAMES)


def test_data_dependent_design_caveats_present(data):
    users, vocab = data
    _, led = build_measurements_v2(users, vocab, 0.5, np.random.default_rng(0))
    text = " ".join(led.caveats).lower()
    for must in ("data-dependent design choice", "outside the formal dp guarantee",
                 "signal-to-noise of a previous run on the real data", "conditional",
                 "coarse time bins", "table list", "not measured"):
        assert must in text, must
    assert {"place_time_users", "stop_transitions_r6", "trav_users_r8", "trav_users_r9"}.isdisjoint(S2.ALL_NAMES)


def test_coarse_time_bins_and_expansion():
    from geolife.outside.fit_v2 import _expand_coarse
    from geolife.transfer.types import local_coarse_bin, tod_coarse

    def h(hour):
        return hour * 3600 - 8 * 3600 + 86400 * 100
    assert [local_coarse_bin(h(x)) for x in (4, 5, 8, 13, 14, 17, 23, 0)] ==         ["pm", "am", "am", "am", "pm", "pm", "pm", "pm"]
    assert tod_coarse("morning") == "am" and tod_coarse("evening") == "pm"
    for w in S2.COARSE_TO_FINE.values():
        assert sum(w.values()) == pytest.approx(1.0)
    p = _expand_coarse([3.0, 1.0])
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["morning"] == pytest.approx(0.75 * 6 / 9)
    assert _expand_coarse([0.0, 0.0]) == {t: 0.25 for t in p}

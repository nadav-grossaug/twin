import json
import math

import numpy as np
import pytest

from geolife.transfer import schema as S
from geolife.eval.fakedata import cached_fake_artifact
from geolife.outside.fit import _fit_zipf, _pmf, fit_behaviour, fit_decay
from geolife.transfer.types import Measurement, PrivacyLedger


def _params():
    _, _, ms, led = cached_fake_artifact()
    return fit_behaviour(ms, led, extent_m=8000.0)


def test_pmf_nonneg_and_normalised():
    p = _pmf([-5.0, 0.2, 30.0, 10.0], sigma=1.0)
    assert (p >= 0).all() and abs(p.sum() - 1) < 1e-12
    assert p[0] == 0 and p[1] == 0          # below sigma -> denoised away
    u = _pmf([-1.0, -2.0, -3.0], sigma=1.0)  # all negative -> uniform
    assert np.allclose(u, 1 / 3)


def test_params_shape_and_json():
    p = _params()
    json.dumps(p, allow_nan=False)           # no inf/nan anywhere
    assert 350 < p["n_users_source"] < 450
    assert 0.65 < p["work_frac"] < 0.95      # truth 0.8
    assert abs(sum(p["other_anchor_pmf"]) - 1) < 1e-9
    for ro in S.ROLES:
        assert abs(sum(p["hour_profile"][ro]) - 1) < 1e-9 and len(p["hour_profile"][ro]) == 24
        assert abs(sum(p["dwell"][ro]["p"]) - 1) < 1e-9
        assert abs(sum(p["role_trans"][ro].values()) - 1) < 1e-9
    for k in S.KINDS:
        assert abs(sum(p["trip_dist"][k]["p"]) - 1) < 1e-9
    assert abs(sum(p["detour"]["p"]) - 1) < 1e-9
    assert 0.5 < p["zipf_s"] < 1.6           # truth 1.0 (data-generating popularity)
    assert int(np.argmax(p["hour_profile"]["work"])) in (9, 10)
    assert p["role_trans"]["home"]["work"] > p["role_trans"]["home"]["other"]
    assert 0 <= p["explore_prob"] <= 0.5
    assert p["privacy"]["rho_total"] > 0


def test_fit_uses_only_measurements_and_checks_ledger():
    _, _, ms, led = cached_fake_artifact()
    bad = PrivacyLedger(entries=[e for e in led.entries if e["name"] != "n_users"])
    with pytest.raises(ValueError):
        fit_behaviour(ms, bad)
    with pytest.raises(ValueError):
        fit_behaviour([m for m in ms if m.name != "place_role_users"], led)
    # works from a plain measurement list with a perturbed copy: no hidden data access
    ms2 = [Measurement(m.name, m.keys, list(m.values), m.sigma, m.l2_sensitivity, m.rho) for m in ms]
    assert fit_behaviour(ms2, led)["n_users_source"] == fit_behaviour(ms, led)["n_users_source"]


def test_zipf_fit_recovers_exponent():
    N, A, M, s = 400.0, 2.5, 120, 1.1
    w = np.arange(1, M + 1) ** -s
    w /= w.sum()
    from scipy.optimize import brentq
    lam = brentq(lambda l: (1 - np.exp(-l * w)).sum() - A, 1e-9, 1e12)
    exp_counts = N * (1 - np.exp(-lam * w))
    top = exp_counts[exp_counts >= 4]
    s_hat, m_hat = _fit_zipf(top, N, A)
    assert abs(s_hat - s) < 0.25
    assert m_hat >= len(top)


def test_decay_monotone_in_mean_distance():
    near = fit_decay(math.log(500), 8000.0, 1.0)
    far = fit_decay(math.log(4000), 8000.0, 1.0)
    assert near > far >= 0

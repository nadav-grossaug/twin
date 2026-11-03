"""Shared fit helpers of outside/fit_v2 (denoising, Zipf, distance decay, weighted rho split).

The fit-on-measurements path itself is covered by tests/geolife/inside/test_learn_v2_pipeline.py
(synthetic measurements) and tests/geolife/outside/test_fit_learned_artifacts.py (committed
artifacts); this file keeps the unit tests of the pure helpers that moved here from the retired v1
outside/fit.py and inside/measure.py.
"""
import math

import numpy as np
import pytest

from geolife.inside.measure_v2 import FLOOR_FRAC, rho_weight, weighted_rho_split
from geolife.outside.fit_v2 import _fit_zipf, _pmf, fit_decay


def test_pmf_nonneg_and_normalised():
    p = _pmf([-5.0, 0.2, 30.0, 10.0], sigma=1.0)
    assert (p >= 0).all() and abs(p.sum() - 1) < 1e-12
    assert p[0] == 0 and p[1] == 0          # below sigma -> denoised away
    u = _pmf([-1.0, -2.0, -3.0], sigma=1.0)  # all negative -> uniform
    assert np.allclose(u, 1 / 3)


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


def test_weighted_rho_split_sums_and_floors():
    weights = [rho_weight(math.sqrt(m), 1.0, v, pri)
               for m, v, pri in ((1, 2, 1.0), (12, 400, 6.0), (50, 20000, 1.0), (1, 60, 1.0))]
    rho = 0.5
    shares = weighted_rho_split(rho, weights)
    assert sum(shares) == pytest.approx(rho)
    floor = rho * FLOOR_FRAC / len(weights)
    assert all(s >= floor * (1 - 1e-12) for s in shares)
    # weights are monotone in sensitivity^2 * ln(vocab) * priority
    order = np.argsort(weights)
    assert list(np.argsort(shares)) == list(order)


def test_rho_weight_uses_public_quantities_only():
    # (l2/clip)^2 = max keys per user; the clip itself cancels
    assert rho_weight(2.0, 1.0, 100) == pytest.approx(4 * math.log(100))
    assert rho_weight(4.0, 2.0, 100) == pytest.approx(rho_weight(2.0, 1.0, 100))
    assert rho_weight(1.0, 1.0, 1) == pytest.approx(math.log(2))        # vocab floor at 2
    assert rho_weight(1.0, 1.0, 100, priority=3.0) == pytest.approx(3 * math.log(100))

"""Tests for the classifiers: separation on a linear toy, uniform score orientation, registry."""

import numpy as np
import pytest

from realfake.metrics import accuracy, auc
from realfake.models import MODELS, fit_gbm, fit_logreg


def linear_toy(n=400, d=4, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(shift, 1, (n // 2, d))])
    y = np.repeat([0, 1], n // 2)
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    half = n // 2
    return x[:half], y[:half], x[half:], y[half:]


def test_logreg_separates_linear_toy():
    xtr, ytr, xte, yte = linear_toy()
    model = fit_logreg(xtr, ytr)
    assert auc(model.scores(xte), yte) > 0.95


def test_gbm_separates_linear_toy():
    pytest.importorskip("sklearn")
    xtr, ytr, xte, yte = linear_toy()
    model = fit_gbm(xtr, ytr, seed=0)
    assert auc(model.scores(xte), yte) > 0.9


def test_registry_logreg_always_gbm_when_sklearn():
    assert "logreg" in MODELS
    try:
        import sklearn  # noqa: F401
    except ImportError:
        assert "gbm" not in MODELS
    else:
        assert "gbm" in MODELS


def test_scores_orientation_uniform_across_models():
    """Higher score = more fake (FAKE=1), log-odds scale: threshold 0.0 classifies well."""
    xtr, ytr, xte, yte = linear_toy()
    for name, fit in MODELS.items():
        model = fit(xtr, ytr, 0)
        s = model.scores(xte)
        assert s.shape == (len(xte),)
        assert auc(s, yte) > 0.9, name
        assert s[yte == 1].mean() > s[yte == 0].mean(), name
        assert accuracy(s, yte) > 0.8, name


def test_logreg_deterministic():
    xtr, ytr, xte, _ = linear_toy(seed=1)
    a = fit_logreg(xtr, ytr).scores(xte)
    b = fit_logreg(xtr, ytr).scores(xte)
    assert np.array_equal(a, b)

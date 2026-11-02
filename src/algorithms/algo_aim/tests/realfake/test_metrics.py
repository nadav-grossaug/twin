"""Tests for the metrics against hand-computed values and sklearn as an independent reference."""

import numpy as np
import pytest

from realfake.metrics import accuracy, auc, auc_by_bucket, bootstrap_auc_ci


def test_auc_hand_computed():
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([1.0, 2.0, 3.0, 4.0]), y) == 1.0
    assert auc(np.array([4.0, 3.0, 2.0, 1.0]), y) == 0.0
    assert auc(np.array([1.0, 1.0, 1.0, 1.0]), y) == 0.5  # all ties
    assert auc(np.array([1.0, 3.0, 2.0, 4.0]), y) == 0.75
    assert np.isnan(auc(np.array([1.0, 2.0]), np.array([1, 1])))  # one class


def test_auc_matches_sklearn_reference():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    scores = rng.normal(size=300) + y  # some signal, plus ties
    scores[:50] = np.round(scores[:50])
    assert auc(scores, y) == pytest.approx(sk.roc_auc_score(y, scores), abs=1e-12)


def test_accuracy_threshold_is_log_odds_zero():
    y = np.array([0, 0, 1, 1])
    assert accuracy(np.array([-1.0, -2.0, 1.0, 2.0]), y) == 1.0
    assert accuracy(np.array([1.0, 2.0, -1.0, -2.0]), y) == 0.0
    assert accuracy(np.array([-1.0, 2.0, 1.0, -2.0]), y) == 0.5


def test_bootstrap_ci_covers_true_auc_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 200)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1, 1, 200)])
    point = auc(scores, y)
    lo, hi = bootstrap_auc_ci(scores, y, n_boot=500, seed=0)
    true_auc = 0.7602  # P(N(1,1) > N(0,1)) = Phi(1/sqrt(2))
    assert lo < point < hi
    assert lo <= true_auc <= hi
    assert 0.0 < hi - lo < 0.15
    assert (lo, hi) == bootstrap_auc_ci(scores, y, n_boot=500, seed=0)


def test_auc_by_bucket():
    # bucket [0, 5): perfectly separated; bucket [5, 10]: inverted
    scores = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    values = np.array([0, 0, 0, 0, 10, 10, 10, 10])
    df = auc_by_bucket(scores, y, values, edges=np.array([0.0, 5.0, 10.0]))
    assert list(df["n"]) == [4, 4]
    assert list(df["n_fake"]) == [2, 2]
    assert list(df["auc"]) == [1.0, 0.0]
    assert list(df["lo"]) == [0.0, 5.0] and list(df["hi"]) == [5.0, 10.0]
    # out-of-range rows dropped, empty buckets give NaN
    df2 = auc_by_bucket(scores, y, values + 100, edges=np.array([0.0, 5.0, 10.0]))
    assert list(df2["n"]) == [0, 0] and df2["auc"].isna().all()

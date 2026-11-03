"""The classifiers (plan section 2): numpy logistic regression, optional sklearn gradient boosting.

Logistic regression is the dependency-free baseline (same numpy/scipy L-BFGS approach as
``graph_aim.eval.utility``); gradient boosting is the stronger model and only registered when
sklearn imports, so light environments fall back silently. Every model exposes the same interface:
``scores(x) -> np.ndarray`` on a log-odds scale, higher = more likely fake (``FAKE`` = 1), so a
threshold of 0.0 is the p = 0.5 decision and downstream code is model-agnostic. ``MODELS`` maps a
model name to a factory ``(x, y, seed) -> model``; report the strongest model present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

try:  # optional: gradient boosting only when sklearn is installed
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # pragma: no cover - depends on the environment
    HistGradientBoostingClassifier = None


@dataclass
class LogRegModel:
    """L2-regularised logistic regression on standardised features."""

    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Log-odds of the fake label; higher = more fake."""
        return ((np.asarray(x, dtype=np.float64) - self.mu) / self.sd) @ self.w + self.b


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> LogRegModel:
    """Standardise, then minimise the mean logistic loss + ``0.5 * l2 * |w|^2`` with L-BFGS."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    d = z.shape[1]

    def loss(wb):
        w, b = wb[:d], wb[d]
        s = z @ w + b
        ll = np.logaddexp(0.0, s) - y * s  # log(1 + exp(s)) - y s, stable
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
        r = p - y
        grad = np.concatenate([z.T @ r / len(y) + l2 * w, [r.mean()]])
        return ll.mean() + 0.5 * l2 * float(w @ w), grad

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return LogRegModel(w=res.x[:d], b=float(res.x[d]), mu=mu, sd=sd)


class GbmModel:
    """Wraps a fitted sklearn gradient-boosting classifier behind the uniform interface."""

    def __init__(self, clf, pos_col: int):
        self._clf = clf
        self._pos_col = pos_col

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Log-odds of the fake label; higher = more fake."""
        p = np.clip(self._clf.predict_proba(np.asarray(x, dtype=np.float64))[:, self._pos_col], 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))


def fit_gbm(x: np.ndarray, y: np.ndarray, seed: int = 0) -> GbmModel:
    """Histogram gradient boosting (sklearn). Raises ``ImportError`` when sklearn is missing."""
    if HistGradientBoostingClassifier is None:
        raise ImportError("scikit-learn is not installed; use the 'logreg' model")
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
    classes = list(clf.classes_)
    return GbmModel(clf, classes.index(1) if 1 in classes else len(classes) - 1)


# name -> factory (x, y, seed) -> model with scores(x); "logreg" is always present
MODELS = {"logreg": lambda x, y, seed=0: fit_logreg(x, y)}
if HistGradientBoostingClassifier is not None:
    MODELS["gbm"] = lambda x, y, seed=0: fit_gbm(x, y, seed)

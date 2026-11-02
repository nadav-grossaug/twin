"""Discrimination metrics (plan section 2): AUC, accuracy, bootstrap CI, AUC by meta bucket.

AUC is the headline number of the two-sample test (0.5 = this model cannot tell real from fake,
1.0 = trivially detectable), computed rank-based (Mann-Whitney) so it only needs score order.
The bootstrap CI resamples rows, so it understates the uncertainty when samples overlap - which is
exactly what the group-aware split and the real-vs-real control are for. ``auc_by_bucket`` slices
the AUC by a per-sample meta value (for example sample size), to show where the classifier wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from realfake.types import FAKE


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC of ``scores`` for the fake label (``FAKE`` = positive). NaN when one
    class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(y) == FAKE
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy(scores: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of rows classified correctly at ``threshold`` on the score scale. Model scores are
    log-odds (``models.py``), so the default 0.0 is the p = 0.5 decision."""
    pred = np.asarray(scores, dtype=np.float64) > threshold
    return float((pred == (np.asarray(y) == FAKE)).mean())


def bootstrap_auc_ci(
    scores: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap ``(lo, hi)`` of the AUC at level ``1 - alpha``, resampling rows with
    replacement (resamples that lose a class are dropped)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = auc(scores[idx], y[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def auc_by_bucket(scores: np.ndarray, y: np.ndarray, values: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    """Per-bucket AUC: rows are binned by ``values`` into ``[edges[i], edges[i+1])`` (the last
    bucket closed on the right); rows outside ``[edges[0], edges[-1]]`` are dropped. Returns a
    DataFrame with ``lo``, ``hi``, ``n``, ``n_fake``, ``auc`` per bucket, empty buckets included
    (``auc`` NaN)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n_buckets = len(edges) - 1
    bucket = np.searchsorted(edges, values, side="right") - 1
    bucket[values == edges[-1]] = n_buckets - 1  # close the last bucket on the right
    rows = []
    for i in range(n_buckets):
        m = bucket == i
        rows.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n": int(m.sum()),
                "n_fake": int((np.asarray(y)[m] == FAKE).sum()),
                "auc": auc(scores[m], y[m]) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)

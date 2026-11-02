"""Grouped permutation importance (plan section 2): what gave the fake away.

When the classifier beats 0.5, the per-group importance points at the statistic the generator got
wrong (and flags giveaway features that are artifacts of generation rather than statistics under
test - see ``docs/realfake/snap.md`` section 2). Features are permuted jointly per group, the same
row permutation for every column in the group, so correlated features are knocked out together and
their shared signal is not hidden by a surviving partner column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realfake.metrics import auc


def permutation_importance(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    seed: int = 0,
    n_rep: int = 5,
) -> pd.DataFrame:
    """Per feature group: baseline test AUC minus the AUC after jointly permuting the group's
    columns, averaged over ``n_rep`` permutations. Returns a DataFrame with ``group``,
    ``auc_drop``, ``auc_drop_std``, ``n_features`` and ``auc_baseline``, sorted by ``auc_drop``
    descending. ``model`` is anything with ``scores(x)`` (see ``models.MODELS``)."""
    x_test = np.asarray(x_test, dtype=np.float64)
    base = auc(model.scores(x_test), y_test)
    col = {name: i for i, name in enumerate(feature_names)}
    rng = np.random.default_rng(seed)
    rows = []
    for group, feats in feature_groups.items():
        cols = [col[f] for f in feats]
        drops = np.empty(n_rep)
        for r in range(n_rep):
            perm = rng.permutation(len(x_test))
            xp = x_test.copy()
            xp[:, cols] = x_test[np.ix_(perm, cols)]
            drops[r] = base - auc(model.scores(xp), y_test)
        rows.append(
            {
                "group": group,
                "auc_drop": float(drops.mean()),
                "auc_drop_std": float(drops.std()),
                "n_features": len(cols),
                "auc_baseline": base,
            }
        )
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False, ignore_index=True)

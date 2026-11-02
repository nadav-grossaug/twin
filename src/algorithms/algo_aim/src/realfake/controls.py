"""Honesty controls (plan section 4): a discriminator AUC means nothing without them.

``real_vs_real`` is the floor: a classifier trained to tell one half of the real data from the
other must sit at AUC about 0.5; anything above means the pipeline itself leaks (overlapping
samples, a featurizer that sees ids) and no real-vs-fake number can be trusted.
``label_permutation`` is the same check on the modelling side: shuffled training labels must give
test AUC about 0.5, otherwise the split leaks or the metric is broken.
"""

from __future__ import annotations

import numpy as np

from realfake.metrics import auc
from realfake.models import MODELS
from realfake.split import group_split
from realfake.types import FAKE, REAL, Dataset


def real_vs_real(ds: Dataset, model_name: str = "logreg", seed: int = 0, test_frac: float = 0.25) -> float:
    """Test AUC of ``model_name`` trained to tell one half of the real groups from the other.

    ``ds`` must be built from the real source alone (all labels ``REAL``, for example
    ``build_dataset(real_source, [], ...)``). Its groups are shuffled (deterministically, sorted
    then permuted with ``seed``) and the first half pseudo-labelled ``FAKE``; rows are then
    group-split with ``test_frac`` stratified by the pseudo-label. Should be about 0.5 on a sound
    pipeline."""
    if (np.asarray(ds.y) != REAL).any():
        raise ValueError("real_vs_real expects a dataset built from the real source only")
    groups = np.asarray(ds.groups).astype(str)
    uniq = np.unique(groups)
    perm = np.random.default_rng([seed, 0]).permutation(len(uniq))
    fake_half = set(uniq[perm[: len(uniq) // 2]])
    y = np.array([FAKE if g in fake_half else REAL for g in groups], dtype=np.int64)
    train, test = group_split(ds.groups, test_frac, seed, y=y)
    model = MODELS[model_name](ds.x[train], y[train], seed)
    return auc(model.scores(ds.x[test]), y[test])


def label_permutation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "logreg",
    seed: int = 0,
) -> float:
    """Test AUC of ``model_name`` trained on permuted training labels. Should be about 0.5."""
    y_perm = np.random.default_rng(seed).permutation(np.asarray(y_train))
    model = MODELS[model_name](x_train, y_perm, seed)
    return auc(model.scores(x_test), y_test)

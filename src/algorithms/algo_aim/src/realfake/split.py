"""Group-aware train/test splits (plan section 4).

Samples that share a group id (SNAP: walks from the same start node) overlap, so a plain row split
leaks test information into training and inflates the AUC. Every split here assigns whole groups to
one side; the real-vs-real control (``controls.py``) is the check that nothing still leaks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from realfake.types import FAKE, Dataset


def group_split(
    groups: np.ndarray, test_frac: float = 0.25, seed: int = 0, y: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(train_mask, test_mask)`` over the rows: a partition in which every group is
    wholly on one side and the test side holds roughly ``test_frac`` of the rows.

    When ``y`` is given the split is stratified: each group is binned by the majority label of its
    rows and the test fraction is met per label, so the test set stays roughly label-balanced.
    Deterministic for a fixed ``seed`` (groups are sorted before shuffling)."""
    groups = np.asarray(groups).astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    if y is None:
        glabel = np.zeros(len(uniq), dtype=np.int64)
    else:
        y = np.asarray(y, dtype=np.int64)
        glabel = np.array([np.bincount(y[inv == g]).argmax() for g in range(len(uniq))])

    rng = np.random.default_rng(seed)
    in_test = np.zeros(len(uniq), dtype=bool)
    for lab in np.unique(glabel):
        idx = np.flatnonzero(glabel == lab)
        order = idx[rng.permutation(len(idx))]
        target = test_frac * sizes[idx].sum()
        got = 0
        for g in order:
            if got >= target:
                break
            in_test[g] = True
            got += sizes[g]
        if len(idx) > 1 and in_test[idx].all():  # keep at least one group on the train side
            in_test[order[-1]] = False
    test = in_test[inv]
    return ~test, test


def split_with_held_out_sources(
    ds: Dataset, test_sources: Iterable[str], test_frac: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(train_mask, test_mask)`` implementing "train on fake seeds 0 and 1, test on
    seed 2": fake rows whose ``meta["source"]`` is in ``test_sources`` go entirely to test, the
    remaining fake rows entirely to train, and the real rows are group-split with ``test_frac``
    (so both sides keep real rows, with no real group on both sides)."""
    held_out = set(test_sources)
    src = np.array([m.get("source", "") for m in ds.meta], dtype=object)
    held = np.array([s in held_out for s in src])
    fake = ds.y == FAKE

    train = (fake & ~held).copy()
    test = (fake & held).copy()
    real_idx = np.flatnonzero(~fake)
    if len(real_idx):
        r_train, r_test = group_split(ds.groups[real_idx], test_frac, seed)
        train[real_idx[r_train]] = True
        test[real_idx[r_test]] = True
    return train, test

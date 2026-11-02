"""Tests for the honesty controls: ~0.5 on an indistinguishable toy, high AUC on a separable one."""

import numpy as np

from realfake.controls import label_permutation, real_vs_real
from realfake.metrics import auc
from realfake.models import MODELS
from realfake.split import group_split


def test_real_vs_real_control_is_near_half(toys):
    ds = toys.make_dataset(shift=0.0, n=400, d=4, seed=0, fake_names=())  # real source only
    assert (ds.y == 0).all()
    a = real_vs_real(ds, "logreg", seed=0)
    assert abs(a - 0.5) < 0.1


def test_label_permutation_control_is_near_half(toys):
    ds = toys.make_dataset(shift=0.0, n=300, d=4, seed=1)  # both labels from the same distribution
    train, test = group_split(ds.groups, test_frac=0.3, seed=0, y=ds.y)
    a = label_permutation(ds.x[train], ds.y[train], ds.x[test], ds.y[test], "logreg", seed=0)
    assert abs(a - 0.5) < 0.1


def test_main_classifier_near_half_on_non_separable_toy(toys):
    ds = toys.make_dataset(shift=0.0, n=300, d=4, seed=2)
    train, test = group_split(ds.groups, test_frac=0.3, seed=0, y=ds.y)
    model = MODELS["logreg"](ds.x[train], ds.y[train], 0)
    assert abs(auc(model.scores(ds.x[test]), ds.y[test]) - 0.5) < 0.1


def test_main_classifier_detects_separable_toy(toys):
    ds = toys.make_dataset(shift=1.5, n=300, d=4, seed=3)
    train, test = group_split(ds.groups, test_frac=0.3, seed=0, y=ds.y)
    model = MODELS["logreg"](ds.x[train], ds.y[train], 0)
    assert auc(model.scores(ds.x[test]), ds.y[test]) > 0.9


def test_controls_deterministic(toys):
    ds = toys.make_dataset(shift=0.0, n=200, d=4, seed=4, fake_names=())
    assert real_vs_real(ds, "logreg", seed=5) == real_vs_real(ds, "logreg", seed=5)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(100, 3)), rng.integers(0, 2, 100)
    assert label_permutation(x, y, x, y, seed=1) == label_permutation(x, y, x, y, seed=1)

"""Tests for grouped permutation importance: the informative group stands out."""

import numpy as np

from realfake.importance import permutation_importance
from realfake.models import fit_logreg
from realfake.split import group_split


def test_informative_group_has_high_importance(toys):
    # signal only in group "a" (first half of the columns); group "b" is pure noise
    ds = toys.make_dataset(n=400, d=4, seed=0, mu_fake=[1.5, 1.5, 0.0, 0.0])
    train, test = group_split(ds.groups, test_frac=0.3, seed=0, y=ds.y)
    model = fit_logreg(ds.x[train], ds.y[train])
    df = permutation_importance(
        model, ds.x[test], ds.y[test], ds.feature_names, ds.feature_groups, seed=0, n_rep=5
    )
    imp = dict(zip(df["group"], df["auc_drop"]))
    assert set(imp) == {"a", "b"}
    assert df["group"].iloc[0] == "a"  # sorted, informative group first
    assert imp["a"] > 0.2
    assert abs(imp["b"]) < 0.05
    assert (df["auc_baseline"] > 0.9).all()
    assert list(df["n_features"]) == [2, 2]


def test_importance_deterministic(toys):
    ds = toys.make_dataset(n=200, d=4, seed=1, shift=1.0)
    model = fit_logreg(ds.x, ds.y)
    kw = dict(seed=3, n_rep=3)
    a = permutation_importance(model, ds.x, ds.y, ds.feature_names, ds.feature_groups, **kw)
    b = permutation_importance(model, ds.x, ds.y, ds.feature_names, ds.feature_groups, **kw)
    assert a.equals(b)

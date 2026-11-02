"""Tests for build_dataset and the group-aware splits on toy Gaussian data."""

import numpy as np

from realfake.split import group_split, split_with_held_out_sources
from realfake.types import FAKE, REAL


# ---------------------------------------------------------------------------------- build_dataset
def test_build_dataset_shapes_labels_and_meta(toys):
    ds = toys.make_dataset(n=30, d=4, fake_names=("synA", "synB"))
    assert ds.x.shape == (60, 4)
    assert (ds.y[:30] == REAL).all() and (ds.y[30:] == FAKE).all()
    assert ds.feature_names == [f"f{i}" for i in range(4)]
    assert set(ds.feature_groups) == {"a", "b"}
    assert all(m["size"] == 4.0 for m in ds.meta)  # sampler meta kept
    assert [m["source"] for m in ds.meta[:30]] == ["real"] * 30


def test_build_dataset_namespaces_groups_per_source(toys):
    ds = toys.make_dataset(n=20, fake_names=("synA",), n_groups=5)
    real_g = set(ds.groups[ds.y == REAL])
    fake_g = set(ds.groups[ds.y == FAKE])
    assert all(g.startswith("real:") for g in real_g)
    assert all(g.startswith("synA:") for g in fake_g)
    assert not real_g & fake_g  # same raw ids (0..4), still distinct groups


def test_build_dataset_even_split_across_fake_sources(toys):
    ds = toys.make_dataset(n=10, fake_names=("s0", "s1", "s2"))
    counts = {name: sum(m["source"] == name for m in ds.meta) for name in ("s0", "s1", "s2")}
    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_build_dataset_deterministic_and_real_stream_stable(toys):
    a = toys.make_dataset(n=25, seed=7, fake_names=("synA",))
    b = toys.make_dataset(n=25, seed=7, fake_names=("synA",))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.groups, b.groups)
    assert np.array_equal(a.y, b.y)
    # adding a fake source changes neither the real rows nor their groups
    c = toys.make_dataset(n=25, seed=7, fake_names=("synA", "synB"))
    assert np.array_equal(a.x[:25], c.x[:25])
    assert np.array_equal(a.groups[:25], c.groups[:25])
    assert toys.make_dataset(n=25, seed=8, fake_names=("synA",)).x[0, 0] != a.x[0, 0]


# ------------------------------------------------------------------------------------ group_split
def test_group_split_no_group_on_both_sides():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 30, size=200).astype(str)
    train, test = group_split(groups, test_frac=0.25, seed=0)
    assert np.array_equal(train, ~test) and test.any() and train.any()
    assert not set(groups[train]) & set(groups[test])
    assert 0.1 < test.mean() < 0.45  # groups are lumpy, so only roughly test_frac


def test_group_split_stratified_by_label():
    rng = np.random.default_rng(1)
    groups = np.concatenate([rng.integers(0, 40, 200), 100 + rng.integers(0, 40, 200)]).astype(str)
    y = np.array([0] * 200 + [1] * 200)
    train, test = group_split(groups, test_frac=0.25, seed=0, y=y)
    assert not set(groups[train]) & set(groups[test])
    for lab in (0, 1):
        frac = test[y == lab].mean()
        assert 0.15 < frac < 0.4


def test_group_split_deterministic():
    groups = (np.arange(100) % 17).astype(str)
    a = group_split(groups, 0.3, seed=3)
    b = group_split(groups, 0.3, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = group_split(groups, 0.3, seed=4)
    assert not np.array_equal(a[1], c[1])


# -------------------------------------------------------------------- split_with_held_out_sources
def test_held_out_fake_source_goes_entirely_to_test(toys):
    ds = toys.make_dataset(n=60, fake_names=("seed0", "seed1", "seed2"))
    train, test = split_with_held_out_sources(ds, ["seed2"], test_frac=0.25, seed=0)
    src = np.array([m["source"] for m in ds.meta])
    assert np.array_equal(train, ~test)
    assert test[src == "seed2"].all()
    assert train[(src == "seed0") | (src == "seed1")].all()
    # real rows on both sides, no real group split across them
    real = ds.y == REAL
    assert train[real].any() and test[real].any()
    assert not set(ds.groups[train]) & set(ds.groups[test])
    frac = test[real].mean()
    assert 0.1 < frac < 0.45

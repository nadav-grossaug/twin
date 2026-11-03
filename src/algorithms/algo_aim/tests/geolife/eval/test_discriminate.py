"""Smoke test of the stage-3 discriminator core (``src/geolife/eval/discriminate.py``).

Everything runs on tiny in-process random-walk traces through ``discriminate.run`` (the testable
core that ``main()`` wraps with file loading), so no real data is read and the test stays offline
and fast. Each toy method has a single fake source, which exercises exactly the fold mode the
committed single synthetic variant uses (stratified group split per fold).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from geolife.eval.discriminate import run
from geolife.eval.discriminator import TraceFeaturizer, WindowSampler
from tests.geolife.eval.test_discriminator import toy_arrays

EXPECTED_ROW_COLS = {
    "method", "epsilon", "seed", "model", "auc", "auc_lo", "auc_hi",
    "accuracy", "n_train", "n_test", "permuted_auc", "real_vs_real_auc",
}


def test_run_writes_tables_with_sane_values(tmp_path):
    real = toy_arrays(n_users=12, n_pts=500, speed=1.5, seed=0)
    entries = [
        # same generator, other seed: should be hard to tell apart
        ("same_speed", math.inf, {"seed0": toy_arrays(n_users=12, n_pts=500, speed=1.5, seed=100)}),
        # three times the speed: trivially detectable
        ("fast", float("nan"), {"seed0": toy_arrays(n_users=12, n_pts=500, speed=4.5, seed=200)}),
    ]
    feat = TraceFeaturizer(include_map=False)
    run(
        real,
        entries,
        out=tmp_path,
        n=60,
        n_boot=50,
        folds=2,
        sampler=WindowSampler(k_min=10, k_max=40),
        featurizer=feat,
        size_edges=(10, 20, 41),
    )

    for name in ("discriminator", "discriminator_summary", "discriminator_importance", "discriminator_by_size"):
        assert (tmp_path / f"{name}.csv").exists(), name

    df = pd.read_csv(tmp_path / "discriminator.csv")
    assert EXPECTED_ROW_COLS <= set(df.columns)
    assert not df["auc"].isna().any()
    assert df["auc"].between(0.0, 1.0).all()
    assert df["real_vs_real_auc"].between(0.0, 1.0).all()

    # the real-vs-real control appears as its own method, once per model
    rvr = df[df["method"] == "real_vs_real"]
    assert len(rvr) == len(df["model"].unique()) and rvr["epsilon"].isna().all()

    # single fake source -> `folds` stratified group-split folds per method
    per = df[df["method"] != "real_vs_real"]
    assert set(per["method"]) == {"same_speed", "fast"}
    assert set(per["seed"]) == {0, 1}
    assert (per["n_train"] > 0).all() and (per["n_test"] > 0).all()
    assert per["permuted_auc"].between(0.0, 1.0).all()
    # the faster generator is easy to detect for the best model
    assert per[per["method"] == "fast"]["auc"].max() > 0.8

    summary = pd.read_csv(tmp_path / "discriminator_summary.csv")
    assert {"method", "epsilon", "model", "auc_mean", "auc_std"} <= set(summary.columns)
    assert "real_vs_real" in set(summary["method"])

    imp = pd.read_csv(tmp_path / "discriminator_importance.csv")
    assert {"method", "epsilon", "model", "group", "auc_drop", "auc_drop_std", "auc_baseline"} <= set(imp.columns)
    assert set(imp["group"]) == set(feat.feature_groups)
    assert imp["auc_baseline"].between(0.0, 1.0).all()
    assert set(imp["method"]) == {"same_speed", "fast"}

    by_size = pd.read_csv(tmp_path / "discriminator_by_size.csv")
    assert {"method", "epsilon", "model", "lo", "hi", "n", "auc"} <= set(by_size.columns)
    assert by_size["auc"].dropna().between(0.0, 1.0).all()
    assert (by_size.groupby("method")["n"].sum() > 0).all()


def test_run_rejects_unknown_model(tmp_path):
    try:
        run(toy_arrays(n_users=3, n_pts=100), [], out=tmp_path, n=5, models=["nope"])
    except ValueError as e:
        assert "unknown models" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown model name")

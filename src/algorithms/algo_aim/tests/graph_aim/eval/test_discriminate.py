"""Smoke test of the stage-3 discriminator CLI (``snap/eval/discriminate.py``).

Everything runs on tiny in-process graphs through ``discriminate.run`` (the testable core that
``main()`` wraps with file loading), so no real data is read and the test stays offline and fast.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from discriminate import run, selected
from graph_aim.eval.discriminator import WalkSampler
from graph_aim.transfer.types import AttributedGraph

DOMAINS = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}


def er_graph(n: int = 150, p: float = 0.05, seed: int = 0) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in DOMAINS.items()})
    return AttributedGraph(n, edges, attrs, dict(DOMAINS))


EXPECTED_ROW_COLS = {
    "method", "epsilon", "seed", "model", "auc", "auc_lo", "auc_hi",
    "accuracy", "n_train", "n_test", "permuted_auc", "real_vs_real_auc",
}


def test_run_writes_tables_with_sane_values(tmp_path):
    real = er_graph(seed=0)
    entries = [
        # same distribution as the real graph: should be hard to tell apart
        ("same_er", math.inf, {"seed0": er_graph(seed=10), "seed1": er_graph(seed=11)}),
        # three times the density: trivially detectable
        ("dense_er", float("nan"), {"seed0": er_graph(p=0.15, seed=20), "seed1": er_graph(p=0.15, seed=21)}),
    ]
    run(
        real,
        entries,
        homophily_attrs=["a"],
        out=tmp_path,
        n=60,
        n_boot=50,
        sampler=WalkSampler(k_min=5, k_max=30),
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

    # one row per method, fold, model for the two fake methods (2 folds: one per fake source)
    per = df[df["method"] != "real_vs_real"]
    assert set(per["method"]) == {"same_er", "dense_er"}
    assert set(per["seed"]) == {0, 1}
    assert (per["n_train"] > 0).all() and (per["n_test"] > 0).all()
    assert per["permuted_auc"].between(0.0, 1.0).all()
    # the denser graph is easy to detect for the best model
    assert per[per["method"] == "dense_er"]["auc"].max() > 0.8

    summary = pd.read_csv(tmp_path / "discriminator_summary.csv")
    assert {"method", "epsilon", "model", "auc_mean", "auc_std"} <= set(summary.columns)
    assert "real_vs_real" in set(summary["method"])

    imp = pd.read_csv(tmp_path / "discriminator_importance.csv")
    assert {"method", "epsilon", "model", "group", "auc_drop", "auc_drop_std", "auc_baseline"} <= set(imp.columns)
    assert set(imp["group"]) == {"size", "degree", "structure", "attributes"}
    assert imp["auc_baseline"].between(0.0, 1.0).all()
    assert set(imp["method"]) == {"same_er", "dense_er"}  # one importance table per method

    by_size = pd.read_csv(tmp_path / "discriminator_by_size.csv")
    assert {"method", "epsilon", "model", "lo", "hi", "n", "auc"} <= set(by_size.columns)
    assert by_size["auc"].dropna().between(0.0, 1.0).all()
    assert (by_size.groupby("method")["n"].sum() > 0).all()


def test_selected_mirrors_score():
    assert selected(None, "anything")
    assert selected(["graph_aim_eps4"], "graph_aim_eps4", "graph_aim")
    assert selected(["graph_aim"], "graph_aim_eps4", "graph_aim")
    assert not selected(["baselines"], "graph_aim_eps4", "graph_aim")
    assert selected(["baselines"], "bter", "baselines")
    assert selected(["bter"], "bter", "baselines")
    assert not selected(["erdos_renyi"], "bter", "baselines")

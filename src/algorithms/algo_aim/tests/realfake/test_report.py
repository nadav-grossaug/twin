"""Tests for the result tables: csv/json layout and the mean/std summary over seeds."""

import json

import pandas as pd

from realfake.report import summarize, write_results

ROWS = [
    {"method": "graph_aim", "epsilon": 4.0, "seed": 0, "model": "logreg", "auc": 0.6},
    {"method": "graph_aim", "epsilon": 4.0, "seed": 1, "model": "logreg", "auc": 0.7},
    {"method": "erdos_renyi", "epsilon": float("inf"), "seed": 0, "model": "logreg", "auc": 0.99},
]


def test_write_results_csv_and_json(tmp_path):
    df = write_results(ROWS, tmp_path, name="discriminator", summary_keys=["method", "epsilon", "model"])
    back = pd.read_csv(tmp_path / "discriminator.csv")
    assert list(back.columns) == ["method", "epsilon", "seed", "model", "auc"]
    assert len(back) == 3 and len(df) == 3
    assert json.loads((tmp_path / "discriminator.json").read_text())[0]["method"] == "graph_aim"
    summ = pd.read_csv(tmp_path / "discriminator_summary.csv")
    assert len(summ) == 2
    row = summ[summ["method"] == "graph_aim"].iloc[0]
    assert row["n_seeds"] == 2
    assert abs(row["auc_mean"] - 0.65) < 1e-12
    assert abs(row["auc_std"] - pd.Series([0.6, 0.7]).std()) < 1e-12


def test_summarize_skips_keys_and_seed():
    df = pd.DataFrame(ROWS)
    summ = summarize(df, ["method", "epsilon", "model"])
    assert "seed_mean" not in summ.columns
    assert {"auc_mean", "auc_std", "n_seeds"} <= set(summ.columns)


def test_write_results_empty_writes_nothing(tmp_path):
    df = write_results([], tmp_path / "sub", name="x")
    assert df.empty and not (tmp_path / "sub").exists()

"""Shared fixtures: a tiny offline dataset spec and splits (no Adult file needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth_bench.datasets.base import DatasetSpec

CODES = {"red": 1, "green": 2, "blue": 3, "rare": 4}


def make_df(n: int, seed: int) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    color = rs.choice(list(CODES), n, p=[0.5, 0.3, 0.17, 0.03])
    x = rs.normal(size=n) + (color == "red")
    z = rs.exponential(size=n).round(2)
    y = ((x + rs.normal(scale=0.5, size=n)) > 0.5).astype(int)
    code = pd.Series(color).map(CODES).to_numpy()
    return pd.DataFrame({"x": x, "z": z, "color": color, "code": code, "y": y})


@pytest.fixture(scope="session")
def spec() -> DatasetSpec:
    return DatasetSpec(
        name="toy",
        dir=Path("toy"),
        target="y",
        task_type="classification",
        categorical=["color"],
        numerical=["x", "z", "code"],
        sensitive_features=["color"],
        loader=lambda: make_df(1500, 0),
        constraints={
            "code_mapping": lambda d: d["color"].map(CODES) != d["code"],
            "z_negative": lambda d: d["z"] < 0,
        },
    )


@pytest.fixture(scope="session")
def splits():
    return make_df(1000, 1), make_df(500, 2)

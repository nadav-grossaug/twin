"""Dataset specification shared by all registered datasets."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

Constraint = Callable[[pd.DataFrame], pd.Series]
"""Maps a DataFrame to a boolean Series that is True for rows violating a rule."""


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the benchmark needs to know about a tabular dataset.

    Attributes:
        name: Registry key, also the dataset folder name under the repo root.
        dir: Dataset folder (holds ``data/``, ``results/``, ...).
        target: Target column (used by utility metrics and conditional generators).
        task_type: Synthcity task type; only ``"classification"`` is used so far.
        categorical: Categorical feature columns (the target is listed separately).
        numerical: Numerical feature columns.
        sensitive_features: Columns Synthcity's privacy metrics treat as the
            sensitive attributes. k-anonymity/l-diversity cluster rows on all
            *other* columns (quasi-identifiers) and count sensitive values per cluster.
        loader: Function returning the cleaned, typed full DataFrame.
        constraints: Domain rules that (almost) always hold in the real data, as
            ``name -> fn(df) -> boolean Series`` that is True where a row *violates*
            the rule. The evaluation reports the violation share per rule.
        method_overrides: Per-method param overrides merged over the defaults.
    """

    name: str
    dir: Path
    target: str
    task_type: str
    categorical: list[str]
    numerical: list[str]
    sensitive_features: list[str]
    loader: Callable[[], pd.DataFrame]
    constraints: dict[str, Constraint] = field(default_factory=dict)
    method_overrides: dict[str, dict] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        """All declared columns: numerical, categorical, target."""
        return [*self.numerical, *self.categorical, self.target]

    @property
    def discrete_columns(self) -> list[str]:
        """Columns that generators must model as discrete (categoricals + classification target)."""
        target = [self.target] if self.task_type == "classification" else []
        return [*self.categorical, *target]

    def load(self) -> pd.DataFrame:
        """Load the cleaned dataset, keep the declared columns (file order) and validate it."""
        df = self.loader()
        missing = set(self.columns) - set(df.columns)
        if missing:
            raise ValueError(f"{self.name}: loader output lacks columns {sorted(missing)}")
        df = df[[c for c in df.columns if c in self.columns]]
        if df.isna().any().any():
            raise ValueError(f"{self.name}: loader output contains NaN")
        return df.reset_index(drop=True)

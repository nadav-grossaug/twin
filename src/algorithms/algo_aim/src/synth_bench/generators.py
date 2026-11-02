"""Fit a Synthcity plugin and sample from it."""

import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path
from unittest import mock

import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models import tabular_ddpm, tabular_encoder
from synthcity.plugins.core.models.tabular_arf import TabularARF
from synthcity.utils.dataframe import discrete_columns as synthcity_discrete_columns
from synthcity.utils.reproducibility import enable_reproducible_results

# Image plugins are stubbed out (no monai), so only these categories are loaded.
PLUGIN_CATEGORIES = ["generic", "privacy"]


@lru_cache(maxsize=1)
def plugins() -> Plugins:
    """Synthcity plugin registry (cached: loading it takes a few seconds)."""
    return Plugins(categories=PLUGIN_CATEGORIES)


def fit_generate(
    method: str,
    train_loader: DataLoader,
    params: dict,
    n_synth: int,
    seed: int,
    discrete_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, float, float]:
    """Fit ``method`` on ``train_loader`` and generate ``n_synth`` rows.

    ``discrete_columns`` are forced to be modelled as categorical (see
    ``force_discrete``). Returns ``(synthetic_df, fit_seconds, generate_seconds)``.
    """
    with tempfile.TemporaryDirectory() as workspace, force_discrete(discrete_columns):
        enable_reproducible_results(seed)
        model = plugins().get(method, random_state=seed, workspace=Path(workspace), **params)
        start = time.perf_counter()
        model.fit(train_loader)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        synth = model.generate(count=n_synth, random_state=seed).dataframe()
        generate_seconds = time.perf_counter() - start
    return synth.reset_index(drop=True), fit_seconds, generate_seconds


@contextmanager
def force_discrete(columns: Sequence[str]) -> Iterator[None]:
    """Make Synthcity treat ``columns`` as discrete, on top of its own heuristic.

    Synthcity label-encodes object columns before fitting, then decides
    "discrete vs continuous" purely by cardinality (<= 10 unique values) in
    ``TabularEncoder`` (GANs, VAEs, flows, Bayesian network), TabDDPM and ARF.
    Categoricals with more levels would be modelled as continuous label codes, so
    their levels get an arbitrary order and decoded samples are truncated floats.
    """
    forced = set(columns)

    def discrete(df: pd.DataFrame, max_classes: int = 10, return_counts: bool = False) -> list:
        found = synthcity_discrete_columns(df, max_classes, return_counts=True)
        low_cardinality = {col for col, _ in found}
        cols = [c for c in df.columns if c in forced or c in low_cardinality]
        return [(c, df[c].nunique()) for c in cols] if return_counts else cols

    def arf_categorical(_self: TabularARF, df: pd.DataFrame, var_threshold: int) -> list:
        return discrete(df, var_threshold)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(tabular_encoder, "find_cat_cols", discrete))
        stack.enter_context(mock.patch.object(tabular_ddpm, "discrete_columns", discrete))
        stack.enter_context(
            mock.patch.object(TabularARF, "get_categorical_cols", arf_categorical)
        )
        yield

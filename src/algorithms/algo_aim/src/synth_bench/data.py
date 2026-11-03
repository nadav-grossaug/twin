"""Deterministic train/test splits and Synthcity data loaders."""

import pandas as pd
from sklearn.model_selection import train_test_split
from synthcity.plugins.core.dataloader import GenericDataLoader

from synth_bench.config import N_TRAIN, SEED, TEST_SIZE
from synth_bench.datasets import DatasetSpec, get_dataset


def load_splits(
    dataset_name: str,
    n_train: int | None = N_TRAIN,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` for a dataset.

    The full data is split once (stratified on the target, ``test_size`` held out),
    then the train part is stratified-subsampled to ``n_train`` rows (``None`` keeps
    all of it). The test set is only ever used for evaluation.
    """
    spec = get_dataset(dataset_name)
    df = spec.load()
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=_stratify(df, spec)
    )
    if n_train is not None and n_train < len(train):
        train, _ = train_test_split(
            train, train_size=n_train, random_state=seed, stratify=_stratify(train, spec)
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def to_loader(df: pd.DataFrame, spec: DatasetSpec) -> GenericDataLoader:
    """Wrap a DataFrame in a Synthcity loader carrying target and sensitive features."""
    return GenericDataLoader(
        df,
        target_column=spec.target,
        sensitive_features=spec.sensitive_features,
        random_state=SEED,
    )


def _stratify(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series | None:
    """Stratification labels (the target for classification, none otherwise)."""
    return df[spec.target] if spec.task_type == "classification" else None

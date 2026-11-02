"""Global paths, seed and quick-budget defaults."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
N_TRAIN = 10_000  # rows used to fit each generator (stratified subsample of the train split)
TEST_SIZE = 0.2  # fraction of the full dataset held out for evaluation
N_SYNTH: int | None = None  # rows to generate; None means "same as n_train"


def dataset_dir(name: str) -> Path:
    """Root folder of a dataset (``<repo>/<name>``)."""
    return REPO_ROOT / name


def results_dir(name: str) -> Path:
    """Benchmark output folder of a dataset (``<repo>/<name>/results``)."""
    return dataset_dir(name) / "results"

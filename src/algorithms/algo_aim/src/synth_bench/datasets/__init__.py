"""Dataset registry: add a ``DatasetSpec`` to ``_REGISTRY`` to make it benchmarkable."""

from synth_bench.datasets.adult_census import ADULT_CENSUS
from synth_bench.datasets.base import DatasetSpec

_REGISTRY: dict[str, DatasetSpec] = {spec.name: spec for spec in [ADULT_CENSUS]}


def get_dataset(name: str) -> DatasetSpec:
    """Return the spec registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; available: {list_datasets()}")
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Names of all registered datasets."""
    return sorted(_REGISTRY)


__all__ = ["DatasetSpec", "get_dataset", "list_datasets"]

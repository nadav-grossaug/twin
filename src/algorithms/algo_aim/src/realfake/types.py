"""Shared contract of the real-vs-fake classifier (``docs/realfake/PLAN.md`` section 2).

A data type plugs in two things: a ``Sampler`` that cuts one unit (a sample) out of a real or a
synthetic source, and a ``Featurizer`` that turns a sample into a fixed-length numeric vector.
Everything else (dataset assembly, group-aware split, models, metrics, controls, importance,
report) is dataset-independent and lives in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, runtime_checkable

import numpy as np

REAL, FAKE = 0, 1


@dataclass
class Sample:
    """One unit cut from a source.

    ``payload`` is dataset-specific (SNAP: a relabelled walk subgraph) and is read only by the
    matching ``Featurizer``. ``label`` is ``REAL`` (0) or ``FAKE`` (1); a ``Sampler`` leaves it at
    ``-1`` and ``build_dataset`` sets it from which source the sample came. ``group_id`` ties
    together samples that must never be split across train and test (SNAP: the start node);
    ``build_dataset`` namespaces it per source, so equal ids from different sources stay distinct
    groups. ``meta`` holds per-sample numbers for reporting (for example ``{"size": k}``), never
    used as features.
    """

    payload: Any
    group_id: Hashable
    meta: dict[str, float] = field(default_factory=dict)
    label: int = -1


@runtime_checkable
class Sampler(Protocol):
    """Cut one sample from ``source``. Must use identical parameters for real and fake sources, so
    unit size and selection bias match by construction. Deterministic given ``rng``."""

    def __call__(self, source: Any, rng: np.random.Generator) -> Sample: ...


@runtime_checkable
class Featurizer(Protocol):
    """Sample -> fixed-length vector, invariant to representation details that differ between real
    and synthetic by construction (ids, ordering, column order). ``feature_names`` names every
    column of the vector, in order; ``feature_groups`` maps a group name to the feature names in it
    (a partition of ``feature_names``), used for grouped permutation importance."""

    feature_names: list[str]
    feature_groups: dict[str, list[str]]

    def __call__(self, sample: Sample) -> np.ndarray: ...


@dataclass
class Dataset:
    """What ``build_dataset`` returns: the design matrix and everything needed downstream.

    ``x`` is ``(n_samples, n_features)``; ``y`` is 0/1 (``REAL``/``FAKE``); ``groups`` are the
    namespaced group ids as strings, one per row; ``meta`` is a list of the samples' ``meta``
    dicts in row order. ``feature_names`` / ``feature_groups`` are copied from the featurizer.
    """

    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: list[dict[str, float]]
    feature_names: list[str]
    feature_groups: dict[str, list[str]]

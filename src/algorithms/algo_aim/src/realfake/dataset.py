"""Assemble a labelled design matrix from one real and several fake sources (plan section 2).

``build_dataset`` draws ``n`` samples from the real source and ``n`` in total from the fake
sources with one shared sampler and featurizer, so unit size and selection bias match between the
labels by construction. Each source gets its own rng stream derived from ``(seed, stream index)``,
so the real samples (and every already-listed fake source's samples) are unchanged when another
fake source is appended. Group ids are namespaced per source (``"real:<gid>"``, ``"<name>:<gid>"``)
so equal ids from different sources stay distinct groups in the group-aware split.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from realfake.types import FAKE, REAL, Dataset, Featurizer, Sampler


def build_dataset(
    real_source: Any,
    fake_sources: list[tuple[str, Any]],
    sampler: Sampler,
    featurizer: Featurizer,
    n: int,
    seed: int = 0,
) -> Dataset:
    """``n`` real samples (label ``REAL``) plus ``n`` fake samples (label ``FAKE``) split as evenly
    as possible across ``fake_sources`` (``(name, source)`` pairs; earlier sources get the
    remainder). Each sample's ``meta`` keeps what the sampler set and gains ``"source"`` (``"real"``
    or the fake source's name). Deterministic for a fixed ``seed``."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    gs: list[str] = []
    metas: list[dict] = []

    def draw(source: Any, name: str, label: int, count: int, stream: int) -> None:
        rng = np.random.default_rng([seed, stream])
        for _ in range(count):
            s = sampler(source, rng)
            s.label = label
            xs.append(np.asarray(featurizer(s), dtype=np.float64).ravel())
            ys.append(label)
            gs.append(f"{name}:{s.group_id}")
            metas.append({**s.meta, "source": name})

    draw(real_source, "real", REAL, n, 0)
    k = len(fake_sources)
    counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
    for i, ((name, source), count) in enumerate(zip(fake_sources, counts)):
        draw(source, name, FAKE, count, i + 1)

    x = np.vstack(xs) if xs else np.zeros((0, len(featurizer.feature_names)))
    return Dataset(
        x=x,
        y=np.asarray(ys, dtype=np.int64),
        groups=np.asarray(gs, dtype=object),
        meta=metas,
        feature_names=list(featurizer.feature_names),
        feature_groups={g: list(f) for g, f in featurizer.feature_groups.items()},
    )

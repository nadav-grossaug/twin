"""Shared toy sampler / featurizer over Gaussian vectors (offline; no graph_aim imports).

A source is ``{"mu": vector, "n_groups": int}``; the sampler draws ``N(mu, I)`` payloads with a
random group id, the featurizer is the identity with two feature groups ("a" = first half of the
columns, "b" = second half).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from realfake.dataset import build_dataset
from realfake.types import Sample


def gauss_sampler(source, rng):
    mu = np.asarray(source["mu"], dtype=np.float64)
    return Sample(
        payload=mu + rng.normal(size=len(mu)),
        group_id=int(rng.integers(0, source["n_groups"])),
        meta={"size": float(len(mu))},
    )


class IdentityFeaturizer:
    def __init__(self, d):
        self.feature_names = [f"f{i}" for i in range(d)]
        self.feature_groups = {"a": self.feature_names[: d // 2], "b": self.feature_names[d // 2 :]}

    def __call__(self, sample):
        return np.asarray(sample.payload, dtype=np.float64)


def make_dataset(shift=0.0, n=200, d=4, seed=0, n_groups=40, fake_names=("synA",), mu_fake=None):
    """Real source at N(0, I); each fake source at N(mu_fake, I) (default: ``shift`` in every
    coordinate). ``fake_names=()`` gives a real-only dataset."""
    real = {"mu": np.zeros(d), "n_groups": n_groups}
    mu = np.full(d, shift) if mu_fake is None else np.asarray(mu_fake, dtype=np.float64)
    fakes = [(name, {"mu": mu, "n_groups": n_groups}) for name in fake_names]
    return build_dataset(real, fakes, gauss_sampler, IdentityFeaturizer(d), n, seed)


@pytest.fixture
def toys():
    return SimpleNamespace(sampler=gauss_sampler, Featurizer=IdentityFeaturizer, make_dataset=make_dataset)

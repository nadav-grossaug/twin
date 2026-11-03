"""Shared data contract of graph-aim-walks (``docs/snap/walks-plan.md`` section 3).

``WalksArtifact`` = the exact (noise-free) graph-AIM base artifact + the public walk spec + the released walk tables.
It is NOT private: every table is an exact statistic of the real graph (up to walk-sampling error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graph_aim.transfer.types import Artifact
from graph_aim_walks.transfer.walkstats import WalkSpec

METHOD = "graph_aim_walks"
SPECTRUM = "transition_spectrum"


@dataclass
class WalksArtifact:
    """Everything the walks generator may see. No per-node data, no paths.

    ``tables``: the normalised walk tables of :func:`graph_aim_walks.transfer.walkstats.normalise` (``return``,
    ``coverage``, ``degree_lag``, ``attr_lag:<g>``, ``share:<a>``, ``dominance:<a>|<b>``) plus ``transition_spectrum``
    (top eigenvalues of ``D^-1 A``, descending). ``n_walks`` is the size of the walk pool the tables were estimated
    from (it sets their sampling variance).
    """

    base: Artifact
    spec: WalkSpec
    tables: dict[str, np.ndarray]
    n_walks: int
    provenance: dict = field(default_factory=dict)


def table_names(spec: WalkSpec) -> list[str]:
    """Names of the walk tables of ``spec`` in release order (``transition_spectrum`` last)."""
    names = ["return", "coverage", "degree_lag"]
    names += ["attr_lag:" + "+".join(g) for g in spec.groups]
    names += ["share:" + a for a in spec.share_attrs]
    names += [f"dominance:{a}|{b}" for a, b in spec.dominance_pairs]
    return names + [SPECTRUM]


def expected_shapes(spec: WalkSpec, domains: dict) -> dict[str, tuple[int, ...] | None]:
    """Shape of every walk table under ``spec`` and the schema ``domains`` (``None`` for the spectrum, whose length
    is at most the requested ``k`` and is checked separately). Raises ``KeyError`` for an attribute not in ``domains``."""
    T, L, W = spec.length, len(spec.lags), len(spec.windows)
    B = len(spec.degree_bin_edges) - 1
    out: dict[str, tuple[int, ...] | None] = {"return": (T,), "coverage": (T,), "degree_lag": (L, B, B)}
    for g in spec.groups:
        G = int(np.prod([len(domains[c]) for c in g]))
        out["attr_lag:" + "+".join(g)] = (L, G, G)
    for a in spec.share_attrs:
        out["share:" + a] = (W, len(domains[a]), spec.share_bins)
    for a, b in spec.dominance_pairs:
        len(domains[a]), len(domains[b])  # KeyError for an unknown attribute
        out[f"dominance:{a}|{b}"] = (W, spec.dominance_bins, spec.dominance_bins)
    out[SPECTRUM] = None
    return out

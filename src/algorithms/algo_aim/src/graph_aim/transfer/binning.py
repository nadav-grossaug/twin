"""Degree binning shared by fitting (inside), generation (outside) and evaluation (eval).

One definition of "which bin is this degree in", so the ``attr_degree`` tables, the generator that reads them and the
metrics that compare against them can never disagree. Bins are ``[e_0, e_1), ..., [e_{B-1}, e_B)`` with ``e_0 = 0``;
a degree at or above the last edge falls in the last bin (open-ended, same convention as ``clustering_by_degree`` and
``joint_degree``).
"""

from __future__ import annotations

import numpy as np


def check_bin_edges(bin_edges) -> np.ndarray:
    """Return the edges as an int64 array, or raise ``ValueError`` if they are not usable bin edges.

    Needs at least two edges, the first equal to 0, strictly increasing.
    """
    e = np.asarray(bin_edges)
    if e.ndim != 1 or len(e) < 2:
        raise ValueError("bin_edges needs at least two edges")
    if not np.issubdtype(e.dtype, np.integer):
        if not np.all(np.equal(np.mod(e, 1), 0)):
            raise ValueError("bin_edges must be integers")
    e = e.astype(np.int64)
    if e[0] != 0:
        raise ValueError("bin_edges must start at 0")
    if np.any(np.diff(e) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    return e


def degree_bin(deg, bin_edges) -> np.ndarray:
    """Bin index (``0..B-1``) of each degree; degrees at or above the last edge fall in the last bin."""
    e = check_bin_edges(bin_edges)
    d = np.asarray(deg)
    return np.minimum(np.searchsorted(e, d, side="right") - 1, len(e) - 2).astype(np.int64)

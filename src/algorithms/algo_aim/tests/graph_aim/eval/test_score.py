from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graph_aim.transfer.types import AttributedGraph
from score import align


def graph(cols: list[str], domains: dict) -> AttributedGraph:
    n = 6
    attrs = pd.DataFrame({c: np.arange(n) % len(domains[c]) for c in cols})
    return AttributedGraph(n, np.array([[0, 1], [1, 2]], dtype=np.int64), attrs, {c: domains[c] for c in cols})


DOMAINS = {"b": ["p", "q", "other", "none"], "a": ["x", "other", "none"]}


def test_align_reorders_columns_by_name_and_keeps_codes():
    real = graph(["b", "a"], DOMAINS)
    syn = graph(["a", "b"], DOMAINS)  # alphabetical, as it comes back from an artifact on disk
    out = align(syn, real)
    assert list(out.attrs.columns) == ["b", "a"] and list(out.domains) == ["b", "a"]
    assert (out.attrs["a"].to_numpy() == syn.attrs["a"].to_numpy()).all()


def test_coupling_columns_are_prefixed_and_policy_is_optional(tmp_path):
    from score import coupling_row, coupling_setup

    real = graph(["b", "a"], DOMAINS)
    kw, null = coupling_setup(real, tmp_path / "no_policy.json", exclude_hubs=1)  # missing policy file: defaults
    row = coupling_row(real, align(graph(["a", "b"], DOMAINS), real), kw, null)
    assert row and all(k.startswith("cpl_") for k in row)
    assert {"cpl_a_lift_real", "cpl_b_eta2_syn", "cpl_mean_lift_syn", "cpl_a_eta2_nohub_syn"} <= set(row)


def test_align_rejects_different_labels_or_columns():
    real = graph(["b", "a"], DOMAINS)
    with pytest.raises(ValueError):
        align(graph(["a", "b"], {**DOMAINS, "a": ["y", "other", "none"]}), real)
    with pytest.raises(ValueError):
        align(graph(["a"], DOMAINS), real)

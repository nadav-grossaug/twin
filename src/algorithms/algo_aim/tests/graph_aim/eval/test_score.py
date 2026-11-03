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


def test_composition_columns_are_prefixed_and_config_keys_are_optional():
    from score import composition_row, composition_setup

    real = graph(["b", "a"], DOMAINS)
    syn = align(graph(["a", "b"], DOMAINS), real)
    kw, nulls = composition_setup(real, {})  # no keys at all: every attribute on its own, public bins, min_degree 1
    assert kw["groups"] == [["b"], ["a"]] and kw["min_degree"] == 1 and kw["bin_edges"][0] == 0
    assert set(nulls) == {"gap", "gapdeg"}
    row = composition_row(real, syn, kw, nulls)
    assert row and all(k.startswith("cmp_") for k in row)
    assert {"cmp_a_simpson_mean_syn", "cmp_b_two_hop_same_rate_real", "cmp_mean_dominant_share_tvd"} <= set(row)
    assert {"cmp_a_gap_simpson_mean", "cmp_a_gapdeg_simpson_mean", "cmp_mean_gap_two_hop_same_rate"} <= set(row)
    # v1-style config: homophily_groups / cluster_* fallbacks; v2 keys win when present
    v1 = {"homophily_attrs": ["a"], "homophily_groups": [["a"], ["a", "b"]], "cluster_bin_edges": [0, 2, 5], "cluster_min_degree": 2}
    assert composition_setup(real, v1)[0] == {"groups": [["a"], ["a", "b"]], "bin_edges": [0, 2, 5], "min_degree": 2}
    kw, _ = composition_setup(real, {"homophily_attrs": ["a"], "cluster_bin_edges": [0, 2, 5]})
    assert kw["groups"] == [["a"]] and kw["min_degree"] == 1
    v2 = {**v1, "neigh_simpson_groups": [["b"]], "neigh_simpson_bin_edges": [0, 3, 9], "neigh_simpson_min_degree": 3}
    assert composition_setup(real, v2)[0] == {"groups": [["b"]], "bin_edges": [0, 3, 9], "min_degree": 3}
    assert "cmp_a+b_simpson_mean_real" in composition_row(real, syn, *composition_setup(real, {"homophily_groups": [["a", "b"]]}))


def test_align_rejects_different_labels_or_columns():
    real = graph(["b", "a"], DOMAINS)
    with pytest.raises(ValueError):
        align(graph(["a", "b"], {**DOMAINS, "a": ["y", "other", "none"]}), real)
    with pytest.raises(ValueError):
        align(graph(["a"], DOMAINS), real)


def test_method_name_keeps_v1_v2_names_and_takes_a_named_method():
    from graph_aim.eval.evaluate import method_name

    assert method_name({"variant": "graph_aim_eps4"}) == "graph_aim"  # v1 / v1.1: no method key
    assert method_name({"variant": "graph_aim_eps4", "method": "v2"}) == "graph_aim"  # v2 generator flag
    assert method_name({"variant": "graph_aim_nocap_nonprivate", "method": "v2"}) == "graph_aim_nocap_nonprivate"
    assert method_name({"variant": "graph_aim_nocap_nonprivate", "method": "v1"}) == "graph_aim_nocap_nonprivate"
    assert method_name({"variant": "graph_aim_walks", "method": "graph_aim_walks"}) == "graph_aim_walks"


def test_community_and_walk_columns():
    import networkx as nx

    from score import community_row, walk_row, walk_setup
    from graph_aim_walks.eval.metrics import community_metrics

    def g(seed):
        G = nx.powerlaw_cluster_graph(80, 3, 0.3, seed=seed)
        e = np.sort(np.array(list(G.edges()), dtype=np.int64), axis=1)
        rng = np.random.default_rng(seed)
        attrs = pd.DataFrame({c: rng.integers(0, len(DOMAINS[c]), size=80) for c in ["b", "a"]})
        return AttributedGraph(80, e, attrs, {c: DOMAINS[c] for c in ["b", "a"]})

    real, syn = g(0), g(1)
    row = community_row(syn, community_metrics(real))
    assert {"com_modularity_louvain", "com_modularity_louvain_real", "com_spectral_gap", "com_n_parts_real"} <= set(row)
    assert all(k.startswith("com_") for k in row)
    assert walk_row(real, syn, walk_setup(real, 0)) == {}
    w = walk_row(real, syn, walk_setup(real, 300))
    assert w and all(k.startswith("wlk_") for k in w)
    assert {"wlk_share_var_ratio_w64", "wlk_share_var_ratio_w128", "wlk_spectrum_abs", "wlk_return_abs"} <= set(w)

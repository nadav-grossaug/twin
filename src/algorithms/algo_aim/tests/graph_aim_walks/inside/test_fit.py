"""fit_walks on a toy graph with attributes a, b."""

from __future__ import annotations

import json

import numpy as np
import pytest

from graph_aim.inside.fit import fit
from graph_aim_walks.inside.fit import fit_walks, load_config, spec_from_config
from graph_aim_walks.transfer import walkstats as W
from graph_aim_walks.transfer.artifact import PROVENANCE_ALLOWED, validate
from graph_aim_walks.transfer.types import SPECTRUM, expected_shapes, table_names
from tests.graph_aim_walks.conftest import CONFIG_DIR


def test_tables_present_with_the_spec_shapes(fitted, graph):
    spec = fitted.spec
    assert spec.share_attrs == ("a", "b")  # "all" = every attribute of the schema
    assert spec.groups == (("a",), ("a", "b")) and spec.dominance_pairs == (("a", "b"),)
    assert list(fitted.tables) == table_names(spec)
    shapes = expected_shapes(spec, graph.domains)
    for name, v in fitted.tables.items():
        if name != SPECTRUM:
            assert v.shape == shapes[name], name
    assert fitted.tables["attr_lag:a+b"].shape == (4, 16, 16)
    assert fitted.tables["share:a"].shape == (3, 4, 10)
    assert fitted.tables[SPECTRUM].shape == (16,)
    assert validate(fitted) == []


def test_tables_equal_the_reference_computation(fitted, graph, config):
    ref = W.walk_tables(graph.edges, graph.n, graph.attrs, graph.domains, fitted.spec, n_walks=config["n_walks"], seed=0)
    for k, v in ref.items():
        np.testing.assert_array_equal(fitted.tables[k], v, err_msg=k)
    np.testing.assert_allclose(fitted.tables[SPECTRUM], W.transition_spectrum(graph.edges, graph.n, k=16))
    assert fitted.tables[SPECTRUM][0] == pytest.approx(1.0)
    assert fitted.n_walks == config["n_walks"] == fitted.provenance["n_walks"]


def test_base_artifact_is_exact(fitted, graph, config):
    exact = fit(graph, dict(config["base_config"], noise=False), seed=0)
    assert list(fitted.base.measurements) == list(exact.measurements)
    for k, m in exact.measurements.items():
        assert fitted.base.measurements[k].sigma == 0
        np.testing.assert_array_equal(fitted.base.measurements[k].values, m.values, err_msg=k)
    assert fitted.base.provenance["noise"] is False
    assert any(k.startswith("attr_neigh_simpson") for k in exact.measurements) and "spectrum" in exact.measurements


def test_provenance_public_and_reproducible(fitted, graph, config):
    assert set(fitted.provenance) <= PROVENANCE_ALLOWED
    assert fitted.provenance["method"] == "graph_aim_walks" and fitted.provenance["seed"] == 0
    again = fit_walks(graph, config, seed=0)
    for k in fitted.tables:
        np.testing.assert_array_equal(again.tables[k], fitted.tables[k])
    other = fit_walks(graph, config, seed=1)
    assert not np.array_equal(other.tables["coverage"], fitted.tables["coverage"])


def test_spec_from_config_rejects_unknown_attributes(graph):
    with pytest.raises(ValueError, match="not in the graph"):
        spec_from_config({"groups": [["school"]]}, graph.domains)
    spec = spec_from_config({"share_attrs": ["b"]}, graph.domains)
    assert spec.share_attrs == ("b",) and spec.length == 32


def test_shipped_config(tmp_path):
    raw = json.loads((CONFIG_DIR / "graph_aim_walks.json").read_text())
    assert raw["base_config"] == "graph_aim_v2.json" and raw["n_walks"] == 200000 and raw["spectrum_k"] == 32
    assert "NOT PRIVATE" in raw["_notes"]
    s = raw["spec"]
    assert s["groups"] == [["school"], ["hometown"], ["location"], ["hometown", "location"]]
    assert s["share_attrs"] == "all"
    assert s["dominance_pairs"] == [["hometown", "location"], ["hometown", "school"], ["school", "locale"]]
    cfg = load_config(CONFIG_DIR / "graph_aim_walks.json")
    assert cfg["base_config"] == json.loads((CONFIG_DIR / "graph_aim_v2.json").read_text())
    # the spec block builds a valid WalkSpec with the defaults of the contract
    domains = {c: ["v", "other", "none"] for c in ("gender", "locale", "birthday", "school", "employer", "location",
                                                   "hometown", "language")}
    spec = spec_from_config(s, domains)
    assert spec == W.WalkSpec(groups=spec.groups, share_attrs=tuple(domains), dominance_pairs=spec.dominance_pairs)

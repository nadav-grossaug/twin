"""Save / load / validate / inspect of the graph-aim-walks artifact (NOT private)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from graph_aim.transfer.artifact import save as save_base
from graph_aim_walks.transfer import artifact as A
from graph_aim_walks.transfer.inspect import inspect_report, main as inspect_main
from graph_aim_walks.transfer.types import SPECTRUM, WalksArtifact, expected_shapes, table_names


def _copy(art: WalksArtifact, **kw) -> WalksArtifact:
    tables = {k: np.array(v, copy=True) for k, v in art.tables.items()}
    return dataclasses.replace(art, tables=kw.pop("tables", tables), provenance=kw.pop("provenance", dict(art.provenance)), **kw)


def test_fitted_artifact_is_valid(fitted):
    assert A.validate(fitted) == []
    assert list(fitted.tables) == table_names(fitted.spec)


def test_round_trip(fitted, tmp_path):
    A.save(fitted, tmp_path / "art")
    back = A.load(tmp_path / "art")
    assert back.spec == fitted.spec and back.n_walks == fitted.n_walks and back.provenance == fitted.provenance
    assert list(back.tables) == list(fitted.tables)
    for k, v in fitted.tables.items():
        np.testing.assert_array_equal(back.tables[k], v, err_msg=k)
    assert list(back.base.measurements) == list(fitted.base.measurements)
    for k, m in fitted.base.measurements.items():
        np.testing.assert_array_equal(back.base.measurements[k].values, m.values)
    assert A.validate(back) == []
    files = {p.relative_to(tmp_path / "art").as_posix() for p in (tmp_path / "art").rglob("*") if p.is_file()}
    assert {"manifest.json", "spec.json", "privacy.json", "provenance.json", "base/manifest.json"} <= files
    assert any(f.startswith("tables/") and f.endswith("_transition_spectrum.npz") for f in files)
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text())
    assert manifest["method"] == "graph_aim_walks" and manifest["format_version"] == A.FORMAT_VERSION
    assert set(manifest["sha256"]) == files - {"manifest.json"}


def test_privacy_json_says_not_private(fitted, tmp_path):
    A.save(fitted, tmp_path / "art")
    priv = json.loads((tmp_path / "art" / "privacy.json").read_text())
    assert priv["private"] is False and priv["epsilon"] is None
    assert "NOT PRIVATE" in priv["warning"] and "exact" in priv["warning"]
    assert fitted.provenance["private"] is False
    assert "NOT PRIVATE" in inspect_report(fitted)


def test_save_refuses_a_non_empty_directory(fitted, tmp_path):
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        A.save(fitted, tmp_path / "art")


@pytest.mark.parametrize("target", ["table", "base", "spec", "privacy"])
def test_tamper_detection(fitted, tmp_path, target):
    path = A.save(fitted, tmp_path / "art")
    f = {
        "table": next((path / "tables").glob("*_return.npz")),
        "base": path / "base" / "schema.json",
        "spec": path / "spec.json",
        "privacy": path / "privacy.json",
    }[target]
    data = bytearray(f.read_bytes())
    data[-2] ^= 0x01
    f.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        A.load(path)
    assert inspect_main([str(path)]) == 1


def test_extra_or_missing_file_detected(fitted, tmp_path):
    path = A.save(fitted, tmp_path / "art")
    (path / "tables" / "999_extra.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected"):
        A.load(path)
    (path / "tables" / "999_extra.npz").unlink()
    next((path / "tables").glob("*_coverage.json")).unlink()
    with pytest.raises(ValueError, match="missing"):
        A.load(path)


def test_privacy_flag_cannot_be_flipped_even_with_a_rehashed_manifest(fitted, tmp_path):
    path = A.save(fitted, tmp_path / "art")
    priv = json.loads((path / "privacy.json").read_text())
    priv["private"] = True
    (path / "privacy.json").write_text(json.dumps(priv))
    m = json.loads((path / "manifest.json").read_text())
    m["sha256"]["privacy.json"] = A._sha256(path / "privacy.json")
    (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="private"):
        A.load(path)


def test_load_rejects_a_plain_graph_aim_artifact(fitted, tmp_path):
    save_base(fitted.base, tmp_path / "base_only")
    with pytest.raises(ValueError, match="method"):
        A.load(tmp_path / "base_only")


def test_shape_errors(fitted):
    bad = _copy(fitted)
    bad.tables["degree_lag"] = bad.tables["degree_lag"][:, :-1]
    bad.tables["share:a"] = bad.tables["share:a"][:2]
    errs = A.validate(bad)
    assert any("'degree_lag'" in e and "shape" in e for e in errs)
    assert any("'share:a'" in e and "shape" in e for e in errs)


def test_missing_and_unexpected_tables(fitted):
    bad = _copy(fitted)
    del bad.tables["attr_lag:a+b"]
    bad.tables["attr_lag:b"] = bad.tables["attr_lag:a"]
    errs = A.validate(bad)
    assert any("missing ['attr_lag:a+b']" in e and "unexpected ['attr_lag:b']" in e for e in errs)


def test_values_finite_and_distributions_sum_to_one(fitted):
    bad = _copy(fitted)
    bad.tables["dominance:a|b"][0, 0, 0] += 0.1
    bad.tables["return"][3] = np.nan
    bad.tables["coverage"][0] = 5.0  # more distinct nodes than t + 1 after one step
    errs = A.validate(bad)
    assert any("'dominance:a|b'" in e and "sum to 1" in e for e in errs)
    assert any("'return'" in e and "non-finite" in e for e in errs)
    assert any("'coverage'" in e for e in errs)


def test_spectrum_checks(fitted):
    bad = _copy(fitted)
    bad.tables[SPECTRUM] = bad.tables[SPECTRUM][::-1].copy()
    assert any("descending" in e for e in A.validate(bad))
    bad.tables[SPECTRUM] = np.concatenate([[1.5], fitted.tables[SPECTRUM][1:]])
    assert any("[-1, 1]" in e for e in A.validate(bad))
    bad.tables[SPECTRUM] = np.ones(fitted.provenance["spectrum_k"] + 1)
    assert any("spectrum_k" in e for e in A.validate(bad))


def test_spec_attributes_must_be_in_the_base_schema(fitted):
    spec = dataclasses.replace(fitted.spec, share_attrs=("a", "zz"))
    errs = A.validate(_copy(fitted, spec=spec))
    assert any("'zz'" in e and "schema" in e for e in errs)
    with pytest.raises(KeyError):
        expected_shapes(spec, fitted.base.schema.domains)


def test_provenance_allowlist(fitted, tmp_path):
    bad = _copy(fitted, provenance=dict(fitted.provenance, edge_count=88234))
    assert any("allowlist" in e and "edge_count" in e for e in A.validate(bad))
    bad = _copy(fitted, provenance=dict(fitted.provenance, private=True))
    assert any("'private' must be false" in e for e in A.validate(bad))
    bad = _copy(fitted, provenance=dict(fitted.provenance, method="graph_aim_v1"))
    assert any("method" in e for e in A.validate(bad))


def test_base_problems_are_reported(fitted):
    m = next(iter(fitted.base.measurements.values()))
    base = dataclasses.replace(fitted.base, measurements=dict(fitted.base.measurements))
    base.measurements[m.name] = dataclasses.replace(m, values=m.values[:-1])
    errs = A.validate(_copy(fitted, base=base))
    assert any(e.startswith("base: ") for e in errs)


def test_inspect_cli(fitted, tmp_path, capsys):
    path = A.save(fitted, tmp_path / "art")
    assert inspect_main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "NOT PRIVATE" in out and "no problems found" in out and "transition spectrum" in out
    assert inspect_main([]) == 2

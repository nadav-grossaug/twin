"""Tests for artifact I/O, validation and inspection."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from graph_aim.transfer.artifact import entry_rho, load, save, validate, zcdp_to_eps
from graph_aim.transfer.inspect import inspect_report, main
from graph_aim.transfer.types import Artifact, Measurement, PrivacyLedger, Schema

SRC = str(Path(__file__).resolve().parents[3] / "src")


def make_artifact(private: bool = True) -> Artifact:
    rng = np.random.default_rng(0)
    D = 6
    domains = {"gender": ["male", "female", "other", "none"], "edu": ["hs", "college", "other", "none"]}
    sigma = 2.0 if private else 0.0

    def sym(g):
        a = rng.uniform(0, 50, (g, g))
        return (a + a.T) / 2

    specs = [
        ("degree_hist", rng.uniform(0, 30, D + 1), {}, 3.0, 0.0),
        ("clustering_by_degree", rng.uniform(0, 30, (3, 2)), {"bin_edges": [0, 2, 4, 7]}, 6.0, 0.0),
        ("joint_degree", sym(3), {"bin_edges": [0, 2, 4, 7]}, 12.0, 0.0),
        ("attr_1way", rng.uniform(0, 99, 4), {"attrs": ["gender"]}, 0.0, math.sqrt(2)),
        ("attr_2way", rng.uniform(0, 99, (4, 4)), {"attrs": ["gender", "edu"]}, 0.0, math.sqrt(2)),
        ("attr_edge_pair", sym(4), {"attrs": ["gender"]}, 3.0, 12.0),
    ]
    ms, entries = {}, []
    for name, vals, params, se, sa in specs:
        ms[name] = Measurement(name, name, vals, "gaussian", sigma, se, sa, params)
        entries.append({"name": name, "rho_edge": entry_rho(sigma, se), "rho_attr": entry_rho(sigma, sa)})
    ledger = PrivacyLedger(
        delta=1e-6,
        rho_edge=sum(e["rho_edge"] for e in entries),
        rho_attr=sum(e["rho_attr"] for e in entries),
        entries=entries,
        caveats=["attribute domains chosen from real data"],
    )
    schema = Schema(n=100, domains=domains, degree_cap=D, cap_salt=7, homophily_attrs=["gender"])
    return Artifact(schema, ms, ledger, {"seed": 1, "versions": {"numpy": np.__version__}})


@pytest.fixture
def art():
    return make_artifact()


def assert_equal(a: Artifact, b: Artifact):
    assert a.schema == b.schema
    assert a.provenance == b.provenance
    assert a.ledger == b.ledger
    assert a.measurements.keys() == b.measurements.keys()
    for k, m in a.measurements.items():
        o = b.measurements[k]
        assert m.values.dtype == o.values.dtype
        assert m.values.tobytes() == o.values.tobytes()
        assert (m.name, m.kind, m.mechanism, m.sigma) == (o.name, o.kind, o.mechanism, o.sigma)
        assert (m.sensitivity_edge, m.sensitivity_attr, m.params) == (o.sensitivity_edge, o.sensitivity_attr, o.params)


def test_fixture_is_valid(art):
    assert validate(art) == []


def test_json_files_use_lf_line_endings(art, tmp_path):
    """Artifacts are byte-exact (.gitattributes -text): no CRLF may appear, whatever the platform."""
    save(art, tmp_path / "a")
    jsons = [p for p in (tmp_path / "a").rglob("*.json")]
    assert jsons
    for p in jsons:
        data = p.read_bytes()
        assert b"\r" not in data and data.endswith(b"\n"), p.name
    assert_equal(art, load(tmp_path / "a"))


def test_round_trip_exact(art, tmp_path):
    save(art, tmp_path / "a")
    assert_equal(art, load(tmp_path / "a"))
    layout = {p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*") if p.is_file()}
    assert {"manifest.json", "schema.json", "privacy.json", "provenance.json"} <= layout
    assert any(f.startswith("measurements/") and f.endswith("_degree_hist.json") for f in layout)
    assert any(f.startswith("measurements/") and f.endswith("_degree_hist.npz") for f in layout)
    man = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert man["format_version"] == 1 and man["method"] == "graph_aim_v1" and man["created"]
    assert set(man["sha256"]) == layout - {"manifest.json"}


def test_no_pickle(art, tmp_path):
    save(art, tmp_path / "a")
    for p in (tmp_path / "a" / "measurements").glob("*.npz"):
        with np.load(p, allow_pickle=False) as z:
            assert z["values"].dtype.kind == "f"
    bad = make_artifact()
    bad.measurements["degree_hist"].values = np.array([object()] * 7, dtype=object)
    with pytest.raises(TypeError):
        save(bad, tmp_path / "b")


def test_non_private_round_trip(tmp_path):
    a = make_artifact(private=False)
    assert math.isinf(a.ledger.rho_edge) and math.isinf(a.ledger.rho_attr)
    assert validate(a) == []
    save(a, tmp_path / "np")
    b = load(tmp_path / "np")
    assert math.isinf(b.ledger.rho_edge) and b.measurements["degree_hist"].sigma == 0
    assert_equal(a, b)
    priv = json.loads((tmp_path / "np" / "privacy.json").read_text())
    assert priv["derived"]["edge"]["epsilon"] == "inf"
    rep = inspect_report(tmp_path / "np")
    assert "NOT differentially private" in rep


def test_derived_epsilon(art, tmp_path):
    save(art, tmp_path / "a")
    priv = json.loads((tmp_path / "a" / "privacy.json").read_text())
    rho = art.ledger.rho_edge
    assert priv["derived"]["edge"]["epsilon"] == pytest.approx(rho + 2 * math.sqrt(rho * math.log(1e6)))
    assert zcdp_to_eps(0.0, 1e-6) == 0.0 and zcdp_to_eps(math.inf, 1e-6) == math.inf
    assert "edge-adjacent" in priv["adjacency"]["edge"] and "attribute" in priv["adjacency"]["attr"]
    assert priv["caveats"] == art.ledger.caveats and len(priv["entries"]) == 6


def _flip_byte(path: Path, offset: int = -20):
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


def test_tamper_npz(art, tmp_path):
    save(art, tmp_path / "a")
    _flip_byte(next((tmp_path / "a" / "measurements").glob("*_attr_1way.npz")), 100)
    with pytest.raises(ValueError, match="hash mismatch"):
        load(tmp_path / "a")


def test_tamper_json(art, tmp_path):
    save(art, tmp_path / "a")
    p = tmp_path / "a" / "privacy.json"
    p.write_text(p.read_text().replace('"rho_edge"', '"rho_edgf"', 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        load(tmp_path / "a")


def test_extra_and_missing_files_and_version(art, tmp_path):
    save(art, tmp_path / "a")
    (tmp_path / "a" / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        load(tmp_path / "a")
    (tmp_path / "a" / "extra.json").unlink()
    (tmp_path / "a" / "provenance.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        load(tmp_path / "a")
    save(art, tmp_path / "b")
    mp = tmp_path / "b" / "manifest.json"
    man = json.loads(mp.read_text())
    man["format_version"] = 99
    mp.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="format version"):
        load(tmp_path / "b")


def test_save_refuses_non_empty(art, tmp_path):
    save(art, tmp_path / "a")
    with pytest.raises(FileExistsError):
        save(art, tmp_path / "a")


def _corrupt(mutator):
    a = make_artifact()
    mutator(a)
    return validate(a)


@pytest.mark.parametrize(
    "mutator, needle",
    [
        (lambda a: a.measurements["degree_hist"].__setattr__("values", np.zeros(5)), "shape"),
        (lambda a: a.measurements["joint_degree"].__setattr__("values", np.zeros((3, 4))), "shape"),
        (lambda a: a.measurements["degree_hist"].values.__setitem__(0, -1.0), "negative"),
        (lambda a: a.measurements["degree_hist"].values.__setitem__(0, np.nan), "non-finite"),
        (lambda a: a.measurements["joint_degree"].values.__setitem__((0, 1), 999.0), "symmetric"),
        (lambda a: a.measurements["attr_edge_pair"].values.__setitem__((1, 0), 999.0), "symmetric"),
        (lambda a: a.measurements["joint_degree"].params.__setitem__("bin_edges", [0, 2, 4, 7, 9]), "shape"),
        (lambda a: a.measurements["clustering_by_degree"].params.__setitem__("bin_edges", [0, 4, 2, 7]), "ascending"),
        (lambda a: a.measurements["attr_1way"].params.__setitem__("attrs", ["nope"]), "attrs"),
        (lambda a: a.measurements["attr_2way"].__setattr__("values", np.zeros((4, 3))), "shape"),
        (lambda a: a.measurements["attr_1way"].__setattr__("kind", "mystery"), "unknown kind"),
        (lambda a: setattr(a.ledger, "rho_edge", a.ledger.rho_edge + 1.0), "rho_edge"),
        (lambda a: setattr(a.ledger, "rho_attr", 0.0), "rho_attr"),
        (lambda a: a.ledger.entries.pop(), "do not match"),
        (lambda a: a.ledger.entries[0].__setitem__("rho_edge", 0.5), "rho_edge"),
        (lambda a: a.measurements["degree_hist"].__setattr__("sigma", -1.0), "sigma"),
        (lambda a: setattr(a.schema, "homophily_attrs", ["ghost"]), "homophily"),
        (lambda a: a.schema.domains.__setitem__("gender", ["male", "female"]), "other"),
    ],
)
def test_validate_catches(mutator, needle):
    problems = _corrupt(mutator)
    assert problems and any(needle in p for p in problems), problems


def test_inspect_report_fields(art, tmp_path):
    save(art, tmp_path / "a")
    rep = inspect_report(tmp_path / "a")
    assert rep == inspect_report(art) or "Provenance" in rep
    for key in ("Schema", "Privacy budget", "gender", "degree_hist", "attr_edge_pair", "noise/signal", "Caveats",
                "attribute domains chosen from real data", "edge | ", "no problems found"):
        assert key in rep
    eps = zcdp_to_eps(art.ledger.rho_edge, art.ledger.delta)
    assert f"{eps:.4g}" in rep
    # sigma 2 over mean magnitude of degree_hist
    m = art.measurements["degree_hist"]
    assert f"{2.0 / np.mean(m.values):.4g}" in rep


def test_cli(art, tmp_path):
    save(art, tmp_path / "a")
    proc = subprocess.run(
        [sys.executable, "-m", "graph_aim.transfer.inspect", str(tmp_path / "a")],
        capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": ""}, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Graph-AIM artifact report" in proc.stdout
    _flip_byte(tmp_path / "a" / "schema.json", -3)
    assert main([str(tmp_path / "a")]) == 1
    assert main([]) == 2


def test_stdlib_inspect_not_shadowed():
    import inspect as stdlib_inspect

    assert hasattr(stdlib_inspect, "signature")


def test_names_with_special_characters_round_trip(tmp_path):
    a = make_artifact()
    m = a.measurements.pop("degree_hist")
    for name in ("attr_1way:gender", "attr_2way:a|b", "../escape"):
        a.measurements[name] = Measurement(**{**m.__dict__, "name": name})
        a.ledger.entries.append({"name": name, "rho_edge": 0.0, "rho_attr": 0.0})
    a.ledger.entries = [e for e in a.ledger.entries if e["name"] != "degree_hist"]
    save(a, tmp_path / "a")
    assert all(p.resolve().is_relative_to((tmp_path / "a").resolve()) for p in (tmp_path / "a").rglob("*"))
    assert not (tmp_path / "escape.json").exists()
    assert set(load(tmp_path / "a").measurements) == set(a.measurements)


def test_degree_hist_shape_uses_max_degree_param(tmp_path):
    """Round 2: degree_hist is uncapped; its length is params['max_degree'] + 1, not schema.degree_cap + 1."""
    a = make_artifact()
    m = a.measurements["degree_hist"]
    m.values = np.linspace(1, 3, 21)
    assert any("shape" in e for e in validate(a))  # length D + 1 = 7 is expected without the param
    m.params = {"max_degree": 20, "cap": None, "hist_edges": [0, 1, 2, 5, 21], "threshold": 1.0, "spread_power": 1.0}
    assert validate(a) == []
    m.params["max_degree"] = 0
    assert any("max_degree" in e for e in validate(a))
    m.params["max_degree"] = 20
    save(a, tmp_path / "a")
    assert load(tmp_path / "a").measurements["degree_hist"].params == m.params  # None and lists survive the round trip


def test_validate_rejects_unlisted_provenance_key():
    a = make_artifact()
    assert validate(a) == []
    a.provenance["capped_edges_by_cap"] = {"150": 80032}  # an exact, un-noised real-data statistic
    assert any("provenance" in e and "capped_edges_by_cap" in e for e in validate(a))


# --------------------------------------------------------------------------- attr_degree


def with_attr_degree(a: Artifact, **override) -> Artifact:
    """``a`` plus an ``attr_degree:gender`` measurement (4 values x 3 bins); ``override`` replaces fields."""
    a = copy.deepcopy(a)
    fields = dict(
        name="attr_degree:gender",
        kind="attr_degree",
        values=np.random.default_rng(3).uniform(0, 30, (4, 3)),
        mechanism="gaussian",
        sigma=2.0,
        sensitivity_edge=2 * math.sqrt(2),
        sensitivity_attr=math.sqrt(2),
        params={"attrs": ["gender"], "cap": None, "bin_edges": [0, 2, 5, 9]},
    )
    fields.update(override)
    m = Measurement(**fields)
    a.measurements[m.name] = m
    e = {"name": m.name, "rho_edge": entry_rho(2.0, m.sensitivity_edge), "rho_attr": entry_rho(2.0, m.sensitivity_attr)}
    a.ledger.entries.append(e)
    a.ledger.rho_edge += e["rho_edge"]
    a.ledger.rho_attr += e["rho_attr"]
    return a


def test_attr_degree_round_trip_and_inspect(art, tmp_path):
    a = with_attr_degree(art)
    assert validate(a) == []
    save(a, tmp_path / "a")
    b = load(tmp_path / "a")
    assert_equal(a, b)
    assert b.measurements["attr_degree:gender"].values.shape == (4, 3)
    rep = inspect_report(tmp_path / "a")
    assert "attr_degree:gender" in rep and "4x3" in rep and "no problems found" in rep


@pytest.mark.parametrize(
    "override, text",
    [
        ({"values": np.ones((3, 3))}, "shape"),
        ({"values": np.ones((4, 4))}, "shape"),
        ({"params": {"attrs": ["gender"], "cap": None, "bin_edges": [0, 5, 2, 9]}}, "bin_edges"),
        ({"params": {"attrs": ["gender"], "cap": None, "bin_edges": [1, 2, 5, 9]}}, "bin_edges"),
        ({"params": {"attrs": ["gender"], "cap": None}}, "bin_edges"),
        ({"params": {"attrs": ["nope"], "cap": None, "bin_edges": [0, 2, 5, 9]}}, "attrs"),
        ({"params": {"attrs": ["gender", "edu"], "cap": None, "bin_edges": [0, 2, 5, 9]}}, "attrs"),
        ({"params": {"attrs": ["gender"], "cap": 5, "bin_edges": [0, 2, 5, 9]}}, "cap"),
    ],
)
def test_attr_degree_validation_rejects(art, override, text):
    errs = validate(with_attr_degree(art, **override))
    assert any(text in e for e in errs), errs


def test_artifact_without_attr_degree_still_validates(art, tmp_path):
    assert "attr_degree:gender" not in art.measurements and validate(art) == []
    save(art, tmp_path / "old")
    assert validate(load(tmp_path / "old")) == []


# --------------------------------------------------------------------------- attr_edge_pair groups


def _add_edge_pair(a: Artifact, name: str, attrs: list[str], size: int) -> Measurement:
    rng = np.random.default_rng(len(name))
    v = rng.uniform(0, 50, (size, size))
    m = Measurement(name, "attr_edge_pair", (v + v.T) / 2, "gaussian", 2.0, 3.0, 12.0, {"attrs": attrs})
    a.measurements[name] = m
    a.ledger.entries.append({"name": name, "rho_edge": entry_rho(2.0, 3.0), "rho_attr": entry_rho(2.0, 12.0)})
    a.ledger.rho_edge += entry_rho(2.0, 3.0)
    a.ledger.rho_attr += entry_rho(2.0, 12.0)
    return m


def with_edge_pair_groups(a: Artifact) -> Artifact:
    """The fixture's bare ``attr_edge_pair`` replaced by two named instances (two homophily groups)."""
    a = copy.deepcopy(a)
    del a.measurements["attr_edge_pair"]
    e = next(e for e in a.ledger.entries if e["name"] == "attr_edge_pair")
    a.ledger.entries.remove(e)
    a.ledger.rho_edge -= e["rho_edge"]
    a.ledger.rho_attr -= e["rho_attr"]
    _add_edge_pair(a, "attr_edge_pair:gender", ["gender"], 4)
    _add_edge_pair(a, "attr_edge_pair:gender+edu", ["gender", "edu"], 16)
    return a


def test_edge_pair_two_instances_round_trip(art, tmp_path):
    a = with_edge_pair_groups(art)
    assert validate(a) == []
    save(a, tmp_path / "a")
    b = load(tmp_path / "a")
    assert_equal(a, b)
    assert b.measurements["attr_edge_pair:gender"].values.shape == (4, 4)
    assert b.measurements["attr_edge_pair:gender+edu"].values.shape == (16, 16)
    rep = inspect_report(tmp_path / "a")
    assert "attr_edge_pair:gender+edu" in rep and "16x16" in rep and "no problems found" in rep


def test_legacy_bare_edge_pair_name_still_validates(art, tmp_path):
    """Old artifacts carry a single instance named exactly ``attr_edge_pair`` (additive change)."""
    assert art.measurements["attr_edge_pair"].kind == "attr_edge_pair" and validate(art) == []
    save(art, tmp_path / "old")
    assert validate(load(tmp_path / "old")) == []


# --------------------------------------------------------------------------- composition round (v2)

ARTIFACTS = Path(__file__).resolve().parents[3] / "snap" / "transfer" / "artifacts"


def _add(a: Artifact, m: Measurement) -> None:
    a.measurements[m.name] = m
    e = {"name": m.name, "rho_edge": entry_rho(m.sigma, m.sensitivity_edge), "rho_attr": entry_rho(m.sigma, m.sensitivity_attr)}
    a.ledger.entries.append(e)
    a.ledger.rho_edge += e["rho_edge"]
    a.ledger.rho_attr += e["rho_attr"]


def simpson_measurement(**override) -> Measurement:
    fields = dict(
        name="attr_neigh_simpson:gender+edu",
        kind="attr_neigh_simpson",
        values=np.column_stack([np.array([0.0, 3.5, 7.0]), np.array([10.0, 20.0, 30.0])]),
        mechanism="gaussian",
        sigma=2.0,
        sensitivity_edge=6 * math.sqrt(2),
        sensitivity_attr=2 * 6 / 2,
        params={"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 2, 4, 7], "min_degree": 2, "counts_from_hist": True},
    )
    fields.update(override)
    return Measurement(**fields)


def spectrum_measurement(**override) -> Measurement:
    fields = dict(
        name="spectrum",
        kind="spectrum",
        values=np.array([5.5, 3.0, 0.2, -1.5]),
        mechanism="gaussian",
        sigma=2.0,
        sensitivity_edge=math.sqrt(2),
        sensitivity_attr=0.0,
        params={"k": 4, "cap": None, "matrix": "adjacency"},
    )
    fields.update(override)
    return Measurement(**fields)


def with_v2_kinds(a: Artifact, simpson=None, spectrum=None) -> Artifact:
    a = copy.deepcopy(a)
    _add(a, simpson_measurement(**(simpson or {})))
    _add(a, spectrum_measurement(**(spectrum or {})))
    return a


def test_v2_kinds_round_trip_and_inspect(art, tmp_path):
    a = with_v2_kinds(art)
    assert validate(a) == []
    save(a, tmp_path / "a")
    b = load(tmp_path / "a")
    assert_equal(a, b)
    assert b.measurements["attr_neigh_simpson:gender+edu"].values.shape == (3, 2)
    assert b.measurements["attr_neigh_simpson:gender+edu"].params["counts_from_hist"] is True
    assert b.measurements["spectrum"].values.shape == (4,) and b.measurements["spectrum"].values[-1] == -1.5
    assert b.measurements["spectrum"].params["cap"] is None
    rep = inspect_report(tmp_path / "a")
    assert "attr_neigh_simpson:gender+edu" in rep and "3x2" in rep
    assert "| spectrum | spectrum | 4 |" in rep and "no problems found" in rep


@pytest.mark.parametrize(
    "override, text",
    [
        ({"values": np.ones((3, 3))}, "shape"),
        ({"values": np.ones((4, 2))}, "shape"),
        ({"values": np.array([[0.0, 1.0], [-1.0, 2.0], [0.0, 3.0]])}, "negative"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 2, 4, 7], "min_degree": 3, "counts_from_hist": True}}, "min_degree"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 2, 4, 7], "min_degree": 0, "counts_from_hist": True}}, "min_degree"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 2, 4, 7], "counts_from_hist": True}}, "min_degree"),
        ({"params": {"attrs": ["gender", "edu"], "cap": None, "bin_edges": [0, 2, 4, 7], "min_degree": 2, "counts_from_hist": True}}, "cap"),
        ({"params": {"attrs": ["gender", "edu"], "bin_edges": [0, 2, 4, 7], "min_degree": 2, "counts_from_hist": True}}, "cap"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [1, 2, 4, 7], "min_degree": 2, "counts_from_hist": True}}, "bin_edges"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 4, 2, 7], "min_degree": 2, "counts_from_hist": True}}, "ascending"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "min_degree": 2, "counts_from_hist": True}}, "bin_edges"),
        ({"params": {"attrs": ["nope"], "cap": 6, "bin_edges": [0, 2, 4, 7], "min_degree": 2, "counts_from_hist": True}}, "attrs"),
        ({"params": {"attrs": ["gender", "edu"], "cap": 6, "bin_edges": [0, 2, 4, 7], "min_degree": 2}}, "counts_from_hist"),
    ],
)
def test_simpson_validation_rejects(art, override, text):
    errs = validate(with_v2_kinds(art, simpson=override))
    assert any("attr_neigh_simpson:gender+edu" in e and text in e for e in errs), errs


@pytest.mark.parametrize(
    "override, text",
    [
        ({"values": np.ones(3)}, "shape"),
        ({"values": np.ones((4, 1))}, "shape"),
        ({"params": {"k": 0, "cap": None, "matrix": "adjacency"}}, "k"),
        ({"params": {"cap": None, "matrix": "adjacency"}}, "k"),
        ({"params": {"k": 4, "cap": 6, "matrix": "adjacency"}}, "cap"),
        ({"params": {"k": 4, "cap": None, "matrix": "laplacian"}}, "matrix"),
        ({"values": np.array([1.0, 2.0, 0.5, 0.0])}, "descending"),
    ],
)
def test_spectrum_validation_rejects(art, override, text):
    errs = validate(with_v2_kinds(art, spectrum=override))
    assert any("'spectrum'" in e and text in e for e in errs), errs


def test_spectrum_may_be_negative_but_other_kinds_may_not(art):
    a = with_v2_kinds(art, spectrum={"values": np.array([1.0, -0.5, -2.0, -3.0])})
    assert validate(a) == []
    a.measurements["degree_hist"].values[0] = -1.0
    assert any("degree_hist" in e and "negative" in e for e in validate(a))


def test_simpson_group_exceeding_max_groups_rejected(art):
    a = copy.deepcopy(art)
    a.schema.domains["zip"] = [f"z{i}" for i in range(6)] + ["other", "none"]  # 8 labels
    m = simpson_measurement(name="attr_neigh_simpson:gender+edu+zip", params={**simpson_measurement().params, "attrs": ["gender", "edu", "zip"]})
    _add(a, m)
    errs = validate(a)
    assert any("attr_neigh_simpson:gender+edu+zip" in e and "64-cell limit" in e for e in errs), errs


def test_shipped_v1_artifact_in_the_tree_still_loads_and_validates():
    """Old artifacts (no v2 kinds, FORMAT_VERSION 1) load with full hash verification and validate unchanged."""
    path = ARTIFACTS / "graph_aim_eps4_seed0"
    assert (path / "manifest.json").is_file()
    a = load(path)
    assert json.loads((path / "manifest.json").read_text())["format_version"] == 1
    assert validate(a) == []
    kinds = {m.kind for m in a.measurements.values()}
    assert "attr_edge_pair" in kinds and not kinds & {"attr_neigh_simpson", "spectrum"}


def test_edge_pair_group_exceeding_max_groups_rejected(art):
    a = copy.deepcopy(art)
    a.schema.domains["zip"] = [f"z{i}" for i in range(6)] + ["other", "none"]  # 8 labels
    _add_edge_pair(a, "attr_edge_pair:gender+edu+zip", ["gender", "edu", "zip"], 128)  # 4*4*8 = 128 > 64
    errs = validate(a)
    assert any("attr_edge_pair:gender+edu+zip" in e and "64-cell limit" in e for e in errs), errs
    # the oversize group is the only problem reported for that measurement besides itself
    assert not any("shape" in e for e in errs), errs

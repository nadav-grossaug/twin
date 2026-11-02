"""The concept spec must agree across dbt vars, the mode-map seed, Python and
Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py) and the mode-map seed.
A failing test here means spec drift.

Every numeric dbt var is an INTEGER by design (speed edges in milli-m/s,
bboxes in centidegrees), so the Lean `Nat` mirrors are exact; the snake_case
-> camelCase name map is built programmatically.
"""
import csv
import re
from pathlib import Path

import pytest

from spec import (MODE_CLASSES, MOTIF_CLASSES, ROLES, load_spec, mode_map,
                  spec_version)

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveGeo" / "Params.lean"

# string / jinja-valued vars have no Lean Nat mirror
NON_NUMERIC = {"spec_version", "points_path"}


def camel(snake: str) -> str:
    """sp_radius_m -> spRadiusM, dw_e1_s -> dwE1S, excl_lat_min_cd -> exclLatMinCd."""
    head, *rest = snake.split("_")
    return head + "".join(p.capitalize() for p in rest)


def numeric_vars() -> dict:
    return {k: v for k, v in load_spec().items() if k not in NON_NUMERIC}


def lean_defs():
    text = PARAMS.read_text()
    defs = {m[1]: int(m[2])
            for m in re.finditer(r"def (\w+) : Nat := (\d+)", text)}
    ver = re.search(r'def specVersion : String := "([^"]+)"', text)
    assert ver, "specVersion missing from Params.lean"
    return defs, ver.group(1)


def seed_rows() -> dict:
    with open(ROOT / "dbt" / "seeds" / "mode_map.csv") as f:
        return {r["raw_mode"]: r["mode_class"] for r in csv.DictReader(f)}


def test_every_numeric_dbt_var_is_an_integer():
    for k, v in numeric_vars().items():
        assert isinstance(v, int), (
            f"dbt var {k}={v!r} is not an integer — the spec mirror must be exact")


def test_lean_params_match_dbt_vars():
    defs, _ = lean_defs()
    for var, value in numeric_vars().items():
        lean_name = camel(var)
        assert lean_name in defs, f"{lean_name} ({var}) missing from Params.lean"
        assert defs[lean_name] == int(value), (
            f"{lean_name}={defs[lean_name]} but dbt {var}={value}")


def test_every_lean_param_is_mapped():
    defs, _ = lean_defs()
    unmapped = set(defs) - {camel(v) for v in numeric_vars()}
    assert not unmapped, f"Params.lean defs without a dbt var: {unmapped}"


def test_spec_version_matches_in_all_three_places():
    _, lean_ver = lean_defs()
    assert lean_ver == spec_version() == str(load_spec()["spec_version"])


def test_seed_matches_python_mode_map():
    rows = seed_rows()
    assert rows == mode_map()
    assert len(rows) == 11, "the seed must map exactly the 11 raw GeoLife labels"
    assert set(rows.values()) <= set(MODE_CLASSES)


def test_mode_map_matches_lean():
    """Every seed row must appear as a modeClassOf match arm and vice versa."""
    body = PARAMS.read_text().split("def modeClassOf", 1)[1]
    arms = dict(re.findall(r'\| "(\w+)" => "(\w+)"', body))
    assert arms == mode_map(), (
        set(arms.items()) ^ set(mode_map().items()))


@pytest.mark.parametrize("lean_name,want", [
    ("roles", ROLES),
    ("motifClasses", MOTIF_CLASSES),
    ("modeClasses", MODE_CLASSES),
])
def test_taxonomy_lists_match_lean(lean_name, want):
    text = PARAMS.read_text()
    m = re.search(rf"def {lean_name} : List String :=\s*\[([^\]]+)\]", text)
    assert m, f"{lean_name} list missing from Params.lean"
    assert re.findall(r'"([^"]+)"', m.group(1)) == want


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src

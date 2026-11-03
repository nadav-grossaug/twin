"""The inside / transfer / outside boundary of graph-aim-walks, enforced (docs/snap/walks-plan.md section 3).

transfer = numpy/scipy + graph_aim.transfer + graph_aim_walks.transfer
outside  = additionally pandas, graph_aim.outside and graph_aim_walks.outside
eval     = inside-only; nothing in inside / transfer / outside imports it (nor realfake)
graph_aim imports nothing from graph_aim_walks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.graph_aim.test_boundary import FORBIDDEN_STDLIB, imported_modules

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "graph_aim_walks"
TRANSFER_OK = ("graph_aim.transfer", "graph_aim_walks.transfer")
OUTSIDE_OK = TRANSFER_OK + ("graph_aim.outside", "graph_aim_walks.outside")


def violations(path: Path, allowed: tuple[str, ...], third_party: set[str]) -> list[str]:
    bad = []
    for m in imported_modules(path):
        top = m.split(".")[0]
        if top in FORBIDDEN_STDLIB:
            bad.append(m)
        elif top in sys.stdlib_module_names or top in third_party:
            continue
        elif any(m == a or m.startswith(a + ".") for a in allowed):
            continue
        else:
            bad.append(m)
    return sorted(bad)


def outside_scripts() -> list[Path]:
    return sorted((REPO / "snap" / "outside").glob("*walks*.py"))


def test_transfer_package_depends_on_nothing_inside():
    files = sorted((PKG / "transfer").glob("*.py"))
    assert {"walkstats.py", "types.py", "artifact.py", "inspect.py"} <= {p.name for p in files}
    for p in files:
        assert violations(p, TRANSFER_OK, {"numpy", "scipy"}) == [], p.name


def test_outside_package_and_scripts_see_only_transfer_and_outside():
    for p in sorted((PKG / "outside").glob("*.py")) + outside_scripts():
        assert violations(p, OUTSIDE_OK, {"numpy", "pandas", "scipy"}) == [], p.name


def test_no_module_is_left_outside_the_four_zones():
    top_level = {p.name for p in PKG.glob("*.py")}
    assert top_level <= {"__init__.py"}, f"unclassified modules: {sorted(top_level - {'__init__.py'})}"
    assert {p.name for p in PKG.iterdir() if p.is_dir() and p.name != "__pycache__"} == {
        "inside", "transfer", "outside", "eval"}


def test_nothing_outside_eval_imports_eval():
    files = [p for z in ("inside", "transfer", "outside") for p in sorted((PKG / z).glob("*.py"))]
    files += [REPO / "snap" / "inside" / "fit_walks.py"] + outside_scripts()
    eval_pkgs = ("graph_aim.eval", "graph_aim_walks.eval", "realfake")
    for p in files:
        used = {m for m in imported_modules(p) if any(m == e or m.startswith(e + ".") for e in eval_pkgs)}
        assert not used, f"{p.relative_to(REPO)} imports {sorted(used)}"


def test_graph_aim_imports_nothing_from_graph_aim_walks():
    files = sorted((REPO / "src" / "graph_aim").rglob("*.py"))
    assert len(files) > 10
    for p in files:
        used = {m for m in imported_modules(p) if m.split(".")[0] == "graph_aim_walks"}
        assert not used, f"{p.relative_to(REPO)} imports {sorted(used)}"


def test_the_rules_themselves_catch_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import os\nimport pickle\nimport pandas\nfrom graph_aim_walks.inside.fit import fit_walks\n"
        "from graph_aim.inside.fit import fit\nfrom graph_aim.outside.generate import generate\nimport json\n"
        "from graph_aim_walks.transfer.walkstats import WalkSpec\n"
    )
    flagged = set(violations(bad, TRANSFER_OK, {"numpy", "scipy"}))
    assert {"os", "pickle", "pandas", "graph_aim_walks.inside.fit", "graph_aim.inside.fit",
            "graph_aim.outside.generate"} <= flagged
    assert not {"json", "graph_aim_walks.transfer.walkstats"} & flagged
    assert "graph_aim.outside.generate" not in violations(bad, OUTSIDE_OK, {"numpy", "pandas", "scipy"})


@pytest.mark.skip(reason="W2: stage 2 (snap/outside/generate_walks.py) does not exist yet; run it in a subprocess "
                         "with only src on the path, from a copied artifacts_walks/ alone")
def test_outside_stage_runs_from_the_artifact_alone():
    raise NotImplementedError

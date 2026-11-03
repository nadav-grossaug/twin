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


def test_outside_stage_runs_from_the_artifact_alone(tmp_path, fitted):
    """Run stage 2 in a subprocess where only `src` is importable (no snap/inside, no data), from a copied
    artifacts_walks/ alone."""
    import importlib.util
    import json
    import subprocess

    if importlib.util.find_spec("graph_aim_walks.transfer.artifact") is None:
        pytest.skip("W1's loader graph_aim_walks.transfer.artifact is not present yet")
    from graph_aim.transfer.graph_io import load_attributed_graph
    from graph_aim_walks.transfer.artifact import save

    artifacts = tmp_path / "artifacts_walks"
    save(fitted, artifacts / "graph_aim_walks_seed0")
    (artifacts / "index.json").write_text(json.dumps(
        [{"variant": "graph_aim_walks", "epsilon": None, "seed": 0, "dir": "graph_aim_walks_seed0",
          "method": "graph_aim_walks"}]))
    out = tmp_path / "synthetic_walks"
    proc = subprocess.run(
        [sys.executable, str(REPO / "snap" / "outside" / "generate_walks.py"), "--artifacts", str(artifacts),
         "--out", str(out), "--seeds", "0", "1", "--cycles", "1", "--pool-size", "2000"],
        cwd=tmp_path, env={"PYTHONPATH": str(REPO / "src"), "PATH": "", "SYSTEMROOT": "", "OMP_NUM_THREADS": "2"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    names = sorted(p.name for p in out.iterdir())
    assert names == ["graph_aim_walks_seed0.info.json", "graph_aim_walks_seed0.npz", "graph_aim_walks_seed1.info.json",
                     "graph_aim_walks_seed1.npz", "index.json"]
    index = json.loads((out / "index.json").read_text())
    assert [(e["variant"], e["epsilon"], e["seed"], e["method"]) for e in index] == [
        ("graph_aim_walks", None, 0, "graph_aim_walks"), ("graph_aim_walks", None, 1, "graph_aim_walks")]
    g = load_attributed_graph(out / "graph_aim_walks_seed0.npz")
    assert g.n == fitted.base.schema.n and len(g.edges) == index[0]["edges"]
    info = json.loads((out / "graph_aim_walks_seed0.info.json").read_text())
    assert set(info["final"]) == set(fitted.tables) - {"transition_spectrum"}

"""The inside / transfer / outside boundary, enforced.

inside   = code that may read the real graph (fit, measure, privacy, evaluation, baselines, attacks, data prep)
transfer = what crosses the boundary: shared types, the artifact format, the graph file format
outside  = code that generates synthetic data from an artifact and never sees the real graph
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from graph_aim.inside.fit import fit
from graph_aim.transfer.artifact import save
from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim.transfer.types import AttributedGraph

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "graph_aim"
THIRD_PARTY_OK = {"numpy", "pandas", "scipy"}
# Standard-library modules that shipped code (transfer, outside) must never import: no processes, shell, network,
# file copying/temp files, environment access, or pickle. Shipped code only reads artifact files and writes graphs.
FORBIDDEN_STDLIB = {
    "os", "subprocess", "socket", "shutil", "tempfile", "urllib", "http", "ftplib", "smtplib", "ssl",
    "ctypes", "multiprocessing", "pickle", "shelve", "importlib", "runpy",
}


def imported_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module)
            mods.update(f"{node.module}.{a.name}" for a in node.names)
    return mods


def violations(path: Path, allowed_graph_aim: tuple[str, ...]) -> list[str]:
    bad = []
    for m in imported_modules(path):
        top = m.split(".")[0]
        if top in FORBIDDEN_STDLIB:
            bad.append(m)
            continue
        if top in sys.stdlib_module_names or top in THIRD_PARTY_OK:
            continue
        if top == "graph_aim":
            # "graph_aim.transfer.artifact.load" style names produced from `from x import y` are fine if x is allowed
            if any(m == a or m.startswith(a + ".") for a in allowed_graph_aim):
                continue
        bad.append(m)
    return sorted(bad)


def test_transfer_package_depends_on_nothing_inside():
    for p in sorted((PKG / "transfer").glob("*.py")):
        assert violations(p, ("graph_aim.transfer",)) == [], p.name


def test_outside_package_sees_only_transfer_and_itself():
    files = sorted((PKG / "outside").glob("*.py"))
    assert any(p.name == "generate.py" for p in files)
    for p in files:
        assert violations(p, ("graph_aim.transfer", "graph_aim.outside")) == [], p.name


def test_outside_scripts_see_only_transfer_and_outside():
    scripts = sorted((REPO / "snap" / "outside").glob("*.py"))
    assert any(p.name == "generate_synthetic.py" for p in scripts)
    for p in scripts:
        assert violations(p, ("graph_aim.transfer", "graph_aim.outside")) == [], p.name


def test_no_module_is_left_outside_the_four_zones():
    top_level = {p.name for p in PKG.glob("*.py")}
    assert top_level <= {"__init__.py"}, f"unclassified modules: {sorted(top_level - {'__init__.py'})}"


def toy_graph() -> AttributedGraph:
    rng = np.random.default_rng(11)
    n = 150
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < 0.06
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
    domains = {"a": ["x", "y", "other", "none"], "b": ["p", "q", "other", "none"]}
    attrs = pd.DataFrame({c: rng.integers(0, len(d), size=n) for c, d in domains.items()})
    return AttributedGraph(n, edges, attrs, domains)


def test_outside_stage_runs_from_the_artifact_alone(tmp_path):
    """Run the real outside script in a subprocess where only `src` is importable (no snap/inside, no data)."""
    cfg = json.loads((REPO / "snap" / "inside" / "config" / "graph_aim_v1.json").read_text())
    cfg.update(D=8, homophily_attrs=None, noise=True, epsilon_edge=4.0, epsilon_attr=4.0)
    art = fit(toy_graph(), cfg, seed=0)

    artifacts = tmp_path / "artifacts"
    save(art, artifacts / "toy_seed0")
    (artifacts / "index.json").write_text(
        json.dumps([{"variant": "toy", "epsilon": 4.0, "seed": 0, "dir": "toy_seed0"}])
    )
    out = tmp_path / "synthetic"
    proc = subprocess.run(
        [sys.executable, str(REPO / "snap" / "outside" / "generate_synthetic.py"),
         "--artifacts", str(artifacts), "--out", str(out)],
        cwd=tmp_path,  # a directory that holds nothing but the artifacts
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "", "SYSTEMROOT": ""},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    g = load_attributed_graph(out / "toy_seed0.npz")
    assert g.n == 150 and len(g.edges) > 0
    assert (g.edges[:, 0] < g.edges[:, 1]).all()
    assert list(g.attrs.columns) == ["a", "b"]
    # nothing but the synthetic graphs and their index is written
    assert sorted(p.name for p in out.iterdir()) == ["index.json", "toy_seed0.npz"]


def test_the_rules_themselves_catch_violations(tmp_path):
    """Guard the guard: shell/network/pickle imports and imports of inside code must be flagged."""
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess\nimport os\nfrom graph_aim.inside.fit import fit\nimport matplotlib\nimport json\n")
    flagged = violations(bad, ("graph_aim.transfer", "graph_aim.outside"))
    assert {"subprocess", "os", "graph_aim.inside.fit", "matplotlib"} <= set(flagged)
    assert "json" not in flagged


def test_nothing_outside_eval_imports_eval():
    """eval needs real and synthetic data side by side; inside, transfer and outside must not depend on it."""
    files = [p for z in ("inside", "transfer", "outside") for p in sorted((PKG / z).glob("*.py"))]
    files += sorted((REPO / "snap" / "inside").glob("*.py")) + sorted((REPO / "snap" / "outside").glob("*.py"))
    assert len(files) > 10
    for p in files:
        used = {m for m in imported_modules(p) if m == "graph_aim.eval" or m.startswith("graph_aim.eval.")}
        assert not used, f"{p.relative_to(REPO)} imports {sorted(used)}"

"""The inside / transfer / outside / eval boundary of the GeoLife code, enforced (modelled on tests/graph_aim/test_boundary.py).

inside   = may read the real traces: loader, DP, fitting (caps, noise, accounting), EDA
transfer = what crosses the boundary: contract types, artifact format, and PURE computation both sides need
           (tokenization, map graph + matcher, geometry helpers, MapProvider protocol)
outside  = turns an artifact into synthetic traces and never sees real data
eval     = needs real and synthetic side by side (method M, comparison, split-half); inside-only zone
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src" / "geolife"
SCRIPTS = REPO / "geolife"
ZONES = ("inside", "transfer", "outside", "eval")

CORE_THIRD_PARTY = {"numpy", "pandas", "scipy"}
# DECISION FOR THE USER TO CONFIRM (CLAUDE.md allows stdlib + numpy/pandas/scipy "whatever the method needs"): pure
# computation libraries the shared GeoLife code needs. None of them opens a socket, starts a process or reads outside
# the paths it is given. h3: hex cells; networkx + shapely: map graph and HMM map matching; pyarrow: parquet tables
# of the artifact format (pandas' parquet engine). The network/duckdb fetcher is NOT here: it is the script
# geolife/outside/fetch_public_map.py (exempt below, still barred from importing inside or eval).
PURE_COMPUTE_ALLOWED = {"h3", "networkx", "shapely", "pyarrow"}
THIRD_PARTY_OK = CORE_THIRD_PARTY | PURE_COMPUTE_ALLOWED
# Standard-library modules that shipped code (transfer, outside) must never import: no processes, shell, network,
# file copying/temp files, environment access, or pickle. Paths are passed in by the stage scripts.
FORBIDDEN_STDLIB = {
    "os", "subprocess", "socket", "shutil", "tempfile", "urllib", "http", "ftplib", "smtplib", "ssl",
    "ctypes", "multiprocessing", "pickle", "shelve", "importlib", "runpy",
}
STAGE2 = SCRIPTS / "outside" / "generate_synthetic.py"
FETCHER = SCRIPTS / "outside" / "fetch_public_map.py"


def imported_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: relative import (use absolute geolife.<zone>... imports)"
            if node.module:
                mods.add(node.module)
                mods.update(f"{node.module}.{a.name}" for a in node.names)
    return mods


def violations(path: Path, allowed_geolife: tuple[str, ...], third_party=THIRD_PARTY_OK) -> list[str]:
    bad = []
    for m in imported_modules(path):
        top = m.split(".")[0]
        if top in FORBIDDEN_STDLIB:
            bad.append(m)
            continue
        if top in sys.stdlib_module_names or top in third_party:
            continue
        if top == "geolife":
            # "geolife.transfer.artifact.load_artifact" style names come from `from x import y`; fine if x is allowed
            if any(m == a or m.startswith(a + ".") for a in allowed_geolife):
                continue
        bad.append(m)
    return sorted(bad)


def py_files(zone: str) -> list[Path]:
    return sorted((PKG / zone).rglob("*.py"))


def test_transfer_package_depends_on_nothing_inside():
    files = py_files("transfer")
    assert len(files) > 10
    for p in files:
        assert violations(p, ("geolife.transfer",)) == [], p.relative_to(REPO)


def test_outside_package_sees_only_transfer_and_itself():
    files = py_files("outside")
    assert any(p.name == "pipeline_v2.py" for p in files)
    for p in files:
        assert violations(p, ("geolife.transfer", "geolife.outside")) == [], p.relative_to(REPO)


def test_stage2_script_sees_only_transfer_and_outside():
    assert STAGE2.is_file()
    assert violations(STAGE2, ("geolife.transfer", "geolife.outside")) == []


def test_public_map_fetcher_may_use_network_libs_but_not_inside_or_eval():
    """The one place that touches the network / duckdb. It handles public map data only."""
    assert FETCHER.is_file()
    bad = [m for m in imported_modules(FETCHER)
           if m.split(".")[0] == "geolife" and not (m.startswith("geolife.transfer") or m.startswith("geolife.outside"))]
    assert bad == []


def test_no_module_is_left_outside_the_four_zones():
    top_level = {p.name for p in PKG.glob("*.py")}
    assert top_level <= {"__init__.py"}, f"unclassified modules: {sorted(top_level - {'__init__.py'})}"
    dirs = {p.name for p in PKG.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert dirs == set(ZONES), f"unexpected packages: {sorted(dirs ^ set(ZONES))}"


def test_nothing_outside_eval_imports_eval():
    """eval needs real and synthetic data side by side; inside, transfer and outside must not depend on it."""
    files = [p for z in ("inside", "transfer", "outside") for p in py_files(z)]
    files += [p for z in ("inside", "outside", "transfer") for p in sorted((SCRIPTS / z).glob("*.py"))]
    assert len(files) > 30
    for p in files:
        used = {m for m in imported_modules(p) if m == "geolife.eval" or m.startswith("geolife.eval.")}
        assert not used, f"{p.relative_to(REPO)} imports {sorted(used)}"


def test_transfer_and_outside_never_import_inside():
    for p in py_files("transfer") + py_files("outside") + sorted((SCRIPTS / "outside").glob("*.py")):
        used = {m for m in imported_modules(p) if m == "geolife.inside" or m.startswith("geolife.inside.")}
        assert not used, f"{p.relative_to(REPO)} imports {sorted(used)}"


def test_the_rules_themselves_catch_violations(tmp_path):
    """Guard the guard: shell/network/pickle imports, inside imports and unlisted third-party libs are flagged."""
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess\nimport os\nfrom geolife.inside.data import load_geolife\n"
                   "import matplotlib\nimport duckdb\nimport json\nimport h3\n")
    flagged = violations(bad, ("geolife.transfer", "geolife.outside"))
    assert {"subprocess", "os", "geolife.inside.data", "matplotlib", "duckdb"} <= set(flagged)
    assert "json" not in flagged and "h3" not in flagged


def test_artifact_provenance_allowlist_rejects_unlisted_keys():
    """An un-noised statistic must not slip into an artifact through a free-form json file."""
    from geolife.transfer.artifact import PROVENANCE_ALLOWED
    assert set(PROVENANCE_ALLOWED) == {"preprocessing", "diagnostics", "grounding", "generation"}
    from geolife.outside.population_v2 import DEFAULT_GEN_V2
    assert set(DEFAULT_GEN_V2) <= PROVENANCE_ALLOWED["generation"]


def _fixture_artifact(dest: Path) -> None:
    from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users
    from geolife.inside.learn_v2 import run
    run(make_users(30, seed=2), dest, epsilon=6.0, bbox=BBOX, categories=CATS, verbose=False, with_snr=False)


def test_stage2_runs_from_the_artifact_alone(tmp_path):
    """Run the real stage-2 script in a subprocess where only `src` is importable (no tests, no inside data)."""
    artifacts = tmp_path / "artifacts"
    _fixture_artifact(artifacts / "toy_seed0")
    out = tmp_path / "synthetic"
    proc = subprocess.run(
        [sys.executable, str(STAGE2), "--artifacts", str(artifacts), "--out", str(out),
         "--n-users", "6", "--map-n", "14", "--map-places", "500", "--gen-json", "{\"max_outings\": 8, \"length_scale\": 0.3}"],
        cwd=tmp_path,  # a directory that holds nothing but the artifacts
        env={"PYTHONPATH": str(PKG.parent), "PATH": "", "SYSTEMROOT": ""},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    d = out / "toy_seed0"
    assert sorted(p.name for p in d.iterdir()) == ["meta.json", "outings.json", "traces.parquet", "truth.json"]
    assert (out / "index.json").is_file()
    import pandas as pd
    tr = pd.read_parquet(d / "traces.parquet")
    assert list(tr.columns) == ["uid", "t", "lat", "lon"] and len(tr) > 0 and tr.uid.nunique() <= 6
    meta = json.loads((d / "meta.json").read_text())
    assert meta["variant"] == "toy" and meta["map"]["kind"] == "grid"
    # the output can be read back (eval side) and carries the planted truth
    from geolife.outside.synthetic_io import read_synthetic
    g = read_synthetic(d)
    assert set(g.result.traces) == set(g.result.outings)
    assert np.isfinite(sum(len(v) for v in g.result.traces.values()))

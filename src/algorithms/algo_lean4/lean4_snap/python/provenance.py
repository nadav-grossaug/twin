"""Fingerprints and dependency pins for provenance records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def code_fingerprint(paths: list[Path] | None = None) -> str:
    """SHA256 over (relative path, bytes) of the encoder's own sources."""
    if paths is None:
        paths = [ROOT / "python", ROOT / "dbt" / "models",
                 ROOT / "dbt" / "dbt_project.yml"]
    h = hashlib.sha256()
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(q for q in sorted(p.rglob("*"))
                         if q.is_file() and q.suffix in {".py", ".sql", ".yml"})
        else:
            files.append(p)
    for f in sorted(set(files)):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def dependency_versions() -> dict:
    deps = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ["torch", "torch_geometric", "duckdb", "pandas", "numpy",
                "networkx", "dbt-duckdb", "marimo", "altair"]:
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            deps[pkg] = "missing"
    tc = ROOT / "lean" / "lean-toolchain"
    if tc.exists():
        deps["lean-toolchain"] = tc.read_text().strip()
    return deps


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None

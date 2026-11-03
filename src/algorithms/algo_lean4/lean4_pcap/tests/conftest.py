"""Shared fixtures. Data-dependent tests skip gracefully when the pipeline
has not been built, so the suite degrades to unit tests on a fresh clone."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def con():
    import duckdb

    db = ROOT / "artifacts" / "enclave.duckdb"
    if not db.exists():
        pytest.skip("pipeline not built — run `make dbt`")
    c = duckdb.connect(str(db), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="session")
def lean_results():
    p = ROOT / "lean" / "out" / "lean_results.json"
    if not p.exists():
        pytest.skip("lean checks not run — run `make lean-checks`")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def lean_bin():
    b = ROOT / "lean" / ".lake" / "build" / "bin"
    if not (b / "release").exists():
        pytest.skip("lean executables not built — run `make lean-build`")
    return b

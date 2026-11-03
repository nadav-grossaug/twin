"""The concept spec: thresholds, units and conventions.

The single source of truth is the ``vars:`` block of dbt/dbt_project.yml.
This module reads it; lean/EnclaveFlows/Params.lean mirrors the numeric
values; tests/test_spec_sync.py locks all three together.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT = ROOT / "dbt" / "dbt_project.yml"


@lru_cache(maxsize=1)
def load_spec() -> dict:
    """Parse the dbt vars block without requiring PyYAML at runtime.

    The block is flat `key: value  # comment` lines between `vars:` and the
    next top-level key; values are strings, ints or floats.
    """
    text = DBT_PROJECT.read_text()
    m = re.search(r"^vars:\n((?:[ \t]+.*\n|\n)+?)^\S", text, re.M)
    if not m:
        raise ValueError(f"no vars block found in {DBT_PROJECT}")
    spec: dict = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith(("'", '"')):
            spec[key.strip()] = raw.strip("'\"")
        elif re.fullmatch(r"-?\d+", raw):
            spec[key.strip()] = int(raw)
        elif re.fullmatch(r"-?\d+\.\d+", raw):
            spec[key.strip()] = float(raw)
        else:
            spec[key.strip()] = raw  # jinja expressions stay as strings
    return spec


def spec_version() -> str:
    return str(load_spec()["spec_version"])


def spec_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(load_spec(), sort_keys=True).encode()
    ).hexdigest()[:16]

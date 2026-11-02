"""The concept spec: thresholds, units and conventions.

The single source of truth is the ``vars:`` block of dbt/dbt_project.yml
plus the raw-label->mode-class seed dbt/seeds/mode_map.csv. This module reads
both; lean/EnclaveGeo/Params.lean mirrors the values; tests/test_spec_sync.py
locks all three together.
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


# Public taxonomies (fixed orders — the Lean mirrors use the same orders)
MODE_CLASSES = ["walk", "bike", "bus", "car", "rail", "other"]
ROLES = ["home", "work", "other"]
MOTIF_CLASSES = ["home_day", "commute", "errand_loop", "tour", "irregular"]
DWELL_CLASSES = ["dw_lt1h", "dw_1_3h", "dw_3_8h", "dw_8_14h", "dw_ge14h"]
DEP_BINS = ["dep_night", "dep_morning", "dep_afternoon", "dep_evening"]
DIST_CLASSES = ["dist_lt1k", "dist_1_3k", "dist_3_10k", "dist_ge10k"]
SPD_BINS = ["spd_1", "spd_2", "spd_3", "spd_4_5", "spd_6p"]
SPEED_CLASSES = ["spd_still", "spd_walk", "spd_city", "spd_fast", "spd_vfast"]
TURN_BINS = [f"tb_{i}" for i in range(8)]

MODE_SEED = ROOT / "dbt" / "seeds" / "mode_map.csv"


@lru_cache(maxsize=1)
def mode_map() -> dict:
    """Raw GeoLife mode label -> public mode class, from the dbt seed."""
    out: dict = {}
    for line in MODE_SEED.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        raw_mode, mode_class = line.split(",")
        out[raw_mode.strip()] = mode_class.strip()
    return out


def mode_class_of(raw_mode: str) -> str:
    return mode_map().get(raw_mode, "XX")


def spec_version() -> str:
    return str(load_spec()["spec_version"])


def spec_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(load_spec(), sort_keys=True).encode()
    ).hexdigest()[:16]

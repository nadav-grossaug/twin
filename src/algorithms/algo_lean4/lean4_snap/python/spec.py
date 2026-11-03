"""The concept spec: thresholds, units and conventions.

The single source of truth is the ``vars:`` block of dbt/dbt_project.yml
plus the raw->public category seed dbt/seeds/category_map.csv. This module
reads both; lean/EnclaveGraph/Params.lean mirrors the values;
tests/test_spec_sync.py locks all three together.
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


# the 11 PUBLIC feature categories, fixed order (the DP cell grid order)
CATEGORIES = [
    "birthday", "education", "first_name", "gender", "hometown",
    "languages", "last_name", "locale", "location", "work", "other",
]

# the affiliation categories whose key-group structure crosses the release
# gate (DP size histogram + suppression-thresholded cohesion rollup). Chosen by
# measured shared-key EDGE coverage (locale 0.76, education 0.68, gender 0.55,
# birthday 0.14, hometown 0.13, location 0.09, work 0.05, languages 0.04,
# last_name 0.03); first_name/other are excluded (<= 0.001). Must match the
# `affil_cats` list in the two affiliation marts (sync-tested).
AFFIL_RELEASE_CATS = ["birthday", "education", "gender", "hometown", "languages",
                      "last_name", "locale", "location", "work"]

# the public affiliation group-size bins (lower edges in the vars block:
# affil_g5/affil_g9/affil_g17), fixed order
AFFIL_BINS = ["g_2_4", "g_5_8", "g_9_16", "g_17p"]

# the 14 raw SNAP featname category prefixes (ingest matches longest-first)
RAW_CATEGORIES = [
    "birthday", "education", "first_name", "gender", "hometown",
    "languages", "last_name", "locale", "location", "work",
    "middle_name", "name", "political", "religion",
]

CATEGORY_SEED = ROOT / "dbt" / "seeds" / "category_map.csv"


@lru_cache(maxsize=1)
def category_map() -> dict:
    """Raw SNAP featname category -> public category, from the dbt seed."""
    out: dict = {}
    for line in CATEGORY_SEED.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        raw, public = line.split(",")
        out[raw.strip()] = public.strip()
    return out


def category_of(raw: str) -> str:
    return category_map().get(raw, "other")


def spec_version() -> str:
    return str(load_spec()["spec_version"])


def spec_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(load_spec(), sort_keys=True).encode()
    ).hexdigest()[:16]

"""The concept spec: thresholds, units and conventions.

The single source of truth is the ``vars:`` block of dbt/dbt_project.yml
plus the tier seeds dbt/seeds/country_tiers.csv and
dbt/seeds/affiliate_tiers.csv. This module reads them; the Lean
EnclaveTrip.Params mirror the values; tests/test_spec_sync.py locks all
three together (frozen interfaces: wiki/booking-contract.md).

Time unit is DAYS (epoch day numbers; DATE 1970-01-01 = day 0). The public
window is window_start_day..window_end_day (exclusive).
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


# ---------------------------------------------------------------------------
# Public cell grids (frozen in wiki/booking-contract.md; every DP mart emits
# EXACTLY its grid, zero cells included, in this fixed order).
# ---------------------------------------------------------------------------
RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
CT_CELLS = [
    "Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
    "Borginia", "Kangan", "Bozatta", "Alvonia", "Holy Britannian Empire",
    "Axphain", "Santa Prisca", "Patusan", "Rolisica", "Kasnia", "ct_other",
]
TRANS_CELLS = [f"{a}__{b}" for a in CT_CELLS for b in CT_CELLS]
NIGHTS_CELLS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
GAP_CELLS = ["g_0", "g_1", "g_2_3", "g_4p"]
MONTH_CELLS = [f"m{i:02d}" for i in range(1, 13)]
DOW_CELLS = [f"d{i}" for i in range(7)]
DEVICE_CELLS = ["desktop", "mobile", "tablet"]
AFF_CELLS = ["a9924", "a359", "a384", "a9452", "a4541", "a8436",
             "a10332", "a3631", "aff_other"]
BOOKER_CELLS = ["Gondal", "The Devilfire Empire", "Elbonia",
                "Tcherkistan", "Bartovia"]
RR_BANDS = ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]

COUNTRY_TIER_SEED = ROOT / "dbt" / "seeds" / "country_tiers.csv"
AFFILIATE_TIER_SEED = ROOT / "dbt" / "seeds" / "affiliate_tiers.csv"


@lru_cache(maxsize=1)
def _country_tiers() -> dict:
    """hotel_country -> tier (the 15 named countries map to themselves;
    everything else maps to ct_other). From the dbt seed."""
    out: dict = {}
    for line in COUNTRY_TIER_SEED.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        country, tier = line.split(",")
        out[country.strip()] = tier.strip()
    return out


@lru_cache(maxsize=1)
def _affiliate_tiers() -> dict:
    """affiliate_id -> aff_tier (the 8 top affiliates map to a<id>;
    everything else maps to aff_other). From the dbt seed."""
    out: dict = {}
    for line in AFFILIATE_TIER_SEED.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        aff, tier = line.split(",")
        out[int(aff)] = tier.strip()
    return out


def ct_of(country: str) -> str:
    """Hotel-country tier of a hotel country (ct_other for every country
    outside the public 15)."""
    return _country_tiers().get(country, "ct_other")


def aff_tier_of(affiliate_id: int) -> str:
    """Affiliate tier of an affiliate id (aff_other outside the public 8)."""
    return _affiliate_tiers().get(int(affiliate_id), "aff_other")


def spec_version() -> str:
    return str(load_spec()["spec_version"])


def spec_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(load_spec(), sort_keys=True).encode()
    ).hexdigest()[:16]

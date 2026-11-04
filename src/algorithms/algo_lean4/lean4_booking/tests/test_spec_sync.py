"""The concept spec must agree across dbt vars, the tier seed, Python and
Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py) and the country-tier
seed. A failing test here means spec drift.
"""
import csv
import re
from pathlib import Path

from spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveTrip" / "Params.lean"

# Lean def -> (dbt var, scale): lean_value == round(dbt_value * scale)
LEAN_TO_DBT = {
    "rcE4": ("rc_e4", 1),
    "rcE5": ("rc_e5", 1),
    "rcE6": ("rc_e6", 1),
    "rcE7": ("rc_e7", 1),
    "rcE9": ("rc_e9", 1),
    "rcE13": ("rc_e13", 1),
    "nightsE4": ("nights_e4", 1),
    "nightsE7": ("nights_e7", 1),
    "gapE2": ("gap_e2", 1),
    "gapE4": ("gap_e4", 1),
    "tourMinLegs": ("tour_min_legs", 1),
    "tourMaxGapD": ("tour_max_gap_d", 1),
    "tourMinCountries": ("tour_min_countries", 1),
    "corridorMinRepeats": ("corridor_min_repeats", 1),
    "corridorCvMaxNum": ("corridor_cv_max_num", 1),
    "corridorCvMaxDen": ("corridor_cv_max_den", 1),
    "longstayMinNights": ("longstay_min_nights", 1),
    "seqbreakGapD": ("seqbreak_gap_d", 1),
    "rrBandWidthPct": ("rr_band_width_pct", 1),
    "topkCityMinUsers": ("topk_city_min_users", 1),
    "topkCityMinRes": ("topk_city_min_res", 1),
    "topkTransMinUsers": ("topk_trans_min_users", 1),
    "topkTransMinTrans": ("topk_trans_min_trans", 1),
    "dpClipTotal": ("dp_clip_total", 1),
    "dpCountClip": ("dp_count_clip", 1),
    "windowStartDay": ("window_start_day", 1),
    "windowEndDay": ("window_end_day", 1),
    "gapMax": ("gap_max_d", 1),
    "nightsMax": ("nights_max", 1),
    "rcMax": ("rc_max", 1),
    "rcDrawHi": ("rc_13p_draw_hi", 1),
    "tpDrawHi": ("tp_draw_hi", 1),
    "intertripGapMaxD": ("intertrip_gap_max_d", 1),
    "cptDrawHi": ("cpt_draw_hi", 1),
    "domesticStayScalePct": ("domestic_stay_scale_pct", 1),
}

# the public hotel-country tier grid (frozen in wiki/booking-contract.md)
CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
            "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan",
            "Rolisica", "Kasnia", "ct_other"]
AFF_CELLS = ["a9924", "a359", "a384", "a9452", "a4541", "a8436", "a10332",
             "a3631", "aff_other"]


def lean_defs():
    text = PARAMS.read_text()
    defs = {m[1]: int(m[2])
            for m in re.finditer(r"def (\w+) : Nat := (\d+)", text)}
    ver = re.search(r'def specVersion : String := "([^"]+)"', text)
    assert ver, "specVersion missing from Params.lean"
    return defs, ver.group(1)


def test_lean_params_match_dbt_vars():
    spec = load_spec()
    defs, _ = lean_defs()
    for lean_name, (var, scale) in LEAN_TO_DBT.items():
        assert lean_name in defs, f"{lean_name} missing from Params.lean"
        assert defs[lean_name] == round(float(spec[var]) * scale), (
            f"{lean_name}={defs[lean_name]} but dbt {var}={spec[var]} (x{scale})")


def test_spec_version_matches():
    _, ver = lean_defs()
    assert ver == str(load_spec()["spec_version"])


def test_every_lean_param_is_mapped():
    defs, _ = lean_defs()
    unmapped = set(defs) - set(LEAN_TO_DBT)
    assert not unmapped, f"Params.lean defs without a dbt mapping: {unmapped}"


def test_ct_grid_matches_lean():
    """Params.lean must mirror the public country-tier grid exactly."""
    text = PARAMS.read_text()
    m = re.search(r"def ctCells : List String := \[([^\]]+)\]", text)
    assert m, "ctCells list missing from Params.lean"
    lean_ct = re.findall(r'"([^"]+)"', m.group(1))
    assert lean_ct == CT_CELLS


def test_aff_grid_matches_lean():
    text = PARAMS.read_text()
    m = re.search(r"def affCells : List String := \[([^\]]+)\]", text)
    assert m, "affCells list missing from Params.lean"
    lean_aff = re.findall(r'"([^"]+)"', m.group(1))
    assert lean_aff == AFF_CELLS


def test_country_tier_seed_covers_the_named_tiers():
    """Every named tier must have exactly one seed row; ct_other may not
    (it is the fallback for unmapped countries)."""
    seed = ROOT / "dbt" / "seeds" / "country_tiers.csv"
    rows = list(csv.DictReader(open(seed)))
    tiers = [r["tier"] for r in rows]
    assert tiers == CT_CELLS[:-1], (
        f"seed tiers {tiers} != the 15 named public tiers")
    assert all(r["country"] == r["tier"] for r in rows), (
        "each named country must be its own tier")


def test_affiliate_tier_seed_matches_grid():
    seed = ROOT / "dbt" / "seeds" / "affiliate_tiers.csv"
    rows = list(csv.DictReader(open(seed)))
    tiers = [r["aff_tier"] for r in rows]
    assert tiers == AFF_CELLS[:-1], (
        f"seed aff tiers {tiers} != the 8 named public tiers")


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src


def test_no_sorry_in_lean_sources():
    lean = ROOT / "lean"
    offenders = []
    for p in list(lean.glob("*.lean")) + list((lean / "EnclaveTrip").glob("*.lean")):
        if re.search(r"\bsorry\b", p.read_text()):
            offenders.append(p.name)
    assert not offenders, f"sorry found in {offenders}"

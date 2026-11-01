"""The concept spec must agree across dbt vars, the region seed, Python and
Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py) and the region seed.
A failing test here means spec drift.
"""
import re
from pathlib import Path

from spec import REGIONS, load_spec, region_map

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveTravel" / "Params.lean"

# Lean def -> (dbt var, scale): lean_value == round(dbt_value * scale)
LEAN_TO_DBT = {
    "gapHopMaxS": ("gap_hop_max_s", 1),
    "gapTripMaxS": ("gap_trip_max_s", 1),
    "gapMonthlyMaxS": ("gap_monthly_max_s", 1),
    "gapQuarterlyMaxS": ("gap_quarterly_max_s", 1),
    "durDayMaxS": ("dur_day_max_s", 1),
    "durWeekMaxS": ("dur_week_max_s", 1),
    "durMonthMaxS": ("dur_month_max_s", 1),
    "durQuarterMaxS": ("dur_quarter_max_s", 1),
    "distShortMaxKm": ("dist_short_max_km", 1),
    "distMidMaxKm": ("dist_mid_max_km", 1),
    "distLongMaxKm": ("dist_long_max_km", 1),
    "tcE2": ("tc_e2", 1),
    "tcE3": ("tc_e3", 1),
    "tcE5": ("tc_e5", 1),
    "tcE9": ("tc_e9", 1),
    "tcE17": ("tc_e17", 1),
    "tcE33": ("tc_e33", 1),
    "episodeGapMaxS": ("episode_gap_max_s", 1),
    "epE4": ("ep_e4", 1),
    "epE6": ("ep_e6", 1),
    "corridorMinRepeats": ("corridor_min_repeats", 1),
    "corridorCvMaxNum": ("corridor_cv_max_num", 1),
    "corridorCvMaxDen": ("corridor_cv_max_den", 1),
    "tourMinHops": ("tour_min_hops", 1),
    "tourMaxGapS": ("tour_max_gap_s", 1),
    "tourMinCountries": ("tour_min_countries", 1),
    "burstMinTransfers": ("burst_min_transfers", 1),
    "burstWindowS": ("burst_window_s", 1),
    "longstayMinS": ("longstay_min_s", 1),
    "seqbreakGapS": ("seqbreak_gap_s", 1),
    "rrBandWidthPct": ("rr_band_width_pct", 1),
    "topkCorridorMinUsers": ("topk_corridor_min_users", 1),
    "topkCorridorMinTransfers": ("topk_corridor_min_transfers", 1),
    "dpClipTotal": ("dp_clip_total", 1),
    "dpCountClip": ("dp_count_clip", 1),
    "windowStartS": ("window_start_s", 1),
    "windowEndS": ("window_end_s", 1),
}


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


def test_region_list_matches_lean():
    text = PARAMS.read_text()
    m = re.search(r"def regions : List String := \[([^\]]+)\]", text)
    assert m, "regions list missing from Params.lean"
    lean_regions = re.findall(r'"(\w+)"', m.group(1))
    assert lean_regions == REGIONS


def test_region_map_matches_lean():
    """Every seed row must appear as a regionOf match arm and vice versa."""
    text = PARAMS.read_text()
    body = text.split("def regionOf", 1)[1]
    arms = dict(re.findall(r'\| "([A-Z]{2})" => "([A-Z]{2})"', body))
    assert arms == region_map(), (
        set(arms.items()) ^ set(region_map().items()))


def test_region_map_covers_the_data():
    import csv
    seen = set()
    with open(ROOT / "country_transfer.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            seen.add(row["prev_country"])
            seen.add(row["next_country"])
    missing = seen - set(region_map())
    assert not missing, f"countries without a region: {missing}"


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src


def test_no_sorry_in_lean_sources():
    lean = ROOT / "lean"
    offenders = []
    for p in list(lean.glob("*.lean")) + list((lean / "EnclaveTravel").glob("*.lean")):
        if re.search(r"\bsorry\b", p.read_text()):
            offenders.append(p.name)
    assert not offenders, f"sorry found in {offenders}"

"""The concept spec must agree across dbt vars, Python and Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py), with explicit unit
scalings. A failing test here means spec drift.
"""
import re
from fractions import Fraction
from pathlib import Path

from spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveFlows" / "Params.lean"

# Lean def -> (dbt var, scale): lean_value == round(dbt_value * scale)
LEAN_TO_DBT = {
    "beaconMinFlows": ("beacon_min_flows", 1),
    "beaconMinPeriodMs": ("beacon_min_period_s", 1000),
    "fanoutMinDsts": ("fanout_min_dsts", 1),
    "returnMaxGapS": ("return_max_gap_s", 1),
    "sessionGapS": ("session_gap_s", 1),
    "dpClip": ("dp_clip", 1),
    "dpClipTotal": ("dp_clip_total", 1),
    "entropyHighCentibits": ("entropy_high", 100),
    "chainMaxGapMs": ("chain_max_gap_s", 1000),
    "tlsRareMaxFlows": ("tls_rare_max_flows", 1),
    "dnsMaxLabelLen": ("dns_max_label_len", 1),
    "dnsMaxNameLen": ("dns_max_name_len", 1),
    "dnsSubfanMinNames": ("dns_subfan_min_names", 1),
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
            f"{lean_name}={defs[lean_name]} but dbt {var}={spec[var]} (×{scale})")


def test_cv_threshold_fraction():
    spec = load_spec()
    defs, _ = lean_defs()
    cv = Fraction(str(spec["beacon_max_cv"]))
    assert (defs["beaconMaxCvNum"], defs["beaconMaxCvDen"]) == (
        cv.numerator, cv.denominator)


def test_spec_version_matches():
    _, ver = lean_defs()
    assert ver == str(load_spec()["spec_version"])


def test_every_lean_param_is_mapped():
    defs, _ = lean_defs()
    unmapped = set(defs) - set(LEAN_TO_DBT) - {"beaconMaxCvNum", "beaconMaxCvDen"}
    assert not unmapped, f"Params.lean defs without a dbt mapping: {unmapped}"


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src

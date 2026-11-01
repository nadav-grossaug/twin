"""The concept spec must agree across dbt vars, Python and Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py). A failing test here
means spec drift.
"""
import re
from pathlib import Path

from spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveSeries" / "Params.lean"

# Lean def -> (dbt var, scale): lean_value == round(dbt_value * scale)
LEAN_TO_DBT = {
    "retFlatBp": ("ret_flat_bp", 1),
    "retBigBp": ("ret_big_bp", 1),
    "shockBp": ("shock_bp", 1),
    "runMinDays": ("run_min_days", 1),
    "drawdownNum": ("drawdown_num", 1),
    "drawdownDen": ("drawdown_den", 1),
    "vrecovDropNum": ("vrecov_drop_num", 1),
    "vrecovDropDen": ("vrecov_drop_den", 1),
    "vrecovLevelNum": ("vrecov_level_num", 1),
    "vrecovLevelDen": ("vrecov_level_den", 1),
    "vrecovMaxDays": ("vrecov_max_days", 1),
    "volburstWindow": ("volburst_window", 1),
    "volburstMinBp": ("volburst_min_bp", 1),
    "gapMaxDays": ("gap_max_days", 1),
    "volCalmMaxBp": ("vol_calm_max_bp", 1),
    "volNormalMaxBp": ("vol_normal_max_bp", 1),
    "volElevatedMaxBp": ("vol_elevated_max_bp", 1),
    "dpClip": ("dp_clip", 1),
    "dpClipTotal": ("dp_clip_total", 1),
    "dpComoveClip": ("dp_comove_clip", 1),
    "comoveLoPct": ("comove_lo_pct", 1),
    "comoveHiPct": ("comove_hi_pct", 1),
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


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src


def test_no_sorry_in_lean_sources():
    lean = ROOT / "lean"
    offenders = []
    for p in list(lean.glob("*.lean")) + list((lean / "EnclaveSeries").glob("*.lean")):
        if re.search(r"\bsorry\b", p.read_text()):
            offenders.append(p.name)
    assert not offenders, f"sorry found in {offenders}"

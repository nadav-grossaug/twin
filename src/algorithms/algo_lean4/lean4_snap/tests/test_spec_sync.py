"""The concept spec must agree across dbt vars, the category seed, Python and
Lean Params.lean.

Sync is test-enforced, not generated: this file regex-parses the Lean source
and compares against the dbt vars (via python/spec.py) and the category seed.
A failing test here means spec drift.
"""
import re
from pathlib import Path

import pytest

from spec import CATEGORIES, category_map, load_spec

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "lean" / "EnclaveGraph" / "Params.lean"

# Lean def -> (dbt var, scale): lean_value == round(dbt_value * scale)
LEAN_TO_DBT = {
    "degE2": ("deg_e2", 1),
    "degE3": ("deg_e3", 1),
    "degE5": ("deg_e5", 1),
    "degE9": ("deg_e9", 1),
    "degE17": ("deg_e17", 1),
    "degE33": ("deg_e33", 1),
    "degE65": ("deg_e65", 1),
    "ccDen": ("cc_den", 1),
    "triE1": ("tri_e1", 1),
    "triE11": ("tri_e11", 1),
    "triE101": ("tri_e101", 1),
    "triE1001": ("tri_e1001", 1),
    "triE10001": ("tri_e10001", 1),
    "richE8": ("rich_e8", 1),
    "circE4": ("circ_e4", 1),
    "csizeE25": ("csize_e25", 1),
    "csizeE51": ("csize_e51", 1),
    "csizeE101": ("csize_e101", 1),
    "csizeE201": ("csize_e201", 1),
    "csizeE401": ("csize_e401", 1),
    "nCommunityRanks": ("n_community_ranks", 1),
    "hubDegreeMin": ("hub_degree_min", 1),
    "leafDegree": ("leaf_degree", 1),
    "brokerDegreeMin": ("broker_degree_min", 1),
    "brokerCcMaxNum": ("broker_cc_max_num", 1),
    "brokerCcMaxDen": ("broker_cc_max_den", 1),
    "bridgeCommunitiesMin": ("bridge_communities_min", 1),
    "butterflyCirclesMin": ("butterfly_circles_min", 1),
    "coreDegreeMin": ("core_degree_min", 1),
    "coreCcMinNum": ("core_cc_min_num", 1),
    "coreCcMinDen": ("core_cc_min_den", 1),
    "suppressMinNodes": ("suppress_min_nodes", 1),
    "dpClipTotal": ("dp_clip_total", 1),
    "dpCountClip": ("dp_count_clip", 1),
    "dpCsizeClip": ("dp_csize_clip", 1),
    "dpClipMotifs": ("dp_clip_motifs", 1),
    "louvainSeed": ("louvain_seed", 1),
    "synthClosureTolPct": ("synth_closure_tol_pct", 1),
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


def test_category_list_matches_lean():
    text = PARAMS.read_text()
    m = re.search(r"def categories : List String :=\s*\n?\s*\[([^\]]+)\]",
                  text)
    assert m, "categories list missing from Params.lean"
    lean_cats = re.findall(r'"(\w+)"', m.group(1))
    assert lean_cats == CATEGORIES


def test_category_map_matches_lean():
    """Every seed row must appear as a categoryOf match arm and vice versa."""
    text = PARAMS.read_text()
    body = text.split("def categoryOf", 1)[1].split("end EnclaveGraph", 1)[0]
    arms = dict(re.findall(r'\| "(\w+)" => "(\w+)"', body))
    assert arms == category_map(), (
        set(arms.items()) ^ set(category_map().items()))


def test_category_map_covers_the_data(con):
    """Every raw category the ingest produced must be in the seed."""
    seen = {r[0] for r in con.execute(
        "select distinct raw_category from stg_node_features").fetchall()}
    missing = seen - set(category_map())
    assert not missing, f"raw categories without a public mapping: {missing}"


def test_community_ranks_match_lean():
    text = PARAMS.read_text()
    m = re.search(r"def communityRanks : List String :=\s*\n?\s*\[([^\]]+)\]",
                  text)
    assert m, "communityRanks list missing from Params.lean"
    lean_ranks = re.findall(r'"(\w+)"', m.group(1))
    assert lean_ranks == [f"c{i}" for i in range(1, 9)] + ["rest"]
    assert len(lean_ranks) - 1 == int(load_spec()["n_community_ranks"])


def test_export_script_uses_spec():
    # the export bridge must read the spec, not hardcode thresholds
    src = (ROOT / "python" / "export_lean_data.py").read_text()
    assert "load_spec" in src


def test_no_sorry_in_lean_sources():
    lean = ROOT / "lean"
    offenders = []
    for p in list(lean.glob("*.lean")) + list((lean / "EnclaveGraph").glob("*.lean")):
        if p.name == "GenCert.lean":
            continue  # generated; checked by its own build
        if re.search(r"\bsorry\b", p.read_text()):
            offenders.append(p.name)
    assert not offenders, f"sorry found in {offenders}"


def test_no_native_decide_in_certificates():
    p = ROOT / "lean" / "GenCert.lean"
    if not p.exists():
        pytest.skip("no certificate generated yet")
    assert "native_decide" not in p.read_text()

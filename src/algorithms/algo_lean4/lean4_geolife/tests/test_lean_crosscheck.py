"""SQL <-> Lean agreement: every exported observation must be decided
identically by the dbt SQL and the formal Lean definitions — plus a LIVE
recomputation of a sample of checks straight from the warehouse."""
import json
import re
from collections import Counter

import pytest


@pytest.fixture(scope="module")
def results(lean_results):
    return lean_results


KEYS = ["motif_days", "revisits", "dwell_classes", "hour_bins", "dist_classes",
        "speed_classes", "spd_bins", "longdwells", "visit_seqs", "assoc",
        "seg_classes", "ja_classes", "circ_classes"]


@pytest.mark.parametrize("key", KEYS)
def test_agreement(results, key):
    rows = results[key]
    assert rows, f"no {key} observations exported"
    bad = [r for r in rows if not r.get("agree", False)]
    assert not bad, f"{len(bad)}/{len(rows)} {key} disagreements: {bad[:3]}"


def test_agreement_totals(results):
    agg = results["agreement"]
    assert agg["n_checked"] > 0
    assert agg["n_disagree"] == 0
    assert agg["n_agree"] == agg["n_checked"]


def test_claims_ledger_shape(results):
    claims = {c["name"]: c["status"] for c in results["claims"]}
    assert claims, "empty claims ledger"
    bad = {n: s for n, s in claims.items() if s not in ("proved", "assumed")}
    assert not bad, f"claims with an unknown status: {bad}"
    # the mechanism, sampler and generator contracts must be proved
    for name in ("privClippedCell_PureDP", "histMech_DP", "sampleCellPMF_law"):
        assert claims.get(name) == "proved", name
    # the route-shape ladder contracts must be proved
    for name in ("jaRange_is_junction", "segRange_resolvable"):
        assert claims.get(name) == "proved", name
    gen = [n for n, s in claims.items()
           if s == "proved" and ("mkStays" in n or "genStay" in n
                                 or "genTrip" in n or "sorted" in n.lower())]
    assert gen, "no proved generator-contract claim in the ledger"
    # the trust boundary is explicit
    for name in ("sql_counts_correct", "entropy_source", "io_wrappers"):
        assert claims.get(name) == "assumed", name


def test_no_sorry_or_native_decide_in_lean_sources(root):
    lean = root / "lean"
    offenders = []
    for p in list(lean.glob("*.lean")) + list((lean / "EnclaveGeo").glob("*.lean")):
        # comments may legitimately SAY "no sorry, no native_decide"
        code = re.sub(r"/-.*?-/", "", p.read_text(), flags=re.S)
        code = re.sub(r"--[^\n]*", "", code)
        if re.search(r"\bsorry\b", code):
            offenders.append(f"{p.name}: sorry")
        if re.search(r"\bnative_decide\b", code):
            offenders.append(f"{p.name}: native_decide")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# LIVE recomputation: trust but verify the exported observations against the
# warehouse they claim to summarize.
# ---------------------------------------------------------------------------

def test_motif_mix_matches_warehouse(results, con):
    """The exported per-day motif labels must reproduce the feat_user_days
    motif-class mix exactly (same rows, recounted independently)."""
    exported = Counter(r["sql_motif"] for r in results["motif_days"])
    warehouse = Counter(dict(con.execute(
        "select motif_class, count(*) from feat_user_days group by 1"
    ).fetchall()))
    assert exported == warehouse


def test_seg_class_sample_matches_warehouse(results, con):
    """The exported route-segment observations (data/motifs.json "segs" ->
    lean_results seg_classes) must reproduce a direct recount of the SAME
    sample from feat_route_segments — identical ORDER BY / LIMIT as
    export_lean_data.py, recounted independently down to the floored
    seg_m values."""
    exported = Counter((r["seg_m"], r["sql_class"])
                       for r in results["seg_classes"])
    warehouse = Counter((int(m), c) for (m, c) in con.execute(
        "select floor(seg_m)::bigint, seg_class from feat_route_segments "
        "order by user_id, trajectory_id, sequence_uuid, junction_idx "
        "limit 2000").fetchall())
    assert exported == warehouse


def test_assoc_chi2_matches_warehouse(root, con):
    """The chi2 statistics shipped to Lean (data/assoc.json) must equal the
    mart_feature_assoc values the SQL computed, recomputed here from the db."""
    p = root / "lean" / "data" / "assoc.json"
    if not p.exists():
        pytest.skip("lean data export missing — run `make train`")
    pairs = json.loads(p.read_text())["pairs"]
    assert pairs, "no association pairs exported"
    db = {(a, b): chi2 for a, b, chi2 in con.execute(
        "select feat_a, feat_b, chi2 from mart_feature_assoc").fetchall()}
    for pr in pairs:
        key = (pr["feat_a"], pr["feat_b"])
        assert key in db, key
        assert abs(float(pr["sql_chi2"]) - db[key]) < 1e-6, (
            key, pr["sql_chi2"], db[key])


def test_assoc_observed_counts_match_warehouse(root, con):
    """Deeper: the observed contingency cells of one exported pair must equal
    a direct recount of the underlying columns."""
    p = root / "lean" / "data" / "assoc.json"
    if not p.exists():
        pytest.skip("lean data export missing — run `make train`")
    pairs = json.loads(p.read_text())["pairs"]
    pair = next((x for x in pairs
                 if (x["feat_a"], x["feat_b"]) == ("role", "dwell_class")), None)
    if pair is None:
        pytest.skip("role x dwell_class pair not exported")
    got = {(a, b): n
           for i, a in enumerate(pair["vals_a"])
           for j, b in enumerate(pair["vals_b"])
           for n in [pair["observed"][i][j]]}
    want = {(a, b): n for a, b, n in con.execute(
        "select role, dwell_class, count(*) from feat_visits "
        "where role is not null and dwell_class is not null "
        "and not (arr_censored or dep_censored) "
        "group by 1, 2").fetchall()}
    # the export may zero-fill cells the warehouse has no rows for
    for key, n in want.items():
        assert got.get(key, 0) == n, (key, got.get(key), n)
    for key, n in got.items():
        assert want.get(key, 0) == n, (key, n, want.get(key))

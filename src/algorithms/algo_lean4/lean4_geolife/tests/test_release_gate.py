"""Release gate: default deny, exact-fraction ledger, attestation failure
modes, egress containment, suppression thresholds, policy-filtered export."""
import csv
import json
from fnmatch import fnmatch
from fractions import Fraction

import pytest

from enclave import (AttestationError, DevNoEnclave, EgressGuard, SimulatedTEE,
                     Verifier, new_nonce)
from release import (check_lifetime_budget, check_release_budget, epsilon_ledger,
                     export_relation, is_allowed, lifetime_spent, load_policy,
                     read_ledger, reconcile_ledger, void_export)


@pytest.fixture(scope="module")
def policy():
    return load_policy()


# ---------------------------------------------------------------- ledger
def test_ledger_within_budget(policy):
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) <= Fraction(ledger["epsilon_max"])


def test_ledger_exactly_saturated(policy):
    # the per-export budget is spent exactly: a new query must displace an
    # old one at review time (or the cap is raised deliberately)
    ledger = epsilon_ledger(policy)
    assert Fraction(ledger["epsilon_total"]) == Fraction(
        policy["dp_total_epsilon_max"])


def test_over_budget_refused(policy):
    bad = json.loads(json.dumps(policy))
    bad["dp_releases"]["dp_motif"]["epsilon_per_cell"] = "2/1"
    with pytest.raises(PermissionError):
        epsilon_ledger(bad)


# ------------------------------------------- lifetime ledger (across exports)
def _spend(eid, eps, sha="d1"):
    return {"kind": "spend", "export_id": eid, "data_sha256": sha, "epsilon": eps}


def test_policy_declares_lifetime_cap(policy):
    cap = Fraction(policy["dp_lifetime_epsilon_max"])
    assert cap >= Fraction(policy["dp_total_epsilon_max"])


def test_lifetime_accumulates_per_dataset():
    entries = [_spend("a", "4/1"), _spend("b", "1/2"), _spend("c", "4/1", "d2")]
    assert lifetime_spent(entries, "d1") == Fraction(9, 2)
    assert lifetime_spent(entries, "d2") == Fraction(4)


def test_second_release_of_same_data_refused():
    entries = [_spend("a", "4/1")]
    with pytest.raises(PermissionError, match="lifetime"):
        check_lifetime_budget(entries, "d1", Fraction(4), Fraction(4))
    # a DIFFERENT dataset (other users) has its own budget
    ok = check_lifetime_budget(entries, "d2", Fraction(4), Fraction(4))
    assert ok["spent_after"] == "4/1"


def test_void_restores_budget_and_is_audited(tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.write_text(json.dumps(_spend("a", "4/1")) + "\n")
    with pytest.raises(ValueError):
        void_export("a", "   ", led)              # reason required
    with pytest.raises(KeyError):
        void_export("nope", "never left", led)    # only recorded exports
    void_export("a", "never left the enclave", led)
    with pytest.raises(ValueError):
        void_export("a", "again", led)            # no double void
    entries = read_ledger(led)
    assert entries[-1]["kind"] == "void" and entries[-1]["reason"]
    assert lifetime_spent(entries, "d1") == 0
    assert len(entries) == 2                      # append-only: spend kept


def test_reconcile_records_unledgered_exports(tmp_path):
    led = tmp_path / "ledger.jsonl"
    rel = tmp_path / "release"
    for eid in ("20260101T000000Z", "20260102T000000Z"):
        (rel / eid).mkdir(parents=True)
        (rel / eid / "manifest.json").write_text(json.dumps(
            {"export_id": eid, "dp_epsilon": {"epsilon_total": "4/1"}}))
    assert reconcile_ledger(led, rel, "d1") == 2
    assert reconcile_ledger(led, rel, "d1") == 0   # idempotent
    assert lifetime_spent(read_ledger(led), "d1") == Fraction(8)
    # deleting an export directory never refunds its spend
    import shutil
    shutil.rmtree(rel / "20260101T000000Z")
    assert lifetime_spent(read_ledger(led), "d1") == Fraction(8)


def test_corrupt_ledger_refuses(tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.write_text(json.dumps(_spend("a", "4/1")) + "\n{not json\n")
    with pytest.raises(PermissionError, match="corrupt"):
        read_ledger(led)


# ----------------------------------------------------------- default deny
@pytest.mark.parametrize("relation", [
    "stg_points", "feat_steps", "feat_legs", "feat_staypoints", "feat_anchors",
    "feat_visits", "feat_user_days", "mart_user_stats", "mart_motif_daily",
    "mart_motif_tours", "mart_motif_revisit", "mart_staypoint_stats",
    "mart_assoc_cells", "mart_seq_transitions", "mart_seq_chains",
    "mart_dp_spd", "mart_dp_distance", "ml_geo_scores",
])
def test_default_deny(policy, relation):
    assert not is_allowed(relation, policy)
    assert any(fnmatch(relation, pat) for pat in policy["denied"]), (
        f"{relation} missing from the audit list")


def test_allowed_artifacts(policy):
    for rel in ("mart_noise_moments", "mart_kinematics", "mart_turn_hist",
                "mart_dist_mix", "q_data_integrity"):
        assert is_allowed(rel, policy), rel


def test_model_weights_denied(policy):
    assert policy["model_weights"] == "deny_unless_reviewed"


# --------------------------------------------- the Lean release vs policy
AUX_OF = {"dp_transitions": "transitions", "dp_dwell": "dwell",
          "dp_departure": "departure", "dp_mode": "mode"}


def test_dp_release_grids_match_lean_output(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    motif = policy["dp_releases"]["dp_motif"]
    assert [c["cell"] for c in rel["cells"]] == motif["grid"]
    assert Fraction(rel["eps1"], rel["eps2"]) == Fraction(motif["epsilon_per_cell"])
    assert rel["clip"] == motif["clip_l1_per_user"]
    by_name = {h["name"]: h for h in rel["aux"]}
    for pol_name, aux_name in AUX_OF.items():
        spec = policy["dp_releases"][pol_name]
        aux = by_name[aux_name]
        assert [c["cell"] for c in aux["cells"]] == spec["grid"], pol_name
        assert Fraction(aux["eps1"], aux["eps2"]) == Fraction(
            spec["epsilon_per_cell"]), pol_name
        assert aux["clip"] == spec["clip_l1_per_user"], pol_name


def test_lean_budget_matches_policy_ledger(root, policy):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    res = check_release_budget(epsilon_ledger(policy), p)
    assert res["matches_policy"] is True
    assert Fraction(res["spent"]) == Fraction(policy["dp_total_epsilon_max"])


def test_tampered_budget_refused(root, policy, tmp_path):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    rel["budget"]["spent_num"] = int(rel["budget"]["spent_num"]) + 1
    bad = tmp_path / "dp_release.json"
    bad.write_text(json.dumps(rel))
    with pytest.raises(PermissionError):
        check_release_budget(epsilon_ledger(policy), bad)


# ------------------------------------------------------------ attestation
KEY = b"test-hardware-key"


def _verifier(measurement):
    return Verifier(trusted_keys={"simulated-tee": KEY},
                    expected_measurements={measurement})


def test_attestation_roundtrip():
    tee = SimulatedTEE(KEY, measurement="good-code")
    nonce = new_nonce()
    claims = _verifier("good-code").verify(tee.evidence(nonce), nonce)
    assert claims.attested is True
    assert claims.platform == "simulated-tee"
    assert claims.measurement == "good-code"


def test_wrong_key_refused():
    tee = SimulatedTEE(b"attacker-key", measurement="good-code")
    nonce = new_nonce()
    with pytest.raises(AttestationError, match="signature"):
        _verifier("good-code").verify(tee.evidence(nonce), nonce)


def test_nonce_replay_refused():
    tee = SimulatedTEE(KEY, measurement="good-code")
    stale = tee.evidence(new_nonce())
    with pytest.raises(AttestationError, match="nonce"):
        _verifier("good-code").verify(stale, new_nonce())   # a fresh nonce


def test_unexpected_measurement_refused():
    tee = SimulatedTEE(KEY, measurement="evil-code")
    nonce = new_nonce()
    with pytest.raises(AttestationError, match="measurement"):
        _verifier("good-code").verify(tee.evidence(nonce), nonce)


def test_untrusted_platform_refused():
    tee = SimulatedTEE(KEY, measurement="good-code")
    nonce = new_nonce()
    verifier = Verifier(trusted_keys={}, expected_measurements={"good-code"})
    with pytest.raises(AttestationError, match="untrusted"):
        verifier.verify(tee.evidence(nonce), nonce)


def test_dev_export_requires_explicit_env(policy, monkeypatch):
    from release import attest
    env = policy["allow_dev_platform_env"]
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(AttestationError):
        attest(policy, DevNoEnclave(), None)
    monkeypatch.setenv(env, "1")
    claims = attest(policy, DevNoEnclave(), None)
    assert claims["attested"] is False   # a dev export is marked, never attested


# ------------------------------------------------------------------ egress
def test_egress_containment(tmp_path):
    guard = EgressGuard(tmp_path / "release")
    inside = guard.check(tmp_path / "release" / "x" / "y.json")
    assert inside == (tmp_path / "release" / "x" / "y.json").resolve()
    with pytest.raises(PermissionError):
        guard.check(tmp_path / "elsewhere" / "y.json")
    with pytest.raises(PermissionError):
        guard.check(tmp_path / "release" / ".." / "escape.json")


# ------------------------------------------------------------- suppression
def test_suppression_thresholds_applied(con, policy, tmp_path):
    spec = policy["artifacts"]["kinematics"]
    out = tmp_path / "kinematics.csv"
    n_rows, suppressed = export_relation(con, spec, out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_rows and suppressed >= 0
    assert rows and list(rows[0].keys()) == spec["columns"]
    for rule in spec["suppress"].values():
        col, val = rule["column"], int(rule["value"])
        for r in rows:
            assert float(r[col]) >= val, (col, r)
    total = con.execute(f"select count(*) from {spec['relation']}").fetchone()[0]
    assert n_rows + suppressed == total


# ----------------------------------------------- route-shape artifacts (new)
ROUTE_ARTIFACTS = ("route_shape", "seg_hist", "junction_hist")
ROUTE_POLICY_SKIP = ("release policy predates the route-shape artifacts — "
                     "update config/release_policy.yaml / run `make train`")


def _require_route_policy(policy):
    missing = [a for a in ROUTE_ARTIFACTS if a not in policy.get("artifacts", {})]
    if missing:
        pytest.skip(ROUTE_POLICY_SKIP)


def test_route_artifacts_allowed(policy):
    _require_route_policy(policy)
    for rel in ("mart_route_shape", "mart_seg_hist", "mart_junction_hist"):
        assert is_allowed(rel, policy), rel


@pytest.mark.parametrize("relation", [
    "feat_route_points", "feat_route_segments", "feat_route_legs",
])
def test_route_feature_tables_denied(policy, relation):
    # default deny holds regardless of the policy version ...
    assert not is_allowed(relation, policy)
    # ... and the updated policy must also carry the audit listing
    _require_route_policy(policy)
    assert any(fnmatch(relation, pat) for pat in policy["denied"]), (
        f"{relation} missing from the audit list")


@pytest.mark.parametrize("name", ROUTE_ARTIFACTS)
def test_route_suppression_thresholds_applied(con, policy, tmp_path, name):
    """The three new category-2 CSVs obey their suppression thresholds and
    the exported+suppressed rows account for the whole mart."""
    _require_route_policy(policy)
    spec = policy["artifacts"][name]
    assert spec.get("enabled", False), name
    assert spec.get("suppress"), f"{name} must carry suppression thresholds"
    out = tmp_path / f"{name}.csv"
    n_rows, suppressed = export_relation(con, spec, out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_rows and suppressed >= 0
    assert rows and list(rows[0].keys()) == spec["columns"]
    for rule in spec["suppress"].values():
        col, val = rule["column"], int(rule["value"])
        for r in rows:
            assert float(r[col]) >= val, (name, col, r)
    total = con.execute(f"select count(*) from {spec['relation']}").fetchone()[0]
    assert n_rows + suppressed == total


# ----------------------------------------------------------- exported files
def _latest_export(root):
    rel_dirs = sorted((root / "release").glob("*/")) if (root / "release").exists() else []
    if not rel_dirs:
        pytest.skip("no export yet — run `make train`")
    return rel_dirs[-1]


def test_released_files_carry_no_user_ids(root):
    latest = _latest_export(root)
    files = list(latest.glob("*.json")) + list(latest.glob("*.csv"))
    assert files
    for f in files:
        txt = f.read_text()
        assert "user_id" not in txt, f.name
        assert "trajectory_id" not in txt, f.name
        assert "latitude" not in txt and "lat_c" not in txt, f.name


def test_export_manifest_ledger(root, policy):
    latest = _latest_export(root)
    man = json.loads((latest / "manifest.json").read_text())
    if man["policy_version"] != policy["policy_version"]:
        pytest.skip(f"latest export {latest.name} predates policy "
                    f"{policy['policy_version']} — run `make export`")
    assert Fraction(man["dp_epsilon"]["epsilon_total"]) == Fraction(
        policy["dp_total_epsilon_max"])
    assert man["dp_epsilon"]["lean_release_budget"]["matches_policy"] is True
    assert Fraction(man["dp_epsilon"]["lifetime"]["spent_after"]) <= Fraction(
        policy["dp_lifetime_epsilon_max"])
    for name in man["files"]:
        assert (latest / name).exists(), name


def test_lean_export_bridge_reads_policy_epsilon(policy):
    """The (eps1, eps2) and budget handed to `lake exe release` come from the
    policy itself — no hand-kept mirror that can drift."""
    import export_lean_data as bridge
    for name, spec in policy["dp_releases"].items():
        assert Fraction(*bridge.LEDGER[name]) == Fraction(spec["epsilon_per_cell"]), name
    assert bridge.BUDGET == Fraction(policy["dp_total_epsilon_max"])

"""Release-gate tests: attestation, restricted egress, policy-gated export,
DP budget enforcement and the default-deny relation list."""
import csv
import json
import re
from pathlib import Path

import pytest

import enclave as E
import release as REL

HW = b"simulated-hardware-root"
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def make_verifier(meas):
    return E.Verifier({"simulated-tee": HW}, {meas})


def attested_pair():
    from provenance import code_fingerprint
    meas = code_fingerprint()
    return E.SimulatedTEE(HW, meas), make_verifier(meas)


# --- attestation -----------------------------------------------------------

def test_attestation_happy_path():
    tee = E.SimulatedTEE(HW, "m1")
    n = E.new_nonce()
    claims = make_verifier("m1").verify(tee.evidence(n), n)
    assert claims.attested and claims.platform == "simulated-tee"


@pytest.mark.parametrize("case", ["replay", "measurement", "platform", "signature"])
def test_attestation_failures(case):
    tee = E.SimulatedTEE(HW, "m1")
    n = E.new_nonce()
    ev = tee.evidence(n)
    v = make_verifier("m1")
    with pytest.raises(E.AttestationError):
        if case == "replay":
            v.verify(ev, E.new_nonce())
        elif case == "measurement":
            make_verifier("other").verify(ev, n)
        elif case == "platform":
            v.verify(E.Evidence("unknown", "m1", n, ev.signature), n)
        else:
            E.Verifier({"simulated-tee": b"wrong"}, {"m1"}).verify(ev, n)


# --- restricted egress -----------------------------------------------------

def test_egress_guard(tmp_path):
    g = E.EgressGuard(tmp_path / "release")
    g.check(tmp_path / "release" / "x.csv")
    with pytest.raises(PermissionError):
        g.check(tmp_path / "elsewhere.csv")
    with pytest.raises(PermissionError):
        g.check(tmp_path / "release" / ".." / "escape.csv")


# --- export gate -----------------------------------------------------------

def test_export_refused_without_dev_override(monkeypatch, tmp_path):
    monkeypatch.delenv("FLOW_ALLOW_DEV_RELEASE", raising=False)
    with pytest.raises(E.AttestationError):
        REL.export(E.DevNoEnclave(), None, release_root=tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_attested_export_respects_policy(con, tmp_path):
    tee, ver = attested_pair()
    out = REL.export(tee, ver, release_root=tmp_path / "release")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["attestation"]["attested"] is True
    assert manifest["policy_version"] == "release-1.3.0"

    pol = REL.load_policy()
    dp_files = {Path(s).name for spec in pol["dp_releases"].values() for s in spec["sources"]}
    enabled = {f"{n}.csv" if "relation" in a else f"{n}.json"
               for n, a in pol["artifacts"].items() if a.get("enabled", False)}
    assert set(manifest["files"]) == dp_files | enabled
    assert {p.name for p in out.iterdir()} == dp_files | enabled | {"manifest.json"}

    # suppression holds and nothing host-level leaves in flow_stats
    rows = list(csv.DictReader(open(out / "flow_stats.csv")))
    assert rows and all(int(r["n_flows"]) >= 10 for r in rows)
    assert manifest["files"]["flow_stats.csv"]["suppressed_rows"] >= 0
    assert not {"sa", "da", "src_ip", "dst_ip", "src", "dst", "mac"} & set(rows[0])
    assert not any(IP_RE.search(v) for r in rows for v in r.values())

    # lean_claims: claims + aggregate counts only, no per-channel rows / IPs
    claims = json.loads((out / "lean_claims.json").read_text())
    assert set(claims) == {"claims", "agreement_counts"}
    assert not IP_RE.search(json.dumps(claims))

    # exact-fraction epsilon ledger within budget
    assert manifest["dp_epsilon"]["epsilon_total"] == "389/100"
    assert manifest["dp_epsilon"]["epsilon_max"] == "4/1"
    for name, meta in manifest["files"].items():
        assert len(meta["sha256"]) == 64, name


def test_budget_enforced(monkeypatch, tmp_path):
    pol = REL.load_policy()
    pol["dp_total_epsilon_max"] = "1/10"
    monkeypatch.setattr(REL, "load_policy", lambda *a, **k: pol)
    tee, ver = attested_pair()
    with pytest.raises(PermissionError, match="budget"):
        REL.export(tee, ver, release_root=tmp_path / "release")
    assert not (tmp_path / "release").exists()


# --- default deny ----------------------------------------------------------

def test_denied_relations():
    pol = REL.load_policy()
    for rel in ["feat_flows", "ml_flow_scores", "mart_dns", "mart_motif_beacon",
                "mart_motif_fanout", "mart_motif_return", "mart_assoc_cells",
                "mart_seq_sessions"]:
        assert rel in pol["denied"]
        assert not REL.is_allowed(rel, pol)
    assert REL.is_allowed("mart_flow_stats", pol)

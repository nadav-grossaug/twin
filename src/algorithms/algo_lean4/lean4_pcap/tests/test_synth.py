"""Synthetic-data layer: the generator declares its assumptions, the plan's
beacons are all detected, validate's exit codes map to its report, and the
certificate is kernel-checked. Tests skip gracefully when the corresponding
artifact (synth/, validation.json, certificate.json) has not been produced."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

SYNTH = ROOT / "synth"


def _run(*args, **kw):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "synth.py"), *args],
                          capture_output=True, text=True, **kw)


def _load(name: str) -> dict:
    p = SYNTH / name
    if not p.exists():
        pytest.skip(f"{p} not produced — run `make synth` (and certify)")
    return json.loads(p.read_text())


def test_meta_marks_assumptions_vs_released_facts():
    meta = _load("meta.json")
    a = meta["assumptions"]
    for k in ("temporal", "record_validity", "tls", "dns", "fields_not_modelled"):
        assert a[k], k
    rf = meta["released_facts"]
    for k in ("service_mix", "transitions", "tls_stack_mix", "dns_shape_mix"):
        assert "genspec" in rf[k]["source"], k
    assert meta["genspec"]["sha256"]
    assert "FlowShape" in a["temporal"]
    assert "service_mix" in meta["released_facts"]
    assert meta["release"]["dp_release_sha256"]
    assert "sampleCellPMF" in meta["sampler"]


def test_planned_beacons_all_detected():
    v = _load("validation.json")
    rt = v["roundtrip"]
    assert rt["planned_beacons"] >= 1
    assert rt["beacon_recall"] == 1.0, rt
    assert v["lean"]["planned_beacons_lean_false"] == []
    assert v["lean"]["sessions_untimely"] == []


def test_validate_exit_codes_map_to_report():
    if not (SYNTH / "plan.json").exists():
        pytest.skip("no synth/ — run `make synth`")
    r = _run("validate", "--synth", str(SYNTH))
    v = json.loads((SYNTH / "validation.json").read_text())
    assert r.returncode in (0, 2), r.stdout[-2000:] + r.stderr[-2000:]
    assert r.returncode == v["exit"]
    assert (r.returncode == 0) == v["passed"]
    if r.returncode == 2:                       # soft: tolerance only, never a Lean/spec violation
        assert v["lean"]["passed"]


def test_validate_schema_violation_exits_1(tmp_path):
    if not (SYNTH / "plan.json").exists():
        pytest.skip("no synth/ — run `make synth`")
    work = tmp_path / "bad"
    work.mkdir()
    for f in ("plan.json", "meta.json"):
        shutil.copy(SYNTH / f, work)
    lines = (SYNTH / "flow_synth.jsonl").read_text().splitlines()
    import re as _re
    lines[1] = _re.sub(r'"pr":\s*(6|17)\b', '"pr":99', lines[1])
    (work / "flow_synth.jsonl").write_text("\n".join(lines) + "\n")
    r = _run("validate", "--synth", str(work))
    assert r.returncode == 1, "a dbt schema-test failure must be a hard exit 1"
    assert not json.loads((work / "validation.json").read_text())["passed"]


def test_certificate_is_kernel_checked():
    c = _load("certificate.json")
    assert c["proved"] is True
    assert c["kernel_only"] is True
    assert c["n_beacons"] >= 1 and c["n_sessions"] >= 1
    assert set(c["axioms"]) == {"beacons_valid", "sessions_timely"}

"""Distillation: the genspec is pure post-processing of released values —
every weight equals a released value or released + the DECLARED rule."""
import hashlib
import json

import pytest

CLASSES = ["crash", "down", "flat", "up", "surge"]
VOLS = ["calm", "normal", "elevated", "burst"]
DOWS = ["mon", "tue", "wed", "thu", "fri"]


@pytest.fixture(scope="module")
def rel_and_spec(root):
    rel_p = root / "lean" / "out" / "dp_release.json"
    gs_p = root / "lean" / "out" / "genspec.json"
    if not rel_p.exists() or not gs_p.exists():
        pytest.skip("run `make lean-release distill` first")
    rel = json.loads(rel_p.read_text())
    gs = json.loads(gs_p.read_text())
    if gs["source"]["dp_release_sha256"] != hashlib.sha256(
            rel_p.read_bytes()).hexdigest():
        pytest.skip("genspec predates the current (non-reproducible) release — "
                    "run `make distill` after `make lean`")
    aux = {h["name"]: {c["cell"]: int(c["released"]) for c in h["cells"]}
           for h in rel["aux"]}
    return rel, gs, aux


def test_classes_equal_release(rel_and_spec):
    rel, gs, _ = rel_and_spec
    released = {c["cell"]: max(0, int(c["released"])) for c in rel["cells"]}
    for cell in gs["classes"]["cells"]:
        assert cell["w"] == released[cell["name"]]


def test_transition_rows_follow_declared_rule(rel_and_spec):
    rel, gs, aux = rel_and_spec
    trans = aux["class_transition"]
    released_mix = {c["cell"]: max(0, int(c["released"])) for c in rel["cells"]}
    mixtot = max(1, sum(released_mix.values()))
    for row in gs["transitions"]:
        p = row["given"]
        raw = {n: max(0, trans.get(f"{p}>{n}", 0)) for n in CLASSES}
        rowsum = max(1, sum(raw.values()))
        for cell in row["cells"]:
            n = cell["name"]
            want = raw[n] + round(2 * rowsum * released_mix[n] / mixtot)
            if n == p:
                want += 2 * rowsum
            assert cell["w"] == want, (p, n)


def test_dow_rows_follow_declared_rescale(rel_and_spec):
    _, gs, aux = rel_and_spec
    dow = aux["dow_class"]
    for row in gs["dow"]:
        w = row["given"]
        raw = {n: max(0, dow.get(f"{w}|{n}", 0)) for n in CLASSES}
        tot = sum(raw.values())
        for cell in row["cells"]:
            want = 2 if tot == 0 else round(10 * raw[cell["name"]] / tot)
            assert cell["w"] == want, (w, cell["name"])


def test_vol_and_mag_equal_release(rel_and_spec):
    _, gs, aux = rel_and_spec
    vol = aux["vol_transition"]
    for row in gs["vol_transitions"]:
        for cell in row["cells"]:
            assert cell["w"] == max(0, vol.get(f"{row['given']}>{cell['name']}", 0))
    mag = aux["mag_bins"]
    for cell in gs["mag_bins"]["cells"]:
        assert cell["w"] == max(0, mag.get(cell["name"], 0))


def test_comove_lambda_follows_declared_rule(rel_and_spec):
    _, gs, aux = rel_and_spec
    com = aux["comove"]
    mids = {"comove_lo": 20, "comove_mid": 55, "comove_hi": 85}
    tot = sum(max(0, com.get(k, 0)) for k in mids)
    share = (sum(max(0, com.get(k, 0)) * m for k, m in mids.items()) / tot
             if tot else 50.0)
    assert gs["comove_lambda_pct"] == max(0, min(100, round(2 * share - 100)))


def test_fidelity_report_accepted(root):
    p = root / "artifacts" / "reports" / "distill_report.json"
    if not p.exists():
        pytest.skip("run `make distill` first")
    rep = json.loads(p.read_text())
    assert rep.get("accepted") is True, rep
    assert rep["student_accuracy"] >= rep["repeat_last_accuracy"] - 0.02

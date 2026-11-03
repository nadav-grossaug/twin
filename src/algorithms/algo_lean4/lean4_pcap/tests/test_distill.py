"""Distillation tests: the genspec is pure post-processing of released
artifacts (provenance), and the distilled semi-Markov student meets the
declared acceptance criterion vs its teacher/baseline (fidelity).
Skips gracefully when `make distill` has not been run.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"


@pytest.fixture(scope="module")
def genspec():
    p = OUT / "genspec.json"
    if not p.exists():
        pytest.skip("no genspec — run `make distill`")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def release():
    return json.loads((OUT / "dp_release.json").read_text())


def test_genspec_provenance(genspec, release):
    """Every weight in the spec equals a released value — the spec is
    post-processing, nothing enters it outside the gate. Only comparable
    against the release the spec was distilled from (DP draws are
    intentionally non-reproducible, so a newer release means re-distill)."""
    import hashlib
    cur = hashlib.sha256((OUT / "dp_release.json").read_bytes()).hexdigest()
    if genspec["source"]["dp_release_sha256"] != cur:
        pytest.skip("genspec predates the current dp_release — run `make distill`")
    aux = {h["name"]: {c["cell"]: int(c["released"]) for c in h["cells"]}
           for h in release.get("aux", [])}
    cells = {c["cell"]: int(c["released"]) for c in release["cells"]}
    for c in genspec["services"]["cells"]:
        assert c["w"] == max(0, cells[c["name"]])
    for row in genspec["transitions"]:
        released = {n: max(0, aux["svc_transition"].get(f"{row['given']}>{n}", 0))
                    for n in [c["name"] for c in row["cells"]]}
        boost = max(1, sum(released.values()))
        for c in row["cells"]:
            expected = released[c["name"]] + (boost if c["name"] == row["given"] else 0)
            assert c["w"] == expected, (row["given"], c)
    for c in genspec["tls_stacks"]["cells"]:
        assert c["w"] == max(0, aux["tls_stack"][c["name"]])
    topk = {n["qname"]: n["n_queries"]
            for n in json.loads((OUT / "dns_topk.json").read_text())["names"]}
    for c in genspec["qname_topk"]:
        assert c["w"] == topk[c["name"]]


def test_genspec_shape(genspec):
    svcs = [c["name"] for c in genspec["services"]["cells"]]
    assert svcs == ["http", "https", "dns", "ntp", "chat", "ephemeral", "other"]
    assert len(genspec["transitions"]) == 7
    assert all(len(r["cells"]) == 7 for r in genspec["transitions"])
    assert genspec["genspec_version"]
    assert "dp_release_sha256" in genspec["source"]


def test_distill_fidelity_accepted():
    p = ROOT / "artifacts" / "reports" / "distill_report.json"
    if not p.exists():
        pytest.skip("no distill report — run `make distill` with the warehouse built")
    ns = json.loads(p.read_text()).get("next_service")
    if ns is None:
        pytest.skip("warehouse was absent at distill time")
    assert ns["n_val_pairs"] > 100
    # declared acceptance criterion: the distilled state machine may trail the
    # repeat-last baseline by at most the declared margin
    assert ns["accepted"], (ns["student_markov_acc"], ns["repeat_last_acc"])

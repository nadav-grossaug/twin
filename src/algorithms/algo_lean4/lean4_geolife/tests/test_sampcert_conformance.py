"""SampCert conformance: the compiled executables' randomness matches the
proved laws.

- catsample: chi-square GOF of empirical draws against the weight law
  P(i) = w_i / sum(w)  (theorem sampleCellPMF_law).
- release: with a huge per-cell epsilon the noised counts equal the clipped
  counts (discrete Laplace concentrates); an over-budget input is refused
  with exit 2 and no output file.
"""
import json
import subprocess

import pytest


def chi2_stat(obs, exp):
    return sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)


def test_catsample_gof(lean_bin):
    n = 20000
    weights = [5, 3, 2]
    r = subprocess.run([str(lean_bin / "catsample")],
                       input=f"{n}\n5,3,2\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    draws = [int(x) for x in r.stdout.split()]
    assert len(draws) == n
    tot = sum(weights)
    obs = [draws.count(i) for i in range(3)]
    exp = [n * w / tot for w in weights]
    # df = 2, alpha = 0.001 -> critical value 13.82
    assert chi2_stat(obs, exp) < 13.82, (obs, exp)


def _release_input(eps1, eps2, budget_num, budget_den, counts):
    return {
        "spec_version": "4.0.0",
        "privacy_unit": "one user (test fixture)",
        "clip": 1, "eps1": eps1, "eps2": eps2,
        "budget_num": budget_num, "budget_den": budget_den,
        "cells": [{"cell": f"c{i}", "clipped_count": c}
                  for i, c in enumerate(counts)],
        "aux": [],
    }


def test_release_huge_epsilon_recovers_counts(lean_bin, tmp_path):
    counts = [17, 0, 42]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dp_counts.json").write_text(
        json.dumps(_release_input(100000, 1, 10 ** 9, 1, counts)))
    r = subprocess.run([str(lean_bin / "release")], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rel = json.loads((tmp_path / "out" / "dp_release.json").read_text())
    got = [c["noised_count"] for c in rel["cells"]]
    assert got == counts


def test_release_refuses_over_budget(lean_bin, tmp_path):
    (tmp_path / "data").mkdir()
    # 3 cells at 2/1 = 6 > budget 4/1
    (tmp_path / "data" / "dp_counts.json").write_text(
        json.dumps(_release_input(2, 1, 4, 1, [1, 2, 3])))
    r = subprocess.run([str(lean_bin / "release")], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert not (tmp_path / "out" / "dp_release.json").exists()


def test_production_release_within_budget(root):
    p = root / "lean" / "out" / "dp_release.json"
    if not p.exists():
        pytest.skip("no release yet — run `make train`")
    rel = json.loads(p.read_text())
    led = rel["budget"]
    assert led["within_budget"] is True
    assert led["spent_num"] / led["spent_den"] <= 4.0


def test_production_samples_track_release(root):
    """The sampler run's weights must be the released motif counts (clamped
    at 0) and every draw must land on a declared cell."""
    sp = root / "lean" / "out" / "samples.json"
    rp = root / "lean" / "out" / "dp_release.json"
    if not sp.exists() or not rp.exists():
        pytest.skip("no sampler run yet — run `make train`")
    samples = json.loads(sp.read_text())
    rel = json.loads(rp.read_text())
    released = {c["cell"]: max(0, int(c["noised_count"])) for c in rel["cells"]}
    got = {c["cell"]: c["weight"] for c in samples["cells"]}
    assert got == released
    assert samples["total_weight"] == sum(released.values())
    assert sum(c["draws"] for c in samples["cells"]) == samples["n_draws"]

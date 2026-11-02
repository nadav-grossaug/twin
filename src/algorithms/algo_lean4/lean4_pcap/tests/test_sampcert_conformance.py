"""Conformance of the SampCert-backed executables: tests the WIRING between
Python-visible inputs and the proved mechanisms (the proofs themselves live
in Lean; see lean/out/lean_results.json claims).

Each test runs the real binaries in a scratch cwd with crafted inputs.
"""
import json
import math
import subprocess

import pytest


def run_bin(lean_bin, name, cwd, args=()):
    return subprocess.run([str(lean_bin / name), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def write_counts(tmp, cells, clip=4, e1=1, e2=5, budget=99.0):
    (tmp / "data").mkdir(exist_ok=True)
    (tmp / "data" / "dp_counts.json").write_text(json.dumps({
        "clip": clip, "eps1": e1, "eps2": e2, "budget_total_eps": budget,
        "cells": [{"cell": name, "clipped_count": c, "n_users": 1}
                  for name, c in cells],
    }))


def test_huge_epsilon_recovers_exact_counts(lean_bin, tmp_path):
    # scale = clip*eps2/eps1 = 4/10^6: discrete Laplace noise is 0 w.p. ~1,
    # so the release must equal the clipped counts exactly.
    cells = [("a", 17), ("b", 0), ("c", 231)]
    write_counts(tmp_path, cells, e1=10**6, e2=1, budget=10**9)
    r = run_bin(lean_bin, "release", tmp_path)
    assert r.returncode == 0, r.stderr
    rel = json.loads((tmp_path / "out" / "dp_release.json").read_text())
    got = {c["cell"]: c["released"] for c in rel["cells"]}
    assert got == dict(cells)


def test_budget_refusal(lean_bin, tmp_path):
    # 3 cells × 1/5 = 0.6 > budget 0.5 -> refuse with exit 2, write nothing
    write_counts(tmp_path, [("a", 1), ("b", 2), ("c", 3)], budget=0.5)
    r = run_bin(lean_bin, "release", tmp_path)
    assert r.returncode == 2
    assert "REFUSED" in r.stderr
    assert not (tmp_path / "out" / "dp_release.json").exists()


def test_sampler_chi_square(lean_bin, tmp_path):
    # draws from the verified sampler must match the proved law w_i/sum(w)
    weights = {"w": 5, "x": 10, "y": 25, "z": 60}
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "dp_release.json").write_text(json.dumps({
        "cells": [{"cell": k, "released": v} for k, v in weights.items()]}))
    n = 20000
    r = run_bin(lean_bin, "sample", tmp_path, args=(str(n),))
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "out" / "samples.json").read_text())
    tot = sum(weights.values())
    chi2 = sum((c["draws"] - n * c["weight"] / tot) ** 2 / (n * c["weight"] / tot)
               for c in s["cells"])
    dof = len(weights) - 1
    # Wilson–Hilferty p-value approximation; reject only below 0.001
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    assert p > 0.001, f"chi2={chi2:.2f} on {dof} dof (p={p:.2e})"


def test_zero_weight_never_drawn(lean_bin, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "dp_release.json").write_text(json.dumps({
        "cells": [{"cell": "a", "released": 50},
                  {"cell": "zero", "released": 0},
                  {"cell": "b", "released": 50}]}))
    r = run_bin(lean_bin, "sample", tmp_path, args=("2000",))
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "out" / "samples.json").read_text())
    zero = next(c for c in s["cells"] if c["cell"] == "zero")
    assert zero["draws"] == 0

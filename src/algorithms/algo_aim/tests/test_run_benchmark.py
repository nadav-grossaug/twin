"""CLI wiring. Uses Synthcity's fast ``marginal_distributions`` plugin on the toy spec."""

import json

import pandas as pd
import pytest

from synth_bench import data, run_benchmark
from synth_bench.datasets import _REGISTRY


@pytest.fixture
def toy_registered(spec, monkeypatch):
    monkeypatch.setitem(_REGISTRY, spec.name, spec)
    monkeypatch.setattr(run_benchmark, "list_datasets", lambda: [spec.name])
    return spec


def test_parse_methods():
    avail = ["a", "b", "c"]
    assert run_benchmark.parse_methods("all", avail) == avail
    assert run_benchmark.parse_methods("a, c", avail) == ["a", "c"]
    with pytest.raises(SystemExit):
        run_benchmark.parse_methods("a,zzz", avail)


def test_load_splits_deterministic(toy_registered):
    a = data.load_splits("toy", 400, seed=1)
    b = data.load_splits("toy", 400, seed=1)
    assert len(a[0]) == 400 and a[0].equals(b[0]) and a[1].equals(b[1])


def test_cli_end_to_end_and_resume(toy_registered, tmp_path):
    args = ["--dataset", "toy", "--methods", "marginal_distributions", "--n-train", "300",
            "--results-dir", str(tmp_path), "--no-synthcity-metrics"]
    assert run_benchmark.main(args) == 0
    out = tmp_path / "marginal_distributions"
    rec = json.loads((out / "metrics.json").read_text())
    assert rec["status"] == "ok" and rec["n_synth"] == 300
    assert len(pd.read_csv(out / "synthetic.csv")) == 300
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert list(summary["method"]) == ["marginal_distributions"]

    mtime = (out / "metrics.json").stat().st_mtime_ns
    assert run_benchmark.main([*args, "--resume"]) == 0
    assert (out / "metrics.json").stat().st_mtime_ns == mtime  # skipped


def test_unknown_method_rejected(toy_registered, tmp_path):
    with pytest.raises(SystemExit):
        run_benchmark.main(["--dataset", "toy", "--methods", "nope", "--results-dir", str(tmp_path)])


def test_failing_method_is_recorded_not_raised(toy_registered, tmp_path, monkeypatch):
    from synth_bench import generators

    def boom(*args, **kwargs):
        raise RuntimeError("kaput")

    monkeypatch.setattr(generators, "fit_generate", boom)
    train, test = data.load_splits("toy", 200, seed=0)
    rec = run_benchmark.run_method("marginal_distributions", toy_registered, train, test,
                                   tmp_path / "m", 200, 0, False)
    assert rec["status"] == "error" and "kaput" in rec["error"]
    summary = run_benchmark.write_summary(tmp_path)
    assert summary.loc[0, "status"] == "error"

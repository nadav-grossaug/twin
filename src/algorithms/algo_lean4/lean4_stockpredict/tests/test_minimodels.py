"""Minimodel discipline: content-addressed immutable artifacts, explicit
adoption gates with honest negatives, registry activation."""
import json
import os
import stat

import pytest


@pytest.fixture(scope="module")
def artifact(root):
    reg_p = root / "models_out" / "registry.json"
    if not reg_p.exists():
        pytest.skip("run `make train` first")
    reg = json.loads(reg_p.read_text())
    active = reg.get("minimodels", {}).get("active")
    assert active, "no active minimodels artifact"
    vdir = root / "models_out" / "minimodels" / active
    return vdir, json.loads((vdir / "meta.json").read_text())


def test_artifact_files_read_only(artifact):
    vdir, _ = artifact
    for f in vdir.iterdir():
        mode = stat.S_IMODE(os.stat(f).st_mode)
        assert not (mode & stat.S_IWUSR), f"{f.name} is writable"


def test_gates_recorded_for_every_model(artifact):
    _, meta = artifact
    for name in ("return_ae", "next_class", "seasonality", "garch", "factor"):
        assert name in meta, name
        assert "adopted" in meta[name], name


def test_gates_are_honest(artifact):
    """Adoption flags must match the recorded metrics (honest negatives)."""
    _, meta = artifact
    nc = meta["next_class"]["metrics"]
    assert meta["next_class"]["adopted"] == (
        nc["val_accuracy"] >= nc["repeat_last_accuracy"]
        and nc["val_accuracy"] >= nc["majority_accuracy"])
    se = meta["seasonality"]["metrics"]
    assert meta["seasonality"]["adopted"] == (
        se["val_accuracy"] >= se["majority_accuracy"])
    fa = meta["factor"]["metrics"]
    assert meta["factor"]["adopted"] == (fa["var_explained_f1"] >= 0.15)


def test_garch_per_ticker_params_recorded(artifact):
    _, meta = artifact
    per = meta["garch"]["per_ticker"]
    assert len(per) >= 2
    for t, p in per.items():
        assert 0 <= p["alpha"] and 0 <= p["beta"], t
        assert p["alpha"] + p["beta"] < 1.0, f"{t} not stationary"


def test_release_status_unreviewed(artifact):
    _, meta = artifact
    assert meta.get("release_status") == "internal_unreviewed"


def test_ml_scores_never_retrain(root):
    src = (root / "dbt" / "models" / "ml" / "ml_series_scores.py").read_text()
    assert "train" not in src.lower().replace("untrained", "").replace(
        "retrain", "RT"), "inference node must not train"
    assert "untrained" in src   # placeholder path exists

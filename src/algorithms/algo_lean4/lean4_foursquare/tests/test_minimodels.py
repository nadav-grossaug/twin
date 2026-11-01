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
    for name in ("next_region", "gap_class", "episode_shape", "profile_ae"):
        assert name in meta, name
        assert "adopted" in meta[name], name


def test_gates_are_honest(artifact):
    """Adoption flags must match the recorded metrics (honest negatives).
    The return-to-previous baseline is brutal (72.1% at country level) —
    a losing model with adopted=False is a finding, not a failure."""
    _, meta = artifact
    nr = meta["next_region"]["metrics"]
    assert meta["next_region"]["adopted"] == (
        nr["val_accuracy"] >= nr["return_prev_accuracy"])
    gc = meta["gap_class"]["metrics"]
    assert meta["gap_class"]["adopted"] == (
        gc["val_accuracy"] >= gc["majority_accuracy"])
    ep = meta["episode_shape"]["metrics"]
    if ep["val_nll"] is None:
        assert meta["episode_shape"]["adopted"] is False
    else:
        assert meta["episode_shape"]["adopted"] == (
            ep["val_nll"] <= ep["unigram_nll"])


def test_user_split_not_time_split(artifact):
    """Leakage guard: the split must be on user id, not time."""
    _, meta = artifact
    assert "user" in str(meta["split"]) and "hash" in str(meta["split"])


def test_release_status_unreviewed(artifact):
    _, meta = artifact
    assert meta.get("release_status") == "internal_unreviewed"


def test_ml_scores_never_retrain(root):
    src = (root / "dbt" / "models" / "ml" / "ml_travel_scores.py").read_text()
    assert "train" not in src.lower().replace("untrained", "").replace(
        "retrain", "RT"), "inference node must not train"
    assert "untrained" in src   # placeholder path exists

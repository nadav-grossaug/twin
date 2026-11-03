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


def test_artifact_has_all_models(artifact):
    vdir, _ = artifact
    for name in ("next_country.pt", "trip_len_class.pt", "stay_nights.pt",
                 "device_class.pt", "profile_ae.pt",
                 "trip_count.pt", "device_seq.pt", "aff_seq.pt",
                 "meta.json"):
        assert (vdir / name).exists(), name


def test_gates_recorded_for_every_model(artifact):
    _, meta = artifact
    for name in ("next_country", "trip_len_class", "stay_nights",
                 "device_class", "profile_ae",
                 "trip_count", "device_seq", "aff_seq"):
        assert name in meta, name
        assert "adopted" in meta[name], name


def test_gates_are_honest(artifact):
    """Adoption flags must match the recorded metrics (honest negatives).
    The return-to-previous baseline is brutal at country-tier level — a
    losing model with adopted=False is a finding, not a failure."""
    _, meta = artifact
    nc = meta["next_country"]["metrics"]
    assert meta["next_country"]["adopted"] == (
        nc["val_accuracy"] >= nc["return_prev_accuracy"])
    for name in ("trip_len_class", "stay_nights", "device_class",
                 "trip_count"):
        m = meta[name]["metrics"]
        assert meta[name]["adopted"] == (
            m["val_accuracy"] >= m["majority_accuracy"]), name
    for name, base_key in (("device_seq", "stay_same_device_accuracy"),
                           ("aff_seq", "stay_same_tier_accuracy")):
        m = meta[name]["metrics"]
        assert meta[name]["adopted"] == (
            m["val_accuracy"] >= m[base_key]), name


def test_user_split_not_time_split(artifact):
    """Leakage guard: the split must be on user id, not time."""
    _, meta = artifact
    assert "user" in str(meta["split"]) and "hash" in str(meta["split"])


def test_release_status_unreviewed(artifact):
    _, meta = artifact
    assert meta.get("release_status") == "internal_unreviewed"


def test_ml_scores_never_retrain(root):
    src = (root / "dbt" / "models" / "ml" / "ml_trip_scores.py").read_text()
    assert "train" not in src.lower().replace("untrained", "").replace(
        "retrain", "RT"), "inference node must not train"
    assert "untrained" in src   # placeholder path exists

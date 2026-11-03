"""Minimodel discipline: content-addressed immutable artifacts, explicit
adoption gates with honest negatives, registry activation."""
import json
import os
import stat

import pytest

MODELS = ("degree_sage", "community_gnn", "triad_vgae", "featcond_encoder",
          "ego_motif_pool",
          # affiliation-community models (2026-11-03)
          "affil_ae", "affil_link_mlp", "key_community_mlp",
          "key_cohesion_mlp")


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
    for name in MODELS:
        assert name in meta, name
        assert "adopted" in meta[name], name
        assert "gate" in meta[name], name


def test_gates_are_honest(artifact):
    """Adoption flags must match the recorded metrics (honest negatives).
    At transitivity 0.52 the common-neighbor link-prediction baseline is
    brutal — a losing triad_vgae with adopted=False is a finding, not a
    failure."""
    _, meta = artifact
    ds = meta["degree_sage"]["metrics"]
    assert meta["degree_sage"]["adopted"] == (
        ds["val_accuracy"] > ds["majority_accuracy"] + 0.02)
    cg = meta["community_gnn"]["metrics"]
    assert meta["community_gnn"]["adopted"] == (cg["ari_vs_louvain"] > 0.30)
    tv = meta["triad_vgae"]["metrics"]
    assert meta["triad_vgae"]["adopted"] == (
        tv["val_auc"] > tv["common_neighbor_auc"] + 0.01)
    fc = meta["featcond_encoder"]["metrics"]
    assert meta["featcond_encoder"]["adopted"] == (fc["mean_val_auc"] > 0.55)
    mp = meta["ego_motif_pool"]["metrics"]
    assert meta["ego_motif_pool"]["adopted"] == (
        mp["macro_f1"] > mp["baseline_macro_f1"])


def test_affiliation_gates_are_honest(artifact):
    """The four affiliation models: adoption flags must match the recorded
    metrics; negatives are findings, kept."""
    _, meta = artifact
    aa = meta["affil_ae"]["metrics"]
    assert meta["affil_ae"]["adopted"] == (
        aa["val_bce"] < aa["marginal_bce"] - 0.001)
    al = meta["affil_link_mlp"]["metrics"]
    assert meta["affil_link_mlp"]["adopted"] == (
        al["val_auc"] > al["total_shared_auc"] + 0.01)
    kc = meta["key_community_mlp"]["metrics"]
    assert meta["key_community_mlp"]["adopted"] == (
        kc["val_accuracy"] > kc["majority_accuracy"] + 0.02)
    kh = meta["key_cohesion_mlp"]["metrics"]
    assert meta["key_cohesion_mlp"]["adopted"] == (
        kh["val_mae"] < kh["mean_baseline_mae"])


def test_affil_ae_key_universe_recorded(artifact):
    """The affil_ae input universe is pinned: key-list file present and its
    sha recorded in the metrics (content addressing of the input space)."""
    vdir, meta = artifact
    assert (vdir / "affil_keys.json").exists()
    keys = json.loads((vdir / "affil_keys.json").read_text())
    assert len(keys) == meta["affil_ae"]["metrics"]["n_keys"]
    import hashlib
    assert (hashlib.sha256("\n".join(keys).encode()).hexdigest()
            == meta["affil_ae"]["metrics"]["keys_sha256"])


def test_node_split_not_time_split(artifact):
    """Leakage guard: the split must be on node id hash."""
    _, meta = artifact
    assert "node" in str(meta["split"]) and "hash" in str(meta["split"])


def test_release_status_unreviewed(artifact):
    _, meta = artifact
    assert meta.get("release_status") == "internal_unreviewed"


def test_ml_scores_never_retrain(root):
    src = (root / "dbt" / "models" / "ml" / "ml_graph_scores.py").read_text()
    # the real property: no gradient steps / fitting happens at inference
    for token in (".backward(", "optim.", "optimizer", ".fit(", "train_loop"):
        assert token not in src, f"inference node must not train ({token})"
    assert "untrained" in src   # placeholder path exists

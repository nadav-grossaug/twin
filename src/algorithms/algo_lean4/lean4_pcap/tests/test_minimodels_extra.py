"""Ten additional minimodels (2026-10-30): artifact entries, adoption-gate
consistency, weight blobs and the new flow-level score columns.

Mirrors test_tls.py / test_dns.py: skips gracefully when the registry or the
warehouse is missing; honest negatives (adopted: false) are valid outcomes —
only the CONSISTENCY of the flag with the recorded baseline is asserted."""
import json

import pytest

# entry -> (metric key, baseline key, direction of the adoption gate)
GATED = {
    "volume_service": ("val_accuracy", "majority_baseline", "gt"),
    "bytes_in": ("val_mae_log", "baseline_mae_log", "lt"),
    "duration_band": ("val_accuracy", "majority_baseline", "gt"),
    "pair_gap": ("val_mae_log_gap", "repeat_last_mae", "lt"),
    "session_len": ("val_accuracy", "majority_baseline", "gt"),
    "unanswered": ("val_accuracy", "majority_baseline", "gt"),
    "tls_setup": ("val_accuracy", "majority_baseline", "gt"),
    "dns_kind": ("val_accuracy", "majority_baseline", "gt"),
    "beacon_channel": ("val_accuracy", "majority_baseline", "gt"),
}
BLOBS = [
    "bytedist_ae.pt", "volume_service.pt", "bytes_in.pt", "duration_band.pt",
    "pair_gap.pt", "session_len.pt", "unanswered.pt", "tls_setup.pt",
    "dns_kind.pt", "beacon_channel.pt",
]
NEW_COLUMNS = {
    "bd_recon_score", "pred_service_vol", "pred_service_vol_confidence",
    "bytes_in_abs_err", "pred_unanswered_conf",
}


@pytest.fixture(scope="session")
def active(root):
    reg = root / "models_out" / "registry.json"
    if not reg.exists():
        pytest.skip("no model registry — run `make train`")
    version = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not version:
        pytest.skip("no active minimodels artifact")
    vdir = root / "models_out" / "minimodels" / version
    return vdir, json.loads((vdir / "meta.json").read_text())


def test_all_ten_entries_present(active):
    _, meta = active
    missing = [k for k in list(GATED) + ["bytedist_ae"] if k not in meta]
    assert not missing, f"active artifact predates the ten additions: {missing}"
    for k in list(GATED) + ["bytedist_ae"]:
        assert "metrics" in meta[k] and "arch" in meta[k]


def test_adoption_gates_consistent_with_baselines(active):
    _, meta = active
    for key, (mk, bk, op) in GATED.items():
        m = meta[key]["metrics"]
        assert {mk, bk, "adopted", "n_train", "n_val"} <= set(m), key
        expect = m[mk] > m[bk] if op == "gt" else m[mk] < m[bk]
        assert m["adopted"] == expect, (key, m[mk], m[bk], m["adopted"])


def test_bytedist_ae_is_ungated_anomaly_scorer(active):
    _, meta = active
    m = meta["bytedist_ae"]["metrics"]
    assert {"val_mse_mean", "val_mse_p95", "n_train", "n_val"} <= set(m)
    assert "adopted" not in m  # unsupervised — no adoption gate (like the AE)


def test_weight_blobs_present(active):
    vdir, _ = active
    missing = [b for b in BLOBS if not (vdir / b).exists()]
    assert not missing, missing


def test_new_flow_scores_materialized(con):
    cols = {r[1] for r in con.execute(
        "pragma table_info('ml_flow_scores')").fetchall()}
    assert NEW_COLUMNS <= cols, NEW_COLUMNS - cols
    model_version = con.execute(
        "select any_value(model_version) from ml_flow_scores").fetchone()[0]
    if model_version == "untrained":
        pytest.skip("scores materialized without an active artifact")
    counts = con.execute(
        "select count(bd_recon_score), count(pred_service_vol),"
        " count(bytes_in_abs_err), count(pred_unanswered_conf)"
        " from ml_flow_scores").fetchone()
    assert all(n > 0 for n in counts), counts

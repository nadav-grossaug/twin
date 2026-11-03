"""Minimodel discipline: content-addressed immutable artifacts, explicit
adoption gates with honest negatives, registry activation, and an inference
node that never retrains."""
import importlib.util
import json
import os
import stat

import pytest

MODELS = ("next_role", "dwell_class", "mode_leg", "profile_ae")
WEIGHT_FILES = ("next_role.pt", "dwell_class.pt", "mode_leg.pt", "profile_ae.pt")
SCORE_COLS = ["user_id", "sp_id", "anomaly_score", "pred_next_role",
              "pred_next_confidence", "p_next_role_correct",
              "pred_dwell_class", "weak_label_role", "model_version"]


@pytest.fixture(scope="module")
def artifact(root):
    reg_p = root / "models_out" / "registry.json"
    if not reg_p.exists():
        pytest.skip("run `make train` first")
    reg = json.loads(reg_p.read_text())
    active = reg.get("minimodels", {}).get("active")
    assert active, "no active minimodels artifact"
    assert active in reg["minimodels"]["history"]
    vdir = root / "models_out" / "minimodels" / active
    return vdir, json.loads((vdir / "meta.json").read_text())


@pytest.fixture(scope="module")
def ml_module(root):
    p = root / "dbt" / "models" / "ml" / "ml_geo_scores.py"
    spec = importlib.util.spec_from_file_location("ml_geo_scores", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_artifact_files_read_only(artifact):
    vdir, _ = artifact
    for name in WEIGHT_FILES + ("meta.json",):
        assert (vdir / name).exists(), f"{name} missing from the artifact"
    for f in vdir.iterdir():
        mode = stat.S_IMODE(os.stat(f).st_mode)
        assert not (mode & stat.S_IWUSR), f"{f.name} is writable"


def test_version_is_content_addressed(artifact):
    """The version dir name must be the recomputable hash of
    (name, data fingerprint, hparams, code fingerprint) — exactly what
    artifacts.candidate_version hashes."""
    from provenance import sha256_json
    vdir, meta = artifact
    want = sha256_json({"name": meta["name"],
                        "data": meta["data_fingerprint"],
                        "hparams": meta["hparams"],
                        "code": meta["code_fingerprint"]})[:12]
    assert vdir.name == want


def test_gates_recorded_for_every_model(artifact):
    _, meta = artifact
    for name in MODELS:
        assert name in meta, name
        assert isinstance(meta[name].get("adopted"), bool), name
        assert meta[name].get("metrics"), name


def test_gates_are_honest(artifact):
    """Adoption flags must match the recorded metrics (honest negatives).
    A losing model with adopted=False is a finding, not a failure."""
    _, meta = artifact
    nr = meta["next_role"]["metrics"]
    assert meta["next_role"]["adopted"] == (
        nr["val_accuracy"] >= nr["return_prev_accuracy"])
    dc = meta["dwell_class"]["metrics"]
    assert meta["dwell_class"]["adopted"] == (
        dc["val_accuracy"] >= dc["majority_accuracy"])
    ml = meta["mode_leg"]["metrics"]
    assert meta["mode_leg"]["adopted"] == (
        ml["val_accuracy"] >= ml["rule_accuracy"])
    # the AE is a scorer, not a generator: no gate, but the metric is recorded
    assert meta["profile_ae"]["adopted"] is True
    assert meta["profile_ae"]["metrics"]["val_mse"] >= 0


def test_scalers_recorded(artifact):
    """The inference node rebuilds the training transform from meta alone."""
    _, meta = artifact
    for name in ("mode_leg", "profile_ae"):
        m = meta[name]["metrics"]
        assert len(m["scaler_mean"]) == len(m["scaler_std"]) == 7, name
        assert all(s > 0 for s in m["scaler_std"]), name


def test_user_split_not_time_split(artifact):
    """Leakage guard: the split must be on user id, not time."""
    _, meta = artifact
    assert "user" in str(meta["split"]) and "hash" in str(meta["split"])


def test_release_status_unreviewed(artifact):
    _, meta = artifact
    assert meta.get("release_status") == "internal_unreviewed"


def test_ml_scores_shape(con, artifact):
    vdir, _ = artifact
    cols = [r[0] for r in con.execute("describe ml_geo_scores").fetchall()]
    assert cols == SCORE_COLS
    n_scores = con.execute("select count(*) from ml_geo_scores").fetchone()[0]
    n_visits = con.execute("select count(*) from feat_visits").fetchone()[0]
    assert n_scores == n_visits, "one score row per visit"
    versions = {v for (v,) in con.execute(
        "select distinct model_version from ml_geo_scores").fetchall()}
    assert versions in ({vdir.name}, {"untrained"}), versions


def test_placeholder_when_no_artifact(ml_module, tmp_path):
    """With no registry at all the node must emit the typed placeholder so
    downstream SQL still compiles."""
    import pandas as pd
    assert ml_module._active_artifact(tmp_path) is None   # empty MODELS_DIR
    df = pd.DataFrame({"user_id": ["001", "001"], "sp_id": [1, 2],
                       "role": ["home", "work"]})
    out = ml_module._placeholder(df)
    assert list(out.columns) == SCORE_COLS
    assert list(out.model_version) == ["untrained", "untrained"]
    assert out.anomaly_score.isna().all()
    assert list(out.weak_label_role) == ["home", "work"]


def test_active_artifact_resolves(ml_module, root, artifact):
    vdir, _ = artifact
    active = ml_module._active_artifact(root / "models_out")
    assert active is not None
    version, path = active
    assert version == vdir.name and path == vdir


def test_inference_node_never_trains(root):
    src = (root / "dbt" / "models" / "ml" / "ml_geo_scores.py").read_text()
    assert "untrained" in src                     # placeholder path exists
    assert ".backward(" not in src, "inference node must not backprop"
    assert "optim" not in src, "inference node must not build an optimizer"


# ---------------------------------------------------------------------------
# Route-shape minimodels (RouteShapeGRU / LegShapeAE) + the leg-grain scores.
# These land after the route-shape layer: skip cleanly on an older artifact.
# ---------------------------------------------------------------------------

ROUTE_SKIP = ("active minimodels artifact predates the route-shape models — "
              "retrain with FLOW_RETRAIN=1 / run `make train`")
LEG_SCORE_COLS = ("leg_id", "shape_anomaly", "pred_next_token",
                  "pred_next_confidence", "model_version")


@pytest.fixture(scope="module")
def route_artifact(artifact):
    vdir, meta = artifact
    if "route_shape" not in meta or "leg_shape" not in meta:
        pytest.skip(ROUTE_SKIP)
    return vdir, meta


def test_route_shape_gate_recorded(route_artifact):
    """RouteShapeGRU: weights shipped, gate recorded (marginal token
    accuracy), adoption an honest bool."""
    vdir, meta = route_artifact
    assert (vdir / "route_shape.pt").exists()
    rs = meta["route_shape"]
    assert isinstance(rs.get("adopted"), bool)
    m = rs.get("metrics")
    assert m, "route_shape gate metrics missing"
    acc_keys = [k for k in m if "accuracy" in k and "marginal" not in k]
    base_keys = [k for k in m if "marginal" in k]
    assert acc_keys, f"no token-accuracy metric recorded: {sorted(m)}"
    assert base_keys, f"no marginal baseline recorded: {sorted(m)}"
    if len(acc_keys) == 1 and len(base_keys) == 1:
        # honest gate: adopted iff the model beats the marginal baseline
        assert rs["adopted"] == (m[acc_keys[0]] >= m[base_keys[0]])


def test_leg_shape_is_scoring_only(route_artifact):
    """LegShapeAE scores, never generates: no gate (adopted like
    profile_ae), and the scaler ships in meta so inference rebuilds the
    training transform exactly."""
    vdir, meta = route_artifact
    assert (vdir / "leg_shape.pt").exists()
    ls = meta["leg_shape"]
    assert ls.get("adopted") is True
    m = ls["metrics"]
    assert len(m["scaler_mean"]) == len(m["scaler_std"]) >= 1
    assert all(s > 0 for s in m["scaler_std"])


def test_route_weight_files_read_only(route_artifact):
    vdir, _ = route_artifact
    for name in ("route_shape.pt", "leg_shape.pt"):
        mode = stat.S_IMODE(os.stat(vdir / name).st_mode)
        assert not (mode & stat.S_IWUSR), f"{name} is writable"


def test_ml_leg_scores_shape(con, artifact):
    """ml_leg_scores: one score row per route leg (leg_id grain)."""
    vdir, _ = artifact
    tables = {t for (t,) in con.execute("show tables").fetchall()}
    if "ml_leg_scores" not in tables:
        pytest.skip("ml_leg_scores not built — "
                    "retrain with FLOW_RETRAIN=1 / run `make train`")
    cols = [r[0] for r in con.execute("describe ml_leg_scores").fetchall()]
    for c in LEG_SCORE_COLS:
        assert c in cols, f"{c} missing from ml_leg_scores ({cols})"
    n, n_ids = con.execute(
        "select count(*), count(distinct leg_id) from ml_leg_scores"
    ).fetchone()
    n_legs = con.execute(
        "select count(*) from feat_route_legs").fetchone()[0]
    assert n == n_ids, "leg_id must be the grain (no duplicates)"
    assert n == n_legs, "one score row per route leg"
    versions = {v for (v,) in con.execute(
        "select distinct model_version from ml_leg_scores").fetchall()}
    assert versions <= {vdir.name, "untrained"}, versions

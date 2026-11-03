# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-leg route-shape scores. Rebuilding this mart never refits a model; if
# no active artifact carries the route-shape heads yet it materializes an
# 'untrained' placeholder so downstream SQL still compiles.
# The model classes mirror the fitting pipeline exactly (RouteShapeGRU
# 12/8/24, LegShapeAE 6-feature 6->8->2->8->6) and reuse the meta leg_shape
# scaler + transform; artifact files: route_shape.pt, leg_shape.pt. Token
# windows are rebuilt identically to training: token = seg_class x ja_class
# (vocab 12), pad id = 12, WIN=8, one token sequence per leg (sequence_uuid)
# ordered by junction_idx. Grain: one row per feat_route_legs leg; legs with
# < 1 route token get null next-token predictions.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEG_CLASSES = ["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"]
JA_CLASSES = ["ja_30_60", "ja_60_120", "ja_120_180"]
TOKENS = [f"{s}|{a}" for s in SEG_CLASSES for a in JA_CLASSES]  # vocab 12
WIN = 8   # window length (route tokens), mirrors the fitting pipeline


def _active_artifact(models_dir: Path):
    reg = models_dir / "registry.json"
    if not reg.exists():
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        return None
    vdir = models_dir / "minimodels" / active
    return (active, vdir) if (vdir / "meta.json").exists() else None


def _placeholder(legs):
    return pd.DataFrame({
        "leg_id": legs.leg_id,
        "user_id": legs.user_id,
        "mode_class": legs.mode_class,
        "shape_anomaly": np.nan,
        "pred_next_token": None,
        "pred_next_confidence": np.nan,
        "model_version": "untrained",
    })


def _scores(legs, segs, version, vdir):
    import torch
    import torch.nn as nn

    meta = json.loads((vdir / "meta.json").read_text())

    class RouteShapeGRU(nn.Module):
        def __init__(self, n_tokens=12, emb=8, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_tokens + 1, emb)   # +1 pad id
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_tokens)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class LegShapeAE(nn.Module):
        def __init__(self, n_in=6):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    legs = legs.sort_values(["user_id", "leg_id"]).reset_index(drop=True)

    rgru = RouteShapeGRU()
    rgru.load_state_dict(torch.load(vdir / "route_shape.pt", map_location="cpu"))
    rgru.eval()
    ae = LegShapeAE()
    ae.load_state_dict(torch.load(vdir / "leg_shape.pt", map_location="cpu"))
    ae.eval()

    # ---- per-leg shape anomaly from the AE (meta scaler + transform) -------
    mu = np.array(meta["leg_shape"]["metrics"]["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["leg_shape"]["metrics"]["scaler_std"], dtype=np.float32)
    raw = np.stack([
        legs.path_m.to_numpy(np.float32),
        legs.circuity.fillna(1).to_numpy(np.float32),
        legs.junctions_per_km.fillna(0).to_numpy(np.float32),
        legs.mean_seg_m.fillna(0).to_numpy(np.float32),
        legs.straightness.fillna(1).to_numpy(np.float32),
        legs.n_junctions.to_numpy(np.float32),
    ], axis=1)
    feats = np.stack([np.log1p(raw[:, 0]), np.minimum(raw[:, 1], 6.0),
                      raw[:, 2], np.log1p(raw[:, 3]), raw[:, 4],
                      np.log1p(raw[:, 5])], axis=1)
    with torch.no_grad():
        x = torch.from_numpy((feats - mu) / sd)
        shape_anomaly = ((ae(x) - x) ** 2).mean(dim=1).numpy().astype(np.float64)

    # ---- next-token head: the GRU over each leg's token prefix -------------
    n = len(legs)
    pred_tok = np.full(n, None, dtype=object)
    pred_conf = np.full(n, np.nan)
    seg_idx = {s: i for i, s in enumerate(SEG_CLASSES)}
    ja_idx = {a: i for i, a in enumerate(JA_CLASSES)}
    segs = segs.sort_values(
        ["user_id", "trajectory_id", "sequence_uuid", "junction_idx"])
    tokens = (segs.seg_class.map(seg_idx) * len(JA_CLASSES)
              + segs.ja_class.map(ja_idx)).astype(np.int64)
    tok_by_leg = {
        key: toks.tolist() for key, toks in tokens.groupby(
            [segs.user_id, segs.trajectory_id, segs.sequence_uuid], sort=False)}
    wins, pos = [], []
    for i, key in enumerate(zip(legs.user_id, legs.trajectory_id,
                                legs.sequence_uuid)):
        toks = tok_by_leg.get(key)
        if not toks:            # a leg with < 1 token gets no prediction
            continue
        win = toks[-WIN:]
        wins.append([len(TOKENS)] * (WIN - len(win)) + win)
        pos.append(i)
    if wins:
        with torch.no_grad():
            probs = torch.softmax(
                rgru(torch.tensor(wins, dtype=torch.int64)), dim=1).numpy()
        for row, i in enumerate(pos):
            k = int(probs[row].argmax())
            pred_tok[i] = TOKENS[k]
            pred_conf[i] = float(probs[row][k])

    return pd.DataFrame({
        "leg_id": legs.leg_id,
        "user_id": legs.user_id,
        "mode_class": legs.mode_class,
        "shape_anomaly": shape_anomaly,
        "pred_next_token": pred_tok,
        "pred_next_confidence": pred_conf,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    legs = dbt.ref("feat_route_legs").df()[
        ["leg_id", "user_id", "trajectory_id", "sequence_uuid", "mode_class",
         "path_m", "circuity", "junctions_per_km", "mean_seg_m",
         "straightness", "n_junctions"]]
    segs = dbt.ref("feat_route_segments").df()[
        ["user_id", "trajectory_id", "sequence_uuid", "junction_idx",
         "seg_class", "ja_class"]]
    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    # an active artifact from before the route-shape layer has no route
    # heads — the placeholder keeps downstream SQL compiling either way
    if active is None or not (active[1] / "route_shape.pt").exists():
        return _placeholder(legs)
    return _scores(legs, segs, active[0], active[1])

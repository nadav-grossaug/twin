# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-transfer predictions. Rebuilding this mart never refits a model; if no
# artifact is active yet it materializes an 'untrained' placeholder so
# downstream SQL still compiles.
# The model classes mirror the fitting pipeline exactly (NextRegionGRU
# 8/6/24, GapClassGRU 5/6/24, UserProfileAE 5-feature) and reuse the meta
# profile_ae scaler + transform; artifact files: next_region.pt, gap_class.pt,
# profile_ae.pt.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["NA", "LA", "EU", "ME", "AF", "SA", "EA", "OC"]
GAPS = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
WIN = 8   # window length (hops), mirrors the fitting pipeline


def _active_artifact(models_dir: Path):
    reg = models_dir / "registry.json"
    if not reg.exists():
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        return None
    vdir = models_dir / "minimodels" / active
    return (active, vdir) if (vdir / "meta.json").exists() else None


def _placeholder(df):
    return pd.DataFrame({
        "obs_id": df.obs_id,
        "userid": df.userid,
        "anomaly_score": np.nan,
        "pred_next_region": None,
        "pred_next_confidence": np.nan,
        "p_next_region_correct": np.nan,
        "pred_next_gap_class": None,
        "weak_label_region": df.next_region,
        "model_version": "untrained",
    })


def _scores(df, users, version, vdir):
    import torch
    import torch.nn as nn

    meta = json.loads((vdir / "meta.json").read_text())

    class NextRegionGRU(nn.Module):
        def __init__(self, n_regions=8, emb=6, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_regions + 1, emb)   # +1 pad id
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_regions)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class GapClassGRU(nn.Module):
        def __init__(self, n_classes=5, emb=6, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_classes + 1, emb)   # +1 pad id
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class UserProfileAE(nn.Module):
        def __init__(self, n_in=5):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    df = df.sort_values(["userid", "travel_idx"]).reset_index(drop=True)
    reg_idx = {r: i for i, r in enumerate(REGIONS)}
    gap_idx = {g: i for i, g in enumerate(GAPS)}
    n = len(df)
    pred_next = np.full(n, None, dtype=object)
    pred_next_conf = np.full(n, np.nan)
    p_next_correct = np.full(n, np.nan)
    pred_gap = np.full(n, None, dtype=object)

    gru = NextRegionGRU()
    gru.load_state_dict(torch.load(vdir / "next_region.pt", map_location="cpu"))
    gru.eval()
    ggru = GapClassGRU()
    ggru.load_state_dict(torch.load(vdir / "gap_class.pt", map_location="cpu"))
    ggru.eval()
    ae = UserProfileAE()
    ae.load_state_dict(torch.load(vdir / "profile_ae.pt", map_location="cpu"))
    ae.eval()

    # ---- per-user anomaly from the profile AE (meta scaler + transform) ----
    mu = np.array(meta["profile_ae"]["metrics"]["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["profile_ae"]["metrics"]["scaler_std"], dtype=np.float32)
    raw = users[["n_travels", "n_countries", "return_rate_pct",
                 "mean_gap_days", "sd_gap_days"]].fillna(0).to_numpy(np.float32)
    feats = np.stack([np.log1p(raw[:, 0]), np.log1p(raw[:, 1]),
                      raw[:, 2] / 100.0,
                      np.log1p(raw[:, 3]), np.log1p(raw[:, 4])], axis=1)
    with torch.no_grad():
        x = torch.from_numpy((feats - mu) / sd)
        err = ((ae(x) - x) ** 2).mean(dim=1).numpy()
    user_anom = dict(zip(users.userid.to_numpy(), err.astype(float)))
    anomaly = df.userid.map(user_anom).to_numpy(dtype=np.float64)

    # ---- sequence heads: region and gap-class windows per user ------------
    with torch.no_grad():
        for _, g in df.groupby("userid", sort=False):
            idx = g.index.to_numpy()
            regions = g.next_region.to_numpy()
            gclasses = g.gap_class.to_numpy()
            seq_r, seq_g, pos = [], [], []
            for row_i, i in enumerate(idx):
                hist = [reg_idx[r] for r in regions[max(0, row_i - WIN + 1):row_i + 1]
                        if r in reg_idx]
                seq_r.append([len(REGIONS)] * (WIN - len(hist)) + hist[-WIN:])
                ghist = [gap_idx[c] for c in gclasses[max(0, row_i - WIN + 1):row_i + 1]
                         if c in gap_idx]
                seq_g.append([len(GAPS)] * (WIN - len(ghist)) + ghist[-WIN:])
                pos.append((row_i, i))
            if not pos:
                continue
            probs = torch.softmax(
                gru(torch.tensor(seq_r, dtype=torch.int64)), dim=1).numpy()
            gprobs = torch.softmax(
                ggru(torch.tensor(seq_g, dtype=torch.int64)), dim=1).numpy()
            for row, (row_i, i) in enumerate(pos):
                k = probs[row].argmax()
                pred_next[i] = REGIONS[k]
                pred_next_conf[i] = float(probs[row][k])
                pred_gap[i] = GAPS[gprobs[row].argmax()]
                # probability mass on the realized next region (rows with a
                # successor whose region is on the public grid)
                if row_i + 1 < len(idx) and regions[row_i + 1] in reg_idx:
                    p_next_correct[i] = float(
                        probs[row][reg_idx[regions[row_i + 1]]])

    return pd.DataFrame({
        "obs_id": df.obs_id,
        "userid": df.userid,
        "anomaly_score": anomaly,
        "pred_next_region": pred_next,
        "pred_next_confidence": pred_next_conf,
        "p_next_region_correct": p_next_correct,
        "pred_next_gap_class": pred_gap,
        "weak_label_region": df.next_region,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    df = dbt.ref("feat_transfers").df()[
        ["obs_id", "userid", "travel_idx", "gap_class", "next_region"]]
    users = dbt.ref("feat_users").df()[
        ["userid", "n_travels", "n_countries", "return_rate_pct",
         "mean_gap_days", "sd_gap_days"]]
    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        return _placeholder(df)
    return _scores(df, users, active[0], active[1])

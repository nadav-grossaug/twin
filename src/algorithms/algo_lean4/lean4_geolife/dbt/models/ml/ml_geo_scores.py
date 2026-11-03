# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-visit predictions. Rebuilding this mart never refits a model; if no
# artifact is active yet it materializes an 'untrained' placeholder so
# downstream SQL still compiles.
# The model classes mirror the fitting pipeline exactly (NextRoleGRU 3/6/24,
# DwellClassGRU 5/6/24, UserProfileAE 7-feature) and reuse the meta
# profile_ae scaler + transform; artifact files: next_role.pt,
# dwell_class.pt, profile_ae.pt (mode_leg.pt is leg-grain and not scored at
# this visit grain). Windows are rebuilt identically to training: pad id =
# vocab size, WIN=8, sequences RESET at seq_ok=false breaks, and the dwell
# stream keeps only uncensored visits.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["home", "work", "other"]
DWELLS = ["dw_lt1h", "dw_1_3h", "dw_3_8h", "dw_8_14h", "dw_ge14h"]
WIN = 8   # window length (visits), mirrors the fitting pipeline


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
        "user_id": df.user_id,
        "sp_id": df.sp_id,
        "anomaly_score": np.nan,
        "pred_next_role": None,
        "pred_next_confidence": np.nan,
        "p_next_role_correct": np.nan,
        "pred_dwell_class": None,
        "weak_label_role": df.role,
        "model_version": "untrained",
    })


def _scores(df, users, version, vdir):
    import torch
    import torch.nn as nn

    meta = json.loads((vdir / "meta.json").read_text())

    class NextRoleGRU(nn.Module):
        def __init__(self, n_roles=3, emb=6, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_roles + 1, emb)   # +1 pad id
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_roles)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class DwellClassGRU(nn.Module):
        def __init__(self, n_classes=5, emb=6, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_classes + 1, emb)   # +1 pad id
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class UserProfileAE(nn.Module):
        def __init__(self, n_in=7):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    df = df.sort_values(["user_id", "ts_arrive_s", "sp_id"]).reset_index(drop=True)
    role_idx = {r: i for i, r in enumerate(ROLES)}
    dwell_idx = {d: i for i, d in enumerate(DWELLS)}
    n = len(df)
    pred_next = np.full(n, None, dtype=object)
    pred_next_conf = np.full(n, np.nan)
    p_next_correct = np.full(n, np.nan)
    pred_dwell = np.full(n, None, dtype=object)

    gru = NextRoleGRU()
    gru.load_state_dict(torch.load(vdir / "next_role.pt", map_location="cpu"))
    gru.eval()
    dgru = DwellClassGRU()
    dgru.load_state_dict(torch.load(vdir / "dwell_class.pt", map_location="cpu"))
    dgru.eval()
    ae = UserProfileAE()
    ae.load_state_dict(torch.load(vdir / "profile_ae.pt", map_location="cpu"))
    ae.eval()

    # ---- per-user anomaly from the profile AE (meta scaler + transform) ----
    mu = np.array(meta["profile_ae"]["metrics"]["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["profile_ae"]["metrics"]["scaler_std"], dtype=np.float32)
    raw = users[["n_days", "mean_sp_per_day", "share_commute", "share_home",
                 "n_anchors", "median_dwell_s", "n_visits"]].fillna(0).to_numpy(np.float32)
    feats = np.stack([np.log1p(raw[:, 0]), raw[:, 1], raw[:, 2], raw[:, 3],
                      np.log1p(raw[:, 4]), np.log1p(raw[:, 5]),
                      np.log1p(raw[:, 6])], axis=1)
    with torch.no_grad():
        x = torch.from_numpy((feats - mu) / sd)
        err = ((ae(x) - x) ** 2).mean(dim=1).numpy()
    user_anom = dict(zip(users.user_id.to_numpy(), err.astype(float)))
    anomaly = df.user_id.map(user_anom).to_numpy(dtype=np.float64)

    # ---- sequence heads: role and dwell-class windows per user -------------
    with torch.no_grad():
        for _, g in df.groupby("user_id", sort=False):
            idx = g.index.to_numpy()
            roles = g.role.to_numpy()
            dwells = g.dwell_class.to_numpy()
            oks = g.seq_ok.to_numpy()
            cens = (g.arr_censored | g.dep_censored).to_numpy()
            seq_r, pos_r = [], []
            seq_d, pos_d = [], []
            r_seg, d_seg = [], []
            for row_i, i in enumerate(idx):
                if not oks[row_i]:            # sequence break: history resets
                    r_seg, d_seg = [], []
                if roles[row_i] in role_idx:
                    r_seg.append(role_idx[roles[row_i]])
                    win = r_seg[-WIN:]
                    seq_r.append([len(ROLES)] * (WIN - len(win)) + win)
                    pos_r.append((row_i, i))
                # the dwell stream keeps only uncensored visits (training
                # convention); censored visits get no dwell prediction
                if not cens[row_i] and dwells[row_i] in dwell_idx:
                    d_seg.append(dwell_idx[dwells[row_i]])
                    win = d_seg[-WIN:]
                    seq_d.append([len(DWELLS)] * (WIN - len(win)) + win)
                    pos_d.append(i)
            if seq_r:
                probs = torch.softmax(
                    gru(torch.tensor(seq_r, dtype=torch.int64)), dim=1).numpy()
                for row, (row_i, i) in enumerate(pos_r):
                    k = probs[row].argmax()
                    pred_next[i] = ROLES[k]
                    pred_next_conf[i] = float(probs[row][k])
                    # probability mass on the realized next role — only when
                    # the next visit continues the sequence (its seq_ok true)
                    if (row_i + 1 < len(idx) and bool(oks[row_i + 1])
                            and roles[row_i + 1] in role_idx):
                        p_next_correct[i] = float(
                            probs[row][role_idx[roles[row_i + 1]]])
            if seq_d:
                dprobs = torch.softmax(
                    dgru(torch.tensor(seq_d, dtype=torch.int64)), dim=1).numpy()
                for row, i in enumerate(pos_d):
                    pred_dwell[i] = DWELLS[dprobs[row].argmax()]

    return pd.DataFrame({
        "user_id": df.user_id,
        "sp_id": df.sp_id,
        "anomaly_score": anomaly,
        "pred_next_role": pred_next,
        "pred_next_confidence": pred_next_conf,
        "p_next_role_correct": p_next_correct,
        "pred_dwell_class": pred_dwell,
        "weak_label_role": df.role,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    df = dbt.ref("feat_visits").df()[
        ["user_id", "sp_id", "role", "dwell_class", "seq_ok",
         "ts_arrive_s", "arr_censored", "dep_censored"]]
    users = dbt.ref("mart_user_stats").df()[
        ["user_id", "n_days", "mean_sp_per_day", "share_commute",
         "share_home", "n_anchors", "median_dwell_s", "n_visits"]]
    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        return _placeholder(df)
    return _scores(df, users, active[0], active[1])

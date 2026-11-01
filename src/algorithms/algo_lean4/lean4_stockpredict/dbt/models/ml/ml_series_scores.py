# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-day predictions. Rebuilding this mart never retrains; if no artifact is
# active yet it materializes an 'untrained' placeholder so downstream SQL
# still compiles.
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["crash", "down", "flat", "up", "surge"]
WIN = 10


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
        "anomaly_score": np.nan,
        "pred_next_class": None,
        "pred_next_confidence": np.nan,
        "pred_season_class": None,
        "pred_season_confidence": np.nan,
        "garch_sigma_bp": np.nan,
        "factor_residual_bp": np.nan,
        "weak_label_class": df.ret_class,
        "model_version": "untrained",
    })


def _scores(df, version, vdir):
    import torch
    import torch.nn as nn

    meta = json.loads((vdir / "meta.json").read_text())

    class ReturnWindowAE(nn.Module):
        def __init__(self, n_in=WIN):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class NextClassGRU(nn.Module):
        def __init__(self, n_classes=5, emb=6, hidden=24):
            super().__init__()
            self.emb = nn.Embedding(n_classes + 1, emb)
            self.gru = nn.GRU(emb, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, x):
            _, h = self.gru(self.emb(x))
            return self.out(h[-1])

    class SeasonalityMLP(nn.Module):
        def __init__(self, n_in=17, n_classes=5):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, 24), nn.ReLU(),
                                     nn.Linear(24, n_classes))

        def forward(self, x):
            return self.net(x)

    df = df.sort_values(["ticker", "d"]).reset_index(drop=True)
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    n = len(df)
    anomaly = np.full(n, np.nan)
    pred_next = np.full(n, None, dtype=object)
    pred_next_conf = np.full(n, np.nan)
    pred_season = np.full(n, None, dtype=object)
    pred_season_conf = np.full(n, np.nan)
    garch_sigma = np.full(n, np.nan)
    factor_resid = np.full(n, np.nan)

    ae = ReturnWindowAE()
    ae.load_state_dict(torch.load(vdir / "return_ae.pt", map_location="cpu"))
    ae.eval()
    gru = NextClassGRU()
    gru.load_state_dict(torch.load(vdir / "next_class.pt", map_location="cpu"))
    gru.eval()
    mlp = SeasonalityMLP()
    mlp.load_state_dict(torch.load(vdir / "seasonality.pt", map_location="cpu"))
    mlp.eval()
    garch = json.loads((vdir / "garch_params.json").read_text())
    factor = json.loads((vdir / "factor_loadings.json").read_text())
    mu = meta["return_ae"]["metrics"]["scaler_mean"]
    sd = meta["return_ae"]["metrics"]["scaler_std"]

    with torch.no_grad():
        for t, g in df.groupby("ticker", sort=False):
            idx = g.index.to_numpy()
            bps = g.ret_bp.to_numpy(dtype=np.float64)
            ok = ~np.isnan(bps)
            # AE anomaly on trailing windows + GRU next-class
            seq_ids, seq_pos = [], []
            for row_i, i in enumerate(idx):
                if not ok[row_i]:
                    continue
                lo = max(0, row_i - WIN + 1)
                win = bps[lo:row_i + 1]
                win = win[~np.isnan(win)]
                if len(win) == WIN:
                    x = torch.tensor((win / 100.0 - mu) / sd,
                                     dtype=torch.float32).unsqueeze(0)
                    anomaly[i] = float(((ae(x) - x) ** 2).mean())
                hist = [cls_idx[c] for c in g.ret_class.to_numpy()[lo:row_i + 1]
                        if c is not None]
                seq_ids.append([5] * (WIN - len(hist)) + hist[-WIN:])
                seq_pos.append(i)
            if seq_ids:
                probs = torch.softmax(
                    gru(torch.tensor(seq_ids, dtype=torch.int64)), dim=1).numpy()
                for row, i in enumerate(seq_pos):
                    k = probs[row].argmax()
                    pred_next[i] = CLASSES[k]
                    pred_next_conf[i] = float(probs[row][k])
            # GARCH filtered sigma
            p = garch.get(t)
            if p and ok.sum() > 2:
                r = bps[ok] / 1e4
                s2 = float(np.var(r)) + 1e-12
                sig = np.full(len(r), np.nan)
                for k in range(1, len(r)):
                    s2 = p["omega"] + p["alpha"] * r[k - 1] ** 2 + p["beta"] * s2
                    sig[k] = math.sqrt(s2) * 1e4
                garch_sigma[idx[ok]] = sig

        # seasonality prediction (vectorized over all rows with a class)
        okm = df.ret_class.notna().to_numpy()
        oh = np.zeros((int(okm.sum()), 17), dtype=np.float32)
        sub = df[okm]
        oh[np.arange(len(sub)), sub.dow.to_numpy() - 1] = 1.0
        oh[np.arange(len(sub)), 4 + sub.month.to_numpy()] = 1.0
        probs = torch.softmax(mlp(torch.from_numpy(oh)), dim=1).numpy()
        ks = probs.argmax(axis=1)
        pred_season[okm] = [CLASSES[k] for k in ks]
        pred_season_conf[okm] = probs[np.arange(len(ks)), ks]

        # factor residual per day (cross-sectional reconstruction)
        tickers = factor["tickers"]
        B = np.array(factor["loadings"])          # 2 x T
        tpos = {t: j for j, t in enumerate(tickers)}
        for d, g in df.groupby("d", sort=False):
            known = g[g.ret_bp.notna() & g.ticker.isin(tpos)]
            if len(known) < 3:
                continue
            r = known.ret_bp.to_numpy(dtype=np.float64) / 1e4
            Bk = B[:, [tpos[t] for t in known.ticker]]
            f, *_ = np.linalg.lstsq(Bk.T, r, rcond=None)
            resid = np.abs(r - Bk.T @ f) * 1e4
            factor_resid[known.index.to_numpy()] = resid

    return pd.DataFrame({
        "obs_id": df.obs_id,
        "anomaly_score": anomaly,
        "pred_next_class": pred_next,
        "pred_next_confidence": pred_next_conf,
        "pred_season_class": pred_season,
        "pred_season_confidence": pred_season_conf,
        "garch_sigma_bp": garch_sigma,
        "factor_residual_bp": factor_resid,
        "weak_label_class": df.ret_class,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    df = dbt.ref("feat_returns").df()[
        ["obs_id", "ticker", "d", "ret_bp", "ret_class", "dow", "month"]]
    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        return _placeholder(df)
    return _scores(df, active[0], active[1])

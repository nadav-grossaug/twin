#!/usr/bin/env python3
"""Fit the stock-series minimodels into the content-addressed artifact store.

Five models, each answering one modelling question, each gated by a frozen
non-ML baseline (honest negatives are kept):

  ReturnWindowAE   anomaly/embedding over 10-day log-return windows
                   (scoring model; stays enclave-side forever)
  NextClassGRU     P(next return class | class history)
                   gate: repeat-last AND majority accuracy
  SeasonalityMLP   P(return class | day-of-week, month)   [seasonality focus]
                   gate: majority accuracy
  GarchVol         GARCH(1,1) per ticker by MLE (torch autograd)
                   [GARCH diffusion focus]; gate: constant-volatility
                   Gaussian NLL on held-out days
  FactorModel      2-factor linear model of the daily cross-section
                   [collinearity focus]; gate: factor 1 explains >= 15% of
                   variance (else the market-coupling story is unsupported)

Split: time-ordered 80/20 by date, fixed before fitting (leakage guideline).
Artifacts: content-addressed via python/artifacts.py (FLOW_RETRAIN=1 to
activate a retrain; FLOW_DEVICE=cpu for the deterministic mode).
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifacts import train_or_reuse  # noqa: E402
from provenance import sha256_file  # noqa: E402
from spec import load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"
TSV = ROOT / "stock_normalized_train.tsv"

CLASSES = ["crash", "down", "flat", "up", "surge"]
WIN = 10          # window length (days) for AE and GRU
EPOCHS = int(os.environ.get("FLOW_EPOCHS", "60"))
SEED = 1234


def pick_device() -> torch.device:
    forced = os.environ.get("FLOW_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# architectures (rebuilt identically by dbt/models/ml/ml_series_scores.py)
# ---------------------------------------------------------------------------
class ReturnWindowAE(nn.Module):
    def __init__(self, n_in: int = WIN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class NextClassGRU(nn.Module):
    def __init__(self, n_classes: int = 5, emb: int = 6, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class SeasonalityMLP(nn.Module):
    def __init__(self, n_in: int = 17, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 24), nn.ReLU(),
                                 nn.Linear(24, n_classes))

    def forward(self, x):
        return self.net(x)


class GarchVol(nn.Module):
    """GARCH(1,1) parameters for one ticker, MLE via autograd:
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}."""

    def __init__(self):
        super().__init__()
        self.raw_omega = nn.Parameter(torch.tensor(-9.0))
        self.raw_alpha = nn.Parameter(torch.tensor(-2.0))
        self.raw_beta = nn.Parameter(torch.tensor(1.0))

    def params(self):
        omega = torch.nn.functional.softplus(self.raw_omega) * 1e-4
        # keep alpha + beta < 1 (stationarity) via a scaled softmax-ish map
        a = torch.sigmoid(self.raw_alpha)
        b = torch.sigmoid(self.raw_beta)
        scale = 0.999 / torch.clamp(a + b, min=1.0)
        cap = torch.clamp(scale, max=1.0)
        return omega, a * cap, b * cap

    def nll(self, r: torch.Tensor) -> torch.Tensor:
        omega, alpha, beta = self.params()
        sigma2 = torch.var(r) + 1e-12
        out = []
        for t in range(1, len(r)):
            sigma2 = omega + alpha * r[t - 1] ** 2 + beta * sigma2
            out.append(0.5 * (torch.log(2 * math.pi * sigma2) + r[t] ** 2 / sigma2))
        return torch.stack(out).mean()


# ---------------------------------------------------------------------------
def build_windows(seqs: list[tuple[list[int], list[int]]], k: int = WIN):
    """Per-ticker class-id sequences -> (window, next) pairs, chronological."""
    xs, ys, ts = [], [], []
    for seq, days in seqs:
        for i in range(1, len(seq)):
            win = seq[max(0, i - k):i]
            xs.append([5] * (k - len(win)) + win)   # 5 = pad id
            ys.append(seq[i])
            ts.append(days[i])
    order = np.argsort(np.array(ts), kind="stable")
    return (np.array(xs, dtype=np.int64)[order],
            np.array(ys, dtype=np.int64)[order])


def train_loop(model, loss_fn, xs, ys, device, epochs, lr=1e-2, batch=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(xs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xs[idx].to(device)), ys[idx].to(device))
            loss.backward()
            opt.step()
    model.eval()


def main() -> int:
    if not DB.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    spec = load_spec()
    con = duckdb.connect(str(DB), read_only=True)

    rows = con.execute(
        "select ticker, d, day_num, ret_bp, ret_class, dow, month "
        "from feat_returns where ret_bp is not null order by ticker, d"
    ).fetchall()
    con.close()
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    by_ticker: dict[str, list] = {}
    for t, d, dn, bp, cl, dow, mon in rows:
        by_ticker.setdefault(t, []).append((int(dn), int(bp), cls_idx[cl],
                                            int(dow), int(mon)))
    tickers = sorted(by_ticker)
    all_days = sorted({r[0] for rs in by_ticker.values() for r in rs})
    split_day = all_days[int(len(all_days) * 0.8)]   # fixed before fitting

    data_fp = sha256_file(TSV)
    hparams = {"win": WIN, "epochs": EPOCHS, "seed": SEED,
               "classes": CLASSES, "split_day": split_day,
               "spec_version": spec_version()}

    def fit():
        files: dict[str, bytes] = {}
        meta: dict = {"spec_version": spec_version(), "classes": CLASSES,
                      "device": str(device), "split_day": split_day}

        # ---- ReturnWindowAE (anomaly scorer; no adoption gate — scoring) ---
        wins, wins_t = [], []
        for t in tickers:
            bps = [r[1] for r in by_ticker[t]]
            dns = [r[0] for r in by_ticker[t]]
            for i in range(WIN, len(bps) + 1):
                wins.append(bps[i - WIN:i])
                wins_t.append(dns[i - 1])
        wins = np.array(wins, dtype=np.float32) / 100.0   # bp -> percent
        wins_t = np.array(wins_t)
        tr = wins[wins_t <= split_day]
        mu, sd = tr.mean(), tr.std() + 1e-6
        ae = ReturnWindowAE().to(device)
        xs = torch.from_numpy((tr - mu) / sd)
        train_loop(ae, nn.MSELoss(), xs, xs, device, EPOCHS)
        with torch.no_grad():
            va = torch.from_numpy((wins[wins_t > split_day] - mu) / sd).to(device)
            val_err = float(((ae(va) - va) ** 2).mean())
        meta["return_ae"] = {"metrics": {"val_mse": val_err, "scaler_mean": float(mu),
                                         "scaler_std": float(sd), "win": WIN},
                             "adopted": True,
                             "role": "scoring only (never distilled)"}
        buf = io.BytesIO(); torch.save(ae.cpu().state_dict(), buf)
        files["return_ae.pt"] = buf.getvalue()

        # ---- NextClassGRU (gate: repeat-last AND majority) -----------------
        seqs = [([r[2] for r in by_ticker[t]], [r[0] for r in by_ticker[t]])
                for t in tickers]
        xs_np, ys_np = build_windows(seqs)
        n_tr = int(len(xs_np) * 0.8)
        gru = NextClassGRU().to(device)
        train_loop(gru, nn.CrossEntropyLoss(),
                   torch.from_numpy(xs_np[:n_tr]), torch.from_numpy(ys_np[:n_tr]),
                   device, EPOCHS)
        with torch.no_grad():
            pred = gru(torch.from_numpy(xs_np[n_tr:]).to(device)).argmax(1).cpu().numpy()
        val_y = ys_np[n_tr:]
        acc = float((pred == val_y).mean())
        repeat_acc = float((xs_np[n_tr:, -1] == val_y).mean())
        maj = np.bincount(ys_np[:n_tr]).argmax()
        maj_acc = float((val_y == maj).mean())
        meta["next_class"] = {
            "k": WIN,
            "metrics": {"val_accuracy": acc, "repeat_last_accuracy": repeat_acc,
                        "majority_accuracy": maj_acc, "n_val": int(len(val_y))},
            "adopted": bool(acc >= repeat_acc and acc >= maj_acc),
            "role": "generative teacher -> distilled to DP transition rows"}
        buf = io.BytesIO(); torch.save(gru.cpu().state_dict(), buf)
        files["next_class.pt"] = buf.getvalue()

        # ---- SeasonalityMLP (gate: majority) -------------------------------
        feats, labels, ts = [], [], []
        for t in tickers:
            for dn, _, ci, dow, mon in by_ticker[t]:
                oh = np.zeros(17, dtype=np.float32)
                oh[dow - 1] = 1.0            # dow 1..5 -> slots 0..4
                oh[4 + mon] = 1.0            # month 1..12 -> slots 5..16
                feats.append(oh); labels.append(ci); ts.append(dn)
        feats = np.array(feats); labels = np.array(labels, dtype=np.int64)
        ts = np.array(ts)
        tr_m = ts <= split_day
        mlp = SeasonalityMLP().to(device)
        train_loop(mlp, nn.CrossEntropyLoss(),
                   torch.from_numpy(feats[tr_m]), torch.from_numpy(labels[tr_m]),
                   device, EPOCHS)
        with torch.no_grad():
            pred = mlp(torch.from_numpy(feats[~tr_m]).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == labels[~tr_m]).mean())
        maj = np.bincount(labels[tr_m]).argmax()
        maj_acc = float((labels[~tr_m] == maj).mean())
        meta["seasonality"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "n_val": int((~tr_m).sum())},
            "adopted": bool(acc >= maj_acc),
            "role": "seasonality probe -> released counterpart is the DP dow_class histogram"}
        buf = io.BytesIO(); torch.save(mlp.cpu().state_dict(), buf)
        files["seasonality.pt"] = buf.getvalue()

        # ---- GarchVol per ticker (gate: constant-vol NLL) ------------------
        garch_meta = {}
        garch_params = {}
        adopted_any = False
        for t in tickers:
            r_all = np.array([r[1] for r in by_ticker[t]], dtype=np.float32) / 1e4
            dns = np.array([r[0] for r in by_ticker[t]])
            r_tr = torch.from_numpy(r_all[dns <= split_day])
            r_va = torch.from_numpy(r_all[dns > split_day])
            g = GarchVol()
            opt = torch.optim.Adam(g.parameters(), lr=5e-2)
            for _ in range(200):
                opt.zero_grad()
                loss = g.nll(r_tr)
                loss.backward()
                opt.step()
            with torch.no_grad():
                omega, alpha, beta = (float(x) for x in g.params())
                val_nll = float(g.nll(r_va)) if len(r_va) > 5 else None
                const_var = float(torch.var(r_tr)) + 1e-12
                base_nll = float(np.mean(
                    0.5 * (np.log(2 * math.pi * const_var)
                           + (r_va.numpy() ** 2) / const_var))) if len(r_va) > 5 else None
            adopted = bool(val_nll is not None and base_nll is not None
                           and val_nll <= base_nll)
            adopted_any = adopted_any or adopted
            garch_params[t] = {"omega": omega, "alpha": alpha, "beta": beta}
            garch_meta[t] = {"omega": omega, "alpha": alpha, "beta": beta,
                             "persistence": alpha + beta,
                             "val_nll": val_nll, "const_vol_nll": base_nll,
                             "adopted": adopted}
        meta["garch"] = {
            "per_ticker": garch_meta,
            "adopted": adopted_any,
            "role": "GARCH diffusion teacher -> distilled to DP vol-state "
                    "transitions + the public vol-boost table",
            "note": "per-ticker parameters are PRIVATE (never released); the "
                    "released counterpart is the vol_transition histogram"}
        files["garch_params.json"] = json.dumps(garch_params, indent=2).encode()

        # ---- FactorModel (2-factor SVD on the train cross-section) ---------
        day_set = sorted({r[0] for t in tickers for r in by_ticker[t]
                          if r[0] <= split_day})
        day_pos = {d: i for i, d in enumerate(day_set)}
        mat = np.zeros((len(day_set), len(tickers)), dtype=np.float32)
        for j, t in enumerate(tickers):
            for dn, bp, *_ in by_ticker[t]:
                if dn in day_pos:
                    mat[day_pos[dn], j] = bp / 1e4
        mat = mat - mat.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        var = s ** 2 / (s ** 2).sum()
        loadings = vt[:2]                     # 2 x n_tickers
        explained_1, explained_2 = float(var[0]), float(var[:2].sum())
        meta["factor"] = {
            "metrics": {"var_explained_f1": explained_1,
                        "var_explained_f1f2": explained_2,
                        "n_tickers": len(tickers), "n_days": len(day_set)},
            "adopted": bool(explained_1 >= 0.15),
            "role": "collinearity teacher -> distilled to comove_lambda_pct "
                    "(market coupling); loadings are PRIVATE"}
        files["factor_loadings.json"] = json.dumps(
            {"tickers": tickers, "loadings": loadings.tolist(),
             "var_explained": var[:4].tolist()}, indent=2).encode()

        meta["numeric_note"] = "all returns modelled as LOG returns (bp)"
        return files, meta

    version, trained = train_or_reuse(
        "minimodels", data_fp, hparams,
        [Path(__file__)], fit)
    print(f"train: minimodels version {version} "
          f"({'trained now' if trained else 'reused existing artifact'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

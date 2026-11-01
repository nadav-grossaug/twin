#!/usr/bin/env python3
"""Fit the travel-sequence minimodels into the content-addressed artifact store.

Four models, each answering one modelling question, each gated by a frozen
non-ML baseline (honest negatives are kept):

  NextRegionGRU     P(next region | the user's region sequence)
                    gate: RETURN-TO-PREVIOUS region — predict the region the
                    user came from (measured 72.1% at next-country direction
                    on the source data; the region-level baseline is computed
                    honestly here on the same held-out users)
  GapClassGRU       P(next gap class | gap-class history)
                    gate: marginal class shares (majority class)
  EpisodeShapeModel P(next episode-length bin | previous bin) — a learned
                    bigram over journey lengths; gate: unigram NLL
  UserProfileAE     anomaly/embedding over per-user profile vectors
                    (n_travels, n_countries, return_rate, mean/sd gap);
                    scoring model, no gate — stays enclave-side forever

Split: 80/20 ON USER ID HASH, fixed before fitting (the privacy unit is the
user, so leakage control must be user-level, not time-level). Duplicate
(window, label) training pairs are deduplicated and weighted — region and
gap-class alphabets are tiny, so dedup cuts the epoch cost sharply without
changing the loss. Artifacts: content-addressed via python/artifacts.py
(FLOW_RETRAIN=1 to activate a retrain; FLOW_DEVICE=cpu for the deterministic
mode).
"""
from __future__ import annotations

import hashlib
import io
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
from spec import REGIONS, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"
TSV = ROOT / "country_transfer.tsv"

GAP_CLASSES = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
EP_BINS = ["ep_1", "ep_2", "ep_3", "ep_4_5", "ep_6p"]
PROFILE_FEATURES = ["n_travels", "n_countries", "return_rate_pct",
                    "mean_gap_days", "sd_gap_days"]
WIN = 8           # window length (hops) for the GRUs
EPOCHS = int(os.environ.get("FLOW_EPOCHS", "30"))
SEED = 1234
VAL_PCT = 20      # users with sha256(userid) % 100 >= 80 are held out


def pick_device() -> torch.device:
    forced = os.environ.get("FLOW_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_train_user(userid: int) -> bool:
    """Deterministic user-level 80/20 split (the whole history of a user is
    one record, so every window of a held-out user is held out)."""
    h = int(hashlib.sha256(str(int(userid)).encode()).hexdigest(), 16)
    return h % 100 < 100 - VAL_PCT


# ---------------------------------------------------------------------------
# architectures (rebuilt identically by dbt/models/ml/ml_travel_scores.py)
# ---------------------------------------------------------------------------
class NextRegionGRU(nn.Module):
    def __init__(self, n_regions: int = 8, emb: int = 6, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_regions + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_regions)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class GapClassGRU(nn.Module):
    def __init__(self, n_classes: int = 5, emb: int = 6, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class EpisodeShapeModel(nn.Module):
    """Learned bigram over episode-length bins: previous bin -> next bin."""

    def __init__(self, n_bins: int = 5, emb: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_bins, emb)
        self.out = nn.Linear(emb, n_bins)

    def forward(self, x):
        return self.out(self.emb(x))


class UserProfileAE(nn.Module):
    def __init__(self, n_in: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
def build_windows(seqs: list[list[int]], n_vocab: int, k: int = WIN):
    """Per-user token sequences -> (window, next) pairs; pad id = n_vocab."""
    xs, ys = [], []
    for seq in seqs:
        for i in range(1, len(seq)):
            win = seq[max(0, i - k):i]
            xs.append([n_vocab] * (k - len(win)) + win)
            ys.append(seq[i])
    return (np.array(xs, dtype=np.int64).reshape(-1, k),
            np.array(ys, dtype=np.int64))


def dedup_weight(xs: np.ndarray, ys: np.ndarray):
    """Collapse duplicate (window, label) pairs into counts (training weights).

    The loss is unchanged — duplicates contribute their multiplicity — but
    over a tiny alphabet the epoch cost drops by an order of magnitude."""
    keyed: dict[tuple, int] = {}
    for x, y in zip(xs, ys):
        keyed[(*x, y)] = keyed.get((*x, y), 0) + 1
    k = xs.shape[1]
    uniq = np.array([key[:k] for key in keyed], dtype=np.int64)
    labs = np.array([key[k] for key in keyed], dtype=np.int64)
    w = np.array(list(keyed.values()), dtype=np.float32)
    return uniq, labs, w


def train_loop(model, xs, ys, ws, device, epochs, lr=1e-2, batch=256):
    """Weighted cross-entropy training (ws = duplicate multiplicities)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.train()
    n = len(xs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            per = loss_fn(model(xs[idx].to(device)), ys[idx].to(device))
            w = ws[idx].to(device)
            loss = (per * w).sum() / w.sum()
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
    con = duckdb.connect(str(DB), read_only=True)

    # per-user region sequences: the region the user starts in, then the
    # destination region of every transfer (XX rows are quality-flagged and
    # excluded — the unknown-region marker is never modelled)
    transfer_rows = con.execute(
        "select userid, prev_region, next_region, gap_class from feat_transfers "
        "where prev_region != 'XX' and next_region != 'XX' "
        "order by userid, ts_s").fetchall()
    user_rows = con.execute(
        "select userid, coalesce(n_travels, 0), coalesce(n_countries, 0), "
        "coalesce(return_rate_pct, 0), coalesce(mean_gap_days, 0), "
        "coalesce(sd_gap_days, 0) from feat_users order by userid").fetchall()
    episode_rows = con.execute(
        "select userid, episode_id, ep_bin from feat_episodes "
        "order by userid, episode_id").fetchall()
    con.close()

    region_idx = {r: i for i, r in enumerate(REGIONS)}
    gap_idx = {g: i for i, g in enumerate(GAP_CLASSES)}
    ep_idx = {e: i for i, e in enumerate(EP_BINS)}

    region_seqs: dict[int, list[int]] = {}
    gap_seqs: dict[int, list[int]] = {}
    for u, prev_r, next_r, gap in transfer_rows:
        u = int(u)
        if u not in region_seqs:
            region_seqs[u] = [region_idx[prev_r]]
        region_seqs[u].append(region_idx[next_r])
        if gap is not None:
            gap_seqs.setdefault(u, []).append(gap_idx[gap])
    ep_seqs: dict[int, list[int]] = {}
    for u, _, bin_ in episode_rows:
        ep_seqs.setdefault(int(u), []).append(ep_idx[bin_])

    data_fp = sha256_file(TSV)
    hparams = {"win": WIN, "epochs": EPOCHS, "seed": SEED,
               "regions": REGIONS, "gap_classes": GAP_CLASSES,
               "ep_bins": EP_BINS, "val_user_pct": VAL_PCT,
               "split": "sha256(userid) % 100", "spec_version": spec_version()}

    def split(seqs: dict[int, list[int]]):
        tr = [s for u, s in sorted(seqs.items()) if is_train_user(u)]
        va = [s for u, s in sorted(seqs.items()) if not is_train_user(u)]
        return tr, va

    def fit():
        files: dict[str, bytes] = {}
        meta: dict = {"spec_version": spec_version(), "device": str(device),
                      "split": f"user-id hash 80/{VAL_PCT} (user-level; the "
                               "whole history of a held-out user is held out)"}

        # ---- NextRegionGRU (gate: return-to-previous region) ---------------
        tr_seqs, va_seqs = split(region_seqs)
        xs_tr, ys_tr = build_windows(tr_seqs, len(REGIONS))
        xs_va, ys_va = build_windows(va_seqs, len(REGIONS))
        ux, uy, uw = dedup_weight(xs_tr, ys_tr)
        gru = NextRegionGRU(n_regions=len(REGIONS)).to(device)
        train_loop(gru, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = gru(torch.from_numpy(xs_va).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == ys_va).mean())
        # the gate baseline: return to the region you came from (the window's
        # second-to-last token); with no came-from region yet, fall back to
        # staying in the current region — honest, not an artificial zero
        base = np.where(xs_va[:, -2] != len(REGIONS), xs_va[:, -2], xs_va[:, -1])
        return_prev_acc = float((base == ys_va).mean())
        repeat_acc = float((xs_va[:, -1] == ys_va).mean())
        maj = int(np.bincount(ys_tr, minlength=len(REGIONS)).argmax())
        maj_acc = float((ys_va == maj).mean())
        meta["next_region"] = {
            "k": WIN,
            "metrics": {"val_accuracy": acc,
                        "return_prev_accuracy": return_prev_acc,
                        "repeat_last_accuracy": repeat_acc,
                        "majority_accuracy": maj_acc,
                        "n_val": int(len(ys_va)),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= return_prev_acc),
            "baseline_note": "return-to-previous measured 72.1% at country "
                             "level on the source data; the region-level "
                             "value above is computed on the held-out users",
            "role": "generative teacher -> distilled to DP region_transition "
                    "rows + the return_rate release (second-order returns)"}
        buf = io.BytesIO(); torch.save(gru.cpu().state_dict(), buf)
        files["next_region.pt"] = buf.getvalue()

        # ---- GapClassGRU (gate: marginal class shares) ---------------------
        tr_seqs, va_seqs = split(gap_seqs)
        xs_tr, ys_tr = build_windows(tr_seqs, len(GAP_CLASSES))
        xs_va, ys_va = build_windows(va_seqs, len(GAP_CLASSES))
        ux, uy, uw = dedup_weight(xs_tr, ys_tr)
        ggru = GapClassGRU(n_classes=len(GAP_CLASSES)).to(device)
        train_loop(ggru, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = ggru(torch.from_numpy(xs_va).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == ys_va).mean())
        shares = np.bincount(ys_tr, minlength=len(GAP_CLASSES)) / max(1, len(ys_tr))
        maj = int(shares.argmax())
        maj_acc = float((ys_va == maj).mean())
        meta["gap_class"] = {
            "k": WIN,
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4) for s in shares],
                        "n_val": int(len(ys_va)),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "tempo teacher -> released counterpart is the DP "
                    "gap_class histogram + suppressed gap moments"}
        buf = io.BytesIO(); torch.save(ggru.cpu().state_dict(), buf)
        files["gap_class.pt"] = buf.getvalue()

        # ---- EpisodeShapeModel (gate: unigram NLL) --------------------------
        tr_seqs, va_seqs = split(ep_seqs)
        pairs_tr = [(s[i - 1], s[i]) for s in tr_seqs for i in range(1, len(s))]
        pairs_va = [(s[i - 1], s[i]) for s in va_seqs for i in range(1, len(s))]
        counts = np.bincount([b for s in tr_seqs for b in s],
                             minlength=len(EP_BINS)).astype(np.float64)
        unigram = (counts + 1) / (counts.sum() + len(EP_BINS))   # Laplace
        epm = EpisodeShapeModel(n_bins=len(EP_BINS)).to(device)
        if pairs_tr and pairs_va:
            xp = np.array([p for p, _ in pairs_tr], dtype=np.int64)
            yp = np.array([n for _, n in pairs_tr], dtype=np.int64)
            ux, uy, uw = dedup_weight(xp.reshape(-1, 1), yp)
            train_loop(epm, torch.from_numpy(ux[:, 0]), torch.from_numpy(uy),
                       torch.from_numpy(uw), device, EPOCHS)
            xv = torch.from_numpy(np.array([p for p, _ in pairs_va], dtype=np.int64))
            yv = torch.from_numpy(np.array([n for _, n in pairs_va], dtype=np.int64))
            with torch.no_grad():
                model_nll = float(nn.CrossEntropyLoss()(epm(xv.to(device)),
                                                        yv.to(device)))
            unigram_nll = float(np.mean([-math.log(unigram[n])
                                         for _, n in pairs_va]))
            adopted = bool(model_nll <= unigram_nll)
        else:
            model_nll, unigram_nll, adopted = None, None, False
        meta["episode_shape"] = {
            "metrics": {"val_nll": model_nll, "unigram_nll": unigram_nll,
                        "train_unigram": [round(float(p), 4) for p in unigram],
                        "n_val_pairs": int(len(pairs_va))},
            "adopted": adopted,
            "role": "journey-shape teacher -> released counterpart is the DP "
                    "episode_len histogram"}
        buf = io.BytesIO(); torch.save(epm.cpu().state_dict(), buf)
        files["episode_shape.pt"] = buf.getvalue()

        # ---- UserProfileAE (anomaly scorer; no adoption gate — scoring) ----
        uids = np.array([int(r[0]) for r in user_rows])
        raw = np.array([[float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5])] for r in user_rows],
                       dtype=np.float32)
        # compress the heavy tails (log1p on counts and gaps; rate -> [0,1])
        feats = np.stack([np.log1p(raw[:, 0]), np.log1p(raw[:, 1]),
                          raw[:, 2] / 100.0,
                          np.log1p(raw[:, 3]), np.log1p(raw[:, 4])], axis=1)
        tr_m = np.array([is_train_user(u) for u in uids])
        mu = feats[tr_m].mean(axis=0)
        sd = feats[tr_m].std(axis=0) + 1e-6
        ae = UserProfileAE(n_in=feats.shape[1]).to(device)
        xs = torch.from_numpy((feats[tr_m] - mu) / sd)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
        ae.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(xs))
            for i in range(0, len(xs), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                x = xs[idx].to(device)
                loss = ((ae(x) - x) ** 2).mean()
                loss.backward()
                opt.step()
        ae.eval()
        with torch.no_grad():
            va = torch.from_numpy((feats[~tr_m] - mu) / sd).to(device)
            val_err = float(((ae(va) - va) ** 2).mean())
        meta["profile_ae"] = {
            "metrics": {"val_mse": val_err,
                        "scaler_mean": [float(x) for x in mu],
                        "scaler_std": [float(x) for x in sd],
                        "n_val_users": int((~tr_m).sum())},
            "features": PROFILE_FEATURES,
            "transform": "log1p on counts and gap days; return_rate_pct / 100",
            "adopted": True,
            "role": "scoring only (never distilled; anomaly scores stay "
                    "enclave-side)"}
        buf = io.BytesIO(); torch.save(ae.cpu().state_dict(), buf)
        files["profile_ae.pt"] = buf.getvalue()

        meta["numeric_note"] = ("gaps and durations modelled in SECONDS; "
                                "profile gaps in days (feat_users convention)")
        return files, meta

    version, trained = train_or_reuse(
        "minimodels", data_fp, hparams,
        [Path(__file__)], fit)
    print(f"train: minimodels version {version} "
          f"({'trained now' if trained else 'reused existing artifact'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

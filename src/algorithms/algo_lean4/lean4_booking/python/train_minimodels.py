#!/usr/bin/env python3
"""Fit the booking minimodels into the content-addressed artifact store.

Five models, each answering one modelling question, each gated by a frozen
non-ML baseline (honest negatives are kept):

  NextCountryMLP    P(next hotel-country tier | prev tier, leg position,
                    booker country, device)
                    gate: RETURN-TO-PREVIOUS country — predict the tier the
                    user came from (the dominant real pattern: 38.07% of
                    trips revisit a city, and consecutive same-country hops
                    are common; computed honestly here on held-out users)
  TripLenMLP        P(user's reservations-per-user bin | first reservation
                    features) gate: majority class
  StayNightsMLP     P(stay nights cell | country tier, month, device)
                    gate: majority class (n_1 dominates: median stay 1 night)
  DeviceMLP         P(device cell | booker country, affiliate tier, month)
                    gate: majority class (desktop dominates)
  UserProfileAE     anomaly/embedding over per-user country-tier histograms;
                    scoring model, no gate — stays enclave-side forever

Split: 80/20 ON USER ID HASH, fixed before fitting (the privacy unit is the
user, so leakage control must be user-level, not time-level). Duplicate
(input, label) training rows are deduplicated and weighted — the alphabets
are tiny, so dedup cuts the epoch cost sharply without changing the loss.
Artifacts: content-addressed via python/artifacts.py (BOOKING_RETRAIN=1 to
activate a retrain; BOOKING_DEVICE=cpu for the deterministic mode).

The architectures below are rebuilt identically by
dbt/models/ml/ml_trip_scores.py.
"""
from __future__ import annotations

import hashlib
import io
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
from spec import (AFF_CELLS, BOOKER_CELLS, CT_CELLS, DEVICE_CELLS,  # noqa: E402
                  MONTH_CELLS, NIGHTS_CELLS, RC_BINS, spec_version)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"
CSV = ROOT / "combined_set.csv"

EPOCHS = int(os.environ.get("BOOKING_EPOCHS", "30"))
SEED = 1234
VAL_PCT = 20      # users with sha256(userid) % 100 >= 80 are held out


def pick_device() -> torch.device:
    forced = os.environ.get("BOOKING_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_train_user(userid: int) -> bool:
    """Deterministic user-level 80/20 split (the whole history of a user is
    one record, so every row of a held-out user is held out)."""
    h = int(hashlib.sha256(str(int(userid)).encode()).hexdigest(), 16)
    return h % 100 < 100 - VAL_PCT


# ---------------------------------------------------------------------------
# architectures (rebuilt identically by dbt/models/ml/ml_trip_scores.py)
# ---------------------------------------------------------------------------
def _mlp(n_in: int, n_hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(),
                         nn.Linear(n_hidden, n_out))


class NextCountryMLP(nn.Module):
    """P(next tier | one-hot prev tier, leg position, booker, device)."""

    def __init__(self, n_tiers: int = 16, n_booker: int = 5,
                 n_device: int = 3, hidden: int = 32):
        super().__init__()
        self.net = _mlp(n_tiers + 1 + n_booker + n_device, hidden, n_tiers)

    def forward(self, x):
        return self.net(x)


class TripLenMLP(nn.Module):
    """P(rc bin | first-leg tier, booker, device, month)."""

    def __init__(self, n_tiers: int = 16, n_booker: int = 5,
                 n_device: int = 3, n_month: int = 12, hidden: int = 32):
        super().__init__()
        self.net = _mlp(n_tiers + n_booker + n_device + n_month,
                        hidden, 7)

    def forward(self, x):
        return self.net(x)


class StayNightsMLP(nn.Module):
    """P(nights cell | tier, month, device)."""

    def __init__(self, n_tiers: int = 16, n_month: int = 12,
                 n_device: int = 3, hidden: int = 32):
        super().__init__()
        self.net = _mlp(n_tiers + n_month + n_device, hidden, 5)

    def forward(self, x):
        return self.net(x)


class DeviceMLP(nn.Module):
    """P(device cell | booker, affiliate tier, month)."""

    def __init__(self, n_booker: int = 5, n_aff: int = 9,
                 n_month: int = 12, hidden: int = 32):
        super().__init__()
        self.net = _mlp(n_booker + n_aff + n_month, hidden, 3)

    def forward(self, x):
        return self.net(x)


class TripCountMLP(nn.Module):
    """P(trips-per-user bin | first-leg tier, booker, device, month)."""

    def __init__(self, n_tiers: int = 16, n_booker: int = 5,
                 n_device: int = 3, n_month: int = 12, hidden: int = 16):
        super().__init__()
        self.net = _mlp(n_tiers + n_booker + n_device + n_month, hidden, 3)

    def forward(self, x):
        return self.net(x)


class DeviceSeqMLP(nn.Module):
    """P(next-leg device | one-hot prev device, leg position) — the
    within-trip persistence teacher (70.7% of real trips are one device)."""

    def __init__(self, n_device: int = 3, hidden: int = 16):
        super().__init__()
        self.net = _mlp(n_device + 1, hidden, n_device)

    def forward(self, x):
        return self.net(x)


class CountriesCountMLP(nn.Module):
    """P(countries-per-trip bin | first-leg tier, booker, device, month)."""

    def __init__(self, n_tiers: int = 16, n_booker: int = 5,
                 n_device: int = 3, n_month: int = 12, hidden: int = 16):
        super().__init__()
        self.net = _mlp(n_tiers + n_booker + n_device + n_month, hidden, 4)

    def forward(self, x):
        return self.net(x)


class DomesticStartMLP(nn.Module):
    """P(first leg domestic | booker country) — the booker->domestic-start
    teacher (real base rate 12.8%, but 18.7% for Gondal bookers vs 2.1%
    for Devilfire)."""

    def __init__(self, n_booker: int = 5, hidden: int = 8):
        super().__init__()
        self.net = _mlp(n_booker, hidden, 2)

    def forward(self, x):
        return self.net(x)


class AffSeqMLP(nn.Module):
    """P(next-leg affiliate tier | one-hot prev tier) — the within-trip
    channel-persistence teacher (75.4% of legs keep the previous tier)."""

    def __init__(self, n_aff: int = 9, hidden: int = 24):
        super().__init__()
        self.net = _mlp(n_aff, hidden, n_aff)

    def forward(self, x):
        return self.net(x)


class UserProfileAE(nn.Module):
    """Autoencoder over a user's hotel-country-tier histogram (shares)."""

    def __init__(self, n_in: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),
                                      nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                     nn.Linear(8, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def dedup_weight(xs: np.ndarray, ys: np.ndarray):
    """Collapse duplicate (input, label) rows into counts (training weights).

    The loss is unchanged — duplicates contribute their multiplicity — but
    over a tiny alphabet the epoch cost drops by an order of magnitude."""
    keyed: dict[tuple, int] = {}
    for x, y in zip(xs, ys):
        keyed[(*x, y)] = keyed.get((*x, y), 0) + 1
    uniq = np.array([k[:-1] for k in keyed], dtype=np.float32)
    labs = np.array([k[-1] for k in keyed], dtype=np.int64)
    w = np.array(list(keyed.values()), dtype=np.float32)
    return uniq, labs, w


def train_loop(model, xs, ys, ws, device, epochs, lr=1e-2, batch=1024):
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

    ct_idx = {c: i for i, c in enumerate(CT_CELLS)}
    booker_idx = {c: i for i, c in enumerate(BOOKER_CELLS)}
    dev_idx = {c: i for i, c in enumerate(DEVICE_CELLS)}
    aff_idx = {c: i for i, c in enumerate(AFF_CELLS)}
    month_idx = {c: i for i, c in enumerate(MONTH_CELLS)}
    nights_idx = {c: i for i, c in enumerate(NIGHTS_CELLS)}
    rc_idx = {c: i for i, c in enumerate(RC_BINS)}

    # transitions with the PREVIOUS leg's device/booker (the booking context
    # of the leg the user is currently on), leg position scaled to [0,1]
    trans_rows = con.execute(
        "select f.userid, f.prev_ct, f.next_ct, f.from_leg, "
        "s.booker_country, s.device_class "
        "from feat_transitions f "
        "join stg_reservations s on s.utrip = f.utrip "
        "and s.leg_idx = f.from_leg "
        "order by f.userid, f.utrip, f.from_leg").fetchall()
    # first reservation per user + the user's rc bin
    first_rows = con.execute(
        "with first_leg as ("
        "  select userid, hotel_ct, booker_country, device_class, "
        "  month_cell, row_number() over (partition by userid "
        "  order by checkin, utrip, leg_idx) rn "
        "  from stg_reservations) "
        "select f.userid, f.hotel_ct, f.booker_country, f.device_class, "
        "f.month_cell, u.rc_bin "
        "from first_leg f join feat_users u on u.userid = f.userid "
        "where f.rn = 1").fetchall()
    # per-reservation nights cells
    nights_rows = con.execute(
        "select userid, hotel_ct, month_cell, device_class, nights_cell "
        "from stg_reservations order by userid").fetchall()
    # per-reservation device cells
    dev_rows = con.execute(
        "select userid, booker_country, aff_tier, month_cell, device_class "
        "from stg_reservations order by userid").fetchall()
    # per-user country-tier histograms (shares over the user's reservations)
    hist_rows = con.execute(
        "select userid, list(hotel_ct order by checkin) "
        "from stg_reservations group by userid order by userid").fetchall()
    # 1.3.0: trips-per-user bins (from feat_users)
    tp_rows = con.execute(
        "select userid, tp_bin from feat_users order by userid").fetchall()
    # 1.3.0: per-leg device WITH the previous leg's device (persistence
    # teacher): the sequence teaches whether the next leg switches
    devseq_rows = con.execute(
        "select userid, prev_device, device_class, leg_idx from ("
        "select userid, device_class, "
        "lag(device_class) over (partition by utrip order by leg_idx) "
        "prev_device, leg_idx from stg_reservations) "
        "where prev_device is not null order by userid").fetchall()
    # 1.4.0: countries-per-trip bins (first trip per user)
    cpt_rows = con.execute(
        "select userid, case n_countries when 1 then 'cpt_1' "
        "when 2 then 'cpt_2' when 3 then 'cpt_3' else 'cpt_4p' end "
        "from (select userid, n_countries, "
        "row_number() over (partition by userid order by start_day, utrip) rn "
        "from feat_trips) where rn = 1 order by userid").fetchall()
    # 1.4.0: first-leg domestic indicator (hotel == booker country)
    dom_rows = con.execute(
        "select userid, booker_country, cast(hotel_country = booker_country as int) "
        "from (select userid, hotel_country, booker_country, "
        "row_number() over (partition by userid order by checkin, utrip, "
        "leg_idx) rn from stg_reservations) where rn = 1 order by userid"
    ).fetchall()
    # 1.3.0: per-leg affiliate tier with the previous leg's tier
    affseq_rows = con.execute(
        "select userid, prev_aff, aff_tier from ("
        "select userid, aff_tier, "
        "lag(aff_tier) over (partition by utrip order by leg_idx) prev_aff "
        "from stg_reservations) "
        "where prev_aff is not null order by userid").fetchall()
    con.close()

    data_fp = sha256_file(CSV)
    hparams = {"epochs": EPOCHS, "seed": SEED,
               "ct_cells": CT_CELLS, "nights_cells": NIGHTS_CELLS,
               "rc_bins": RC_BINS, "device_cells": DEVICE_CELLS,
               "booker_cells": BOOKER_CELLS, "aff_cells": AFF_CELLS,
               "month_cells": MONTH_CELLS,
               "val_user_pct": VAL_PCT,
               "split": "sha256(userid) % 100", "spec_version": spec_version()}

    def fit():
        files: dict[str, bytes] = {}
        meta: dict = {"spec_version": spec_version(), "device": str(device),
                      "split": f"user-id hash 80/{VAL_PCT} (user-level; the "
                               "whole history of a held-out user is held out)"}

        # ---- NextCountryMLP (gate: return-to-previous) ----------------------
        feats = []
        for _, p, nxt, leg, bk, dv in trans_rows:
            feats.append((onehot(ct_idx[p], len(CT_CELLS)),
                          np.array([min(float(leg), 16.0) / 16.0],
                                   dtype=np.float32),
                          onehot(booker_idx[bk], len(BOOKER_CELLS)),
                          onehot(dev_idx[dv], len(DEVICE_CELLS)),
                          ct_idx[nxt]))
        X = np.array([np.concatenate(f[:4]) for f in feats], dtype=np.float32)
        Y = np.array([f[4] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in trans_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        mlp = NextCountryMLP().to(device)
        train_loop(mlp, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = mlp(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        # gate baseline: return to the tier you came from = stay in the
        # current tier (prev tier of the transition) — honest, measured on
        # the held-out users
        stay_acc = float((np.array([ct_idx[r[1]] for r in trans_rows])[~m_tr]
                          == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=len(CT_CELLS)).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        meta["next_country"] = {
            "metrics": {"val_accuracy": acc,
                        "return_prev_accuracy": stay_acc,
                        "majority_accuracy": maj_acc,
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= max(stay_acc, maj_acc)),
            "baseline_note": "return-to-previous measured 38.07% "
                             "repeat-city trips / 99.11% contiguous legs on "
                             "the source data; the tier-level value above is "
                             "computed on the held-out users",
            "role": "generative teacher -> distilled to DP "
                    "country_transition rows + the return_rate release "
                    "(second-order returns)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(mlp.cpu().state_dict(), buf)
        files["next_country.pt"] = buf.getvalue()

        # ---- TripLenMLP (gate: majority class) ------------------------------
        feats = [(onehot(ct_idx[r[1]], len(CT_CELLS)),
                  onehot(booker_idx[r[2]], len(BOOKER_CELLS)),
                  onehot(dev_idx[r[3]], len(DEVICE_CELLS)),
                  onehot(month_idx[r[4]], len(MONTH_CELLS)),
                  rc_idx[r[5]]) for r in first_rows]
        X = np.array([np.concatenate(f[:4]) for f in feats], dtype=np.float32)
        Y = np.array([f[4] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in first_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        tlm = TripLenMLP().to(device)
        train_loop(tlm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = tlm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=len(RC_BINS)).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        shares = np.bincount(Y[m_tr], minlength=len(RC_BINS)) / max(1, m_tr.sum())
        meta["trip_len_class"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4)
                                                  for s in shares],
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "trip-shape teacher -> released counterpart is the DP "
                    "res_count histogram",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(tlm.cpu().state_dict(), buf)
        files["trip_len_class.pt"] = buf.getvalue()

        # ---- StayNightsMLP (gate: majority class) ---------------------------
        feats = [(onehot(ct_idx[r[1]], len(CT_CELLS)),
                  onehot(month_idx[r[2]], len(MONTH_CELLS)),
                  onehot(dev_idx[r[3]], len(DEVICE_CELLS)),
                  nights_idx[r[4]]) for r in nights_rows]
        X = np.array([np.concatenate(f[:3]) for f in feats], dtype=np.float32)
        Y = np.array([f[3] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in nights_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        snm = StayNightsMLP().to(device)
        train_loop(snm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = snm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=len(NIGHTS_CELLS)).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        shares = np.bincount(Y[m_tr], minlength=len(NIGHTS_CELLS)) / max(1, m_tr.sum())
        meta["stay_nights"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4)
                                                  for s in shares],
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "stay-length teacher -> released counterpart is the DP "
                    "nights histogram",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(snm.cpu().state_dict(), buf)
        files["stay_nights.pt"] = buf.getvalue()

        # ---- DeviceMLP (gate: majority class) -------------------------------
        feats = [(onehot(booker_idx[r[1]], len(BOOKER_CELLS)),
                  onehot(aff_idx[r[2]], len(AFF_CELLS)),
                  onehot(month_idx[r[3]], len(MONTH_CELLS)),
                  dev_idx[r[4]]) for r in dev_rows]
        X = np.array([np.concatenate(f[:3]) for f in feats], dtype=np.float32)
        Y = np.array([f[3] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in dev_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        dvm = DeviceMLP().to(device)
        train_loop(dvm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = dvm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=len(DEVICE_CELLS)).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        shares = np.bincount(Y[m_tr], minlength=len(DEVICE_CELLS)) / max(1, m_tr.sum())
        meta["device_class"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4)
                                                  for s in shares],
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "booking-device teacher -> released counterpart is the DP "
                    "device histogram (+ the chi2 association cells)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(dvm.cpu().state_dict(), buf)
        files["device_class.pt"] = buf.getvalue()

        # ---- TripCountMLP (gate: majority class; 1.3.0) --------------------
        # P(trips-per-user bin | first-reservation features); teacher for
        # the DP trips_per_user release
        tp_idx = {"tp_1": 0, "tp_2": 1, "tp_3p": 2}
        tp_of = {int(u): tp_idx[b] for u, b in tp_rows}
        feats = [(onehot(ct_idx[r[1]], len(CT_CELLS)),
                  onehot(booker_idx[r[2]], len(BOOKER_CELLS)),
                  onehot(dev_idx[r[3]], len(DEVICE_CELLS)),
                  onehot(month_idx[r[4]], len(MONTH_CELLS)),
                  tp_of[int(r[0])]) for r in first_rows]
        X = np.array([np.concatenate(f[:4]) for f in feats], dtype=np.float32)
        Y = np.array([f[4] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in first_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        tcm = TripCountMLP().to(device)
        train_loop(tcm, torch.from_numpy(ux), torch.from_numpy(uy),
                    torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = tcm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=3).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        shares = np.bincount(Y[m_tr], minlength=3) / max(1, m_tr.sum())
        meta["trip_count"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4)
                                                  for s in shares],
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "trips-per-user teacher -> released counterpart is the "
                    "DP trips_per_user histogram (the multi-trip rendering)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(tcm.cpu().state_dict(), buf)
        files["trip_count.pt"] = buf.getvalue()

        # ---- DeviceSeqMLP (gate: stay-on-same-device; 1.3.0) ---------------
        # P(next-leg device | prev device, leg position) — the persistence
        # teacher for the DP device_persist release
        feats = [(onehot(dev_idx[r[1]], len(DEVICE_CELLS)),
                  np.array([min(float(r[3]), 16.0) / 16.0], dtype=np.float32),
                  dev_idx[r[2]]) for r in devseq_rows]
        X = np.array([np.concatenate(f[:2]) for f in feats], dtype=np.float32)
        Y = np.array([f[2] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in devseq_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        dsm = DeviceSeqMLP().to(device)
        train_loop(dsm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = dsm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        same_acc = float((np.array([dev_idx[r[1]] for r in devseq_rows])[~m_tr]
                          == Y[~m_tr]).mean())
        meta["device_seq"] = {
            "metrics": {"val_accuracy": acc,
                        "stay_same_device_accuracy": same_acc,
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= same_acc),
            "role": "device-persistence teacher -> released counterpart is "
                    "the DP device_persist histogram (per-trip device plans)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(dsm.cpu().state_dict(), buf)
        files["device_seq.pt"] = buf.getvalue()

        # ---- AffSeqMLP (gate: stay-on-same-tier; 1.3.0) --------------------
        # P(next-leg affiliate tier | prev tier) — the persistence teacher
        # for the DP affiliate_persist release (75.4% of real legs keep
        # the previous leg's tier)
        feats = [(onehot(aff_idx[r[1]], len(AFF_CELLS)),
                  aff_idx[r[2]]) for r in affseq_rows]
        X = np.array([f[0] for f in feats], dtype=np.float32)
        Y = np.array([f[1] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in affseq_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        asm = AffSeqMLP().to(device)
        train_loop(asm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = asm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        same_acc = float((np.array([aff_idx[r[1]] for r in affseq_rows])[~m_tr]
                          == Y[~m_tr]).mean())
        meta["aff_seq"] = {
            "metrics": {"val_accuracy": acc,
                        "stay_same_tier_accuracy": same_acc,
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= same_acc),
            "role": "affiliate-persistence teacher -> released counterpart "
                    "is the DP affiliate_persist histogram (per-trip "
                    "affiliate plans)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(asm.cpu().state_dict(), buf)
        files["aff_seq.pt"] = buf.getvalue()

        # ---- CountriesCountMLP (gate: majority; 1.4.0) --------------------
        # P(countries-per-trip bin | first-reservation features); teacher
        # for the DP countries_per_trip release (the trip's country budget)
        cpt_idx = {"cpt_1": 0, "cpt_2": 1, "cpt_3": 2, "cpt_4p": 3}
        cpt_of = {int(u): cpt_idx[b] for u, b in cpt_rows}
        feats = [(onehot(ct_idx[r[1]], len(CT_CELLS)),
                  onehot(booker_idx[r[2]], len(BOOKER_CELLS)),
                  onehot(dev_idx[r[3]], len(DEVICE_CELLS)),
                  onehot(month_idx[r[4]], len(MONTH_CELLS)),
                  cpt_of[int(r[0])]) for r in first_rows]
        X = np.array([np.concatenate(f[:4]) for f in feats], dtype=np.float32)
        Y = np.array([f[4] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in first_rows], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        ccm = CountriesCountMLP().to(device)
        train_loop(ccm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = ccm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        maj = int(np.bincount(Y[m_tr], minlength=4).argmax())
        maj_acc = float((Y[~m_tr] == maj).mean())
        meta["countries_count"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "countries-per-trip teacher -> released counterpart is "
                    "the DP countries_per_trip histogram (the walk's "
                    "country budget)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(ccm.cpu().state_dict(), buf)
        files["countries_count.pt"] = buf.getvalue()

        # ---- DomesticStartMLP (gate: base rate; 1.4.0) -------------------
        # P(first leg domestic | booker); teacher for the DP domestic_start
        # release (the generator's domestic first-leg draw)
        feats = [(onehot(booker_idx[r[1]], len(BOOKER_CELLS)), int(r[2]))
                 for r in dom_rows if r[1] in booker_idx]
        X = np.array([f[0] for f in feats], dtype=np.float32)
        Y = np.array([f[1] for f in feats], dtype=np.int64)
        U = np.array([int(r[0]) for r in dom_rows
                      if r[1] in booker_idx], dtype=np.int64)
        m_tr = np.array([is_train_user(u) for u in U])
        ux, uy, uw = dedup_weight(X[m_tr], Y[m_tr])
        dsm = DomesticStartMLP().to(device)
        train_loop(dsm, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = dsm(torch.from_numpy(X[~m_tr]).to(device)) \
                .argmax(1).cpu().numpy()
        acc = float((pred == Y[~m_tr]).mean())
        base = float((Y[~m_tr] == 0).mean())
        meta["domestic_start"] = {
            "metrics": {"val_accuracy": acc, "base_rate_accuracy": base,
                        "n_val": int((~m_tr).sum()),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= base),
            "role": "booker->domestic-start teacher -> released counterpart "
                    "is the DP domestic_start histogram (P(domestic | "
                    "booker) with the booker release)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(dsm.cpu().state_dict(), buf)
        files["domestic_start.pt"] = buf.getvalue()

        # ---- UserProfileAE (anomaly scorer; no gate — scoring) -------------
        uids = np.array([int(r[0]) for r in hist_rows], dtype=np.int64)
        H = np.zeros((len(hist_rows), len(CT_CELLS)), dtype=np.float32)
        for i, (_, cts) in enumerate(hist_rows):
            for c in cts:
                H[i, ct_idx[c]] += 1.0
        tot = H.sum(axis=1, keepdims=True)
        H = H / np.maximum(tot, 1.0)          # shares in [0,1]
        m_tr = np.array([is_train_user(u) for u in uids])
        mu = H[m_tr].mean(axis=0)
        sd = H[m_tr].std(axis=0) + 1e-6
        ae = UserProfileAE(n_in=H.shape[1]).to(device)
        xs = torch.from_numpy((H[m_tr] - mu) / sd)
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
            va = torch.from_numpy((H[~m_tr] - mu) / sd).to(device)
            val_err = float(((ae(va) - va) ** 2).mean())
        meta["profile_ae"] = {
            "metrics": {"val_mse": val_err,
                        "scaler_mean": [float(x) for x in mu],
                        "scaler_std": [float(x) for x in sd],
                        "n_val_users": int((~m_tr).sum())},
            "features": [f"ct_share_{c}" for c in CT_CELLS],
            "transform": "per-user country-tier histogram shares",
            "adopted": True,
            "role": "scoring only (never distilled; anomaly scores stay "
                    "enclave-side)",
            "release_status": "internal_unreviewed"}
        buf = io.BytesIO(); torch.save(ae.cpu().state_dict(), buf)
        files["profile_ae.pt"] = buf.getvalue()

        meta["numeric_note"] = ("nights and gaps modelled in DAYS "
                                "(epoch-day convention); nights in [1,30], "
                                "gaps in [0,33]")
        return files, meta

    version, trained = train_or_reuse(
        "minimodels", data_fp, hparams,
        [Path(__file__)], fit)
    print(f"train: minimodels version {version} "
          f"({'trained now' if trained else 'reused existing artifact'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

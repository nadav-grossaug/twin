# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-trip predictions. Rebuilding this mart never refits a model; if no
# artifact is active yet it materializes an 'untrained' placeholder so
# downstream SQL still compiles.
# The model classes mirror the fitting pipeline in python/ EXACTLY (it is
# the source of truth for the saved state_dicts):
#   NextCountryMLP  25-dim one-hot input (prev tier 16 + leg position 1 +
#                   booker 5 + device 3) -> 16, hidden 32
#   TripLenMLP      36-dim one-hot input (first-leg tier 16 + booker 5 +
#                   device 3 + month 12) -> 7, hidden 32
#   StayNightsMLP   31-dim one-hot input (tier 16 + month 12 + device 3)
#                   -> 5, hidden 32
#   UserProfileAE   16-dim per-user country-tier share histogram (scaler
#                   from meta.json profile_ae.metrics)
# artifact files: next_country.pt, trip_len_class.pt, stay_nights.pt,
# profile_ae.pt (device_class.pt is scored only by scripts/realism_audit.py).
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib",
            "Elbonia", "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca",
            "Patusan", "Rolisica", "Kasnia", "ct_other"]
RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
NIGHTS_CELLS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
BOOKERS = ["Gondal", "The Devilfire Empire", "Elbonia", "Tcherkistan",
           "Bartovia"]
DEVICES = ["desktop", "mobile", "tablet"]
MONTHS = [f"m{i:02d}" for i in range(1, 13)]


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
        "userid": df.userid,
        "utrip": df.utrip,
        "anomaly_score": np.nan,
        "pred_next_ct": None,
        "pred_next_ct_confidence": np.nan,
        "p_next_ct_correct": np.nan,
        "pred_trip_len_bin": None,
        "actual_trip_len_bin": df.actual_trip_len_bin,
        "pred_nights_cell": None,
        "actual_nights_cell": df.actual_nights_cell,
        "model_version": "untrained",
    })


def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def _scores(df, legs, trans, users, version, vdir):
    import torch
    import torch.nn as nn

    meta = json.loads((vdir / "meta.json").read_text())

    # architectures copied verbatim from the fitting pipeline
    def _mlp(n_in, n_hidden, n_out):
        return nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(),
                             nn.Linear(n_hidden, n_out))

    class NextCountryMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(16 + 1 + 5 + 3, 32, 16)

        def forward(self, x):
            return self.net(x)

    class TripLenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(16 + 5 + 3 + 12, 32, 7)

        def forward(self, x):
            return self.net(x)

    class StayNightsMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(16 + 12 + 3, 32, 5)

        def forward(self, x):
            return self.net(x)

    class UserProfileAE(nn.Module):
        def __init__(self, n_in=16):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),
                                         nn.Linear(8, 2))
            self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                         nn.Linear(8, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    ct_idx = {c: i for i, c in enumerate(CT_CELLS)}
    bk_idx = {c: i for i, c in enumerate(BOOKERS)}
    dv_idx = {c: i for i, c in enumerate(DEVICES)}
    mo_idx = {c: i for i, c in enumerate(MONTHS)}
    n = len(df)
    pred_next = np.full(n, None, dtype=object)
    pred_next_conf = np.full(n, np.nan)
    p_next_correct = np.full(n, np.nan)
    pred_len = np.full(n, None, dtype=object)
    pred_nights = np.full(n, None, dtype=object)

    ncm = NextCountryMLP()
    ncm.load_state_dict(torch.load(vdir / "next_country.pt",
                                   map_location="cpu"))
    ncm.eval()
    tlm = TripLenMLP()
    tlm.load_state_dict(torch.load(vdir / "trip_len_class.pt",
                                   map_location="cpu"))
    tlm.eval()
    snm = StayNightsMLP()
    snm.load_state_dict(torch.load(vdir / "stay_nights.pt",
                                   map_location="cpu"))
    snm.eval()
    ae = UserProfileAE()
    ae.load_state_dict(torch.load(vdir / "profile_ae.pt",
                                  map_location="cpu"))
    ae.eval()

    row_of = {u: i for i, u in enumerate(df.utrip)}

    # ---- per-user anomaly: tier-share histogram through the profile AE ----
    # mirrors the fitting step: shares over the user's reservations, scaled by the
    # fitting-side scaler frozen in meta.json
    legs_s = legs.sort_values(["userid", "utrip", "leg_idx"])
    hist = np.zeros((len(users), len(CT_CELLS)), dtype=np.float32)
    uid_pos = {u: i for i, u in enumerate(users.userid)}
    for uid, ct in zip(legs_s.userid.to_numpy(), legs_s.hotel_ct.to_numpy()):
        if ct in ct_idx:
            hist[uid_pos[uid], ct_idx[ct]] += 1.0
    tot = hist.sum(axis=1, keepdims=True)
    hist = hist / np.maximum(tot, 1.0)
    mu = np.array(meta["profile_ae"]["metrics"]["scaler_mean"],
                  dtype=np.float32)
    sd = np.array(meta["profile_ae"]["metrics"]["scaler_std"],
                  dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy((hist - mu) / sd)
        err = ((ae(x) - x) ** 2).mean(dim=1).numpy()
    user_anom = dict(zip(users.userid.to_numpy(), err.astype(float)))
    anomaly = df.userid.map(user_anom).to_numpy(dtype=np.float64)

    with torch.no_grad():
        # ---- next-country head: the trip's LAST transition ----------------
        # (mirrors the fitting join over feat_transitions x stg_reservations:
        #  booker/device come from the leg the user is currently on)
        t_s = trans.sort_values(["utrip", "from_leg"])
        last_t = t_s.groupby("utrip", sort=False).tail(1)
        f_rows, f_tgt, f_refs = [], [], []
        for _, t in last_t.iterrows():
            if t.prev_ct not in ct_idx or t.next_ct not in ct_idx:
                continue
            if t.utrip not in row_of:
                continue
            bk = t.booker_country if t.booker_country in bk_idx else None
            dv = t.device_class if t.device_class in dv_idx else None
            if bk is None or dv is None:
                continue
            f_rows.append(np.concatenate([
                _onehot(ct_idx[t.prev_ct], 16),
                np.array([min(float(t.from_leg), 16.0) / 16.0],
                         dtype=np.float32),
                _onehot(bk_idx[bk], 5), _onehot(dv_idx[dv], 3)]))
            f_tgt.append(ct_idx[t.next_ct])
            f_refs.append(row_of[t.utrip])
        if f_rows:
            probs = torch.softmax(
                ncm(torch.from_numpy(np.array(f_rows, dtype=np.float32))),
                dim=1).numpy()
            for r, ut, tgt in zip(range(len(f_rows)), f_refs, f_tgt):
                k = int(probs[r].argmax())
                pred_next[ut] = CT_CELLS[k]
                pred_next_conf[ut] = float(probs[r][k])
                p_next_correct[ut] = float(probs[r][tgt])

        # ---- trip-length head: the USER'S first reservation ---------------
        # (mirrors the fitting step: first leg per user by checkin/utrip/leg_idx,
        #  target = the user's rc bin; one prediction per trip of that user)
        first = legs.sort_values(
            ["checkin", "utrip", "leg_idx"]).groupby("userid", sort=False
                                                     ).head(1)
        fl_rows, fl_refs = [], []
        for _, r in first.iterrows():
            bk = r.booker_country if r.booker_country in bk_idx else None
            dv = r.device_class if r.device_class in dv_idx else None
            mo = r.month_cell if r.month_cell in mo_idx else None
            if r.hotel_ct not in ct_idx or bk is None or dv is None \
                    or mo is None:
                continue
            fl_rows.append(np.concatenate([
                _onehot(ct_idx[r.hotel_ct], 16), _onehot(bk_idx[bk], 5),
                _onehot(dv_idx[dv], 3), _onehot(mo_idx[mo], 12)]))
            fl_refs.append(r.userid)
        if fl_rows:
            lprobs = torch.softmax(
                tlm(torch.from_numpy(np.array(fl_rows, dtype=np.float32))),
                dim=1).numpy()
            uid2pred = {}
            for uid, pr in zip(fl_refs, lprobs):
                uid2pred[uid] = RC_BINS[int(pr.argmax())]
            df_uids = df.userid.to_numpy()
            for i, uid in enumerate(df_uids):
                if uid in uid2pred:
                    pred_len[i] = uid2pred[uid]

        # ---- stay-nights head: the trip's FINAL leg ------------------------
        last_leg = legs.sort_values(["utrip", "leg_idx"]).groupby(
            "utrip", sort=False).tail(1)
        fn_rows, fn_refs = [], []
        for _, r in last_leg.iterrows():
            dv = r.device_class if r.device_class in dv_idx else None
            mo = r.month_cell if r.month_cell in mo_idx else None
            if r.hotel_ct not in ct_idx or dv is None or mo is None:
                continue
            if r.utrip not in row_of:
                continue
            fn_rows.append(np.concatenate([
                _onehot(ct_idx[r.hotel_ct], 16), _onehot(mo_idx[mo], 12),
                _onehot(dv_idx[dv], 3)]))
            fn_refs.append(row_of[r.utrip])
        if fn_rows:
            nprobs = torch.softmax(
                snm(torch.from_numpy(np.array(fn_rows, dtype=np.float32))),
                dim=1).numpy()
            for r, ut in zip(nprobs, fn_refs):
                pred_nights[ut] = NIGHTS_CELLS[int(r.argmax())]

    return pd.DataFrame({
        "userid": df.userid,
        "utrip": df.utrip,
        "anomaly_score": anomaly,
        "pred_next_ct": pred_next,
        "pred_next_ct_confidence": pred_next_conf,
        "p_next_ct_correct": p_next_correct,
        "pred_trip_len_bin": pred_len,
        "actual_trip_len_bin": df.actual_trip_len_bin,
        "pred_nights_cell": pred_nights,
        "actual_nights_cell": df.actual_nights_cell,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    trips = dbt.ref("feat_trips").df()[["userid", "utrip"]]
    legs_df = dbt.ref("stg_reservations").df()[
        ["userid", "utrip", "leg_idx", "checkin", "hotel_ct", "month_cell",
         "device_class", "booker_country", "nights_cell"]]
    trans_df = dbt.ref("feat_transitions").df()[
        ["utrip", "from_leg", "prev_ct", "next_ct", "prev_country",
         "next_country"]]
    # booker/device of the FROM leg (the leg the user is on), as in fitting
    legs_bd = legs_df[["utrip", "leg_idx", "booker_country", "device_class"]]
    trans_df = trans_df.merge(
        legs_bd, left_on=["utrip", "from_leg"],
        right_on=["utrip", "leg_idx"], how="left")
    users_df = dbt.ref("feat_users").df()[["userid", "rc_bin"]]
    df = trips.copy()
    # the trip-length target is the USER's rc bin (the fitting step's target)
    df["actual_trip_len_bin"] = df.userid.map(
        dict(zip(users_df.userid, users_df.rc_bin)))
    last_leg_nights = legs_df.sort_values(
        ["utrip", "leg_idx"]).groupby("utrip").tail(1).set_index(
        "utrip")["nights_cell"]
    df["actual_nights_cell"] = df.utrip.map(last_leg_nights)
    models_dir = Path(os.environ.get(
        "MODELS_DIR",
        os.environ.get("BOOKING_ARTIFACT_ROOT", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        return _placeholder(df)
    return _scores(df, legs_df, trans_df, users_df, active[0], active[1])

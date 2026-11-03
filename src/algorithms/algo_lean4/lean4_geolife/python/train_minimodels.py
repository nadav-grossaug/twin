#!/usr/bin/env python3
"""Fit the GeoLife mobility minimodels into the content-addressed artifact store.

Six models, each answering one modelling question, each gated by a frozen
non-ML baseline (honest negatives are kept) unless it is a pure scorer:

  NextRoleGRU    P(next anchor role | the user's role sequence)
                 gate: RETURN-TO-PREVIOUS role — predict the role the user
                 came from (daily mobility is dominated by home<->work
                 pendulums, so coming-from is a brutal baseline; with no
                 came-from role yet the gate repeats the current role);
                 repeat-last and majority accuracies are reported alongside
  DwellClassGRU  P(next dwell class | dwell-class history) over UNCENSORED
                 visits only (a censored dwell has no honest class);
                 gate: marginal class shares (majority class)
  ModeLegMLP     P(mode class | leg kinematics) from 7 standardized features;
                 gate: a fixed speed-threshold rule on median_speed_mps
                 (< 1.8 walk, < 5 bike, < 12 bus, < 25 car, else rail) — the
                 rule never predicts 'other', which is its honest weakness
  UserProfileAE  anomaly/embedding over per-user profile vectors (n_days,
                 staypoints/day, motif shares, anchors, median dwell, visits);
                 scoring model, no gate — stays enclave-side forever
  RouteShapeGRU  P(next route token | token history) where a token is a
                 seg_class x ja_class pair (vocab 12), one token sequence
                 per leg (sequence_uuid), ordered by junction;
                 gate: marginal majority token accuracy
  LegShapeAE     per-leg route-shape anomaly scorer (path, circuity,
                 junction density, segment structure from feat_route_legs);
                 scoring model, no gate — stays enclave-side forever

Split: 80/20 ON USER ID HASH, fixed before fitting (the privacy unit is the
user, so leakage control must be user-level, not time-level). GeoLife user
ids are zero-padded strings ('010') and are hashed as-is. Visit streams are
split into separate sequences at seq_ok=false (an observation break: the
visit does not continue from the previous one), so no window ever spans a
break. Duplicate (window, label) training pairs are deduplicated and
weighted — role and dwell alphabets are tiny, so dedup cuts the epoch cost
sharply without changing the loss. Artifacts: content-addressed via
python/artifacts.py (FLOW_RETRAIN=1 to activate a retrain; FLOW_DEVICE=cpu
for the deterministic mode).
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
from spec import DWELL_CLASSES, MODE_CLASSES, ROLES, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"
TSV = ROOT / "geolife_data.tsv"

LEG_FEATURES = ["log1p_path_m", "log1p_duration_s", "median_speed_mps",
                "p95_speed_mps", "median_abs_turn_deg", "straightness",
                "log1p_median_dt_s"]
PROFILE_FEATURES = ["n_days", "mean_sp_per_day", "share_commute",
                    "share_home", "n_anchors", "median_dwell_s", "n_visits"]
# route-shape vocabulary (fixed orders — token id = seg_idx * 3 + ja_idx)
SEG_CLASSES = ["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"]
JA_CLASSES = ["ja_30_60", "ja_60_120", "ja_120_180"]
ROUTE_TOKENS = [f"{s}|{a}" for s in SEG_CLASSES for a in JA_CLASSES]  # vocab 12
LEG_SHAPE_FEATURES = ["log1p_path_m", "circuity_clip6", "junctions_per_km",
                      "log1p_mean_seg_m", "straightness", "log1p_n_junctions"]
WIN = 8           # window length (visits) for the GRUs
EPOCHS = int(os.environ.get("FLOW_EPOCHS", "30"))
SEED = 1234
VAL_PCT = 20      # users with sha256(user_id) % 100 >= 80 are held out


def pick_device() -> torch.device:
    forced = os.environ.get("FLOW_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_train_user(user_id: str) -> bool:
    """Deterministic user-level 80/20 split (the whole history of a user is
    one record, so every window and leg of a held-out user is held out).
    The zero-padded string id ('010') is hashed verbatim."""
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return h % 100 < 100 - VAL_PCT


# ---------------------------------------------------------------------------
# architectures (rebuilt identically by dbt/models/ml/ml_geo_scores.py)
# ---------------------------------------------------------------------------
class NextRoleGRU(nn.Module):
    def __init__(self, n_roles: int = 3, emb: int = 6, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_roles + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_roles)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class DwellClassGRU(nn.Module):
    def __init__(self, n_classes: int = 5, emb: int = 6, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class ModeLegMLP(nn.Module):
    """Leg kinematics -> mode class (7 standardized features)."""

    def __init__(self, n_in: int = 7, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Linear(32, n_classes))

    def forward(self, x):
        return self.net(x)


class UserProfileAE(nn.Module):
    def __init__(self, n_in: int = 7):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class RouteShapeGRU(nn.Module):
    """P(next route token | history); token = seg_class x ja_class (vocab 12)."""

    def __init__(self, n_tokens: int = 12, emb: int = 8, hidden: int = 24):
        super().__init__()
        self.emb = nn.Embedding(n_tokens + 1, emb)   # +1 pad id
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_tokens)

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.out(h[-1])


class LegShapeAE(UserProfileAE):
    """Per-leg route-shape anomaly scorer (same AE shape over 6 features)."""

    def __init__(self, n_in: int = 6):
        super().__init__(n_in=n_in)


# ---------------------------------------------------------------------------
def build_windows(seqs: list[list[int]], n_vocab: int, k: int = WIN):
    """Per-sequence token lists -> (window, next) pairs; pad id = n_vocab."""
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


def mode_rule(median_speed_mps: np.ndarray) -> np.ndarray:
    """The frozen non-ML gate for ModeLegMLP: median-speed thresholds only.

    < 1.8 walk, < 5 bike, < 12 bus, < 25 car, else rail. The rule never
    predicts 'other' — that is its honest weakness, recorded as-is."""
    idx = {m: i for i, m in enumerate(MODE_CLASSES)}
    return np.select(
        [median_speed_mps < 1.8, median_speed_mps < 5.0,
         median_speed_mps < 12.0, median_speed_mps < 25.0],
        [idx["walk"], idx["bike"], idx["bus"], idx["car"]],
        default=idx["rail"]).astype(np.int64)


def main() -> int:
    if not DB.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    con = duckdb.connect(str(DB), read_only=True)

    # per-user visit streams, in arrival order; seq_ok=false marks the start
    # of a new sequence (the visit does not continue from the previous one)
    visit_rows = con.execute(
        "select user_id, role, dwell_class, seq_ok, arr_censored, dep_censored "
        "from feat_visits order by user_id, ts_arrive_s, sp_id").fetchall()
    # labelled legs; XX rows are quality-flagged and excluded — the
    # unknown-mode marker is never modelled
    leg_rows = con.execute(
        "select user_id, mode_class, path_m, duration_s, median_speed_mps, "
        "p95_speed_mps, coalesce(median_abs_turn_deg, 0), "
        "coalesce(straightness, 1), median_dt_s from feat_legs "
        "where mode_class != 'XX' order by user_id, leg_id").fetchall()
    user_rows = con.execute(
        "select user_id, coalesce(n_days, 0), coalesce(mean_sp_per_day, 0), "
        "coalesce(share_commute, 0), coalesce(share_home, 0), "
        "coalesce(n_anchors, 0), coalesce(median_dwell_s, 0), "
        "coalesce(n_visits, 0) from mart_user_stats order by user_id").fetchall()
    # route tokens: ordered junction-to-junction segments, one sequence per
    # leg (sequence_uuid); token = seg_class x ja_class
    seg_rows = con.execute(
        "select user_id, trajectory_id, sequence_uuid, seg_class, ja_class "
        "from feat_route_segments "
        "order by user_id, trajectory_id, sequence_uuid, junction_idx").fetchall()
    # per-leg shape vectors; XX rows are quality-flagged and excluded — the
    # unknown-mode marker is never modelled
    rleg_rows = con.execute(
        "select user_id, path_m, coalesce(circuity, 1), "
        "coalesce(junctions_per_km, 0), coalesce(mean_seg_m, 0), "
        "coalesce(straightness, 1), n_junctions from feat_route_legs "
        "where mode_class != 'XX' order by user_id, leg_id").fetchall()
    con.close()

    role_idx = {r: i for i, r in enumerate(ROLES)}
    dwell_idx = {d: i for i, d in enumerate(DWELL_CLASSES)}
    mode_idx = {m: i for i, m in enumerate(MODE_CLASSES)}

    by_user: dict[str, list[tuple]] = {}
    for u, role, dwell, seq_ok, arr_c, dep_c in visit_rows:
        by_user.setdefault(u, []).append((role, dwell, seq_ok, arr_c, dep_c))

    role_seqs: dict[str, list[list[int]]] = {}
    dwell_seqs: dict[str, list[list[int]]] = {}
    for u, rows in by_user.items():
        r_all: list[list[int]] = []
        d_all: list[list[int]] = []
        r_cur: list[int] = []
        d_cur: list[int] = []
        for role, dwell, seq_ok, arr_c, dep_c in rows:
            if not seq_ok:                       # sequence break
                r_all.append(r_cur)
                d_all.append(d_cur)
                r_cur, d_cur = [], []
            if role in role_idx:
                r_cur.append(role_idx[role])
            # censored dwells have no honest class — the dwell stream keeps
            # only fully uncensored visits (still breaking at seq_ok=false)
            if not arr_c and not dep_c and dwell in dwell_idx:
                d_cur.append(dwell_idx[dwell])
        r_all.append(r_cur)
        d_all.append(d_cur)
        role_seqs[u] = [s for s in r_all if len(s) >= 2]
        dwell_seqs[u] = [s for s in d_all if len(s) >= 2]

    # route-token sequences: one per leg (sequence_uuid), already ordered by
    # (user, trajectory, sequence_uuid, junction_idx); only legs with >= 2
    # tokens yield a (window, next) pair
    seg_i = {s: i for i, s in enumerate(SEG_CLASSES)}
    ja_i = {a: i for i, a in enumerate(JA_CLASSES)}
    route_seqs: dict[str, list[list[int]]] = {}
    cur_key: tuple | None = None
    cur: list[int] = []
    for u, traj, suuid, sc, jc in seg_rows:
        key = (u, traj, suuid)
        if key != cur_key:
            if cur_key is not None and len(cur) >= 2:
                route_seqs.setdefault(cur_key[0], []).append(cur)
            cur_key, cur = key, []
        cur.append(seg_i[sc] * len(JA_CLASSES) + ja_i[jc])
    if cur_key is not None and len(cur) >= 2:
        route_seqs.setdefault(cur_key[0], []).append(cur)

    data_fp = sha256_file(TSV)
    hparams = {"win": WIN, "epochs": EPOCHS, "seed": SEED,
               "roles": ROLES, "dwell_classes": DWELL_CLASSES,
               "mode_classes": MODE_CLASSES, "leg_features": LEG_FEATURES,
               "profile_features": PROFILE_FEATURES,
               "seg_classes": SEG_CLASSES, "ja_classes": JA_CLASSES,
               "leg_shape_features": LEG_SHAPE_FEATURES, "val_user_pct": VAL_PCT,
               "split": "sha256(user_id) % 100", "spec_version": spec_version()}

    def split(seqs: dict[str, list[list[int]]]):
        tr = [s for u, ss in sorted(seqs.items()) if is_train_user(u) for s in ss]
        va = [s for u, ss in sorted(seqs.items()) if not is_train_user(u) for s in ss]
        return tr, va

    def fit():
        files: dict[str, bytes] = {}
        meta: dict = {"spec_version": spec_version(), "device": str(device),
                      "split": f"user-id hash 80/{VAL_PCT} (user-level; the "
                               "whole history of a held-out user is held out)"}

        # ---- NextRoleGRU (gate: return-to-previous role) --------------------
        tr_seqs, va_seqs = split(role_seqs)
        xs_tr, ys_tr = build_windows(tr_seqs, len(ROLES))
        xs_va, ys_va = build_windows(va_seqs, len(ROLES))
        ux, uy, uw = dedup_weight(xs_tr, ys_tr)
        gru = NextRoleGRU(n_roles=len(ROLES)).to(device)
        train_loop(gru, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = gru(torch.from_numpy(xs_va).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == ys_va).mean())
        # the gate baseline: return to the role you came from (the window's
        # second-to-last token); with no came-from role yet, fall back to
        # repeating the current role — honest, not an artificial zero
        base = np.where(xs_va[:, -2] != len(ROLES), xs_va[:, -2], xs_va[:, -1])
        return_prev_acc = float((base == ys_va).mean())
        repeat_acc = float((xs_va[:, -1] == ys_va).mean())
        maj = int(np.bincount(ys_tr, minlength=len(ROLES)).argmax())
        maj_acc = float((ys_va == maj).mean())
        meta["next_role"] = {
            "k": WIN,
            "metrics": {"val_accuracy": acc,
                        "return_prev_accuracy": return_prev_acc,
                        "repeat_last_accuracy": repeat_acc,
                        "majority_accuracy": maj_acc,
                        "n_val": int(len(ys_va)),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= return_prev_acc),
            "baseline_note": "home<->work pendulums make return-to-previous "
                             "a brutal baseline on anchor roles; the value "
                             "above is computed on the held-out users",
            "role": "generative teacher -> released counterpart is the DP "
                    "role-transition grid (mart_dp_transitions)"}
        buf = io.BytesIO(); torch.save(gru.cpu().state_dict(), buf)
        files["next_role.pt"] = buf.getvalue()

        # ---- DwellClassGRU (gate: marginal class shares) --------------------
        tr_seqs, va_seqs = split(dwell_seqs)
        xs_tr, ys_tr = build_windows(tr_seqs, len(DWELL_CLASSES))
        xs_va, ys_va = build_windows(va_seqs, len(DWELL_CLASSES))
        ux, uy, uw = dedup_weight(xs_tr, ys_tr)
        dgru = DwellClassGRU(n_classes=len(DWELL_CLASSES)).to(device)
        train_loop(dgru, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = dgru(torch.from_numpy(xs_va).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == ys_va).mean())
        shares = np.bincount(ys_tr, minlength=len(DWELL_CLASSES)) / max(1, len(ys_tr))
        maj = int(shares.argmax())
        maj_acc = float((ys_va == maj).mean())
        meta["dwell_class"] = {
            "k": WIN,
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4) for s in shares],
                        "n_val": int(len(ys_va)),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "tempo teacher -> released counterpart is the DP "
                    "dwell_class histogram (mart_dp_dwell)"}
        buf = io.BytesIO(); torch.save(dgru.cpu().state_dict(), buf)
        files["dwell_class.pt"] = buf.getvalue()

        # ---- ModeLegMLP (gate: fixed speed-threshold rule) ------------------
        leg_uids = np.array([r[0] for r in leg_rows])
        labels = np.array([mode_idx[r[1]] for r in leg_rows], dtype=np.int64)
        raw = np.array([[float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                         float(r[6]), float(r[7]), float(r[8])]
                        for r in leg_rows], dtype=np.float32)
        # compress the heavy tails (log1p on path, duration and sample step)
        feats = np.stack([np.log1p(raw[:, 0]), np.log1p(raw[:, 1]),
                          raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5],
                          np.log1p(raw[:, 6])], axis=1)
        tr_m = np.array([is_train_user(u) for u in leg_uids])
        mu = feats[tr_m].mean(axis=0)
        sd = feats[tr_m].std(axis=0) + 1e-6
        mlp = ModeLegMLP(n_in=feats.shape[1], n_classes=len(MODE_CLASSES)).to(device)
        train_loop(mlp,
                   torch.from_numpy((feats[tr_m] - mu) / sd),
                   torch.from_numpy(labels[tr_m]),
                   torch.ones(int(tr_m.sum())), device, EPOCHS)
        with torch.no_grad():
            xv = torch.from_numpy((feats[~tr_m] - mu) / sd).to(device)
            pred = mlp(xv).argmax(1).cpu().numpy()
        acc = float((pred == labels[~tr_m]).mean())
        rule_acc = float((mode_rule(raw[~tr_m, 2]) == labels[~tr_m]).mean())
        meta["mode_leg"] = {
            "metrics": {"val_accuracy": acc, "rule_accuracy": rule_acc,
                        "scaler_mean": [float(x) for x in mu],
                        "scaler_std": [float(x) for x in sd],
                        "n_val": int((~tr_m).sum()),
                        "n_train": int(tr_m.sum())},
            "features": LEG_FEATURES,
            "transform": "log1p on path_m, duration_s, median_dt_s; turn "
                         "fillna 0; straightness fillna 1",
            "adopted": bool(acc >= rule_acc),
            "baseline_note": "the speed rule never predicts 'other' — its "
                             "honest weakness; both scores on held-out users",
            "role": "kinematics teacher -> released counterpart is the DP "
                    "mode histogram (mart_dp_mode)"}
        buf = io.BytesIO(); torch.save(mlp.cpu().state_dict(), buf)
        files["mode_leg.pt"] = buf.getvalue()

        # ---- UserProfileAE (anomaly scorer; no adoption gate — scoring) ----
        uids = np.array([r[0] for r in user_rows])
        raw = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                         float(r[5]), float(r[6]), float(r[7])]
                        for r in user_rows], dtype=np.float32)
        # compress the heavy tails (log1p on counts and dwell; shares stay)
        feats = np.stack([np.log1p(raw[:, 0]), raw[:, 1], raw[:, 2],
                          raw[:, 3], np.log1p(raw[:, 4]),
                          np.log1p(raw[:, 5]), np.log1p(raw[:, 6])], axis=1)
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
            "transform": "log1p on n_days, n_anchors, median_dwell_s, "
                         "n_visits; shares and staypoints/day untouched",
            "adopted": True,
            "role": "scoring only (never distilled; anomaly scores stay "
                    "enclave-side)"}
        buf = io.BytesIO(); torch.save(ae.cpu().state_dict(), buf)
        files["profile_ae.pt"] = buf.getvalue()

        # ---- RouteShapeGRU (gate: marginal majority token) ------------------
        tr_seqs, va_seqs = split(route_seqs)
        xs_tr, ys_tr = build_windows(tr_seqs, len(ROUTE_TOKENS))
        xs_va, ys_va = build_windows(va_seqs, len(ROUTE_TOKENS))
        ux, uy, uw = dedup_weight(xs_tr, ys_tr)
        rgru = RouteShapeGRU(n_tokens=len(ROUTE_TOKENS)).to(device)
        train_loop(rgru, torch.from_numpy(ux), torch.from_numpy(uy),
                   torch.from_numpy(uw), device, EPOCHS)
        with torch.no_grad():
            pred = rgru(torch.from_numpy(xs_va).to(device)).argmax(1).cpu().numpy()
        acc = float((pred == ys_va).mean())
        shares = np.bincount(ys_tr, minlength=len(ROUTE_TOKENS)) / max(1, len(ys_tr))
        maj = int(shares.argmax())
        maj_acc = float((ys_va == maj).mean())
        meta["route_shape"] = {
            "k": WIN,
            "tokens": ROUTE_TOKENS,
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_marginal_shares": [round(float(s), 4) for s in shares],
                        "n_val": int(len(ys_va)),
                        "n_train_unique": int(len(ux))},
            "adopted": bool(acc >= maj_acc),
            "role": "road-shape teacher -> released counterparts are the "
                    "seg_hist + junction_hist + per-mode circuity (category-2)"}
        buf = io.BytesIO(); torch.save(rgru.cpu().state_dict(), buf)
        files["route_shape.pt"] = buf.getvalue()

        # ---- LegShapeAE (anomaly scorer; no adoption gate — scoring) --------
        rleg_uids = np.array([r[0] for r in rleg_rows])
        raw = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                         float(r[5]), float(r[6])] for r in rleg_rows],
                       dtype=np.float32)
        # compress the heavy tails; circuity is clipped at 6 (loop artefacts)
        feats = np.stack([np.log1p(raw[:, 0]), np.minimum(raw[:, 1], 6.0),
                          raw[:, 2], np.log1p(raw[:, 3]), raw[:, 4],
                          np.log1p(raw[:, 5])], axis=1)
        tr_m = np.array([is_train_user(u) for u in rleg_uids])
        mu = feats[tr_m].mean(axis=0)
        sd = feats[tr_m].std(axis=0) + 1e-6
        lae = LegShapeAE(n_in=feats.shape[1]).to(device)
        xs = torch.from_numpy((feats[tr_m] - mu) / sd)
        opt = torch.optim.Adam(lae.parameters(), lr=1e-2)
        lae.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(xs))
            for i in range(0, len(xs), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                x = xs[idx].to(device)
                loss = ((lae(x) - x) ** 2).mean()
                loss.backward()
                opt.step()
        lae.eval()
        with torch.no_grad():
            va = torch.from_numpy((feats[~tr_m] - mu) / sd).to(device)
            val_err = float(((lae(va) - va) ** 2).mean())
        meta["leg_shape"] = {
            "metrics": {"val_mse": val_err,
                        "scaler_mean": [float(x) for x in mu],
                        "scaler_std": [float(x) for x in sd],
                        "n_val_legs": int((~tr_m).sum()),
                        "n_train_legs": int(tr_m.sum())},
            "features": LEG_SHAPE_FEATURES,
            "transform": "log1p on path_m, mean_seg_m, n_junctions; circuity "
                         "coalesce 1 then clip at 6; junctions_per_km "
                         "coalesce 0; straightness fillna 1",
            "adopted": True,
            "role": "scoring only (shape anomaly stays enclave-side)"}
        buf = io.BytesIO(); torch.save(lae.cpu().state_dict(), buf)
        files["leg_shape.pt"] = buf.getvalue()

        meta["numeric_note"] = ("dwells, gaps and leg durations modelled in "
                                "SECONDS; distances in meters; speeds in m/s "
                                "(feat_visits/feat_legs convention)")
        return files, meta

    version, trained = train_or_reuse(
        "minimodels", data_fp, hparams,
        [Path(__file__)], fit)
    print(f"train: minimodels version {version} "
          f"({'trained now' if trained else 'reused existing artifact'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

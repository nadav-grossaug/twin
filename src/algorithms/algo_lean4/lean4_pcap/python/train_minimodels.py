#!/usr/bin/env python3
"""Train the two PyTorch minimodels on features materialized by dbt.

Model A  FlowAutoencoder   — numeric per-flow features -> 3-d bottleneck;
                             reconstruction error is the anomaly score.
Model B  ByteDistClassifier — joy's 256-bin byte distribution -> service class
                             (weak labels from destination port taxonomy).
Model C  NextServiceGRU    — per-host flow sequences (service ids + gaps) ->
                             next service class (goal.md guideline 8:
                             P(next | history, time), the sequence model).
Model D  TlsRecordGRU      — TLS record structure (direction/type/size/timing
                             of each record, no payload) -> client-stack
                             fingerprint class: encrypted-traffic
                             fingerprinting from packet shape alone.
Model E  DnsCharLM         — character-level language model over DNS query
                             names; per-name perplexity is the name-shape
                             anomaly score (the DGA/tunneling detector shape).

Ten additions (2026-10-30), same conventions — frozen eval definitions,
time-ordered 80/20 split, a declared non-ML baseline per gated model, honest
negatives kept: ByteDistAE (payload-shape anomaly, ungated), VolumeServiceMLP,
BytesInMLP, DurationBandMLP, PairGapGRU, SessionLenMLP, UnansweredMLP,
TlsSetupBandMLP, DnsKindClassifier, BeaconChannelMLP.

Training is separate from inference (goal.md guideline 3): this script writes a
versioned artifact under models_out/<version>/ and updates models_out/latest.txt;
the dbt Python model ml_flow_scores loads that artifact to materialize scores.
Split is time-based (guideline 6): first 80% of flows train, last 20% validate;
the scaler is fit on train only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from artifacts import artifact_dir, candidate_version, maybe_activate, write_artifact

ROOT = Path(__file__).resolve().parent.parent
SEED = 1216691467  # first flow timestamp of the Nitroba capture

# Frozen evaluation definitions (recorded into every artifact's metadata;
# changing them requires a new spec version). A learned model is ADOPTED only
# if it beats its non-ML baseline on the validation criterion.
EVAL_DEFINITIONS = {
    "classifier": "val accuracy vs majority-class baseline, last 20% of flows by time",
    "next_service": "val accuracy vs repeat-last baseline, last 20% of windows by target time",
    "autoencoder": "val reconstruction MSE (unsupervised anomaly score; no adoption gate)",
    "tls_fingerprint": "val accuracy vs majority-class baseline, last 20% of TLS flows by"
                       " time; inputs are record direction/type/size/timing only (no payload)",
    "dns_name_lm": "val bits/char vs order-0 (unigram) char baseline, last 20% of DNS"
                   " query records by time; adopted only if the LM beats the unigram model",
    "bytedist_ae": "val reconstruction MSE on the normalized byte_dist of payload-bearing"
                   " flows (unsupervised payload-shape anomaly; no adoption gate)",
    "volume_service": "val accuracy vs majority-class baseline, last 20% of flows by time;"
                      " standardized numeric volume/timing features only — no ports",
    "bytes_in": "val MAE of log1p(bytes_in) vs per-service TRAIN-median baseline,"
                " last 20% of flows by time",
    "duration_band": "val accuracy vs majority-class baseline over 5 public duration"
                     " bands (<=0.02/<=0.09/<=0.4/<=2s/longer), last 20% of flows by time",
    "pair_gap": "val MAE of the next log1p(gap_s) vs repeat-last-gap baseline, last 20%"
                " of windows by target time; (sa,da,dp) channels with >=8 flows, K=5",
    "session_len": "val accuracy vs majority-class baseline over session flow-count"
                   " buckets {2,3,4,6,9+}, last 20% of sessions by start time; inputs are"
                   " the session's FIRST flow's features only",
    "unanswered": "val accuracy vs majority-class baseline, last 20% of flows by time;"
                  " OUTBOUND-only features (log bytes_out, log pkts_out, service onehot)",
    "tls_setup": "val accuracy vs majority-class baseline over hs_setup_ms bands"
                 " (<=50/<=200/<=1000ms/longer), last 20% of TLS flows by time",
    "dns_kind": "val accuracy vs majority-class baseline over query kinds (ordinary/"
                " reverse_lookup/service_discovery), last 20% of DNS query records by"
                " time; numeric name-shape features only",
    "beacon_channel": "val accuracy vs majority-class baseline on is_beacon, last 20% of"
                      " channels by first_seen (tiny N — an honest negative is expected)",
}

NUMERIC_FEATURES = [
    "log1p_bytes_out", "log1p_bytes_in", "log1p_pkts_out", "log1p_pkts_in",
    "log1p_duration_ms", "log1p_bytes_per_s", "entropy_filled", "has_entropy",
    "is_zero_byte", "is_unanswered", "src_internal", "dst_internal",
    "hour_sin", "hour_cos",
]

FEATURE_BODY = """
select
    flow_id,
    sa,
    time_start,
    duration_s,
    ln(1 + bytes_out)                        as log1p_bytes_out,
    ln(1 + bytes_in)                         as log1p_bytes_in,
    ln(1 + num_pkts_out)                     as log1p_pkts_out,
    ln(1 + num_pkts_in)                      as log1p_pkts_in,
    ln(1 + duration_s * 1000)                as log1p_duration_ms,
    ln(1 + coalesce(bytes_per_s, 0))         as log1p_bytes_per_s,
    coalesce(entropy, 0)                     as entropy_filled,
    (entropy is not null)::int               as has_entropy,
    is_zero_byte::int                        as is_zero_byte,
    is_unanswered::int                       as is_unanswered,
    src_internal::int                        as src_internal,
    dst_internal::int                        as dst_internal,
    sin(2 * pi() * hour_utc / 24)            as hour_sin,
    cos(2 * pi() * hour_utc / 24)            as hour_cos,
    service_class
from feat_flows
"""

FEATURE_SQL = FEATURE_BODY + "order by time_start, flow_id"

BYTEDIST_SQL = """
select f.flow_id, s.byte_dist, f.service_class
from feat_flows f join stg_flows s using (flow_id)
where s.byte_dist is not null
order by f.time_start, f.flow_id
"""

SEQ_SQL = """
select sa, service_class, time_start
from feat_flows
order by sa, time_start, flow_id
"""

SEQ_K = 5             # window: K previous flows predict the next one
SEQ_MAX_GAP = 600.0   # do not predict across session breaks (dbt session_gap_s)

TLS_K = 12            # TLS records per flow fed to the record-structure model
TLS_MIN_CLASS = 10    # fingerprints with fewer flows fold into 'other'

DNS_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-._"
DNS_PAD, DNS_BOS, DNS_EOS, DNS_UNK = 0, 1, 2, 3
DNS_VOCAB = 4 + len(DNS_CHARS)
DNS_MAX_CHARS = 64    # qname prefix fed to the char LM

DNS_SQL = """
select qname, time_start from feat_dns order by time_start, flow_id, rec_idx
"""

# ---- ten additional minimodels (2026-10-30) --------------------------------

DUR_BANDS_S = [0.02, 0.09, 0.4, 2.0]         # 5 public duration bands
SESS_BUCKETS = [2, 3, 4, 6, 9]               # FlowShape sessionFlowsChoices
TLS_SETUP_BANDS_MS = [50.0, 200.0, 1000.0]   # handshake setup-time bands
PAIR_K = 5                                   # previous log-gaps per window
PAIR_MIN_FLOWS = 8                           # channels below this are skipped
DNS_KIND_CLASSES = ["ordinary", "reverse_lookup", "service_discovery"]

PAIRGAP_SQL = f"""
with ch as (
    select sa, da, dp from feat_flows
    group by 1, 2, 3 having count(*) >= {PAIR_MIN_FLOWS}
)
select f.sa, f.da, f.dp, f.time_start
from feat_flows f join ch using (sa, da, dp)
order by f.sa, f.da, f.dp, f.time_start, f.flow_id
"""

SESSION_SQL = f"""
with flow_feats as ({{body}}),
first_flow as (
    select s.session_id, s.n_flows as session_flows, s.session_start_t, ff.*,
           row_number() over (partition by s.sa, s.session_id
                              order by ff.flow_id) as rn
    from mart_seq_sessions s
    join flow_feats ff on ff.sa = s.sa and ff.time_start = s.session_start_t
    where s.n_flows >= 2
)
select * from first_flow where rn = 1 order by session_start_t, session_id
""".format(body=FEATURE_BODY)

TLS_SETUP_SQL = """
select fingerprint, n_records,
       cast(tls_bytes_out + tls_bytes_in as double) as tls_bytes,
       cast(hs_setup_ms as double)                  as hs_setup_ms
from feat_tls
where hs_setup_ms is not null
order by time_start, flow_id
"""

DNS_KIND_SQL = """
select qname_len, n_labels, max_label_len, min_label_len, digit_frac,
       case when is_reverse_lookup then 1
            when is_service_discovery then 2 else 0 end as kind
from feat_dns
order by time_start, flow_id, rec_idx
"""

BEACON_SQL = """
select ln(1 + n_flows)               as log1p_n_flows,
       coalesce(cv_gap, 0)           as cv_gap,
       (cv_gap is not null)::int     as has_cv,
       ln(1 + median_gap_s)          as log1p_median_gap,
       ln(1 + coalesce(sd_gap_s, 0)) as log1p_sd_gap,
       is_beacon::int                as is_beacon
from mart_motif_beacon
order by first_seen, sa, da, dp
"""


def svc_onehot(labels, classes):
    """One-hot service class; classes outside the list fold into 'other'."""
    idx = {c: i for i, c in enumerate(classes)}
    oh = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for n, c in enumerate(labels):
        oh[n, idx.get(c, idx["other"])] = 1.0
    return oh


def standardize(x, n_train):
    """z-score with TRAIN-only statistics (guideline 6)."""
    mu = x[:n_train].mean(axis=0)
    sd = x[:n_train].std(axis=0) + 1e-6
    return ((x - mu) / sd).astype(np.float32), mu, sd


def build_gap_windows(rows, k=PAIR_K):
    """rows: (sa, da, dp, time_start) sorted by channel, time. One window per
    flow with k full previous gaps on its channel: x = the previous k
    log1p-gaps, y = the next log1p-gap, last = the most recent gap (the
    repeat-last baseline). Returns (x[n,k], y[n], last[n]) time-ordered by
    the target flow's time."""
    by_ch = {}
    for sa, da, dp, t in rows:
        by_ch.setdefault((sa, da, dp), []).append(t)
    xs, ys, lasts, ts = [], [], [], []
    for times in by_ch.values():
        gaps = np.log1p(np.diff(np.asarray(times)))
        for i in range(k, len(gaps)):
            xs.append(gaps[i - k:i])
            ys.append(gaps[i])
            lasts.append(gaps[i - 1])
            ts.append(times[i + 1])
    order = np.argsort(np.asarray(ts), kind="stable")
    return (np.asarray(xs, dtype=np.float32)[order],
            np.asarray(ys, dtype=np.float32)[order],
            np.asarray(lasts, dtype=np.float32)[order])



def encode_qnames(qnames, max_chars=DNS_MAX_CHARS):
    """Char-encode names for the LM: inputs[n,L] (BOS+chars) and
    targets[n,L] (chars+EOS), PAD-filled; cased folded, UNK for
    out-of-charset bytes."""
    idx = {c: 4 + i for i, c in enumerate(DNS_CHARS)}
    L = max_chars + 1
    xs = np.full((len(qnames), L), DNS_PAD, dtype=np.int64)
    ys = np.full((len(qnames), L), DNS_PAD, dtype=np.int64)
    for n, q in enumerate(qnames):
        ids = [idx.get(c, DNS_UNK) for c in q.lower()[:max_chars]]
        xs[n, :len(ids) + 1] = [DNS_BOS] + ids
        ys[n, :len(ids) + 1] = ids + [DNS_EOS]
    return xs, ys

TLS_SQL = """
select flow_id, time_start, fingerprint, recs
from feat_tls
order by time_start, flow_id
"""


def build_tls_tensors(rows, k=TLS_K):
    """rows: (flow_id, time_start, fingerprint, recs) time-ordered; recs are
    joy srlt dicts. Returns (x[n,k,7], fps[n], flow_ids[n]): per record
    [dir(±1), onehot tp 20/21/22/23, log1p(b), log1p(ipt)], zero-padded — the
    packet structure only, no handshake payload."""
    tp_slot = {20: 1, 21: 2, 22: 3, 23: 4}
    x = np.zeros((len(rows), k, 7), dtype=np.float32)
    fps, fids = [], []
    for n, (fid, _, fp, recs) in enumerate(rows):
        for i, r in enumerate(recs[:k]):
            x[n, i, 0] = 1.0 if r["dir"] == ">" else -1.0
            slot = tp_slot.get(int(r["tp"]))
            if slot:
                x[n, i, slot] = 1.0
            x[n, i, 5] = np.log1p(float(r["b"]))
            x[n, i, 6] = np.log1p(float(r["ipt"]))
        fps.append(fp)
        fids.append(fid)
    return x, np.array(fps, dtype=object), np.array(fids)


def build_seq_windows(rows, classes, k=SEQ_K, max_gap=SEQ_MAX_GAP):
    """rows: (sa, service_class, time_start) sorted by sa, time.
    Returns (ids[n,k], gaps[n,k], targets[n], target_times[n], last_ids[n]).
    Windows are left-padded with the PAD id len(classes)."""
    cls_idx = {c: i for i, c in enumerate(classes)}
    pad = len(classes)
    ids, gaps, tgts, times, lasts = [], [], [], [], []
    by_sa = {}
    for sa, cls, t in rows:
        by_sa.setdefault(sa, []).append((cls_idx.get(cls, cls_idx["other"]), t))
    for seq in by_sa.values():
        for i in range(1, len(seq)):
            tgt_id, tgt_t = seq[i]
            if tgt_t - seq[i - 1][1] > max_gap:
                continue
            win = seq[max(0, i - k):i]
            w_ids = [pad] * (k - len(win)) + [c for c, _ in win]
            w_gaps = [0.0] * (k - len(win)) + [
                float(np.log1p(win[j][1] - win[j - 1][1])) if j > 0 else 0.0
                for j in range(len(win))
            ]
            ids.append(w_ids)
            gaps.append(w_gaps)
            tgts.append(tgt_id)
            times.append(tgt_t)
            lasts.append(win[-1][0])
    order = np.argsort(np.array(times), kind="stable")  # time-ordered split
    return (np.array(ids, dtype=np.int64)[order],
            np.array(gaps, dtype=np.float32)[order],
            np.array(tgts, dtype=np.int64)[order],
            np.array(lasts, dtype=np.int64)[order])


def pick_device() -> torch.device:
    want = os.environ.get("FLOW_DEVICE")
    if want:
        if want == "cpu":
            # deterministic mode for reproduce_check retrains
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlowAutoencoder(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, n_in),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ByteDistClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class NextServiceGRU(nn.Module):
    """P(next service | last K services and inter-flow gaps)."""

    def __init__(self, n_classes: int, emb: int = 8, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, emb)  # +1 = left-pad token
        self.gru = nn.GRU(emb + 1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        ids, gaps = x
        e = torch.cat([self.emb(ids), gaps.unsqueeze(-1)], dim=-1)
        _, h = self.gru(e)
        return self.out(h[-1])


class TlsRecordGRU(nn.Module):
    """Client-stack fingerprint from the TLS record sequence (encrypted-
    traffic fingerprinting: direction/type/size/timing, no payload bytes)."""

    def __init__(self, n_classes: int, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(7, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.out(h[-1])


class DnsCharLM(nn.Module):
    """Next-char LM over query names; per-name bits/char = shape anomaly."""

    def __init__(self, vocab: int = DNS_VOCAB, emb: int = 16, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=DNS_PAD)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h)


class ByteDistAE(nn.Module):
    """256->32->256 autoencoder on the normalized byte histogram; recon MSE
    is the payload-shape anomaly score (unsupervised, no adoption gate)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(256, 32)
        self.decoder = nn.Linear(32, 256)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


class MiniMLP(nn.Module):
    """Shared one-hidden-layer shape for the small gated heads."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class PairGapGRU(nn.Module):
    """Next log1p(gap_s) on a (sa,da,dp) channel from its previous K gaps."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))
        return self.out(h[-1])


def dns_bits_per_char(model, xs, ys, device, batch=512):
    """Mean NLL (bits) per non-pad target symbol, per name."""
    import torch.nn.functional as F
    out = np.zeros(len(xs))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = torch.from_numpy(xs[i:i + batch]).to(device)
            yb = torch.from_numpy(ys[i:i + batch]).to(device)
            logits = model(xb)
            nll = F.cross_entropy(logits.transpose(1, 2), yb,
                                  ignore_index=DNS_PAD, reduction="none")
            mask = (yb != DNS_PAD).float()
            per = (nll * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out[i:i + batch] = (per / math.log(2)).cpu().numpy()
    return out


def train_loop(model, loss_fn, xs, ys, device, epochs, lr, batch=256,
               logits_transform=None, sample_weights=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = xs[0].shape[0] if isinstance(xs, tuple) else xs.shape[0]

    def to_dev(data, idx):
        if isinstance(data, tuple):
            return tuple(t[idx].to(device) for t in data)
        return data[idx].to(device)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = to_dev(xs, idx)
            yb = ys[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            if logits_transform is not None:
                logits = logits_transform(logits)
            if sample_weights is None:
                loss = loss_fn(logits, yb)
            else:
                loss = loss_fn(logits, yb, sample_weights[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  loss {total / n:.5f}")
    model.eval()


def recon_errors(model, xt, device, batch=8192):
    """Per-row reconstruction MSE, batched (val sets can be large)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            xb = xt[i:i + batch].to(device)
            out.append(((model(xb) - xb) ** 2).mean(dim=1).cpu())
    return torch.cat(out)


def fit_majority_classifier(name, model, x, y, device, epochs, lr, batch=512):
    """Shared gated-classifier recipe: time-ordered 80/20 split, majority
    baseline from TRAIN, adopted = val accuracy > majority baseline."""
    n_train = int(len(x) * 0.8)
    print(f"training {name} on {n_train} rows ...")
    train_loop(model, nn.CrossEntropyLoss(),
               torch.from_numpy(x[:n_train]), torch.from_numpy(y[:n_train]),
               device, epochs, lr, batch=batch)
    preds = []
    with torch.no_grad():
        for i in range(n_train, len(x), 8192):
            preds.append(model(torch.from_numpy(x[i:i + 8192]).to(device))
                         .argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = float((pred == y[n_train:]).mean())
    majority = float(np.bincount(y[:n_train]).max() / max(n_train, 1))
    print(f"  val accuracy {acc:.3f}  (majority baseline {majority:.3f})  "
          f"-> {'ADOPTED' if acc > majority else 'not adopted'}")
    return {
        "val_accuracy": acc,
        "majority_baseline": majority,
        "adopted": acc > majority,
        "n_train": n_train,
        "n_val": int(len(x) - n_train),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "artifacts" / "enclave.duckdb"))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    if not Path(args.db).exists():
        print(f"ERROR: {args.db} not found — run `make dbt` first", file=sys.stderr)
        return 1
    con = duckdb.connect(args.db, read_only=True)

    # ---- content-addressed artifact: skip training if inputs unchanged ------
    flow_path = ROOT / "flow.jsonl"
    fingerprint = hashlib.sha256(flow_path.read_bytes()).hexdigest()
    hparams = {
        "epochs": args.epochs,
        "seed": SEED,
        "device": str(device),
        "split": "time-ordered 80/20",
        "ae_arch": "14-16-8-3-8-16-14",
        "clf_arch": "256-64-C",
        "gru_arch": f"emb8-GRU32, K={SEQ_K}",
        "seq_max_gap_s": SEQ_MAX_GAP,
        "tls_arch": f"rec7-GRU16, K={TLS_K}",
        "tls_min_class": TLS_MIN_CLASS,
        "dns_arch": f"emb16-GRU32 char LM, vocab={DNS_VOCAB}, max={DNS_MAX_CHARS}",
        "bdae_arch": "256-32-256",
        "volsvc_arch": f"{len(NUMERIC_FEATURES)}-32-C",
        "bytesin_arch": "svc+2-32-1 (L1)",
        "durband_arch": "svc+2-32-5",
        "durband_bands_s": DUR_BANDS_S,
        "pairgap_arch": f"GRU16, K={PAIR_K}, min_flows={PAIR_MIN_FLOWS}",
        "sesslen_arch": f"{len(NUMERIC_FEATURES)}-32-{len(SESS_BUCKETS)}",
        "sesslen_buckets": SESS_BUCKETS,
        "unans_arch": "2+svc-16-2",
        "tlssetup_arch": "fp+2-32-4",
        "tlssetup_bands_ms": TLS_SETUP_BANDS_MS,
        "dnskind_arch": "5-16-3",
        "beacon_arch": "5-16-2",
    }
    version = candidate_version("minimodels", fingerprint, hparams, [Path(__file__)])
    if artifact_dir("minimodels", version).exists():
        activated = maybe_activate("minimodels", version)
        print(f"artifact minimodels/{version} already exists — "
              f"{'activated' if activated else 'kept current active'} "
              f"(set FLOW_RETRAIN=1 to activate on retrain)")
        return 0

    # ---- Model A: autoencoder on numeric features --------------------------
    df = con.execute(FEATURE_SQL).fetch_df()
    x = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    n_train = int(len(x) * 0.8)  # time-ordered split
    mu, sd = x[:n_train].mean(axis=0), x[:n_train].std(axis=0) + 1e-6
    xz = (x - mu) / sd
    xt_train = torch.from_numpy(xz[:n_train])
    xt_val = torch.from_numpy(xz[n_train:])

    ae = FlowAutoencoder(len(NUMERIC_FEATURES)).to(device)
    print(f"training FlowAutoencoder on {n_train} flows ...")
    train_loop(ae, nn.MSELoss(), xt_train, xt_train, device, args.epochs, 1e-3)
    with torch.no_grad():
        val_err = ((ae(xt_val.to(device)) - xt_val.to(device)) ** 2).mean(dim=1)
        train_err = ((ae(xt_train.to(device)) - xt_train.to(device)) ** 2).mean(dim=1)
    ae_metrics = {
        "train_mse_mean": float(train_err.mean()),
        "val_mse_mean": float(val_err.mean()),
        "val_mse_p95": float(val_err.quantile(0.95)),
        "n_train": n_train,
        "n_val": len(x) - n_train,
    }
    print(f"  val MSE mean {ae_metrics['val_mse_mean']:.4f}  p95 {ae_metrics['val_mse_p95']:.4f}")

    # ---- Model B: service classifier on byte_dist ---------------------------
    bd = con.execute(BYTEDIST_SQL).fetch_df()
    mat = np.stack(bd["byte_dist"].to_numpy()).astype(np.float32)
    row_sums = mat.sum(axis=1)
    keep = row_sums > 0
    mat, labels = mat[keep] / row_sums[keep, None], bd["service_class"].to_numpy()[keep]
    mat_payload = mat  # every payload-bearing flow — ByteDistAE input below
    class_counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    classes = sorted(c for c, n in class_counts.items() if n >= 30)
    sel = np.isin(labels, classes)
    mat, labels = mat[sel], labels[sel]
    y = np.array([classes.index(c) for c in labels], dtype=np.int64)
    n_train_b = int(len(mat) * 0.8)  # still time-ordered
    xb_train = torch.from_numpy(mat[:n_train_b])
    yb_train = torch.from_numpy(y[:n_train_b])
    xb_val = torch.from_numpy(mat[n_train_b:])
    yb_val = torch.from_numpy(y[n_train_b:])

    clf = ByteDistClassifier(len(classes)).to(device)
    print(f"training ByteDistClassifier on {n_train_b} flows, {len(classes)} classes: {classes}")
    train_loop(clf, nn.CrossEntropyLoss(), xb_train, yb_train, device, args.epochs, 1e-3)
    with torch.no_grad():
        pred = clf(xb_val.to(device)).argmax(dim=1).cpu()
        acc = float((pred == yb_val).float().mean())
        majority = float(np.bincount(y[:n_train_b]).max() / max(n_train_b, 1))
    clf_metrics = {
        "val_accuracy": acc,
        "majority_baseline": majority,
        "adopted": acc > majority,
        "n_train": n_train_b,
        "n_val": len(mat) - n_train_b,
        "classes": classes,
    }
    print(f"  val accuracy {acc:.3f}  (majority baseline {majority:.3f})")

    # ---- Model C: next-service sequence model -------------------------------
    seq_rows = con.execute(SEQ_SQL).fetchall()
    svc_counts = con.execute(
        "select service_class, count(*) from feat_flows group by 1"
    ).fetchall()
    seq_classes = sorted(c for c, cnt in svc_counts if cnt >= 50 and c != "other")
    seq_classes.append("other")
    ids_a, gaps_a, tgt_a, last_a = build_seq_windows(seq_rows, seq_classes)
    n_train_c = int(len(tgt_a) * 0.8)  # windows already time-ordered
    xc_train = (torch.from_numpy(ids_a[:n_train_c]), torch.from_numpy(gaps_a[:n_train_c]))
    yc_train = torch.from_numpy(tgt_a[:n_train_c])
    xc_val = (torch.from_numpy(ids_a[n_train_c:]).to(device),
              torch.from_numpy(gaps_a[n_train_c:]).to(device))
    yc_val = torch.from_numpy(tgt_a[n_train_c:])

    gru = NextServiceGRU(len(seq_classes)).to(device)
    print(f"training NextServiceGRU on {n_train_c} windows, "
          f"{len(seq_classes)} classes: {seq_classes}")
    train_loop(gru, nn.CrossEntropyLoss(), xc_train, yc_train, device,
               max(args.epochs // 2, 20), 2e-3)
    with torch.no_grad():
        pred_c = gru(xc_val).argmax(dim=1).cpu().numpy()
    acc_c = float((pred_c == tgt_a[n_train_c:]).mean())
    mode_id = int(np.bincount(tgt_a[:n_train_c]).argmax())
    marginal_c = float((tgt_a[n_train_c:] == mode_id).mean())
    repeat_last_c = float((last_a[n_train_c:] == tgt_a[n_train_c:]).mean())
    gru_metrics = {
        "val_accuracy": acc_c,
        "marginal_baseline": marginal_c,
        "repeat_last_baseline": repeat_last_c,
        "adopted": acc_c > repeat_last_c,
        "n_train": n_train_c,
        "n_val": int(len(tgt_a) - n_train_c),
        "classes": seq_classes,
    }
    print(f"  val accuracy {acc_c:.3f}  (marginal {marginal_c:.3f}, "
          f"repeat-last {repeat_last_c:.3f})")

    # ---- Model D: TLS client-stack fingerprint from record structure --------
    tls_rows = con.execute(TLS_SQL).fetchall()
    xd, fps_d, _ = build_tls_tensors(tls_rows)
    fp_counts = {f: int((fps_d == f).sum()) for f in set(fps_d)}
    tls_classes = sorted(f for f, n in fp_counts.items() if n >= TLS_MIN_CLASS)
    tls_classes.append("other")
    yd = np.array([tls_classes.index(f) if f in tls_classes else len(tls_classes) - 1
                   for f in fps_d], dtype=np.int64)
    n_train_d = int(len(xd) * 0.8)  # rows already time-ordered
    xd_train, yd_train = torch.from_numpy(xd[:n_train_d]), torch.from_numpy(yd[:n_train_d])
    xd_val, yd_val = torch.from_numpy(xd[n_train_d:]), torch.from_numpy(yd[n_train_d:])

    tlsm = TlsRecordGRU(len(tls_classes)).to(device)
    print(f"training TlsRecordGRU on {n_train_d} TLS flows, "
          f"{len(tls_classes)} stacks: {tls_classes}")
    train_loop(tlsm, nn.CrossEntropyLoss(), xd_train, yd_train, device,
               args.epochs, 2e-3, batch=32)
    with torch.no_grad():
        pred_d = tlsm(xd_val.to(device)).argmax(dim=1).cpu().numpy()
    acc_d = float((pred_d == yd[n_train_d:]).mean())
    majority_d = float(np.bincount(yd[:n_train_d]).max() / max(n_train_d, 1))
    tls_metrics = {
        "val_accuracy": acc_d,
        "majority_baseline": majority_d,
        "adopted": acc_d > majority_d,
        "n_train": n_train_d,
        "n_val": int(len(xd) - n_train_d),
        "classes": tls_classes,
    }
    print(f"  val accuracy {acc_d:.3f}  (majority baseline {majority_d:.3f})")

    # ---- Model E: DNS char LM (name-shape anomaly) ---------------------------
    dns_rows = con.execute(DNS_SQL).fetchall()
    qnames_all = [r[0] for r in dns_rows]  # already time-ordered
    xe, ye = encode_qnames(qnames_all)
    n_train_e = int(len(xe) * 0.8)

    # Train on DISTINCT names weighted by query count: the same record-level
    # objective, but orders of magnitude fewer sequences on large captures
    # where the same names repeat (the evaluation below stays record-level).
    train_counts = Counter(qnames_all[:n_train_e])
    train_names = sorted(train_counts)
    xu, yu = encode_qnames(train_names)
    wu = torch.tensor([train_counts[q] for q in train_names], dtype=torch.float32)
    wu = wu / wu.mean()

    def weighted_lm_loss(logits, yb, wb):
        # logits [B, vocab, L]: per-name masked mean NLL, query-count-weighted
        nll = F.cross_entropy(logits, yb, ignore_index=DNS_PAD, reduction="none")
        mask = (yb != DNS_PAD).float()
        per = (nll * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return (per * wb).sum() / wb.sum()

    dnslm = DnsCharLM().to(device)
    print(f"training DnsCharLM on {len(train_names)} distinct names "
          f"({n_train_e} query records, count-weighted) ...")
    train_loop(dnslm, weighted_lm_loss,
               torch.from_numpy(xu), torch.from_numpy(yu),
               device, args.epochs, 2e-3, batch=256,
               logits_transform=lambda z: z.transpose(1, 2),
               sample_weights=wu)
    val_bits = dns_bits_per_char(dnslm, xe[n_train_e:], ye[n_train_e:], device)
    train_bits = dns_bits_per_char(dnslm, xe[:n_train_e], ye[:n_train_e], device)

    # order-0 baseline: unigram distribution of target symbols on train
    _counts = np.bincount(ye[:n_train_e][ye[:n_train_e] != DNS_PAD].ravel(),
                          minlength=DNS_VOCAB).astype(np.float64)
    _probs = (_counts + 1) / (_counts.sum() + DNS_VOCAB)
    _val_tgt = ye[n_train_e:]
    _mask = _val_tgt != DNS_PAD
    unigram_bits = float((-np.log2(_probs[_val_tgt]) * _mask).sum() / _mask.sum())
    val_bits_mean = float(
        (val_bits * _mask.sum(axis=1)).sum() / _mask.sum())  # symbol-weighted
    dns_metrics = {
        "val_bits_per_char": val_bits_mean,
        "train_bits_per_char": float(np.mean(train_bits)),
        "unigram_baseline_bits": unigram_bits,
        "adopted": val_bits_mean < unigram_bits,
        "n_train": n_train_e,
        "n_train_names": len(train_names),
        "n_val": int(len(xe) - n_train_e),
        "charset": DNS_CHARS,
        "max_chars": DNS_MAX_CHARS,
    }
    print(f"  val {val_bits_mean:.3f} bits/char  (unigram baseline {unigram_bits:.3f})")

    # ---- ten additional minimodels (2026-10-30) ------------------------------
    quick = max(args.epochs // 3, 15)   # big simple heads converge fast
    half = max(args.epochs // 2, 20)

    # ByteDistAE: unsupervised payload-shape anomaly (no adoption gate)
    n_train_bd = int(len(mat_payload) * 0.8)
    bd_train = torch.from_numpy(mat_payload[:n_train_bd])
    bd_val = torch.from_numpy(mat_payload[n_train_bd:])
    bdae = ByteDistAE().to(device)
    print(f"training ByteDistAE on {n_train_bd} payload-bearing flows ...")
    train_loop(bdae, nn.MSELoss(), bd_train, bd_train, device, half, 1e-3,
               batch=512)
    bd_val_err = recon_errors(bdae, bd_val, device)
    bd_train_err = recon_errors(bdae, bd_train, device)
    bdae_metrics = {
        "train_mse_mean": float(bd_train_err.mean()),
        "val_mse_mean": float(bd_val_err.mean()),
        "val_mse_p95": float(bd_val_err.quantile(0.95)),
        "n_train": n_train_bd,
        "n_val": int(len(mat_payload) - n_train_bd),
    }
    print(f"  val MSE mean {bdae_metrics['val_mse_mean']:.6f}  "
          f"p95 {bdae_metrics['val_mse_p95']:.6f}")

    # VolumeServiceMLP: service class from volume/timing numerics (no ports)
    labels_all = df["service_class"].to_numpy()
    vol_counts = Counter(labels_all)
    vol_classes = sorted(c for c, n in vol_counts.items() if n >= 30)
    sel_v = np.isin(labels_all, vol_classes)
    y_vol = np.array([vol_classes.index(c) for c in labels_all[sel_v]],
                     dtype=np.int64)
    volm = MiniMLP(len(NUMERIC_FEATURES), len(vol_classes)).to(device)
    vol_metrics = fit_majority_classifier(
        f"VolumeServiceMLP ({len(vol_classes)} classes)", volm,
        xz[sel_v], y_vol, device, quick, 1e-3)
    vol_metrics["classes"] = vol_classes

    # BytesInMLP: log1p(bytes_in) ~ service onehot + outbound volume + duration
    svc_oh = svc_onehot(labels_all, seq_classes)
    x_bi = np.concatenate(
        [svc_oh, df[["log1p_bytes_out", "log1p_duration_ms"]]
                 .to_numpy(dtype=np.float32)], axis=1)
    y_bi = df["log1p_bytes_in"].to_numpy(dtype=np.float32)
    n_train_bi = int(len(x_bi) * 0.8)
    bim = MiniMLP(x_bi.shape[1], 1).to(device)
    print(f"training BytesInMLP on {n_train_bi} flows ...")
    train_loop(bim, nn.L1Loss(),
               torch.from_numpy(x_bi[:n_train_bi]),
               torch.from_numpy(y_bi[:n_train_bi]),
               device, quick, 1e-3, batch=512,
               logits_transform=lambda z: z.squeeze(-1))
    with torch.no_grad():
        pred_bi = np.concatenate([
            bim(torch.from_numpy(x_bi[i:i + 8192]).to(device))
            .squeeze(-1).cpu().numpy()
            for i in range(n_train_bi, len(x_bi), 8192)])
    mae_bi = float(np.abs(pred_bi - y_bi[n_train_bi:]).mean())
    folded = np.where(np.isin(labels_all, seq_classes), labels_all, "other")
    med_global = float(np.median(y_bi[:n_train_bi]))
    bi_medians = {c: float(np.median(y_bi[:n_train_bi][folded[:n_train_bi] == c]))
                  for c in seq_classes if (folded[:n_train_bi] == c).any()}
    base_bi = np.array([bi_medians.get(c, med_global)
                        for c in folded[n_train_bi:]], dtype=np.float32)
    base_mae_bi = float(np.abs(base_bi - y_bi[n_train_bi:]).mean())
    bytes_in_metrics = {
        "val_mae_log": mae_bi,
        "baseline_mae_log": base_mae_bi,
        "adopted": mae_bi < base_mae_bi,
        "n_train": n_train_bi,
        "n_val": int(len(x_bi) - n_train_bi),
    }
    print(f"  val MAE {mae_bi:.3f} log-units  (per-service train median "
          f"{base_mae_bi:.3f})  -> "
          f"{'ADOPTED' if mae_bi < base_mae_bi else 'not adopted'}")

    # DurationBandMLP: 5 public duration bands from service onehot + log bytes
    y_dur = np.digitize(df["duration_s"].to_numpy(), DUR_BANDS_S,
                        right=True).astype(np.int64)
    x_dur = np.concatenate(
        [svc_oh, df[["log1p_bytes_out", "log1p_bytes_in"]]
                 .to_numpy(dtype=np.float32)], axis=1)
    durm = MiniMLP(x_dur.shape[1], len(DUR_BANDS_S) + 1).to(device)
    dur_metrics = fit_majority_classifier("DurationBandMLP (5 bands)", durm,
                                          x_dur, y_dur, device, quick, 1e-3)
    dur_metrics["bands_s"] = DUR_BANDS_S

    # PairGapGRU: next channel gap from the previous K=5 log-gaps
    gap_rows = con.execute(PAIRGAP_SQL).fetchall()
    xg, yg, lastg = build_gap_windows(gap_rows)
    n_train_g = int(len(xg) * 0.8)
    gapm = PairGapGRU().to(device)
    print(f"training PairGapGRU on {n_train_g} windows "
          f"({len(gap_rows)} flows on >={PAIR_MIN_FLOWS}-flow channels) ...")
    train_loop(gapm, nn.L1Loss(),
               torch.from_numpy(xg[:n_train_g]), torch.from_numpy(yg[:n_train_g]),
               device, half, 2e-3, batch=512,
               logits_transform=lambda z: z.squeeze(-1))
    with torch.no_grad():
        pred_g = np.concatenate([
            gapm(torch.from_numpy(xg[i:i + 8192]).to(device))
            .squeeze(-1).cpu().numpy()
            for i in range(n_train_g, len(xg), 8192)])
    mae_g = float(np.abs(pred_g - yg[n_train_g:]).mean())
    base_g = float(np.abs(lastg[n_train_g:] - yg[n_train_g:]).mean())
    gap_metrics = {
        "val_mae_log_gap": mae_g,
        "repeat_last_mae": base_g,
        "adopted": mae_g < base_g,
        "n_train": n_train_g,
        "n_val": int(len(xg) - n_train_g),
    }
    print(f"  val MAE {mae_g:.3f} log-gap units  (repeat-last {base_g:.3f})  "
          f"-> {'ADOPTED' if mae_g < base_g else 'not adopted'}")

    # SessionLenMLP: session flow-count bucket from its FIRST flow's features
    sess = con.execute(SESSION_SQL).fetch_df()
    x_sess_raw = sess[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    n_train_s = int(len(x_sess_raw) * 0.8)
    x_sess, _, _ = standardize(x_sess_raw, n_train_s)
    y_sess = np.digitize(sess["session_flows"].to_numpy(),
                         SESS_BUCKETS[1:]).astype(np.int64)
    sessm = MiniMLP(len(NUMERIC_FEATURES), len(SESS_BUCKETS)).to(device)
    sess_metrics = fit_majority_classifier(
        "SessionLenMLP (buckets 2/3/4-5/6-8/9+)", sessm, x_sess, y_sess,
        device, args.epochs, 1e-3, batch=256)
    sess_metrics["buckets"] = SESS_BUCKETS

    # UnansweredMLP: is_unanswered from OUTBOUND-only features
    x_un = np.concatenate(
        [df[["log1p_bytes_out", "log1p_pkts_out"]].to_numpy(dtype=np.float32),
         svc_oh], axis=1)
    y_un = df["is_unanswered"].to_numpy(dtype=np.int64)
    unm = MiniMLP(x_un.shape[1], 2, hidden=16).to(device)
    un_metrics = fit_majority_classifier("UnansweredMLP", unm, x_un, y_un,
                                         device, quick, 1e-3)

    # TlsSetupBandMLP: handshake-setup band from fingerprint + record volume
    ts_df = con.execute(TLS_SETUP_SQL).fetch_df()
    fp_all = ts_df["fingerprint"].to_numpy()
    setup_stacks = sorted(f for f, n in Counter(fp_all).items()
                          if n >= TLS_MIN_CLASS)
    fp_idx = {f: i for i, f in enumerate(setup_stacks)}
    oh_fp = np.zeros((len(ts_df), len(setup_stacks) + 1), dtype=np.float32)
    for n, f in enumerate(fp_all):
        oh_fp[n, fp_idx.get(f, len(setup_stacks))] = 1.0  # last slot = other
    x_ts = np.concatenate(
        [oh_fp, np.log1p(ts_df[["n_records", "tls_bytes"]]
                         .to_numpy(dtype=np.float32))], axis=1)
    y_ts = np.digitize(ts_df["hs_setup_ms"].to_numpy(), TLS_SETUP_BANDS_MS,
                       right=True).astype(np.int64)
    tsm = MiniMLP(x_ts.shape[1], len(TLS_SETUP_BANDS_MS) + 1).to(device)
    ts_metrics = fit_majority_classifier(
        f"TlsSetupBandMLP ({len(setup_stacks)} stacks + other, 4 bands)", tsm,
        x_ts, y_ts, device, half, 1e-3, batch=256)
    ts_metrics["classes"] = setup_stacks
    ts_metrics["bands_ms"] = TLS_SETUP_BANDS_MS

    # DnsKindClassifier: query kind from numeric name-shape features only
    dk = con.execute(DNS_KIND_SQL).fetch_df()
    x_dk_raw = dk[["qname_len", "n_labels", "max_label_len", "min_label_len",
                   "digit_frac"]].to_numpy(dtype=np.float32)
    n_train_k = int(len(x_dk_raw) * 0.8)
    x_dk, _, _ = standardize(x_dk_raw, n_train_k)
    y_dk = dk["kind"].to_numpy(dtype=np.int64)
    dkm = MiniMLP(x_dk.shape[1], len(DNS_KIND_CLASSES), hidden=16).to(device)
    dk_metrics = fit_majority_classifier("DnsKindClassifier", dkm, x_dk, y_dk,
                                         device, quick, 1e-3)
    dk_metrics["classes"] = DNS_KIND_CLASSES

    # BeaconChannelMLP: is_beacon from channel gap statistics (tiny N — the
    # result, adopted or not, is recorded honestly)
    bc = con.execute(BEACON_SQL).fetch_df()
    x_bc_raw = bc[["log1p_n_flows", "cv_gap", "has_cv", "log1p_median_gap",
                   "log1p_sd_gap"]].to_numpy(dtype=np.float32)
    n_train_bc = int(len(x_bc_raw) * 0.8)
    x_bc, _, _ = standardize(x_bc_raw, n_train_bc)
    y_bc = bc["is_beacon"].to_numpy(dtype=np.int64)
    bcm = MiniMLP(x_bc.shape[1], 2, hidden=16).to(device)
    bc_metrics = fit_majority_classifier("BeaconChannelMLP", bcm, x_bc, y_bc,
                                         device, args.epochs, 1e-3, batch=256)

    # ---- immutable, content-addressed artifact ------------------------------
    def blob(sd) -> bytes:
        buf = io.BytesIO()
        torch.save(sd, buf)
        return buf.getvalue()

    meta = {
        "version": version,
        "data_fingerprint_sha256": fingerprint,
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "split": "time-ordered 80/20",
        "numeric_features": NUMERIC_FEATURES,
        "eval_definitions": EVAL_DEFINITIONS,
        "scaler_mean": mu.tolist(),
        "scaler_std": sd.tolist(),
        "autoencoder": {"arch": hparams["ae_arch"], "metrics": ae_metrics},
        "classifier": {"arch": hparams["clf_arch"], "metrics": clf_metrics},
        "next_service": {"arch": hparams["gru_arch"], "k": SEQ_K,
                         "max_gap_s": SEQ_MAX_GAP, "metrics": gru_metrics},
        "tls_fingerprint": {"arch": hparams["tls_arch"], "k": TLS_K,
                            "min_class": TLS_MIN_CLASS, "metrics": tls_metrics},
        "dns_name_lm": {"arch": hparams["dns_arch"], "metrics": dns_metrics},
        "bytedist_ae": {"arch": hparams["bdae_arch"], "metrics": bdae_metrics},
        "volume_service": {"arch": hparams["volsvc_arch"], "metrics": vol_metrics},
        "bytes_in": {"arch": hparams["bytesin_arch"], "svc_classes": seq_classes,
                     "train_medians": bi_medians, "metrics": bytes_in_metrics},
        "duration_band": {"arch": hparams["durband_arch"], "svc_classes": seq_classes,
                          "bands_s": DUR_BANDS_S, "metrics": dur_metrics},
        "pair_gap": {"arch": hparams["pairgap_arch"], "k": PAIR_K,
                     "min_flows": PAIR_MIN_FLOWS, "metrics": gap_metrics},
        "session_len": {"arch": hparams["sesslen_arch"], "buckets": SESS_BUCKETS,
                        "metrics": sess_metrics},
        "unanswered": {"arch": hparams["unans_arch"], "svc_classes": seq_classes,
                       "metrics": un_metrics},
        "tls_setup": {"arch": hparams["tlssetup_arch"],
                      "bands_ms": TLS_SETUP_BANDS_MS, "metrics": ts_metrics},
        "dns_kind": {"arch": hparams["dnskind_arch"], "metrics": dk_metrics},
        "beacon_channel": {"arch": hparams["beacon_arch"], "metrics": bc_metrics},
        "epochs": args.epochs,
    }
    vdir = write_artifact("minimodels", version, {
        "autoencoder.pt": blob(ae.state_dict()),
        "classifier.pt": blob(clf.state_dict()),
        "nextservice.pt": blob(gru.state_dict()),
        "tlsrecord.pt": blob(tlsm.state_dict()),
        "dnslm.pt": blob(dnslm.state_dict()),
        "bytedist_ae.pt": blob(bdae.state_dict()),
        "volume_service.pt": blob(volm.state_dict()),
        "bytes_in.pt": blob(bim.state_dict()),
        "duration_band.pt": blob(durm.state_dict()),
        "pair_gap.pt": blob(gapm.state_dict()),
        "session_len.pt": blob(sessm.state_dict()),
        "unanswered.pt": blob(unm.state_dict()),
        "tls_setup.pt": blob(tsm.state_dict()),
        "dns_kind.pt": blob(dkm.state_dict()),
        "beacon_channel.pt": blob(bcm.state_dict()),
    }, {**meta, "data_fingerprint": fingerprint, "hparams": hparams})
    activated = maybe_activate("minimodels", version)
    print(f"artifact: {vdir} ({'active' if activated else 'written, not activated'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

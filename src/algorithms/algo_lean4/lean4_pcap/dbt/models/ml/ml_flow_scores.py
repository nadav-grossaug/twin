# Inference node (goal.md guideline 3): loads the ACTIVE minimodel artifact
# from the content-addressed registry (models_out/registry.json) and
# materializes per-flow predictions. Rebuilding this mart never retrains; if
# no artifact is active yet it materializes an 'untrained' placeholder so
# downstream SQL still compiles.
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _active_artifact(models_dir: Path):
    reg = models_dir / "registry.json"
    if not reg.exists():
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        return None
    vdir = models_dir / "minimodels" / active
    return (active, vdir) if (vdir / "meta.json").exists() else None


def _scores(df, bd_df, tls_df, dns_df, version, vdir):
    import torch

    meta = json.loads((vdir / "meta.json").read_text())

    feats = pd.DataFrame({
        "log1p_bytes_out": np.log1p(df.bytes_out),
        "log1p_bytes_in": np.log1p(df.bytes_in),
        "log1p_pkts_out": np.log1p(df.num_pkts_out),
        "log1p_pkts_in": np.log1p(df.num_pkts_in),
        "log1p_duration_ms": np.log1p(df.duration_s * 1000),
        "log1p_bytes_per_s": np.log1p(df.bytes_per_s.fillna(0)),
        "entropy_filled": df.entropy.fillna(0),
        "has_entropy": df.entropy.notna().astype(int),
        "is_zero_byte": df.is_zero_byte.astype(int),
        "is_unanswered": df.is_unanswered.astype(int),
        "src_internal": df.src_internal.astype(int),
        "dst_internal": df.dst_internal.astype(int),
        "hour_sin": np.sin(2 * math.pi * df.hour_utc / 24),
        "hour_cos": np.cos(2 * math.pi * df.hour_utc / 24),
    })[meta["numeric_features"]]
    mu = np.array(meta["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["scaler_std"], dtype=np.float32)
    xz = torch.from_numpy(((feats.to_numpy(dtype=np.float32) - mu) / sd))

    # rebuild architectures from the training script's definitions
    n_in = len(meta["numeric_features"])
    ae = torch.nn.Sequential()  # container for loading; use functional modules
    import torch.nn as nn

    class FlowAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(),
                                         nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 8), nn.ReLU(),
                                         nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    classes = meta["classifier"]["metrics"]["classes"]

    class ByteDistClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(64, len(classes)))

        def forward(self, x):
            return self.net(x)

    ns_meta = meta.get("next_service")
    ns_classes = ns_meta["metrics"]["classes"] if ns_meta else []

    tls_meta = meta.get("tls_fingerprint")
    tls_classes = tls_meta["metrics"]["classes"] if tls_meta else []

    dns_meta = meta.get("dns_name_lm")
    _dns_charset = dns_meta["metrics"]["charset"] if dns_meta else ""
    _dns_vocab = 4 + len(_dns_charset)

    class DnsCharLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(_dns_vocab, 16, padding_idx=0)
            self.gru = nn.GRU(16, 32, batch_first=True)
            self.out = nn.Linear(32, _dns_vocab)

        def forward(self, x):
            h, _ = self.gru(self.emb(x))
            return self.out(h)

    class TlsRecordGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(7, 16, batch_first=True)
            self.out = nn.Linear(16, len(tls_classes))

        def forward(self, x):
            _, h = self.gru(x)
            return self.out(h[-1])

    class NextServiceGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(ns_classes) + 1, 8)
            self.gru = nn.GRU(9, 32, batch_first=True)
            self.out = nn.Linear(32, len(ns_classes))

        def forward(self, x):
            ids, gaps = x
            e = torch.cat([self.emb(ids), gaps.unsqueeze(-1)], dim=-1)
            _, h = self.gru(e)
            return self.out(h[-1])

    # 2026-10-30 additions. Only the FLOW-level models score here:
    # ByteDistAE (bd_recon_score), VolumeServiceMLP (pred_service_vol),
    # BytesInMLP (bytes_in_abs_err), UnansweredMLP (pred_unanswered_conf).
    # PairGapGRU (channel), SessionLenMLP (session), TlsSetupBandMLP
    # (TLS handshake), DnsKindClassifier (DNS record) and BeaconChannelMLP
    # (channel) are not per-flow scorers — their frozen evaluations live in
    # the artifact's meta.json only.
    class MiniMLP(nn.Module):
        def __init__(self, n_in, n_out, hidden=32):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_out))

        def forward(self, x):
            return self.net(x)

    class ByteDistAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(256, 32)
            self.decoder = nn.Linear(32, 256)

        def forward(self, x):
            return self.decoder(torch.relu(self.encoder(x)))

    def _svc_onehot(classes):
        sidx = {c: i for i, c in enumerate(classes)}
        oh = np.zeros((len(df), len(classes)), dtype=np.float32)
        for n, c in enumerate(df.service_class):
            oh[n, sidx.get(c, sidx["other"])] = 1.0
        return oh

    ae = FlowAutoencoder()
    ae.load_state_dict(torch.load(vdir / "autoencoder.pt", map_location="cpu"))
    ae.eval()
    clf = ByteDistClassifier()
    clf.load_state_dict(torch.load(vdir / "classifier.pt", map_location="cpu"))
    clf.eval()

    with torch.no_grad():
        recon = ae(xz)
        anomaly = ((recon - xz) ** 2).mean(dim=1).numpy()
        emb = ae.encoder(xz).numpy()

        pos = {fid: i for i, fid in enumerate(df.flow_id.to_numpy())}
        pred_service = np.full(len(df), None, dtype=object)
        pred_conf = np.full(len(df), np.nan)
        bd_recon = np.full(len(df), np.nan)
        bd_idx = bd_df.index[bd_df.byte_dist.notna()]
        if len(bd_idx):
            mat = np.stack(bd_df.loc[bd_idx, "byte_dist"].to_numpy()).astype(np.float32)
            sums = mat.sum(axis=1, keepdims=True)
            ok = sums[:, 0] > 0
            xnorm = torch.from_numpy(np.divide(mat, np.where(sums == 0, 1, sums)))
            probs = torch.softmax(clf(xnorm), dim=1).numpy()
            flow_ids = bd_df.loc[bd_idx, "flow_id"].to_numpy()
            for row, fid in enumerate(flow_ids):
                if ok[row] and fid in pos:
                    k = probs[row].argmax()
                    pred_service[pos[fid]] = classes[k]
                    pred_conf[pos[fid]] = float(probs[row][k])

            # ByteDistAE payload-shape anomaly (payload-bearing flows only)
            if meta.get("bytedist_ae") and (vdir / "bytedist_ae.pt").exists():
                bdae = ByteDistAE()
                bdae.load_state_dict(torch.load(vdir / "bytedist_ae.pt",
                                                map_location="cpu"))
                bdae.eval()
                err = ((bdae(xnorm) - xnorm) ** 2).mean(dim=1).numpy()
                for row, fid in enumerate(flow_ids):
                    if ok[row] and fid in pos:
                        bd_recon[pos[fid]] = float(err[row])

        # VolumeServiceMLP: service from volume/timing numerics (all flows)
        pred_vol = np.full(len(df), None, dtype=object)
        pred_vol_conf = np.full(len(df), np.nan)
        vol_meta = meta.get("volume_service")
        if vol_meta and (vdir / "volume_service.pt").exists():
            vol_classes = vol_meta["metrics"]["classes"]
            volm = MiniMLP(n_in, len(vol_classes))
            volm.load_state_dict(torch.load(vdir / "volume_service.pt",
                                            map_location="cpu"))
            volm.eval()
            vprobs = torch.softmax(volm(xz.float()), dim=1).numpy()
            vk = vprobs.argmax(axis=1)
            pred_vol[:] = [vol_classes[k] for k in vk]
            pred_vol_conf[:] = vprobs[np.arange(len(vk)), vk]

        # BytesInMLP: |log1p(bytes_in) - prediction| per flow
        bytes_in_err = np.full(len(df), np.nan)
        bi_meta = meta.get("bytes_in")
        if bi_meta and (vdir / "bytes_in.pt").exists():
            x_bi = np.concatenate(
                [_svc_onehot(bi_meta["svc_classes"]),
                 feats[["log1p_bytes_out", "log1p_duration_ms"]]
                 .to_numpy(dtype=np.float32)], axis=1)
            bim = MiniMLP(x_bi.shape[1], 1)
            bim.load_state_dict(torch.load(vdir / "bytes_in.pt",
                                           map_location="cpu"))
            bim.eval()
            pred_log = bim(torch.from_numpy(x_bi)).squeeze(-1).numpy()
            bytes_in_err[:] = np.abs(
                pred_log - feats["log1p_bytes_in"].to_numpy(dtype=np.float32))

        # UnansweredMLP: P(unanswered) from outbound-only features
        unans_conf = np.full(len(df), np.nan)
        un_meta = meta.get("unanswered")
        if un_meta and (vdir / "unanswered.pt").exists():
            x_un = np.concatenate(
                [feats[["log1p_bytes_out", "log1p_pkts_out"]]
                 .to_numpy(dtype=np.float32),
                 _svc_onehot(un_meta["svc_classes"])], axis=1)
            unm = MiniMLP(x_un.shape[1], 2, hidden=16)
            unm.load_state_dict(torch.load(vdir / "unanswered.pt",
                                           map_location="cpu"))
            unm.eval()
            unans_conf[:] = torch.softmax(unm(torch.from_numpy(x_un)),
                                          dim=1).numpy()[:, 1]

        # next-service prediction: window ending at each flow -> what follows
        pred_next = np.full(len(df), None, dtype=object)
        pred_next_conf = np.full(len(df), np.nan)
        if ns_meta and (vdir / "nextservice.pt").exists():
            gru = NextServiceGRU()
            gru.load_state_dict(torch.load(vdir / "nextservice.pt", map_location="cpu"))
            gru.eval()
            k_win = ns_meta["k"]
            cls_idx = {c: i for i, c in enumerate(ns_classes)}
            pad_id = len(ns_classes)
            sdf = df[["flow_id", "sa", "time_start", "service_class"]].sort_values(
                ["sa", "time_start", "flow_id"])
            ids_l, gaps_l, fid_l = [], [], []
            for _, g in sdf.groupby("sa", sort=False):
                seq = [(cls_idx.get(c, cls_idx["other"]), t)
                       for c, t in zip(g.service_class, g.time_start)]
                for i, fid in enumerate(g.flow_id):
                    win = seq[max(0, i - k_win + 1): i + 1]
                    ids_l.append([pad_id] * (k_win - len(win)) + [c for c, _ in win])
                    gaps_l.append([0.0] * (k_win - len(win)) + [
                        math.log1p(win[j][1] - win[j - 1][1]) if j > 0 else 0.0
                        for j in range(len(win))])
                    fid_l.append(fid)
            nprobs = torch.softmax(gru((
                torch.tensor(ids_l, dtype=torch.int64),
                torch.tensor(gaps_l, dtype=torch.float32))), dim=1).numpy()
            for row, fid in enumerate(fid_l):
                kk = nprobs[row].argmax()
                pred_next[pos[fid]] = ns_classes[kk]
                pred_next_conf[pos[fid]] = float(nprobs[row][kk])

        # TLS client-stack prediction from record structure (TLS flows only)
        pred_tls = np.full(len(df), None, dtype=object)
        pred_tls_conf = np.full(len(df), np.nan)
        if tls_meta and (vdir / "tlsrecord.pt").exists() and len(tls_df):
            tlsm = TlsRecordGRU()
            tlsm.load_state_dict(torch.load(vdir / "tlsrecord.pt", map_location="cpu"))
            tlsm.eval()
            k_tls = tls_meta["k"]
            tp_slot = {20: 1, 21: 2, 22: 3, 23: 4}
            xt = np.zeros((len(tls_df), k_tls, 7), dtype=np.float32)
            for n, recs in enumerate(tls_df.recs):
                for i, r in enumerate(list(recs)[:k_tls]):
                    xt[n, i, 0] = 1.0 if r["dir"] == ">" else -1.0
                    slot = tp_slot.get(int(r["tp"]))
                    if slot:
                        xt[n, i, slot] = 1.0
                    xt[n, i, 5] = math.log1p(float(r["b"]))
                    xt[n, i, 6] = math.log1p(float(r["ipt"]))
            tprobs = torch.softmax(tlsm(torch.from_numpy(xt)), dim=1).numpy()
            for row, fid in enumerate(tls_df.flow_id.to_numpy()):
                if fid in pos:
                    kk = tprobs[row].argmax()
                    pred_tls[pos[fid]] = tls_classes[kk]
                    pred_tls_conf[pos[fid]] = float(tprobs[row][kk])

        # DNS name-shape anomaly: per-name char-LM bits/char, flow-level max
        dns_score = np.full(len(df), np.nan)
        dns_top_qn = np.full(len(df), None, dtype=object)
        if dns_meta and (vdir / "dnslm.pt").exists() and len(dns_df):
            lm = DnsCharLM()
            lm.load_state_dict(torch.load(vdir / "dnslm.pt", map_location="cpu"))
            lm.eval()
            max_chars = dns_meta["metrics"]["max_chars"]
            cidx = {c: 4 + i for i, c in enumerate(_dns_charset)}
            names = sorted(dns_df.qname.unique())
            L = max_chars + 1
            xs = np.zeros((len(names), L), dtype=np.int64)
            ys = np.zeros((len(names), L), dtype=np.int64)
            for n, q in enumerate(names):
                ids = [cidx.get(c, 3) for c in q.lower()[:max_chars]]
                xs[n, :len(ids) + 1] = [1] + ids
                ys[n, :len(ids) + 1] = ids + [2]
            with torch.no_grad():
                logits = lm(torch.from_numpy(xs))
                nll = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), torch.from_numpy(ys),
                    ignore_index=0, reduction="none")
                msk = (torch.from_numpy(ys) != 0).float()
                bits = ((nll * msk).sum(dim=1) / msk.sum(dim=1).clamp(min=1)
                        / math.log(2)).numpy()
            name_bits = dict(zip(names, bits))
            for fid, grp in dns_df.groupby("flow_id"):
                if fid not in pos:
                    continue
                scored = [(name_bits[q], q) for q in grp.qname]
                b, q = max(scored)
                dns_score[pos[fid]] = b
                dns_top_qn[pos[fid]] = q

    return pd.DataFrame({
        "flow_id": df.flow_id,
        "anomaly_score": anomaly,
        "emb_x": emb[:, 0], "emb_y": emb[:, 1], "emb_z": emb[:, 2],
        "pred_service": pred_service,
        "pred_confidence": pred_conf,
        "pred_next_service": pred_next,
        "pred_next_confidence": pred_next_conf,
        "pred_tls_stack": pred_tls,
        "pred_tls_confidence": pred_tls_conf,
        "dns_name_score": dns_score,
        "dns_top_qname": dns_top_qn,
        "bd_recon_score": bd_recon,
        "pred_service_vol": pred_vol,
        "pred_service_vol_confidence": pred_vol_conf,
        "bytes_in_abs_err": bytes_in_err,
        "pred_unanswered_conf": unans_conf,
        "weak_label_service": df.service_class,
        "model_version": version,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    df = dbt.ref("feat_flows").df()
    bd_df = dbt.ref("stg_flows").df()[["flow_id", "byte_dist"]]
    tls_df = dbt.ref("feat_tls").df()[["flow_id", "recs"]]
    dns_df = dbt.ref("feat_dns").df()[["flow_id", "qname"]]

    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        # untrained placeholder — supports `unknown` outcomes (guideline 4)
        return pd.DataFrame({
            "flow_id": df.flow_id,
            "anomaly_score": np.nan,
            "emb_x": np.nan, "emb_y": np.nan, "emb_z": np.nan,
            "pred_service": None,
            "pred_confidence": np.nan,
            "pred_next_service": None,
            "pred_next_confidence": np.nan,
            "pred_tls_stack": None,
            "pred_tls_confidence": np.nan,
            "dns_name_score": np.nan,
            "dns_top_qname": None,
            "bd_recon_score": np.nan,
            "pred_service_vol": None,
            "pred_service_vol_confidence": np.nan,
            "bytes_in_abs_err": np.nan,
            "pred_unanswered_conf": np.nan,
            "weak_label_service": df.service_class,
            "model_version": "untrained",
        })
    return _scores(df, bd_df, tls_df, dns_df, active[0], active[1])

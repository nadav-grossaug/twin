"""Realism audit of synthetic Joy-schema flows, run INSIDE the enclave.

    python3 scripts/realism_audit.py --synth synth      (make realism-audit)

The synthetic file is generated from the DP release only, so it may enter the
enclave. The audit treats it like real traffic and asks the enclave's models
and the real data how realistic it is:

1. Replay: the production dbt pipeline builds the synthetic file into a
   scratch DuckDB (same models, same tests).
2. Anomaly autoencoder (the ACTIVE minimodels artifact, trained on REAL
   flows): share of synthetic flows whose anomaly score exceeds the real
   data's p95 / p99 anomaly percentiles (from artifacts/enclave.duckdb
   ml_flow_scores, read-only). The architecture is rebuilt from the artifact's
   meta (numeric_features / scaler) exactly as dbt/models/ml/ml_flow_scores.py
   does.
3. TV distance real-vs-synth of the duration and bytes_total histograms on
   fixed PUBLIC log-spaced bins.

Output: artifacts/reports/realism_audit.json — aggregate numbers only, and
ENCLAVE-ONLY: it contains non-DP aggregates of real data, so the release
policy does not export it. A graded audit: no pass/fail.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from artifacts import load_active  # noqa: E402
import synth as S  # noqa: E402  (run_dbt, read_meta — same replay path as validate)

REAL_DB = ROOT / "artifacts" / "enclave.duckdb"
OUT = ROOT / "artifacts" / "reports" / "realism_audit.json"
# fixed public bins (log-spaced), data-independent
DURATION_MS_EDGES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
BYTES_EDGES = [30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]


def features(df: pd.DataFrame, meta: dict) -> np.ndarray:
    """Featurization copied from dbt/models/ml/ml_flow_scores.py (same model contract)."""
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
    return (feats.to_numpy(dtype=np.float32) - mu) / sd


def anomaly_scores(xz: np.ndarray, vdir: Path, meta: dict) -> np.ndarray:
    import torch
    import torch.nn as nn

    n_in = len(meta["numeric_features"])

    class FlowAutoencoder(nn.Module):               # arch as in ml_flow_scores.py / training
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(),
                                         nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 8), nn.ReLU(),
                                         nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, n_in))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = FlowAutoencoder()
    ae.load_state_dict(torch.load(vdir / "autoencoder.pt", map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        x = torch.from_numpy(xz)
        return ((ae(x) - x) ** 2).mean(dim=1).numpy()


def hist_tv(real: np.ndarray, syn: np.ndarray, edges: list) -> dict:
    bins = [-np.inf] + list(edges) + [np.inf]
    pr, _ = np.histogram(real, bins=bins)
    ps, _ = np.histogram(syn, bins=bins)
    pr = pr / max(pr.sum(), 1)
    ps = ps / max(ps.sum(), 1)
    return {"tv": round(float(0.5 * np.abs(pr - ps).sum()), 4), "edges": edges,
            "real_p": [round(float(x), 4) for x in pr],
            "synth_p": [round(float(x), 4) for x in ps]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, default=Path("synth"))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if not REAL_DB.exists():
        raise SystemExit("needs the real enclave warehouse: run `make dbt ml` first")

    db = a.synth / "scratch_audit.duckdb"                   # 1. replay through the production dbt
    r = S.run_dbt(a.synth / S.FLOW_JSONL, db)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-2000:])
        raise SystemExit("dbt replay of the synthetic file failed")
    scon = duckdb.connect(str(db), read_only=True)
    syn = scon.execute("""select bytes_out, bytes_in, num_pkts_out, num_pkts_in, duration_s,
                                 bytes_per_s, entropy, is_zero_byte, is_unanswered,
                                 src_internal, dst_internal, hour_utc, bytes_total
                          from feat_flows""").df()
    scon.close()

    version, vdir, meta = load_active("minimodels")         # 2. ACTIVE real-data-trained models
    scores = anomaly_scores(features(syn, meta), vdir, meta)

    rcon = duckdb.connect(str(REAL_DB), read_only=True)
    real_scores = rcon.execute(
        "select anomaly_score from ml_flow_scores where anomaly_score is not null").df().anomaly_score.to_numpy()
    real = rcon.execute("select duration_s, bytes_total from feat_flows").df()
    rcon.close()
    if not len(real_scores):
        raise SystemExit("real ml_flow_scores has no scores: run `make train ml` first")
    p95, p99 = float(np.percentile(real_scores, 95)), float(np.percentile(real_scores, 99))

    meta_s = S.read_meta(a.synth)
    report = {
        "synth": str(a.synth), "n_synth_flows": int(len(syn)),
        "synth_release": (meta_s.get("release") or {}).get("export_id"),
        "model_version": version,
        "anomaly_autoencoder": {
            "real_p95": round(p95, 4), "real_p99": round(p99, 4),
            "synth_share_above_real_p95": round(float((scores > p95).mean()), 4),
            "synth_share_above_real_p99": round(float((scores > p99).mean()), 4),
            "synth_median_score": round(float(np.median(scores)), 4),
            "real_median_score": round(float(np.median(real_scores)), 4),
            "reading": "shares near 5% / 1% would mean synthetic flows look as typical as real ones; "
                       "note synthetic flows carry entropy=null in the JSONL, which feat_flows' "
                       "least(entropy, 8.0) (NULL-ignoring in DuckDB) turns into entropy=8.0, so the "
                       "model sees a uniform, slightly atypical entropy for all synthetic flows"},
        "tv_histograms": {
            "duration_ms": hist_tv(real.duration_s.to_numpy() * 1000,
                                   syn.duration_s.to_numpy() * 1000, DURATION_MS_EDGES),
            "bytes_total": hist_tv(real.bytes_total.to_numpy(),
                                   syn.bytes_total.to_numpy(), BYTES_EDGES)},
        "note": "ENCLAVE-ONLY (contains non-DP aggregates of real data); graded audit, no pass/fail",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    an = report["anomaly_autoencoder"]
    print(f"anomaly: {an['synth_share_above_real_p95']:.1%} of {len(syn)} synthetic flows above the real "
          f"p95, {an['synth_share_above_real_p99']:.1%} above the real p99")
    print(f"TV duration_ms {report['tv_histograms']['duration_ms']['tv']}, "
          f"bytes_total {report['tv_histograms']['bytes_total']['tv']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

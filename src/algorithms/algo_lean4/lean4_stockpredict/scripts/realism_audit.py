#!/usr/bin/env python3
"""Realism audit (ENCLAVE-ONLY): judge the synthetic series with the
production scoring models and enclave statistics. Graded, not pass/fail —
the generator targets the spec contract and the release, not realism;
the gap is reported honestly.

Measures:
  anomaly        share of synthetic 10-day windows scoring above the real
                 data's p99 reconstruction error (ReturnWindowAE)
  class mix      TV distance real vs synth over the public 5 classes
  vol clustering lag-1 autocorrelation of |ret| (GARCH footprint), real vs synth
  collinearity   mean pairwise same-sign share and Pearson r, real vs synth
                 (the declared fields_not_modelled gap, quantified)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def load_tsv(path: Path) -> dict[str, list[float]]:
    out: dict[str, list] = defaultdict(list)
    for r in csv.DictReader(open(path), delimiter="\t"):
        out[r["stockname"]].append(float(r["stockprice"]))
    return dict(out)


def rets_bp(prices: list[float]) -> list[float]:
    return [round(math.log(b / a) * 10000) for a, b in zip(prices, prices[1:])]


def class_of(bp: float) -> str:
    if bp <= -300:
        return "crash"
    if bp < -50:
        return "down"
    if bp <= 50:
        return "flat"
    if bp < 300:
        return "up"
    return "surge"


def abs_autocorr1(rs: list[float]) -> float | None:
    a = np.abs(np.array(rs, dtype=float))
    if len(a) < 3 or a.std() == 0:
        return None
    x, y = a[:-1], a[1:]
    denom = x.std() * y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / denom) if denom else None


def pairwise(series: dict[str, list[float]]):
    ticks = sorted(series)
    rs = {t: rets_bp(series[t]) for t in ticks}
    same, rhos = [], []
    for i, a in enumerate(ticks):
        for b in ticks[i + 1:]:
            n = min(len(rs[a]), len(rs[b]))
            xa, xb = np.array(rs[a][:n]), np.array(rs[b][:n])
            nz = (np.abs(xa) > 50) | (np.abs(xb) > 50)
            if nz.sum():
                same.append(float((np.sign(xa[nz]) == np.sign(xb[nz])).mean()))
            if xa.std() and xb.std():
                rhos.append(float(np.corrcoef(xa, xb)[0, 1]))
    return (float(np.mean(same)) if same else None,
            float(np.mean(rhos)) if rhos else None)


def ae_scores(series: dict[str, list[float]], meta, ae) -> np.ndarray:
    import torch
    mu = meta["return_ae"]["metrics"]["scaler_mean"]
    sd = meta["return_ae"]["metrics"]["scaler_std"]
    win = meta["return_ae"]["metrics"]["win"]
    scores = []
    for t, prices in series.items():
        rs = np.array(rets_bp(prices), dtype=np.float32) / 100.0
        for i in range(win, len(rs) + 1):
            x = torch.tensor((rs[i - win:i] - mu) / sd).unsqueeze(0)
            with torch.no_grad():
                scores.append(float(((ae(x) - x) ** 2).mean()))
    return np.array(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "stock_synth.tsv"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    real = load_tsv(ROOT / "stock_normalized_train.tsv")
    synth = load_tsv(synth_p)

    report: dict = {"note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)"}
    # class mix TV
    def mix(series):
        return Counter(class_of(bp) for p in series.values() for bp in rets_bp(p))
    mr, ms = mix(real), mix(synth)
    tr, ts = sum(mr.values()), sum(ms.values())
    report["class_mix_tv_real_vs_synth"] = round(0.5 * sum(
        abs(mr[c] / tr - ms[c] / ts)
        for c in ("crash", "down", "flat", "up", "surge")), 4)
    # vol clustering
    report["abs_autocorr1"] = {
        "real_mean": round(float(np.mean([a for p in real.values()
                                          if (a := abs_autocorr1(rets_bp(p))) is not None])), 4),
        "synth_mean": round(float(np.mean([a for p in synth.values()
                                           if (a := abs_autocorr1(rets_bp(p))) is not None])), 4),
    }
    # collinearity (the declared gap, measured)
    r_same, r_rho = pairwise(real)
    s_same, s_rho = pairwise(synth)
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731
    report["collinearity"] = {
        "real": {"mean_same_sign_share": rnd(r_same),
                 "mean_pearson_r": rnd(r_rho)},
        "synth": {"mean_same_sign_share": rnd(s_same),
                  "mean_pearson_r": rnd(s_rho)},
        "note": "synth coupling is driven only by the DP co-movement release "
                "(comove_lambda_pct); exact correlation structure is a "
                "declared fields_not_modelled gap",
    }
    # anomaly scores through the production AE
    reg = ROOT / "models_out" / "registry.json"
    if reg.exists():
        import torch.nn as nn
        import torch
        active = json.loads(reg.read_text())["minimodels"]["active"]
        vdir = ROOT / "models_out" / "minimodels" / active
        meta = json.loads((vdir / "meta.json").read_text())

        class ReturnWindowAE(nn.Module):
            def __init__(self, n_in=10):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),
                                             nn.Linear(8, 2))
                self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                             nn.Linear(8, n_in))

            def forward(self, x):
                return self.decoder(self.encoder(x))

        ae = ReturnWindowAE()
        ae.load_state_dict(torch.load(vdir / "return_ae.pt", map_location="cpu"))
        ae.eval()
        real_s = ae_scores(real, meta, ae)
        synth_s = ae_scores(synth, meta, ae)
        p99 = float(np.quantile(real_s, 0.99))
        report["anomaly"] = {
            "real_p99_recon": round(p99, 4),
            "synth_share_above_real_p99": round(float((synth_s > p99).mean()), 4),
            "model_version": active,
        }
    out = ROOT / "artifacts" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "realism_audit.json").write_text(json.dumps(report, indent=2))
    print(f"realism-audit: {json.dumps(report, indent=2)[:400]} ...")
    print("-> artifacts/reports/realism_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Realism audit (ENCLAVE-ONLY): judge the synthetic transfers with the
production scoring models and enclave statistics. Graded, not pass/fail —
the generator targets the spec contract and the release, not realism;
the gap is reported honestly.

Measures:
  anomaly       share of synthetic user profiles scoring above the real
                data's p99 reconstruction error (UserProfileAE)
  mixes         TV distance real vs synth over the public travel-count bins,
                gap classes and destination regions
  return rate   mean per-user return rate (A->B then B->A), real vs synth
                (the second-order behavior the return_rate release carries)
  tempo         median inter-transfer gap (days), real vs synth
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec, region_of  # noqa: E402

# the full public TSV schema, in order (validated on load)
COLUMNS = ["userid", "dt", "prev_country", "prev_last_venue_id",
           "prev_last_lat", "prev_last_long", "prev_duration",
           "next_country", "next_first_venue_id", "next_first_lat",
           "next_first_long", "next_duration"]
TC_BINS = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
GAP_CLASSES = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
REGIONS = ["NA", "LA", "EU", "ME", "AF", "SA", "EA", "OC"]

_SPEC = load_spec()


def load_tsv(path: Path) -> dict[int, list[dict]]:
    """userid -> transfers ordered by dt. Validates the public schema."""
    out: dict[int, list] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: header differs from the public "
                             f"schema (got {reader.fieldnames})")
        for r in reader:
            ts = datetime.strptime(r["dt"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
            out[int(r["userid"])].append(
                {"ts_s": int(ts), "a": r["prev_country"], "b": r["next_country"]})
    return {u: sorted(rows, key=lambda x: x["ts_s"]) for u, rows in out.items()}


def tc_bin(n: int) -> str:
    edges = [int(_SPEC[k]) for k in ("tc_e2", "tc_e3", "tc_e5", "tc_e9",
                                     "tc_e17", "tc_e33")]
    for name, lo in zip(reversed(TC_BINS), reversed([1] + edges)):
        if n >= lo:
            return name
    return TC_BINS[0]


def gap_class(gap_s: int) -> str:
    if gap_s < int(_SPEC["gap_hop_max_s"]):
        return "gap_hop"
    if gap_s < int(_SPEC["gap_trip_max_s"]):
        return "gap_trip"
    if gap_s < int(_SPEC["gap_monthly_max_s"]):
        return "gap_monthly"
    if gap_s < int(_SPEC["gap_quarterly_max_s"]):
        return "gap_quarterly"
    return "gap_rare"


def gaps_of(rows: list[dict]) -> list[int]:
    return [b["ts_s"] - a["ts_s"] for a, b in zip(rows, rows[1:])]


def return_rate(rows: list[dict]) -> float | None:
    pairs = list(zip(rows, rows[1:]))
    if not pairs:
        return None
    hits = sum(1 for x, y in pairs if y["a"] == x["b"] and y["b"] == x["a"])
    return hits / len(pairs)


def profile(rows: list[dict]) -> list[float]:
    """Mirror of feat_users -> UserProfileAE features (n_travels, n_countries,
    return_rate_pct, mean/sd gap days), nulls coalesced to 0 as in training.
    n_countries here = distinct countries on either side of a transfer."""
    gaps_d = [g / 86400.0 for g in gaps_of(rows)]
    rr = return_rate(rows)
    return [float(len(rows)),
            float(len({r["a"] for r in rows} | {r["b"] for r in rows})),
            0.0 if rr is None else 100.0 * rr,
            float(np.mean(gaps_d)) if gaps_d else 0.0,
            float(np.std(gaps_d, ddof=1)) if len(gaps_d) > 1 else 0.0]


def transform(raw: np.ndarray) -> np.ndarray:
    """Must match train_minimodels.py: log1p on counts and gaps, pct -> [0,1]."""
    return np.stack([np.log1p(raw[:, 0]), np.log1p(raw[:, 1]),
                     raw[:, 2] / 100.0,
                     np.log1p(raw[:, 3]), np.log1p(raw[:, 4])], axis=1)


def tv(a: Counter, b: Counter, grid: list[str]) -> float:
    ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
    return 0.5 * sum(abs(a[c] / ta - b[c] / tb) for c in grid)


def ae_scores(users: dict[int, list[dict]], meta, ae) -> np.ndarray:
    import torch
    m = meta["profile_ae"]["metrics"]
    mu = np.array(m["scaler_mean"], dtype=np.float32)
    sd = np.array(m["scaler_std"], dtype=np.float32)
    raw = np.array([profile(rows) for rows in users.values()], dtype=np.float32)
    x = torch.from_numpy((transform(raw) - mu) / sd)
    with torch.no_grad():
        return ((ae(x) - x) ** 2).mean(dim=1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "travel_synth.tsv"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    real = load_tsv(ROOT / "country_transfer.tsv")
    synth = load_tsv(synth_p)

    report: dict = {"note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)"}

    # public class mixes, TV real vs synth
    def mixes(users: dict[int, list[dict]]):
        tc = Counter(tc_bin(len(rows)) for rows in users.values())
        gc = Counter(gap_class(g) for rows in users.values()
                     for g in gaps_of(rows))
        rg = Counter(region_of(r["b"]) for rows in users.values() for r in rows)
        return tc, gc, rg

    r_tc, r_gc, r_rg = mixes(real)
    s_tc, s_gc, s_rg = mixes(synth)
    report["mix_tv_real_vs_synth"] = {
        "travel_count": round(tv(r_tc, s_tc, TC_BINS), 4),
        "gap_class": round(tv(r_gc, s_gc, GAP_CLASSES), 4),
        "region": round(tv(r_rg, s_rg, REGIONS), 4),
    }
    # second-order return behavior (the return_rate release, measured)
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731

    def mean_rr(users):
        vals = [rr for rows in users.values()
                if (rr := return_rate(rows)) is not None]
        return float(np.mean(vals)) if vals else None

    report["return_rate"] = {
        "real_mean": rnd(mean_rr(real)),
        "synth_mean": rnd(mean_rr(synth)),
        "note": "synth returns are driven only by the released return_rate "
                "bands + the return_boost prior; exact per-corridor return "
                "structure is a declared gap",
    }
    # tempo
    def med_gap_days(users):
        g = [x / 86400.0 for rows in users.values() for x in gaps_of(rows)]
        return float(np.median(g)) if g else None

    report["median_gap_days"] = {"real": rnd(med_gap_days(real)),
                                 "synth": rnd(med_gap_days(synth))}

    # countries: marginal TV (both sides pooled), coverage, in/out balance,
    # hour-of-day TV — the four 2026-11-03 quality briefs, measured honestly
    def country_stats(users):
        pooled, cin, cout = Counter(), Counter(), Counter()
        hod = Counter()
        for rows in users.values():
            for r in rows:
                pooled[r["a"]] += 1
                pooled[r["b"]] += 1
                cin[r["b"]] += 1
                cout[r["a"]] += 1
                hod[datetime.fromtimestamp(
                    r["ts_s"], tz=timezone.utc).hour] += 1
        imbal = [abs(cin[c] - cout[c]) / max(1, cin[c] + cout[c])
                 for c in set(cin) | set(cout)]
        return pooled, hod, float(np.mean(imbal)) if imbal else None

    r_cc, r_hod, r_imb = country_stats(real)
    s_cc, s_hod, s_imb = country_stats(synth)
    grid_c = sorted(set(r_cc) | set(s_cc))
    report["countries"] = {
        "tv_real_vs_synth": round(tv(r_cc, s_cc, grid_c), 4),
        "n_real": len(r_cc), "n_synth": len(s_cc),
        "note": "synth countries come only from the suppressed topk_countries "
                "release (70 of 77; 99.7% of appearances)",
    }
    report["inout_imbalance_mean"] = {"real": rnd(r_imb), "synth": rnd(s_imb)}
    report["hod"] = {
        "tv_real_vs_synth": round(
            tv(r_hod, s_hod, list(range(24))), 4),
        "note": "synth hours come from the DP hod histogram (ledger v2.1); "
                "minutes/seconds are a declared uniform prior",
    }

    # anomaly scores through the production profile AE
    reg = ROOT / "models_out" / "registry.json"
    if reg.exists():
        import torch
        import torch.nn as nn
        active = json.loads(reg.read_text())["minimodels"]["active"]
        vdir = ROOT / "models_out" / "minimodels" / active
        meta = json.loads((vdir / "meta.json").read_text())

        class UserProfileAE(nn.Module):
            def __init__(self, n_in=5):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),
                                             nn.Linear(8, 2))
                self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                             nn.Linear(8, n_in))

            def forward(self, x):
                return self.decoder(self.encoder(x))

        ae = UserProfileAE()
        ae.load_state_dict(torch.load(vdir / "profile_ae.pt", map_location="cpu"))
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

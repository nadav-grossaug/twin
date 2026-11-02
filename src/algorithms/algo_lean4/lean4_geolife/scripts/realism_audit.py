#!/usr/bin/env python3
"""Realism audit (ENCLAVE-ONLY): judge the synthetic GPS trajectories with
the production scoring models and enclave statistics. Graded, not pass/fail —
the generator targets the spec contract and the release, not realism;
the gap is reported honestly, and so is what this audit does NOT measure.

Measured:
  mixes         TV distance real vs synth over the public mode classes
                (point share, labels mapped through spec.mode_class_of) and
                the public step-speed classes; when `make synth` validation
                left its re-ingested scratch DB (synth/scratch.duckdb), also
                over the daily-motif classes, dwell classes and route
                circuity classes (circ_class) as the
                PRODUCTION dbt models re-detect them
  route shape   per-mode median straightness real (feat_route_legs) vs synth
                (scratch feat_route_legs) — are the trips road-like or
                straight vectors?
  mode model    ModeLegMLP (the production kinematics teacher) run on real
                legs (feat_legs) and on synthetic legs rebuilt from the TSV
                (consecutive same-mode points per trajectory): agreement with
                the declared mode labels, plus median speed per mode class
Skipped (honestly):
  anomaly       share of synthetic users above the real users' p99
                UserProfileAE reconstruction error needs the per-user profile
                features (mart_user_stats) for the SYNTHETIC cohort, i.e. a
                full dbt profile build — not recomputed here
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from spec import (DWELL_CLASSES, MODE_CLASSES, MOTIF_CLASSES,  # noqa: E402
                  SPEED_CLASSES, load_spec, mode_class_of)

# the full public TSV schema, in order (validated on load)
COLUMNS = ["user_id", "sequence_uuid", "trajectory_id", "datetime",
           "latitude", "longitude", "altitude_ft", "label_start",
           "label_end", "mode"]

DB = ROOT / "artifacts" / "enclave.duckdb"
_SPEC = load_spec()
# public route circuity classes (mirror of the dbt circ_e* spec vars)
CIRC_CLASSES = ["circ_lt120", "circ_120_150", "circ_150_250", "circ_ge250"]
# training-order leg features; log1p on path, duration and sample step,
# turn fillna 0, straightness fillna 1 (mirror of train_minimodels.py)
LEG_RAW = ("path_m", "duration_s", "median_speed_mps", "p95_speed_mps",
           "median_abs_turn_deg", "straightness", "median_dt_s")


def _mode_class(label: str) -> str:
    """Raw GeoLife labels (train, taxi, ...) map through the public seed;
    the Lean generator declares the public class labels (rail, other, ...)
    directly, so an already-public class passes through unchanged."""
    c = mode_class_of(label)
    if c == "XX" and label in MODE_CLASSES:
        return label
    return c


def _ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp())


def haversine_m(la1, lo1, la2, lo2) -> float:
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(a)))


def speed_class(v_mps: float) -> str:
    mmps = v_mps * 1000.0
    if mmps < int(_SPEC["spd_still_max_mmps"]):
        return "spd_still"
    if mmps < int(_SPEC["spd_walk_max_mmps"]):
        return "spd_walk"
    if mmps < int(_SPEC["spd_city_max_mmps"]):
        return "spd_city"
    if mmps < int(_SPEC["spd_fast_max_mmps"]):
        return "spd_fast"
    return "spd_vfast"


def load_synth(path: Path) -> dict[tuple, list[tuple]]:
    """(user_id, trajectory_id) -> [(ts_s, lat, lon, mode_class), ...] in
    time order. Validates the public 10-column schema."""
    out: dict[tuple, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: header differs from the public "
                             f"schema (got {reader.fieldnames})")
        for r in reader:
            out.setdefault((r["user_id"], r["trajectory_id"]), []).append(
                (_ts(r["datetime"]), float(r["latitude"]),
                 float(r["longitude"]), _mode_class(r["mode"])))
    return {k: sorted(v) for k, v in out.items()}


def synth_legs(trajs: dict[tuple, list[tuple]]) -> list[tuple[str, list[float]]]:
    """Consecutive same-mode-class runs per trajectory -> (mode_class,
    training-order raw leg features). Mirrors the feat_legs conventions:
    unknown-mode (XX) runs dropped, turn fillna 0, straightness fillna 1."""
    legs = []
    for pts in trajs.values():
        i = 0
        while i < len(pts):
            j = i
            while j + 1 < len(pts) and pts[j + 1][3] == pts[i][3]:
                j += 1
            run = pts[i:j + 1]
            i = j + 1
            mode = run[0][3]
            if mode == "XX" or len(run) < 2:
                continue
            dists, dts, speeds, headings = [], [], [], []
            for (t0, la0, lo0, _), (t1, la1, lo1, _) in zip(run, run[1:]):
                d, dt = haversine_m(la0, lo0, la1, lo1), t1 - t0
                dists.append(d)
                if dt > 0:
                    dts.append(float(dt))
                    speeds.append(d / dt)
                if d > 0:
                    headings.append(math.degrees(math.atan2(
                        math.radians(lo1 - lo0)
                        * math.cos(math.radians((la0 + la1) / 2)),
                        math.radians(la1 - la0))))
            duration = run[-1][0] - run[0][0]
            if duration <= 0 or not speeds:
                continue
            turns = [min(abs(h1 - h0), 360 - abs(h1 - h0))
                     for h0, h1 in zip(headings, headings[1:])]
            path = float(sum(dists))
            net = haversine_m(run[0][1], run[0][2], run[-1][1], run[-1][2])
            legs.append((mode, [
                path, float(duration), float(np.median(speeds)),
                float(np.quantile(speeds, 0.95)),
                float(np.median(turns)) if turns else 0.0,
                net / path if path > 0 else 1.0,
                float(np.median(dts))]))
    return legs


def synth_speed_mix(trajs: dict[tuple, list[tuple]]) -> Counter:
    """Step-speed class mix from the TSV, with the public plausibility
    filters feat_steps applies (dt within a stream break, speed cap)."""
    brk, cap = int(_SPEC["sp_stream_break_s"]), float(_SPEC["max_speed_mps"])
    mix: Counter = Counter()
    for pts in trajs.values():
        for (t0, la0, lo0, _), (t1, la1, lo1, _) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0 or dt > brk:
                continue
            v = haversine_m(la0, lo0, la1, lo1) / dt
            if v > cap:
                continue
            mix[speed_class(v)] += 1
    return mix


def tv(a: Counter, b: Counter, grid: list[str]) -> float:
    ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
    return 0.5 * sum(abs(a[c] / ta - b[c] / tb) for c in grid)


def featurize(raw: np.ndarray) -> np.ndarray:
    """Must match train_minimodels.py: log1p on path_m, duration_s and
    median_dt_s; speeds, turn and straightness untouched."""
    return np.stack([np.log1p(raw[:, 0]), np.log1p(raw[:, 1]), raw[:, 2],
                     raw[:, 3], raw[:, 4], raw[:, 5],
                     np.log1p(raw[:, 6])], axis=1)


def mode_agreement(legs: list[tuple[str, list[float]]], meta, mlp) -> dict:
    import torch
    m = meta["mode_leg"]["metrics"]
    mu = np.array(m["scaler_mean"], dtype=np.float32)
    sd = np.array(m["scaler_std"], dtype=np.float32)
    raw = np.array([f for _, f in legs], dtype=np.float32)
    labels = np.array([MODE_CLASSES.index(c) for c, _ in legs], dtype=np.int64)
    x = torch.from_numpy((featurize(raw) - mu) / sd)
    with torch.no_grad():
        pred = mlp(x).argmax(1).numpy()
    return {"n_legs": int(len(legs)),
            "agreement_with_declared_mode": round(float((pred == labels).mean()), 4)}


def median_speed_per_mode(legs: list[tuple[str, list[float]]]) -> dict:
    by: dict[str, list[float]] = {}
    for mode, f in legs:
        by.setdefault(mode, []).append(f[2])
    return {m: round(float(np.median(by[m])), 3)
            for m in MODE_CLASSES if m in by}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "geolife_synth.tsv"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    if not DB.exists():
        print("ERROR: no artifacts/enclave.duckdb — this audit is "
              "ENCLAVE-ONLY; run `make train` first", file=sys.stderr)
        return 1
    import duckdb  # enclave-side only
    con = duckdb.connect(str(DB), read_only=True)
    r_mode = Counter(dict(con.execute(
        "select mode_class, count(*) from stg_points "
        "where mode_class != 'XX' group by 1").fetchall()))
    r_spd = Counter(dict(con.execute(
        "select speed_class, count(*) from feat_steps "
        "where not qf_implausible_speed group by 1").fetchall()))
    r_motif = Counter(dict(con.execute(
        "select motif_class, count(*) from mart_motif_daily "
        "group by 1").fetchall()))
    r_dwell = Counter(dict(con.execute(
        "select dwell_class, count(*) from feat_visits "
        "where not arr_censored and not dep_censored group by 1").fetchall()))
    real_leg_rows = con.execute(
        "select mode_class, path_m, duration_s, median_speed_mps, "
        "p95_speed_mps, coalesce(median_abs_turn_deg, 0), "
        "coalesce(straightness, 1), median_dt_s from feat_legs "
        "where mode_class != 'XX'").fetchall()
    # route-shape ground truth (new tables — skip honestly on old warehouses)
    r_circ = r_straight = None
    try:
        r_circ = Counter(dict(con.execute(
            "select circ_class, count(*) from feat_route_legs "
            "where circ_class is not null group by 1").fetchall()))
        r_straight = {m: round(float(v), 4) for m, v in con.execute(
            "select mode_class, median(straightness) from feat_route_legs "
            "where straightness is not null group by 1").fetchall()}
    except Exception:
        pass
    con.close()
    real_legs = [(r[0], [float(v) for v in r[1:]]) for r in real_leg_rows]

    trajs = load_synth(synth_p)
    s_mode = Counter(c for pts in trajs.values()
                     for *_, c in pts if c != "XX")
    s_spd = synth_speed_mix(trajs)
    s_legs = synth_legs(trajs)

    report: dict = {
        "note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)",
        "measured": ["mode_point_share TV", "step_speed_class TV",
                     "ModeLegMLP agreement real vs synth",
                     "median speed per mode real vs synth"],
        "skipped": [],
    }
    mixes = {
        "mode_point_share": round(tv(r_mode, s_mode, MODE_CLASSES), 4),
        "step_speed_class": round(tv(r_spd, s_spd, SPEED_CLASSES), 4),
    }
    # motif / dwell mixes need staypoint re-detection; reuse the production
    # dbt re-ingestion that `make synth` validation already ran, if present
    scratch = ROOT / args.synth / "scratch.duckdb"
    s_circ = s_straight = None
    if scratch.exists():
        con2 = duckdb.connect(str(scratch), read_only=True)
        s_motif = Counter(dict(con2.execute(
            "select motif_class, count(*) from mart_motif_daily "
            "group by 1").fetchall()))
        s_dwell = Counter(dict(con2.execute(
            "select dwell_class, count(*) from feat_visits "
            "where not arr_censored and not dep_censored group by 1").fetchall()))
        try:
            s_circ = Counter(dict(con2.execute(
                "select circ_class, count(*) from feat_route_legs "
                "where circ_class is not null group by 1").fetchall()))
            s_straight = {m: round(float(v), 4) for m, v in con2.execute(
                "select mode_class, median(straightness) from feat_route_legs "
                "where straightness is not null group by 1").fetchall()}
        except Exception:
            pass
        con2.close()
        mixes["motif_class"] = round(tv(r_motif, s_motif, MOTIF_CLASSES), 4)
        mixes["dwell_class"] = round(tv(r_dwell, s_dwell, DWELL_CLASSES), 4)
        report["measured"] += ["motif_class TV (via synth scratch DB)",
                               "dwell_class TV (via synth scratch DB)"]
    else:
        report["skipped"].append(
            "motif_class / dwell_class TV: no synth scratch DB "
            f"({args.synth}/scratch.duckdb) — run `make synth` validation "
            "first; staypoint/motif re-detection is not recomputed here")
    # route-shape comparisons: circ_class mix + per-mode median straightness
    if r_circ is not None and s_circ is not None:
        mixes["circ_class"] = round(tv(r_circ, s_circ, CIRC_CLASSES), 4)
        report["measured"].append("circ_class TV (via synth scratch DB)")
    else:
        report["skipped"].append(
            "circ_class TV: feat_route_legs absent from the "
            + ("enclave DB" if r_circ is None else "synth scratch DB")
            + " — route-shape re-detection is not recomputed here")
    if r_straight is not None and s_straight is not None:
        report["route_median_straightness_per_mode"] = {
            "real": {m: r_straight[m] for m in MODE_CLASSES if m in r_straight},
            "synth": {m: s_straight[m] for m in MODE_CLASSES if m in s_straight},
        }
        report["measured"].append(
            "per-mode median straightness real vs synth (feat_route_legs)")
    else:
        report["skipped"].append(
            "per-mode median straightness: feat_route_legs absent from the "
            + ("enclave DB" if r_straight is None else "synth scratch DB"))
    report["mix_tv_real_vs_synth"] = mixes

    # the production kinematics teacher on real vs synthetic legs
    reg = ROOT / "models_out" / "registry.json"
    if reg.exists() and s_legs:
        import torch
        import torch.nn as nn
        active = json.loads(reg.read_text())["minimodels"]["active"]
        vdir = ROOT / "models_out" / "minimodels" / active
        meta = json.loads((vdir / "meta.json").read_text())

        class ModeLegMLP(nn.Module):
            def __init__(self, n_in=7, n_classes=len(MODE_CLASSES)):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                         nn.Linear(32, n_classes))

            def forward(self, x):
                return self.net(x)

        mlp = ModeLegMLP()
        mlp.load_state_dict(torch.load(vdir / "mode_leg.pt", map_location="cpu"))
        mlp.eval()
        report["mode_model"] = {
            "model_version": active,
            "real_feat_legs": mode_agreement(real_legs, meta, mlp),
            "synth_legs": mode_agreement(s_legs, meta, mlp),
            "note": "synthetic leg features are rebuilt from the TSV "
                    "(consecutive same-mode points per trajectory, same "
                    "fillna conventions); small drift vs the dbt feat_legs "
                    "definitions is methodological, not behavioral",
        }
    else:
        report["skipped"].append(
            "ModeLegMLP agreement: no active minimodel registry "
            "or no scorable synthetic legs")
    report["median_speed_mps_per_mode"] = {
        "real": median_speed_per_mode(real_legs),
        "synth": median_speed_per_mode(s_legs),
    }
    report["anomaly"] = {
        "skipped": True,
        "reason": "share of synthetic users above the real p99 UserProfileAE "
                  "reconstruction error needs mart_user_stats profile "
                  "features for the synthetic cohort (a full dbt profile "
                  "build); the cheap mirror measured instead is the "
                  "ModeLegMLP agreement above",
    }
    out = ROOT / "artifacts" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "realism_audit.json").write_text(json.dumps(report, indent=2))
    print(f"realism-audit: {json.dumps(report, indent=2)[:400]} ...")
    print("-> artifacts/reports/realism_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Realism audit (ENCLAVE-ONLY): judge the synthetic reservations with the
production scoring model and enclave statistics. Graded, not pass/fail —
the generator targets the spec contract and the release, not realism;
the gap is reported honestly.

Measures:
  anomaly       share of synthetic user profiles scoring above the real
                data's p99 reconstruction error (UserProfileAE)
  mixes         TV distance real vs synth over the public reservations-per-
                user bins, stay-night classes, inter-leg gap classes,
                hotel-country tiers, device classes and checkin months
  return rate   mean per-user share of trips with a repeat city, real vs
                synth (the second-order behavior the return_rate release
                carries — 38.07% of real trips revisit a city)
  tempo         median stay nights and median inter-leg gap (days)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec  # noqa: E402

# the full public CSV schema, in order (validated on load)
COLUMNS = ["user_id", "checkin", "checkout", "city_id", "device_class",
           "affiliate_id", "booker_country", "hotel_country", "utrip_id"]
RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
NIGHTS_CELLS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
GAP_CELLS = ["g_0", "g_1", "g_2_3", "g_4p"]
CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
            "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan",
            "Rolisica", "Kasnia", "ct_other"]

_SPEC = load_spec()


def ct_of(country: str) -> str:
    return country if country in CT_CELLS else "ct_other"


def rc_bin_expr(n_col: str) -> str:
    edges = [int(_SPEC[k]) for k in ("rc_e4", "rc_e5", "rc_e6", "rc_e7",
                                     "rc_e9", "rc_e13")]
    e4, e5, e6, e7, e9, e13 = edges
    return (f"case when {n_col} < {e4} then 'rc_1_3' "
            f"when {n_col} < {e5} then 'rc_4' "
            f"when {n_col} < {e6} then 'rc_5' "
            f"when {n_col} < {e7} then 'rc_6' "
            f"when {n_col} < {e9} then 'rc_7_8' "
            f"when {n_col} < {e13} then 'rc_9_12' "
            f"else 'rc_13p' end")


def nights_expr() -> str:
    e4, e7 = int(_SPEC["nights_e4"]), int(_SPEC["nights_e7"])
    return (f"case when nights < {e4} then 'n_' || nights "
            f"when nights < {e7} then 'n_4_6' else 'n_7p' end")


def gap_expr() -> str:
    e2, e4 = int(_SPEC["gap_e2"]), int(_SPEC["gap_e4"])
    return (f"case when gap_days = 0 then 'g_0' when gap_days < {e2} then 'g_1' "
            f"when gap_days < {e4} then 'g_2_3' else 'g_4p' end")


def _assert_schema(con, path: Path) -> None:
    header = [c[0] for c in con.execute(
        f"select * from read_csv_auto('{path}') limit 0").description]
    if header != COLUMNS:
        raise ValueError(f"{path.name}: header differs from the public schema "
                         f"(got {header})")


def _mixes(con, path: Path) -> dict:
    _assert_schema(con, path)
    src = f"read_csv_auto('{path}')"
    rc = dict(con.execute(
        f"select {rc_bin_expr('n_res')}, count(*) from ("
        f"select user_id, count(*) n_res from {src} group by 1) group by 1"
    ).fetchall())
    nights = dict(con.execute(
        f"select {nights_expr()}, count(*) from ("
        f"select date_diff('day', checkin, checkout) nights from {src}) "
        f"group by 1").fetchall())
    gap = dict(con.execute(
        f"select {gap_expr()}, count(*) from ("
        f"select utrip_id, checkin, "
        f"date_diff('day', lag(checkout) over (partition by utrip_id "
        f"order by checkin), checkin) gap_days from {src}) "
        f"where gap_days is not null group by 1").fetchall())
    ct = {ct_of(c): n for c, n in con.execute(
        f"select hotel_country, count(*) from {src} group by 1").fetchall()}
    dev = dict(con.execute(
        f"select device_class, count(*) from {src} group by 1").fetchall())
    mon = dict(con.execute(
        f"select strftime(checkin, '%m'), count(*) from {src} "
        f"group by 1").fetchall())
    return {"res_count": rc, "nights": nights, "gap": gap, "country": ct,
            "device": dev, "month": mon}


def _rr(con, path: Path) -> dict:
    """Per-user share of trips with a repeat city (a trip revisits a city)."""
    src = f"read_csv_auto('{path}')"
    rows = con.execute(
        f"select user_id, avg(repeat_flag) from ("
        f"select user_id, utrip_id, "
        f"cast(count(distinct city_id) < count(*) as int) repeat_flag "
        f"from {src} group by 1, 2) group by 1 order by 1").fetchall()
    return {int(u): float(r) for u, r in rows}


def _tempo(con, path: Path) -> dict:
    src = f"read_csv_auto('{path}')"
    med_nights = con.execute(
        f"select median(date_diff('day', checkin, checkout)) from {src}"
    ).fetchone()[0]
    med_gap = con.execute(
        f"select median(gap_days) from ("
        f"select date_diff('day', lag(checkout) over (partition by utrip_id "
        f"order by checkin), checkin) gap_days from {src}) "
        f"where gap_days is not null").fetchone()[0]
    return {"median_nights": float(med_nights), "median_gap_days": float(med_gap)}


def _tier_hists(con, path: Path) -> np.ndarray:
    """Mirror of train_minimodels.py's UserProfileAE features: per-user
    country-tier share histograms over the 16 public tiers (hotel_country
    mapped through the country_tiers seed, unmapped -> ct_other)."""
    named = {}
    seed = ROOT / "dbt" / "seeds" / "country_tiers.csv"
    for line in seed.read_text().splitlines()[1:]:
        if line.strip():
            c, t = line.split(",")
            named[c.strip()] = t.strip()
    rows = con.execute(
        f"select user_id, hotel_country, count(*) n "
        f"from read_csv_auto('{path}') group by 1, 2 order by 1").fetchall()
    hists: dict[int, np.ndarray] = {}
    for u, c, n in rows:
        tier = named.get(c, "ct_other")
        if tier not in CT_CELLS:
            tier = "ct_other"
        h = hists.setdefault(int(u), np.zeros(len(CT_CELLS), dtype=np.float32))
        h[CT_CELLS.index(tier)] += float(n)
    out = np.zeros((len(hists), len(CT_CELLS)), dtype=np.float32)
    for i, h in enumerate(hists.values()):
        tot = h.sum()
        out[i] = h / tot if tot > 0 else h
    return out


def tv(a: dict, b: dict, grid: list[str]) -> float:
    ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
    return 0.5 * sum(abs(a.get(c, 0) / ta - b.get(c, 0) / tb) for c in grid)


def ae_scores(hists: np.ndarray, meta, ae) -> np.ndarray:
    """Per-user reconstruction error of the tier-share histogram through the
    production profile AE, scaled by the train-side scaler frozen in
    meta.json (mirrors train_minimodels.py)."""
    import torch
    m = meta["profile_ae"]["metrics"]
    mu = np.array(m["scaler_mean"], dtype=np.float32)
    sd = np.array(m["scaler_std"], dtype=np.float32)
    x = torch.from_numpy((hists - mu) / sd)
    with torch.no_grad():
        return ((ae(x) - x) ** 2).mean(dim=1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "booking_synth.csv"
    real_p = ROOT / "combined_set.csv"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    if not real_p.exists():
        print(f"ERROR: {real_p} missing", file=sys.stderr)
        return 1

    import duckdb
    con = duckdb.connect()
    report: dict = {"note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)"}

    r_mix = _mixes(con, real_p)
    s_mix = _mixes(con, synth_p)
    report["mix_tv_real_vs_synth"] = {
        "res_count": round(tv(r_mix["res_count"], s_mix["res_count"], RC_BINS), 4),
        "nights": round(tv(r_mix["nights"], s_mix["nights"], NIGHTS_CELLS), 4),
        "gap": round(tv(r_mix["gap"], s_mix["gap"], GAP_CELLS), 4),
        "country": round(tv(r_mix["country"], s_mix["country"], CT_CELLS), 4),
        "device": round(tv(r_mix["device"], s_mix["device"],
                           ["desktop", "mobile", "tablet"]), 4),
        "month": round(tv(r_mix["month"], s_mix["month"],
                          [f"{i:02d}" for i in range(1, 13)]), 4),
    }
    # second-order return behavior (the return_rate release, measured)
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731

    def mean_rr(d):
        vals = list(_rr(con, d).values())
        return float(np.mean(vals)) if vals else None

    report["repeat_city_trip_share"] = {
        "real_mean": rnd(mean_rr(real_p)),
        "synth_mean": rnd(mean_rr(synth_p)),
        "note": "synth repeat behavior is driven only by the released "
                "return_rate bands + repeat_struct anatomy (spec 1.2.0)",
    }

    # consecutive same-city legs (the stay-extension structure): 9.01% of
    # real legs; the pre-1.2.0 generator rendered 1.66% — this row is why
    # the repeat_struct release exists
    def consec_share(p):
        src = f"read_csv_auto('{p}')"
        r = con.execute(
            f"select count(*) filter (where prev = city_id), count(*) from ("
            f"select city_id, lag(city_id) over (partition by utrip_id "
            f"order by checkin) prev from {src}) where prev is not null"
        ).fetchone()
        return round(r[0] / r[1], 4) if r[1] else None

    report["consecutive_same_city_leg_share"] = {
        "real": consec_share(real_p),
        "synth": consec_share(synth_p),
        "note": "A,A stay-extensions (real trips split long stays across "
                "back-to-back bookings in one city); driven by the released "
                "repeat_struct ext-chain cells",
    }

    # v1.3.0: trips per user, per-trip device-class count, affiliate
    # persistence — the structure the per-leg marginals hide
    def trips_per_user(p):
        src = f"read_csv_auto('{p}')"
        n = con.execute(
            f"select count(*) from (select user_id from {src} group by 1 "
            f"having count(distinct utrip_id) > 1)").fetchone()[0]
        t = con.execute(
            f"select count(distinct user_id) from {src}").fetchone()[0]
        return round(n / t, 4) if t else None

    def device_const_trips(p):
        src = f"read_csv_auto('{p}')"
        r = con.execute(
            f"select count(*) filter (where n_dc = 1), count(*) from ("
            f"select utrip_id, count(distinct device_class) n_dc from "
            f"{src} group by 1)").fetchone()
        return round(r[0] / r[1], 4) if r[1] else None

    def aff_same_share(p):
        src = f"read_csv_auto('{p}')"
        # TIER level — the released quantity (affiliate_persist counts
        # tiers per trip; raw id level is confounded by the synthetic tail
        # ids the generator must invent for the ~3,600-id aff_other mass)
        r = con.execute(
            f"select count(*) filter (where a = pa), count(*) from ("
            f"select case when affiliate_id in "
            f"(9924,359,384,9452,4541,8436,10332,3631) "
            f"then 'a' || affiliate_id::varchar else 'aff_other' end a, "
            f"lag(case when affiliate_id in "
            f"(9924,359,384,9452,4541,8436,10332,3631) "
            f"then 'a' || affiliate_id::varchar else 'aff_other' end) over "
            f"(partition by utrip_id order by checkin) pa from {src}) "
            f"where pa is not null").fetchone()
        return round(r[0] / r[1], 4) if r[1] else None

    report["trips_per_user_share"] = {
        "real": trips_per_user(real_p), "synth": trips_per_user(synth_p),
        "note": "multi-trip users (real 1.11 trips/user); driven by the "
                "released trips_per_user cells (spec 1.3.0)"}
    report["single_device_trip_share"] = {
        "real": device_const_trips(real_p),
        "synth": device_const_trips(synth_p),
        "note": "trips booked entirely on one device class (real 70.7%); "
                "driven by the released device_persist cells (spec 1.3.0)"}
    report["affiliate_same_prev_tier_share"] = {
        "real": aff_same_share(real_p), "synth": aff_same_share(synth_p),
        "note": "legs whose affiliate TIER equals the previous leg's "
                "(real 75.4%; the released quantity). Residual over-"
                "persistence (~10pp) is the within-trip tier ORDER the "
                "count-only release cannot carry — declared gap. Raw id "
                "level is NOT comparable: the generator invents per-"
                "segment synthetic ids for the ~3,600-id suppressed tail."}
    # month-of-window placement (the observation ramp): TV over the 15
    # absolute window months — month-of-year TV hides the censoring shape
    def month_window_mix(p):
        src = f"read_csv_auto('{p}')"
        return dict(con.execute(
            f"select strftime(checkin, '%Y-%m'), count(*) from {src} "
            f"group by 1").fetchall())

    mw_grid = sorted(set(month_window_mix(real_p))
                     | set(month_window_mix(synth_p)))
    report["month_window_tv_real_vs_synth"] = round(
        tv(month_window_mix(real_p), month_window_mix(synth_p), mw_grid), 4)
    # tempo
    report["tempo"] = {
        "real": {k: rnd(v) for k, v in _tempo(con, real_p).items()},
        "synth": {k: rnd(v) for k, v in _tempo(con, synth_p).items()}}

    # coverage: how much of the synthetic city mass sits on released ids
    n_synth_cities = con.execute(
        f"select count(distinct city_id) from read_csv_auto('{synth_p}')"
    ).fetchone()[0]
    report["n_synth_cities"] = int(n_synth_cities)

    # anomaly scores through the production profile AE
    reg = ROOT / "models_out" / "registry.json"
    if reg.exists():
        import torch
        import torch.nn as nn
        active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
        if active:
            vdir = ROOT / "models_out" / "minimodels" / active
            meta = json.loads((vdir / "meta.json").read_text())

            class UserProfileAE(nn.Module):
                def __init__(self, n_in=16):
                    super().__init__()
                    self.encoder = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(),
                                                 nn.Linear(8, 2))
                    self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(),
                                                 nn.Linear(8, n_in))

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            n_in = len(meta["profile_ae"]["metrics"]["scaler_mean"])
            ae = UserProfileAE(n_in=n_in)
            ae.load_state_dict(torch.load(vdir / "profile_ae.pt",
                                          map_location="cpu"))
            ae.eval()
            real_s = ae_scores(_tier_hists(con, real_p), meta, ae)
            synth_s = ae_scores(_tier_hists(con, synth_p), meta, ae)
            p99 = float(np.quantile(real_s, 0.99))
            report["anomaly"] = {
                "real_p99_recon": round(p99, 4),
                "synth_share_above_real_p99": round(
                    float((synth_s > p99).mean()), 4),
                "model_version": active,
            }
            # the classifier minimodels stay enclave-side (ml_trip_scores
            # scores the enclave data); record their gates, not new scores
            report["minimodel_gates_recorded"] = sorted(
                k for k in meta if k not in ("name", "version", "created_utc",
                                             "dependencies", "release_status",
                                             "data_fingerprint", "hparams",
                                             "code_fingerprint", "split",
                                             "device", "numeric_note"))
    out = ROOT / "artifacts" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "realism_audit.json").write_text(json.dumps(report, indent=2))
    print(f"realism-audit: {json.dumps(report, indent=2)[:400]} ...")
    print("-> artifacts/reports/realism_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

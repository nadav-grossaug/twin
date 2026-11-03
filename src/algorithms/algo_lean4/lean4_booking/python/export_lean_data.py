#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes (under lean/data/ by default):
  dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT
      leave the enclave). Main histogram: reservations-per-user bins (one
      cell per user, clip 1). Aux histograms: hotel-country tiers (16),
      country-tier transitions (16x16), stay nights (5), inter-leg gaps
      (4), checkin months (12), checkin weekdays (7), device classes (3),
      affiliate tiers (9), booker countries (5, one cell per user), and
      return-rate bands (5, one cell per user) — proportional L1 clip 8
      except the one-cell-per-user grids.
  motifs.json    — return/corridor/tour/longstay observations plus the
      public thresholds, re-checked by `lake exe checks` against the
      formal motif definitions.
  assoc.json     — contingency cells for the top association pairs (Lean
      recomputes chi-square exactly over ℚ).
  trips.json     — longest trips (per-leg checkin days, nights, cities,
      countries) for the IsWellFormedTrip / IsContiguousTrip checks.

Exact-fraction ledger (enforced again in Lean and in release.py; a new
query must displace an old one):
  7*(4/100) + 16*(3/100) + 256*(3/100) + 5*(6/100) + 4*(6/100)
  + 12*(1/100) + 7*(1/100) + 3*(5/100) + 9*(4/100) + 5*(4/100) + 5*(4/100)
  = 28 + 48 + 768 + 30 + 24 + 12 + 7 + 15 + 36 + 20 + 20 = 1008/100 <= 14/1
(NOT saturated: head-room for one small future query; the 14/1 cap is the
user's explicit fidelity-over-privacy choice — the bound is weak, eps ~10.)

Per goal.md guideline 10: Lean proves properties of the mechanism and the
motif predicates; it does NOT prove these inputs were correctly computed by
SQL.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import (AFF_CELLS, BOOKER_CELLS, CT_CELLS, DEVICE_CELLS,  # noqa: E402
                  GAP_CELLS, NIGHTS_CELLS, RC_BINS, RR_BANDS,
                  TRANS_CELLS, load_spec, spec_version)

ROOT = Path(__file__).resolve().parent.parent

# public DP parameters: per-cell epsilon = EPS1/EPS2 (SampCert takes positive
# naturals); the country-tier transition grid gets the lion's share of the
# budget — it is the distilled replacement for the NextCountry teacher, and
# with 256 cells even a small per-cell epsilon dominates the ledger.
# Ledger v1.2.0 (epsilon REALLOCATION, fidelity over privacy):
#   7*(4/100) rc + 16*(3/100) country + 256*(4/100) transitions
# + 5*(6/100) nights + 4*(6/100) gap + 15*(2/100) month_window
# + 3*(5/100) device + 9*(4/100) affiliate + 5*(4/100) booker
# + 5*(6/100) return_rate + 6*(5/100) repeat_struct = 1315/100
# Ledger v1.3.0 (second fidelity pass — new structure releases):
# + 3*(6/100) trips_per_user + 3*(5/100) device_persist
# + 4*(5/100) affiliate_persist
# = 1315 + 18 + 15 + 20 = 1368/100 = 13.68 <= 14/1 (the dow release is
# DROPPED: measured uniform, never enforced — the uniform public prior
# covers it).
EPS1 = int(os.environ.get("DP_EPS1", "4"))
EPS2 = int(os.environ.get("DP_EPS2", "100"))
BUDGET = float(os.environ.get("DP_BUDGET", "14.0"))
RC_EPS1 = int(os.environ.get("DP_RC_EPS1", "4"))
COUNTRY_EPS1 = int(os.environ.get("DP_COUNTRY_EPS1", "3"))
TRANS_EPS1 = int(os.environ.get("DP_TRANS_EPS1", "4"))
NIGHTS_EPS1 = int(os.environ.get("DP_NIGHTS_EPS1", "6"))
GAP_EPS1 = int(os.environ.get("DP_GAP_EPS1", "6"))
MONTH_WINDOW_EPS1 = int(os.environ.get("DP_MONTH_WINDOW_EPS1", "2"))
DEVICE_EPS1 = int(os.environ.get("DP_DEVICE_EPS1", "5"))
AFF_EPS1 = int(os.environ.get("DP_AFF_EPS1", "4"))
BOOKER_EPS1 = int(os.environ.get("DP_BOOKER_EPS1", "4"))
RR_EPS1 = int(os.environ.get("DP_RR_EPS1", "6"))
REPEAT_STRUCT_EPS1 = int(os.environ.get("DP_REPEAT_STRUCT_EPS1", "5"))
TRIPS_EPS1 = int(os.environ.get("DP_TRIPS_EPS1", "6"))
DEVICE_PERSIST_EPS1 = int(os.environ.get("DP_DEVICE_PERSIST_EPS1", "5"))
AFF_PERSIST_EPS1 = int(os.environ.get("DP_AFF_PERSIST_EPS1", "5"))

# public cell grids, in the fixed order the Lean release iterates
MONTH_WINDOW_CELLS = [f"mw{i:02d}" for i in range(15)]
REPEAT_STRUCT_CELLS = ["r_ext1", "r_ext2", "r_ext3p",
                       "r_ret1", "r_ret2p", "r_both"]
TRIPS_CELLS = ["tp_1", "tp_2", "tp_3p"]
DEVICE_PERSIST_CELLS = ["dv_1", "dv_2", "dv_3"]
AFF_PERSIST_CELLS = ["ap_1", "ap_2", "ap_3", "ap_4p"]

_SPEC = load_spec()


def _params() -> dict:
    return {
        "corridor_min_repeats": int(_SPEC["corridor_min_repeats"]),
        "corridor_cv_max_num": int(_SPEC["corridor_cv_max_num"]),
        "corridor_cv_max_den": int(_SPEC["corridor_cv_max_den"]),
        "tour_min_legs": int(_SPEC["tour_min_legs"]),
        "tour_max_gap_d": int(_SPEC["tour_max_gap_d"]),
        "tour_min_countries": int(_SPEC["tour_min_countries"]),
        "longstay_min_nights": int(_SPEC["longstay_min_nights"]),
        "seqbreak_gap_d": int(_SPEC["seqbreak_gap_d"]),
    }


def _hist(con, relation: str, grid: list[str]) -> list[tuple]:
    """Fetch a DP input mart and order it by its public grid.

    Every mart must emit EXACTLY the public grid (zero cells included) —
    the Lean release iterates the full grid in this fixed order, so a
    missing or extra cell is a pipeline bug, not data.
    """
    try:
        rows = con.execute(
            f"select cell, clipped_count, n_contributing_users, clip "
            f"from {relation}"
        ).fetchall()
    except duckdb.CatalogException as e:
        raise SystemExit(
            f"ERROR: relation {relation!r} is missing from the enclave "
            f"warehouse — artifacts/enclave.duckdb predates the current dbt "
            f"models. Run `make dbt` (or `make train`) first.\n  [{e}]") from e
    by_cell = {r[0]: r for r in rows}
    if set(by_cell) != set(grid):
        missing = sorted(set(grid) - set(by_cell))
        extra = sorted(set(by_cell) - set(grid))
        raise ValueError(
            f"{relation} does not match its public grid "
            f"(missing={missing}, extra={extra})")
    return [by_cell[c] for c in grid]


def _cells(rows: list[tuple]) -> list[dict]:
    return [{"cell": c, "clipped_count": int(n), "n_users": int(u)}
            for c, n, u, _ in rows]


def export(db: Path, out_dir: Path) -> None:
    """Export every Lean input from the DuckDB at `db` into `out_dir`."""
    con = duckdb.connect(str(db), read_only=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DP inputs ---------------------------------------------------------
    main = _hist(con, "mart_dp_res_count", RC_BINS)
    dp = {
        "privacy_unit": "one user (add/remove-one Neighbour; their whole "
                        "reservation history is one record); aux contribution = "
                        "proportional L1 allocation over dp_clip_total",
        "grid": "public reservations-per-user bins",
        "spec_version": spec_version(),
        "clip": int(main[0][3]),
        "eps1": EPS1,
        "eps2": EPS2,
        "budget_total_eps": BUDGET,
        "budget_num": 14,
        "budget_den": 1,
        "cells": _cells(main),
        "aux": [
            {"name": name, "clip": int(rows[0][3]), "eps1": e1, "eps2": e2,
             "cells": _cells(rows)}
            for name, e1, e2, rows in (
                ("country", COUNTRY_EPS1, EPS2, _hist(con, "mart_dp_country", CT_CELLS)),
                ("country_transition", TRANS_EPS1, EPS2, _hist(con, "mart_dp_country_trans", TRANS_CELLS)),
                ("nights", NIGHTS_EPS1, EPS2, _hist(con, "mart_dp_nights", NIGHTS_CELLS)),
                ("gap", GAP_EPS1, EPS2, _hist(con, "mart_dp_gap", GAP_CELLS)),
                ("month_window", MONTH_WINDOW_EPS1, EPS2, _hist(con, "mart_dp_month_window", MONTH_WINDOW_CELLS)),
                ("device", DEVICE_EPS1, EPS2, _hist(con, "mart_dp_device", DEVICE_CELLS)),
                ("affiliate", AFF_EPS1, EPS2, _hist(con, "mart_dp_affiliate", AFF_CELLS)),
                ("booker", BOOKER_EPS1, EPS2, _hist(con, "mart_dp_booker", BOOKER_CELLS)),
                ("return_rate", RR_EPS1, EPS2, _hist(con, "mart_dp_return_rate", RR_BANDS)),
                ("repeat_struct", REPEAT_STRUCT_EPS1, EPS2, _hist(con, "mart_dp_repeat_struct", REPEAT_STRUCT_CELLS)),
                ("trips_per_user", TRIPS_EPS1, EPS2, _hist(con, "mart_dp_trips", TRIPS_CELLS)),
                ("device_persist", DEVICE_PERSIST_EPS1, EPS2, _hist(con, "mart_dp_device_persist", DEVICE_PERSIST_CELLS)),
                ("affiliate_persist", AFF_PERSIST_EPS1, EPS2, _hist(con, "mart_dp_aff_persist", AFF_PERSIST_CELLS)),
            )
        ],
    }
    (out_dir / "dp_counts.json").write_text(json.dumps(dp, indent=2))

    # ---- motif observations ------------------------------------------------
    # returns: repeat-city trips — the cities list (in checkin order) and the
    # per-leg checkin epoch days; the mart keeps near-miss rows too
    # (repeat_city = false) so the Lean agreement check exercises both
    # verdicts. sql_is_return is the SQL-side decision.
    returns = [
        {"user": int(u), "utrip": str(trip), "cities": [str(c) for c in cs],
         "days": [int(d) for d in ds],
         "sql_is_return": bool(f)}
        for u, trip, cs, ds, f in con.execute(
            "select userid, utrip, cities, days, repeat_city "
            "from mart_motif_returns "
            "order by repeat_city desc, len(cities) desc, utrip limit 50").fetchall()
    ]
    corridors = [
        {"user": int(u), "a": a, "b": b,
         "gaps_d": [int(g) for g in gaps], "n_repeats": int(nr),
         "sql_is_corridor": bool(f)}
        for u, a, b, gaps, nr, f in con.execute(
            "select userid, prev_country, next_country, gaps_days, "
            "n_repeats, is_corridor "
            "from mart_motif_corridors order by n_repeats desc limit 25").fetchall()
    ]
    tours = [
        {"user": int(u), "utrip": str(trip),
         "days": [int(d) for d in ds], "nights": [int(n) for n in ns],
         "countries": list(cs), "sql_is_tour": bool(f)}
        for u, trip, ds, ns, cs, f in con.execute(
            "select userid, utrip, days, nights_list as nights, countries, is_tour "
            "from mart_motif_tours order by len(days) desc, utrip limit 25").fetchall()
    ]
    longstays = [
        {"user": int(u), "utrip": str(trip), "leg": int(leg),
         "nights": int(n), "sql_is_longstay": bool(f)}
        for u, trip, leg, n, f in con.execute(
            "select userid, utrip, leg, nights, is_longstay "
            "from mart_motif_longstays order by nights desc limit 25").fetchall()
    ]
    (out_dir / "motifs.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "params": _params(),
        "returns": returns,
        "corridors": corridors,
        "tours": tours,
        "longstays": longstays,
    }, indent=2))

    # ---- contingency cells for the top association pairs -------------------
    assoc_pairs = []
    for fa, fb, chi2, dof, v in con.execute(
        "select feat_a, feat_b, chi2, dof, cramers_v from mart_feature_assoc "
        "order by cramers_v desc limit 6"
    ).fetchall():
        pcells = con.execute(
            "select val_a, val_b, observed from mart_assoc_cells "
            "where feat_a=? and feat_b=? order by val_a, val_b", [fa, fb]
        ).fetchall()
        vals_a = sorted({c[0] for c in pcells})
        vals_b = sorted({c[1] for c in pcells})
        m = {(a, b): int(o) for a, b, o in pcells}
        assoc_pairs.append({
            "feat_a": fa, "feat_b": fb,
            "sql_chi2": float(chi2), "dof": int(dof), "cramers_v": float(v),
            "rows": vals_a, "cols": vals_b,
            "observed": [[m.get((a, b), 0) for b in vals_b] for a in vals_a],
        })
    (out_dir / "assoc.json").write_text(json.dumps({"pairs": assoc_pairs}, indent=2))

    # ---- longest trips for the Lean well-formedness check -------------------
    # checkin epoch days = date 1970-01-01 = day 0 (spec convention);
    # feat_reservations carries checkin_day already. The per-leg device and
    # affiliate-class lists ride along for the 1.3.0 constancy motifs
    # (IsDeviceConstantTrip / IsAffiliateConstantTrip re-decided in Lean).
    trips = [
        {"user": int(u), "utrip": str(trip),
         "days": [int(d) for d in ds], "nights": [int(n) for n in ns],
         "cities": [str(c) for c in cs], "countries": list(cos),
         "devices": list(dvs), "affs": list(afs),
         "sql_all_contiguous": bool(f),
         "sql_is_devconst": bool(fd), "sql_is_affconst": bool(fa)}
        for u, trip, ds, ns, cs, cos, dvs, afs, f, fd, fa in con.execute(
            "select t.userid, t.utrip, "
            "list(s.checkin_day order by s.leg_idx), "
            "list(s.nights order by s.leg_idx), "
            "list(s.city_id order by s.leg_idx), "
            "list(s.hotel_country order by s.leg_idx), "
            "list(s.device_class order by s.leg_idx), "
            "list(s.aff_tier order by s.leg_idx), "
            "t.all_contiguous, t.is_devconst, t.is_affconst "
            "from (select userid, utrip, n_legs, all_contiguous, "
            "  is_devconst, is_affconst, "
            "  row_number() over (order by n_legs desc, utrip) rn "
            "  from feat_trips) t "
            "join feat_reservations s on s.utrip = t.utrip "
            "where t.rn <= 12 "
            "group by t.userid, t.utrip, t.all_contiguous, t.is_devconst, "
            "  t.is_affconst, t.rn "
            "order by t.rn").fetchall()
    ]
    (out_dir / "trips.json").write_text(json.dumps({
        "seqbreak_gap_d": int(_SPEC["seqbreak_gap_d"]),
        "trips": trips,
    }, indent=2))
    con.close()


def main() -> int:
    db = Path(os.environ.get("ENCLAVE_DB_ABS", ROOT / "artifacts" / "enclave.duckdb"))
    if not db.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    export(db, ROOT / "lean" / "data")
    print("export-lean: wrote dp_counts/motifs/assoc/trips -> lean/data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

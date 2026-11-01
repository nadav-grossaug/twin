#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes (under lean/data/ by default):
  dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT
      leave the enclave). Main histogram: travels-per-user bins (one cell
      per user, clip 1). Aux histograms: destination region (8), region
      transitions (8x8), gap classes (5), stay-duration classes (5), hop
      distance classes (4), months (12), episode lengths (5), return-rate
      bands (5) — proportional L1 clip except the one-cell-per-user grids.
  motifs.json    — return/corridor/tour/burst/longstay observations plus
      the public thresholds, re-checked by `lake exe checks` against the
      formal motif definitions.
  assoc.json     — contingency cells for the top association pairs (Lean
      recomputes chi-square exactly over ℚ).
  episodes.json  — longest journeys (transfer timestamps; IsTimelyEpisode).

Exact-fraction ledger (enforced again in Lean and in release.py):
  7*(4/100) + 8*(4/100) + 64*(3/100) + 5*(6/100) + 5*(6/100)
  + 4*(6/100) + 12*(2/100) + 5*(4/100) + 5*(4/100)
  = 28/100 + 32/100 + 192/100 + 30/100 + 30/100 + 24/100 + 24/100
    + 20/100 + 20/100 = 400/100 <= 4/1
(exactly saturated: a new query must displace an old one at review time)

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
from spec import REGIONS, load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# public DP parameters: per-cell epsilon = EPS1/EPS2 (SampCert takes positive
# naturals); the region-transition grid gets the lion's share of the budget —
# it is the distilled replacement for the NextRegionGRU teacher, and with 64
# cells even a small per-cell epsilon dominates the ledger.
EPS1 = int(os.environ.get("DP_EPS1", "4"))
EPS2 = int(os.environ.get("DP_EPS2", "100"))
BUDGET = float(os.environ.get("DP_BUDGET", "4.0"))
REGION_EPS1 = int(os.environ.get("DP_REGION_EPS1", "4"))
TRANS_EPS1 = int(os.environ.get("DP_TRANS_EPS1", "3"))
CLASS_EPS1 = int(os.environ.get("DP_CLASS_EPS1", "6"))    # gap/duration/distance
MONTH_EPS1 = int(os.environ.get("DP_MONTH_EPS1", "2"))
EPISODE_EPS1 = int(os.environ.get("DP_EPISODE_EPS1", "4"))
RR_EPS1 = int(os.environ.get("DP_RR_EPS1", "4"))

_SPEC = load_spec()

# public cell grids, in the fixed order the Lean release iterates
TC_BINS = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
TRANS_CELLS = [f"{a}__{b}" for a in REGIONS for b in REGIONS]
GAP_CLASSES = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
DUR_CLASSES = ["dur_day", "dur_week", "dur_month", "dur_quarter", "dur_long"]
DIST_CLASSES = ["dist_short", "dist_mid", "dist_long", "dist_ultra"]
MONTH_CELLS = [f"m{i:02d}" for i in range(1, 13)]
EP_BINS = ["ep_1", "ep_2", "ep_3", "ep_4_5", "ep_6p"]
RR_BANDS = ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]


def _params() -> dict:
    return {
        "corridor_min_repeats": int(_SPEC["corridor_min_repeats"]),
        "corridor_cv_max_num": int(_SPEC["corridor_cv_max_num"]),
        "corridor_cv_max_den": int(_SPEC["corridor_cv_max_den"]),
        "tour_min_hops": int(_SPEC["tour_min_hops"]),
        "tour_max_gap_s": int(_SPEC["tour_max_gap_s"]),
        "tour_min_countries": int(_SPEC["tour_min_countries"]),
        "burst_min_transfers": int(_SPEC["burst_min_transfers"]),
        "burst_window_s": int(_SPEC["burst_window_s"]),
        "longstay_min_s": int(_SPEC["longstay_min_s"]),
    }


def _hist(con, relation: str, grid: list[str]) -> list[tuple]:
    """Fetch a DP input mart and order it by its public grid.

    Every mart must emit EXACTLY the public grid (zero cells included) —
    the Lean release iterates the full grid in this fixed order, so a
    missing or extra cell is a pipeline bug, not data.
    """
    rows = con.execute(
        f"select cell, clipped_count, n_contributing_users, clip "
        f"from {relation}"
    ).fetchall()
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
    main = _hist(con, "mart_dp_travel_count", TC_BINS)
    dp = {
        "privacy_unit": "one user (add/remove-one Neighbour; their whole "
                        "travel history is one record); aux contribution = "
                        "proportional L1 allocation over dp_clip_total",
        "grid": "public travels-per-user bins",
        "spec_version": spec_version(),
        "clip": int(main[0][3]),
        "eps1": EPS1,
        "eps2": EPS2,
        "budget_total_eps": BUDGET,
        "budget_num": 4,
        "budget_den": 1,
        "cells": _cells(main),
        "aux": [
            {"name": name, "clip": int(rows[0][3]), "eps1": e1, "eps2": e2,
             "cells": _cells(rows)}
            for name, e1, e2, rows in (
                ("region", REGION_EPS1, EPS2, _hist(con, "mart_dp_region", REGIONS)),
                ("region_transition", TRANS_EPS1, EPS2, _hist(con, "mart_dp_transitions", TRANS_CELLS)),
                ("gap_class", CLASS_EPS1, EPS2, _hist(con, "mart_dp_gap", GAP_CLASSES)),
                ("duration_class", CLASS_EPS1, EPS2, _hist(con, "mart_dp_duration", DUR_CLASSES)),
                ("distance_class", CLASS_EPS1, EPS2, _hist(con, "mart_dp_distance", DIST_CLASSES)),
                ("month", MONTH_EPS1, EPS2, _hist(con, "mart_dp_month", MONTH_CELLS)),
                ("episode_len", EPISODE_EPS1, EPS2, _hist(con, "mart_dp_episode", EP_BINS)),
                ("return_rate", RR_EPS1, EPS2, _hist(con, "mart_dp_return_rate", RR_BANDS)),
            )
        ],
    }
    (out_dir / "dp_counts.json").write_text(json.dumps(dp, indent=2))

    # ---- motif observations ------------------------------------------------
    # returns: the two consecutive transfers (pair1 a->b, pair2 c->d); the
    # second pair's countries come from feat_transfers at the next travel_idx.
    returns = [
        {"user": int(u), "a": a, "b": b, "c": c, "d": d,
         "ts1_s": int(t1), "ts2_s": int(t2), "sql_is_return": bool(f)}
        for u, a, b, c, d, t1, t2, f in con.execute(
            "select r.userid, r.a, r.b, t.prev_country, t.next_country, "
            "r.ts1_s, r.ts2_s, r.is_return "
            "from mart_motif_returns r "
            "join feat_transfers t on t.userid = r.userid "
            "and t.travel_idx = r.idx + 1 "
            "order by r.ts2_s desc limit 50").fetchall()
    ]
    corridors = [
        {"user": int(u), "a": a, "b": b, "gaps_s": [int(g) for g in gaps],
         "sql_is_corridor": bool(f)}
        for u, a, b, gaps, f in con.execute(
            "select userid, prev_country, next_country, gaps_s, is_corridor "
            "from mart_motif_corridors order by n_repeats desc limit 25").fetchall()
    ]
    tours = [
        {"user": int(u), "ts_s": [int(t) for t in ts],
         "countries": list(cs), "sql_is_tour": bool(f)}
        for u, ts, cs, f in con.execute(
            "select userid, ts_list_s, countries, is_tour "
            "from mart_motif_tours order by n_hops desc limit 25").fetchall()
    ]
    bursts = [
        {"user": int(u), "ts_s": [int(t) for t in ts], "sql_is_burst": bool(f)}
        for u, ts, f in con.execute(
            "select userid, ts_list_s, is_burst "
            "from mart_motif_bursts order by window_s asc limit 25").fetchall()
    ]
    longstay = int(_SPEC["longstay_min_s"])
    longstays = [
        {"user": int(u), "duration_s": int(d), "sql_is_longstay": bool(f)}
        for u, d, f in con.execute(
            "select userid, next_duration_s, next_duration_s >= ? "
            "from feat_transfers order by next_duration_s desc limit 25",
            [longstay]).fetchall()
    ]
    (out_dir / "motifs.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "params": _params(),
        "returns": returns,
        "corridors": corridors,
        "tours": tours,
        "bursts": bursts,
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

    # ---- longest journeys for the Lean episode check -----------------------
    episodes = [
        {"user": int(u), "n_hops": int(n), "ts_s": [int(t) for t in ts]}
        for u, n, ts in con.execute(
            "select e.userid, e.n_hops, list(t.ts_s order by t.ts_s) "
            "from feat_episodes e join feat_transfers t "
            "on t.userid = e.userid "
            "and t.ts_s between e.start_ts_s and e.end_ts_s "
            "group by e.userid, e.episode_id, e.n_hops "
            "order by e.n_hops desc limit 12").fetchall()
    ]
    (out_dir / "episodes.json").write_text(json.dumps({
        "episode_gap_max_s": int(_SPEC["episode_gap_max_s"]),
        "episodes": episodes,
    }, indent=2))
    con.close()


def main() -> int:
    db = Path(os.environ.get("ENCLAVE_DB_ABS", ROOT / "artifacts" / "enclave.duckdb"))
    if not db.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    export(db, ROOT / "lean" / "data")
    print("export-lean: wrote dp_counts/motifs/assoc/episodes -> lean/data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

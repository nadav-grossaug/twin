#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes (under lean/data/ by default):
  dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT
      leave the enclave). Main histogram: daily-motif classes. Aux
      histograms: role transitions (3x3), dwell classes (5), departure bins
      (4), mode classes (6) — all proportional L1 clip (clip 10).
  motifs.json    — per-user-day visit lists (the Lean daily-motif classifier
      re-decides every day's class), revisit-beacon day lists, and class
      ladders (dwell / hour / displacement / speed / staypoints-per-day),
      re-checked by `lake exe checks` against the formal definitions.
  assoc.json     — contingency cells for the spec'd association pairs (Lean
      recomputes chi-square exactly over ℚ).
  seqs.json      — maximal sequence-ok visit runs (IsVisitSeq re-check).

Exact-fraction ledger (enforced again in Lean and in release.py):
  5*(30/100) + 9*(15/100) + 5*(13/100) + 4*(5/100) + 6*(5/100)
  = 150/100 + 135/100 + 65/100 + 20/100 + 30/100 = 400/100 <= 4/1
(exactly saturated: a new query must displace an old one at review time)

Per goal.md guideline 10: Lean proves properties of the mechanism and the
motif predicates; it does NOT prove these inputs were correctly computed by
SQL/Python.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"
OUT = ROOT / "lean" / "data"

MOTIF_CELLS = ["home_day", "commute", "errand_loop", "tour", "irregular"]
TRANS_CELLS = [f"{a}__{b}" for a in ("home", "work", "other")
               for b in ("home", "work", "other")]
DWELL_CELLS = ["dw_lt1h", "dw_1_3h", "dw_3_8h", "dw_8_14h", "dw_ge14h"]
DEP_CELLS = ["dep_night", "dep_morning", "dep_afternoon", "dep_evening"]
MODE_CELLS = ["walk", "bike", "bus", "car", "rail", "other"]

# (eps1, eps2) per cell for each histogram — mirrors config/release_policy.yaml
LEDGER = {
    "dp_motif": (30, 100),
    "dp_transitions": (15, 100),
    "dp_dwell": (13, 100),
    "dp_departure": (5, 100),
    "dp_mode": (5, 100),
}


def _hist(con, relation: str, grid: list[str]) -> list[dict]:
    rows = dict(con.execute(
        f"select cell, clipped_count from {relation}").fetchall())
    missing = [c for c in grid if c not in rows]
    if missing:
        raise SystemExit(f"export: {relation} is missing cells {missing}")
    return [{"cell": c, "clipped_count": int(rows[c])} for c in grid]


def export(db: Path, out_dir: Path) -> None:
    spec = load_spec()
    con = duckdb.connect(str(db), read_only=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dp_counts.json ----------------------------------------------------
    clip_days = int(spec["dp_clip_days"])
    clip_visits = int(spec["dp_clip_visits"])
    clip_legs = int(spec["dp_clip_legs"])
    dp = {
        "spec_version": spec_version(),
        "privacy_unit": "user (add/remove-one; the whole multi-year GPS "
                        "trace of one user is one record)",
        "clip": clip_days,
        "eps1": LEDGER["dp_motif"][0],
        "eps2": LEDGER["dp_motif"][1],
        "budget_num": 4,
        "budget_den": 1,
        "cells": _hist(con, "mart_dp_motif", MOTIF_CELLS),
        "aux": [
            {"name": "transitions", "clip": clip_visits,
             "eps1": LEDGER["dp_transitions"][0],
             "eps2": LEDGER["dp_transitions"][1],
             "cells": _hist(con, "mart_dp_transitions", TRANS_CELLS)},
            {"name": "dwell", "clip": clip_visits,
             "eps1": LEDGER["dp_dwell"][0], "eps2": LEDGER["dp_dwell"][1],
             "cells": _hist(con, "mart_dp_dwell", DWELL_CELLS)},
            {"name": "departure", "clip": clip_visits,
             "eps1": LEDGER["dp_departure"][0],
             "eps2": LEDGER["dp_departure"][1],
             "cells": _hist(con, "mart_dp_departure", DEP_CELLS)},
            {"name": "mode", "clip": clip_legs,
             "eps1": LEDGER["dp_mode"][0], "eps2": LEDGER["dp_mode"][1],
             "cells": _hist(con, "mart_dp_mode", MODE_CELLS)},
        ],
    }
    (out_dir / "dp_counts.json").write_text(json.dumps(dp, indent=1))

    # ---- motifs.json -------------------------------------------------------
    params = {k: int(spec[k]) for k in (
        "sp_min_dwell_s", "visit_gap_break_s", "tour_max_gap_s",
        "tour_min_anchors", "errand_max_dwell_s", "revisit_min_days",
        "revisit_window_days", "longdwell_min_s",
        "dw_e1_s", "dw_e2_s", "dw_e3_s", "dw_e4_s",
        "dep_night_max_h", "dep_morning_max_h", "dep_afternoon_max_h",
        "dist_e1_m", "dist_e2_m", "dist_e3_m",
        "spd_still_max_mmps", "spd_walk_max_mmps", "spd_city_max_mmps",
        "spd_fast_max_mmps", "spdbin_e2", "spdbin_e3", "spdbin_e4",
        "spdbin_e6", "route_resample_m", "junction_min_turn_deg",
        "ja_e1_deg", "ja_e2_deg", "seg_e1_m", "seg_e2_m", "seg_e3_m",
        "circ_e1_pm", "circ_e2_pm", "circ_e3_pm", "route_min_net_m")}

    # every user-day: the visit list the Lean classifier re-decides
    visit_rows = con.execute("""
        select v.user_id, v.day_idx, v.role, v.anchor_id,
               v.ts_arrive_s, v.ts_leave_s, v.prev_role, v.prev_leave_s,
               d.motif_class
        from feat_visits v
        join feat_user_days d using (user_id, day_idx)
        order by v.user_id, v.ts_arrive_s, v.sp_id
    """).fetchall()
    days: dict = {}
    for (uid, day, role, anchor, arr, lv, prole, pleave, motif) in visit_rows:
        key = (uid, int(day))
        d = days.setdefault(key, {"user": uid, "day_idx": int(day),
                                  "sql_motif": motif, "visits": []})
        d["visits"].append({
            "role": role, "anchor": int(anchor),
            "arrive_s": int(arr), "leave_s": int(lv),
            "prev_role": prole,
            "prev_leave_s": int(pleave) if pleave is not None else None})
    day_list = [days[k] for k in sorted(days)]

    revisit_days = con.execute("""
        select v.user_id, v.anchor_id, list(distinct v.day_idx order by v.day_idx),
               any_value(m.is_revisit)
        from feat_visits v
        join mart_motif_revisit m using (user_id, anchor_id)
        where v.role != 'home'
        group by v.user_id, v.anchor_id
        order by v.user_id, v.anchor_id
    """).fetchall()
    revisits = [{"user": u, "anchor": int(a),
                 "days": [int(d) for d in ds], "sql_is_revisit": bool(r)}
                for (u, a, ds, r) in revisit_days]

    dwells = [{"dwell_s": int(d), "sql_class": c} for (d, c) in con.execute(
        "select dwell_s, dwell_class from feat_visits "
        "where not arr_censored and not dep_censored "
        "order by user_id, sp_id limit 2000").fetchall()]
    hours = [{"hour": int(h), "sql_bin": b} for (h, b) in con.execute(
        "select arrive_hour, arr_bin from feat_visits "
        "order by user_id, sp_id limit 2000").fetchall()]
    dists = [{"disp_m": int(m), "sql_class": c} for (m, c) in con.execute(
        "select floor(trip_disp_m)::bigint, dist_class from feat_visits "
        "where dist_class is not null order by user_id, sp_id limit 2000"
    ).fetchall()]
    speeds = [{"mmps": int(s), "sql_class": c} for (s, c) in con.execute(
        "select floor(speed_mps * 1000)::bigint, speed_class from feat_steps "
        "order by obs_id limit 2000").fetchall()]
    spds = [{"n": int(n), "sql_bin": b} for (n, b) in con.execute(
        "select n_staypoints, spd_bin from feat_user_days "
        "order by user_id, day_idx limit 2000").fetchall()]
    longdwells = [{"dwell_s": int(d), "sql_is_longdwell": bool(f)}
                  for (d, f) in con.execute(
        "select dwell_s, is_longdwell from feat_visits "
        "order by user_id, sp_id limit 2000").fetchall()]
    segs = [{"seg_m": int(m), "sql_class": c} for (m, c) in con.execute(
        "select floor(seg_m)::bigint, seg_class from feat_route_segments "
        "order by user_id, trajectory_id, sequence_uuid, junction_idx "
        "limit 2000").fetchall()]
    jas = [{"abs_deg": int(a), "sql_class": c} for (a, c) in con.execute(
        "select floor(junction_abs_deg)::bigint, ja_class "
        "from feat_route_segments "
        "order by user_id, trajectory_id, sequence_uuid, junction_idx "
        "limit 2000").fetchall()]
    circs = [{"circ_pm": int(p), "sql_class": c} for (p, c) in con.execute(
        "select circ_pm, circ_class from feat_route_legs "
        "where circ_class is not null order by leg_id limit 2000").fetchall()]

    (out_dir / "motifs.json").write_text(json.dumps({
        "params": params,
        "days": day_list,
        "revisits": revisits,
        "dwells": dwells,
        "hours": hours,
        "dists": dists,
        "speeds": speeds,
        "spd_bins": spds,
        "segs": segs,
        "jas": jas,
        "circs": circs,
        "longdwells": longdwells,
    }, indent=1))

    # ---- assoc.json --------------------------------------------------------
    pairs = []
    pair_keys = con.execute(
        "select distinct feat_a, feat_b from mart_assoc_cells "
        "order by feat_a, feat_b").fetchall()
    for fa, fb in pair_keys:
        vals_a = [r[0] for r in con.execute(
            "select distinct val_a from mart_assoc_cells "
            "where feat_a = ? and feat_b = ? order by val_a", [fa, fb]).fetchall()]
        vals_b = [r[0] for r in con.execute(
            "select distinct val_b from mart_assoc_cells "
            "where feat_a = ? and feat_b = ? order by val_b", [fa, fb]).fetchall()]
        cells = {(a, b): int(o) for (a, b, o) in con.execute(
            "select val_a, val_b, observed from mart_assoc_cells "
            "where feat_a = ? and feat_b = ?", [fa, fb]).fetchall()}
        table = [[cells.get((a, b), 0) for b in vals_b] for a in vals_a]
        chi2 = con.execute(
            "select chi2 from mart_feature_assoc "
            "where feat_a = ? and feat_b = ?", [fa, fb]).fetchone()
        pairs.append({"feat_a": fa, "feat_b": fb,
                      "vals_a": vals_a, "vals_b": vals_b,
                      "observed": table,
                      "sql_chi2": float(chi2[0]) if chi2 else 0.0})
    (out_dir / "assoc.json").write_text(json.dumps({"pairs": pairs}, indent=1))

    # ---- seqs.json ---------------------------------------------------------
    # maximal sequence-ok runs: consecutive visits chained while seq_ok holds
    runs = con.execute("""
        with marked as (
            select user_id, sp_id, ts_arrive_s, ts_leave_s,
                   sum(case when seq_ok then 0 else 1 end)
                     over (partition by user_id
                           order by ts_arrive_s, sp_id) as run_id
            from feat_visits
        )
        select user_id, run_id,
               list(ts_arrive_s order by ts_arrive_s) as arrives,
               list(ts_leave_s order by ts_arrive_s) as leaves
        from marked
        group by user_id, run_id
        having count(*) >= 2
        order by user_id, run_id
        limit 500
    """).fetchall()
    seqs = [{"user": u, "intervals": [[int(a), int(l)]
                                      for a, l in zip(arrs, lvs)]}
            for (u, _, arrs, lvs) in runs]
    (out_dir / "seqs.json").write_text(json.dumps(
        {"visit_gap_break_s": int(spec["visit_gap_break_s"]),
         "runs": seqs}, indent=1))

    con.close()
    print(f"export: wrote dp_counts.json ({len(dp['cells'])} + "
          f"{sum(len(a['cells']) for a in dp['aux'])} cells), motifs.json "
          f"({len(day_list)} days, {len(revisits)} revisit anchors), "
          f"assoc.json ({len(pairs)} pairs), seqs.json ({len(seqs)} runs) "
          f"-> {out_dir}")


def main() -> int:
    if not DB.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    out = Path(os.environ.get("LEAN_DATA_DIR", str(OUT)))
    export(DB, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

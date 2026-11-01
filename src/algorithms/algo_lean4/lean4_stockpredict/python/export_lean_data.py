#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes (under lean/data/ by default):
  dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT
      leave the enclave). Main histogram: the 5-class return mix
      (proportional L1 clip). Aux histograms: class transitions (5x5),
      dow seasonality (5x5), vol-state transitions (4x4, the discretized
      GARCH diffusion), |log return| magnitude bins (6), co-movement bins (3).
  motifs.json    — run/shock/drawdown/V-recovery/vol-burst observations plus
      the public thresholds, re-checked by `lake exe checks` against the
      formal motif definitions.
  assoc.json     — contingency cells for the top association pairs (Lean
      recomputes chi-square exactly over ℚ).
  sessions.json  — contiguous trading runs (day numbers; IsTimelySession).

Exact-fraction ledger (enforced again in Lean and in release.py):
  5*(27/50) + 25*(1/100) + 25*(1/100) + 16*(1/100) + 6*(1/150) + 3*(1/5)
  = 270/100 + 25/100 + 25/100 + 16/100 + 4/100 + 60/100 = 400/100 <= 4/1
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
from spec import load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# public DP parameters: per-cell epsilon = EPS1/EPS2 (SampCert takes positive
# naturals); the 5-class return mix gets the lion's share of the budget —
# with only 8 protected tickers the SNR is thin, so the grid is coarse and
# the budget concentrated (goal.md: the n=8 calibration gate).
EPS1 = int(os.environ.get("DP_EPS1", "27"))
EPS2 = int(os.environ.get("DP_EPS2", "50"))
BUDGET = float(os.environ.get("DP_BUDGET", "4.0"))
AUX_EPS1 = int(os.environ.get("DP_AUX_EPS1", "1"))
AUX_EPS2 = int(os.environ.get("DP_AUX_EPS2", "100"))
COMOVE_EPS2 = int(os.environ.get("DP_COMOVE_EPS2", "5"))
MAG_EPS2 = int(os.environ.get("DP_MAG_EPS2", "150"))

_SPEC = load_spec()


def _params() -> dict:
    return {
        "ret_flat_bp": int(_SPEC["ret_flat_bp"]),
        "ret_big_bp": int(_SPEC["ret_big_bp"]),
        "shock_bp": int(_SPEC["shock_bp"]),
        "run_min_days": int(_SPEC["run_min_days"]),
        "volburst_min_bp": int(_SPEC["volburst_min_bp"]),
        "drawdown_num": int(_SPEC["drawdown_num"]),
        "drawdown_den": int(_SPEC["drawdown_den"]),
        "vrecov_drop_num": int(_SPEC["vrecov_drop_num"]),
        "vrecov_drop_den": int(_SPEC["vrecov_drop_den"]),
        "vrecov_level_num": int(_SPEC["vrecov_level_num"]),
        "vrecov_level_den": int(_SPEC["vrecov_level_den"]),
        "vrecov_max_days": int(_SPEC["vrecov_max_days"]),
    }


def _hist(con, relation: str) -> list[tuple]:
    return con.execute(
        f"select cell, clipped_count, n_contributing_users, clip "
        f"from {relation} order by cell"
    ).fetchall()


def _cells(rows: list[tuple]) -> list[dict]:
    return [{"cell": c, "clipped_count": int(n), "n_users": int(u)}
            for c, n, u, _ in rows]


def micro(x: float) -> int:
    return int(round(float(x) * 1_000_000))


def export(db: Path, out_dir: Path) -> None:
    """Export every Lean input from the DuckDB at `db` into `out_dir`."""
    con = duckdb.connect(str(db), read_only=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DP inputs ---------------------------------------------------------
    main = _hist(con, "mart_dp_counts")
    dp = {
        "privacy_unit": "one ticker (add/remove-one Neighbour; its whole "
                        "series is one record); class-mix contribution = "
                        "proportional L1 allocation over dp_clip_total",
        "grid": "public 5-class log-return taxonomy",
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
                ("class_transition", AUX_EPS1, AUX_EPS2, _hist(con, "mart_dp_transitions")),
                ("dow_class", AUX_EPS1, AUX_EPS2, _hist(con, "mart_dp_dow")),
                ("vol_transition", AUX_EPS1, AUX_EPS2, _hist(con, "mart_dp_voltrans")),
                ("mag_bins", AUX_EPS1, MAG_EPS2, _hist(con, "mart_dp_mag")),
                ("comove", AUX_EPS1, COMOVE_EPS2, _hist(con, "mart_dp_comove")),
            )
        ],
    }
    (out_dir / "dp_counts.json").write_text(json.dumps(dp, indent=2))

    # ---- motif observations ------------------------------------------------
    runs = [
        {"ticker": t, "rets_bp": [int(r) for r in rets], "sql_is_run": bool(f)}
        for t, rets, f in con.execute(
            "select ticker, rets_bp, is_run from mart_motif_runs "
            "order by len_days desc limit 25").fetchall()
    ]
    shocks = [
        {"ticker": t, "d": str(d), "ret_bp": int(r), "sql_is_shock": bool(f)}
        for t, d, r, f in con.execute(
            "select ticker, d, ret_bp, is_shock from mart_motif_shock "
            "order by abs(ret_bp) desc limit 40").fetchall()
    ]
    drawdowns = [
        {"ticker": t, "prices_micro": [micro(p) for p in ps],
         "sql_has_drawdown": bool(f)}
        for t, ps, f in con.execute(
            "select ticker, prices, has_drawdown from mart_motif_drawdown "
            "order by ticker").fetchall()
    ]
    w = int(_SPEC["vrecov_max_days"])
    vrecs = []
    for t, idx, peak, f in con.execute(
        "select ticker, trough_idx, peak, is_vrecovery from mart_motif_vrecovery "
        "order by ticker, trough_d limit 15").fetchall():
        window = [micro(p) for (p,) in con.execute(
            "select price from feat_returns where ticker = ? "
            "and day_idx between ? and ? order by d",
            [t, int(idx), int(idx) + w]).fetchall()]
        vrecs.append({"ticker": t, "peak_micro": micro(peak),
                      "window_micro": window, "sql_is_vrecovery": bool(f)})
    volbursts = [
        {"ticker": t, "d": str(d), "window_rets_bp": [int(r) for r in rets],
         "sql_is_volburst": bool(f)}
        for t, d, rets, f in con.execute(
            "select ticker, d, window_rets_bp, is_volburst from mart_motif_volburst "
            "order by vol5_bp desc limit 25").fetchall()
    ]
    (out_dir / "motifs.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "params": _params(),
        "runs": runs,
        "shocks": shocks,
        "drawdowns": drawdowns,
        "vrecoveries": vrecs,
        "volbursts": volbursts,
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

    # ---- contiguous trading runs for the Lean session check ----------------
    sessions = [
        {"ticker": t, "n_days": int(n), "day_nums": [int(x) for x in nums]}
        for t, n, nums in con.execute(
            "select ticker, n_days, day_nums from mart_seq_sessions "
            "order by n_days desc limit 12").fetchall()
    ]
    (out_dir / "sessions.json").write_text(json.dumps({
        "max_gap_days": int(_SPEC["gap_max_days"]),
        "sessions": sessions,
    }, indent=2))
    con.close()


def main() -> int:
    db = Path(os.environ.get("ENCLAVE_DB_ABS", ROOT / "artifacts" / "enclave.duckdb"))
    if not db.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    export(db, ROOT / "lean" / "data")
    print(f"export-lean: wrote dp_counts/motifs/assoc/sessions -> lean/data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

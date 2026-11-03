#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes (under lean/data/ by default; GRAPH_LEAN_DATA overrides):
  dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT
      leave the enclave). Main histogram: degree bins (one cell per node,
      clip 1). Aux histograms: clustering (7), triangles (6), richness (9),
      category presence (11), circles (5), community sizes (6), mixing (45),
      wedge closure (16), motif presence (6) — proportional L1 clip except
      the one-cell-per-node grids.
  nodes.json     — ALL per-node observations (degree/tri/communities/circles
      + the SQL motif flags and classes), re-checked by `lake exe checks`
      against the formal motif definitions and class maps.
  assoc.json     — contingency cells for the top association grids (Lean
      recomputes chi-square exactly over ℚ).

Exact-fraction ledger (enforced again in Lean and in release.py):
  8*(5/100) + 7*(5/100) + 6*(5/100) + 9*(4/100) + 11*(4/100)
  + 5*(4/100) + 6*(4/100) + 45*(2/100) + 16*(3/100) + 6*(3/100)
  = 40 + 35 + 30 + 36 + 44 + 20 + 24 + 90 + 48 + 18 = 385/100 <= 4/1
(nearly saturated: a new query must displace an old one at review time)

Per goal.md guideline 10: Lean proves properties of the mechanism and the
motif predicates; it does NOT prove these inputs were correctly computed by
SQL. The privacy unit is ONE DERIVED NODE ROW (the node-DP gap is declared
in goal.md).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import CATEGORIES, load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# public DP parameters: per-cell epsilon = EPS1/EPS2 (SampCert takes positive
# naturals); the mixing grid gets the lion's share of the budget — it is the
# distilled replacement for the CommunityGNN teacher, and with 45 cells even
# a small per-cell epsilon dominates the ledger.
EPS2 = int(os.environ.get("DP_EPS2", "100"))
DEGREE_EPS1 = int(os.environ.get("DP_DEGREE_EPS1", "5"))
CC_EPS1 = int(os.environ.get("DP_CC_EPS1", "5"))
TRI_EPS1 = int(os.environ.get("DP_TRI_EPS1", "5"))
RICH_EPS1 = int(os.environ.get("DP_RICH_EPS1", "4"))
CAT_EPS1 = int(os.environ.get("DP_CAT_EPS1", "4"))
CIRC_EPS1 = int(os.environ.get("DP_CIRC_EPS1", "4"))
CSIZE_EPS1 = int(os.environ.get("DP_CSIZE_EPS1", "4"))
MIX_EPS1 = int(os.environ.get("DP_MIX_EPS1", "2"))
CLO_EPS1 = int(os.environ.get("DP_CLO_EPS1", "3"))
MOTIF_EPS1 = int(os.environ.get("DP_MOTIF_EPS1", "3"))
BUDGET = float(os.environ.get("DP_BUDGET", "4.0"))

_SPEC = load_spec()

# public cell grids, in the fixed order the Lean release iterates
DEG_BINS = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
            "deg_33_64", "deg_65p"]
CC_CLASSES = ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80",
              "cc_80_99", "cc_one"]
TRI_CLASSES = ["tri_0", "tri_1_10", "tri_11_100", "tri_101_1000",
               "tri_1001_10000", "tri_10001p"]
RICH_CLASSES = [f"rich_{i}" for i in range(8)] + ["rich_8p"]
CAT_CELLS = [f"cat_{c}" for c in CATEGORIES]
CIRC_CLASSES = ["circ_0", "circ_1", "circ_2", "circ_3", "circ_4p"]
CSIZE_CLASSES = ["csize_lt25", "csize_25_50", "csize_51_100", "csize_101_200",
                 "csize_201_400", "csize_gt400"]
RANKS = [f"c{i}" for i in range(1, 9)] + ["rest"]
MIX_CELLS = [f"{RANKS[i]}__{RANKS[j]}"
             for i in range(len(RANKS)) for j in range(i, len(RANKS))]
CLOSURE_CELLS = [c for d in DEG_BINS
                 for c in (f"clo_{d}__closed", f"clo_{d}__open")]
MOTIF_CELLS = ["motif_hub", "motif_leaf", "motif_broker", "motif_bridge",
               "motif_butterfly", "motif_core"]


def _hist(con, relation: str, grid: list[str]) -> list[tuple]:
    """Fetch a DP input mart and order it by its public grid.

    Every mart must emit EXACTLY the public grid (zero cells included) —
    the Lean release iterates the full grid in this fixed order, so a
    missing or extra cell is a pipeline bug, not data.
    """
    rows = con.execute(
        f"select cell, clipped_count, n_contributing_nodes, clip "
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
    return [{"cell": c, "clipped_count": int(n), "n_nodes": int(u)}
            for c, n, u, _ in rows]


def export(db: Path, out_dir: Path) -> None:
    """Export every Lean input from the DuckDB at `db` into `out_dir`."""
    con = duckdb.connect(str(db), read_only=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DP inputs ---------------------------------------------------------
    main = _hist(con, "mart_dp_degree", DEG_BINS)
    dp = {
        "privacy_unit": "node_row",
        "privacy_note": "add/remove ONE derived node row; edge/wedge/category "
                        "contributions bounded by proportional L1 allocation "
                        "over the clip; the neighbor-induced node-DP gap is "
                        "declared in goal.md (checks claim "
                        "row_level_privacy_unit)",
        "grid": "public degree bins",
        "spec_version": spec_version(),
        "clip": int(main[0][3]),
        "eps1": DEGREE_EPS1,
        "eps2": EPS2,
        "budget_total_eps": BUDGET,
        "budget_num": 4,
        "budget_den": 1,
        "cells": _cells(main),
        "aux": [
            {"name": name, "clip": int(rows[0][3]), "eps1": e1, "eps2": e2,
             "cells": _cells(rows)}
            for name, e1, e2, rows in (
                ("clustering", CC_EPS1, EPS2, _hist(con, "mart_dp_clustering", CC_CLASSES)),
                ("triangles", TRI_EPS1, EPS2, _hist(con, "mart_dp_triangles", TRI_CLASSES)),
                ("richness", RICH_EPS1, EPS2, _hist(con, "mart_dp_richness", RICH_CLASSES)),
                ("category", CAT_EPS1, EPS2, _hist(con, "mart_dp_categories", CAT_CELLS)),
                ("circles", CIRC_EPS1, EPS2, _hist(con, "mart_dp_circles", CIRC_CLASSES)),
                ("community_sizes", CSIZE_EPS1, EPS2, _hist(con, "mart_dp_community_sizes", CSIZE_CLASSES)),
                ("mixing", MIX_EPS1, EPS2, _hist(con, "mart_dp_mixing", MIX_CELLS)),
                ("closure", CLO_EPS1, EPS2, _hist(con, "mart_dp_closure", CLOSURE_CELLS)),
                ("motifs", MOTIF_EPS1, EPS2, _hist(con, "mart_dp_motifs", MOTIF_CELLS)),
            )
        ],
    }
    (out_dir / "dp_counts.json").write_text(json.dumps(dp, indent=2))

    # ---- per-node observations for the Lean deciders -----------------------
    nodes = [
        {"node_id": int(nid), "degree": int(deg), "tri_count": int(tri),
         "neighbor_communities": int(nbc), "max_circles_one_ego": int(mco),
         "n_circles": int(nc), "richness": int(rich),
         "is_hub": bool(h), "is_leaf": bool(le), "is_broker": bool(br),
         "is_bridge": bool(bg), "is_butterfly": bool(bu), "is_core": bool(co),
         "degree_class": dc, "cc_class": cc, "tri_class": tc,
         "richness_class": rc, "circle_class": cic}
        for (nid, deg, tri, nbc, mco, nc, rich, h, le, br, bg, bu, co,
             dc, cc, tc, rc, cic) in con.execute(
            "select node_id, degree, tri_count, neighbor_communities, "
            "max_circles_one_ego, n_circles, richness, "
            "is_hub, is_leaf, is_broker, is_bridge, is_butterfly, is_core, "
            "degree_class, cc_class, tri_class, richness_class, circle_class "
            "from feat_nodes order by node_id").fetchall()
    ]
    (out_dir / "nodes.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "nodes": nodes,
    }, indent=2))

    # ---- contingency cells for the top association grids -------------------
    # fixed grids first, then the top category pairs by chi2
    fixed = [("cc_class", "degree_class"), ("degree_class", "richness_class"),
             ("community_rank", "motif")]
    top_cats = con.execute(
        "select feat_a, feat_b from mart_feature_assoc "
        "where feat_a like 'cat_%' order by chi2 desc limit 5").fetchall()
    assoc_pairs = []
    for fa, fb in fixed + [tuple(r) for r in top_cats]:
        row = con.execute(
            "select chi2, dof, cramers_v from mart_feature_assoc "
            "where feat_a=? and feat_b=?", [fa, fb]).fetchone()
        if row is None:
            continue
        chi2, dof, v = row
        pcells = con.execute(
            "select val_a, val_b, observed from mart_assoc_cells "
            "where feat_a=? and feat_b=? order by val_a, val_b", [fa, fb]
        ).fetchall()
        vals_a = sorted({c[0] for c in pcells})
        vals_b = sorted({c[1] for c in pcells})
        m = {(a, b): int(o) for a, b, o in pcells}
        assoc_pairs.append({
            "grid": f"{fa}__x__{fb}",
            "feat_a": fa, "feat_b": fb,
            "sql_chi2": float(chi2), "dof": int(dof), "cramers_v": float(v),
            "rows": vals_a, "cols": vals_b,
            "observed": [[m.get((a, b), 0) for b in vals_b] for a in vals_a],
        })
    (out_dir / "assoc.json").write_text(json.dumps({"pairs": assoc_pairs}, indent=2))
    con.close()


def main() -> int:
    db = Path(os.environ.get("GRAPH_DB",
                             ROOT / "artifacts" / "enclave.duckdb"))
    if not db.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    out_dir = Path(os.environ.get("GRAPH_LEAN_DATA", ROOT / "lean" / "data"))
    export(db, out_dir)
    print(f"export-lean: wrote dp_counts/nodes/assoc -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (class mix + the aux
class_transition / dow_class / vol_transition / mag_bins / comove
histograms). No raw data, weights or embeddings are read for the spec, so the
genspec inherits the release's privacy accounting by post-processing.

Outputs:
  lean/out/genspec.json                   the restricted generative spec
                                          consumed by `lake exe generate`
  artifacts/reports/distill_report.json   ENCLAVE-ONLY fidelity report:
      the distilled semi-Markov student vs the NextClassGRU teacher vs the
      repeat-last baseline on the same held-out transitions; per-row TV
      between DP-released and exact transition matrices; GARCH persistence
      (released vol-state diagonal vs the enclave's exact one).

Distillation table (teacher -> executable statistical replacement):
  NextClassGRU     -> probabilistic state machine (DP class-transition rows
                      blended with DP dow-seasonality rows in the generator)
  SeasonalityMLP   -> DP dow_class rows
  GarchVol         -> DP vol-state transition rows + public vol-boost table
                      (the discretized GARCH diffusion)
  FactorModel      -> comove_lambda_pct (market-coupling probability,
                      post-processed from the DP co-movement histogram)
  ReturnWindowAE   -> scoring model: NOT generative, stays enclave-side.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
CLASSES = ["crash", "down", "flat", "up", "surge"]
VOLS = ["calm", "normal", "elevated", "burst"]
DOWS = ["mon", "tue", "wed", "thu", "fri"]
GAP_MAX_DAYS = 4
ACCEPT_MARGIN = 0.02   # student may trail repeat-last by at most this much
# public bin midpoints (%) of the co-movement cells [<40, 40-70, >=70]
COMOVE_MIDPOINTS = {"comove_lo": 20, "comove_mid": 55, "comove_hi": 85}


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def main() -> int:
    rel_p = OUT / "dp_release.json"
    if not rel_p.exists():
        print("ERROR: run `make lean-release` first", file=sys.stderr)
        return 1
    rel = json.loads(rel_p.read_text())
    classes = {c["cell"]: int(c["released"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["released"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in ("class_transition", "dow_class", "vol_transition",
                 "mag_bins", "comove"):
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1

    trans = aux["class_transition"]

    def trans_row(p: str) -> list[dict]:
        """DP-released row blended with (a) a PUBLIC persistence prior
        (derived_from_data: false): the diagonal gets a boost equal to
        max(1, sum of the released row) — calm days persist and big moves
        cluster; and (b) the RELEASED marginal class mix, rescaled to the
        row's own mass (post-processing of released values): with n = 8
        tickers the per-row DP counts are noise-dominated, so anchoring each
        row to the released marginal keeps the chain's stationary
        distribution near the released mix."""
        row = {n: max(0, int(trans.get(f"{p}>{n}", 0))) for n in CLASSES}
        rowsum = max(1, sum(row.values()))
        mixtot = max(1, sum(max(0, classes.get(n, 0)) for n in CLASSES))
        for n in CLASSES:
            row[n] += round(2 * rowsum * max(0, classes.get(n, 0)) / mixtot)
        row[p] += 2 * rowsum
        return [{"name": n, "w": row[n]} for n in CLASSES]

    dow_rows = aux["dow_class"]
    DOW_ROW_TOTAL = 10   # public rescale target (post-processing)

    def dow_row(w: str) -> list[dict]:
        """DP dow row rescaled to a fixed small total (post-processing of the
        release): the dow noise scale is large, so left raw the seasonality
        rows would dominate the transition blend in the generator — rescaled,
        they modulate it. Zero rows fall back to uniform (declared)."""
        raw = {n: max(0, int(dow_rows.get(f"{w}|{n}", 0))) for n in CLASSES}
        tot = sum(raw.values())
        if tot == 0:
            return [{"name": n, "w": 2} for n in CLASSES]
        return [{"name": n, "w": round(DOW_ROW_TOTAL * raw[n] / tot)}
                for n in CLASSES]

    vol_rows = aux["vol_transition"]

    def vol_row(s: str) -> list[dict]:
        return [{"name": n, "w": max(0, int(vol_rows.get(f"{s}>{n}", 0)))}
                for n in VOLS]

    # co-movement -> market-coupling probability λ (post-processing):
    # with prob λ the sign is forced to the market mode, else independent
    # (independence gives ~50% same-sign agreement), so share s = λ + (1-λ)/2
    # i.e. λ = 2s - 1, with s the weighted midpoint of the released bins.
    com = aux["comove"]
    tot = sum(max(0, com.get(k, 0)) for k in COMOVE_MIDPOINTS)
    if tot > 0:
        share_pct = sum(max(0, com.get(k, 0)) * m
                        for k, m in COMOVE_MIDPOINTS.items()) / tot
    else:
        share_pct = 50.0
    lam_pct = max(0, min(100, round(2 * share_pct - 100)))

    genspec = {
        "genspec_version": "1.0.0",
        "kind": "restricted generative specification (no executable code)",
        "source": {
            "dp_release_sha256": hashlib.sha256(rel_p.read_bytes()).hexdigest(),
            "privacy": "all distributions below are DP-released histograms; "
                       "the spec is post-processing of the release",
        },
        "schema": {"format": "stock-tsv",
                   "constraints": ["first price of every ticker = 100.0",
                                   "prices > 0 (log-return random walk)",
                                   "dates strictly increasing weekdays from the public anchor",
                                   "planned class = classOfRet(ret_bp) (mkDay_valid)"]},
        "classes": {"cells": cells_of(classes, CLASSES)},
        "transitions": [{"given": p, "cells": trans_row(p)} for p in CLASSES],
        "transitions_prior": "per-row: released-marginal anchor (2x row mass, "
                             "post-processing) + diagonal persistence boost "
                             "(2x row mass, public prior, derived_from_data: "
                             "false; strong enough to be robust to per-draw "
                             "DP noise)",
        "dow": [{"given": w, "cells": dow_row(w)} for w in DOWS],
        "vol_transitions": [{"given": s, "cells": vol_row(s)} for s in VOLS],
        "mag_bins": {"cells": cells_of(aux["mag_bins"],
                                       [f"mag_{i}" for i in range(6)])},
        "comove_lambda_pct": lam_pct,
        "comove_rule": "lambda = clamp(2*share - 1); share = weighted midpoint "
                       "of the released co-movement bins (post-processing)",
        "temporal": "weekday calendar + magnitude-bin edges + vol-boost table "
                    "from lean/EnclaveSeries/SeriesShape.lean (fixed, "
                    "derived_from_data: false)",
    }
    (OUT / "genspec.json").write_text(json.dumps(genspec, indent=2))

    # ---- enclave-only fidelity report -------------------------------------
    db = ROOT / "artifacts" / "enclave.duckdb"
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    if db.exists():
        import duckdb  # enclave-side only; consumers of the release don't need it
        con = duckdb.connect(str(db), read_only=True)
        rows = con.execute(
            "select ticker, ret_class, d, "
            "lag(ret_class) over w as prev, "
            "d - lag(d) over w as gap "
            "from feat_returns where ret_class is not null "
            "window w as (partition by ticker order by d) "
            "qualify prev is not null and gap <= ?", [GAP_MAX_DAYS]
        ).fetchall()
        rows = [(p, c, d) for (_, c, d, p, _) in rows]
        rows.sort(key=lambda r: r[2])
        val = rows[int(len(rows) * 0.8):]
        # student: argmax of the blended (prior + DP) transition row — the
        # same law the Lean generator samples from (before dow blending)
        blended = {p: {c["name"]: c["w"] for c in trans_row(p)} for p in CLASSES}
        student = {p: max(CLASSES, key=lambda n: blended[p][n]) for p in CLASSES}
        n = len(val)
        student_acc = sum(student[p] == s for p, s, _ in val) / n
        repeat_acc = sum(p == s for p, s, _ in val) / n
        # per-row TV between released and exact in-enclave transition rows
        exact: dict[str, dict[str, int]] = {p: {} for p in CLASSES}
        for p, s, _ in rows:
            exact[p][s] = exact[p].get(s, 0) + 1
        tv = {}
        for p in CLASSES:
            te = sum(exact[p].values())
            tr = sum(max(0, trans.get(f"{p}>{nn}", 0)) for nn in CLASSES)
            if te and tr:
                tv[p] = round(0.5 * sum(
                    abs(exact[p].get(nn, 0) / te
                        - max(0, trans.get(f"{p}>{nn}", 0)) / tr)
                    for nn in CLASSES), 4)
        # teacher accuracy, read from the frozen artifact metadata if present
        teacher_acc = None
        reg = ROOT / "models_out" / "registry.json"
        if reg.exists():
            active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
            if active:
                meta_p = ROOT / "models_out" / "minimodels" / active / "meta.json"
                if meta_p.exists():
                    meta = json.loads(meta_p.read_text())
                    teacher_acc = (meta.get("next_class") or {}).get(
                        "metrics", {}).get("val_accuracy")
        # GARCH persistence: released P(stay | elevated-or-burst) vs exact
        def persistence(d: dict[str, int]) -> float | None:
            hi = ["elevated", "burst"]
            stay = sum(max(0, d.get(f"{a}>{b}", 0)) for a in hi for b in hi)
            leave = sum(max(0, d.get(f"{a}>{b}", 0)) for a in hi for b in VOLS
                        if b not in hi)
            return round(stay / (stay + leave), 4) if stay + leave else None
        exact_vol = {f"{a}>{b}": n for a, b, n in con.execute(
            "with s as (select ticker, vol_state as cur, "
            "lead(vol_state) over (partition by ticker order by d) as nxt "
            "from feat_returns where vol_state is not null) "
            "select cur, nxt, count(*) from s where nxt is not null "
            "group by 1, 2").fetchall()}
        con.close()
        report.update({
            "held_out_transitions": n,
            "student_accuracy": round(student_acc, 4),
            "repeat_last_accuracy": round(repeat_acc, 4),
            "teacher_accuracy": teacher_acc,
            "accepted": bool(student_acc >= repeat_acc - ACCEPT_MARGIN),
            "acceptance_rule": f"student >= repeat_last - {ACCEPT_MARGIN}",
            "per_row_tv_released_vs_exact": tv,
            "garch_persistence_released": persistence(vol_rows),
            "garch_persistence_exact": persistence(exact_vol),
            "comove_lambda_pct": lam_pct,
        })
    rep_dir = ROOT / "artifacts" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "distill_report.json").write_text(json.dumps(report, indent=2))
    print(f"distill: wrote lean/out/genspec.json (lambda={lam_pct}%) and "
          f"artifacts/reports/distill_report.json "
          f"(accepted={report.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

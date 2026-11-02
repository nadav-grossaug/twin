#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (daily-motif mix + the
aux role-transition / dwell / departure / mode histograms) and the seven
suppressed category-2 release artifacts (noise_moments, kinematics,
turn_hist, dist_mix, route_shape, seg_hist, junction_hist). No raw data,
weights or embeddings are read for the
spec, so the genspec inherits the release's privacy accounting by
post-processing. The module is stdlib-only so the off-enclave bundle can run
it as shipped.

Outputs:
  lean/out/genspec.json        (or $GENSPEC_OUT) the restricted generative
                               spec consumed by `lake exe generate`. Keyed by
                               the hash of the DP release: because the DP
                               draws are intentionally non-reproducible, a
                               genspec whose recorded dp_release_sha256 still
                               matches the live release is NOT rewritten
                               (provenance-skip; re-running distill is
                               idempotent).
  lean/out/genspec_meta.json   provenance sidecar, written next to the
                               genspec: the release dir, per-file sha256s of
                               every released input, and the DECLARED public
                               structural priors (derived_from_data: false).
  models_out/fidelity.json     ENCLAVE-ONLY fidelity report: the distilled
                               first-order role-transition student vs the
                               NextRoleGRU teacher vs the return-to-previous
                               baseline on held-out user visit sequences.
                               Skipped (with a message) when the enclave
                               database is absent — in the bundle the spec
                               needs only the release + the public spec.

Distillation table (teacher -> executable statistical replacement):
  NextRoleGRU    -> DP role-transition rows + the home_return prior (the
                    return-home structure DP noise tends to destroy at
                    n = 64 users; declared, not measured)
  DwellClassGRU  -> DP dwell classes (grid-uniform within the public ranges)
  kinematics / GPS noise / turn bins / displacement mix / route shape
                 -> the released category-2 moment summaries (route polylines
                    are rendered from the released seg_hist / junction_hist /
                    route_shape summaries; declared fallback priors stand in
                    on old releases that predate the route-shape artifacts)
  scoring models -> NOT generative, stay enclave-side.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
DB = ROOT / "artifacts" / "enclave.duckdb"
MODELS_OUT = ROOT / "models_out"

MOTIF_CELLS = ["home_day", "commute", "errand_loop", "tour", "irregular"]
ROLES = ["home", "work", "other"]
DWELL_CELLS = ["dw_lt1h", "dw_1_3h", "dw_3_8h", "dw_8_14h", "dw_ge14h"]
DEP_CELLS = ["dep_night", "dep_morning", "dep_afternoon", "dep_evening"]
MODE_CELLS = ["walk", "bike", "bus", "car", "rail", "other"]
TURN_BINS = [f"tb_{i}" for i in range(8)]
DIST_CELLS = ["dist_lt1k", "dist_1_3k", "dist_3_10k", "dist_ge10k"]
SEG_CELLS = ["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"]
JA_CELLS = ["ja_30_60", "ja_60_120", "ja_120_180"]
AUX_NEEDED = ("transitions", "dwell", "departure", "mode")
ACCEPT_MARGIN = 0.02   # student may trail return-to-previous by at most this

# PUBLIC constants (mirrors of the spec; _check_spec_sync verifies them
# against dbt/dbt_project.yml whenever the spec is readable — the bundle
# ships without it and relies on these literals + tests/test_spec_sync.py).
# n = 64 enclave users; the hint is a public constant, never a count.
N_USERS_HINT = 64
WINDOW = {"start_s": 1176336000, "end_s": 1325376000}
DWELL_RANGES_S = {"dw_lt1h": [900, 3599], "dw_1_3h": [3600, 10799],
                  "dw_3_8h": [10800, 28799], "dw_8_14h": [28800, 50399],
                  "dw_ge14h": [50400, 54000]}
DEP_RANGES_H = {"dep_night": [0, 5], "dep_morning": [6, 11],
                "dep_afternoon": [12, 16], "dep_evening": [17, 23]}
DIST_RANGES_M = {"dist_lt1k": [200, 999], "dist_1_3k": [1000, 2999],
                 "dist_3_10k": [3000, 9999], "dist_ge10k": [10000, 25000]}

# the seven category-2 artifacts the genspec post-processes (policy columns)
ARTIFACT_COLUMNS = {
    "noise_moments": ["metric", "value", "n_staypoints", "n_users"],
    "kinematics": ["mode_class", "n_legs", "n_points", "n_users",
                   "median_speed_mps", "p95_speed_mps", "median_dt_s",
                   "median_abs_turn_deg", "median_straightness",
                   "median_path_m", "implausible_rate"],
    "turn_hist": ["turn_bin", "n_points", "n_users"],
    "dist_mix": ["dist_class", "n_trips", "n_users"],
    "route_shape": ["mode_class", "n_legs", "n_users", "median_circuity",
                    "p90_circuity", "median_straightness",
                    "junctions_per_km", "share_left"],
    "seg_hist": ["seg_class", "n_segments", "n_users"],
    "junction_hist": ["ja_class", "n_junctions", "n_users"],
}

# route-shape artifacts are NEW: an old release (or bundle) may lack them, in
# which case the DECLARED fallback priors below stand in (genspec_meta says so)
ROUTE_ARTIFACTS = ("route_shape", "seg_hist", "junction_hist")
FALLBACK_SEG_W = [5, 3, 2, 1]       # vs SEG_CELLS (short segments dominate)
FALLBACK_JA_W = [5, 3, 2]           # vs JA_CELLS (gentle turns dominate)
FALLBACK_SHARE_LEFT_PCT = 50        # no left/right asymmetry assumed
FALLBACK_CIRC_PM = 1400             # circuity 1.4 per-mille for every mode

PRIORS = [
    {"name": "home_return_boost", "derived_from_data": False,
     "rule": "transition row given g != 'home': the 'home' cell gets "
             "max(1, row_sum // 2) added — the return-home structure that "
             "per-draw DP noise tends to destroy at n = 64 users"},
    {"name": "motif_floor", "derived_from_data": False,
     "rule": "any motif cell released as 0 is floored at 1 so the generator "
             "can still plant that behavior"},
    {"name": "turn_rescale", "derived_from_data": False,
     "rule": "turn-bin weights are released n_points // 1000 (min 1 for "
             "released rows, 0 for suppressed bins) — a monotone public "
             "rescale, not new information"},
]


def _check_spec_sync() -> None:
    """Fail loudly if the embedded public edges drift from the live spec
    (no-op in the bundle, where the spec file is not shipped)."""
    try:
        from spec import load_spec  # noqa: PLC0415
        s = load_spec()
    except Exception:
        return
    e1, e2, e3, e4 = (int(s[k]) for k in ("dw_e1_s", "dw_e2_s", "dw_e3_s",
                                          "dw_e4_s"))
    want_dw = {"dw_lt1h": [DWELL_RANGES_S["dw_lt1h"][0], e1 - 1],
               "dw_1_3h": [e1, e2 - 1], "dw_3_8h": [e2, e3 - 1],
               "dw_8_14h": [e3, e4 - 1],
               "dw_ge14h": [e4, DWELL_RANGES_S["dw_ge14h"][1]]}
    if DWELL_RANGES_S != want_dw:
        raise ValueError("distill dwell ranges drifted from the spec")
    hn, hm, ha = (int(s[k]) for k in ("dep_night_max_h", "dep_morning_max_h",
                                      "dep_afternoon_max_h"))
    if DEP_RANGES_H != {"dep_night": [0, hn], "dep_morning": [hn + 1, hm],
                        "dep_afternoon": [hm + 1, ha],
                        "dep_evening": [ha + 1, 23]}:
        raise ValueError("distill departure bins drifted from the spec")
    d1, d2, d3 = (int(s[k]) for k in ("dist_e1_m", "dist_e2_m", "dist_e3_m"))
    want_dist = {"dist_lt1k": [DIST_RANGES_M["dist_lt1k"][0], d1 - 1],
                 "dist_1_3k": [d1, d2 - 1], "dist_3_10k": [d2, d3 - 1],
                 "dist_ge10k": [d3, DIST_RANGES_M["dist_ge10k"][1]]}
    if DIST_RANGES_M != want_dist:
        raise ValueError("distill displacement ranges drifted from the spec")
    if WINDOW != {"start_s": int(s["window_start_s"]),
                  "end_s": int(s["window_end_s"])}:
        raise ValueError("distill window drifted from the spec")


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def _release_dir() -> Path | None:
    """$RELEASE_DIR, or the latest release/<export_id>/ export, or None (a
    fresh enclave run where `make distill` precedes `make export`)."""
    env = os.environ.get("RELEASE_DIR")
    if env:
        return Path(env)
    dirs = sorted(d for d in ROOT.glob("release/*") if d.is_dir())
    return dirs[-1] if dirs else None


def _released_artifact(name: str, rel_dir: Path | None) -> tuple[list[dict] | None, str]:
    """Rows of a policy-filtered release artifact, with its provenance.

    Preferred source: <release_dir>/<name>.csv (or <name>.csv at the repo
    root — the bundle layout). On a fresh enclave run `make distill` precedes
    `make export`, so as a fallback the SAME rows are read from the mart with
    the policy's own suppression thresholds applied — byte-for-byte what
    release.py will export."""
    candidates = []
    if rel_dir is not None:
        candidates.append(rel_dir / f"{name}.csv")
    candidates.append(ROOT / f"{name}.csv")
    for p in candidates:
        if p.exists():
            with open(p, newline="") as f:
                return [dict(r) for r in csv.DictReader(f)], str(p)
    if not DB.exists():
        return None, ""
    import duckdb  # enclave-side only
    import yaml  # enclave-side only
    policy = yaml.safe_load((ROOT / "config" / "release_policy.yaml").read_text())
    spec = policy["artifacts"][name]
    if not spec.get("enabled", False):
        raise PermissionError(f"release policy does not enable artifact {name!r}")
    columns = ARTIFACT_COLUMNS[name]
    where = " AND ".join(
        f"{r['column']} >= {int(r['value'])}" for r in spec.get("suppress", {}).values())
    con = duckdb.connect(str(DB), read_only=True)
    try:
        rows = con.execute(
            f"select {', '.join(columns)} from {spec['relation']}"
            + (f" where {where}" if where else "")).fetchall()
    finally:
        con.close()
    return ([dict(zip(columns, r)) for r in rows],
            f"duckdb:{spec['relation']} (policy suppression applied; pre-export)")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    _check_spec_sync()
    rel_dir = _release_dir()
    rel_p = None
    if rel_dir is not None and (rel_dir / "dp_release.json").exists():
        rel_p = rel_dir / "dp_release.json"
    elif (OUT / "dp_release.json").exists():
        rel_p = OUT / "dp_release.json"
    if rel_p is None:
        print("ERROR: no dp_release.json (neither release/<id>/ nor lean/out/)"
              " — run `make train` (or at least `make lean-release`) first",
              file=sys.stderr)
        return 1
    rel_sha = _sha(rel_p)
    rel = json.loads(rel_p.read_text())
    motif = {c["cell"]: int(c["noised_count"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in AUX_NEEDED:
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1

    trans = aux["transitions"]

    def trans_row(g: str) -> list[dict]:
        """DP-released row plus the PUBLIC home_return_boost prior
        (derived_from_data: false): for every row given g != 'home' the
        'home' cell gets max(1, row_sum // 2) added — people return home,
        and at n = 64 users the per-draw DP noise routinely erases that
        structure from the raw row."""
        row = {b: max(0, int(trans.get(f"{g}__{b}", 0))) for b in ROLES}
        if g != "home":
            row["home"] += max(1, sum(row.values()) // 2)
        return [{"name": b, "w": row[b]} for b in ROLES]

    # motif_floor prior: a 0-weight motif cell would make the generator
    # unable to plant that behavior at all — floor at 1 (declared).
    motif_cells = [{"name": n, "w": max(1, max(0, int(motif.get(n, 0))))}
                   for n in MOTIF_CELLS]

    # category-2 suppressed artifacts (CSV release files, or the equivalent
    # policy-filtered mart rows on a pre-export enclave run). The route-shape
    # trio is new: when an OLD release lacks it, declared fallback priors
    # stand in (recorded below in genspec_meta) instead of failing.
    art_rows, art_src, route_fallbacks = {}, {}, []
    for name in ARTIFACT_COLUMNS:
        rows, src = _released_artifact(name, rel_dir)
        if rows is None:
            if name in ROUTE_ARTIFACTS:
                route_fallbacks.append(name)
                continue
            print("ERROR: no release artifacts (release/<id>/*.csv or "
                  "<name>.csv at the root) and no database to policy-filter "
                  "— run `make train` (or `make export`) first",
                  file=sys.stderr)
            return 1
        art_rows[name], art_src[name] = rows, src

    turn_pts = {r["turn_bin"]: max(0, int(r["n_points"]))
                for r in art_rows["turn_hist"]}
    turn_cells = [{"name": b,
                   "w": max(1, turn_pts[b] // 1000) if b in turn_pts else 0}
                  for b in TURN_BINS]
    dist_trips = {r["dist_class"]: max(0, int(r["n_trips"]))
                  for r in art_rows["dist_mix"]}
    dist_cells = cells_of(dist_trips, DIST_CELLS)
    kinematics = [{"mode": r["mode_class"],
                   "speed_mmps": max(0, int(float(r["median_speed_mps"]) * 1000)),
                   "dt_ds": max(0, int(float(r["median_dt_s"]) * 10))}
                  for r in art_rows["kinematics"]]
    moments = {r["metric"]: float(r["value"]) for r in art_rows["noise_moments"]
               if r.get("value") not in (None, "")}
    sd_mm = 2000
    for key in ("tight_resid_sd_m_median", "resid_sd_m_median"):
        if key in moments:
            sd_mm = max(1, int(moments[key] * 1000))
            break
    lag1_pct = 78
    if "tight_resid_lag1_median" in moments:
        lag1_pct = min(95, max(0, int(moments["tight_resid_lag1_median"] * 100)))

    # ROUTE SHAPE: released seg-length / junction-angle histograms + per-mode
    # circuity and junction density, so the generator renders road-like
    # polylines instead of straight vectors. Old releases lack these
    # artifacts: the DECLARED fallback priors stand in (genspec_meta records
    # them; derived_from_data: false).
    seg_rows = art_rows.get("seg_hist") or []
    if seg_rows:
        seg_n = {r["seg_class"]: max(0, int(r["n_segments"])) for r in seg_rows}
        seg_cells = cells_of(seg_n, SEG_CELLS)
    else:
        if "seg_hist" not in route_fallbacks:
            route_fallbacks.append("seg_hist")
        seg_cells = [{"name": n, "w": w}
                     for n, w in zip(SEG_CELLS, FALLBACK_SEG_W)]
    ja_rows = art_rows.get("junction_hist") or []
    if ja_rows:
        ja_n = {r["ja_class"]: max(0, int(r["n_junctions"])) for r in ja_rows}
        ja_cells = cells_of(ja_n, JA_CELLS)
    else:
        if "junction_hist" not in route_fallbacks:
            route_fallbacks.append("junction_hist")
        ja_cells = [{"name": n, "w": w} for n, w in zip(JA_CELLS, FALLBACK_JA_W)]
    rs_rows = [r for r in (art_rows.get("route_shape") or [])
               if r.get("median_circuity") not in (None, "")]
    if rs_rows:
        legs_total = sum(max(1, int(r["n_legs"])) for r in rs_rows)
        share_mean = sum(float(r["share_left"]) * max(1, int(r["n_legs"]))
                         for r in rs_rows) / legs_total
        share_left_pct = min(65, max(35, round(100 * share_mean)))
        route_circ = [{"mode": r["mode_class"],
                       "circ_pm": max(0, int(float(r["median_circuity"]) * 1000)),
                       "junc_pkm_c": max(0, int(float(r["junctions_per_km"]) * 100))}
                      for r in rs_rows]
    else:
        if "route_shape" not in route_fallbacks:
            route_fallbacks.append("route_shape")
        share_left_pct = FALLBACK_SHARE_LEFT_PCT
        route_circ = [{"mode": m, "circ_pm": FALLBACK_CIRC_PM,
                       "junc_pkm_c": 300} for m in MODE_CELLS]

    genspec = {
        "genspec_version": "4.0.0",
        "kind": "restricted generative specification (no executable code)",
        "spec_version": str(rel.get("spec_version", "4.0.0")),
        "source": {
            "dp_release_sha256": rel_sha,
            "privacy": "all weights below are DP-released counts, "
                       "suppression-thresholded category-2 summaries, or "
                       "released + a declared public prior; the spec is "
                       "post-processing of the release",
        },
        "n_users_hint": N_USERS_HINT,
        "motif": {"cells": motif_cells},
        "transitions": {"rows": [{"given": g, "cells": trans_row(g)}
                                 for g in ROLES]},
        "dwell": {"cells": cells_of(aux["dwell"], DWELL_CELLS)},
        "departure": {"cells": cells_of(aux["departure"], DEP_CELLS)},
        "mode": {"cells": cells_of(aux["mode"], MODE_CELLS)},
        "turn": {"cells": turn_cells},
        "seg_hist": {"cells": seg_cells},
        "junction_hist": {"cells": ja_cells},
        "routes": {"share_left_pct": share_left_pct, "circ_pm": route_circ},
        "dist_mix": {"cells": dist_cells},
        "kinematics": kinematics,
        "noise": {"sd_mm": sd_mm, "lag1_pct": lag1_pct},
        "priors": PRIORS,
        "window": WINDOW,
        "dwell_ranges_s": DWELL_RANGES_S,
        "dep_ranges_h": DEP_RANGES_H,
        "dist_ranges_m": DIST_RANGES_M,
    }

    # provenance-skip: the DP draws are non-reproducible, so a genspec keyed
    # to the live release hash is authoritative — do not rewrite it.
    gs_path = Path(os.environ.get("GENSPEC_OUT") or (OUT / "genspec.json"))
    if gs_path.exists():
        try:
            old = json.loads(gs_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if (old.get("source", {}).get("dp_release_sha256") == rel_sha
                and all(k in old for k in ("seg_hist", "junction_hist",
                                           "routes"))):
            print(f"distill: {gs_path.name} already keyed to dp_release "
                  f"{rel_sha[:12]} — reusing (provenance-skip)")
            return _fidelity(old)
    gs_path.parent.mkdir(parents=True, exist_ok=True)
    gs_path.write_text(json.dumps(genspec, indent=2))

    meta = {
        "kind": "distill provenance (sidecar of the genspec)",
        "spec_version": genspec["spec_version"],
        "source": {
            "release_dir": str(rel_dir) if rel_dir is not None else None,
            "dp_release": str(rel_p),
            "dp_release_sha256": rel_sha,
            "artifacts": {
                name: ({"source": src, "sha256": _sha(Path(src))}
                       if not src.startswith("duckdb:")
                       else {"source": src, "sha256": None,
                             "note": "pre-export enclave run: policy-filtered "
                                     "mart rows, byte-for-byte what release.py "
                                     "exports"})
                for name, src in art_src.items()},
        },
        "priors": PRIORS,
        "fallback_priors": [
            {"name": name, "derived_from_data": False,
             "rule": {
                 "seg_hist": f"release predates the seg_hist artifact: "
                             f"declared weights {FALLBACK_SEG_W} over "
                             f"{SEG_CELLS}",
                 "junction_hist": f"release predates the junction_hist "
                                  f"artifact: declared weights "
                                  f"{FALLBACK_JA_W} over {JA_CELLS}",
                 "route_shape": f"release predates the route_shape artifact: "
                                f"share_left_pct {FALLBACK_SHARE_LEFT_PCT}, "
                                f"circ_pm {FALLBACK_CIRC_PM} and junc_pkm_c "
                                f"300 for every mode class",
             }[name]}
            for name in sorted(route_fallbacks)],
        "public_constants": {"n_users_hint": N_USERS_HINT, "window": WINDOW},
    }
    meta_path = gs_path.with_name("genspec_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"distill: wrote {gs_path} + {meta_path.name} "
          f"(n_users_hint={N_USERS_HINT}, {len(kinematics)} kinematic modes, "
          f"noise sd_mm={sd_mm} lag1_pct={lag1_pct}, "
          f"routes share_left_pct={share_left_pct} "
          f"{len(route_circ)} circ modes"
          + (f", route fallbacks: {sorted(route_fallbacks)}"
             if route_fallbacks else "") + ")")
    return _fidelity(genspec)


def _fidelity(genspec: dict) -> int:
    """ENCLAVE-ONLY: the distilled first-order role-transition student vs the
    NextRoleGRU teacher vs the return-to-previous baseline, on held-out user
    visit sequences (the same user-id-hash split train_minimodels.py froze).
    Skipped cleanly off-enclave: the bundle carries no database and no
    models_out, and the genspec does not depend on this report."""
    if not DB.exists():
        print("distill: fidelity skipped — no enclave database (bundle mode: "
              "the genspec needs only the release + the public spec)")
        return 0
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    import duckdb  # enclave-side only
    from train_minimodels import is_train_user  # noqa: PLC0415
    con = duckdb.connect(str(DB), read_only=True)
    rows = con.execute(
        "select user_id, role from feat_visits "
        "order by user_id, ts_arrive_s, sp_id").fetchall()
    con.close()
    seqs: dict[str, list[str]] = {}
    for u, r in rows:
        seqs.setdefault(str(u), []).append(str(r))
    held = [s for u, s in sorted(seqs.items()) if not is_train_user(u)]
    blended = {row["given"]: {c["name"]: c["w"] for c in row["cells"]}
               for row in genspec["transitions"]["rows"]}
    student = {r: max(ROLES, key=lambda n: blended[r][n]) for r in ROLES}
    n = ok_student = ok_base = 0
    for s in held:
        for i in range(1, len(s)):
            came_from = s[i - 2] if i >= 2 else s[i - 1]
            n += 1
            ok_student += student.get(s[i - 1], "home") == s[i]
            ok_base += came_from == s[i]
    if n:
        student_acc = ok_student / n
        baseline_acc = ok_base / n
        # teacher accuracy from the frozen artifact metadata: computed on
        # the SAME deterministic user split, so directly comparable
        teacher_acc = None
        reg = MODELS_OUT / "registry.json"
        if reg.exists():
            active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
            if active:
                meta_p = MODELS_OUT / "minimodels" / active / "meta.json"
                if meta_p.exists():
                    meta = json.loads(meta_p.read_text())
                    teacher_acc = (meta.get("next_role") or {}).get(
                        "metrics", {}).get("val_accuracy")
        report.update({
            "held_out_transitions": n,
            "student_accuracy": round(student_acc, 4),
            "return_prev_accuracy": round(baseline_acc, 4),
            "teacher_accuracy": teacher_acc,
            "accepted": bool(student_acc >= baseline_acc - ACCEPT_MARGIN),
            "acceptance_rule": f"student >= return_to_previous - {ACCEPT_MARGIN}",
            "student_rule": "argmax of the blended (released + "
                            "home_return_boost prior) role-transition row",
        })
    else:
        print("distill: fidelity skipped — no held-out visit sequences in the "
              "enclave database")
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    (MODELS_OUT / "fidelity.json").write_text(json.dumps(report, indent=2))
    print(f"distill: wrote models_out/fidelity.json "
          f"(accepted={report.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

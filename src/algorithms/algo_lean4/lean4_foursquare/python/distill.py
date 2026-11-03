#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (travel-count mix + the
aux region / region_transition / gap_class / duration_class / distance_class
/ month / episode_len / return_rate histograms) and the two suppressed
release artifacts (topk_corridors, gap_moment_stats). No raw data, weights
or embeddings are read for the spec, so the genspec inherits the release's
privacy accounting by post-processing. The module is stdlib-only so the
off-enclave bundle can run it as shipped.

Outputs:
  lean/out/genspec.json       the restricted generative spec consumed by
                              `lake exe generate`. Keyed by the hash of the
                              DP release: because the DP draws are
                              intentionally non-reproducible, a genspec
                              whose recorded dp_release_sha256 still matches
                              the live release is NOT rewritten (provenance-
                              skip; re-running distill is idempotent).
  models_out/fidelity.json    ENCLAVE-ONLY fidelity report: the distilled
                              first-order transition-matrix student vs the
                              NextRegionGRU teacher vs the return-to-previous
                              baseline on held-out user sequences.

Distillation table (teacher -> executable statistical replacement):
  NextRegionGRU     -> DP region_transition rows + the return_boost prior
                       (the generator applies second-order return behavior:
                       a per-user return-rate band drawn from the released
                       return_rate cells, with that probability the next hop
                       goes back to the previous country — a first-order
                       matrix cannot express it, the band release can)
  GapClassGRU       -> DP gap_class rows + suppressed gap moments
  EpisodeShapeModel -> DP episode_len rows
  UserProfileAE     -> scoring model: NOT generative, stays enclave-side.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
DB = ROOT / "artifacts" / "enclave.duckdb"
MODELS_OUT = ROOT / "models_out"

REGIONS = ["NA", "LA", "EU", "ME", "AF", "SA", "EA", "OC"]
TC_BINS = ["tc_1", "tc_2", "tc_3_4", "tc_5_8", "tc_9_16", "tc_17_32", "tc_33p"]
GAP_CLASSES = ["gap_hop", "gap_trip", "gap_monthly", "gap_quarterly", "gap_rare"]
DUR_CLASSES = ["dur_day", "dur_week", "dur_month", "dur_quarter", "dur_long"]
DIST_CLASSES = ["dist_short", "dist_mid", "dist_long", "dist_ultra"]
MONTH_CELLS = [f"m{i:02d}" for i in range(1, 13)]
HOD_CELLS = [f"h{i:02d}" for i in range(24)]
EP_BINS = ["ep_1", "ep_2", "ep_3", "ep_4_5", "ep_6p"]
RR_BANDS = ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]
AUX_NEEDED = ("region", "region_transition", "gap_class", "duration_class",
              "distance_class", "month", "hod", "episode_len", "return_rate")
ACCEPT_MARGIN = 0.02   # student may trail return-to-previous by at most this

# PUBLIC bin edges (mirrors of the spec; _check_spec_sync verifies them
# against dbt/dbt_project.yml whenever the spec is readable — the bundle
# ships without it and relies on these literals + tests/test_spec_sync.py).
WINDOW = {"start_s": 1333497600, "end_s": 1379376000}
TC_BIN_RANGES = {"tc_1": [1, 1], "tc_2": [2, 2], "tc_3_4": [3, 4],
                 "tc_5_8": [5, 8], "tc_9_16": [9, 16], "tc_17_32": [17, 32],
                 "tc_33p": [33, 64]}
GAP_RANGES_S = {"gap_hop": [3600, 172800], "gap_trip": [172800, 604800],
                "gap_monthly": [604800, 2592000],
                "gap_quarterly": [2592000, 7776000],
                "gap_rare": [7776000, 15552000]}
DUR_RANGES_S = {"dur_day": [0, 86400], "dur_week": [86400, 604800],
                "dur_month": [604800, 2592000],
                "dur_quarter": [2592000, 7776000],
                "dur_long": [7776000, 15552000]}
DIST_RANGES_KM = {"dist_short": [0, 500], "dist_mid": [500, 2000],
                  "dist_long": [2000, 7000], "dist_ultra": [7000, 20000]}


def _check_spec_sync() -> None:
    """Fail loudly if the embedded public edges drift from the live spec
    (no-op in the bundle, where the spec file is not shipped)."""
    try:
        from spec import load_spec  # noqa: PLC0415
        s = load_spec()
    except Exception:
        return
    want = {
        ("gap_hop", 1): s["gap_hop_max_s"], ("gap_trip", 1): s["gap_trip_max_s"],
        ("gap_monthly", 1): s["gap_monthly_max_s"],
        ("gap_quarterly", 1): s["gap_quarterly_max_s"],
    }
    for (cls, i), v in want.items():
        if GAP_RANGES_S[cls][i] != int(v):
            raise ValueError(f"distill gap range {cls} drifted from the spec")
    for cls, key in (("dur_day", "dur_day_max_s"), ("dur_week", "dur_week_max_s"),
                     ("dur_month", "dur_month_max_s"),
                     ("dur_quarter", "dur_quarter_max_s")):
        if DUR_RANGES_S[cls][1] != int(s[key]):
            raise ValueError(f"distill duration range {cls} drifted from the spec")
    for cls, key in (("dist_short", "dist_short_max_km"),
                     ("dist_mid", "dist_mid_max_km"),
                     ("dist_long", "dist_long_max_km")):
        if DIST_RANGES_KM[cls][1] != int(s[key]):
            raise ValueError(f"distill distance range {cls} drifted from the spec")
    edges = [int(s[k]) for k in ("tc_e2", "tc_e3", "tc_e5", "tc_e9",
                                 "tc_e17", "tc_e33")]
    if edges != [2, 3, 5, 9, 17, 33] or \
            WINDOW != {"start_s": int(s["window_start_s"]),
                       "end_s": int(s["window_end_s"])}:
        raise ValueError("distill tc bins / window drifted from the spec")


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def _released_artifact(name: str, columns: list[str]) -> list[dict] | None:
    """Rows of a policy-filtered release artifact.

    Preferred source: the latest release/<export_id>/<name>.csv (or
    <name>.csv at the repo root — the bundle layout). On a fresh enclave run
    `make distill` precedes `make export`, so as a fallback the SAME rows are
    read from the mart with the policy's own suppression thresholds applied —
    byte-for-byte what release.py will export."""
    candidates = sorted(ROOT.glob(f"release/*/{name}.csv"))
    if (ROOT / f"{name}.csv").exists():
        candidates.append(ROOT / f"{name}.csv")
    if candidates:
        with open(candidates[-1], newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if not DB.exists():
        return None
    import duckdb  # enclave-side only
    import yaml  # enclave-side only
    policy = yaml.safe_load((ROOT / "config" / "release_policy.yaml").read_text())
    spec = policy["artifacts"][name]
    if not spec.get("enabled", False):
        raise PermissionError(f"release policy does not enable artifact {name!r}")
    where = " AND ".join(
        f"{r['column']} >= {int(r['value'])}" for r in spec.get("suppress", {}).values())
    con = duckdb.connect(str(DB), read_only=True)
    try:
        rows = con.execute(
            f"select {', '.join(columns)} from {spec['relation']}"
            + (f" where {where}" if where else "")).fetchall()
    finally:
        con.close()
    return [dict(zip(columns, r)) for r in rows]


def main() -> int:
    _check_spec_sync()
    rel_p = OUT / "dp_release.json"
    if not rel_p.exists():
        print("ERROR: run `make lean-release` first", file=sys.stderr)
        return 1
    rel_sha = hashlib.sha256(rel_p.read_bytes()).hexdigest()
    rel = json.loads(rel_p.read_text())
    tc = {c["cell"]: int(c["noised_count"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in AUX_NEEDED:
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1

    trans = aux["region_transition"]

    def trans_row(r: str) -> list[dict]:
        """DP-released row plus the PUBLIC return-boost prior
        (derived_from_data: false): the DIAGONAL gets a boost equal to
        max(1, sum of the released row) — intra-region hops dominate the
        real pattern (e.g. EA__EA for MY<->SG commuting), and the boost is
        strong enough to be robust to per-draw DP noise. The second-order
        part of return behavior (going BACK where you came from) cannot
        live in a first-order matrix: the generator applies it per user
        from the released return_rate bands."""
        row = {n: max(0, int(trans.get(f"{r}__{n}", 0))) for n in REGIONS}
        row[r] += max(1, sum(row.values()))
        return [{"name": n, "w": row[n]} for n in REGIONS]

    corridors_rows = _released_artifact(
        "topk_corridors",
        ["prev_country", "next_country", "prev_region", "next_region"])
    countries_rows = _released_artifact(
        "topk_countries", ["country", "region", "n_in", "n_out"])
    moments_rows = _released_artifact(
        "gap_moment_stats",
        ["gap_class", "n_gaps", "mean_days", "median_days", "sd_days"])
    if corridors_rows is None or moments_rows is None or countries_rows is None:
        print("ERROR: no release artifacts (release/<id>/*.csv) and no "
              "database to policy-filter — run `make export` (or `make dbt`) "
              "first", file=sys.stderr)
        return 1
    corridors = [{"a": r["prev_country"], "b": r["next_country"],
                  "region_a": r["prev_region"], "region_b": r["next_region"]}
                 for r in corridors_rows]
    countries = [{"c": r["country"], "region": r["region"],
                  "w_in": int(r["n_in"]), "w_out": int(r["n_out"])}
                 for r in countries_rows]
    gap_moments = [{"gap_class": r["gap_class"],
                    "mean_days": float(r["mean_days"]),
                    "sd_days": float(r["sd_days"])} for r in moments_rows]

    genspec = {
        "genspec_version": "2.0.0",
        "kind": "restricted generative specification (no executable code)",
        "spec_version": str(rel.get("spec_version", "2.0.0")),
        "source": {
            "dp_release_sha256": rel_sha,
            "privacy": "all weights below are DP-released counts (or released "
                       "+ a declared public prior); the spec is "
                       "post-processing of the release",
        },
        "n_users_hint": sum(max(0, tc.get(n, 0)) for n in TC_BINS),
        "travel_count": {"cells": cells_of(tc, TC_BINS)},
        "region_mix": {"cells": cells_of(aux["region"], REGIONS)},
        "transitions": {"rows": [{"given": r, "cells": trans_row(r)}
                                 for r in REGIONS]},
        "gap_class": {"cells": cells_of(aux["gap_class"], GAP_CLASSES)},
        "duration_class": {"cells": cells_of(aux["duration_class"], DUR_CLASSES)},
        "distance_class": {"cells": cells_of(aux["distance_class"], DIST_CLASSES)},
        "month": {"cells": cells_of(aux["month"], MONTH_CELLS)},
        "hod": {"cells": cells_of(aux["hod"], HOD_CELLS)},
        "episode_len": {"cells": cells_of(aux["episode_len"], EP_BINS)},
        "return_rate": {"cells": cells_of(aux["return_rate"], RR_BANDS)},
        "corridors": corridors,
        "countries": countries,
        "gap_moments": gap_moments,
        "priors": [
            {"name": "return_boost", "derived_from_data": False,
             "rule": "transition row r: diagonal cell r += max(1, row sum) "
                     "(intra-region persistence); generator: draw each "
                     "user's return-rate band from the return_rate cells and "
                     "with that probability return to the previous country "
                     "(RETURN-FIRST: decided before the region walk — "
                     "second-order, outside the first-order matrix)"},
            {"name": "uniform_dow", "derived_from_data": False,
             "rule": "day-of-week is NOT released; dates carry a uniform "
                     "public prior over the week; time-of-day comes from the "
                     "DP hod histogram, minutes/seconds uniform within the "
                     "drawn hour"},
        ],
        "window": WINDOW,
        "tc_bin_ranges": TC_BIN_RANGES,
        "gap_ranges_s": GAP_RANGES_S,
        "dur_ranges_s": DUR_RANGES_S,
        "dist_ranges_km": DIST_RANGES_KM,
    }

    # provenance-skip: the DP draws are non-reproducible, so a genspec keyed
    # to the live release hash is authoritative — do not rewrite it.
    gs_path = OUT / "genspec.json"
    if gs_path.exists():
        try:
            old = json.loads(gs_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if old.get("source", {}).get("dp_release_sha256") == rel_sha:
            print(f"distill: genspec.json already keyed to dp_release "
                  f"{rel_sha[:12]} — reusing (provenance-skip)")
            return _fidelity(old)
    gs_path.write_text(json.dumps(genspec, indent=2))
    print(f"distill: wrote lean/out/genspec.json "
          f"({len(corridors)} corridors, {len(countries)} countries, "
          f"n_users_hint={genspec['n_users_hint']})")
    return _fidelity(genspec)


def _fidelity(genspec: dict) -> int:
    """ENCLAVE-ONLY: the distilled first-order student vs the NextRegionGRU
    teacher vs the return-to-previous baseline, on held-out user sequences
    (the same user-id-hash split train_minimodels.py froze)."""
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    if DB.exists():
        import duckdb  # enclave-side only
        from train_minimodels import is_train_user  # noqa: PLC0415
        con = duckdb.connect(str(DB), read_only=True)
        rows = con.execute(
            "select userid, prev_region, next_region from feat_transfers "
            "where prev_region != 'XX' and next_region != 'XX' "
            "order by userid, ts_s").fetchall()
        con.close()
        seqs: dict[int, list[str]] = {}
        for u, p, n in rows:
            u = int(u)
            if u not in seqs:
                seqs[u] = [p]
            seqs[u].append(n)
        held = [s for u, s in sorted(seqs.items()) if not is_train_user(u)]
        blended = {row["given"]: {c["name"]: c["w"] for c in row["cells"]}
                   for row in genspec["transitions"]["rows"]}
        student = {r: max(REGIONS, key=lambda n: blended[r][n]) for r in REGIONS}
        n = ok_student = ok_base = 0
        for s in held:
            for i in range(1, len(s)):
                came_from = s[i - 2] if i >= 2 else s[i - 1]
                n += 1
                ok_student += student[s[i - 1]] == s[i]
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
                        teacher_acc = (meta.get("next_region") or {}).get(
                            "metrics", {}).get("val_accuracy")
            report.update({
                "held_out_transitions": n,
                "student_accuracy": round(student_acc, 4),
                "return_prev_accuracy": round(baseline_acc, 4),
                "teacher_accuracy": teacher_acc,
                "accepted": bool(student_acc >= baseline_acc - ACCEPT_MARGIN),
                "acceptance_rule": f"student >= return_to_previous - {ACCEPT_MARGIN}",
                "student_rule": "argmax of the blended (released + "
                                "return_boost prior) transition row",
            })
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    (MODELS_OUT / "fidelity.json").write_text(json.dumps(report, indent=2))
    print(f"distill: wrote models_out/fidelity.json "
          f"(accepted={report.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

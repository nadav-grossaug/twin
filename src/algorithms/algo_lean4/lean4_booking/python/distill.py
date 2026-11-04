#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (res_count mix + the
aux country / country_transition / nights / gap / month / dow / device /
affiliate / booker / return_rate histograms) and the three suppressed
release artifacts (topk_cities, topk_city_transitions, gap_moment_stats).
No raw data, weights or embeddings are read for the spec, so the genspec
inherits the release's privacy accounting by post-processing. The module
is stdlib-only so the off-enclave bundle can run it as shipped.

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
                              NextCountry teacher vs the return-to-previous
                              baseline on held-out user sequences.

Distillation table (teacher -> executable statistical replacement):
  NextCountryMLP     -> DP country_transition rows + the return_boost prior
                        (the generator applies second-order return behavior:
                        a per-trip repeat decision at the released
                        return_rate band midpoint plants ONE hop back to
                        the previous city/tier (trip-level, matching the
                        released share of trips with a repeat city — a
                        first-order matrix cannot express it, the band
                        release can)
  TripLenMLP         -> DP res_count rows (per-user reservations bins)
  StayNightsMLP      -> DP nights rows
  DeviceMLP          -> DP device rows (+ the chi2 association cells)
  UserProfileAE      -> scoring model: NOT generative, stays enclave-side.
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

RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
CT_CELLS = [
    "Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
    "Borginia", "Kangan", "Bozatta", "Alvonia", "Holy Britannian Empire",
    "Axphain", "Santa Prisca", "Patusan", "Rolisica", "Kasnia", "ct_other",
]
NIGHTS_CELLS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
GAP_CELLS = ["g_0", "g_1", "g_2_3", "g_4p"]
MONTH_WINDOW_CELLS = [f"mw{i:02d}" for i in range(15)]
DEVICE_CELLS = ["desktop", "mobile", "tablet"]
AFF_CELLS = ["a9924", "a359", "a384", "a9452", "a4541", "a8436",
             "a10332", "a3631", "aff_other"]
BOOKER_CELLS = ["Gondal", "The Devilfire Empire", "Elbonia",
                "Tcherkistan", "Bartovia"]
RR_BANDS = ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]
REPEAT_STRUCT_CELLS = ["r_ext1", "r_ext2", "r_ext3p",
                       "r_ret1", "r_ret2p", "r_both"]
TRIPS_CELLS = ["tp_1", "tp_2", "tp_3p"]
DEVICE_PERSIST_CELLS = ["dv_1", "dv_2", "dv_3"]
AFF_PERSIST_CELLS = ["ap_1", "ap_2", "ap_3", "ap_4p"]
CPT_CELLS = ["cpt_1", "cpt_2", "cpt_3", "cpt_4p"]
DEVICE_PLAN_CELLS = ["1:desktop", "1:mobile", "1:tablet",
                     "2:dm", "2:dt", "2:mt", "3:all"]
DOMESTIC_START_CELLS = ["Gondal", "The Devilfire Empire", "Elbonia",
                        "Tcherkistan", "Bartovia"]
AUX_NEEDED = ("country", "country_transition", "nights", "gap",
              "month_window", "device", "affiliate", "booker",
              "return_rate", "repeat_struct", "trips_per_user",
              "device_persist", "affiliate_persist", "countries_per_trip",
              "device_plan", "domestic_start")
ACCEPT_MARGIN = 0.02   # student may trail return-to-previous by at most this

# PUBLIC bin edges (mirrors of the spec; _check_spec_sync verifies them
# against dbt/dbt_project.yml whenever the spec is readable — the bundle
# ships without it and relies on these literals + tests/test_spec_sync.py).
# rc_13p's DRAW range ends at rc_13p_draw_hi (24, ~p99 of the measured
# per-user reservation counts): uniform-in-bin draws over [13,64] put the
# bin's synthetic mean at 38.5 vs the real ~17-20, inflating the synthetic
# records/user mean by ~17% (spec 1.1.0; the BIN still covers 13+ —
# membership is uncapped, only the draw range narrowed).
WINDOW = {"start_day": 16770, "end_day": 17226}
RC_BIN_RANGES = {"rc_1_3": [1, 3], "rc_4": [4, 4], "rc_5": [5, 5],
                 "rc_6": [6, 6], "rc_7_8": [7, 8], "rc_9_12": [9, 12],
                 "rc_13p": [13, 24]}
GAP_RANGES_D = {"g_0": [0, 0], "g_1": [1, 1], "g_2_3": [2, 3],
                "g_4p": [4, 33]}
NIGHTS_RANGES = {"n_1": [1, 1], "n_2": [2, 2], "n_3": [3, 3],
                 "n_4_6": [4, 6], "n_7p": [7, 30]}


def _check_spec_sync() -> None:
    """Fail loudly if the embedded public edges drift from the live spec
    (no-op in the bundle, where the spec file is not shipped)."""
    try:
        from spec import load_spec  # noqa: PLC0415
        s = load_spec()
    except Exception:
        return
    edges = [int(s[k]) for k in ("rc_e4", "rc_e5", "rc_e6",
                                 "rc_e7", "rc_e9", "rc_e13")]
    if edges != [4, 5, 6, 7, 9, 13]:
        raise ValueError("distill rc bin edges drifted from the spec")
    if RC_BIN_RANGES["rc_13p"][1] != int(s["rc_13p_draw_hi"]):
        raise ValueError("distill rc_13p draw range drifted from the spec")
    if int(s["nights_e4"]) != 4 or int(s["nights_e7"]) != 7:
        raise ValueError("distill nights bin edges drifted from the spec")
    if int(s["gap_e2"]) != 2 or int(s["gap_e4"]) != 4:
        raise ValueError("distill gap bin edges drifted from the spec")
    if WINDOW != {"start_day": int(s["window_start_day"]),
                  "end_day": int(s["window_end_day"])}:
        raise ValueError("distill window drifted from the spec")


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def _policy_version() -> str:
    """The current policy version (bundle fallback: unknown)."""
    try:
        import yaml  # noqa: PLC0415
        return str(yaml.safe_load(
            (ROOT / "config" / "release_policy.yaml").read_text()
        )["policy_version"])
    except Exception:
        return "unknown"


def _released_artifact(name: str, columns: list[str]) -> list[dict] | None:
    """Rows of a policy-filtered release artifact.

    Preferred source: the latest release/<export_id>/<name>.csv (or
    <name>.csv at the repo root — the bundle layout) — but ONLY if that
    export was made under the CURRENT policy version: a release CSV from an
    older policy (older suppression thresholds, older spec) must not feed a
    newer genspec. On a fresh enclave run `make distill` precedes
    `make export`, so as a fallback the SAME rows are read from the mart
    with the policy's own suppression thresholds applied — byte-for-byte
    what release.py will export."""
    try:
        import yaml  # enclave-side only
        _pol_now = yaml.safe_load(
            (ROOT / "config" / "release_policy.yaml").read_text()
        )["policy_version"]
    except Exception:
        _pol_now = None
    candidates = sorted(ROOT.glob(f"release/*/{name}.csv"))
    if (ROOT / f"{name}.csv").exists():
        candidates.append(ROOT / f"{name}.csv")
    for cand in reversed(candidates):
        try:
            _man = json.loads(
                (cand.parent / "manifest.json").read_text())
            if _pol_now is None or \
                    _man.get("policy_version") == _pol_now:
                with open(cand, newline="") as f:
                    return [dict(r) for r in csv.DictReader(f)]
        except Exception:
            continue
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
    try:
        from spec import load_spec  # noqa: PLC0415  (absent in the bundle)
        spec_now = load_spec()
    except Exception:
        spec_now = {}
    rel_p = OUT / "dp_release.json"
    if not rel_p.exists():
        print("ERROR: run `make lean-release` first", file=sys.stderr)
        return 1
    rel_sha = hashlib.sha256(rel_p.read_bytes()).hexdigest()
    rel = json.loads(rel_p.read_text())
    rc = {c["cell"]: int(c["noised_count"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["noised_count"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in AUX_NEEDED:
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1

    trans = aux["country_transition"]

    def trans_row(r: str) -> list[dict]:
        """DP-released row plus the PUBLIC return-boost prior
        (derived_from_data: false): the DIAGONAL gets a boost equal to
        max(1, sum of the released row) — intra-country tier moves dominate
        the real pattern (38.07% of trips revisit a city; contiguous legs),
        and the boost is strong enough to be robust to per-draw DP noise. The
        second-order part of return behavior (going BACK where you came
        from) cannot live in a first-order matrix: the generator applies it
        per user from the released return_rate bands."""
        row = {n: max(0, int(trans.get(f"{r}__{n}", 0))) for n in CT_CELLS}
        row[r] += max(1, sum(row.values()))
        return [{"name": n, "w": row[n]} for n in CT_CELLS]

    cities_rows = _released_artifact(
        "topk_cities",
        ["city_id", "hotel_country", "hotel_ct", "n_users", "n_res"])
    corridors_rows = _released_artifact(
        "topk_city_transitions",
        ["prev_city", "next_city", "hotel_ct", "n_users", "n_trans"])
    moments_rows = _released_artifact(
        "gap_moment_stats",
        ["gap_class", "n_gaps", "mean_days", "median_days", "sd_days"])
    if cities_rows is None or corridors_rows is None or moments_rows is None:
        print("ERROR: no release artifacts (release/<id>/*.csv) and no "
              "database to policy-filter — run `make export` (or `make dbt`) "
              "first", file=sys.stderr)
        return 1
    cities = [{"city_id": int(r["city_id"]), "country": r["hotel_country"],
               "ct": r["hotel_ct"], "w": int(r["n_res"])}
              for r in cities_rows]
    corridors = [{"a_city": int(r["prev_city"]), "b_city": int(r["next_city"]),
                  "ct": r["hotel_ct"], "n_users": int(r["n_users"]),
                  "n_trans": int(r["n_trans"])}
                 for r in corridors_rows]
    gap_moments = [{"gap_class": r["gap_class"],
                    "mean_days": float(r["mean_days"]),
                    "sd_days": float(r["sd_days"])} for r in moments_rows]

    genspec = {
        "genspec_version": "1.4.0",
        "kind": "restricted generative specification (no executable code)",
        # the genspec reflects the CURRENT spec structure; the release it
        # was distilled from may have been drawn under an older spec
        # version (the grids/eps are unchanged between 1.0.0 and 1.1.0 —
        # only generator draw ranges and tolerances moved)
        "spec_version": str(spec_now.get("spec_version", "1.4.0")),
        "source": {
            "dp_release_sha256": rel_sha,
            "release_spec_version": str(rel.get("spec_version", "1.0.0")),
            "policy_version": str(_policy_version()),
            "privacy": "all weights below are DP-released counts (or released "
                       "+ a declared public prior); the spec is "
                       "post-processing of the release",
        },
        "n_users_hint": sum(max(0, rc.get(n, 0)) for n in RC_BINS),
        "res_count": {"cells": cells_of(rc, RC_BINS)},
        "country_mix": {"cells": cells_of(aux["country"], CT_CELLS)},
        "transitions": {"rows": [{"given": r, "cells": trans_row(r)}
                                 for r in CT_CELLS]},
        "nights": {"cells": cells_of(aux["nights"], NIGHTS_CELLS)},
        "gap": {"cells": cells_of(aux["gap"], GAP_CELLS)},
        "month_window": {"cells": cells_of(aux["month_window"],
                                           MONTH_WINDOW_CELLS)},
        "device": {"cells": cells_of(aux["device"], DEVICE_CELLS)},
        "affiliate": {"cells": cells_of(aux["affiliate"], AFF_CELLS)},
        "booker": {"cells": cells_of(aux["booker"], BOOKER_CELLS)},
        "return_rate": {"cells": cells_of(aux["return_rate"], RR_BANDS)},
        "repeat_struct": {"cells": cells_of(aux["repeat_struct"],
                                            REPEAT_STRUCT_CELLS)},
        "trips_per_user": {"cells": cells_of(aux["trips_per_user"],
                                             TRIPS_CELLS)},
        "device_persist": {"cells": cells_of(aux["device_persist"],
                                              DEVICE_PERSIST_CELLS)},
        "affiliate_persist": {"cells": cells_of(aux["affiliate_persist"],
                                                AFF_PERSIST_CELLS)},
        "countries_per_trip": {"cells": cells_of(aux["countries_per_trip"],
                                                 CPT_CELLS)},
        "device_plan": {"cells": cells_of(aux["device_plan"],
                                          DEVICE_PLAN_CELLS)},
        "domestic_start": {"cells": cells_of(aux["domestic_start"],
                                             DOMESTIC_START_CELLS)},
        "cities": cities,
        "corridors": corridors,
        "gap_moments": gap_moments,
        "priors": [
            {"name": "return_boost", "derived_from_data": False,
             "rule": "transition row r: diagonal cell r += max(1, row sum) "
                     "(intra-tier persistence); generator: per trip, a repeat "
                     "decision at the return_rate band midpoint; a repeat "
                     "trip's STRUCTURE is drawn from the repeat_struct cells "
                     "— r_ext*/r_ret*/r_both render consecutive same-city "
                     "stay-extension chains at the current city and/or hops "
                     "back to the previous city (RETURN-FIRST: decided "
                     "before the tier walk, trip-level like the releases — "
                     "second-order, outside the first-order matrix)"},
            {"name": "uniform_dow", "derived_from_data": False,
             "rule": "day-of-week is NOT released (v1.2.0 dropped the near-"
                     "uniform dow histogram; measured spread over 7 is "
                     "2.08e6-scale noise): weekday is a uniform public "
                     "prior; the trip start day is drawn from the DP "
                     "month-of-window cells (day-of-month uniform 1..28)"},
            {"name": "multi_trip", "derived_from_data": False,
             "rule": "trips per user from the DP trips_per_user cells "
                     "(spec 1.3.0; real 91.1/7.4/1.5% for 1/2/3+); a "
                     "user's reservations split near-equally across their "
                     "trips (measured: both trips >= 4 legs in 99.6% of "
                     "2-trip users); the inter-trip gap is uniform over "
                     "[1, 94] days (median-matched to the measured 47, "
                     "IQR 4-117); utrip_id = '<user>_<k>'"},
            {"name": "channel_persistence", "derived_from_data": False,
             "rule": "per-trip device / affiliate CLASS PLANS from the DP "
                     "device_persist / affiliate_persist cells (spec "
                     "1.3.0): k distinct classes laid over drawn segments "
                     "— real trips are 70.7% single-device and keep the "
                     "previous leg's affiliate tier on 75.4% of legs; "
                     "per-leg marginal draws hide this structure"},
            {"name": "within_bin_decay", "derived_from_data": False,
             "rule": "draws inside multi-width public bins are NOT uniform: "
                     "real within-bin mass decays geometrically toward the "
                     "upper edge (measured per-step ratios: rc bins ~0.7, "
                     "nights ~0.45, gaps ~0.5 — e.g. rc_9_12 real shares "
                     "9:10:11:12 = 2.67:1.63:1.03:0.66%); the generator "
                     "draws from integer weights decaying at those per-"
                     "family ratios (a structural prior; uniform draws "
                     "inflated the records/user mean and the 10-12 leg "
                     "mass)"},
        ],
        "window": WINDOW,
        "rc_bin_ranges": RC_BIN_RANGES,
        "gap_ranges_d": GAP_RANGES_D,
        "nights_ranges": NIGHTS_RANGES,
    }

    # provenance-skip: the DP draws are non-reproducible, so a genspec keyed
    # to the live release hash is authoritative — do not rewrite it. The
    # skip ALSO requires the recorded spec version to match, so a spec bump
    # (e.g. a narrowed draw range) always regenerates the genspec.
    gs_path = OUT / "genspec.json"
    if gs_path.exists():
        try:
            old = json.loads(gs_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if (old.get("source", {}).get("dp_release_sha256") == rel_sha
                and old.get("spec_version") == genspec["spec_version"]
                and old.get("source", {}).get("policy_version")
                    == genspec["source"]["policy_version"]):
            print(f"distill: genspec.json already keyed to dp_release "
                  f"{rel_sha[:12]} at spec {genspec['spec_version']} — "
                  f"reusing (provenance-skip)")
            return _fidelity(old)
    gs_path.write_text(json.dumps(genspec, indent=2))
    print(f"distill: wrote lean/out/genspec.json "
          f"({len(cities)} cities, {len(corridors)} city corridors, "
          f"n_users_hint={genspec['n_users_hint']})")
    return _fidelity(genspec)


def _fidelity(genspec: dict) -> int:
    """ENCLAVE-ONLY: the distilled first-order student vs the NextCountry
    teacher vs the return-to-previous baseline, on held-out user sequences
    (the same user-id-hash split train_minimodels.py froze)."""
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    if DB.exists():
        import duckdb  # enclave-side only
        from train_minimodels import is_train_user  # noqa: PLC0415
        con = duckdb.connect(str(DB), read_only=True)
        rows = con.execute(
            "select userid, prev_ct, next_ct from feat_transitions "
            "order by userid, utrip, from_leg").fetchall()
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
        student = {r: max(CT_CELLS, key=lambda n: blended[r][n])
                   for r in CT_CELLS}
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
                        teacher_acc = (meta.get("next_country") or {}).get(
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

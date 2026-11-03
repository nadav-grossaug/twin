#!/usr/bin/env python3
"""Synthetic booking reservations: generate / validate / replicate / certify.

generate  — invoke the standalone Lean generator (`lake exe generate`) from
            the distilled genspec (or, with --bundle DIR, the bundle's own
            rebuilt generator and genspec); write meta.json with full
            provenance and DECLARED generation gaps (fields_not_modelled).
validate  — re-ingest the synthetic CSV through the PRODUCTION dbt pipeline
            into a scratch DuckDB, verify the planted motifs are recalled
            (repeat-city return, multi-country tour, long stay; recall must
            be 1.0), check the structural invariants (bad dates, overlapping
            presence, public window, the single-trip pattern), GOF-check the
            synthetic reservations-per-user, country-tier, nights and gap
            mixes against the genspec, and run the Lean deciders on the
            re-exported observations. Exit codes: 0 pass / 2 soft tolerance
            / 1 hard (schema, planted-pattern or Lean disagreement).
replicates— the repeat-city share is a stochastic property of the GENERATOR,
            so it is judged over replicate runs, not one sample (band
            verdict: fraction in band + margin over the lower edge).
certify   — emit GenCert.lean asserting (by decide, kernel-checked) that the
            planted plan structures satisfy the formal motif definitions and
            that every planned user's reservation count falls in its public
            bin; compile it and record certificate.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN = ROOT / "lean"
GENSPEC = LEAN / "out" / "genspec.json"
GENERATE_BIN = LEAN / ".lake" / "build" / "bin" / "generate"
CHECKS_BIN = LEAN / ".lake" / "build" / "bin" / "checks"
RC_BINS = ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]
CT_CELLS = ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia",
            "Borginia", "Kangan", "Bozatta", "Alvonia",
            "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan",
            "Rolisica", "Kasnia", "ct_other"]
NIGHTS_CELLS = ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]
GAP_CELLS = ["g_0", "g_1", "g_2_3", "g_4p"]
# the synth CSV carries the FULL public input schema (9 columns)
CSV_COLUMNS = ["user_id", "checkin", "checkout", "city_id", "device_class",
               "affiliate_id", "booker_country", "hotel_country", "utrip_id"]
MOTIF_KEYS = ("returns", "corridors", "tours", "longstays", "assoc")
RR_MIDPOINTS = {"rr_0_19": 10, "rr_20_39": 30, "rr_40_59": 50,
                "rr_60_79": 70, "rr_80_100": 90}

sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec, spec_version  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ct_of(country: str) -> str:
    """Public hotel-country tier (frozen in the contract; mirrors the dbt
    seed country_tiers.csv: the 15 named countries, everything else ct_other)."""
    return country if country in CT_CELLS else "ct_other"


def gap_cell(gap_days: int) -> str:
    if gap_days == 0:
        return "g_0"
    if gap_days == 1:
        return "g_1"
    if gap_days <= 3:
        return "g_2_3"
    return "g_4p"


def _lean_env() -> dict:
    """PATH with the shared Lean store's elan first (LEAN_HOME policy),
    falling back to a user-local elan."""
    lean_home = Path(os.environ.get("LEAN_HOME", "/opt/data/lean"))
    env = dict(os.environ)
    prefix = []
    if lean_home.is_dir():
        env.setdefault("ELAN_HOME", str(lean_home / "elan"))
        env.setdefault("XDG_CACHE_HOME", str(lean_home / "cache"))
        prefix.append(str(lean_home / "elan" / "bin"))
    prefix.append(str(Path.home() / ".elan" / "bin"))
    env["PATH"] = ":".join(prefix + [env.get("PATH", "")])
    return env


def cmd_generate(args) -> int:
    base = Path(args.bundle).resolve() if args.bundle else ROOT
    lean_dir = base / "lean"
    genspec = lean_dir / "out" / "genspec.json"
    gen_bin = lean_dir / ".lake" / "build" / "bin" / "generate"
    if not genspec.exists():
        hint = ("run `python3 python/distill.py` inside the bundle first"
                if args.bundle else "run `make distill` first")
        print(f"ERROR: {genspec} missing — {hint}", file=sys.stderr)
        return 1
    if not gen_bin.exists():
        hint = ("run `./build_from_source.sh` inside the bundle first"
                if args.bundle else "run `make lean-build` first")
        print(f"ERROR: {gen_bin} missing — {hint}", file=sys.stderr)
        return 1
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(gen_bin), str(genspec), str(args.users), str(out)],
        cwd=str(lean_dir))
    if r.returncode != 0:
        return r.returncode
    spec = json.loads(genspec.read_text())
    meta = {
        "generator": "lake exe generate (Lean; genspec-driven; every draw "
                      "through the verified sampler sampleCellPMF)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "genspec_sha256": sha(genspec),
        "dp_release_sha256": spec["source"]["dp_release_sha256"],
        "spec_version": str(spec.get("spec_version", spec_version())),
        "n_users": args.users,
        "bundle": str(base) if args.bundle else None,
        "released_fact_sources": {
            "res_count": "dp_release.json cells",
            "country_mix": "aux country",
            "transitions": "aux country_transition (+ declared return_boost "
                           "diagonal prior)",
            "return_rate": "aux return_rate (trip-level repeat decision)",
            "repeat_struct": "aux repeat_struct (the anatomy of repeat trips: "
                             "consecutive stay-extension chains vs return "
                             "hops; spec 1.2.0)",
            "trips_per_user": "aux trips_per_user (multi-trip rendering; "
                              "spec 1.3.0)",
            "device_persist": "aux device_persist (per-trip device plans; "
                              "spec 1.3.0)",
            "affiliate_persist": "aux affiliate_persist (per-trip affiliate "
                                 "plans; spec 1.3.0)",
            "nights": "aux nights",
            "gap": "aux gap (+ suppressed gap moments)",
            "month_window": "aux month_window (start-month placement on the "
                            "real observation ramp; spec 1.2.0)",
            "device": "aux device",
            "affiliate": "aux affiliate",
            "booker": "aux booker",
            "cities": "topk_cities release artifact (suppressed)",
            "corridors": "topk_city_transitions release artifact (suppressed)",
        },
        "assumptions": {
            "fields_not_modelled": [
                "within-tier city choice beyond the released top-k lists "
                "(cities outside topk_cities get synthetic counter ids)",
                "day-of-week placement (uniform_dow public prior; dow is NOT "
                "released — measured near-uniform)",
                "sub-day time (dates only, day granularity; no time of day)",
                "within-trip affiliate order beyond the per-trip tier-count "
                "plans (segment lengths and tier assignment are drawn, not "
                "released)",
                "correlation structure between legs beyond the released "
                "associations (chi2 pairs are re-checked in Lean, not "
                "enforced by the generator)",
            ],
            "combination_rules": [
                "next country tier = verified draw from the blended "
                "transition row (return_boost diagonal prior); per trip, one "
                "repeat decision at the drawn return-rate band probability; "
                "GIVEN a repeat trip the anatomy is drawn from the "
                "repeat_struct cells — a chain of consecutive same-city "
                "stay-extension hops and/or hops back to the previous city "
                "(second-order, trip-level like the releases)",
                "trips per user from the trips_per_user cells; a user's "
                "reservations split near-equally across trips; inter-trip "
                "gap uniform [1, 94] days; utrip_id = '<user>_<k>'",
                "per-trip device / affiliate plans from the "
                "device_persist / affiliate_persist cells: k distinct "
                "classes over drawn segments",
                "nights per leg = verified decay-weighted draw within the "
                "public nights range (within_bin_decay prior); checkout = "
                "checkin + nights",
                "gap days = verified decay-weighted draw within the public "
                "gap-class range (within_bin_decay prior); "
                "checkin(i+1) = checkout(i) + gap (temporal flow by "
                "construction; 99.1% of real legs are contiguous)",
                "trip start day = month-of-window drawn from the DP "
                "month_window histogram (the real observation ramp), "
                "day-of-month uniform in 1..28",
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed FFI, "
                                    "consistent with non-reproducible DP draws)",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/booking_synth.csv + plan.json + meta.json")
    return 0


# ---------------------------------------------------------------------------
# Validation pieces (module-level + pure, so tests can drive them against an
# in-memory DuckDB without any pipeline artifacts — the snap pattern).
# ---------------------------------------------------------------------------

# user-count window the fidelity tolerances are validated on. Below it the
# GOF mixes cannot be met by construction (planted users distort a few
# hundred-user population), so they are recorded as unjudged, not failed.
VALIDATED_MIN_USERS = 200
GOF_TV_TOL = 0.25
# sampling allowance: with few rows the observed mix is noisy by construction
# (expected TV ~ sqrt((k-1)/(2 pi n))); three standard errors, matching the
# lean4_snap calibration.
GOF_TV_TOL_SIGMA = 3


def mix_tv(want: dict[str, int], got: dict[str, int], grid: list[str]) -> float:
    tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
    return 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                     for c in grid)


def check_mixes(gs: dict, con, notes: dict, soft: list,
                n_rows: int = 10**9, judged: bool = True) -> None:
    """GOF (soft): synthetic mixes vs the released genspec weights (TV),
    with the sampling tolerance widened for small runs. A cell the spec
    expects to be populated (>= 3 expected rows) must not be empty."""
    def flag(msg: str) -> None:
        if judged:
            soft.append(msg)
        else:
            notes.setdefault("unjudged_small_n", []).append(msg)

    def tv_tol(k: int) -> float:
        return max(GOF_TV_TOL,
                   GOF_TV_TOL_SIGMA * math.sqrt((k - 1) / (2 * math.pi * max(1, n_rows))))

    mixes = (
        ("res_count", "rc_bin", "feat_users",
         {c["name"]: c["w"] for c in gs["res_count"]["cells"]}, RC_BINS),
        ("country_mix", "hotel_ct", "feat_reservations",
         {c["name"]: c["w"] for c in gs["country_mix"]["cells"]}, CT_CELLS),
        ("nights", "nights_cell", "stg_reservations",
         {c["name"]: c["w"] for c in gs["nights"]["cells"]}, NIGHTS_CELLS),
    )
    for name, col, rel, want, grid in mixes:
        got = dict(con.execute(
            f"select {col}, count(*) from {rel} group by 1").fetchall())
        notes[f"{name}_tv_vs_genspec"] = round(mix_tv(want, got, grid), 4)
        tol = tv_tol(len(grid))
        notes[f"{name}_tv_tol"] = round(tol, 4)
        if notes[f"{name}_tv_vs_genspec"] > tol:
            flag(f"{name}_tv_vs_genspec = {notes[name + '_tv_vs_genspec']:.3f} "
                 f"> {tol:.2f}")
        # coverage: an expected-populated cell may not be empty
        n_got = sum(got.values())
        for c in grid:
            expected = want.get(c, 0) / max(1, sum(want.values())) * n_got
            if expected >= 3.0 and got.get(c, 0) == 0:
                flag(f"{name} cell {c}: ~{expected:.0f} rows expected, "
                     f"none generated")
    # gap needs the class computed from gap_days (the transitions table has
    # no gap_cell column; the classes are public constants)
    got_gap = {c: n for c, n in con.execute(
        "select case when gap_days = 0 then 'g_0' "
        "when gap_days = 1 then 'g_1' when gap_days <= 3 then 'g_2_3' "
        "else 'g_4p' end, count(*) from feat_transitions group by 1").fetchall()}
    want_gap = {c["name"]: c["w"] for c in gs["gap"]["cells"]}
    notes["gap_tv_vs_genspec"] = round(mix_tv(want_gap, got_gap, GAP_CELLS), 4)
    tol = tv_tol(len(GAP_CELLS))
    notes["gap_tv_tol"] = round(tol, 4)
    if notes["gap_tv_vs_genspec"] > tol:
        flag(f"gap_tv_vs_genspec = {notes['gap_tv_vs_genspec']:.3f} > {tol:.2f}")
    n_got = sum(got_gap.values())
    for c in GAP_CELLS:
        expected = want_gap.get(c, 0) / max(1, sum(want_gap.values())) * n_got
        if expected >= 3.0 and got_gap.get(c, 0) == 0:
            flag(f"gap cell {c}: ~{expected:.0f} rows expected, none generated")


def check_planted(plan: dict, con, notes: dict, hard: list) -> None:
    """Planted motifs recalled by the PRODUCTION motif marts (recall 1.0)."""
    planted = plan.get("planted") or {}
    # repeat-city return: presence in mart_motif_returns IS the observation
    ret = planted.get("return")
    if ret:
        n_ret = con.execute(
            "select count(*) from mart_motif_returns where userid = ?",
            [int(ret["user"])]).fetchone()[0]
        notes["planted_return_recall"] = 1.0 if n_ret else 0.0
        if not n_ret:
            hard.append(f"planted repeat-city return of user {ret['user']} "
                        f"not recalled")
    tour = planted.get("tour")
    if tour:
        n_tour = con.execute(
            "select count(*) from mart_motif_tours where userid=? and is_tour",
            [int(tour["user"])]).fetchone()[0]
        notes["planted_tour_recall"] = 1.0 if n_tour else 0.0
        if not n_tour:
            hard.append(f"planted tour of user {tour['user']} not recalled")
    ls = planted.get("longstay")
    if ls:
        n_ls = con.execute(
            "select count(*) from mart_motif_longstays "
            "where userid=? and is_longstay",
            [int(ls["user"])]).fetchone()[0]
        notes["planted_longstay_recall"] = 1.0 if n_ls else 0.0
        if not n_ls:
            hard.append(f"planted long stay of user {ls['user']} not recalled")


def check_structure(con, spec: dict, notes: dict, hard: list) -> None:
    """Structural invariants: clean dates, no overlapping presence, public
    window, the "<user>_<k>" utrip naming the generator guarantees, and a
    multi-trip rate consistent with the released trips_per_user cells."""
    n_bad = con.execute(
        "select count(*) from stg_reservations where qf_bad_dates").fetchone()[0]
    if n_bad:
        hard.append(f"{n_bad} rows with bad dates (checkout <= checkin)")
    n_overlap = con.execute(
        "select count(*) from stg_reservations where qf_overlap").fetchone()[0]
    notes["overlap_rows"] = int(n_overlap)
    if n_overlap:
        hard.append(f"{n_overlap} rows with overlapping presence "
                    f"(same trip in two stays at once)")
    w0, w1 = int(spec["window_start_day"]), int(spec["window_end_day"])
    n_outside = con.execute(
        "select count(*) from stg_reservations "
        "where checkin < (DATE '1970-01-01' + ?) "
        "or checkin >= (DATE '1970-01-01' + ?)", [w0, w1]).fetchone()[0]
    if n_outside:
        hard.append(f"{n_outside} rows outside the public window")
    # utrip naming: every trip is "<user>_<k>", k a 1-based trip index
    n_bad_utrip = con.execute(
        "select count(*) from stg_reservations where not regexp_matches("
        "utrip, '^' || userid::varchar || '_[0-9]+$')").fetchone()[0]
    if n_bad_utrip:
        hard.append(f"{n_bad_utrip} rows whose utrip_id is not '<user>_<k>'")
    # multi-trip users (v1.3.0): the rate must match the released
    # trips_per_user cells — the release is the ground truth here, not 0
    n_users = con.execute(
        "select count(distinct userid) from stg_reservations").fetchone()[0]
    n_multi = con.execute(
        "select count(*) from (select userid from stg_reservations "
        "group by 1 having count(distinct utrip) > 1)").fetchone()[0]
    notes["multi_trip_users"] = int(n_multi)
    if n_users:
        notes["multi_trip_user_share"] = round(n_multi / n_users, 4)
    # contiguity rate: the temporal-flow statistic (99.11% in the real data)
    contig = con.execute(
        "select count(*) filter (where gap_days = 0), count(*) "
        "from feat_transitions").fetchone()
    if contig and contig[1]:
        notes["contiguous_leg_share"] = round(contig[0] / contig[1], 4)


def cmd_validate(args) -> int:
    synth = Path(args.synth)
    csvp = synth / "booking_synth.csv"
    plan_p = synth / "plan.json"
    if not csvp.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    if not GENSPEC.exists():
        print("ERROR: run `make distill` first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    scratch = (synth / "scratch.duckdb").resolve()
    if scratch.exists():
        scratch.unlink()
    # re-ingest through the PRODUCTION dbt pipeline (same models, scratch DB)
    env = dict(os.environ,
               BOOKING_CSV=str(csvp.resolve()),
               ENCLAVE_DB=str(scratch),
               DBT_PROFILES_DIR=".")
    r = subprocess.run(["dbt", "build", "--exclude", "ml_trip_scores"],
                       cwd=str(ROOT / "dbt"), env=env,
                       capture_output=True, text=True)
    hard, soft, notes = [], [], {}
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-15:])
        hard.append(f"dbt re-ingestion failed:\n{tail}")
    else:
        import duckdb
        con = duckdb.connect(str(scratch), read_only=True)
        spec = load_spec()
        n_users = len(plan.get("users", []))
        notes["n_users"] = n_users
        notes["validated_size_window"] = [VALIDATED_MIN_USERS, None]
        small = n_users < VALIDATED_MIN_USERS
        check_planted(plan, con, notes, hard)
        check_structure(con, spec, notes, hard)
        # GOF (soft) vs the released genspec weights
        gs = json.loads(GENSPEC.read_text())
        n_rows = con.execute(
            "select count(*) from stg_reservations").fetchone()[0]
        check_mixes(gs, con, notes, soft, n_rows=n_rows, judged=not small)
        if small:
            notes["size_note"] = (f"n_users={n_users} below the validated "
                                  f"fidelity window ({VALIDATED_MIN_USERS}); "
                                  f"GOF mixes recorded as unjudged")
        con.close()
        # Lean deciders on the re-ingested observations
        leandir = synth / "leancheck"
        (leandir / "out").mkdir(parents=True, exist_ok=True)
        from export_lean_data import export as export_lean
        export_lean(scratch, leandir / "data")
        rc = subprocess.run([str(CHECKS_BIN)], cwd=str(leandir),
                            capture_output=True, text=True)
        if rc.returncode != 0:
            hard.append(f"lean checks failed: "
                         f"{rc.stderr[-400:] or rc.stdout[-400:]}")
        else:
            res = json.loads((leandir / "out" / "lean_results.json").read_text())
            disagree = {
                k: sum(1 for x in res[k] if not x.get("agree", True))
                for k in MOTIF_KEYS}
            disagree["trips"] = sum(
                1 for x in res["trips"] if not x.get("agree", True))
            notes["lean_disagreements"] = disagree
            if any(disagree.values()):
                hard.append(f"SQL<->Lean disagreement on synth: {disagree}")
    verdict = {"hard_failures": hard, "soft_failures": soft, "notes": notes,
               "csv_sha256": sha(csvp)}
    (synth / "validation.json").write_text(json.dumps(verdict, indent=2))
    if hard:
        print("validate: HARD FAIL\n - " + "\n - ".join(hard), file=sys.stderr)
        return 1
    if soft:
        print("validate: soft tolerance\n - " + "\n - ".join(soft))
        return 2
    print(f"validate: PASS ({json.dumps(notes)})")
    return 0


# ---------------------------------------------------------------------------
# replicates: the repeat-city share and res-count mix are judged over
# replicate runs, not one sample (snap's closure pattern).
# ---------------------------------------------------------------------------
REPLICATE_MIN_FRACTION = 0.85
REPLICATE_MIN_MARGIN = 0.02
REPLICATE_MAX_RC_TV = 0.20


def band_verdict(values: list[float], released: float, tol: float) -> dict:
    """Pass/fail on the distribution of a per-run statistic against the
    released band [released - tol, released + tol]: at least MIN_FRACTION of
    runs inside the band AND the mean at least MIN_MARGIN above the lower
    edge (a mean hugging the lower edge fails even with all runs inside)."""
    n = len(values)
    lower, upper = released - tol, released + tol
    mean = sum(values) / n
    n_in = sum(1 for v in values if lower <= v <= upper)
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    margin = mean - lower
    return {
        "n_runs": n, "mean": round(mean, 4), "min": round(min(values), 4),
        "max": round(max(values), 4), "std": round(std, 4),
        "band": [round(lower, 4), round(upper, 4)],
        "n_in_band": n_in, "fraction_in_band": round(n_in / n, 4),
        "margin_over_lower_edge": round(margin, 4),
        "rule": f"fraction_in_band >= {REPLICATE_MIN_FRACTION} and margin >= "
                f"{REPLICATE_MIN_MARGIN}",
        "passed": bool(n_in / n >= REPLICATE_MIN_FRACTION
                       and margin >= REPLICATE_MIN_MARGIN),
    }


def released_repeat_share(gs: dict) -> float:
    """The released mean repeat-city share: return_rate band weights times
    their public midpoints (band -> midpoint is the declared rule)."""
    cells = gs["return_rate"]["cells"]
    tw = sum(max(0, c["w"]) for c in cells)
    if not tw:
        return 0.0
    return sum(max(0, c["w"]) * RR_MIDPOINTS.get(c["name"], 50)
               for c in cells) / tw / 100.0


def csv_trip_stats(csv_path: Path) -> dict:
    """Per-run statistics straight from the rendered CSV: the share of trips
    with a repeat city and the reservations-per-user bin mix (public bins)."""
    res_per_user: dict[int, int] = {}
    trip_cities: dict[str, list[int]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{csv_path.name}: header differs from the public "
                             f"schema (got {reader.fieldnames})")
        for r in reader:
            u = int(r["user_id"])
            res_per_user[u] = res_per_user.get(u, 0) + 1
            trip_cities.setdefault(r["utrip_id"], []).append(int(r["city_id"]))

    def rc_bin(n: int) -> str:
        if n <= 3: return "rc_1_3"
        if n == 4: return "rc_4"
        if n == 5: return "rc_5"
        if n == 6: return "rc_6"
        if n <= 8: return "rc_7_8"
        if n <= 12: return "rc_9_12"
        return "rc_13p"

    rc_mix: dict[str, int] = {}
    for n in res_per_user.values():
        rc_mix[rc_bin(n)] = rc_mix.get(rc_bin(n), 0) + 1
    trips = trip_cities.values()
    repeat = sum(1 for cs in trips if len(set(cs)) < len(cs))
    share = repeat / len(trips) if trips else 0.0
    return {"repeat_share": share, "rc_mix": rc_mix, "n_trips": len(trips)}


def cmd_replicates(args) -> int:
    import shutil
    base = Path(args.bundle).resolve() if args.bundle else ROOT
    genspec = base / "lean" / "out" / "genspec.json"
    gen_bin = base / "lean" / ".lake" / "build" / "bin" / "generate"
    if not genspec.exists() or not gen_bin.exists():
        print("ERROR: genspec or generator missing — run the synth flow first",
              file=sys.stderr)
        return 1
    gs = json.loads(genspec.read_text())
    released = released_repeat_share(gs)
    # public soft tolerance for the repeat-share band (percent); spec var
    # optional so the command degrades gracefully before the var lands
    tol = int(load_spec().get("synth_repeat_tol_pct", 15)) / 100.0
    out = Path(args.synth).resolve()
    scratch = out / "replicates"
    shares, rc_tvs = [], []
    for _ in range(args.runs):
        shutil.rmtree(scratch, ignore_errors=True)
        r = subprocess.run([str(gen_bin), str(genspec), str(args.users), str(scratch)],
                           cwd=str(base / "lean"), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"replicates: generator failed: {r.stderr[-300:]}",
                  file=sys.stderr)
            return 1
        stats = csv_trip_stats(scratch / "booking_synth.csv")
        shares.append(stats["repeat_share"])
        want = {c["name"]: c["w"] for c in gs["res_count"]["cells"]}
        rc_tvs.append(mix_tv(want, stats["rc_mix"], RC_BINS))
    shutil.rmtree(scratch, ignore_errors=True)
    verdict = band_verdict(shares, released, tol)
    verdict["released_repeat_share"] = round(released, 4)
    verdict["values"] = [round(v, 4) for v in shares]
    verdict["res_count_tv"] = {"mean": round(sum(rc_tvs) / len(rc_tvs), 4),
                               "max": round(max(rc_tvs), 4),
                               "max_allowed": REPLICATE_MAX_RC_TV}
    verdict["rc_tv_passed"] = bool(
        sum(rc_tvs) / len(rc_tvs) <= REPLICATE_MAX_RC_TV)
    verdict["repeat_passed"] = verdict["passed"]
    verdict["passed"] = bool(verdict["passed"] and verdict["rc_tv_passed"])
    (out / "replicates.json").write_text(json.dumps(verdict, indent=2))
    tag = "PASS" if verdict["passed"] else "soft tolerance"
    print(f"replicates: {tag} — repeat-city share mean {verdict['mean']} "
          f"({verdict['n_in_band']}/{verdict['n_runs']} in band "
          f"{verdict['band']}, margin {verdict['margin_over_lower_edge']}); "
          f"res_count TV mean {verdict['res_count_tv']['mean']}")
    return 0 if verdict["passed"] else 2


def cmd_certify(args) -> int:
    synth = Path(args.synth)
    plan = json.loads((synth / "plan.json").read_text())
    gs = json.loads(GENSPEC.read_text())
    planted = plan["planted"]
    ret, tour, ls = planted["return"], planted["tour"], planted["longstay"]
    ret_cities = [str(c) for c in ret["cities"]]
    tour_days = [int(d) for d in tour["days"]]
    # the tour predicate judges ALL countries touched, in visit order
    tour_cs = [str(c) for c in tour["countries"]]
    ls_nights = [int(n) for n in ls["nights"]]
    ranges = gs["rc_bin_ranges"]
    users = plan["users"][: int(args.n)]
    n = len(users)

    def bin_of(count: int) -> tuple[int, int]:
        for lo, hi in ranges.values():
            if int(lo) <= count <= int(hi):
                return int(lo), int(hi)
        raise SystemExit(f"certify: planned count {count} outside every "
                         f"public rc bin")

    triples = ", ".join(
        "({0}, {1}, {2})".format(int(u["n_res"]), *bin_of(int(u["n_res"])))
        for u in users)
    cities = "[" + ", ".join(f'"{c}"' for c in ret_cities) + "]"
    countries = "[" + ", ".join(f'"{c}"' for c in tour_cs) + "]"
    cert = f"""/- AUTO-GENERATED by scripts/synth.py certify — kernel-checked certificate
that the emitted plan's structures satisfy the formal spec. -/
import EnclaveTrip.Motifs

open EnclaveTrip

def plantedReturnCities : List String := {cities}

theorem planted_is_return :
    HasRepeatCity plantedReturnCities := by decide

def plantedTourDays : List Nat := {tour_days}

def plantedTourCountries : List String := {countries}

theorem planted_is_tour :
    IsTour plantedTourDays plantedTourCountries := by decide

def plantedLongstayNights : List Nat := {ls_nights}

theorem planted_is_longstay :
    AnyLongStay plantedLongstayNights := by decide

/-- (planned reservation count, public bin lower edge, public bin upper edge) -/
def plannedCounts : List (Nat × Nat × Nat) := [{triples}]

theorem planned_counts_in_bins :
    plannedCounts.all
      (fun p => decide (p.2.1 ≤ p.1) && decide (p.1 ≤ p.2.2)) = true := by
  decide
"""
    (LEAN / "GenCert.lean").write_text(cert)
    r = subprocess.run(["lake", "env", "lean", "GenCert.lean"], cwd=str(LEAN),
                       capture_output=True, text=True, env=_lean_env())
    ok = r.returncode == 0
    (synth / "certificate.json").write_text(json.dumps({
        "proved": ok,
        "kernel_only": True,
        "theorems": ["planted_is_return", "planted_is_tour",
                     "planted_is_longstay", "planned_counts_in_bins"],
        "n_users_checked": n,
        "axioms": "decide-based (propext only); no sorry, no native_decide",
        "stderr": r.stderr[-400:] if not ok else "",
    }, indent=2))
    print(f"certify: {'PROVED' if ok else 'FAILED'} -> {synth}/certificate.json")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--users", type=int, default=2000)
    g.add_argument("--out", default="synth")
    g.add_argument("--bundle", default=None,
                   help="use DIR/lean/.lake/build/bin/generate and "
                        "DIR/lean/out/genspec.json instead of the repo's")
    v = sub.add_parser("validate")
    v.add_argument("--synth", default="synth")
    rp = sub.add_parser("replicates")
    rp.add_argument("--users", type=int, default=2000)
    rp.add_argument("--synth", default="synth")
    rp.add_argument("--runs", type=int, default=8)
    rp.add_argument("--bundle", default=None)
    c = sub.add_parser("certify")
    c.add_argument("--synth", default="synth")
    c.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    return {"generate": cmd_generate, "validate": cmd_validate,
            "replicates": cmd_replicates, "certify": cmd_certify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

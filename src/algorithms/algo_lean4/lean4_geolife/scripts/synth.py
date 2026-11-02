#!/usr/bin/env python3
"""Synthetic GeoLife trajectories: generate / manipulate / validate / certify.

generate   — invoke the standalone Lean generator (`lake exe generate`) from
             the distilled genspec (or, with --bundle DIR, the bundle's own
             rebuilt generator and genspec); the generator emits a NEUTRAL
             local frame (degrees around 0°N 0°E, isotropic), so generate
             AUTOMATICALLY runs the manipulate step: the raw frame TSV is
             kept as geolife_synth_frame.tsv and the final Tokyo-placed TSV
             becomes geolife_synth.tsv. Writes meta.json with full provenance
             and DECLARED generation gaps (fields_not_modelled).
manipulate — the Tokyo juxtaposition on its own: place each user's neutral
             frame on a public Tokyo-metro district centroid (deterministic
             per user id), undoing the isotropic frame's longitude scale with
             1/cos(lat). Hard-asserts BEFORE writing: zero points inside the
             Beijing exclusion bbox, all points inside the Tokyo target bbox
             (both are public spec vars).
validate   — re-ingest the synthetic TSV through the PRODUCTION dbt pipeline
             into a scratch DuckDB, verify the planted motifs are recalled
             (commute days, tour day, revisit beacon; recall must be 1.0),
             check the structural invariants (bboxes, time ordering, public
             mode labels, altitude sentinel, path realism: no straight-line
             runs, no ±90° turn atoms), GOF-check the synthetic motif and
             mode mixes against the released weights, compare the re-detected
             GPS-noise sd with the released moments, and run the Lean
             deciders on the re-exported observations. Exit codes: 0 pass /
             2 soft tolerance / 1 hard (schema, planted-pattern, structural
             or Lean disagreement).
certify    — emit GenCert.lean asserting (by decide, kernel-checked) that a
             K-point sample of the rendered rows is sane (timestamps strictly
             increasing per trajectory after exact-duplicate collapse — the
             same collapse production ingestion applies; every raw mode label
             maps to a public mode class through the formal mode map) and
             that the emitted plan satisfies the formal motif spec (the
             planted beacon days ARE a revisit beacon; the planted commuter's
             days are all commute days); compile it and record
             certificate.json.
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
from calendar import timegm
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN = ROOT / "lean"
GENSPEC = LEAN / "out" / "genspec.json"
GENERATE_BIN = LEAN / ".lake" / "build" / "bin" / "generate"
CHECKS_BIN = LEAN / ".lake" / "build" / "bin" / "checks"
SYNTH_TSV = "geolife_synth.tsv"
FRAME_TSV = "geolife_synth_frame.tsv"
DT_FMT = "%Y-%m-%d %H:%M:%S"
# lean_results.json row arrays that carry per-row agree verdicts
LEAN_CHECK_KEYS = ("motif_days", "revisits", "dwell_classes", "hour_bins",
                   "dist_classes", "speed_classes", "spd_bins", "longdwells",
                   "visit_seqs", "assoc")

sys.path.insert(0, str(ROOT / "python"))
from spec import MODE_CLASSES, MOTIF_CLASSES, load_spec, mode_map, spec_version  # noqa: E402

# PUBLIC Tokyo-metro district centroids (open-map geography, no relation to
# any enclave datum). A user's placement is a deterministic function of the
# synthetic user id, so the manipulate step is reproducible.
TOKYO_PLACEMENTS = [
    ("Shinjuku", 35.6938, 139.7034), ("Shibuya", 35.6580, 139.7016),
    ("Ueno", 35.7141, 139.7774), ("Ikebukuro", 35.7295, 139.7109),
    ("Shinagawa", 35.6285, 139.7387), ("Nakano", 35.7074, 139.6638),
    ("Kichijoji", 35.7032, 139.5800), ("Machida", 35.5461, 139.4466),
    ("Kawasaki", 35.5308, 139.7029), ("Funabashi", 35.6947, 139.9826),
    ("Urawa", 35.8617, 139.6455), ("Tachikawa", 35.6939, 139.4138),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ts(dt: str) -> int:
    """Epoch seconds of a 'YYYY-MM-DD HH:MM:SS' TSV timestamp (the local
    naive convention the Lean generator renders)."""
    return timegm(time.strptime(dt, DT_FMT))


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


def _released_route_shape() -> tuple[dict[str, dict] | None, str]:
    """Per-mode rows of the released route_shape artifact: $RELEASE_DIR,
    else the latest release/<export_id>/, else route_shape.csv at the repo
    root (the bundle layout). None when no release carries it (old releases
    predate the route-shape artifacts)."""
    candidates = []
    env = os.environ.get("RELEASE_DIR")
    if env:
        candidates.append(Path(env) / "route_shape.csv")
    else:
        dirs = sorted(d for d in ROOT.glob("release/*") if d.is_dir())
        if dirs:
            candidates.append(dirs[-1] / "route_shape.csv")
    candidates.append(ROOT / "route_shape.csv")
    for p in candidates:
        if p.exists():
            with open(p, newline="") as f:
                return {r["mode_class"]: dict(r)
                        for r in csv.DictReader(f)}, str(p)
    return None, ""


def _bboxes(spec: dict) -> tuple[tuple, tuple]:
    """(Beijing exclusion, Tokyo target) bboxes as (lat_min, lat_max,
    lon_min, lon_max) degrees, from the public centidegree spec vars."""
    excl = tuple(int(spec[k]) / 100.0 for k in
                 ("excl_lat_min_cd", "excl_lat_max_cd",
                  "excl_lon_min_cd", "excl_lon_max_cd"))
    tgt = tuple(int(spec[k]) / 100.0 for k in
                ("tgt_lat_min_cd", "tgt_lat_max_cd",
                 "tgt_lon_min_cd", "tgt_lon_max_cd"))
    return excl, tgt


def _placement_seed(user: str) -> tuple[int, float, float]:
    """Deterministic (district index, lat jitter, lon jitter) for a user id:
    sha256 of the id, so the step is reproducible (python's hash() is
    salted). Jitter stays below 0.01°."""
    h = hashlib.sha256(user.encode()).digest()
    idx = h[0] % len(TOKYO_PLACEMENTS)
    jlat = (h[1] / 255.0 - 0.5) * 0.018
    jlon = (h[2] / 255.0 - 0.5) * 0.018
    return idx, jlat, jlon


def _manipulate(synth: Path, target: str = "tokyo") -> dict:
    """The Tokyo juxtaposition: every user's neutral local frame (degrees
    around 0°N 0°E, isotropic: 1° = 111320 m on both axes) is re-based onto a
    public Tokyo district centroid:

        lat = base_lat + frame_lat
        lon = base_lon + frame_lon / cos(radians(base_lat))

    (the cos correction restores metric distances at Tokyo's latitude).
    Timestamps and every other column are unchanged. The district is the
    user-hash pick, rotated forward deterministically until the user's whole
    frame extent fits the Tokyo target bbox. HARD-ASSERTS before writing:
    zero points inside the Beijing exclusion bbox and all points inside the
    Tokyo target bbox (public spec vars); exits 1 otherwise."""
    if target != "tokyo":
        print(f"ERROR: unknown manipulate target {target!r} (only 'tokyo')",
              file=sys.stderr)
        raise SystemExit(1)
    spec = load_spec()
    excl, tgt = _bboxes(spec)
    frame_p, out_p = synth / FRAME_TSV, synth / SYNTH_TSV
    if not frame_p.exists():
        if not out_p.exists():
            print(f"ERROR: {out_p} missing — run generate first",
                  file=sys.stderr)
            raise SystemExit(1)
        os.replace(out_p, frame_p)   # keep the raw neutral-frame TSV

    # pass 1: each user's frame extent (so the placement provably fits)
    ext: dict[str, list[float]] = {}
    with open(frame_p) as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\n").split("\t")
            lat, lon = float(cols[4]), float(cols[5])
            e = ext.setdefault(cols[0], [lat, lat, lon, lon])
            e[0], e[1] = min(e[0], lat), max(e[1], lat)
            e[2], e[3] = min(e[2], lon), max(e[3], lon)

    margin = 0.002
    placements: dict[str, tuple[str, float, float]] = {}
    for user, (lat0, lat1, lon0, lon1) in sorted(ext.items()):
        idx, jlat, jlon = _placement_seed(user)
        for k in range(len(TOKYO_PLACEMENTS)):
            name, dlat, dlon = TOKYO_PLACEMENTS[(idx + k) % len(TOKYO_PLACEMENTS)]
            blat, blon = dlat + jlat, dlon + jlon
            c = math.cos(math.radians(blat))
            if (tgt[0] + margin <= blat + lat0 and blat + lat1 <= tgt[1] - margin
                    and tgt[2] + margin <= blon + lon0 / c
                    and blon + lon1 / c <= tgt[3] - margin):
                placements[user] = (name, blat, blon)
                break
        else:
            print(f"ERROR: user {user}'s frame extent "
                  f"({lat1 - lat0:.3f}° x {lon1 - lon0:.3f}°) fits no public "
                  f"Tokyo placement inside the target bbox", file=sys.stderr)
            raise SystemExit(1)

    # pass 2: transform, assert, and only then write
    out_lines, n_excl, n_out, n_pts = [], 0, 0, 0
    with open(frame_p) as f:
        out_lines.append(f.readline().rstrip("\n"))
        for line in f:
            cols = line.rstrip("\n").split("\t")
            _, blat, blon = placements[cols[0]]
            lat = blat + float(cols[4])
            lon = blon + float(cols[5]) / math.cos(math.radians(blat))
            if excl[0] <= lat <= excl[1] and excl[2] <= lon <= excl[3]:
                n_excl += 1
            if not (tgt[0] <= lat <= tgt[1] and tgt[2] <= lon <= tgt[3]):
                n_out += 1
            cols[4], cols[5] = f"{lat:.6f}", f"{lon:.6f}"
            out_lines.append("\t".join(cols))
            n_pts += 1
    if n_excl or n_out:
        print(f"manipulate: HARD FAIL — {n_excl} points inside the Beijing "
              f"exclusion bbox, {n_out} outside the Tokyo target bbox; "
              f"{out_p} NOT written", file=sys.stderr)
        raise SystemExit(1)
    out_p.write_text("\n".join(out_lines) + "\n")
    districts = Counter(name for name, _, _ in placements.values())
    print(f"manipulate: {n_pts} points of {len(placements)} users placed on "
          f"{len(districts)} Tokyo districts -> {out_p}")
    return {
        "step": "tokyo juxtaposition",
        "derived_from_data": False,
        "rule": "lat = base_lat + frame_lat; lon = base_lon + frame_lon / "
                "cos(radians(base_lat)) — the frame is isotropic, so the cos "
                "correction restores metric distances at Tokyo's latitude; "
                "timestamps unchanged",
        "placement": "sha256(user_id) picks a PUBLIC Tokyo-metro district "
                     "centroid (+ a deterministic jitter < 0.01°), rotated "
                     "forward until the user's frame extent fits the target "
                     "bbox — fully reproducible, no data dependence",
        "districts": dict(sorted(districts.items())),
        "n_points": n_pts,
        "n_users": len(placements),
        "asserted": {"beijing_exclusion_bbox_deg": list(excl),
                     "tokyo_target_bbox_deg": list(tgt),
                     "violations": 0},
        "frame_tsv": FRAME_TSV,
    }


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
    # the generator just wrote a FRESH neutral-frame TSV to out_p: move it
    # over any stale frame from a previous run (a stale frame left in place
    # would silently win inside _manipulate — it only falls back to the move
    # for standalone `manipulate` invocations)
    os.replace(out / SYNTH_TSV, out / FRAME_TSV)
    # the generator's frame is neutral by construction: the manipulate step
    # is mandatory and runs automatically (it hard-asserts the bboxes)
    manip = _manipulate(out, target="tokyo")
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
        "tsv_sha256": sha(out / SYNTH_TSV),
        "frame_tsv_sha256": sha(out / FRAME_TSV),
        "released_fact_sources": {
            "motif": "dp_release.json cells (daily-motif mix)",
            "transitions": "aux transitions (+ declared home_return_boost "
                           "prior)",
            "dwell": "aux dwell (grid-uniform within the public dwell "
                     "ranges)",
            "departure": "aux departure (uniform within the public hour "
                         "bins)",
            "mode": "aux mode",
            "turn": "turn_hist release artifact (category-2, suppressed)",
            "dist_mix": "dist_mix release artifact (category-2, suppressed)",
            "kinematics": "kinematics release artifact (category-2, "
                          "suppressed)",
            "noise": "noise_moments release artifact (category-2, "
                     "suppressed)",
            "routes": "seg_hist / junction_hist / route_shape release "
                      "artifacts (category-2, suppressed; declared fallback "
                      "priors on releases that predate them)",
        },
        "assumptions": {
            "fields_not_modelled": [
                "street/transit geometry",
                "altitude dynamics (sentinel -777 only)",
                "label-window sub-structure",
                "cross-user relations",
                "within-bin dwell shapes beyond released moments",
                "home-day device-off behaviour",
                "route shapes from released seg/junction histograms; "
                "within-segment micro-curvature not modelled",
            ],
            "combination_rules": [
                "next role = verified draw from the blended transition row "
                "(irregular days); motif templates fix the commute / errand "
                "/ tour chains",
                "trip time = route path length / released per-mode cruise "
                "speed; paths are road-like polylines — segment lengths from "
                "the released segment histogram scaled by the mode's "
                "released junction density, junction angles from the "
                "released junction-angle histogram (in-class jitter, never "
                "on the 90° lattice), left/right from the released share "
                "with the cumulative deviation clamped to ±75°, path budget "
                "from the released per-mode circuity, affine-fitted onto "
                "the anchor pair",
                "GPS noise = per-axis AR(1) matched to the released sd / "
                "lag-1 moments, stay wobble clamped into the detection box",
            ],
            "seed_reproducibility": "intentionally absent (urandom-backed "
                                    "FFI, consistent with non-reproducible "
                                    "DP draws); the manipulate placement IS "
                                    "deterministic given the frame TSV",
        },
        "manipulation": manip,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"synth: generated {out}/{SYNTH_TSV} (+ {FRAME_TSV}, plan.json, "
          f"meta.json)")
    return 0


def cmd_manipulate(args) -> int:
    synth = Path(args.synth)
    manip = _manipulate(synth, target=args.target)
    meta_p = synth / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta["manipulation"] = manip
        meta["tsv_sha256"] = sha(synth / SYNTH_TSV)
        meta_p.write_text(json.dumps(meta, indent=2))
    return 0


def _scan_tsv(tsv: Path, spec: dict) -> dict:
    """One streaming pass over the final TSV: bbox containment, per-(user,
    trajectory) time ordering, public mode labels, the altitude sentinel, and
    a trajectory sample for the path-realism checks (collinear runs and
    ±90° turn atoms are the fingerprints of a naive generator)."""
    excl, tgt = _bboxes(spec)
    public_modes = set(mode_map())
    n = n_excl = n_out = n_alt = order_bad = dup_ts = 0
    bad_modes: Counter = Counter()
    last_ts: dict[tuple, int] = {}
    sample: dict[tuple, list] = {}
    with open(tsv) as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\n").split("\t")
            key = (cols[0], cols[2])
            ts = _ts(cols[3])
            prev = last_ts.get(key)
            if prev is not None:
                if ts < prev:
                    order_bad += 1
                elif ts == prev:
                    dup_ts += 1   # exact duplicates collapse at ingestion
            last_ts[key] = ts
            lat, lon = float(cols[4]), float(cols[5])
            if excl[0] <= lat <= excl[1] and excl[2] <= lon <= excl[3]:
                n_excl += 1
            if not (tgt[0] <= lat <= tgt[1] and tgt[2] <= lon <= tgt[3]):
                n_out += 1
            if float(cols[6]) != -777:
                n_alt += 1
            if cols[9] not in public_modes:
                bad_modes[cols[9]] += 1
            if key in sample:
                if len(sample[key]) < 400:
                    sample[key].append(
                        (lon * math.cos(math.radians(lat)), lat))
            elif len(sample) < 200:
                sample[key] = [(lon * math.cos(math.radians(lat)), lat)]
            n += 1
    collinear = triples = atoms = turns = 0
    for pts in sample.values():
        for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            if v1 == (0.0, 0.0) or v2 == (0.0, 0.0):
                continue
            triples += 1
            if v1[0] * v2[1] - v1[1] * v2[0] == 0.0:
                collinear += 1
            a = math.degrees(math.atan2(v2[1], v2[0])
                             - math.atan2(v1[1], v1[0]))
            a = (a + 180.0) % 360.0 - 180.0
            turns += 1
            if abs(abs(a) - 90.0) <= 0.5:
                atoms += 1
    return {
        "n_points": n, "n_in_beijing_excl": n_excl, "n_outside_tokyo": n_out,
        "n_altitude_not_sentinel": n_alt, "order_violations": order_bad,
        "duplicate_timestamps": dup_ts, "bad_modes": dict(bad_modes),
        "sampled_trajectories": len(sample), "triples": triples,
        "collinear_frac": round(collinear / triples, 4) if triples else 0.0,
        "right_angle_atom_frac": round(atoms / turns, 4) if turns else 0.0,
    }


def cmd_validate(args) -> int:
    synth = Path(args.synth)
    tsv = synth / SYNTH_TSV
    plan_p = synth / "plan.json"
    if not tsv.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    if not GENSPEC.exists():
        print("ERROR: run `make distill` first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    planted = plan.get("planted") or {}
    scratch = (synth / "scratch.duckdb").resolve()
    for p in (scratch, Path(str(scratch) + ".wal")):
        if p.exists():
            p.unlink()
    # re-ingest through the PRODUCTION dbt pipeline (same models, scratch DB,
    # scratch target/log dirs so the enclave's own dbt artifacts stay intact)
    env = dict(os.environ,
               GEOLIFE_TSV=str(tsv.resolve()),
               ENCLAVE_DB=str(scratch),
               DBT_PROFILES_DIR=".",
               DBT_TARGET_PATH=str((synth / "dbt_target").resolve()),
               DBT_LOG_PATH=str((synth / "dbt_logs").resolve()))
    r = subprocess.run(["dbt", "build", "--exclude", "ml_geo_scores"],
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
        gs = json.loads(GENSPEC.read_text())
        # planted recall through the production feature pipeline (hard)
        cu, tu, bu = (planted.get("commute_user"), planted.get("tour_user"),
                      planted.get("beacon_user"))
        if not (cu and tu and bu):
            hard.append("plan.json lacks the planted block")
        else:
            n_com = con.execute(
                "select count(*) from feat_user_days "
                "where user_id=? and motif_class='commute'", [cu]).fetchone()[0]
            n_tour = con.execute(
                "select count(*) from feat_user_days "
                "where user_id=? and motif_class='tour'", [tu]).fetchone()[0]
            n_bea = con.execute(
                "select count(*) from mart_motif_revisit "
                "where user_id=? and is_revisit", [bu]).fetchone()[0]
            found = [n_com >= 1, n_tour >= 1, n_bea >= 1]
            notes["planted_recall"] = round(sum(found) / 3, 4)
            if not found[0]:
                hard.append(f"planted commute days of user {cu} not recalled")
            if not found[1]:
                hard.append(f"planted tour day of user {tu} not recalled")
            if not found[2]:
                hard.append(f"planted revisit beacon of user {bu} not recalled")
        # GOF (soft): re-ingested mixes vs the released (noised, clamped)
        # weights the genspec carries (TV distance)

        def tv(want: dict, got: dict, grid: list[str]) -> float:
            tw, tg = max(1, sum(want.values())), max(1, sum(got.values()))
            return 0.5 * sum(abs(want.get(c, 0) / tw - got.get(c, 0) / tg)
                             for c in grid)

        want_m = {c["name"]: c["w"] for c in gs["motif"]["cells"]}
        got_m = dict(con.execute(
            "select motif_class, count(*) from feat_user_days "
            "group by 1").fetchall())
        notes["motif_mix_tv_vs_release"] = round(tv(want_m, got_m,
                                                    MOTIF_CLASSES), 4)
        want_md = {c["name"]: c["w"] for c in gs["mode"]["cells"]}
        got_md = dict(con.execute(
            "select mode_class, count(*) from feat_legs "
            "where mode_class != 'XX' group by 1").fetchall())
        notes["mode_mix_tv_vs_release"] = round(tv(want_md, got_md,
                                                   MODE_CLASSES), 4)
        # the generator blends motif templates, transitions and kinematics
        # on top of the marginal mixes, so tolerate drift
        for key in ("motif_mix_tv_vs_release", "mode_mix_tv_vs_release"):
            if notes[key] > 0.35:
                soft.append(f"{key} = {notes[key]:.3f} > 0.35")
        # structural invariants + path realism (hard), one TSV pass
        sc = _scan_tsv(tsv, spec)
        notes["structural"] = sc
        if sc["n_in_beijing_excl"]:
            hard.append(f"{sc['n_in_beijing_excl']} points inside the "
                        f"Beijing exclusion bbox")
        if sc["n_outside_tokyo"]:
            hard.append(f"{sc['n_outside_tokyo']} points outside the Tokyo "
                        f"target bbox")
        if sc["order_violations"]:
            hard.append(f"{sc['order_violations']} out-of-order timestamps "
                        f"within a (user, trajectory)")
        if sc["bad_modes"]:
            hard.append(f"raw mode labels outside the public map: "
                        f"{sc['bad_modes']}")
        if sc["n_altitude_not_sentinel"]:
            hard.append(f"{sc['n_altitude_not_sentinel']} rows without the "
                        f"-777 altitude sentinel")
        if sc["triples"] and sc["collinear_frac"] >= 0.2:
            hard.append(f"collinear triple fraction {sc['collinear_frac']} "
                        f">= 0.2 (straight-line generator fingerprint)")
        if sc["triples"] and sc["right_angle_atom_frac"] > 0.05:
            hard.append(f"±90° turn atom fraction "
                        f"{sc['right_angle_atom_frac']} > 0.05 (grid-walk "
                        f"generator fingerprint)")
        # noise (soft): re-detected tight-staypoint residual sd vs released
        got_sd = None
        for metric in ("tight_resid_sd_m_median", "resid_sd_m_median"):
            row = con.execute(
                "select value from mart_noise_moments where metric=?",
                [metric]).fetchone()
            if row and row[0] is not None:
                got_sd = float(row[0])
                break
        rel_sd = gs["noise"]["sd_mm"] / 1000.0
        if got_sd is None:
            soft.append("re-ingested noise moments unavailable (no tight "
                        "staypoints detected)")
        else:
            notes["resid_sd_m"] = {"released": rel_sd,
                                   "reingested": round(got_sd, 3)}
            if not (0.2 * rel_sd <= got_sd <= 5 * rel_sd):
                soft.append(f"re-detected resid sd {got_sd:.2f} m outside "
                            f"[0.2x, 5x] of released {rel_sd:.2f} m")
        # ROUTE SHAPE — junction presence (hard): the production junction
        # detection must re-fire on the synthetic polylines; an empty
        # feat_route_segments means the trips rendered as straight vectors
        try:
            n_rseg = con.execute(
                "select count(*) from feat_route_segments").fetchone()[0]
        except Exception as e:
            n_rseg = None
            hard.append(f"feat_route_segments missing from the re-ingested "
                        f"scratch DB: {e}")
        notes["route_segments_reingested"] = n_rseg
        if n_rseg == 0:
            hard.append("re-ingested feat_route_segments is empty — "
                        "junction detection found no junctions on the "
                        "synthetic polylines")
        # ROUTE-SHAPE GOF (soft): per-mode median straightness of the
        # re-ingested legs. Ground modes must come in BELOW 0.97 (road-like,
        # not straight vectors) and, where the released route_shape rows are
        # available, within ±0.25 of the released medians.
        try:
            s_shape = {m: (float(st), float(ci) if ci is not None else None)
                       for m, st, ci in con.execute(
                           "select mode_class, median(straightness), "
                           "median(circuity) from feat_route_legs "
                           "where straightness is not null "
                           "group by 1").fetchall()}
        except Exception:   # older model set: fall back to feat_legs
            s_shape = {m: (float(st), None) for m, st in con.execute(
                "select mode_class, median(straightness) from feat_legs "
                "where straightness is not null and mode_class != 'XX' "
                "group by 1").fetchall()}
        rel_shape, rel_shape_src = _released_route_shape()
        notes["route_shape"] = {
            "synth_median_straightness": {m: round(v[0], 4)
                                          for m, v in sorted(s_shape.items())},
            "released_source": rel_shape_src or None,
        }
        for m in ("walk", "bike", "bus", "car"):
            if m not in s_shape:
                soft.append(f"route-shape: no re-ingested {m} legs to "
                            f"check straightness")
            elif s_shape[m][0] >= 0.97:
                soft.append(f"route-shape: {m} median straightness "
                            f"{s_shape[m][0]:.3f} >= 0.97 — synthetic trips "
                            f"render as straight vectors")
        rel_circ: dict[str, float] = {}
        if rel_shape is None:
            notes["route_shape"]["band_check"] = (
                "skipped — no released route_shape.csv (release predates "
                "the route-shape artifacts)")
        else:
            band = {}
            for m, r in sorted(rel_shape.items()):
                try:
                    want = float(r["median_straightness"])
                except (KeyError, TypeError, ValueError):
                    continue
                if r.get("median_circuity") not in (None, ""):
                    rel_circ[m] = float(r["median_circuity"])
                if m in s_shape:
                    band[m] = {"released": round(want, 4),
                               "synth": round(s_shape[m][0], 4)}
                    if abs(s_shape[m][0] - want) > 0.25:
                        soft.append(f"route-shape: {m} median straightness "
                                    f"{s_shape[m][0]:.3f} outside ±0.25 of "
                                    f"released {want:.3f}")
            notes["route_shape"]["straightness_band"] = band
        # circuity medians per mode, real (released) vs synth — report only
        notes["route_circuity_median"] = {
            "released": {m: round(v, 4)
                         for m, v in sorted(rel_circ.items())} or None,
            "synth": {m: round(ci, 4) for m, (st, ci) in sorted(s_shape.items())
                      if ci is not None} or None,
        }
        # Lean deciders on the re-ingested observations
        leandir = synth / "leancheck"
        (leandir / "out").mkdir(parents=True, exist_ok=True)
        from export_lean_data import export as export_lean
        export_lean(scratch, leandir / "data")
        rc = subprocess.run([str(CHECKS_BIN)], cwd=str(leandir),
                            capture_output=True, text=True)
        res_p = leandir / "out" / "lean_results.json"
        if not res_p.exists():
            hard.append(f"lean checks failed: {rc.stderr[-400:]}")
        else:
            res = json.loads(res_p.read_text())
            disagree = {k: sum(1 for x in res.get(k, [])
                               if not x.get("agree", True))
                        for k in LEAN_CHECK_KEYS}
            notes["lean_disagreements"] = disagree
            if any(disagree.values()) or rc.returncode != 0:
                hard.append(f"SQL<->Lean disagreement on synth: {disagree}")
        con.close()
    verdict = {"hard_failures": hard, "soft_failures": soft, "notes": notes,
               "tsv_sha256": sha(tsv)}
    (synth / "validation.json").write_text(json.dumps(verdict, indent=2))
    if hard:
        print("validate: HARD FAIL\n - " + "\n - ".join(hard), file=sys.stderr)
        return 1
    if soft:
        print("validate: soft tolerance\n - " + "\n - ".join(soft))
        return 2
    print(f"validate: PASS ({json.dumps(notes)})")
    return 0


def cmd_certify(args) -> int:
    synth = Path(args.synth)
    tsv = synth / SYNTH_TSV
    plan_p = synth / "plan.json"
    if not tsv.exists() or not plan_p.exists():
        print("ERROR: run generate first", file=sys.stderr)
        return 1
    plan = json.loads(plan_p.read_text())
    planted = plan.get("planted") or {}
    n_target = max(1, int(args.n))
    # K-point sample of rendered rows, grouped per (user, trajectory);
    # exact adjacent duplicate timestamps collapse exactly as the production
    # staging dedup does, so strict ascent is the honest decidable statement
    trajs: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    modes: list[str] = []
    n_rows = 0
    with open(tsv) as f:
        f.readline()
        for line in f:
            if n_rows >= n_target:
                break
            cols = line.rstrip("\n").split("\t")
            key = (cols[0], cols[2])
            ts = _ts(cols[3])
            if key not in trajs:
                trajs[key] = []
                order.append(key)
            if not trajs[key] or ts != trajs[key][-1]:
                trajs[key].append(ts)
            if cols[9] not in modes:
                modes.append(cols[9])
            n_rows += 1
    if not n_rows:
        print("ERROR: empty synthetic TSV", file=sys.stderr)
        return 1
    # the planted plan structures: the beacon anchor is visited on every
    # errand_loop / tour day of the beacon user (slot 2 by construction)
    beacon_days: list[int] = []
    commute_motifs: list[str] = []
    for u in plan.get("users", []):
        if u["user"] == planted.get("beacon_user"):
            beacon_days = [int(u["day0"]) + i
                           for i, m in enumerate(u["motifs"])
                           if m in ("errand_loop", "tour")]
        if u["user"] == planted.get("commute_user"):
            commute_motifs = [str(m) for m in u["motifs"]]
    if not beacon_days or not commute_motifs:
        print("ERROR: plan.json lacks the planted users", file=sys.stderr)
        return 1
    ts_lists = ",\n    ".join(
        "[" + ", ".join(str(t) for t in trajs[k]) + "]" for k in order)
    modes_src = ", ".join(f'"{m}"' for m in modes)
    days_src = ", ".join(str(d) for d in beacon_days)
    motifs_src = ", ".join(f'"{m}"' for m in commute_motifs)
    cert = f"""/- AUTO-GENERATED by scripts/synth.py certify — kernel-checked certificate
that a sample of the rendered synthetic rows and the emitted plan satisfy
the formal spec. Timestamps are epoch seconds parsed back from the TSV, with
exact adjacent duplicates collapsed (the same collapse production staging
dedup applies). decide only: propext, no sorry, no native_decide. -/
import EnclaveGeo.Motifs

open EnclaveGeo

/-- Strict ascent, decidable by evaluation. -/
def sortedStrict (l : List Nat) : Bool :=
  (l.zip l.tail).all (fun p => decide (p.1 < p.2))

/-- Per-(user, trajectory) timestamp samples of the rendered TSV. -/
def sampledTrajTs : List (List Nat) := [
    {ts_lists}]

theorem sampled_traj_times_sorted :
    sampledTrajTs.all sortedStrict = true := by decide

/-- Raw mode labels seen in the sample. -/
def sampledModes : List String := [{modes_src}]

/-- Every sampled raw mode label maps to a public mode class through the
formal mode map (never the quality-flag class "XX"). -/
theorem sampled_modes_public :
    sampledModes.all (fun m => Params.modeClassOf m != "XX") = true := by
  decide

/-- The epoch days the plan visits the planted beacon anchor. -/
def plantedBeaconDays : List Nat := [{days_src}]

theorem planted_is_revisit_beacon : IsRevisit plantedBeaconDays := by decide

/-- The planted commuter's planned daily motifs. -/
def plantedCommuteMotifs : List String := [{motifs_src}]

theorem planted_commuter_all_commute :
    plantedCommuteMotifs.all (· == "commute") = true := by decide
"""
    (LEAN / "GenCert.lean").write_text(cert)
    r = subprocess.run(["lake", "env", "lean", "GenCert.lean"], cwd=str(LEAN),
                       capture_output=True, text=True, env=_lean_env())
    ok = r.returncode == 0
    (synth / "certificate.json").write_text(json.dumps({
        "proved": ok,
        "kernel_only": True,
        "theorems": ["sampled_traj_times_sorted", "sampled_modes_public",
                     "planted_is_revisit_beacon",
                     "planted_commuter_all_commute"],
        "n_rows_checked": n_rows,
        "n_trajectories": len(order),
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
    m = sub.add_parser("manipulate")
    m.add_argument("--synth", default="synth")
    m.add_argument("--target", default="tokyo")
    v = sub.add_parser("validate")
    v.add_argument("--synth", default="synth")
    c = sub.add_parser("certify")
    c.add_argument("--synth", default="synth")
    c.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    return {"generate": cmd_generate, "manipulate": cmd_manipulate,
            "validate": cmd_validate, "certify": cmd_certify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

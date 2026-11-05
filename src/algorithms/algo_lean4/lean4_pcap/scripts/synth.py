"""Synthetic Cisco-Joy-schema network flows from the DP release only.

    python3 scripts/synth.py generate --hosts 10000 --out synth
    python3 scripts/synth.py validate --synth synth
    python3 scripts/synth.py certify  --synth synth --n 400 [--native]

Inputs: the DP release (release/<latest>/dp_release.json, fallback
lean/out/dp_release.json) — the ONLY data-derived input; the generator shape
parameters regex-read from lean/EnclaveFlows/FlowShape.lean (Lean is the single
source: the simulation cannot drift from the file the linking theorems are
about); and the concept spec via python/spec.py. No raw or per-host data is
read.

Output: synth/flow_synth.jsonl in the Cisco Joy schema (config header line
first, one JSON object per flow), so the production dbt pipeline ingests it
as-is.

generate  1. Plan (latent), per synthetic host sa = 10.99.x.y (host i ↦ 10.99.(i-1)/254 . (i-1)%254+1):
             * service mix: every flow's service is drawn from the RELEASED
               histogram (weights = released counts of genServices, matched by
               name; cells released as 0 stay 0 — the proved sampler law
               sampleCellPMF_law gives P(i) = w_i / Σw);
             * sessions: a few per host; per session the flow count comes from
               sessionFlowsChoices/Weights, intra-session steps are uniform on
               [1, sessionStepMsMax] ms, inter-session silences are
               >= interSessionGapMsMin ms (interSessionGap_breaks);
             * ONE beacon channel for ~half the hosts: period from
               beaconPeriodMsChoices, exactly beaconFlowsMin flows to one
               dedicated external da at dp 443, and every inter-flow GAP drawn
               directly as g = period - jitter + catsample(2*jitter+1) so each
               gap lies in [period-jitter, period+jitter] — the hypothesis of
               the theorem genBeacon_isBeacon (drawing absolute times
               t0 + k*period + jitter_k would be WRONG: consecutive-jitter
               differences reach 2*jitter).
             The linking theorems (genBeacon_isBeacon, genSession_isTimely,
             sessionStep_lt_chainGap) prove the encoder MUST accept these
             planned structures: "synthetic data passes the spec" is a theorem.
          2. Render Joy JSONL: (pr, dp) per service, sizes/durations from the
             FlowShape choice/weight lists. Topology (GenSpec.lean `Topology`):
             a few dozen busy internal hosts (10.99.x.y) FAN OUT to a large
             long-tailed external set — popular heads 198.51.100.0/24, tail
             198.18.0.0/15 — plus internalPeerPct% internal peers, so busy
             hosts are flow SOURCES, not sinks. Every random draw goes through the verified
             sampler binary catsample (SampCert); no Python RNG anywhere.
          synth/plan.json keeps the latent plan as ground truth for the
          round-trip check (not the product); synth/meta.json records every
          ASSUMPTION vs released fact.
validate  (0) schema: the PRODUCTION dbt pipeline (incl. its tests) builds the
              synthetic file into a scratch DuckDB — any failure = exit 1;
          (1) Lean deciders: beacon/session observations exported from the
              scratch marts exactly as python/export_lean_data.py does, decided
              by `lake exe checks` — a planned beacon with lean_is_beacon=false
              or an untimely session = exit 1;
          (2) round trip vs the latent plan: planned beacons all detected
              (recall 1.0 — a theorem), per-host service counts exact,
              session counts within a declared tolerance;
          (3) statistics: TV distance between the detected clipped service
              histogram (same clipping as mart_dp_counts; planned beacon
              channels excluded — they are a separate latent structure, not
              service-mix traffic) and the released probabilities over
              genServices; TV > 0.20 = exit 2 (soft).
certify   writes lean/GenCert.lean with the DETECTED beacon gap lists and
          session time lists as Lean literals and proves IsBeacon /
          IsTimelySession for all of them by `decide` (kernel; --native for
          native_decide). GenCert.lean is generated: it is excluded from the
          no-sorry pytest and is NOT a lake build target (checked standalone
          via `lake env lean GenCert.lean`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec  # noqa: E402

LEAN = ROOT / "lean"
CHECKS = LEAN / ".lake" / "build" / "bin" / "checks"
FLOWSHAPE = LEAN / "EnclaveFlows" / "FlowShape.lean"
FLOW_JSONL = "flow_synth.jsonl"
EPOCH0 = 1216691467                      # base epoch (same era as the real capture)
# public, data-independent service -> (pr, dp) map, inverse of the feat_flows taxonomy
SERVICE_PORT = {"http": (6, 80), "https": (6, 443), "dns": (17, 53), "ntp": (17, 123),
                "chat": (6, 5222), "ephemeral": (6, None), "other": (6, 8888)}
# Topology/volume are DECLARED assumptions (GenSpec.lean / FlowShape.lean), set so the
# synthetic capture is comparable in shape to a busy-LAN capture: tens of hosts, ~8 h,
# ~14 flows/s, hubs fan out to thousands of external IPs. NOT read from any data.
# Large per-host volume saturates the dp_clip=4 per-host cell clipping, so the released
# service mix is no longer exactly recoverable from the detected histogram (TV check is
# soft, exit 2) — the price of the busy-host topology, recorded in meta.json.
TV_TOLERANCE = 0.35                      # declared (DP noise + Markov rendering), exit 2 beyond
SESSION_COUNT_TOLERANCE = 1              # max |detected - expected| sessions per host (expect 0)


def flowshape_params() -> dict:
    """Generator shape parameters: the `def`s of lean/EnclaveFlows/FlowShape.lean
    (Nat, List Nat, and genServices) — Lean is the single source."""
    src = FLOWSHAPE.read_text()
    out = {k: int(v) for k, v in re.findall(r"^def (\w+) : Nat := (\d+)", src, re.M)}
    out |= {k: [int(x) for x in v.split(",") if x.strip()]
            for k, v in re.findall(r"^def (\w+) : List Nat := \[([\d, ]*)\]", src, re.M)}
    m = re.search(r"^def genServices : List String := \[([^\]]*)\]", src, re.M)
    out["genServices"] = re.findall(r'"([^"]+)"', m.group(1))
    return out


def find_release() -> Path:
    rel = ROOT / "release"
    if rel.is_dir():
        cands = sorted(p for p in rel.iterdir() if (p / "dp_release.json").exists())
        if cands:
            return cands[-1] / "dp_release.json"
    fb = LEAN / "out" / "dp_release.json"
    if fb.exists():
        return fb
    raise SystemExit("no DP release found (release/<id>/dp_release.json or lean/out/dp_release.json)")


def load_topk(path: Path, stem: str, key: str, weight: str) -> list[tuple[str, int]]:
    """Common-knowledge top-k histogram next to the release: <stem>.csv in a
    policy export, <stem>.json in lean/out. Empty when absent (the generator
    then falls back to synthetic rendering, declared in meta)."""
    csv_p, json_p = path.parent / f"{stem}.csv", path.parent / f"{stem}.json"
    if csv_p.exists():
        import csv
        with open(csv_p) as f:
            return [(r[key], int(r[weight])) for r in csv.DictReader(f)]
    if json_p.exists():
        return [(n[key], int(n[weight]))
                for n in json.loads(json_p.read_text())["names"]]
    return []


def load_release(path: Path) -> dict:
    rel = json.loads(path.read_text())
    cells = {c["cell"]: int(c["released"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["released"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    man_p = path.parent / "manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else None
    return {"path": path, "cells": cells, "aux": aux,
            "topk": load_topk(path, "dns_topk", "qname", "n_queries"),
            "sni_topk": load_topk(path, "tls_sni_topk", "sni", "n_flows"),
            "release": rel, "manifest": manifest,
            "dp_release_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(man_p.read_bytes()).hexdigest() if man_p.exists() else None}


def read_meta(synth: Path) -> dict:
    p = synth / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------- generate

def generate(a) -> None:
    """Adapter only: generation runs ENTIRELY in Lean (`lake exe generate`)
    from the distilled genspec (python/distill.py). This function checks the
    inputs exist, invokes the Lean generator, and records provenance metadata
    (meta.json). No sampling or record construction happens in Python."""
    if not 1 <= a.hosts <= 10000:
        raise SystemExit("--hosts must be in 1..10000")
    bundle = Path(a.bundle).resolve() if a.bundle else None
    lean_dir = (bundle or ROOT) / "lean"
    if bundle and not a.release:
        a.release = bundle / "lean" / "out" / "dp_release.json"
    rel = load_release(a.release or find_release())
    genspec = lean_dir / "out" / "genspec.json"
    if not genspec.exists():
        hint = ("run `python3 python/distill.py` inside the bundle first"
                if bundle else "run `make distill` (python/distill.py) first")
        raise SystemExit(f"no genspec at {genspec} — {hint}")
    gen_bin = lean_dir / ".lake" / "build" / "bin" / "generate"
    if not gen_bin.exists():
        hint = ("run `./build_from_source.sh` inside the bundle first"
                if bundle else "cd lean && lake build generate")
        raise SystemExit(f"generator not built at {gen_bin} — {hint}")
    a.out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(gen_bin), str(genspec), str(a.hosts), str(a.out)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"lean generator failed ({r.returncode})")

    spec_j = json.loads(genspec.read_text())
    plan = json.loads((a.out / "plan.json").read_text())
    man = rel["manifest"] or {}

    def hist(k):
        return {c["name"]: c["w"] for c in spec_j[k]["cells"]}

    meta = {
        "generator": "lake exe generate — sampling and record construction run "
                     "ENTIRELY in Lean from the distilled genspec; Python only "
                     "records this metadata",
        "bundle": str(bundle) if bundle else None,
        "release": {"source": str(rel["path"]), "export_id": man.get("export_id"),
                    "policy_version": man.get("policy_version"),
                    "manifest_sha256": rel["manifest_sha256"],
                    "dp_release_sha256": rel["dp_release_sha256"]},
        "genspec": {"path": str(genspec), "version": spec_j["genspec_version"],
                    "sha256": hashlib.sha256(genspec.read_bytes()).hexdigest(),
                    "source": spec_j["source"]},
        "hosts": a.hosts, "beacon_hosts": plan["n_beacon_hosts"],
        "flows": plan["n_flows"],
        "sampler": "sampleCellPMF in-process (SampCert; law sampleCellPMF_law)",
        "output": {"file": FLOW_JSONL,
                   "schema": "Cisco Joy JSONL (config header + one object per flow)",
                   "base_epoch": EPOCH0},
        "released_facts": {
            "service_mix": {"source": "genspec services (DP service histogram)",
                            "weights_over_genServices": hist("services"),
                            "law": "FIRST flow of each session ~ released mix; "
                                   "later flows ~ transitions[prev] (catsample law "
                                   "sampleCellPMF_law)"},
            "transitions": {"source": "genspec transitions (DP aux svc_transition)",
                            "law": "distilled semi-Markov replacement for the "
                                   "NextServiceGRU teacher; zero rows fall back "
                                   "to the marginal (declared)"},
            "tls_stack_mix": {"source": "genspec tls_stacks (DP aux tls_stack)",
                              "weights": hist("tls_stacks")},
            "dns_shape_mix": {"source": "genspec dns_kind/labels/len (DP aux dns_shape)",
                              "kind_weights": hist("dns_kind"),
                              "label_weights": hist("dns_labels"),
                              "len_weights": hist("dns_len")},
            "dns_qname_topk": {"source": "genspec qname_topk (policy artifact dns_topk, "
                                         "common-knowledge, outside the epsilon ledger)",
                               "n_names": len(spec_j["qname_topk"])},
            "tls_sni_topk": {"source": "genspec sni_topk (policy artifact tls_sni_topk)",
                             "n_names": len(spec_j["sni_topk"])}},
        "assumptions": {
            "topology": "declared constants (GenSpec.lean Topology block, derived_from_data: "
                        "false): busy internal hosts fan OUT to a long-tailed external set "
                        "(head 198.51.100.0/24, tail 198.18.0.0/15) + internal peers; hosts, "
                        "sessions/host and session shape sized for ~8 h at ~14 flows/s. "
                        "Known cost: per-host volume saturates dp_clip, so the released "
                        "service mix is only approximately recovered (TV soft-checked). A "
                        "data-derived topology would need new DP aux histograms (budget event).",
            "temporal": "FlowShape.lean + GenSpec.lean constants; proved: "
                        "genBeacon_isBeacon, genSession_isTimely, stepTimes_sorted "
                        "(ordering), genChain fuel-bounded (termination)",
            "record_validity": "mkFlow_valid — structural validity by construction "
                               "in the Lean generator",
            "tls": "canonical-handshake srlt skeleton + six synthetic stacks; mix "
                   "from the released tls_stacks histogram, sni from sni_topk",
            "dns": "kind ~ released dns_shape; names from qname_topk (gibberish "
                   ".test shape-fallback), reverse lookups from the synthetic da",
            "fields_not_modelled": "packets/byte_dist/entropy are null (no payload "
                                   "model); tls/dns blocks are structural renders"},
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    for stale in ("validation.json", "detected.json", "certificate.json",
                  "scratch.duckdb", "scratch.duckdb.wal", "scratch_audit.duckdb",
                  "scratch_audit.duckdb.wal"):
        (a.out / stale).unlink(missing_ok=True)
    shutil.rmtree(a.out / "leancheck", ignore_errors=True)


# ----------------------------------------------------------------------------- validate

def run_dbt(jsonl: Path, db: Path) -> subprocess.CompletedProcess:
    """Build the PRODUCTION dbt pipeline (with its tests) on `jsonl` into scratch `db`."""
    for p in (db, db.with_suffix(db.suffix + ".wal")):
        p.unlink(missing_ok=True)
    env = os.environ | {"FLOW_JSONL": str(jsonl.resolve()), "ENCLAVE_DB": str(db.resolve()),
                        "DBT_PROFILES_DIR": "."}
    return subprocess.run(["dbt", "build", "--exclude", "ml_flow_scores"],
                          cwd=ROOT / "dbt", env=env, capture_output=True, text=True)


def spec_params() -> dict:
    """Public motif thresholds exactly as python/export_lean_data.py exports them."""
    spec = load_spec()
    cv = Fraction(str(spec["beacon_max_cv"]))
    return {"beacon_min_flows": int(spec["beacon_min_flows"]),
            "beacon_max_cv_num": cv.numerator, "beacon_max_cv_den": cv.denominator,
            "beacon_min_period_ms": int(round(float(spec["beacon_min_period_s"]) * 1000)),
            "fanout_min_dsts": int(spec["fanout_min_dsts"]),
            "chain_max_gap_ms": int(float(spec["chain_max_gap_s"]) * 1000)}


def export_lean_inputs(con, workdir: Path) -> tuple[list, list]:
    """lean-check inputs from the scratch DB, with export_lean_data.py's logic:
    beacon candidates (gaps_ms from mart_motif_beacon + feat_flows), sessions
    from mart_seq_sessions, params from the spec."""
    p = spec_params()
    beacons = []
    for sa, da, dpport, n, is_b in con.execute(
            "select sa, da, dp, n_flows, is_beacon from mart_motif_beacon order by n_flows desc").fetchall():
        starts = [r[0] for r in con.execute(
            "select time_start from feat_flows where sa=? and da=? and dp=? order by time_start",
            [sa, da, dpport]).fetchall()]
        beacons.append({"sa": sa, "da": da, "dp": int(dpport), "n_flows": int(n),
                        "sql_is_beacon": bool(is_b),
                        "gaps_ms": [round((b - a) * 1000) for a, b in zip(starts, starts[1:])]})
    sessions = []
    for sa, n, times in con.execute(
            "select sa, n_flows, start_times from mart_seq_sessions where n_flows >= 2 "
            "order by sa, session_id").fetchall():
        sessions.append({"sa": sa, "n_flows": int(n), "times_ms": [round(t * 1000) for t in times]})
    data = workdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "motifs.json").write_text(json.dumps({
        "params": {k: p[k] for k in ("beacon_min_flows", "beacon_max_cv_num", "beacon_max_cv_den",
                                     "beacon_min_period_ms", "fanout_min_dsts")},
        "beacons": beacons, "fanout": [], "returns": []}, indent=2))
    (data / "sessions.json").write_text(json.dumps({
        "max_gap_ms": p["chain_max_gap_ms"] + 1,                  # +1 ms absorbs s->ms rounding
        "sessions": sessions}, indent=2))
    return beacons, sessions


def tv_distance(p: list[float], q: list[float]) -> float:
    sp, sq = sum(p), sum(q)
    return 0.5 * sum(abs(x / sp - y / sq) for x, y in zip(p, q))


def validate(a) -> None:
    import duckdb

    plan_p = a.synth / "plan.json"
    if not plan_p.exists():
        raise SystemExit("run generate first (validate needs synth/plan.json)")
    plan = json.loads(plan_p.read_text())
    shape = flowshape_params()
    services = shape["genServices"]
    rel = load_release(a.release or find_release())

    db = a.synth / "scratch.duckdb"                               # (0) production dbt pipeline + tests
    r = run_dbt(a.synth / FLOW_JSONL, db)
    schema = {"dbt_build_ok": r.returncode == 0,
              "log_tail": (r.stdout + r.stderr).splitlines()[-12:] if r.returncode else
                          [l for l in r.stdout.splitlines() if "Done." in l or "PASS=" in l][-2:]}
    if r.returncode != 0:
        report = {"schema": schema, "passed": False, "exit": 1,
                  "reason": "dbt build (schema/spec tests) failed on the synthetic file"}
        (a.synth / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        sys.exit(1)

    con = duckdb.connect(str(db), read_only=True)
    work = a.synth / "leancheck"                                  # (1) Lean deciders
    shutil.rmtree(work, ignore_errors=True)
    beacons, sessions = export_lean_inputs(con, work)
    if not CHECKS.exists():
        raise SystemExit("checks not built: cd lean && lake build checks")
    subprocess.run([str(CHECKS)], cwd=work, capture_output=True, text=True, check=True)
    lres = json.loads((work / "out" / "lean_results.json").read_text())
    lean_b = {(b["sa"], b["da"], int(b["dp"]) if b["dp"] is not None else None): b
              for b in lres["beacons"]}
    planned = {(b["sa"], b["da"], b["dp"]): b for b in plan["beacons"]}
    missing = [k for k in planned if k not in lean_b]
    lean_false = [k for k in planned if k in lean_b and not lean_b[k]["lean_is_beacon"]]
    untimely = [s for s in lres["sessions"] if not s["lean_is_timely_session"]]
    lean_rep = {"beacon_candidates_checked": len(lres["beacons"]),
                "planned_beacons_lean_true": len(planned) - len(lean_false) - len(missing),
                "planned_beacons_lean_false": [list(k) for k in lean_false],
                "sessions_checked": len(lres["sessions"]),
                "sessions_untimely": [s["sa"] for s in untimely],
                "passed": not lean_false and not untimely}

    sql_b = {(b["sa"], b["da"], b["dp"]): b for b in beacons if b["sql_is_beacon"]}
    recall = sum(1 for k in planned if k in sql_b) / max(len(planned), 1)   # (2) round trip
    det_svc = {}
    for sa, svc, n in con.execute(
            "select sa, service_class, count(*) from feat_flows group by sa, service_class").fetchall():
        det_svc.setdefault(sa, {})[svc] = int(n)
    svc_mismatch = []
    for sa, h in plan["hosts"].items():
        if h["service_counts"] != det_svc.get(sa, {}):
            svc_mismatch.append({"sa": sa, "planned": h["service_counts"],
                                 "detected": det_svc.get(sa, {})})
    det_sess = dict(con.execute("select sa, count(*) from mart_seq_sessions group by sa").fetchall())
    sess_diffs = {}
    for sa, h in plan["hosts"].items():
        expected = h["n_sessions"] + (h["beacon"]["n_flows"] if h["beacon"] else 0)
        sess_diffs[sa] = int(det_sess.get(sa, 0)) - expected
    max_sess_diff = max(abs(d) for d in sess_diffs.values())
    roundtrip = {"planned_beacons": len(planned), "detected_as_beacon": len(sql_b),
                 "beacon_recall": round(recall, 4),
                 "beacon_recall_note": "1.0 expected — genBeacon_isBeacon is a theorem",
                 "service_counts_exact": not svc_mismatch,
                 "service_count_mismatches": svc_mismatch[:10],
                 "session_count": {"expected": "planned sessions + beacon flows (isolated singletons)",
                                   "max_abs_diff_per_host": max_sess_diff,
                                   "tolerance": SESSION_COUNT_TOLERANCE,
                                   "within": max_sess_diff <= SESSION_COUNT_TOLERANCE},
                 "passed": recall == 1.0 and not svc_mismatch
                           and max_sess_diff <= SESSION_COUNT_TOLERANCE}

    clip = int(load_spec()["dp_clip"])                            # (3) statistics (clipped TV)
    rows = con.execute("select sa, da, dp, service_class, count(*) n from feat_flows "
                       "group by sa, da, dp, service_class").fetchall()
    per_user: dict = {}
    for sa, da, dpport, svc, n in rows:
        if (sa, da, int(dpport)) in planned:                      # beacon channels: separate structure
            continue
        per_user[(sa, svc)] = per_user.get((sa, svc), 0) + int(n)
    clipped = {s: 0 for s in services}
    for (sa, svc), n in per_user.items():
        if svc in clipped:
            clipped[svc] += min(n, clip)
    rel_w = [max(0, rel["cells"][s]) for s in services]
    tv = tv_distance(rel_w, [max(clipped[s], 0) for s in services]) if sum(clipped.values()) else 1.0
    tot = sum(clipped.values())
    statistical = {"tv_service_mix": round(tv, 4), "tolerance": TV_TOLERANCE, "within": tv <= TV_TOLERANCE,
                   "released_p": {s: round(w / sum(rel_w), 4) for s, w in zip(services, rel_w)},
                   "detected_clipped_p": {s: round(clipped[s] / tot, 4) for s in services},
                   "clip": clip,
                   "note": "detected = per-host clipped counts (as mart_dp_counts) excluding planned "
                           "beacon channels; released values are DP-noised — declared, not proved"}

    (a.synth / "detected.json").write_text(json.dumps({           # ground truth for certify
        "beacons": [{"sa": b["sa"], "da": b["da"], "dp": b["dp"], "gaps_ms": b["gaps_ms"]}
                    for b in beacons if b["sql_is_beacon"]],
        "sessions": [{"sa": s["sa"], "times_ms": s["times_ms"]} for s in sessions]}, indent=2))
    con.close()

    report = {"schema": schema, "lean": lean_rep, "roundtrip": roundtrip, "statistical": statistical}
    report["passed"] = lean_rep["passed"] and roundtrip["passed"] and statistical["within"]
    code = 1 if not lean_rep["passed"] else (0 if report["passed"] else 2)
    report["exit"] = code
    (a.synth / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(code)


# ----------------------------------------------------------------------------- certify

def lake() -> str:
    exe = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(exe).exists():
        raise SystemExit("lake not found: install elan or add ~/.elan/bin to PATH")
    return exe


def certify(a) -> None:
    det_p = a.synth / "detected.json"
    if not det_p.exists():
        raise SystemExit("run validate first (certify proves the DETECTED structures)")
    det = json.loads(det_p.read_text())
    beacons, sessions, n = [], [], 0
    for b in det["beacons"]:
        g = b["gaps_ms"][:15]                                     # keep decide tractable
        if n + len(g) > a.n:
            break
        beacons.append(g)
        n += len(g)
    for s in det["sessions"]:
        t = s["times_ms"][:40]
        if n + len(t) > a.n:
            break
        sessions.append(t)
        n += len(t)
    if not beacons or not sessions:
        raise SystemExit("no detected beacons/sessions to certify (or --n too small)")

    def lst(xs: list[list[int]]) -> str:
        return ",\n  ".join("[" + ", ".join(str(x) for x in row) + "]" for row in xs)

    tactic = "native_decide" if a.native else "decide"
    # core Rat.add/sub/mul/inv are @[irreducible]; `unseal` lets the kernel-backed
    # `decide` reduce the ℚ arithmetic (no effect on soundness, none on native_decide)
    unseal = "unseal Rat.add Rat.sub Rat.mul Rat.inv in\n" if not a.native else ""
    src = f"""import EnclaveFlows.FlowShape
/-! Auto-generated by `scripts/synth.py certify` from synth/detected.json
    ({len(beacons)} beacon gap lists, {len(sessions)} session time lists, {n} entries).
    NOT a lake build target; checked standalone: `lake env lean GenCert.lean`.
    Excluded from the no-sorry pytest (generated certificate). -/
set_option maxRecDepth 1000000

def sampleBeacons : List (List ℚ) := [
  {lst(beacons)}]

def sampleSessions : List (List ℚ) := [
  {lst(sessions)}]

{unseal}theorem beacons_valid :
    ∀ gaps ∈ sampleBeacons,
      EnclaveFlows.IsBeacon EnclaveFlows.FlowShape.specBeaconParams gaps := by
  {tactic}

{unseal}theorem sessions_timely :
    ∀ ts ∈ sampleSessions,
      EnclaveFlows.IsTimelySession ((EnclaveFlows.Params.chainMaxGapMs : ℚ)) ts := by
  {tactic}

#print axioms beacons_valid
#print axioms sessions_timely
"""
    cert = LEAN / "GenCert.lean"
    cert.write_text(src)
    t0 = time.time()
    env = os.environ | {"PATH": f"{Path.home() / '.elan' / 'bin'}:{os.environ.get('PATH', '')}"}
    r = subprocess.run([lake(), "env", "lean", cert.name], cwd=LEAN,
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    ax = {m.group(1): [x.strip() for x in m.group(2).split(",") if x.strip()]
          for m in re.finditer(r"'(\w+)' depends on axioms: \[([^\]]*)\]", out)}
    for m in re.finditer(r"'(\w+)' does not depend on any axioms", out):
        ax[m.group(1)] = []
    names = ("beacons_valid", "sessions_timely")
    proved = r.returncode == 0 and all(t in ax for t in names) and "error" not in out.lower()
    kernel_only = proved and not any("ofReduceBool" in x or "trustCompiler" in x or "native" in x.lower()
                                     for t in names for x in ax.get(t, []))
    summary = {"theorems": [
                   "beacons_valid : ∀ gaps ∈ sampleBeacons, IsBeacon specBeaconParams gaps",
                   "sessions_timely : ∀ ts ∈ sampleSessions, IsTimelySession chainMaxGapMs ts"],
               "tactic": tactic, "n_beacons": len(beacons), "n_sessions": len(sessions),
               "n_entries": n, "source": "encoder-detected structures (synth/detected.json)",
               "proved": proved, "axioms": ax, "kernel_only": kernel_only,
               "seconds": round(time.time() - t0, 1),
               "lean_toolchain": (LEAN / "lean-toolchain").read_text().strip(),
               "lean_output_tail": out[-2000:] if not proved else ""}
    (a.synth / "certificate.json").write_text(json.dumps(summary, indent=2))
    print(f"{cert}: beacons_valid + sessions_timely ({len(beacons)} beacons, {len(sessions)} sessions, "
          f"by {tactic}, {summary['seconds']}s) -> {'PROVED' if proved else 'FAILED'}"
          + (f"; axioms {sorted(set(x for v in ax.values() for x in v))}; kernel_only={kernel_only}"
             if proved else ""))
    if not proved:
        print(out[-3000:])
    sys.exit(0 if proved else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="plan from the DP release, render Joy-schema JSONL")
    g.add_argument("--hosts", type=int, default=10000)
    g.add_argument("--out", type=Path, default=Path("synth"))
    g.add_argument("--release", type=Path, default=None, help="path to dp_release.json (default: latest)")
    g.add_argument("--bundle", default=None,
                   help="generate from a source-only bundle dir: its genspec, its "
                        "lean-built generator and (unless --release) its own release")
    v = sub.add_parser("validate", help="production dbt + Lean deciders + round trip + statistics")
    v.add_argument("--synth", type=Path, default=Path("synth"))
    v.add_argument("--release", type=Path, default=None)
    c = sub.add_parser("certify", help="kernel-checked Lean certificate of the detected structures")
    c.add_argument("--synth", type=Path, default=Path("synth"))
    c.add_argument("--n", type=int, default=400)
    c.add_argument("--native", action="store_true")
    a = ap.parse_args()
    {"generate": generate, "validate": validate, "certify": certify}[a.cmd](a)


if __name__ == "__main__":
    main()

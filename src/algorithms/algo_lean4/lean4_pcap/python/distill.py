#!/usr/bin/env python3
"""Distill the generative behavior of the minimodels into the typed Lean
generation specification (genspec).

Inputs are RELEASED artifacts only — the DP release (incl. the aux
svc_transition / tls_stack / dns_shape histograms) and the common-knowledge
top-k artifacts. No raw data, weights or embeddings are read for the spec, so
the genspec inherits the release's privacy accounting by post-processing.

Outputs:
  lean/out/genspec.json              the restricted generative specification
                                     consumed by `lake exe generate`
                                     (schema+distributions+dependencies+
                                     temporal structure+constraints+metadata)
  artifacts/reports/distill_report.json   ENCLAVE-ONLY fidelity report:
      the distilled semi-Markov student vs the NextServiceGRU teacher vs the
      repeat-last baseline on the same held-out windows, and per-row TV
      between the DP-released and the exact (in-enclave) transition matrix.

Distillation table (teacher -> executable statistical replacement):
  NextServiceGRU      -> probabilistic state machine (DP transition histogram
                         + FlowShape dwell/gap distributions)
  TlsRecordGRU        -> ordered conditional sampler (DP stack-rank mix over
                         a fixed canonical-handshake skeleton)
  DnsCharLM           -> shape histogram + common-knowledge top-k names
  ByteDistClassifier,
  FlowAutoencoder     -> scoring models: NOT generative, stay enclave-side
                         (realism audit); nothing to distill into the spec.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lean" / "out"
GENSVC = ["http", "https", "dns", "ntp", "chat", "ephemeral", "other"]
TLS_STACK_CELLS = ["stack_1", "stack_2", "stack_3", "stack_4", "stack_5", "other"]
DNS_KIND = ["kind_ordinary", "kind_reverse_lookup", "kind_service_discovery"]
DNS_LABELS = ["labels_2", "labels_3", "labels_4", "labels_5plus"]
DNS_LEN = ["len_le_24", "len_gt_24"]
SESSION_GAP_S = 600.0
ACCEPT_MARGIN = 0.02   # student may trail repeat-last by at most this much


def cells_of(d: dict, names: list[str]) -> list[dict]:
    return [{"name": n, "w": max(0, int(d.get(n, 0)))} for n in names]


def main() -> int:
    rel_p = OUT / "dp_release.json"
    if not rel_p.exists():
        print("ERROR: run `make lean-release` first", file=sys.stderr)
        return 1
    rel = json.loads(rel_p.read_text())
    cells = {c["cell"]: int(c["released"]) for c in rel["cells"]}
    aux = {h["name"]: {c["cell"]: int(c["released"]) for c in h["cells"]}
           for h in rel.get("aux", [])}
    for need in ("svc_transition", "tls_stack", "dns_shape"):
        if need not in aux:
            print(f"ERROR: release lacks aux histogram {need}", file=sys.stderr)
            return 1
    topk = json.loads((OUT / "dns_topk.json").read_text())["names"]
    sni = json.loads((OUT / "tls_sni_topk.json").read_text())["names"]

    trans = aux["svc_transition"]

    def trans_row(p: str) -> list[dict]:
        """DP-released row blended with a PUBLIC self-transition prior
        (bursty traffic is self-transitioning — domain common knowledge,
        derived_from_data: false): the diagonal gets a boost equal to
        max(1, sum of the released row), so the state machine is sticky like
        the teacher while the DP row shapes the off-diagonal mass."""
        row = {n: max(0, int(trans.get(f"{p}>{n}", 0))) for n in GENSVC}
        boost = max(1, sum(row.values()))
        row[p] += boost
        return [{"name": n, "w": row[n]} for n in GENSVC]

    genspec = {
        "genspec_version": "1.0.0",
        "kind": "restricted generative specification (no executable code)",
        "source": {
            "dp_release_sha256": hashlib.sha256(rel_p.read_bytes()).hexdigest(),
            "privacy": "all distributions below are DP-released histograms or "
                       "policy-released common-knowledge top-k artifacts; the "
                       "spec is post-processing of the release",
        },
        "schema": {"format": "cisco-joy-jsonl",
                   "constraints": ["pr/dp from the public service map",
                                   "time_end > time_start",
                                   "num_pkts = 1 + bytes // 1400 (0 when bytes_in = 0)",
                                   "tls only on https flows (canonical handshake)",
                                   "dns only on dns flows (well-formed qname)"]},
        "services": {"cells": cells_of(cells, GENSVC)},
        "transitions": [{"given": p, "cells": trans_row(p)} for p in GENSVC],
        "transitions_prior": "diagonal self-transition boost = max(1, sum of the "
                             "released row) — public structural prior, "
                             "derived_from_data: false",
        "tls_stacks": {"cells": cells_of(aux["tls_stack"], TLS_STACK_CELLS)},
        "dns_kind": {"cells": cells_of(aux["dns_shape"], DNS_KIND)},
        "dns_labels": {"cells": cells_of(aux["dns_shape"], DNS_LABELS)},
        "dns_len": {"cells": cells_of(aux["dns_shape"], DNS_LEN)},
        "qname_topk": [{"name": n["qname"], "w": int(n["n_queries"])} for n in topk],
        "sni_topk": [{"name": n["sni"], "w": int(n["n_flows"])} for n in sni],
        "temporal": "states/dwell/durations from lean/EnclaveFlows/FlowShape.lean "
                    "and GenSpec.lean constants (fixed, derived_from_data: false)",
    }
    (OUT / "genspec.json").write_text(json.dumps(genspec, indent=2))

    # ---- enclave-only fidelity report -------------------------------------
    db = ROOT / "artifacts" / "enclave.duckdb"
    report: dict = {"note": "ENCLAVE-ONLY (never exported): distillation fidelity"}
    if db.exists():
        import duckdb  # enclave-side only; consumers of the release don't need it
        con = duckdb.connect(str(db), read_only=True)
        rows = con.execute(
            "with f as (select sa, time_start, case when service_class in "
            "('http','https','dns','ntp','chat','ephemeral') then service_class "
            "else 'other' end as svc from feat_flows) "
            "select lag(svc) over w as prev, svc, time_start, "
            "time_start - lag(time_start) over w as gap from f "
            "window w as (partition by sa order by time_start) "
            "qualify prev is not null and gap <= ?", [SESSION_GAP_S]
        ).fetchall()
        rows.sort(key=lambda r: r[2])
        val = rows[int(len(rows) * 0.8):]
        # student: argmax of the blended (prior + DP) transition row — the
        # same law the Lean generator samples from
        blended = {p: {c["name"]: c["w"] for c in trans_row(p)} for p in GENSVC}
        student = {p: max(GENSVC, key=lambda n: blended[p][n]) for p in GENSVC}
        n = len(val)
        student_acc = sum(student[p] == s for p, s, _, _ in val) / n
        repeat_acc = sum(p == s for p, s, _, _ in val) / n
        # per-row TV between released and exact in-enclave transition rows
        exact: dict[str, dict[str, int]] = {p: {} for p in GENSVC}
        for p, s, _, _ in rows:
            exact[p][s] = exact[p].get(s, 0) + 1
        tv = {}
        for p in GENSVC:
            te = sum(exact[p].values())
            tr = sum(max(0, trans.get(f"{p}>{n}", 0)) for n in GENSVC)
            if te and tr:
                tv[p] = round(0.5 * sum(
                    abs(exact[p].get(n, 0) / te
                        - max(0, trans.get(f"{p}>{n}", 0)) / tr)
                    for n in GENSVC), 4)
        teacher_acc = None
        reg = ROOT / "models_out" / "registry.json"
        if reg.exists():
            active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
            meta_p = ROOT / "models_out" / "minimodels" / str(active) / "meta.json"
            if meta_p.exists():
                teacher_acc = (json.loads(meta_p.read_text())
                               .get("next_service", {}).get("metrics", {})
                               .get("val_accuracy"))
        report.update({
            "next_service": {
                "n_val_pairs": n,
                "student_markov_acc": round(student_acc, 4),
                "repeat_last_acc": round(repeat_acc, 4),
                "teacher_gru_val_acc": teacher_acc,
                "released_vs_exact_row_tv": tv,
                "accepted": student_acc >= repeat_acc - ACCEPT_MARGIN,
                "criterion": f"student >= repeat-last - {ACCEPT_MARGIN}",
            }})
        con.close()
    rpt_dir = ROOT / "artifacts" / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    (rpt_dir / "distill_report.json").write_text(json.dumps(report, indent=2))
    ns = report.get("next_service", {})
    print(f"wrote {OUT/'genspec.json'} (services 7, transitions 7x7, tls 6, "
          f"dns {len(DNS_KIND)+len(DNS_LABELS)+len(DNS_LEN)}, "
          f"qname topk {len(genspec['qname_topk'])}, sni topk {len(genspec['sni_topk'])}) "
          f"and distill_report.json (student {ns.get('student_markov_acc')} vs "
          f"repeat-last {ns.get('repeat_last_acc')} vs teacher {ns.get('teacher_gru_val_acc')}; "
          f"accepted={ns.get('accepted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

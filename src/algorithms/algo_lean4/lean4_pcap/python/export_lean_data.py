#!/usr/bin/env python3
"""Export in-enclave aggregates to the Lean formal layer.

Writes:
  lean/data/dp_counts.json — clipped per-cell counts (input to `lake exe release`,
      which adds SampCert discrete-Laplace noise; the JSON itself must NOT leave
      the enclave).
  lean/data/motifs.json   — beacon gap sequences, fan-out and return-motif
      observations plus the public thresholds, re-checked by `lake exe checks`
      against the formal motif definitions.

Per goal.md guideline 10: Lean proves properties of the mechanism and the motif
predicates; it does NOT prove these inputs were correctly computed by SQL.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import load_spec, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "lean" / "data"

# public DP parameters: per-cell epsilon = EPS1/EPS2 (SampCert takes positive
# naturals); total budget = n_cells * EPS1/EPS2 by sequential composition.
# The service mix (7 public cells, proportional L1 clip dp_clip_total) gets
# the lion's share of the budget so the released mix survives the noise.
EPS1 = int(os.environ.get("DP_EPS1", "2"))
EPS2 = int(os.environ.get("DP_EPS2", "5"))
BUDGET = float(os.environ.get("DP_BUDGET", "4.0"))
# auxiliary shape histograms are released at smaller per-cell epsilons so the
# exact-rational ledger stays within budget:
# 7*(2/5) + 6*(1/25) + 9*(1/25) + 49*(1/100) = 389/100 <= 4/1
AUX_EPS1 = int(os.environ.get("DP_AUX_EPS1", "1"))
AUX_EPS2 = int(os.environ.get("DP_AUX_EPS2", "25"))
TRANS_EPS2 = int(os.environ.get("DP_TRANS_EPS2", "100"))

# public motif thresholds — from the concept spec (dbt vars; sync-tested
# against lean/EnclaveFlows/Params.lean by tests/test_spec_sync.py)
_SPEC = load_spec()
BEACON_MIN_FLOWS = int(_SPEC["beacon_min_flows"])
_CV = Fraction(str(_SPEC["beacon_max_cv"]))
BEACON_MAX_CV_NUM, BEACON_MAX_CV_DEN = _CV.numerator, _CV.denominator
BEACON_MIN_PERIOD_MS = int(round(float(_SPEC["beacon_min_period_s"]) * 1000))
FANOUT_MIN_DSTS = int(_SPEC["fanout_min_dsts"])
CHAIN_MAX_GAP_S = float(_SPEC["chain_max_gap_s"])
TLS_RARE_MAX_FLOWS = int(_SPEC["tls_rare_max_flows"])
DNS_MAX_LABEL_LEN = int(_SPEC["dns_max_label_len"])
DNS_MAX_NAME_LEN = int(_SPEC["dns_max_name_len"])
DNS_SUBFAN_MIN_NAMES = int(_SPEC["dns_subfan_min_names"])


def main() -> int:
    db = ROOT / "artifacts" / "enclave.duckdb"
    if not db.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    con = duckdb.connect(str(db), read_only=True)
    DATA.mkdir(parents=True, exist_ok=True)

    cells = con.execute(
        "select cell, clipped_count, n_contributing_users, clip from mart_dp_counts order by cell"
    ).fetchall()
    dp = {
        "privacy_unit": "one monitored (internal) host = one user, add/remove-one "
                        "Neighbour; each flow attributed to its internal endpoint; "
                        "contribution = proportional L1 allocation over dp_clip_total",
        "grid": "public 7-class generator service taxonomy (genServices)",
        "spec_version": spec_version(),
        "clip": int(cells[0][3]),
        "eps1": EPS1,
        "eps2": EPS2,
        "budget_total_eps": BUDGET,
        "budget_num": Fraction(str(BUDGET)).numerator,
        "budget_den": Fraction(str(BUDGET)).denominator,
        "cells": [
            {"cell": c, "clipped_count": int(n), "n_users": int(u)}
            for c, n, u, _ in cells
        ],
        # auxiliary shape histograms, same mechanism at AUX_EPS per cell; the
        # release executable charges every aux cell in the same exact ledger
        "aux": [
            {
                "name": name,
                "clip": int(rows[0][3]),
                "eps1": e1,
                "eps2": e2,
                "cells": [{"cell": c, "clipped_count": int(n), "n_users": int(u)}
                          for c, n, u, _ in rows],
            }
            for name, e1, e2, rows in (
                ("tls_stack", AUX_EPS1, AUX_EPS2, con.execute(
                    "select cell, clipped_count, n_contributing_users, clip "
                    "from mart_dp_tls order by cell").fetchall()),
                ("dns_shape", AUX_EPS1, AUX_EPS2, con.execute(
                    "select cell, clipped_count, n_contributing_users, clip "
                    "from mart_dp_dns order by cell").fetchall()),
                ("svc_transition", AUX_EPS1, TRANS_EPS2, con.execute(
                    "select cell, clipped_count, n_contributing_users, clip "
                    "from mart_dp_transitions order by cell").fetchall()),
            )
        ],
    }
    (DATA / "dp_counts.json").write_text(json.dumps(dp, indent=2))

    # beacon candidates: ordered start times -> integer millisecond gaps
    pairs = con.execute(
        """
        select sa, da, dp, n_flows, cv_gap, is_beacon
        from mart_motif_beacon order by n_flows desc limit 25
        """
    ).fetchall()
    beacons = []
    for sa, da, dpport, n, cv, is_b in pairs:
        starts = [r[0] for r in con.execute(
            "select time_start from feat_flows where sa=? and da=? and dp=? order by time_start",
            [sa, da, dpport]).fetchall()]
        gaps_ms = [round((b - a) * 1000) for a, b in zip(starts, starts[1:])]
        beacons.append({
            "sa": sa, "da": da, "dp": int(dpport),
            "n_flows": int(n),
            "sql_cv": None if cv is None else float(cv),
            "sql_is_beacon": bool(is_b),
            "gaps_ms": gaps_ms,
        })

    fanout = []
    for sa, n, f in con.execute(
        "select sa, n_distinct_dst, is_fanout from mart_motif_fanout order by n_distinct_dst desc limit 15"
    ).fetchall():
        dsts = [r[0] for r in con.execute(
            "select distinct da from feat_flows where sa=? order by da", [sa]).fetchall()]
        fanout.append({"sa": sa, "n_distinct_dst": int(n),
                       "sql_is_fanout": bool(f), "dsts": dsts})

    # destination sequences (session-gap-broken upstream) for return-motif check
    returns = []
    for (sa,) in con.execute(
        "select distinct sa from mart_motif_return order by sa limit 10"
    ).fetchall():
        seq = [r[0] for r in con.execute(
            "select da from feat_flows where sa=? order by time_start, flow_id limit 400",
            [sa]).fetchall()]
        returns.append({"sa": sa, "dst_seq": seq})

    motifs = {
        "spec_version": spec_version(),
        "params": {
            "beacon_min_flows": BEACON_MIN_FLOWS,
            "beacon_max_cv_num": BEACON_MAX_CV_NUM,
            "beacon_max_cv_den": BEACON_MAX_CV_DEN,
            "beacon_min_period_ms": BEACON_MIN_PERIOD_MS,
            "fanout_min_dsts": FANOUT_MIN_DSTS,
        },
        "beacons": beacons,
        "fanout": fanout,
        "returns": returns,
    }
    (DATA / "motifs.json").write_text(json.dumps(motifs, indent=2))

    # contingency cells for the top association pairs — Lean recomputes
    # chi-square from these exactly over ℚ (EnclaveFlows/Assoc.lean)
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
    (DATA / "assoc.json").write_text(json.dumps({"pairs": assoc_pairs}, indent=2))

    # SQL-maximal timely sessions for the Lean Chain' check; +1 ms tolerance
    # absorbs the seconds->ms rounding at session boundaries
    sessions = []
    for sa, n, times in con.execute(
        "select sa, n_flows, start_times from mart_seq_sessions "
        "where n_flows between 3 and 300 order by n_flows desc limit 12"
    ).fetchall():
        sessions.append({"sa": sa, "n_flows": int(n),
                         "times_ms": [round(t * 1000) for t in times]})
    (DATA / "sessions.json").write_text(json.dumps({
        "max_gap_ms": int(CHAIN_MAX_GAP_S * 1000) + 1,
        "sessions": sessions,
    }, indent=2))

    # TLS: per-flow fingerprint components + record sequences for the Lean
    # IsCanonicalHandshake / IsRareStack checks (EnclaveFlows/Tls.lean)
    tls_flows = []
    for fid, cv, cs, ext, fph, canon, recs in con.execute(
        "select flow_id, c_version, cs, ext_names, fingerprint, is_canonical_hs, recs "
        "from feat_tls order by flow_id"
    ).fetchall():
        tls_flows.append({
            "flow_id": int(fid),
            "fp": {"v": 0 if cv is None else int(cv),
                   "cs": list(cs), "ext": list(ext)},
            "fp_hash": fph,
            "sql_is_canonical": bool(canon),
            "recs": [[1 if r["dir"] == ">" else 0, int(r["tp"]),
                      int(r["b"]), int(r["ipt"])] for r in recs],
        })
    fp_stats = [
        {"fp_hash": fph, "n_flows": int(n), "sql_is_rare": bool(rare)}
        for fph, n, rare in con.execute(
            "select fingerprint, n_flows, is_rare_stack from mart_tls_fingerprints "
            "order by n_flows desc"
        ).fetchall()
    ]
    (DATA / "tls.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "params": {"tls_rare_max_flows": TLS_RARE_MAX_FLOWS},
        "flows": tls_flows,
        "fp_stats": fp_stats,
    }, indent=2))

    # DNS: distinct qnames (as label lists) + per-domain fans for the Lean
    # IsWellFormedQName / IsSubdomainFan checks (EnclaveFlows/Dns.lean)
    qnames = [
        {"qname": qn, "labels": list(labels), "sql_is_wellformed": bool(wf)}
        for qn, labels, wf in con.execute(
            "select qname, any_value(labels), bool_and(is_wellformed) "
            "from feat_dns group by qname order by qname"
        ).fetchall()
    ]
    fans = []
    for dom, is_fan in con.execute(
        "select registered_domain, is_subdomain_fan from mart_dns_domains "
        "order by n_distinct_names desc limit 12"
    ).fetchall():
        names = [r[0] for r in con.execute(
            "select distinct qname from feat_dns where registered_domain=? order by qname",
            [dom]).fetchall()]
        fans.append({"registered_domain": dom, "names": names,
                     "sql_is_fan": bool(is_fan)})
    (DATA / "dns.json").write_text(json.dumps({
        "spec_version": spec_version(),
        "params": {"dns_max_label_len": DNS_MAX_LABEL_LEN,
                   "dns_max_name_len": DNS_MAX_NAME_LEN,
                   "dns_subfan_min_names": DNS_SUBFAN_MIN_NAMES},
        "qnames": qnames,
        "fans": fans,
    }, indent=2))

    # Common-knowledge top-k query names -> lean/out (the release staging dir,
    # same place the DP release lands): popular names queried by many hosts are
    # policy-released metadata (release_policy.yaml artifacts.dns_topk,
    # suppression-thresholded, OUTSIDE the epsilon ledger); *.arpa is excluded
    # in the mart because reverse-lookup names encode internal IP octets.
    topk = con.execute(
        "select qname, n_queries, n_src_hosts from mart_dns_topk order by n_queries desc, qname"
    ).fetchall()
    out_dir = ROOT / "lean" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dns_topk.json").write_text(json.dumps({
        "kind": "common_knowledge_metadata",
        "basis": "names queried by >=2 distinct hosts and >=10 times "
                 "(suppression-thresholded; *.arpa and single-label names excluded); "
                 "released via release_policy.yaml artifacts.dns_topk, outside the "
                 "epsilon ledger",
        "names": [{"qname": q, "n_queries": int(n), "n_src_hosts": int(h)}
                  for q, n, h in topk],
    }, indent=2))
    sni_topk = con.execute(
        "select sni, n_flows, n_src_hosts from mart_tls_sni_topk order by n_flows desc, sni"
    ).fetchall()
    (out_dir / "tls_sni_topk.json").write_text(json.dumps({
        "kind": "common_knowledge_metadata",
        "basis": "SNI hostnames seen from >=2 distinct hosts in >=10 TLS flows "
                 "(suppression-thresholded; single-label names excluded); released "
                 "via release_policy.yaml artifacts.tls_sni_topk, outside the "
                 "epsilon ledger",
        "names": [{"sni": s, "n_flows": int(n), "n_src_hosts": int(h)}
                  for s, n, h in sni_topk],
    }, indent=2))

    print(f"wrote {DATA/'dp_counts.json'} ({len(cells)} cells), "
          f"{DATA/'motifs.json'} ({len(beacons)} beacon candidates), "
          f"{DATA/'assoc.json'} ({len(assoc_pairs)} pairs), "
          f"{DATA/'sessions.json'} ({len(sessions)} sessions), "
          f"{DATA/'tls.json'} ({len(tls_flows)} tls flows, {len(fp_stats)} stacks), "
          f"{DATA/'dns.json'} ({len(qnames)} qnames, {len(fans)} domains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

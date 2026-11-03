#!/usr/bin/env python3
"""Realism audit (ENCLAVE-ONLY): judge the synthetic graph against the
enclave's measured structure. Graded, not pass/fail — the generator targets
the spec contract and the release, not realism; the gap is reported honestly.

Measures:
  degree       TV distance real vs synth over the public degree bins
  clustering   average local clustering + transitivity, real vs synth (the
               headline statistic: published real value 0.6055)
  communities  Louvain (seed 7) community count + modularity, real vs synth
  motifs       motif shares (hub/leaf/broker/core via spec thresholds),
               real vs synth
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from spec import load_spec  # noqa: E402

DEG_BINS = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
            "deg_33_64", "deg_65p"]

_SPEC = load_spec()


def deg_class(d: int) -> str:
    edges = [int(_SPEC[k]) for k in ("deg_e2", "deg_e3", "deg_e5", "deg_e9",
                                     "deg_e17", "deg_e33", "deg_e65")]
    for name, lo in zip(reversed(DEG_BINS), reversed([1] + edges)):
        if d >= lo:
            return name
    return DEG_BINS[0]


def tv(a: Counter, b: Counter, grid: list[str]) -> float:
    ta, tb = max(1, sum(a.values())), max(1, sum(b.values()))
    return 0.5 * sum(abs(a[c] / ta - b[c] / tb) for c in grid)


def graph_stats(g) -> dict:
    import networkx as nx
    deg = dict(g.degree())
    cc = nx.clustering(g)
    tris = nx.triangles(g)
    hub_min = int(_SPEC["hub_degree_min"])
    broker_min = int(_SPEC["broker_degree_min"])
    comms = nx.community.louvain_communities(g, seed=int(_SPEC["louvain_seed"]))
    n = g.number_of_nodes()
    return {
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "degree_classes": Counter(deg_class(d) for d in deg.values()),
        "avg_clustering": round(nx.average_clustering(g), 4),
        "transitivity": round(nx.transitivity(g), 4),
        "n_communities": len(comms),
        "modularity": round(nx.community.modularity(g, comms), 4),
        "motif_shares": {
            "hub": round(sum(1 for d in deg.values() if d >= hub_min) / n, 4),
            "leaf": round(sum(1 for d in deg.values() if d == 1) / n, 4),
            "broker": round(sum(
                1 for v, d in deg.items()
                if d >= broker_min and cc[v] < 1 / 5) / n, 4),
            "core": round(sum(
                1 for v, d in deg.items()
                if d >= int(_SPEC["core_degree_min"]) and
                (d > 1 and 2 * tris[v] * 5 >= 3 * d * (d - 1))) / n, 4),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "synth_graph.graphml"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    import networkx as nx
    real = graph_stats(nx.read_graphml(ROOT / "facebook_full_features.graphml"))
    synth = graph_stats(nx.read_graphml(synth_p))

    report = {
        "note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)",
        "degree_tv_real_vs_synth": round(
            tv(real.pop("degree_classes"), synth.pop("degree_classes"),
               DEG_BINS), 4),
        "real": real,
        "synth": synth,
        "gaps_note": "declared gaps (degree-community dependence, feature "
                     "identity, circle rosters, assortativity) are expected "
                     "to show here; the clustering line is the generator's "
                     "closure-pass target",
    }
    out = ROOT / "artifacts" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "realism_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("-> artifacts/reports/realism_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

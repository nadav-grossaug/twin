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
  affiliation  shared-attribute key-group structure (education/work): group
               counts, mean internal density per public size bin, and the
               share of edges whose endpoints share >= 1 feature key — the
               local-community-connectivity moments the affiliation release
               targets
  link judge   the enclave's affil_link_mlp teacher scores edge vs non-edge
               pairs by their shared-category profiles on BOTH graphs; a
               realistic synthetic graph should show a similar score
               separation (requires torch + the trained artifact; skipped
               otherwise)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from ingest_graphml import FEATURE_KEY_RE, raw_category_of  # noqa: E402
from spec import AFFIL_RELEASE_CATS, CATEGORIES, category_of, load_spec  # noqa: E402

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


def node_keys(g) -> dict:
    """node -> {(public_category, feature_key)} from GraphML attributes
    (the ingest's own key convention — works on real and synthetic alike)."""
    out = {}
    for v, d in g.nodes(data=True):
        ks = set()
        for key, val in d.items():
            m = FEATURE_KEY_RE.match(key)
            if m and val:
                ks.add((category_of(raw_category_of(m.group(3) or "")), key))
        out[v] = ks
    return out


def affil_stats(g, keys: dict) -> dict:
    """Affiliation key-group moments over the released categories, on the
    PUBLIC size bins — plus the shared-key edge share (all categories)."""
    g5, g9, g17 = (int(_SPEC[k]) for k in ("affil_g5", "affil_g9", "affil_g17"))

    def bin_of(sz: int) -> str:
        return ("g_2_4" if sz < g5 else "g_5_8" if sz < g9
                else "g_9_16" if sz < g17 else "g_17p")

    holders: dict[str, list] = {}
    for v, ks in keys.items():
        for cat, key in ks:
            if cat in AFFIL_RELEASE_CATS:
                holders.setdefault(key, []).append(v)
    per_cat: dict[str, dict] = {}
    for key, hs in holders.items():
        if len(hs) < 2:
            continue
        cat = category_of(raw_category_of(
            FEATURE_KEY_RE.match(key).group(3) or ""))
        internal = sum(1 for i, a in enumerate(hs) for b in hs[i + 1:]
                       if g.has_edge(a, b))
        dens = internal / (len(hs) * (len(hs) - 1) / 2)
        cell = per_cat.setdefault(cat, {}).setdefault(
            bin_of(len(hs)), {"n_groups": 0, "dsum": 0.0})
        cell["n_groups"] += 1
        cell["dsum"] += dens
    report = {}
    for cat, bins in sorted(per_cat.items()):
        report[cat] = {
            b: {"n_groups": c["n_groups"],
                "mean_density": round(c["dsum"] / c["n_groups"], 4)}
            for b, c in sorted(bins.items())}
    shared = sum(1 for a, b in g.edges() if keys[a] & keys[b])
    report["shared_key_edge_frac"] = round(
        shared / max(1, g.number_of_edges()), 4)
    return report


def link_judge(g, keys: dict, rng) -> dict | None:
    """Score edge vs non-edge pairs with the trained affil_link_mlp (the
    affiliation link teacher): shared-key counts per public category +
    min/max richness -> P(edge) logit. ENCLAVE-ONLY judge."""
    reg = ROOT / "models_out" / "registry.json"
    if not reg.exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    pt = ROOT / "models_out" / "minimodels" / (active or "") / "affil_link_mlp.pt"
    if not active or not pt.exists():
        return None
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    rich = {v: len({c for c, _ in ks}) for v, ks in keys.items()}
    nodes = list(g.nodes())
    edges = list(g.edges())
    k = min(5000, len(edges))
    sample = [edges[int(i)] for i in rng.choice(len(edges), size=k,
                                                replace=False)]
    non_edges = []
    while len(non_edges) < k:
        a, b = (nodes[int(i)] for i in rng.integers(0, len(nodes), 2))
        if a != b and not g.has_edge(a, b):
            non_edges.append((a, b))

    def score(pairs) -> float:
        xs = []
        for a, b in pairs:
            shared = Counter(c for c, key in keys[a] & keys[b])
            xs.append([float(shared.get(c, 0)) for c in CATEGORIES]
                      + [float(min(rich[a], rich[b])),
                         float(max(rich[a], rich[b]))])
        x = torch.tensor(xs, dtype=torch.float32)
        h = torch.relu(x @ sd["net.0.weight"].T + sd["net.0.bias"])
        return float(torch.sigmoid(h @ sd["net.2.weight"].T
                                   + sd["net.2.bias"]).mean())

    return {"mean_p_edge": round(score(sample), 4),
            "mean_p_nonedge": round(score(non_edges), 4),
            "n_pairs_each": k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth")
    args = ap.parse_args()
    synth_p = ROOT / args.synth / "synth_graph.graphml"
    if not synth_p.exists():
        print("ERROR: run `make synth` first", file=sys.stderr)
        return 1
    import networkx as nx
    import numpy as np
    real_g = nx.read_graphml(ROOT / "facebook_full_features.graphml")
    synth_g = nx.read_graphml(synth_p)
    real = graph_stats(real_g)
    synth = graph_stats(synth_g)
    real_keys = node_keys(real_g)
    synth_keys = node_keys(synth_g)
    rng = np.random.default_rng(7)

    report = {
        "note": "ENCLAVE-ONLY realism audit (graded, not pass/fail)",
        "degree_tv_real_vs_synth": round(
            tv(real.pop("degree_classes"), synth.pop("degree_classes"),
               DEG_BINS), 4),
        "real": real,
        "synth": synth,
        "affiliation": {
            "real": affil_stats(real_g, real_keys),
            "synth": affil_stats(synth_g, synth_keys),
            "note": "group moments over the released education/work size "
                    "bins; shared_key_edge_frac spans ALL categories",
        },
        "affil_link_judge": {
            "real": link_judge(real_g, real_keys, rng),
            "synth": link_judge(synth_g, synth_keys, rng),
            "note": "affil_link_mlp teacher score separation (edges vs "
                    "uniform non-edges); None = artifact or torch missing",
        },
        "gaps_note": "declared gaps (degree-community dependence, "
                     "non-affiliation feature identity, circle rosters, "
                     "assortativity) are expected to show here; the "
                     "clustering line is the generator's closure-pass "
                     "target and the affiliation block its key-group "
                     "target",
    }
    out = ROOT / "artifacts" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "realism_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("-> artifacts/reports/realism_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

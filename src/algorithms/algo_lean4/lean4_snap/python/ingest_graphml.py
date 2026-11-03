#!/usr/bin/env python3
"""Ingest the GraphML social graph into enclave TSVs for dbt staging.

Writes (under artifacts/ingest/ by default):
  nodes.tsv          node_id, is_ego, nx_degree, nx_triangles, community_id,
                     n_circles, n_egos
  edges.tsv          src, dst  (canonical src < dst, deduplicated)
  node_features.tsv  node_id, feature_key, raw_category, value (always 1 —
                     sparse encoding: absent means 0/unobserved)
  circles.tsv        node_id, ego_id, circle_label
  ingest_meta.json   graphml sha256 fingerprint, counts, Louvain stats

The nx_* columns are CROSS-CHECK values: dbt feat_nodes recomputes degree,
triangles and wedges in SQL from edges.tsv and q_data_integrity asserts
agreement. Community detection (Louvain, seeded — the spec var louvain_seed)
runs here because it is iterative graph clustering, not relational algebra;
labels are re-indexed 0..k-1 by descending size so ranks are stable.

Synthetic round-trip: the SAME script re-ingests generated GraphML
(--graphml synth/synth_graph.graphml --out <scratch>/ingest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import RAW_CATEGORIES, load_spec  # noqa: E402

FEATURE_KEY_RE = re.compile(r"^ego_(\d+)__f(\d+)(?:_(.*))?$")
CIRCLES_KEY_RE = re.compile(r"^ego_(\d+)__circles$")

# longest-first so 'first_name' wins over 'name', 'location' over 'locale'
_CATS_BY_LEN = sorted(RAW_CATEGORIES, key=len, reverse=True)


def raw_category_of(slug: str) -> str:
    for cat in _CATS_BY_LEN:
        if slug == cat or slug.startswith(cat + "_"):
            return cat
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphml", type=Path,
                    default=ROOT / "facebook_full_features.graphml")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "ingest")
    args = ap.parse_args()

    seed = int(load_spec()["louvain_seed"])
    data = args.graphml.read_bytes()
    fingerprint = hashlib.sha256(data).hexdigest()
    g = nx.read_graphml(args.graphml)
    g = nx.relabel_nodes(g, {v: int(v) for v in g.nodes})

    # --- edges (canonical, deduplicated, no self-loops) ---------------------
    edges = sorted({(min(u, v), max(u, v)) for u, v in g.edges() if u != v})

    # --- cross-check structure numbers --------------------------------------
    degree = dict(g.degree())
    triangles = nx.triangles(g)

    # --- communities (Louvain, seeded; labels 0..k-1 by size desc) ----------
    comms = nx.community.louvain_communities(g, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    modularity = nx.community.modularity(g, comms)
    community_of = {v: i for i, c in enumerate(comms) for v in c}

    # --- features / circles / ego flags -------------------------------------
    feature_rows: list[tuple[int, str, str, int]] = []
    circle_rows: list[tuple[int, int, str]] = []
    node_rows: list[tuple[int, int, int, int, int, int, int]] = []
    for v in sorted(g.nodes):
        d = g.nodes[v]
        n_circles = 0
        n_egos = 0
        for key, val in d.items():
            m = CIRCLES_KEY_RE.match(key)
            if m:
                n_egos += 1
                ego = int(m.group(1))
                if val:
                    for label in str(val).split("|"):
                        circle_rows.append((v, ego, label))
                        n_circles += 1
                continue
            m = FEATURE_KEY_RE.match(key)
            if m:
                slug = m.group(3) or ""
                feature_rows.append((v, key, raw_category_of(slug), int(val)))
        node_rows.append((
            v,
            1 if d.get("is_dataset_ego") else 0,
            degree[v],
            triangles[v],
            community_of[v],
            n_circles,
            n_egos,
        ))

    # --- write ---------------------------------------------------------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def tsv(name: str, header: list[str], rows) -> None:
        with (out / name).open("w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")

    tsv("nodes.tsv",
        ["node_id", "is_ego", "nx_degree", "nx_triangles", "community_id",
         "n_circles", "n_egos"], node_rows)
    tsv("edges.tsv", ["src", "dst"], edges)
    tsv("node_features.tsv",
        ["node_id", "feature_key", "raw_category", "value"], feature_rows)
    tsv("circles.tsv", ["node_id", "ego_id", "circle_label"], circle_rows)

    meta = {
        "source_path": str(args.graphml),
        "graphml_sha256": fingerprint,
        "n_nodes": g.number_of_nodes(),
        "n_edges": len(edges),
        "n_feature_rows": len(feature_rows),
        "n_circle_rows": len(circle_rows),
        "n_communities": len(comms),
        "modularity": round(modularity, 6),
        "louvain_seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "ingest_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ingest: {meta['n_nodes']} nodes, {meta['n_edges']} edges, "
          f"{len(feature_rows)} feature rows, {len(circle_rows)} circle rows, "
          f"{meta['n_communities']} communities (mod {meta['modularity']}) -> {out}")


if __name__ == "__main__":
    main()

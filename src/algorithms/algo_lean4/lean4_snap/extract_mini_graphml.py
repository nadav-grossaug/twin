#!/usr/bin/env python3
"""Extract a mini GraphML sample by SNOWBALL sampling: seed one node, add
up to k random neighbors, then repeatedly pick a random node already in the
sample and add up to k of its neighbors, until N nodes are collected; write
the induced subgraph.

    python3 extract_mini_graphml.py -i facebook_full_features.graphml --nodes 60

writes facebook_full_features.60.graphml next to the input: the induced
subgraph on the 60 snowball-sampled nodes, keeping every node attribute
(sparse feature keys, circles, ego markers) and only the edges between
sampled nodes. Key ids are rewritten to the repo's descriptive `dN_<short>`
convention so the mini file matches the full file's format. Snowball
sampling keeps the sample connected-ish and locally dense (unlike uniform
sampling, which keeps almost no edges).
"""
from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx

GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"


def descriptive_key_ids(path: Path) -> None:
    """Rename auto-generated GraphML key ids (d0, d1, ...) to carry a short
    description of the attribute (same convention as the full converter)."""
    ET.register_namespace("", GRAPHML_NS)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(path)
    root = tree.getroot()
    rename = {}
    for key in root.findall(f"{{{GRAPHML_NS}}}key"):
        short = key.get("attr.name", "")
        short = re.sub(r"^ego_\d+__(f\d+_)?", "", short)
        short = re.sub(r"(_id)?_anonymized_feature(_\d+)?$", "", short)
        short = re.sub(r"[^A-Za-z0-9]+", "_", short).strip("_")[:40]
        old = key.get("id")
        rename[old] = f"{old}_{short}" if short else old
        key.set("id", rename[old])
    for data in root.iter(f"{{{GRAPHML_NS}}}data"):
        data.set("key", rename.get(data.get("key"), data.get("key")))
    tree.write(path, xml_declaration=True, encoding="utf-8")


def snowball(g: "nx.Graph", n_target: int, k: int, rng: random.Random) -> list:
    """Seed one node from src; then repeat: pick a random node already in the
    sample (the seed first), add up to k of its not-yet-sampled neighbors;
    until n_target nodes are collected. Re-seeds from src if the frontier is
    ever exhausted (cannot happen on a connected graph before n_target)."""
    all_nodes = sorted(g.nodes)
    dst: list = []
    dst_set: set = set()

    def add(v) -> None:
        if v not in dst_set and len(dst) < n_target:
            dst_set.add(v)
            dst.append(v)

    add(rng.choice(all_nodes))
    fuel = n_target * 100
    while len(dst) < n_target and fuel > 0:
        fuel -= 1
        cur = dst[rng.randrange(len(dst))]
        cands = [u for u in g.neighbors(cur) if u not in dst_set]
        if cands:
            for u in rng.sample(cands, min(k, len(cands))):
                add(u)
        elif all(u in dst_set for v in dst for u in g.neighbors(v)):
            add(rng.choice(all_nodes))  # whole frontier exhausted: re-seed
    if len(dst) < n_target:
        raise SystemExit("snowball: fuel exhausted before reaching the target")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="source GraphML file")
    ap.add_argument("--nodes", type=int, required=True,
                    help="number of nodes to sample")
    ap.add_argument("-k", "--neighbors", type=int, default=5,
                    help="neighbors added per snowball step (default 5)")
    ap.add_argument("--seed", type=int, default=7,
                    help="sampling seed (default 7, reproducible)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: <input stem>.<nodes>.graphml)")
    args = ap.parse_args()

    g = nx.read_graphml(args.input)
    if not 0 < args.nodes <= g.number_of_nodes():
        ap.error(f"--nodes must be in 1..{g.number_of_nodes()}")
    if args.neighbors < 1:
        ap.error("--neighbors must be >= 1")

    sampled = snowball(g, args.nodes, args.neighbors, random.Random(args.seed))
    mini = g.subgraph(sampled).copy()
    mini.graph.update(g.graph)
    mini.graph["sampled_from"] = args.input.name
    mini.graph["sample_seed"] = args.seed
    mini.graph["sample_mode"] = "snowball"
    mini.graph["sample_k"] = args.neighbors

    out = args.out or args.input.with_suffix("").with_suffix(
        f".{args.nodes}.graphml")
    nx.write_graphml(mini, out)
    descriptive_key_ids(out)
    print(f"{out}: {mini.number_of_nodes()} nodes, "
          f"{mini.number_of_edges()} induced edges "
          f"(snowball k={args.neighbors} from {g.number_of_nodes()}/"
          f"{g.number_of_edges()}, seed {args.seed})")


if __name__ == "__main__":
    main()

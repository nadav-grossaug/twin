"""Package E: community strength of synthetic graphs vs the real graph (the diagnostic of docs/snap/v2-diagnostic.md
section 1, Louvain only). Eval zone: reads the real graph. Run from the repo root:
  PYTHONPATH=src:snap/inside:snap/eval /opt/venv/bin/python3 tools/v2_runs/e/community_check.py [graph.npz ...]
Default graphs: the real graph, v1.1 eps-inf seed 0, E eps-inf seed 0 (tools/v2_runs/e/synthetic).
Writes tools/v2_runs/e/community.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from graph_aim.transfer.graph_io import load_attributed_graph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT = {
    "real": ROOT / "snap" / "inside" / "data" / "processed" / "attributed_graph.npz",
    "v1.1 eps inf s0": ROOT / "snap" / "outside" / "synthetic_v1_1" / "graph_aim_epsinf_seed0.npz",
    "v1.1 eps 4 s0": ROOT / "snap" / "outside" / "synthetic_v1_1" / "graph_aim_eps4_seed0.npz",
    "E eps inf s0": HERE / "synthetic" / "graph_aim_epsinf_seed0.npz",
    "E eps inf s1": HERE / "synthetic" / "graph_aim_epsinf_seed1.npz",
    "E eps inf s2": HERE / "synthetic" / "graph_aim_epsinf_seed2.npz",
    "E eps 4 s0": HERE / "synthetic" / "graph_aim_eps4_seed0.npz",
    "E eps 4 s1": HERE / "synthetic" / "graph_aim_eps4_seed1.npz",
    "E eps 4 s2": HERE / "synthetic" / "graph_aim_eps4_seed2.npz",
}


def stats(name: str, path: Path) -> dict:
    g = load_attributed_graph(path)
    n, e = g.n, np.asarray(g.edges)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, e.tolist()))
    parts = [set(p) for p in nx.community.louvain_communities(G, seed=0)]
    sizes = np.array(sorted((len(p) for p in parts), reverse=True))
    tot = deg.sum()
    cond, w = [], []
    for p in parts:
        idx = np.fromiter(p, dtype=np.int64)
        vol = deg[idx].sum()
        internal = adj[idx][:, idx].sum()
        den = min(vol, tot - vol)
        cond.append((vol - internal) / den if den > 0 else 0.0)
        w.append(len(idx))
    cond, w = np.asarray(cond), np.asarray(w, dtype=float)
    lam = np.sort(eigsh(adj, k=32, which="LA", return_eigenvectors=False, v0=np.ones(n) / np.sqrt(n)))[::-1]
    return {
        "graph": name, "edges": len(e), "modularity": nx.community.modularity(G, parts), "parts": len(parts),
        "size_median": float(np.median(sizes)), "size_q75": float(np.percentile(sizes, 75)), "size_max": int(sizes[0]),
        "size_node_weighted_median": float(np.median(np.repeat(sizes, sizes))),
        "share_top5": sizes[:5].sum() / sizes.sum(), "share_parts_ge100": sizes[sizes >= 100].sum() / sizes.sum(),
        "conductance_node_weighted": float((cond * w).sum() / w.sum()),
        "eig1": lam[0], "eig2": lam[1], "eig3": lam[2], "eig4": lam[3], "eig10": lam[9], "eig20": lam[19], "eig32": lam[31],
        "avg_clustering": nx.average_clustering(G),
    }


def main() -> None:
    graphs = {Path(p).stem: Path(p) for p in sys.argv[1:]} if len(sys.argv) > 1 else DEFAULT
    rows = [stats(k, p) for k, p in graphs.items() if p.exists()]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    df.to_csv(HERE / "community.csv", index=False)


if __name__ == "__main__":
    main()

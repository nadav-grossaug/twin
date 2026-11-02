"""Throwaway: diagnose the eps=inf degree-assortativity regression.

(i)  assortativity of real, before-ablation (tools/before_graphs) and new (snap/outside/synthetic) epsinf
     graphs, plus any sweep-variant epsinf graphs found under tools/sweep/*/synthetic.
(ii) who holds the hubs: attribute values of the top-degree nodes, real vs new synthetic.
(iii) degree mixing around hubs: mean neighbour degree of the top-50 nodes, and share of hub edges whose
     other end has degree < 16, real vs before vs after.

Run (repo root): PYTHONPATH=src python3 tools/assort_diag.py
"""

from pathlib import Path

import numpy as np

from graph_aim.eval.evaluate import adjacency, align_columns, degrees, _degree_assortativity
from graph_aim.transfer.graph_io import load_attributed_graph

ROOT = Path(__file__).resolve().parent.parent


def assort(g) -> float:
    return _degree_assortativity(adjacency(g), degrees(g))


def hub_stats(g, top: int = 50) -> tuple[float, float]:
    """(mean neighbour degree of the top-`top` nodes, share of their edges to nodes of degree < 16)."""
    d = degrees(g)
    hubs = set(np.argsort(d)[-top:].tolist())
    mask = np.isin(g.edges, list(hubs))
    nb = []
    for (u, v), (mu, mv) in zip(g.edges, mask):
        if mu:
            nb.append(d[v])
        if mv:
            nb.append(d[u])
    nb = np.array(nb)
    return float(nb.mean()), float((nb < 16).mean())


def main() -> None:
    real = load_attributed_graph(ROOT / "snap" / "inside" / "data" / "processed" / "attributed_graph.npz")
    print(f"real: assort {assort(real):+.4f}")

    groups = {
        "before (no attr_degree)": ROOT / "tools" / "before_graphs",
        "after (shipped cfg)": ROOT / "snap" / "outside" / "synthetic",
    }
    for p in sorted((ROOT / "tools" / "sweep").glob("*/synthetic")):
        groups[f"sweep {p.parent.name}"] = p

    results = {}
    for name, folder in groups.items():
        vals = []
        for s in range(3):
            f = folder / f"graph_aim_epsinf_seed{s}.npz"
            if not f.exists():
                continue
            vals.append(assort(align_columns(load_attributed_graph(f), real)))
        if vals:
            results[name] = vals
            print(f"{name}: epsinf assort {np.mean(vals):+.4f} ± {np.std(vals, ddof=1):.4f} "
                  f"(abs err {np.mean([abs(v - assort(real)) for v in vals]):.4f})")

    # hubs: which attribute values hold the top degrees
    print("\ntop-20 hub attribute values (real vs after epsinf seed0):")
    syn = align_columns(load_attributed_graph(ROOT / "snap" / "outside" / "synthetic" / "graph_aim_epsinf_seed0.npz"), real)
    for g, label in [(real, "real"), (syn, "after")]:
        d = degrees(g)
        top = np.argsort(d)[-20:][::-1]
        for col in ["birthday", "school"]:
            labels = [g.domains[col][c] for c in g.attrs[col].to_numpy()[top]]
            from collections import Counter
            print(f"  {label} {col}: {dict(Counter(labels))}")

    print("\nhub wiring (top-50 nodes): mean neighbour degree / share of hub edges to degree<16 nodes")
    for label, g in [("real", real),
                     ("before s0", align_columns(load_attributed_graph(ROOT / "tools" / "before_graphs" / "graph_aim_epsinf_seed0.npz"), real)),
                     ("after s0", syn)]:
        mnd, lowshare = hub_stats(g)
        print(f"  {label}: {mnd:.1f} / {lowshare:.3f}")


if __name__ == "__main__":
    main()

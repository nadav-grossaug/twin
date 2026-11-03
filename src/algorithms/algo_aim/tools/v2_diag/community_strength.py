"""Throwaway (package A, composition round): how strong are the synthetic structure's communities, and would painting
attributes onto them reproduce the real graph's local concentration?  Eval zone: reads the real graph.

Run (repo root): PYTHONPATH=src:snap/inside:snap/eval /opt/venv/bin/python3 tools/v2_diag/community_strength.py
Writes tools/v2_diag/community_strength.csv (long form: graph, section, metric, value) and prints the tables quoted
in docs/snap/v2-diagnostic.md.  Spec: docs/snap/composition-plan.md section 3.

Graphs: the real graph, the v1.1 synthetic graphs (eps 4 and inf, seed 0), and derived ones: ``real_permuted``
(real structure, whole attribute rows permuted), ``<syn>_painted`` (synthetic structure, synthetic rows re-dealt
over the structure's Louvain communities, the painted upper bound) and ``real_painted`` (the same re-dealing on the
real structure, to show what the greedy painting does to a graph whose communities are strong enough).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from graph_aim.eval.composition import composition_metrics
from graph_aim.eval.discriminator import WalkSampler
from graph_aim.eval.evaluate import adjacency, align_columns, degrees
from graph_aim.transfer.binning import degree_bin
from graph_aim.transfer.graph_io import load_attributed_graph
from graph_aim.transfer.types import AttributedGraph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools" / "v2_diag" / "community_strength.csv"
REAL = ROOT / "snap" / "inside" / "data" / "processed" / "attributed_graph.npz"
SYN = ROOT / "snap" / "outside" / "synthetic_v1_1"

GROUPS = [["school"], ["hometown"], ["location"], ["hometown", "location"]]
SIMPSON_BINS = [0, 16, 35, 60, 110, 1201]
MIN_DEGREE = 16
STRATA = [0, 1, 11, 113, 1201]  # attr_degree bins of the shipped config
PRIMARY = ["hometown", "location", "school"]  # sort keys of the painting, primary first
WALK_ATTRS = ["hometown", "location", "school"]
N_WALKS = 2000
BUCKETS = [(10, 40), (40, 120), (120, 201)]  # walk size k in [lo, hi)
N_EIGS = 32
THRESHOLD = 0.70

rows: list[dict] = []


def rec(graph: str, section: str, metric: str, value) -> None:
    rows.append({"graph": graph, "section": section, "metric": metric, "value": float(value)})


def nx_graph(g: AttributedGraph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(map(tuple, np.asarray(g.edges).tolist()))
    return G


# ------------------------------------------------------------------------------------ structure
def conductances(adj: sp.csr_matrix, deg: np.ndarray, parts: list[set]) -> np.ndarray:
    vol_total = deg.sum()
    out = []
    for p in parts:
        idx = np.fromiter(p, dtype=np.int64)
        vol = deg[idx].sum()
        internal = adj[idx][:, idx].sum()  # each internal edge twice
        cut = vol - internal
        den = min(vol, vol_total - vol)
        out.append(cut / den if den > 0 else 0.0)
    return np.asarray(out)


def partition_stats(name: str, method: str, G: nx.Graph, adj, deg, parts: list[set]) -> None:
    sizes = np.sort(np.array([len(p) for p in parts]))[::-1]
    rec(name, method, "modularity", nx.community.modularity(G, parts))
    rec(name, method, "n_parts", len(parts))
    for q in (0, 10, 25, 50, 75, 90, 100):
        rec(name, method, f"size_q{q}", np.percentile(sizes, q))
    rec(name, method, "size_top5_share", sizes[:5].sum() / sizes.sum())
    big = sizes >= 20
    rec(name, method, "n_parts_ge20", int(big.sum()))
    rec(name, method, "share_nodes_in_parts_ge20", sizes[big].sum() / sizes.sum())
    c = conductances(adj, deg, parts)
    w = np.array([len(p) for p in parts], dtype=float)
    rec(name, method, "conductance_mean", c.mean())
    rec(name, method, "conductance_node_weighted", (c * w).sum() / w.sum())
    rec(name, method, "conductance_median", np.median(c))


def structure_section(name: str, g: AttributedGraph) -> list[set]:
    """Louvain / label-propagation partitions and the top adjacency eigenvalues; returns the Louvain parts."""
    G = nx_graph(g)
    adj, deg = adjacency(g), degrees(g)
    louvain = [set(p) for p in nx.community.louvain_communities(G, seed=0)]
    partition_stats(name, "louvain", G, adj, deg, louvain)
    lpa = [set(p) for p in nx.community.label_propagation_communities(G)]
    partition_stats(name, "label_propagation", G, adj, deg, lpa)
    vals = eigsh(adj, k=N_EIGS, which="LA", return_eigenvectors=False, v0=np.ones(g.n) / np.sqrt(g.n))
    vals = np.sort(vals)[::-1]
    for i, v in enumerate(vals, start=1):
        rec(name, "spectrum", f"eig_{i:02d}", v)
    rec(name, "spectrum", "mean_degree", deg.mean())
    rec(name, "spectrum", "sqrt_mean_sq_over_mean", np.sqrt((deg.astype(float) ** 2).mean() / deg.mean()))
    return louvain


# -------------------------------------------------------------------------------------- painting
def paint(g: AttributedGraph, parts: list[set]) -> AttributedGraph:
    """Re-deal the attribute rows of ``g`` within each degree stratum: rows sorted by the primary group codes,
    nodes ordered community by community (largest first); the i-th sorted row goes to the i-th node."""
    deg = degrees(g)
    strata = degree_bin(deg, STRATA)
    comm = np.full(g.n, -1, dtype=np.int64)
    for i, p in enumerate(sorted(parts, key=len, reverse=True)):
        comm[np.fromiter(p, dtype=np.int64)] = i
    assert (comm >= 0).all()
    values = g.attrs.to_numpy().copy()
    new = values.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        node_order = idx[np.argsort(comm[idx], kind="stable")]
        keys = tuple(g.attrs[c].to_numpy()[idx] for c in reversed(PRIMARY))  # lexsort: last key is primary
        row_order = idx[np.lexsort(keys)]
        new[node_order] = values[row_order]
    attrs = pd.DataFrame(new, columns=g.attrs.columns)
    return AttributedGraph(g.n, g.edges, attrs, {c: list(d) for c, d in g.domains.items()})


def permute_rows(g: AttributedGraph, seed: int = 0) -> AttributedGraph:
    rng = np.random.default_rng(seed)
    attrs = g.attrs.iloc[rng.permutation(g.n)].reset_index(drop=True)
    return AttributedGraph(g.n, g.edges, attrs, {c: list(d) for c, d in g.domains.items()})


# ------------------------------------------------------------------------------------- composition
def simpson_section(name: str, real: AttributedGraph, g: AttributedGraph) -> None:
    m = composition_metrics(real, g, groups=GROUPS, bin_edges=SIMPSON_BINS, min_degree=MIN_DEGREE)
    for label, gm in m["groups"].items():
        sc = gm["scalars"]
        rec(name, f"simpson:{label}", "mean", sc["simpson_mean_syn"])
        for b in range(len(SIMPSON_BINS) - 1):
            rec(name, f"simpson:{label}", f"bin{b}", sc[f"simpson_bin{b}_syn"])
        rec(name, f"simpson:{label}", "two_hop_same_rate", sc["two_hop_same_rate_syn"])
        rec(name, f"simpson:{label}", "two_hop_indep", sc["two_hop_indep_syn"])
        rec(name, f"simpson:{label}", "dominant_share_tvd_vs_real", sc["dominant_share_tvd"])


def walk_shares(g: AttributedGraph, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(k, {attr: (N_WALKS, K) share matrix})`` for ``N_WALKS`` walks sampled exactly as the discriminator does."""
    sampler = WalkSampler()
    rng = np.random.default_rng(seed)
    ks, shares = [], {a: [] for a in WALK_ATTRS}
    for _ in range(N_WALKS):
        s = sampler(g, rng)
        p = s.payload
        ks.append(p.k)
        for a in WALK_ATTRS:
            codes = p.attrs[a].to_numpy(dtype=np.int64)
            shares[a].append(np.bincount(codes, minlength=len(g.domains[a])) / p.k)
    return np.asarray(ks), {a: np.asarray(v) for a, v in shares.items()}


def walk_section(name: str, g: AttributedGraph, seed: int = 0) -> dict[str, float]:
    """Mean over values of the across-walk variance of the value share, per attribute and bucket (seed 0 is the
    headline; another seed only records the per-bucket means, with a ``_seed<k>`` suffix, to show the spread)."""
    ks, shares = walk_shares(g, seed)
    suffix = "" if seed == 0 else f"_seed{seed}"
    out = {}
    for lo, hi in BUCKETS + [(10, 201)]:
        sel = (ks >= lo) & (ks < hi)
        tag = f"k{lo}-{hi - 1}" if (lo, hi) != (10, 201) else "k10-200"
        per_attr = []
        for a in WALK_ATTRS:
            v = shares[a][sel].var(axis=0, ddof=1).mean()
            if seed == 0:
                rec(name, "walks", f"share_var_{a}_{tag}", v)
            per_attr.append(v)
        out[tag] = float(np.mean(per_attr))
        if seed == 0:
            rec(name, "walks", f"n_walks_{tag}", sel.sum())
        rec(name, "walks", f"share_var_mean_{tag}{suffix}", out[tag])
    return out


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    real = load_attributed_graph(REAL)
    graphs = {"real": real}
    for eps in ("eps4", "epsinf"):
        graphs[f"syn_{eps}"] = align_columns(load_attributed_graph(SYN / f"graph_aim_{eps}_seed0.npz"), real)

    parts = {}
    for name, g in graphs.items():
        parts[name] = structure_section(name, g)
        print(f"structure {name} ({time.time() - t0:.0f}s)", flush=True)

    derived = {"real_permuted": permute_rows(real), "real_painted": paint(real, parts["real"])}
    for eps in ("eps4", "epsinf"):
        derived[f"syn_{eps}_painted"] = paint(graphs[f"syn_{eps}"], parts[f"syn_{eps}"])
    order = ["real", "real_permuted", "real_painted", "syn_eps4", "syn_eps4_painted", "syn_epsinf", "syn_epsinf_painted"]
    all_graphs = {**graphs, **derived}
    source = {n: "real" if n.startswith("real") else n.replace("_painted", "") for n in order}
    for name in order:
        g = all_graphs[name]
        # painting and permutation keep every attribute row: the 1-way and 2-way tables are the source graph's
        assert np.array_equal(np.sort(g.attrs.to_numpy(), axis=0), np.sort(all_graphs[source[name]].attrs.to_numpy(), axis=0))
        simpson_section(name, real, g)
    print(f"simpson done ({time.time() - t0:.0f}s)", flush=True)

    var = {0: {}, 1: {}}
    for seed in (0, 1):
        for name in order:
            var[seed][name] = walk_section(name, all_graphs[name], seed)
            print(f"walks {name} seed {seed} {var[seed][name]} ({time.time() - t0:.0f}s)", flush=True)

    # decision rule: painted synthetic >= THRESHOLD x real per-walk share variance in every size bucket
    tags = [f"k{lo}-{hi - 1}" for lo, hi in BUCKETS]
    for name in order:
        for tag in tags + ["k10-200"]:
            v, r, p = var[0][name][tag], var[0]["real"][tag], var[0]["real_permuted"][tag]
            rec(name, "decision", f"ratio_to_real_{tag}", v / r)
            rec(name, "decision", f"excess_over_null_{tag}", (v - p) / (r - p))
            rec(name, "decision", f"ratio_to_real_{tag}_seed1", var[1][name][tag] / var[1]["real"][tag])
    for eps in ("eps4", "epsinf"):
        name = f"syn_{eps}_painted"
        for seed in (0, 1):
            ok = all(var[seed][name][t] / var[seed]["real"][t] >= THRESHOLD for t in tags)
            rec(name, "decision", f"painted_reaches_70pct_all_buckets{'' if seed == 0 else '_seed1'}", ok)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("\n=== partitions ===")
    sub = df[df["section"].isin(["louvain", "label_propagation"])]
    print(sub.pivot_table(index=["section", "metric"], columns="graph", values="value", sort=False).round(3).to_string())
    print("\n=== spectrum (top adjacency eigenvalues) ===")
    sub = df[df["section"] == "spectrum"]
    print(sub.pivot_table(index="metric", columns="graph", values="value", sort=False).round(2).to_string())
    print("\n=== simpson by degree (min_degree 16) ===")
    sub = df[df["section"].str.startswith("simpson:")]
    print(sub.pivot_table(index=["section", "metric"], columns="graph", values="value", sort=False)[order].round(3).to_string())
    print("\n=== per-walk share variance (mean over values; x1000) ===")
    sub = df[(df["section"] == "walks") & df["metric"].str.startswith("share_var")].copy()
    sub["value"] *= 1000
    print(sub.pivot_table(index="metric", columns="graph", values="value", sort=False)[order].round(3).to_string())
    print("\n=== decision ===")
    sub = df[df["section"] == "decision"]
    print(sub.pivot_table(index="metric", columns="graph", values="value", sort=False)[order].round(3).to_string())
    print(f"\nwrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())

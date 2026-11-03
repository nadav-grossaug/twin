"""Throwaway (package E): structure-only exploration of (R, q) on a real v2 artifact.

Eval zone (reads the real graph for the reference numbers). Run from the repo root:
  PYTHONPATH=src:snap/inside:snap/eval /opt/venv/bin/python3 tools/v2_runs/e/explore_rq.py [artifact_dir]
Prints, per (R, q): spectrum score against the released spectrum, Louvain modularity / parts / median size,
node-weighted conductance, mean clustering, edge count, degree KS to the pool.
"""

from __future__ import annotations

import sys
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from graph_aim.outside import generate as G
from graph_aim.transfer.artifact import load

ART = sys.argv[1] if len(sys.argv) > 1 else "snap/transfer/artifacts_v2/graph_aim_epsinf_seed0"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
REWIRE = int(sys.argv[3]) if len(sys.argv) > 3 else 40
GRID_R = [1, 8, 16, 32, 64]
GRID_Q = [0.0, 0.5, 0.75, 0.9]

art = load(ART)
spec = [m for m in art.measurements.values() if m.kind == "spectrum"][0]
lam_rel = np.sort(np.asarray(spec.values, dtype=float))[::-1]
K = len(lam_rel)
n = art.schema.n
rng = np.random.default_rng(0)
attrs = G._sample_attributes(art, n, rng)
deg = G._sample_degrees(art, n, rng, attrs)
MODE = sys.argv[4] if len(sys.argv) > 4 else "blocks"  # "nodes" (spec 7.1) or "blocks" (Z constant per phase-one block)
perm = np.random.default_rng([0, 3]).permutation(n)
grp0, _ = G._group_codes(art, attrs, n)
blocks = G._phase_one_blocks(deg, grp0, np.random.default_rng([0, 1]))


def regions(R):
    z = np.empty(n, dtype=np.int64)
    if MODE == "nodes":
        z[perm] = np.arange(n) * R // n
        return z
    rng3 = np.random.default_rng([0, 3])
    units = [np.asarray(b) for b in blocks]
    rest = np.setdiff1d(np.arange(n), np.concatenate(units))
    units += [np.array([u]) for u in rest]
    order = rng3.permutation(len(units))
    load_ = np.zeros(R)
    for i in order:  # deal units in random order to the currently smallest region: near-equal sizes
        r = int(np.argmin(load_))
        z[units[i]] = r
        load_[r] += len(units[i])
    return z


def adjacency(edges):
    a = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return (a + a.T).tocsr()


def conductance_nw(adj, d, parts):
    tot = d.sum()
    out, w = [], []
    for p in parts:
        idx = np.fromiter(p, dtype=np.int64)
        vol = d[idx].sum()
        internal = adj[idx][:, idx].sum()
        den = min(vol, tot - vol)
        out.append((vol - internal) / den if den > 0 else 0.0)
        w.append(len(idx))
    out, w = np.asarray(out), np.asarray(w, dtype=float)
    return float((out * w).sum() / w.sum())


print(f"artifact {ART}  n={n}  K={K}  sigma={spec.sigma:.3f}  rounds={ROUNDS} rewire={REWIRE}")
print("released eig 1-6:", np.round(lam_rel[:6], 1), " 10/20/32:", np.round(lam_rel[[9, 19, K - 1]], 1))
print(f"{'R':>3} {'q':>5} {'score':>9} {'eig1-4':>24} {'mod':>6} {'parts':>5} {'med':>5} {'nwmed':>5} {'ge100':>5} {'cond':>6} {'clus':>6} {'m':>7} {'degKS':>7} {'assort':>6} {'trans':>6} {'short':>5} {'t':>5}")
print("real: modularity 0.835, parts 16, nwmed 423, ge100 0.95, cond 0.038, clus 0.606, m 88234, assort 0.064, trans 0.519")
for R in GRID_R:
    for q in GRID_Q:
        if R == 1 and q > 0:
            continue
        t0 = time.time()
        z = regions(R) if R > 1 else None
        e = G._calibrated_structure(art, deg, attrs, 0, rounds=ROUNDS, rewire_rounds=REWIRE,
                                    regions=z, within_region_frac=q)
        adj = adjacency(e)
        lam = np.sort(eigsh(adj, k=K, which="LA", return_eigenvectors=False, v0=np.ones(n) / np.sqrt(n)))[::-1]
        score = float(((lam - lam_rel) ** 2).sum() / (spec.sigma**2 + 1))
        Gx = nx.Graph()
        Gx.add_nodes_from(range(n))
        Gx.add_edges_from(map(tuple, e.tolist()))
        parts = nx.community.louvain_communities(Gx, seed=0)
        mod = nx.community.modularity(Gx, parts)
        d = np.asarray(adj.sum(axis=1)).ravel()
        sizes = sorted(len(p) for p in parts)
        cond = conductance_nw(adj, d, parts)
        clus = nx.average_clustering(Gx)
        ks = np.abs(np.sort(d) - np.sort(deg)).max()
        sz = np.asarray(sizes)
        nwmed = int(np.median(np.repeat(sz, sz)))  # part size of the median node
        ge100 = sz[sz >= 100].sum() / sz.sum()
        assort = nx.degree_assortativity_coefficient(Gx)
        trans = nx.transitivity(Gx)
        short = int((deg - d).max())  # largest degree shortfall of a node
        print(f"{R:>3} {q:>5.2f} {score:>9.1f} {str(np.round(lam[:4]).astype(int)):>24} {mod:>6.3f} {len(parts):>5} "
              f"{int(np.median(sizes)):>5} {nwmed:>5} {ge100:>5.2f} {cond:>6.3f} {clus:>6.3f} {len(e):>7} {int(ks):>7} "
              f"{assort:>6.3f} {trans:>6.3f} {short:>5} {time.time() - t0:>5.1f}")

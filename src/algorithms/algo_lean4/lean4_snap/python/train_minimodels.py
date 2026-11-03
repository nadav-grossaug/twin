#!/usr/bin/env python3
"""Fit the social-graph minimodels into the content-addressed artifact store.

Five models, one per learning target, each gated by a frozen non-ML baseline
(honest negatives are kept):

  degree_sage        P(degree class | neighborhood structure) via 2-layer
                     GraphSAGE over degree-free fixed-random node features
                     (mean-aggregated random vectors concentrate with degree,
                     so degree is inferable from message passing alone, never
                     fed as an input). gate: val acc > majority share + 0.02
  community_gnn      unsupervised GAE embeddings + seeded KMeans(k = number
                     of Louvain communities). gate: ARI vs the seeded Louvain
                     labels > 0.30 (all nodes)
  triad_vgae         PyG VGAE link prediction with RandomLinkSplit (val edges
                     never enter message passing). gate: val AUC >
                     common-neighbor-count baseline AUC + 0.01 — at
                     transitivity 0.52 the baseline is strong and the VGAE is
                     EXPECTED to lose; the negative is kept
  featcond_encoder   P(category presence | degree class one-hot + structural
                     embedding) -> 11 sigmoid heads. gate: mean per-category
                     val AUC > 0.55
  ego_motif_pool     motif presence from the mean-pooled 1-hop ego feature
                     vector (degree-class + cc-class one-hots, self included).
                     gate: macro-F1 > per-motif majority baseline

Split: 80/20 ON NODE ID HASH, fixed before fitting (the privacy unit is the
person/node, so leakage control is node-level). Artifacts: content-addressed
via python/artifacts.py (FLOW_RETRAIN=1 to activate a retrain;
FLOW_DEVICE=cpu for the deterministic mode).
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifacts import train_or_reuse  # noqa: E402
from spec import CATEGORIES, spec_version  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "artifacts" / "enclave.duckdb"

DEGREE_CLASSES = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
                  "deg_17_32", "deg_33_64", "deg_65p"]
CC_CLASSES = ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80",
              "cc_80_99", "cc_one"]
MOTIFS = ["motif_hub", "motif_leaf", "motif_broker", "motif_bridge",
          "motif_butterfly", "motif_core"]
MOTIF_COLS = ["is_hub", "is_leaf", "is_broker", "is_bridge",
              "is_butterfly", "is_core"]
# affiliation categories: feature keys here name shared real-world anchors
# (same school/degree, same employer, same hometown value, ...)
AFFIL_CATS = ["education", "work", "hometown", "location", "languages"]
AFFIL_MIN_NODES = 5       # affil_ae input keys must be held by >= 5 nodes
NEG_RATIO = 3             # affil_link_mlp non-edge : edge sampling ratio

EPOCHS = int(os.environ.get("FLOW_EPOCHS", "80"))
SEED = 1234
FEAT_SEED = 4242          # fixed-random node features (degree-free inputs)
FEAT_DIM = 16
EMB_DIM = 32
VAL_PCT = 20              # nodes with sha256(node_id) % 100 >= 80 held out


def pick_device() -> torch.device:
    forced = os.environ.get("FLOW_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_train_node(node_id: int) -> bool:
    """Deterministic node-level 80/20 split."""
    h = int(hashlib.sha256(str(int(node_id)).encode()).hexdigest(), 16)
    return h % 100 < 100 - VAL_PCT


def fixed_random_features(n: int, dim: int = FEAT_DIM,
                          seed: int = FEAT_SEED) -> torch.Tensor:
    """Degree-free, node-varying inputs: a fixed seeded Gaussian per node.

    With mean aggregation, the variance of an aggregated neighborhood shrinks
    like 1/degree — degree-class signal reaches the model only through
    message passing, never as an explicit feature."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g)


# ---------------------------------------------------------------------------
# metrics without sklearn (numpy only; deterministic)
# ---------------------------------------------------------------------------
def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC with average ranks for ties."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    c = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(c, (ia, ib), 1)

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(c.astype(np.float64)).sum()
    a_i = comb2(c.sum(1).astype(np.float64)).sum()
    b_j = comb2(c.sum(0).astype(np.float64)).sum()
    n = comb2(float(len(a)))
    if n == 0:
        return 1.0
    expected = a_i * b_j / n
    maximum = (a_i + b_j) / 2.0
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 100):
    """Seeded Lloyd's algorithm (numpy; no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        new_centers = np.stack([
            x[labels == j].mean(0) if (labels == j).any()
            else x[rng.integers(len(x))]
            for j in range(k)])
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def binary_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


# ---------------------------------------------------------------------------
# architectures (rebuilt identically by dbt/models/ml/ml_graph_scores.py)
# ---------------------------------------------------------------------------
class DegreeSage(nn.Module):
    def __init__(self, n_in: int = FEAT_DIM, hidden: int = 32,
                 n_classes: int = 8):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(n_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h)


class GCNEncoder(nn.Module):
    def __init__(self, n_in: int = FEAT_DIM, hidden: int = 64,
                 out: int = EMB_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(n_in, hidden)
        self.conv2 = GCNConv(hidden, out)

    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)


class VariationalGCNEncoder(nn.Module):
    def __init__(self, n_in: int = FEAT_DIM, hidden: int = 64,
                 out: int = EMB_DIM):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(n_in, hidden)
        self.conv_mu = GCNConv(hidden, out)
        self.conv_logstd = GCNConv(hidden, out)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class FeatCondEncoder(nn.Module):
    """degree-class one-hot + structural embedding -> category presence."""

    def __init__(self, n_in: int = 8 + EMB_DIM, hidden: int = 32,
                 n_cats: int = 11):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_cats))

    def forward(self, x):
        return self.net(x)


class EgoMotifPool(nn.Module):
    """mean-pooled 1-hop ego feature vector -> motif presence logits."""

    def __init__(self, n_in: int = len(DEGREE_CLASSES) + len(CC_CLASSES),
                 hidden: int = 32, n_motifs: int = len(MOTIFS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_motifs))

    def forward(self, x):
        return self.net(x)


class AffilAE(nn.Module):
    """affiliation-profile autoencoder: binary key vector -> 16-dim code ->
    reconstruction logits."""

    def __init__(self, n_keys: int, hidden: int = 64, code: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_keys, hidden), nn.ReLU(),
                                 nn.Linear(hidden, code))
        self.dec = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_keys))

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        return self.dec(self.enc(x))


class PairMLP(nn.Module):
    """shared-key pair features -> P(edge) logit."""

    def __init__(self, n_in: int = len(CATEGORIES) + 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class KeyCommunityMLP(nn.Module):
    """affiliation representation -> structural community rank (9-way)."""

    def __init__(self, n_in: int, hidden: int = 32, n_ranks: int = 9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_ranks))

    def forward(self, x):
        return self.net(x)


class KeyCohesionMLP(nn.Module):
    """(category one-hot + log2 group size) -> internal density."""

    def __init__(self, n_in: int = len(AFFIL_CATS) + 1, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
def train_loop(model, xs, ys, device, epochs, lr=1e-2, batch=512,
               sample_weights=None, binary=False):
    """Mini-batch supervised training; sample_weights is the dedup/weighting
    seam (None = uniform). binary=True uses BCEWithLogits on float targets."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if binary:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    ws = (torch.ones(len(xs)) if sample_weights is None
          else torch.as_tensor(sample_weights, dtype=torch.float32))
    model.train()
    n = len(xs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            per = loss_fn(model(xs[idx].to(device)), ys[idx].to(device))
            if per.dim() > 1:
                per = per.mean(dim=1)
            w = ws[idx].to(device)
            loss = (per * w).sum() / w.sum()
            loss.backward()
            opt.step()
    model.eval()


def state_bytes(model: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.cpu().state_dict(), buf)
    return buf.getvalue()


def main() -> int:
    if not DB.exists():
        print("ERROR: run `make dbt` first", file=sys.stderr)
        return 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    if str(device) == "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)

    con = duckdb.connect(str(DB), read_only=True)
    node_rows = con.execute(
        "select node_id, degree_class, cc_class, community_id, "
        "       is_hub, is_leaf, is_broker, is_bridge, is_butterfly, is_core "
        "from feat_nodes order by node_id").fetchall()
    edge_rows = con.execute(
        "select src, dst from stg_edges order by src, dst").fetchall()
    cat_rows = con.execute(
        "select distinct node_id, category from stg_node_features").fetchall()
    key_rows = con.execute(
        "select node_id, feature_key, category from stg_node_features").fetchall()
    kg_rows = con.execute(
        "select feature_key, category, n_nodes, density from mart_key_groups "
        f"where category in ({','.join(repr(c) for c in AFFIL_CATS)})").fetchall()
    con.close()

    node_ids = np.array([int(r[0]) for r in node_rows], dtype=np.int64)
    n = len(node_ids)
    idx_of = {v: i for i, v in enumerate(node_ids)}
    deg_idx = {c: i for i, c in enumerate(DEGREE_CLASSES)}
    cc_idx = {c: i for i, c in enumerate(CC_CLASSES)}
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}

    y_deg = np.array([deg_idx[r[1]] for r in node_rows], dtype=np.int64)
    y_cc = np.array([cc_idx[r[2]] for r in node_rows], dtype=np.int64)
    community = np.array([int(r[3]) for r in node_rows], dtype=np.int64)
    y_motif = np.array([[int(bool(r[4 + j])) for j in range(len(MOTIFS))]
                        for r in node_rows], dtype=np.float32)
    y_cat = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
    for node, cat in cat_rows:
        i = idx_of.get(int(node))
        if i is not None and cat in cat_idx:
            y_cat[i, cat_idx[cat]] = 1.0

    edges = np.array([[idx_of[int(s)], idx_of[int(d)]] for s, d in edge_rows],
                     dtype=np.int64)
    edge_index = torch.from_numpy(
        np.concatenate([edges.T, edges.T[::-1]], axis=1).copy())

    # data fingerprint: sorted canonical edge list + node count
    h = hashlib.sha256()
    h.update(f"n={n}\n".encode())
    for s, d in edge_rows:
        h.update(f"{int(s)},{int(d)}\n".encode())
    data_fp = h.hexdigest()

    train_mask = np.array([is_train_node(v) for v in node_ids])
    val_mask = ~train_mask

    hparams = {"epochs": EPOCHS, "seed": SEED, "feat_seed": FEAT_SEED,
               "feat_dim": FEAT_DIM, "emb_dim": EMB_DIM,
               "degree_classes": DEGREE_CLASSES, "cc_classes": CC_CLASSES,
               "categories": CATEGORIES, "motifs": MOTIFS,
               "val_node_pct": VAL_PCT, "split": "sha256(node_id) % 100",
               "affil_categories": AFFIL_CATS,
               "affil_min_nodes": AFFIL_MIN_NODES,
               "affil_neg_ratio": NEG_RATIO,
               "cohesion_split": "sha256(feature_key) % 100",
               "spec_version": spec_version()}

    def fit():
        files: dict[str, bytes] = {}
        meta: dict = {"spec_version": spec_version(), "device": str(device),
                      "n_nodes": int(n), "n_edges": int(len(edges)),
                      "split": f"node-id hash 80/{VAL_PCT} (node-level)"}
        x_rand = fixed_random_features(n)
        tr = torch.from_numpy(train_mask)
        va = torch.from_numpy(val_mask)

        # ---- degree_sage (gate: majority share + 0.02) ----------------------
        sage = DegreeSage(n_classes=len(DEGREE_CLASSES)).to(device)
        y = torch.from_numpy(y_deg)
        opt = torch.optim.Adam(sage.parameters(), lr=1e-2)
        xg, eig = x_rand.to(device), edge_index.to(device)
        sage.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            out = sage(xg, eig)
            loss = F.cross_entropy(out[tr], y[tr].to(device))
            loss.backward()
            opt.step()
        sage.eval()
        with torch.no_grad():
            pred = sage(xg, eig).argmax(1).cpu().numpy()
        acc = float((pred[val_mask] == y_deg[val_mask]).mean())
        shares = np.bincount(y_deg[train_mask],
                             minlength=len(DEGREE_CLASSES)) / max(1, train_mask.sum())
        maj = int(shares.argmax())
        maj_acc = float((y_deg[val_mask] == maj).mean())
        meta["degree_sage"] = {
            "metrics": {"val_accuracy": acc, "majority_accuracy": maj_acc,
                        "train_class_shares": [round(float(s), 4) for s in shares],
                        "n_val": int(val_mask.sum())},
            "gate": "val_accuracy > majority_accuracy + 0.02",
            "adopted": bool(acc > maj_acc + 0.02),
            "inputs": "fixed-random features (seed above); degree never fed",
            "role": "structure teacher -> distilled to the DP degree histogram"}
        files["degree_sage.pt"] = state_bytes(sage)

        # ---- community_gnn (gate: ARI vs Louvain > 0.30) --------------------
        from torch_geometric.nn import GAE
        torch.manual_seed(SEED)  # re-seed: negative sampling determinism
        gae = GAE(GCNEncoder()).to(device)
        opt = torch.optim.Adam(gae.parameters(), lr=1e-2)
        from torch_geometric.utils import negative_sampling
        gae.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            z = gae.encode(xg, eig)
            neg = negative_sampling(eig, num_nodes=n,
                                    num_neg_samples=eig.shape[1])
            loss = gae.recon_loss(z, eig, neg)
            loss.backward()
            opt.step()
        gae.eval()
        with torch.no_grad():
            z_comm = gae.encode(xg, eig).cpu().numpy().astype(np.float32)
        k = int(len(np.unique(community)))
        clusters, centers = kmeans(z_comm, k=k, seed=SEED)
        ari = adjusted_rand_index(clusters, community)
        # majority-community map per cluster (for the inference agreement bit)
        cluster_to_comm = {}
        for j in range(k):
            members = community[clusters == j]
            cluster_to_comm[str(j)] = int(np.bincount(members).argmax()) \
                if len(members) else -1
        meta["community_gnn"] = {
            "metrics": {"ari_vs_louvain": float(ari), "k": k,
                        "modularity_note": "Louvain labels are the seeded "
                                           "ingest labels (seed in spec)"},
            "gate": "ari_vs_louvain > 0.30",
            "adopted": bool(ari > 0.30),
            "role": "community teacher -> distilled to DP community sizes + "
                    "rank-labeled mixing matrix"}
        files["community_gnn.pt"] = state_bytes(gae)
        files["embeddings.npy"] = _npy_bytes(z_comm)
        files["node_ids.npy"] = _npy_bytes(node_ids)
        files["clusters.npy"] = _npy_bytes(clusters.astype(np.int64))
        files["kmeans_centers.npy"] = _npy_bytes(centers.astype(np.float32))
        import json as _json
        files["cluster_to_community.json"] = _json.dumps(
            cluster_to_comm, indent=2).encode()

        # ---- triad_vgae (gate: common-neighbor AUC + 0.01; honest negative) -
        from torch_geometric.data import Data
        from torch_geometric.nn import VGAE
        from torch_geometric.transforms import RandomLinkSplit
        torch.manual_seed(SEED)
        data = Data(x=x_rand, edge_index=edge_index, num_nodes=n)
        split = RandomLinkSplit(num_val=0.2, num_test=0.0, is_undirected=True,
                                split_labels=True,
                                add_negative_train_samples=False)
        train_data, val_data, _ = split(data)
        vgae = VGAE(VariationalGCNEncoder()).to(device)
        opt = torch.optim.Adam(vgae.parameters(), lr=1e-2)
        tei = train_data.edge_index.to(device)
        tpos = train_data.pos_edge_label_index.to(device)
        vgae.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            zv = vgae.encode(xg, tei)
            loss = vgae.recon_loss(zv, tpos) + vgae.kl_loss() / n
            loss.backward()
            opt.step()
        vgae.eval()
        vpos = val_data.pos_edge_label_index.cpu().numpy()
        vneg = val_data.neg_edge_label_index.cpu().numpy()
        pairs = np.concatenate([vpos, vneg], axis=1)
        y_link = np.concatenate([np.ones(vpos.shape[1]),
                                 np.zeros(vneg.shape[1])])
        with torch.no_grad():
            zv = vgae.encode(xg, tei)
            score = (zv[pairs[0]] * zv[pairs[1]]).sum(1).cpu().numpy()
        vgae_auc = auc_score(y_link, score)
        # common-neighbor baseline over the TRAIN message-passing graph
        nbrs: list[set] = [set() for _ in range(n)]
        tei_np = train_data.edge_index.cpu().numpy()
        for s, d in tei_np.T:
            nbrs[s].add(int(d))
        cn = np.array([len(nbrs[a] & nbrs[b]) for a, b in pairs.T],
                      dtype=np.float64)
        cn_auc = auc_score(y_link, cn)
        meta["triad_vgae"] = {
            "metrics": {"val_auc": float(vgae_auc),
                        "common_neighbor_auc": float(cn_auc),
                        "n_val_pos": int(vpos.shape[1]),
                        "n_val_neg": int(vneg.shape[1])},
            "gate": "val_auc > common_neighbor_auc + 0.01",
            "adopted": bool(vgae_auc > cn_auc + 0.01),
            "baseline_note": "at transitivity 0.52 the common-neighbor count "
                             "is a strong link predictor; an honest negative "
                             "here is expected and kept",
            "role": "closure teacher -> distilled to the DP wedge-closure "
                    "grid (P(closed | center degree class))"}
        files["triad_vgae.pt"] = state_bytes(vgae)

        # ---- featcond_encoder (gate: mean per-category val AUC > 0.55) ------
        deg_onehot = np.eye(len(DEGREE_CLASSES), dtype=np.float32)[y_deg]
        x_fc = torch.from_numpy(np.concatenate([deg_onehot, z_comm], axis=1))
        fce = FeatCondEncoder(n_in=x_fc.shape[1],
                              n_cats=len(CATEGORIES)).to(device)
        train_loop(fce, x_fc[tr], torch.from_numpy(y_cat[train_mask]),
                   device, EPOCHS, binary=True)
        with torch.no_grad():
            p_val = torch.sigmoid(fce(x_fc[va].to(device))).cpu().numpy()
        aucs = []
        for j in range(len(CATEGORIES)):
            a = auc_score(y_cat[val_mask][:, j], p_val[:, j])
            if not np.isnan(a):
                aucs.append(a)
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        marginal = y_cat[train_mask].mean(0)
        meta["featcond_encoder"] = {
            "metrics": {"mean_val_auc": mean_auc,
                        "per_category_auc_n": len(aucs),
                        "train_marginal_rates":
                            [round(float(r), 4) for r in marginal]},
            "gate": "mean per-category val AUC > 0.55",
            "adopted": bool(mean_auc > 0.55),
            "inputs": "degree-class one-hot + community_gnn embedding",
            "role": "feature teacher -> distilled to DP category/richness "
                    "histograms + suppressed degree-conditioned rows"}
        files["featcond_encoder.pt"] = state_bytes(fce)

        # ---- ego_motif_pool (gate: macro-F1 > per-motif majority baseline) --
        cc_onehot = np.eye(len(CC_CLASSES), dtype=np.float32)[y_cc]
        node_feat = np.concatenate([deg_onehot, cc_onehot], axis=1)
        pooled = np.zeros_like(node_feat)
        deg_count = np.zeros(n, dtype=np.float64)
        for s, d in edges:
            pooled[s] += node_feat[d]
            pooled[d] += node_feat[s]
            deg_count[s] += 1
            deg_count[d] += 1
        pooled = (pooled + node_feat) / (deg_count[:, None] + 1.0)  # self incl.
        x_mp = torch.from_numpy(pooled.astype(np.float32))
        emp = EgoMotifPool(n_in=x_mp.shape[1]).to(device)
        train_loop(emp, x_mp[tr], torch.from_numpy(y_motif[train_mask]),
                   device, EPOCHS, binary=True)
        with torch.no_grad():
            p_val = (torch.sigmoid(emp(x_mp[va].to(device)))
                     .cpu().numpy() >= 0.5).astype(int)
        yv = y_motif[val_mask].astype(int)
        f1s = [binary_f1(yv[:, j], p_val[:, j]) for j in range(len(MOTIFS))]
        macro_f1 = float(np.mean(f1s))
        # baseline: per-motif constant majority prediction (train prevalence)
        base_major = (y_motif[train_mask].mean(0) >= 0.5).astype(int)
        base_f1s = [binary_f1(yv[:, j], np.full(len(yv), base_major[j]))
                    for j in range(len(MOTIFS))]
        base_macro = float(np.mean(base_f1s))
        meta["ego_motif_pool"] = {
            "metrics": {"macro_f1": macro_f1, "baseline_macro_f1": base_macro,
                        "per_motif_f1": [round(f, 4) for f in f1s],
                        "train_motif_rates":
                            [round(float(r), 4)
                             for r in y_motif[train_mask].mean(0)]},
            "gate": "macro_f1 > per-motif-majority baseline macro-F1",
            "adopted": bool(macro_f1 > base_macro),
            "role": "motif teacher -> distilled to the DP motif-presence "
                    "histogram + conditional rules"}
        files["ego_motif_pool.pt"] = state_bytes(emp)

        # ================= affiliation-community models =================
        # key holders and per-category key sets (full observed profiles)
        import json as _json2
        key_holders: dict[str, list[int]] = {}
        key_cat: dict[str, str] = {}
        node_cat_keys: list[dict[str, set]] = [
            {c: set() for c in CATEGORIES} for _ in range(n)]
        for node, key, cat in key_rows:
            i = idx_of.get(int(node))
            if i is None or cat not in cat_idx:
                continue
            key_holders.setdefault(key, []).append(i)
            key_cat[key] = cat
            node_cat_keys[i][cat].add(key)
        richness = y_cat.sum(1)

        # ---- affil_ae (gate: val BCE < marginal baseline BCE - 0.001) -------
        affil_keys = sorted(k for k, hs in key_holders.items()
                            if key_cat[k] in AFFIL_CATS
                            and len(hs) >= AFFIL_MIN_NODES)
        kdim = len(affil_keys)
        x_aff = np.zeros((n, kdim), dtype=np.float32)
        for j, k in enumerate(affil_keys):
            for i in key_holders[k]:
                x_aff[i, j] = 1.0
        aae = AffilAE(kdim).to(device)
        xa = torch.from_numpy(x_aff)
        train_loop(aae, xa[tr], xa[tr], device, EPOCHS, binary=True)
        with torch.no_grad():
            rec = aae(xa[va].to(device)).cpu()
            val_bce = float(F.binary_cross_entropy_with_logits(
                rec, xa[va]).item())
            z_aff = aae.encode(xa.to(device)).cpu().numpy().astype(np.float32)
        p_marg = np.clip(x_aff[train_mask].mean(0), 1e-4, 1 - 1e-4)
        yv_aff = x_aff[val_mask]
        base_bce = float(np.mean(
            -(yv_aff * np.log(p_marg) + (1 - yv_aff) * np.log(1 - p_marg))))
        meta["affil_ae"] = {
            "metrics": {"val_bce": val_bce, "marginal_bce": base_bce,
                        "n_keys": kdim,
                        "keys_sha256": hashlib.sha256(
                            "\n".join(affil_keys).encode()).hexdigest()},
            "gate": "val_bce < marginal_bce - 0.001",
            "adopted": bool(val_bce < base_bce - 0.001),
            "inputs": f"binary vector over the {kdim} affiliation keys "
                      f"(categories {AFFIL_CATS}) held by >= {AFFIL_MIN_NODES} "
                      "nodes; key list stored as affil_keys.json",
            "role": "affiliation-profile compressor; embedding feeds "
                    "key_community_mlp"}
        files["affil_ae.pt"] = state_bytes(aae)
        files["affil_keys.json"] = _json2.dumps(affil_keys).encode()
        files["affil_embeddings.npy"] = _npy_bytes(z_aff)

        # ---- affil_link_mlp (gate: total-shared ranker AUC + 0.01) ----------
        rng = np.random.default_rng(SEED)
        edge_set = {(int(s), int(d)) for s, d in edges}
        neg: list[tuple[int, int]] = []
        seen = set(edge_set)
        while len(neg) < NEG_RATIO * len(edges):
            a = int(rng.integers(n))
            b = int(rng.integers(n))
            if a == b:
                continue
            p = (a, b) if a < b else (b, a)
            if p in seen:
                continue
            seen.add(p)
            neg.append(p)
        pair_list = [(int(s), int(d), 1) for s, d in edges] + \
                    [(a, b, 0) for a, b in neg]
        feats = np.zeros((len(pair_list), len(CATEGORIES) + 2),
                         dtype=np.float32)
        y_pair = np.zeros(len(pair_list), dtype=np.float32)
        keep_tr, keep_va = [], []
        for r, (a, b, lbl) in enumerate(pair_list):
            for ci, c in enumerate(CATEGORIES):
                feats[r, ci] = len(node_cat_keys[a][c] & node_cat_keys[b][c])
            ra, rb = richness[a], richness[b]
            feats[r, len(CATEGORIES)] = min(ra, rb)
            feats[r, len(CATEGORIES) + 1] = max(ra, rb)
            y_pair[r] = lbl
            ta, tb = train_mask[a], train_mask[b]
            if ta and tb:
                keep_tr.append(r)
            elif (not ta) and (not tb):
                keep_va.append(r)
        itr = np.array(keep_tr, dtype=np.int64)
        iva = np.array(keep_va, dtype=np.int64)
        plm = PairMLP().to(device)
        train_loop(plm, torch.from_numpy(feats[itr]),
                   torch.from_numpy(y_pair[itr][:, None]), device, EPOCHS,
                   binary=True)
        with torch.no_grad():
            s_val = plm(torch.from_numpy(feats[iva]).to(device)) \
                .cpu().numpy().ravel()
        link_auc = auc_score(y_pair[iva], s_val)
        total_shared = feats[iva, :len(CATEGORIES)].sum(1)
        base_auc = auc_score(y_pair[iva], total_shared)
        meta["affil_link_mlp"] = {
            "metrics": {"val_auc": float(link_auc),
                        "total_shared_auc": float(base_auc),
                        "n_train_pairs": int(len(itr)),
                        "n_val_pairs": int(len(iva)),
                        "neg_ratio": NEG_RATIO},
            "gate": "val_auc > total_shared_auc + 0.01",
            "adopted": bool(link_auc > base_auc + 0.01),
            "inputs": "shared-key counts per public category + min/max "
                      "richness (13 dims); pairs = edges + uniform non-edges",
            "role": "does WHAT is shared predict friendship beyond HOW MUCH "
                    "is shared (affiliation link teacher)"}
        files["affil_link_mlp.pt"] = state_bytes(plm)

        # ---- key_community_mlp (gate: majority rank share + 0.02) -----------
        if meta["affil_ae"]["adopted"]:
            x_kc = z_aff
            kc_inputs = "affil_ae 16-dim embedding"
        else:
            x_kc = np.array([[len(node_cat_keys[i][c]) for c in CATEGORIES]
                             for i in range(n)], dtype=np.float32)
            kc_inputs = "11-dim per-category key counts (affil_ae fallback)"
        y_rank = np.minimum(community, 8).astype(np.int64)
        kcm = KeyCommunityMLP(n_in=x_kc.shape[1]).to(device)
        train_loop(kcm, torch.from_numpy(x_kc[train_mask]),
                   torch.from_numpy(y_rank[train_mask]), device, EPOCHS)
        with torch.no_grad():
            pred = kcm(torch.from_numpy(x_kc[val_mask]).to(device)) \
                .argmax(1).cpu().numpy()
        kc_acc = float((pred == y_rank[val_mask]).mean())
        rank_shares = np.bincount(y_rank[train_mask], minlength=9) \
            / max(1, train_mask.sum())
        kc_maj = float((y_rank[val_mask] == int(rank_shares.argmax())).mean())
        meta["key_community_mlp"] = {
            "metrics": {"val_accuracy": kc_acc, "majority_accuracy": kc_maj,
                        "n_ranks": 9},
            "gate": "val_accuracy > majority_accuracy + 0.02",
            "adopted": bool(kc_acc > kc_maj + 0.02),
            "inputs": kc_inputs,
            "role": "do affiliations determine the structural community "
                    "(rank labels c1..c8/rest = size-ordered ingest ids)"}
        files["key_community_mlp.pt"] = state_bytes(kcm)

        # ---- key_cohesion_mlp (gate: MAE < train-mean-density baseline) -----
        def is_train_key(key: str) -> bool:
            hk = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            return hk % 100 < 100 - VAL_PCT

        aff_idx = {c: i for i, c in enumerate(AFFIL_CATS)}
        xg_rows, yg_rows, g_train = [], [], []
        for key, cat, sz, dens in kg_rows:
            onehot = [0.0] * len(AFFIL_CATS)
            onehot[aff_idx[cat]] = 1.0
            xg_rows.append(onehot + [float(np.log2(float(sz)))])
            yg_rows.append(float(dens))
            g_train.append(is_train_key(key))
        xg_arr = np.array(xg_rows, dtype=np.float32)
        yg_arr = np.array(yg_rows, dtype=np.float32)
        g_tr = np.array(g_train)
        kcoh = KeyCohesionMLP().to(device)
        opt = torch.optim.Adam(kcoh.parameters(), lr=1e-2)
        xt = torch.from_numpy(xg_arr[g_tr]).to(device)
        yt = torch.from_numpy(yg_arr[g_tr][:, None]).to(device)
        kcoh.train()
        for _ in range(EPOCHS * 4):     # tiny table; more epochs, still fast
            opt.zero_grad()
            loss = F.mse_loss(kcoh(xt), yt)
            loss.backward()
            opt.step()
        kcoh.eval()
        with torch.no_grad():
            pv = kcoh(torch.from_numpy(xg_arr[~g_tr]).to(device)) \
                .cpu().numpy().ravel()
        mae = float(np.mean(np.abs(pv - yg_arr[~g_tr])))
        base_mae = float(np.mean(np.abs(float(yg_arr[g_tr].mean())
                                        - yg_arr[~g_tr])))
        meta["key_cohesion_mlp"] = {
            "metrics": {"val_mae": mae, "mean_baseline_mae": base_mae,
                        "n_groups": int(len(yg_arr)),
                        "n_val_groups": int((~g_tr).sum()),
                        "train_mean_density":
                            round(float(yg_arr[g_tr].mean()), 4)},
            "gate": "val_mae < global-train-mean-density baseline MAE",
            "adopted": bool(mae < base_mae),
            "inputs": "category one-hot + log2(group size) over "
                      "mart_key_groups rows (affiliation categories)",
            "role": "affiliation cohesion teacher (release-ledger candidate; "
                    "NOT released — key-group marts are on the denied list)"}
        files["key_cohesion_mlp.pt"] = state_bytes(kcoh)

        meta["numeric_note"] = ("all grids/classes follow wiki/snap-contract.md; "
                                "class orders recorded in hparams")
        return files, meta

    version, trained = train_or_reuse(
        "minimodels", data_fp, hparams, [Path(__file__)], fit)
    print(f"train: minimodels version {version} "
          f"({'trained now' if trained else 'reused existing artifact'})")
    return 0


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


if __name__ == "__main__":
    sys.exit(main())

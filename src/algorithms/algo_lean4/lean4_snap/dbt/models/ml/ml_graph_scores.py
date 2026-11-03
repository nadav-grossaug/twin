# Inference node (guideline 3): loads the ACTIVE minimodel artifact from the
# content-addressed registry (models_out/registry.json) and materializes
# per-node scores. Rebuilding this mart never refits a model; if no artifact
# is active yet it materializes an 'untrained' placeholder so downstream SQL
# still compiles.
# The model classes mirror python/train_minimodels.py exactly (DegreeSage
# SAGE 16/32/8, GCNEncoder 16/64/32, FeatCondEncoder 40/32/11); artifact
# files: degree_sage.pt, community_gnn.pt (+ embeddings.npy, node_ids.npy,
# clusters.npy, cluster_to_community.json), featcond_encoder.pt.
# Per-model `adopted` flags in meta.json gate which columns are populated:
#   pred_degree_class / confidence   <- degree_sage (if adopted)
#   community_agree                  <- community_gnn (if adopted)
#   anomaly_score                    <- featcond_encoder BCE reconstruction
#                                       (if adopted), else degree-class
#                                       surprise -log p(class) from the
#                                       degree_sage train marginals.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEGREE_CLASSES = ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16",
                  "deg_17_32", "deg_33_64", "deg_65p"]
CATEGORIES = ["birthday", "education", "first_name", "gender", "hometown",
              "languages", "last_name", "locale", "location", "work", "other"]
FEAT_DIM = 16
EMB_DIM = 32
FEAT_SEED = 4242


def _active_artifact(models_dir: Path):
    reg = models_dir / "registry.json"
    if not reg.exists():
        return None
    active = json.loads(reg.read_text()).get("minimodels", {}).get("active")
    if not active:
        return None
    vdir = models_dir / "minimodels" / active
    return (active, vdir) if (vdir / "meta.json").exists() else None


def _placeholder(nodes):
    return pd.DataFrame({
        "node_id": nodes.node_id,
        "anomaly_score": np.nan,
        "pred_degree_class": None,
        "community_agree": np.nan,
        "model_version": "untrained",
        "confidence": np.nan,
    })


def _scores(nodes, feats_long, edges, version, vdir):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    meta = json.loads((vdir / "meta.json").read_text())

    class DegreeSage(nn.Module):
        def __init__(self, n_in=FEAT_DIM, hidden=32, n_classes=8):
            super().__init__()
            self.conv1 = SAGEConv(n_in, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.out = nn.Linear(hidden, n_classes)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            return self.out(h)

    class FeatCondEncoder(nn.Module):
        def __init__(self, n_in=8 + EMB_DIM, hidden=32, n_cats=11):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_cats))

        def forward(self, x):
            return self.net(x)

    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    n = len(nodes)
    node_ids = nodes.node_id.to_numpy(dtype=np.int64)
    idx_of = {int(v): i for i, v in enumerate(node_ids)}
    deg_idx = {c: i for i, c in enumerate(DEGREE_CLASSES)}
    y_deg = nodes.degree_class.map(deg_idx).to_numpy(dtype=np.int64)

    pred_class = np.full(n, None, dtype=object)
    confidence = np.full(n, np.nan)
    agree = np.full(n, np.nan)
    anomaly = np.full(n, np.nan)

    # ---- degree_sage: predicted degree class + confidence -------------------
    ds_meta = meta.get("degree_sage", {})
    if ds_meta.get("adopted") and (vdir / "degree_sage.pt").exists():
        e = edges[["src", "dst"]].to_numpy(dtype=np.int64)
        pairs = np.array([[idx_of[int(s)], idx_of[int(d)]] for s, d in e
                          if int(s) in idx_of and int(d) in idx_of],
                         dtype=np.int64)
        edge_index = torch.from_numpy(
            np.concatenate([pairs.T, pairs.T[::-1]], axis=1).copy()) \
            if len(pairs) else torch.zeros((2, 0), dtype=torch.int64)
        g = torch.Generator().manual_seed(FEAT_SEED)
        x = torch.randn(n, FEAT_DIM, generator=g)
        sage = DegreeSage(n_classes=len(DEGREE_CLASSES))
        sage.load_state_dict(torch.load(vdir / "degree_sage.pt",
                                        map_location="cpu"))
        sage.eval()
        with torch.no_grad():
            probs = torch.softmax(sage(x, edge_index), dim=1).numpy()
        k = probs.argmax(1)
        pred_class = np.array([DEGREE_CLASSES[j] for j in k], dtype=object)
        confidence = probs.max(1)

    # ---- community_gnn: kmeans-vs-louvain agreement (stored assignment) -----
    cg_meta = meta.get("community_gnn", {})
    if cg_meta.get("adopted") and (vdir / "clusters.npy").exists():
        stored_ids = np.load(vdir / "node_ids.npy")
        clusters = np.load(vdir / "clusters.npy")
        c2c = json.loads((vdir / "cluster_to_community.json").read_text())
        mapped = {int(v): c2c.get(str(int(c)), -1)
                  for v, c in zip(stored_ids, clusters)}
        cur_comm = nodes.community_id.to_numpy()
        for i, v in enumerate(node_ids):
            m = mapped.get(int(v))
            if m is not None and m >= 0:
                agree[i] = float(int(m) == int(cur_comm[i]))

    # ---- anomaly: featcond BCE reconstruction, else degree-class surprise ---
    fc_meta = meta.get("featcond_encoder", {})
    used_fallback = True
    if fc_meta.get("adopted") and (vdir / "featcond_encoder.pt").exists() \
            and (vdir / "embeddings.npy").exists():
        stored_ids = np.load(vdir / "node_ids.npy")
        emb = np.load(vdir / "embeddings.npy")
        emb_of = {int(v): emb[i] for i, v in enumerate(stored_ids)}
        have = np.array([int(v) in emb_of for v in node_ids])
        if have.any():
            y_cat = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
            cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
            for node, cat in feats_long[["node_id", "category"]].itertuples(
                    index=False):
                i = idx_of.get(int(node))
                if i is not None and cat in cat_idx:
                    y_cat[i, cat_idx[cat]] = 1.0
            deg_onehot = np.eye(len(DEGREE_CLASSES), dtype=np.float32)[y_deg]
            z = np.stack([emb_of[int(v)] if int(v) in emb_of
                          else np.zeros(EMB_DIM, dtype=np.float32)
                          for v in node_ids])
            fce = FeatCondEncoder(n_in=len(DEGREE_CLASSES) + EMB_DIM,
                                  n_cats=len(CATEGORIES))
            fce.load_state_dict(torch.load(vdir / "featcond_encoder.pt",
                                           map_location="cpu"))
            fce.eval()
            with torch.no_grad():
                logits = fce(torch.from_numpy(
                    np.concatenate([deg_onehot, z], axis=1)))
                bce = F.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(y_cat),
                    reduction="none").mean(dim=1).numpy()
            anomaly = np.where(have, bce.astype(np.float64), np.nan)
            used_fallback = False
    if used_fallback:
        shares = ds_meta.get("metrics", {}).get("train_class_shares")
        if shares:
            p = np.array(shares, dtype=np.float64)
            p = np.clip(p, 1e-6, None)
            anomaly = -np.log(p[y_deg])

    return pd.DataFrame({
        "node_id": nodes.node_id,
        "anomaly_score": anomaly,
        "pred_degree_class": pred_class,
        "community_agree": agree,
        "model_version": version,
        "confidence": confidence,
    })


def model(dbt, session):
    dbt.config(materialized="table")
    nodes = dbt.ref("feat_nodes").df()[["node_id", "degree_class",
                                        "cc_class", "community_id"]]
    feats_long = dbt.ref("stg_node_features").df()
    feats_long = feats_long[["node_id", "category"]].drop_duplicates()
    edges = dbt.ref("stg_edges").df()[["src", "dst"]]
    models_dir = Path(os.environ.get(
        "FLOW_ARTIFACT_ROOT", os.environ.get("MODELS_DIR", "../models_out"))).resolve()
    active = _active_artifact(models_dir)
    if active is None:
        return _placeholder(nodes)
    return _scores(nodes, feats_long, edges, active[0], active[1])

---
title: PyTorch minimodels
type: component
updated: 2026-11-03
sources: [../python/train_minimodels.py, ../models_out]
---

# PyTorch minimodels — snap graph build (nine models)

Content-addressed artifact (`sha256(name, data, hparams, code)`), immutable
dirs under `models_out/minimodels/<version>/`, registry-activated; node-level
80/20 split by id hash; CPU-deterministic under `FLOW_DEVICE=cpu`;
`FLOW_RETRAIN=1` to force a retrain. Every model is gated against a FROZEN
non-ML baseline and recorded with its honest `adopted` flag. Inference
(`ml_graph_scores`) never retrains. Previous builds' model inventories:
[[travel-pipeline]], [[stock-pipeline]].

How each adopted model reaches the Lean generator (or deliberately doesn't):
[[distillation]].

## Structural set (the five learning targets)

| model | question | metric vs baseline | adopted | distilled to |
|---|---|---|---|---|
| degree_sage | degree class from neighborhood structure only (fixed random inputs, degree never fed) | val acc 0.584 vs majority 0.192 | ✅ | DP degree histogram |
| community_gnn | unsupervised GCN/GAE embeddings + seeded KMeans vs Louvain | ARI 0.307 vs gate 0.30 | ✅ | DP community sizes + 45-cell mixing matrix |
| triad_vgae | VGAE link prediction (RandomLinkSplit, no val-edge leakage) | val AUC 0.946 vs common-neighbor 0.990 (+0.01 gate) | ❌ **honest negative** | — (the empirical DP closure grid replaced the teacher) |
| featcond_encoder | 11-dim category presence from degree class ⊕ embedding | mean per-category val AUC 0.670 vs 0.55 gate | ✅ | DP richness + category histograms (+ suppressed conditional rows) |
| ego_motif_pool | 6-dim motif presence from mean-pooled 1-hop class one-hots | macro-F1 0.420 vs majority 0.0 | ✅ | DP motif-presence histogram (validation anchor) |

## Affiliation set (shared-key communities, 2026-11-03 — [[affiliation-communities]])

| model | question | metric vs baseline | adopted | distilled to |
|---|---|---|---|---|
| affil_ae | are affiliation profiles (951 education/work/hometown/location/languages keys) compressible? | val BCE 0.0170 vs marginal 0.0350 | ✅ | — scoring model, enclave-side |
| affil_link_mlp | does WHICH category you share predict friendship beyond HOW MANY keys? | val AUC 0.9507 vs total-shared ranker 0.9347 (+0.01 gate) | ✅ | — ledger candidate (homophily lift); key-group marts on the denied list |
| key_community_mlp | do affiliations determine the structural community? | val acc 0.734 vs majority 0.178 | ✅ | — evidence for the `community_scoped_features` prior |
| key_cohesion_mlp | key-group internal density from category ⊕ log size | val MAE 0.123 vs mean-predictor 0.355 | ✅ | — named ledger candidate (cohesion by category×size) |

## Rules carried from earlier builds

- Gates are data-dependent; read `adopted` from the artifact's meta.json,
  never from docs.
- Honest negatives are kept and cited: triad_vgae is the expected loss —
  at transitivity 0.52 the common-neighbor count is a formidable baseline.
- Scoring models never leave the enclave; generative teachers are replaced
  by RELEASED constructs only (post-processing); anything else is a
  declared gap or a named ledger candidate — see [[distillation]].

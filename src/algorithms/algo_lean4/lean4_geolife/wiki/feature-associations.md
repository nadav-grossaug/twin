---
title: Feature associations (χ²)
type: concept
updated: 2026-10-26
sources: [../dbt/models/marts/mart_feature_assoc.sql, ../lean/EnclaveFlows/Assoc.lean]
---

# Feature associations (χ²)

Pairwise chi-square over all categorical flow features **and the motif flags**
([[motifs]]), computed in SQL on full contingency grids (zero-observed cells
included) with Cramér's V as effect size; the statistic is recomputed
**exactly over ℚ** in Lean (`chiSq`, with `chiSq_nonneg` and
`chiSq_eq_zero_of_indep` proved) and agreed with SQL to <1e-6 relative on all
exported tables ([[lean-formal-layer]]).

Strongest associations in [[dataset-nitroba]] (Cramér's V):

- dst_loc × service_class **0.97** — service taxonomy nearly determines
  internal/external destination (multicast/broadcast services are internal).
- dst_loc × fanout_source 0.95; fanout_source × service_class 0.93 — the
  fan-out hosts ([[fanout-scan]]) dominate external HTTP.
- entropy_band × payload **0.81** — zero-byte flows have no entropy, and
  payload flows split low/high entropy cleanly.
- answered × payload 0.67 — unanswered flows are mostly the zero-byte noise
  floor ([[data-quality]]).
- beacon_channel × service_class — beacon flows concentrate in keepalive
  services ([[beacons]]); residual heatmap in the notebook.

Models: `mart_assoc_cells` (cells + standardized residuals),
`mart_feature_assoc` (per-pair χ², dof, V), fed by `feat_flow_cats`.

---
title: dbt + DuckDB pipeline
type: component
updated: 2026-10-30
sources: [../dbt]
---

# dbt + DuckDB pipeline

`make dbt` → `dbt build` in `dbt/` against `artifacts/enclave.duckdb`
(the in-enclave store, [[enclave-boundary]]). ~25 models + data tests +
a unit test, all green; thresholds are dbt vars (the versioned concept spec).

- **staging/stg_flows** — `read_json_auto` over `../flow.jsonl`
  ([[cisco-joy-features]]); filters the config header; stable `flow_id` +
  provenance `flow_key` (md5 of 5-tuple+start).
- **features** — `feat_flows` (durations/volumes/entropy + quality flags +
  the public 17-class **service taxonomy** = the DP cell grid), `feat_pairs`
  (per-(sa,da,dp) gap stats → [[beacons]]), `feat_tls` (fingerprint + record
  structure, [[tls-fingerprinting]]), `feat_dns` (qname wire shape,
  [[dns-structure]]), `feat_flow_cats` (categorical substrate for χ²).
- **marts** — stats (`mart_flow_stats`), motifs (`mart_motif_beacon`/
  `_fanout`/`_return`), associations (`mart_feature_assoc`,
  `mart_assoc_cells`), sequences (`mart_seq_transitions`/`_chains`/
  `_sessions`), TLS/DNS populations (`mart_tls_fingerprints`,
  `mart_tls_records`, `mart_dns`, `mart_dns_domains`).
- **DP inputs** — `mart_dp_counts`, `mart_dp_tls` (stack RANK cells),
  `mart_dp_dns` (shape cells): clipped per-user counts for `lake exe release`
  ([[dp-release]]).
- **Common-knowledge top-k** — `mart_dns_topk`, `mart_tls_sni_topk`:
  suppression-thresholded popular names for the policy artifacts
  ([[architecture]]).
- **quality/q_data_integrity** — the named-counter audit ledger.
- **ml/ml_flow_scores** — dbt **Python** model: loads the ACTIVE versioned
  artifact from `models_out/` and materializes [[minimodels]] predictions;
  rebuilding never retrains; emits an 'untrained' placeholder if no artifact
  exists.

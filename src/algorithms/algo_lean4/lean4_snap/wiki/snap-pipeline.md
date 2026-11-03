# snap-pipeline — the as-built lean4_snap pipeline (fifth instantiation)

As built 2026-11-03. Layer map per [[architecture]]; plan was
[[migration-plan-snap]]; data profile [[dataset-snap]]; interfaces
[[snap-contract]].

## Layer map (as built)

- **Ingest**: `python/ingest_graphml.py` → `artifacts/ingest/*.tsv` (canonical
  edges, feature long table, circles, Louvain seed 7 relabeled by size,
  sha256 fingerprint). Re-used verbatim for synthetic re-ingestion.
- **dbt** (DuckDB `artifacts/enclave.duckdb`): 25 models + 98 schema tests +
  seed, all green. SQL recomputes degree/triangles/wedges from edges (9.3M
  wedge join); `q_data_integrity` cross-checks networkx exactly (0
  mismatches); `q_temporal_fields` proves the static-snapshot claim (507
  time-looking keys, all binary flags). Motif counts: hub 215, leaf 75,
  broker 32, bridge 499, butterfly 302, core 368. Top associations:
  cc×degree Cramér's V 0.525, education×work 0.406.
- **Minimodels** (PyG 2.8.1, CPU-deterministic, content-addressed): nine
  models — eight adopted, one honest negative —
  degree_sage val acc 0.584 vs majority 0.192 ✓; community_gnn ARI 0.307 vs
  gate 0.30 ✓; **triad_vgae val AUC 0.946 LOST to the common-neighbor
  baseline 0.990** (expected at transitivity 0.52; kept, adopted=false);
  featcond_encoder mean AUC 0.670 ✓; ego_motif_pool macro-F1 0.420 vs 0 ✓;
  and the AFFILIATION set over shared-key communities (see
  [[affiliation-communities]]): affil_ae (951 keys → 16-dim, BCE 0.017 vs
  marginal 0.035) ✓, affil_link_mlp (AUC 0.951 vs the total-shared-keys
  ranker 0.935 — WHAT you share beats HOW MUCH) ✓, key_community_mlp
  (affiliations alone predict the Louvain community at 73.4% vs majority
  17.8%) ✓, key_cohesion_mlp (density MAE 0.123 vs mean-predictor 0.355) ✓.
  Key-group marts stay enclave-side (denied list; ledger-review candidate).
- **Lean formal layer** (SampCert a2c53bb / Lean v4.34.1, shared
  LEAN_HOME store): `lake exe checks` re-decided 24,234 motif + 20,195
  class + 8 exact-ℚ χ² checks — **0 disagreements**. 28-claim ledger
  (24 proved incl. histMech_DP, sampleCellPMF_law, edgesCanonical_sound,
  isBroker_not_isCore; 4 assumed incl. the DECLARED `row_level_privacy_unit`
  node-DP gap).
- **DP release**: discrete Laplace per cell, main degree grid + 9 aux grids,
  exact-fraction ledger **385/100 = 77/20 ≤ 4/1** enforced in Lean (exit-2
  refusal) and Python (`Fraction`); per-node proportional L1 clip (8) on
  category/mixing/closure grids, clip 4 on motifs, clip 2 on community
  sizes, clip 1 on one-cell-per-node grids.
- **Distill**: genspec keyed to the release sha; weight-by-weight provenance
  test; fidelity accepted (released-vs-exact degree TV ≈ 0.016; released
  closure level ≈ 0.59 vs exact transitivity 0.519 — the per-node
  equal-weighting semantics, declared).
- **Generator** (`lake exe generate`, Lean-only): community sizes → degree
  targets (sorted per community) → **near-clique blocks** (size ≈ d/q+1 at
  density q = 5/3·released closure p, min-target pair wiring, homogeneity
  cap) → mixing-row stubs (deficit-adaptive, receiver-capped) → one light
  closure pass → planted hub(0)/broker(2)/butterfly(1) with DECIDABLE
  self-verification → `edgesCanonical` proved output check → GraphML +
  plan.json. ~2 s for 4,039 nodes / ~90k edges.
- **Validation**: `make synth` AND `make synth_from_bundle` → validate
  **PASS exit 0**: planted recall 1.0 / 1.0 / 1.0 (both), degree TV
  0.087 / 0.093 (≤ 0.25), per-node mean closure 0.516 / 0.503 within ±0.15
  of released 0.567, **0 Lean disagreements** on the re-ingested synthetic
  graphs; kernel-checked `certificate.json` (PROVED, `by decide`). Realism
  audit (graded): synth transitivity 0.501 vs real 0.519, modularity 0.809
  vs 0.835, degree TV real-vs-synth 0.059; residuals: hub share 0.0005 vs
  0.053, leaf share 0 vs 0.019 (declared, top improvement candidates).
  Full suite 68 passed / 0 skipped.
- **Ops**: six-flow Makefile + `make help`; single source-only bundle
  (`dist/bundle/`); `synth_from_bundle` rebuilds the generator from bundle
  sources via the shared store; marimo notebooks validated headless.

## Gotchas earned here (→ [[practices]])

- dbt-duckdb's Python-model ref extractor MISSES `dbt.ref(...)` calls whose
  statement continues with a backslash — keep each ref on its own simple
  statement.
- A lone leftover node at a community tail forced block size 2 past the
  array end (Lean panic → silent default in release builds): clamp block
  size to the remaining run, let singleton blocks be empty.
- Generator realism knobs that mattered, in order: block-clique structure
  (uniform wedge-closing alone plateaus at transitivity ≈ 0.24), min-target
  pair wiring (protects the degree histogram), receiver caps on deficit
  stubs (protects the LOW bins — the stub flood had wiped deg_1..deg_4),
  densification 5/3 (compensates stub dilution).
- Compare like with like: the released closure grid is per-node
  equal-weighted, so synth is judged on per-node mean closure, not
  edge-weighted transitivity (which sits lower by construction — same
  effect as real avg-clustering 0.6055 vs transitivity 0.5192).

## Declared gaps (meta.json `fields_not_modelled`)

degree–community dependence; feature identity structure beyond category
level; circle rosters; inter-ego overlap; degree assortativity. Realism
audit quantifies the residuals; transitivity sits below the real value by
construction (declared above).

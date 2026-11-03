---
title: Migration plan: foursquare to SNAP graph
type: synthesis
updated: 2026-11-03
sources: [../goal.md, ../METHOD.md]
---

# migration-plan-snap — foursquare travel pipeline → SNAP Facebook graph

**The ACTIVE plan** (2026-11-02). Port `../lean4_foursquare` (ops substrate:
six-flow Makefile, `LEAN_HOME=/opt/data/lean`, single source-only bundle,
`synth_from_bundle`) to the [[dataset-snap]] social graph. Fifth
instantiation; METHOD §12 substitution map applied below.

## Substitution map (travel → graph)

| foursquare concept | snap concept |
| --- | --- |
| protected entity: user | person (node) — row-level DP on the derived node table; node-DP gap declared |
| record: transfer | derived node row; edge/wedge contributions clipped per node |
| region taxonomy (8, public) | feature-category taxonomy (11, public) |
| travel-count bins | degree bins (8) |
| gap classes | local clustering classes (7) |
| stay-duration classes | triangle-count bins (6) |
| distance classes | feature-richness bins (9) + circle bins (5) |
| region transition matrix 8×8 | community mixing matrix (rank-labeled, 9×9 sym → 45 cells) + community-size bins (6) |
| return motif (A→B→A) | triangle / wedge closure (the dominant structure: closure 0.5192) |
| corridors / tours / bursts / long-stays | hub ≥150, leaf =1, broker (deg≥17 ∧ cc<1/5), bridge (≥3 communities), butterfly (≥3 circles in one ego) |
| episode grouping | (none — static graph) wedge-closure grid replaces the sequence layer |
| NextRegionGRU / GapClassGRU / EpisodeShape / ProfileAE | DegreeSAGE / CommunityGNN / TriadVGAE / FeatCondEncoder / EgoMotifPool (PyG available: torch-geometric 2.8.1) |
| TSV renderer in Lean | GraphML renderer in Lean (nodes + sparse feature keys + circles + canonical edges) |
| top-k corridors (suppression) | degree-conditioned category rows + motif summaries (suppression ≥ 50 nodes) |
| ε ledger 400/100 | draft 385/100 (see goal.md; finalized after pre-noise simulation) |

## Phases

0. ✅ Profile the data ([[dataset-snap]]; clustering 0.6055 reproduced).
1. ✅ Rewrite goal.md for the graph domain.
2. Ingest: `python/ingest_graphml.py` → nodes/edges/features/circles TSVs +
   networkx cross-check columns (triangles, clustering, Louvain seed 7);
   dbt staging + `q_data_integrity` + `q_temporal_fields`.
3. Spec three ways: `dbt_project.yml vars` = `python/spec.py` =
   `lean/EnclaveGraph/Params.lean` (+ category taxonomy ×3), sync tests.
4. dbt features/marts: feat_nodes, feat_edges, motif marts, wedge closure,
   community sizes/mixing, χ² assoc cells, mart_dp_* inputs (proportional L1
   clip, largest-remainder, `dp_clip_total = 8`).
5. Minimodels (PyG, CPU-deterministic, content-addressed, gated).
6. Lean formal layer: Motifs/Stats/GraphShape/Assoc + checks exe; DP release
   (SampCert discrete Laplace, exact-rational ledger both sides, refusal
   path); verified sampler; **pre-noise SNR simulation before committing the
   ledger**.
7. Distill → genspec.json (provenance cell-exact; declared public priors).
8. Lean generator → synthetic GraphML + plan.json; round-trip validation
   (planted broker + butterfly + hub recall 1.0; clustering band; GOF).
9. Ops: six-flow Makefile + `make help`, HOWTO_run.md, source-only bundle,
   `synth_from_bundle` green.
10. Wiki/METHOD/goal updates as built; acceptance table flipped to ✅.

## Risks / watch items

- n = 4,039 is 7× smaller than foursquare: SNR on the 45-cell mixing grid
  needs the phase-6 simulation before trusting ε_cell = 2/100.
- Triangle-count heavy tail (max 30,025) — clip before believing any
  wedge-derived release.
- VGAE vs common-neighbor baseline: transitivity 0.52 makes the baseline
  strong; expect an honest negative and lean on the empirical closure grid.
- GraphML rendering in Lean: plain string building, but key ids must match
  the descriptive `dN_<short>` convention and the sparse-feature encoding.
- Re-ingestion of synthetic GraphML goes through the SAME
  `ingest_graphml.py` (networkx), so Louvain on a synthetic graph may find a
  different k — validation compares histograms over public bins, never raw
  labels.

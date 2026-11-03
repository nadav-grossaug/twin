---
title: Migration plan (stock/flows → travel transfers)
type: synthesis
updated: 2026-11-01
sources: ["../METHOD.md", "[[dataset-foursquare]]", "../goal.md", "[[migration-plan-stock]]"]
---

# Migration plan — pipeline → [[dataset-foursquare]]

Status 2026-11-02: **implemented** — see [[travel-pipeline]] for the as-built
map and numbers (65 pytest green, validate exit 0, planted recall 1.0 + 1.0,
certificate kernel-checked, `synth_from_bundle` green). Headline findings:
the NextRegionGRU beat the 72% return-to-previous baseline (75.7%) but its
first-order distilled student honestly FAILED the fidelity gate (56.5% —
accepted: false, as this page predicted); the realism audit quantifies the
resulting return-rate gap (synth 0.20 vs real 0.82) with a return-first
generator redesign as the top candidate fix. Original plan below kept for
provenance; the build log is in [[log]] (2026-11-01/02 entries).

`../goal.md` is the full task framing;
this page is the working checklist. METHOD.md §12's substitution map
instantiated, then the phases in METHOD §4 order — each phase's tests lock
the previous one. The stock build ([[stock-pipeline]],
[[migration-plan-stock]]) is the code base being ported; its incident
lessons (noise-anchored marginals, sensitivity redesigns) carry over.

## This migration's extra briefs (beyond METHOD)

1. **Focus**: user-level travel-count distribution; travel sequences /
   recurring motifs / sequence models; inter-travel gap features, moments,
   motifs.
2. **Simplified ops**: exactly six make flows — `clean`, `train` (whole
   enclave flow incl. bundle), `synth`, `synth_from_bundle`,
   `notebook-enclave`, `notebook-synth`.
3. **`LEAN_HOME=/opt/data/lean`**: elan, mathlib cache and the pinned
   package store live there, shared and reused by every build (repo and
   bundle alike); never re-downloaded per project.
4. **Source-only deploy**: ONE bundle format; no binaries ship; synth
   consumes only Lean sources + released JSON.

## Substitution map (METHOD §12 applied)

| stock repo | travel repo |
|---|---|
| (ticker, day, price) row; unit = daily return | one country-transfer row; unit = the transfer (hop) |
| ticker (n = 8!) as protected unit | the user (n = 29,640) — whole history = one `Neighbour` record |
| 5-class return grid | 8-region public map (ISO→UN-M49-derived, total, versioned) + public bins: travel-count ×7, gap ×5, duration ×5, distance ×4, episode ×5 |
| runs · shock · drawdown · V-recovery · vol-burst | return trip (72.1%!) · commuter corridor ≥5 reps CV≤0.8 · tour 3 hops/≤7d/3 countries (rarest → planted) · burst 3/14d · long-stay ≥90d |
| weekly Mon–Fri shape vector | episode (journey): gaps ≤14d grouped; length histogram 1,2,3,4–5,6+ |
| sequences broken at calendar gap >4d | sequences broken at gap >90d (`rare` never creates a transition) |
| rank labels `stock_1..8`, SYN names | users: nothing per-user released, fresh synth ids; countries: public taxonomy — DP at region level, names only via top-k corridors (≥50 users, ≥100 transfers); venue ids structurally excluded, synth 24-hex |
| trading calendar + base-100 | observation window 2012-04-04 → 2013-09-16 + country centroids (public geography) + TSV schema + duration-chaining convention |
| planted shock-in-drawdown, recall 1.0 | planted tour + commuter corridor, recall 1.0 |

Kept unchanged: five-layer architecture, three release categories, spec
mirrored three ways, content-addressed gated minimodels,
distill-to-typed-spec, Lean-only generator with validity by construction,
validation loop, every §6 calibration lesson.

## Phases

0. **Profile the data** — ✅ done 2026-11-01, findings in
   [[dataset-foursquare]]; all goal.md thresholds measured, not defaulted.
1. **Ingest**: `stg_transfers` via DuckDB `read_csv`; clean TSV (measured),
   no repairer; flags anyway (zero-duration stays, censored first/last).
2. **Concept spec ×3**: dbt `vars:` + region-map seed = `python/spec.py` =
   `lean/EnclaveTravel/Params.lean`, regex sync test, `spec_version`.
   Rationals in Lean (CV ≤ 4/5; gap edges in whole seconds).
3. **dbt layers**: features (`feat_transfers`, `feat_users`,
   `feat_episodes`), marts (user stats, 5 motif marts, episodes,
   `mart_seq_transitions` 8×8, assoc χ² grids, `mart_dp_*`), ml scores.
4. **Minimodels** (4, each gated vs a frozen baseline): NextRegionGRU (vs
   **return-to-previous**, which alone scores 72.1% — the GRU may honestly
   lose), GapClassGRU (vs marginal), EpisodeShapeModel (vs unigram),
   UserProfileAE (anomaly, enclave-only). Honest negatives kept.
5. **Lean mirrors**: decidable motif predicates + `lake exe checks`
   agreement counts; χ² recomputed over ℚ from exported contingency cells.
6. **DP release.** n = 29,640 ⇒ SNR comfortable (≈139 for a 10%-share cell
   on the 64-cell grid at ε=3/100); the risk is the heavy tail (max 195
   travels) ⇒ proportional L1 clipping with `dp_clip_total = 8` (≈p95).
   Travel-count histogram: sensitivity 1, no clipping. Hard gate: pre-noise
   simulation before the ledger commits. Planned ledger 400/100 exactly
   (goal.md): counts 28 + mix 32 + transitions 192 + gaps 30 + durations 30
   + distance 24 + months 24 + episodes 20 + return-bands 20. Dow = declared
   uniform prior, 0ε.
7. **Verified sampler**: `sampleCellPMF` pattern + χ² GOF conformance test.
8. **Distillation**: released-only inputs; NextRegionGRU → DP 8×8 rows;
   GapClassGRU → DP gap histogram + suppressed within-class moments;
   EpisodeShapeModel → DP episode histogram; corridors → released top-k
   list; priors declared `derived_from_data: false`; weight-by-weight
   provenance test; fidelity criterion declared up front (student ≥
   return-baseline − 0.02).
9. **Lean generator**: `mkTravelRow`/`mkUserHistory` smart constructors
   (timestamps strictly increasing per user, prev ≠ next, durations ≥ 0,
   duration chaining, window containment — theorems); renders source-format
   TSV; emits `plan.json`; Python adapter writes provenance +
   `fields_not_modelled` (intra-country geography, venue structure,
   cross-user relations, within-bin gap shapes, non-top-k country
   transitions).
10. **Validation loop**: round-trip re-ingestion (planted tour + corridor
    recall = 1.0), per-column GOF, kernel certificates, realism audit by the
    enclave's own scorers, run manifests, two marimo notebooks headless,
    wiki updated every ingest.

## Watch-list (where this dataset differs from stock)

- **Tail, not SNR, is the DP risk**: clip total 8 vs max 195 — measure the
  clipped-vs-raw divergence in phase 6; the per-user-average vs row-weighted
  mix gap is REAL here (unlike stock's equal row counts) and must be
  released/documented as the chosen semantics.
- **The 72.1% return baseline** is brutal for sequence models — honest
  negatives likely; the empirical transition matrix + return-boost prior may
  beat the GRU as teacher.
- Countries are PUBLIC taxonomy (unlike tickers) — identity leaves only at
  region level (DP) or via suppressed corridors; the generator needs a
  region→country instantiation rule (corridor-first, centroid fallback).
- Users are generated entities: synth must reproduce the travel-count
  histogram (focus 1) — a per-user draw, a dimension stock synth didn't
  have.
- Durations chain across rows — a structural invariant to prove, test, and
  re-check on re-ingestion.

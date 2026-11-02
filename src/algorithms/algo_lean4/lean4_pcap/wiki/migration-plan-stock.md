---
title: Migration plan (flows → stock series)
type: synthesis
updated: 2026-10-31
sources: ["../METHOD.md", "[[dataset-stock]]", "../goal.md"]
---

# Migration plan — flows pipeline → [[dataset-stock]]

Status 2026-10-31 (evening): **implemented** — see [[stock-pipeline]] for
the as-built map and numbers (51 tests green, validate exit 0, certificate
kernel-checked). Deferred: robustness harness, frozen-rebuild reproduce
check. Original plan below kept for provenance; incidents earned during the
build are in [[log]] (2026-10-31 implementation entry) — headline lessons:
pair-count co-movement release drowned (sensitivity n−1) and was redesigned
to per-ticker mean-share bins (clip 1); noise-dominated DP rows need a
released-marginal anchor + rescaled seasonality rows or the generator's
marginal drifts (TV 0.305 → 0.146).

`../goal.md` is the full task
framing; this page is the working checklist. METHOD.md §12's substitution map
instantiated, then the phases in METHOD §4 order — each phase's tests lock the
previous one.

## Substitution map (METHOD §12 applied)

| flows repo | stock repo |
|---|---|
| flow record (Joy JSONL line) | one (ticker, trading-day, price) row; derived unit = daily return |
| internal host (protected unit) | the ticker (whole series = one `Neighbour` record); n = 8 |
| 7/17-class service taxonomy | 5-class daily return grid: crash ≤−3% / down / flat ±0.5% / up / surge ≥3% |
| motifs: beacon / fan-out / A→B→A / sessions | runs ≥5 · shock \|r\|≥10% · drawdown ≥20% · V-recovery 10%/95%/20d · vol-burst 5d-sd ≥4%; sequences broken at calendar gaps >4d |
| TLS/DNS structural sub-records | weekly shape: Mon–Fri return-class vector, shape histograms over public cells, canonical week skeleton |
| top-k qnames / SNI | none releasable (8 entities < any sane k): tickers structurally excluded, rank labels `stock_1..8`, synthetic names `SYN01…` |
| FlowShape.lean constants | public trading calendar + base-100 start + class edges in `Params.lean` |
| beacon recall round-trip | plant a shock day inside a drawdown; recall 1.0 after re-ingestion |

Kept unchanged: five-layer architecture, three release categories, spec
mirrored three ways, content-addressed gated minimodels, distill-to-typed-spec,
Lean-only generator with validity by construction, validation loop, every §6
calibration lesson.

## Phases

0. **Profile the data** — ✅ done 2026-10-31, findings in [[dataset-stock]];
   thresholds in goal.md chosen from measured distributions.
1. **Ingest**: `stg_prices` via DuckDB `read_csv`; no repairer needed (clean
   TSV — measured); quality flags anyway (first-day=100 check, gap flags).
2. **Concept spec ×3**: dbt `vars:` = `python/spec.py` =
   `lean/EnclaveSeries/Params.lean`, regex sync test, `spec_version`.
   Thresholds as exact rationals (shock = 1/10, edges 1/200 and 3/100).
3. **dbt layers**: features (returns, vol, drawdown, class), marts (stats,
   5 motif marts, week shapes, `mart_seq_transitions`, assoc χ² grids,
   `mart_dp_counts`), ml scores table.
4. **Minimodels** (~4–5, each gated vs a frozen baseline): ReturnWindowAE,
   NextClassGRU (vs repeat-last), VolRegimeClassifier (vs majority),
   WeekShapeModel (vs unigram). Honest negatives kept. Scoring models stay
   enclave-side; generative ones are teachers for phase 8.
5. **Lean mirrors**: decidable motif predicates + `lake exe checks` agreement
   counts; χ² recomputed over ℚ from exported contingency cells.
6. **DP release — THE RISK PHASE.** n = 8 ⇒ SNR ≈ `share × n × ε / k`.
   **Hard gate**: pre-noise simulation (clipped counts vs raw, per-cell SNR)
   BEFORE committing the ledger. Proportional L1 clipping
   (`dp_clip_total`, largest-remainder — never uniform per-cell, see
   [[dp-release]] inversion incident). Fallback if 5-cell utility fails:
   3-cell grid (down/flat/up), never a silent privacy-unit switch. Draft
   ledger 39/10 ≤ 4/1 (mix 5×2/5, transitions 25×1/25, week shapes 10×1/20,
   motif bands 4×1/10) — nearly saturated by design.
7. **Verified sampler**: `sampleCellPMF` pattern + χ² GOF conformance test.
8. **Distillation**: released-only inputs; NextClassGRU → DP 5×5 transition
   rows; WeekShapeModel → shape hist over fixed skeleton; magnitude →
   per-class |return| bins; public priors declared
   `derived_from_data: false`; provenance test weight-by-weight; fidelity
   criterion declared up front (student ≥ repeat-last − 0.02).
9. **Lean generator**: `mkSeries` smart constructor (first price = 100.0,
   prices > 0, dates sorted within the public calendar — theorems); renders
   source-format TSV; emits `plan.json`; Python `generate` = thin adapter
   writing provenance + `fields_not_modelled` (cross-ticker correlation,
   absolute price levels, volume).
10. **Validation loop**: round-trip re-ingestion (planted shock-in-drawdown
    recall = 1.0), per-column GOF tests, kernel certificates, realism audit
    by the enclave's own scorers (graded, not pass/fail), run manifests,
    two marimo notebooks headless, wiki updated every ingest.

## Watch-list (where this dataset differs from flows)

- **n = 8 protected entities** vs 15 hosts: every utility number from the
  reference is optimistic here. Coarse grids, concentrated budget.
- Equal per-entity row counts ⇒ no per-entity-average vs record-weighted
  divergence on this file (the flows repo's DNS-server problem) — but assert
  it, don't assume it.
- No name modelling at all (char-LM dropped); rank labels + `SYN` names.
- Cross-ticker correlation: the realism property quants will notice first is
  a declared gap in v1. Candidate future ledger entry: DP same-day class
  co-movement counts — must displace an existing query.

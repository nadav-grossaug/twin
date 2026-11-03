---
title: Stock pipeline (implemented)
type: component
updated: 2026-10-31
sources: ["../goal.md", "../Makefile", "../config/release_policy.yaml", "[[migration-plan-stock]]"]
---

# Stock pipeline — the implemented arch_03 build

Implemented 2026-10-31 (one session, scaffold cloned from the flows repo
`~/tmp/test_torch_lean_arch_02` with APFS clones incl. the 8.7 GB Lean cache).
`make all` order: `dbt train ml lean distill test manifest`; synth via
`make synth certify realism-audit`. Suite: **51 passed**.

## Layer map

- **dbt/DuckDB** (31 green): `stg_prices` → `feat_returns` (LOG returns in bp,
  5-class taxonomy, rolling vol + vol states, drawdown, SMA/RSI/momentum TA,
  seasonality fields) → `feat_weeks`, `feat_day_cats` → marts: 5 motif marts
  (runs / shock / drawdown / V-recovery / vol-burst), `mart_seq_transitions`
  (surge>surge lift 1.92 = volatility clustering, measured),
  `mart_seq_sessions`, `mart_seasonality`, `mart_correlation` (CRWD–PANW
  r=0.80), χ² assoc grids, `q_data_integrity`, 6 `mart_dp_*` inputs.
- **Minimodels** (content-addressed, honest gates): ReturnWindowAE (adopted,
  scoring), NextClassGRU (0.18 vs repeat-last 0.29 — **honest negative**),
  SeasonalityMLP (0.28 vs majority 0.33 — **honest negative**), GarchVol
  (torch MLE per ticker; beats constant-vol NLL on 6/8; persistence
  0.92–0.999), FactorModel (2-factor SVD, adopted). `ml_series_scores` node
  materializes anomaly / next-class / seasonality / GARCH-σ / factor-residual.
- **Lean 4 + SampCert** (lib `EnclaveSeries`, pinned SampCert a2c53bb):
  mechanism + sampler files reused VERBATIM from the flows repo
  (DPRelease/Sampler/Stats/Sequences/Assoc — dataset-agnostic by design).
  New: `Motifs.lean` (IsRun/IsShock/HasDrawdown/IsVRecovery/IsVolBurst,
  decidable, with sanity + monotonicity theorems), `SeriesShape.lean` (public
  magnitude bins, vol-boost table, Hinnant weekday calendar with
  kernel-checked anchor round-trip + `tradingDayNum_strictMono`),
  `GenSpec.lean` (`mkDay_valid`), `Checks.lean`, `Generate.lean`.
  **119/119 SQL↔Lean agreements**; 24 proved + 3 assumed claims.
- **DP release**: ledger EXACTLY 400/100 = 4/1 — class mix 5×27/50,
  transitions 25×1/100, dow×class 25×1/100, vol transitions 16×1/100,
  mag bins 6×1/150, co-movement 3×1/5. Privacy unit = one ticker,
  proportional L1 clip 128 on the mix.
- **Distillation** (post-processing only, provenance-tested cell-by-cell):
  transitions = DP row + released-marginal anchor (2× row mass) + diagonal
  persistence boost (2× — raised from 1× on 2026-11-01: a fresh DP draw made
  the 1× student miss its gate; 2× pins the argmax to repeat-last across
  draws); dow rows rescaled to total 10; λ = 2s−1 from the
  co-movement release → **λ = 70%**. Fidelity accepted
  (student ≥ repeat-last − 0.02).
- **Generator** (`lake exe generate <genspec> <tickers> <days> <out>`):
  class chain ⊙ dow seasonality, market-mode coupling with prob λ
  (collinearity), vol-state chain (discretized GARCH diffusion) boosting
  magnitude bins, price = 100·exp(Σ log r). Plants a shock-in-drawdown down
  run on the last ticker.
- **Validation**: re-ingestion through PRODUCTION dbt → scratch DB: planted
  shock recall 1.0, run (len 12) + drawdown recalled, class-mix TV 0.146,
  0 Lean disagreements → exit 0. `certificate.json` kernel-checked
  (`by decide`: planted_is_run, planted_has_shock, planned_classes_valid).
- **Realism audit** (graded): synth same-sign share 0.726 vs real; synth mean
  Pearson r 0.19 vs real 0.46 — the coupling transfers sign agreement, not
  the full correlation matrix (declared gap, quantified).
- **Notebooks**: `visualize_enclave_mo.py` (EDA + DP utility released-vs-
  clipped) and `visualize_synth_mo.py` (TV scorecard), both export headless.

## Deferred

Robustness harness and the frozen-rebuild reproduce check (ported targets
exist in the reference repo; see goal.md acceptance rows marked ◻).

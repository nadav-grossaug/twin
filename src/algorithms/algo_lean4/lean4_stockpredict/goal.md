# torch-lean-arch: Enclave Stock-Series Encoder and synthetic data generator

## Goal

Build a reproducible encoder that converts a private portfolio of daily stock
price series (`stock_normalized_train.tsv`: 1,344 rows = 8 tickers × 168
trading days, 2025-10-06 → 2026-06-05, every series normalized to 100.0 on day
one) into a versioned market-behavior representation: per-day return features,
per-ticker regime statistics, price motifs, and fitted distributions — for
quantitative-analysis training and downstream synthetic price-series
generation. Run the encoder inside an enclave so raw series and sensitive
intermediate artifacts remain within the protected processing boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

This is a migration of the method proven on network-flow telemetry: see
`METHOD.md` (the dataset-agnostic replication guide, especially §12's
substitution map), `wiki/migration-plan-stock.md` (the phased plan), and
`wiki/dataset-stock.md` (the measured data profile). The five-layer
architecture, three release categories, spec-mirror discipline, Lean-only
generator, and every calibration lesson in METHOD §6 carry over unchanged.

**What is sensitive here:** the composition of the portfolio (which tickers
are held) and each ticker's price trajectory. Tickers are therefore released
only as rank labels; the protected entity is the ticker.

The encoder learns and measures data behavior from the input dataset,
producing an explicit specification that a later synthetic-data generator can
target.

### Outputs (planned)

- Per-day and per-ticker features: daily simple and log returns, rolling
  volatility (5-day sd of returns), drawdown from running peak, calendar-gap
  flags, quality flags.
- A **public return-class taxonomy** (the DP cell grid, fixed in the spec,
  independent of the data): 5 classes on daily return r —
  `crash` (r ≤ −3%), `down` (−3% < r ≤ −0.5%), `flat` (|r| < 0.5%),
  `up` (0.5% ≤ r < 3%), `surge` (r ≥ 3%).
- Price motifs with public thresholds (SQL and Lean mirrors):
  - **runs** — ≥ 5 consecutive same-sign non-flat daily returns (momentum
    streak; the beacon analog);
  - **shock days** — |daily return| ≥ 10% (earnings-style jumps; the rarest
    motif, used as the planted round-trip pattern);
  - **drawdowns** — running peak-to-trough decline ≥ 20%;
  - **V-recoveries** — drop ≥ 10% from a local peak with recovery to ≥ 95% of
    that peak within 20 trading days (the A→B→A return-trip analog);
  - **volatility bursts** — 5-day rolling sd of daily returns ≥ 4%.
- **Weekly shape** structure: the Mon–Fri vector of return classes per ticker
  per week, histogrammed over public cells (the analog of TLS record-sequence
  shapes); canonical week skeletons for rendering.
- Sequence analysis: the 5×5 return-class transition matrix with lift;
  sequences broken at calendar gaps > 4 days (long weekends never create
  false transitions).
- Conditional regime and anomaly scores from versioned mini-models.
- Versioned mini-models, evaluation results, and candidate formal
  specifications.
- Differentially private releases of aggregate distributions, with the noise
  mechanism formally verified with
  [SampCert](https://github.com/leanprover/SampCert) (Lean 4, pinned commit
  `a2c53bb` and its toolchain, as in the reference implementation).
- A formally verified sampler for drawing from released distributions, built
  on SampCert's samplers — the entry point for the downstream generator.
- A standalone Lean generator (`lake exe generate`) that renders synthetic
  series **in the source TSV format** (`stockname\tdate-yyyy-mm-dd\tstockprice`)
  from the distilled genspec alone, validated by re-ingestion through the
  production pipeline, with kernel-checked certificates.

## Processing Architecture (planned)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | DuckDB `read_csv` in dbt staging (`stg_prices`) | Canonical daily rows with stable `obs_id` and content-hash `obs_key`; typed date/price |
| Cleaning and features | dbt + DuckDB (`feat_returns`, `feat_regimes`, `feat_weeks`) | Daily/log returns, rolling volatility, drawdown, return class from the public taxonomy, calendar-gap flags |
| Mini-model fitting | PyTorch (`python/train_minimodels.py`), CPU-deterministic mode supported | ReturnWindowAE (anomaly/embedding), NextClassGRU (next return class from history), VolRegimeClassifier (regime from recent window), WeekShapeModel (weekly class-vector structure); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_series_scores`) | Per-day predictions with confidence and model version; never retrains on rebuild |
| marts | dbt SQL | `mart_series_stats`, `mart_motif_runs`, `mart_motif_shocks`, `mart_motif_drawdown`, `mart_motif_vrecovery`, `mart_motif_volburst`, `mart_week_shapes`, `mart_dp_counts` |
| Association analysis | dbt SQL (`mart_assoc_cells`, `mart_feature_assoc`) | χ²/Cramér's V between discretized features (return class × vol regime × motif flags) on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Sequence analysis | dbt SQL (`mart_seq_transitions`) | 5×5 return-class transition matrix with lift, gap-aware |
| Formal specification | Lean 4 + Mathlib (`lean/EnclaveSeries/Motifs.lean`, `Params.lean`) plus `lake exe checks` | Decidable motif definitions, proof-status ledger, SQL↔Lean agreement on exported observations |
| Private release | Lean 4 + SampCert discrete Laplace over clipped per-ticker contributions (`lake exe release`) | Ticker-level ε-DP histograms over the public grids, proved mechanism theorem, exact-rational budget ledger enforced in Lean AND Python |
| Verified sampling | SampCert `UniformSample` + bucketing map, law proved (`sampleCellPMF_law` pattern) (`lake exe sample`) | Samples from released distributions for generation and GOF checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: typed weighted cells, conditional transition rows, week-shape histograms, schema constraints, public calendar pointer |
| Generation | `lake exe generate <genspec> <n> <out>` | Synthetic TSV series + `plan.json` ground truth (planted motifs) |
| Validation | re-ingest through production dbt + Lean checks; pytest per-column | Round-trip motif recall, GOF vs released weights, kernel-checked certificate |

Supporting infrastructure: `Makefile` (`all`: `dbt train ml lean distill test
manifest` — distill AFTER lean, as DP draws are non-reproducible), a marimo
EDA notebook and a synth-vs-real comparison notebook (validated headless), and
the persistent **LLM-Wiki** under `wiki/` (Karpathy-style: index, append-only
log, schema, interlinked pages — updated on every ingest and incident).

## Enclave Boundary

Keep the following inside the enclave by default:

- The raw TSV (`stock_normalized_train.tsv`), real ticker symbols, and the
  rank→ticker map.
- Individual returns, per-ticker features, motif hits, per-day rows
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Per-ticker embeddings and predictions; training checkpoints and fitted
  parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required, exact-
fraction ε budget). The three exit categories from METHOD §6 apply:

1. **ε-DP histograms** over the public grids (return-class mix, class
   transitions, dow×class seasonality, vol-state transitions, magnitude
   bins, co-movement share bins), noised by a Lean executable using SampCert
   discrete Laplace, with the mechanism theorem proved and the ledger
   enforced in both Python and Lean.
2. **Suppression-thresholded aggregates** (no formal guarantee — declared as
   such).
3. **Common-knowledge metadata**: the trading calendar (public NYSE-style
   weekday calendar — a fixed public temporal constant), the normalization
   convention (every series starts at 100.0 — a property of the dataset's
   construction, not of any ticker), and the public taxonomy thresholds.

**Privacy unit (the critical design decision — METHOD §6 lessons applied):**

- One protected entity = **one ticker** (its entire 168-day series is one
  record in SampCert's add/remove-one `Neighbour` relation). The protected
  population is the portfolio: membership and trajectory.
- **n = 8 entities is tiny.** METHOD's SNR estimate `share × n × ε_total / k`
  gives ≈ 1.3 on the 5-cell grid even spending the whole ε = 4 budget there.
  Phase 6 therefore REQUIRES a pre-noise simulation (clipped counts vs raw
  distribution, SNR per cell) before the ledger is committed; if ticker-level
  utility fails, the fallback is a coarser 3-cell grid (down/flat/up), NOT a
  silent switch of privacy unit. A per-(ticker, day) event-level unit would
  be a weaker guarantee and may only be adopted as an explicitly declared,
  documented decision.
- **Bounded contributions** via proportional L1 allocation: each ticker's
  167-return class vector is allocated over a total budget `dp_clip_total`
  (largest-remainder rounding in SQL), so per-cell sensitivity equals the
  total and the proved mechanism applies unchanged. Uniform per-cell clipping
  is known to flatten the mix (METHOD §6) — not used.
- **Semantics**: equal per-ticker budgets estimate the per-ticker-average
  class distribution, not the row-weighted one. Here all tickers have equal
  row counts (168 each), so the two coincide — measure and document this
  identity rather than assuming it survives future data.
- **Rank labels**: DP histograms that are per-entity use `stock_1..8` rank
  labels; the rank→ticker map never leaves the enclave. Real tickers are
  structurally excluded from release (with 8 entities, no top-k suppression
  threshold is meaningful); synthetic series carry generated names
  (`SYN01…`) matching the released name-shape convention.
- **Cross-ticker correlation**: the exact pairwise matrix is a protected
  relation and never leaves. What IS released: a 3-cell DP histogram of each
  ticker's mean same-sign-day share (clip 1), post-processed into a single
  market-coupling probability λ for the generator (comove_lambda_pct — the
  2s−1 rule, declared). The residual gap (full correlation structure) stays
  in `assumptions.fields_not_modelled` and is quantified by the realism
  audit.

**Final ε ledger** (exact fractions; enforced in Lean AND Python; refusal
path tested):
class mix 5 × (27/50) = 270/100; class transitions 25 × (1/100) = 25/100;
dow×class 25 × (1/100) = 25/100; vol transitions 16 × (1/100) = 16/100;
magnitude bins 6 × (1/150) = 4/100; co-movement 3 × (1/5) = 60/100.
Total 400/100 = 4/1 — EXACTLY saturated: a new query must displace an old
one. `lake exe release` refuses exports over budget (exit 2).

Suppression thresholds alone carry no formal guarantee; use them only for
artifacts explicitly marked non-DP. DP noise randomness comes from a
non-replayable source (urandom-backed FFI); DP outputs are excluded from
bitwise reproducibility by design.

Require attestation and controlled key release, encrypted persistent storage,
restricted egress, and logs that redact ticker symbols (`<ticker>`). CPU
mini-models suffice at this scale; GPU training stays outside the enclave
threat model.

## Implementation Guidelines

### 0. Understand the data before modelling it — ✅ done (see `wiki/dataset-stock.md`)

Measured profile: 8 tickers × 168 trading days each, no duplicate dates, no
missing business days (max calendar gap 4 days = long weekends), no
unparseable prices. Daily return sd per ticker 2.4–4.4%; shock days exist
(OKTA +30.1%, FTNT +20.0%, ZS −31.5%, CHKP −19.6%) — the thresholds above
were picked from these measured distributions, not defaults.

### 1. Define Concepts Before Fitting Models

Version every threshold (return-class edges, run length 5, shock 10%,
drawdown 20%, V-recovery 10%/95%/20d, vol-burst 4%/5d, gap 4d) as dbt `vars:`,
mirrored in `python/spec.py` and `lean/EnclaveSeries/Params.lean`, locked by a
sync test; bump `spec_version` on any change. Thresholds as exact rationals in
Lean (e.g. shock = 1/10). Document data construction facts (base-100
normalization, weekday-only calendar) in `wiki/data-quality` terms: flags,
never silent fixes. Ingest needs no repairer (the TSV parses clean) — record
that as a measured fact, keep the verified-repairer pattern in reserve.

### 2. Start with Deterministic Baselines

SQL motif detection and empirical distributions first (runs, drawdowns,
transition counts, per-class return stats). Mini-models are added on top and
gated: NextClassGRU vs repeat-last, VolRegimeClassifier vs majority,
WeekShapeModel vs unigram marginals. Gates are data-dependent — read adoption
flags from artifacts, not docs. Keep honest negatives.

### 3. Separate Training from Inference

`make train` produces content-addressed immutable artifacts
(`sha256(name, data, hparams, code)`) under `models_out/`, registry-activated;
the inference node `ml_series_scores` materializes prediction tables.
Rebuilding a mart never retrains.

### 4. Preserve Uncertainty and Provenance

Stable `obs_id`, content-hash `obs_key`, data fingerprint (sha256 of the TSV),
model version and confidence on every prediction, quality flags, `untrained`
placeholder supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

Break sequences at calendar gaps > 4 days; never let a gap create a false
transition (mirror the lemma pattern: "breaks split transitions"). The file
is a *train* split of a larger series — treat series ends as censored, not as
terminal events.

### 6. Prevent Leakage

Time-ordered 80/20 split (split date fixed before fitting); scalers fit on
train only; weak labels derived from the public taxonomy, never from
evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

```text
P(next_class | class history)       — NextClassGRU; and the empirical 5×5
                                      transition matrix with lift
P(class      | recent vol window)   — VolRegimeClassifier
P(week shape | prior week)          — WeekShapeModel / empirical shape rows
P(|return|   | class)               — empirical per-class magnitude bins
```

### 9. Make Execution Reproducible

Data fingerprints, code versions, seeds, hyperparameters, split definitions in
`models_out/<v>/meta.json`; immutable artifact paths; `make reproduce`
(frozen rebuild hash-identical; CPU retrain bit-stable ≤ 1e-6). DP outputs
excluded by design. SampCert pinned; its toolchain dictates the Lean/Mathlib
versions.

### 10. Keep Formal Claims Precise

Distinguish: proved in Lean (mechanism DP, sampler law, motif decidability,
generator validity-by-construction) / checked on this snapshot (`lake exe
checks` SQL↔Lean agreement) / statistical estimates (model metrics) / assumed
(entropy source, I/O wrappers, and the SQL computing the clipped counts — DP
holds for whatever counts are fed in; their meaning is not proved). State
trust boundary 2 explicitly, as the reference implementation does.

### 11. Use Verified Randomness

DP noise and all generation-time draws only through SampCert-backed Lean
executables (`lake exe release`, `lake exe sample`, `lake exe generate`).
Never ad-hoc floating-point noise.

## Synthetic generation contract (planned)

- `python/distill.py` reads ONLY `dp_release.json` + released artifacts
  (post-processing theorem; `source.dp_release_sha256` recorded). Teachers map
  to spec constructs: NextClassGRU → 5×5 DP transition rows; WeekShapeModel →
  shape histogram over a fixed canonical week skeleton; magnitude model →
  per-class |return| bin histograms. Public structural priors (e.g. a
  flat-class diagonal boost) are declared `derived_from_data: false`.
- `lake exe generate` renders TSV rows directly: price path =
  100.0 × cumulative product of (1 + sampled return), returns drawn by class
  via the verified sampler then mapped to a magnitude bin; dates from the
  public calendar. **Validity by construction**: a `mkSeries` smart
  constructor with theorems — first price = 100.0 exactly, all prices > 0,
  dates strictly increasing and within the public calendar
  (`mkSeries_valid`, `dates_sorted` via `List.Pairwise`).
- Plant the rarest motif (a shock day inside a drawdown) in `plan.json`;
  round-trip re-ingestion through the production dbt pipeline must recall it
  at 1.0 and `lake exe checks` must re-verify (exit 0/2 soft, 1 hard).
- Per-column tests: every output column matches its declared distribution
  (χ² GOF vs released weights), exhibits its declared property, or satisfies
  a structural invariant; gaps declared in `meta.json
  assumptions.fields_not_modelled` (initially: cross-ticker correlation,
  absolute price levels, volume — the dataset has no volume column).
- Kernel-checked certificate (`by decide` over the exported plan;
  `proved: true, kernel_only: true` + axiom list).
- Seed-reproducibility intentionally absent (consistent with
  non-reproducible DP draws) — a documented caveat.

## Acceptance Criteria — status

| Area | Required evidence | Status |
| --- | --- | --- |
| Data integrity | Parsing, units, calendar, duplicates, gaps documented and checked | ✅ dbt 31/31 green; `q_data_integrity` counters; clean parse (no repairs needed, measured) |
| Feature extraction | Comparison against rule baselines and a reviewed sample | ✅ log returns + TA features + moments; EDA notebook exports headless |
| Motif extraction | Known sequences produce expected motifs; calendar gaps create no false transitions | ✅ 119/119 SQL↔Lean agreements (runs/shocks/drawdowns/V-recoveries/vol-bursts/χ²); gap-aware sequences |
| Distribution modeling | Held-out fit outperforms or justifies empirical baselines | ✅ time-ordered split; honest negatives kept (GRU and SeasonalityMLP fail their baselines on 1,336 rows); GARCH beats constant-vol NLL on 6/8 tickers; factor model adopted |
| Robustness | Controlled noise/missingness produce declared responses | ◻ deferred (harness not yet ported from the reference repo) |
| Formal layer | Lean compiles, no `sorry`; verified claims and unproved obligations separately listed | ✅ 24 proved + 3 assumed claims in `lean_results.json`; no-sorry test green |
| Enclave operation | Attestation abstraction + SimulatedTEE; policy-gated export | ✅ enclave.py abstraction + default-deny policy export (`make export`, dev override marked `attested: false`); platform still unselected |
| Private release | Mechanism theorem on SampCert; exact-rational budget enforcement; pre-noise SNR documented (n = 8!) | ✅ ledger 400/100 = 4/1 exactly, enforced in Lean AND Python; over-budget refusal tested; released-vs-clipped SNR shown in the notebook |
| Verified sampling | Sampler law proved; executable draws pass χ² GOF | ✅ `sampleCellPMF_law`; catsample χ² conformance test green |
| Distillation | Weight-by-weight provenance test green; fidelity report `accepted: true` | ✅ provenance tested cell-by-cell incl. declared priors; fidelity accepted (student ≥ repeat-last − 0.02) |
| Generation | `lake exe generate` renders schema-valid TSV; planted motif recall 1.0; kernel-checked certificate | ✅ validate exit 0 (shock recall 1.0, run + drawdown recalled, class-mix TV 0.146, 0 Lean disagreements); `certificate.json` kernel-checked |
| Reproducibility | Pinned inputs reproduce within declared tolerances; DP outputs excluded by design | ◻ partial: content-addressed artifacts + run manifest; frozen-rebuild reproduce script deferred |
| Wiki discipline | Pages + append-only log updated on every ingest/incident | ✅ ongoing (`wiki/`) |

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible stock-series encoder plus a
verified synthetic-series generator with an explicit output contract.

- Enclave execution protects processing within the selected platform's threat
  model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit (one ticker) and assumptions. They do not establish that
  the released numbers are useful — with 8 entities, utility is the open risk
  and is measured, not assumed.
- Statistical evaluation establishes empirical usefulness.

None of these alone guarantees realism of future synthetic series. Released
non-DP artifacts and evaluation metrics remain outside the DP guarantee.
Cross-ticker correlation is a declared gap until a ledger entry covers it.

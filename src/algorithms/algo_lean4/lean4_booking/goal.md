# torch-lean-arch: Enclave Booking-Trip Encoder and synthetic data generator

## Goal

Build a reproducible encoder that converts a private corpus of
multi-destination accommodation reservations (`combined_set.csv`:
1,545,502 reservations, 259,159 users, 288,348 trips, window
2015-12-31 → 2017-02-27 — the Booking.com WSDM/SIGIR'21 multi-destination
trips challenge data, anonymized) into a versioned travel-behavior
representation: per-trip leg structure (contiguous stays), hotel-country
tier sequences, next-destination transitions, stay-nights and gap
distributions, seasonality, device/affiliate/booker marginals and their
χ² associations, repeat-city return motifs — for recommendation research
training and downstream synthetic reservation generation. The challenge
learning target: **predict the next in-trip destination in real time**;
the encoder must capture exactly the structure a next-destination model
needs. Run the encoder inside an enclave so raw reservations and sensitive
intermediate artifacts remain within the protected processing boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

This is the sixth instantiation of the method (pcap flows → stock series →
Foursquare travel → GeoLife trajectories → SNAP social graph → Booking
trips). See `METHOD.md` (the dataset-agnostic replication guide, especially
§12's substitution map), `wiki/booking-contract.md` (the frozen cross-layer
interfaces for THIS dataset), `wiki/migration-plan-booking.md` (the phased
plan) and `wiki/dataset-booking.md` (the measured data profile). The
five-layer architecture, three release categories, spec-mirror discipline,
Lean-only generator, and every calibration lesson in METHOD §6 carry over
unchanged. The porting base is `../lean4_foursquare` (travel domain + the
current simplified ops substrate).

**What is sensitive here:** each user's reservation history — which
countries and cities they visit, in what order, when, how long they stay,
from which country and device they book, through which affiliate channel.
Trip sequences are strong re-identification vectors (a multi-city itinerary
narrows a person quickly). The protected entity is the **user**; user ids
and city-level itineraries never cross the boundary in any form. City ids
are released only as suppression-thresholded top-k lists (no formal
guarantee, declared as such) and as declared synthetic labels elsewhere.

The encoder learns and measures data behavior from the input dataset,
producing an explicit specification that a later synthetic-data generator
can target.

### Dataset-specific emphases (this migration's brief)

1. **Trip as the behavioral backbone** — a `utrip_id` is an ordered chain
   of stays; 99.11% of consecutive legs are CONTIGUOUS
   (next checkin == previous checkout), median 5 legs, max 48. Synthetic
   users must reproduce the chained-trip rhythm (contiguity by
   construction), not just marginal distributions.
2. **Feature co-occurrence and correlation** — χ² statistical association
   analysis between discretized columns/features/motifs on full contingency
   grids (device_class × booker_country, device_class × hotel tier,
   affiliate tier × device, affiliate tier × booker country, booker
   country × hotel tier, month × hotel tier), with Cramér's V; Lean
   recomputes every χ² exactly over ℚ.
3. **Timely sequence flow** — input records follow time-ordered structure:
   leg ordering within a trip (leg_idx by checkin), checkout→checkin
   chaining, and the next-destination conditional
   P(next hotel tier | prev tier) — the distilled replacement for the
   NextCountry teacher and the core of the recommendation use case.
4. **Repeat-city returns** — 38.07% of trips revisit a city already
   visited in the same trip; the generator carries this second-order
   return behavior via the released return-rate bands (a first-order
   transition matrix cannot express it).
5. **Next-destination realism** — the synthetic data must support
   next-destination training: country-tier chains from the DP transition
   rows + return-first behavior, city choice from the released top-k city
   and city-transition lists (suppression-only), stay nights from the DP
   nights cells, dates contiguous inside the public window.

### Outputs (planned)

- Per-reservation, per-transition, per-trip and per-user features: leg
  index, nights, gap days to the next leg, contiguity flag, repeat-city
  flag, tier encodings (hotel country tier, affiliate tier), month/dow
  cells, quality flags (`qf_bad_dates`, `qf_overlap`).
- **Public class bins** (all fixed in the spec; edges chosen from measured
  distributions, then frozen as public):
  - **reservations-per-user bins** `rc_1_3, rc_4, rc_5, rc_6, rc_7_8,
    rc_9_12, rc_13p` (edges 4/5/6/7/9/13; cap 64);
  - **nights bins** `n_1, n_2, n_3, n_4_6, n_7p` (edges 4/7; range 1–30);
  - **gap bins** `g_0, g_1, g_2_3, g_4p` (edges 2/4; range 0–33);
  - **hotel-country tiers** — the 15 most common anonymized countries +
    `ct_other` (public seed `country_tiers.csv`);
  - **affiliate tiers** — the 8 dominant affiliate ids + `aff_other`
    (public seed `affiliate_tiers.csv`);
  - month cells `m01..m12`, dow cells `d0..d6` (dow measured ~uniform —
    an honest public uniform prior), device cells
    `desktop/mobile/tablet`, booker cells (the 5 booker countries).
- **Travel motifs** with public thresholds (SQL and Lean mirrors):
  - **repeat city** — a trip revisits a city it already visited (38.07%
    of trips; `HasRepeatCity`);
  - **corridor** — an ordered country pair repeated ≥ 3 times by one user
    with gap CV ≤ 4/5 (`IsCorridor`);
  - **tour** — a trip with ≥ 3 distinct hotel countries, ≥ 4 legs, every
    gap ≤ 7 days (`IsTour`);
  - **long stay** — a stay of ≥ 7 nights (`IsLongStay`);
  - **well-formed trip** — strictly increasing checkin days, no overlap;
    contiguous variant: every gap 0 (`IsWellFormedTrip` /
    `IsContiguousTrip` — the temporal-flow invariant, proved by
    construction in the generator).
- **Sequence analysis**: the 16×16 hotel-tier transition matrix
  (`mart_seq_transitions`, exact, enclave-only; DP counterpart
  `country_transition`), next-destination conditional structure,
  month seasonality.
- χ²/Cramér's V association tables on full contingency grids; Lean
  recomputes χ² exactly over ℚ on exported observations.
- Conditional sequence and anomaly scores from versioned mini-models.
- Versioned mini-models, evaluation results, and candidate formal
  specifications.
- Differentially private releases of aggregate distributions, with the
  noise mechanism formally verified with
  [SampCert](https://github.com/leanprover/SampCert) (Lean 4, pinned commit
  `a2c53bb` and its toolchain, as in the reference implementation).
- A formally verified sampler for drawing from released distributions,
  built on SampCert's samplers — the entry point for the downstream
  generator.
- A standalone Lean generator (`lake exe generate`) that renders
  synthetic reservations **in the source CSV format**
  (`user_id,checkin,checkout,city_id,device_class,affiliate_id,
  booker_country,hotel_country,utrip_id`) from the distilled genspec
  alone; validated by re-ingestion through the production pipeline, with
  kernel-checked certificates. **The only artifacts the generator consumes
  are Lean sources and the released JSON — no compiled binaries, no Python
  models, no raw data.**

## Processing Architecture (planned)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | DuckDB `read_csv` in dbt staging (`stg_reservations`) | Canonical reservations with `leg_idx`, typed dates, nights, epoch-day columns, tier joins, month/dow/nights cells, quality flags |
| Trip features | dbt SQL (`feat_reservations`, `feat_transitions`, `feat_trips`, `feat_users`) | Per-leg prev-city/tier context, consecutive-leg transitions with gap/contiguity/repeat flags, per-trip structure (legs, cities, countries, contiguity, repeat, tour, longstay), per-user summaries (rc bin, rr band) |
| marts | dbt SQL | `mart_dp_*` inputs (11 grid-complete histograms), `mart_feature_assoc` + `mart_assoc_cells` (χ²), `mart_seq_transitions`, motif marts, `mart_topk_cities` / `mart_topk_city_transitions` / `mart_gap_moments`, `mart_user_stats` |
| Mini-model fitting | PyTorch (`python/train_minimodels.py`), CPU-deterministic | NextCountry (next hotel tier; gate: return-to-previous), TripLenClass (rc bin from first leg; gate: marginal), StayNights (gate: marginal), DeviceClass (gate: marginal), UserProfileAE (anomaly; scoring-only); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_trip_scores`) | Per-trip predictions with confidence and model version; never retrains on rebuild |
| Association analysis | dbt SQL (`mart_assoc_cells`, `mart_feature_assoc`) | χ²/Cramér's V between discretized features on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Sequence analysis | dbt SQL (`mart_seq_transitions`) | tier→tier transition counts (exact, enclave-only), month seasonality |
| Formal specification | Lean 4 + Mathlib (`lean/EnclaveTrip/`) plus `lake exe checks` | Decidable motif/trip definitions, proof-status ledger, SQL↔Lean agreement on exported observations |
| Private release | Lean 4 + SampCert discrete Laplace over clipped per-user contributions (`lake exe release`) | User-level ε-DP histograms over the public grids, proved mechanism theorem, exact-rational budget ledger enforced in Lean AND Python |
| Verified sampling | SampCert `UniformSample` + bucketing map, law proved (`sampleCellPMF_law` pattern) (`lake exe sample`) | Samples from released distributions for generation and GOF checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: rc/nights/gap/month/dow/device/affiliate/booker/rr histograms, 16×16 tier transitions (+ declared return-boost diagonal), top-k cities and city transitions, priors, window, bin ranges |
| Generation | `lake exe generate <genspec> <n_users> <out>` | Synthetic booking CSV + `plan.json` ground truth (planted repeat-city / tour / longstay motifs) |
| Validation | re-ingest through production dbt + Lean checks; pytest per-column | Round-trip motif recall, GOF vs released weights, structural invariants, kernel-checked certificate |

Supporting infrastructure: the simplified `Makefile` (six flows + `make
help`), a marimo EDA notebook and a synth-vs-real comparison notebook
(validated headless), and the persistent **LLM-Wiki** under `wiki/`
(Karpathy-style: index, append-only log, schema, interlinked pages — updated
on every ingest and incident).

## Build and run contract (simplified)

Exactly six user-facing flows; everything else is an internal dependency.
`make help` prints this summary.

| Target | What it does |
| --- | --- |
| `make clean` | remove ALL generated data (warehouse, models, Lean build outputs & exported data, release, bundle, synth, reports, caches); keeps source data, code and the shared Lean store |
| `make train` | the entire enclave flow: dbt build → minimodel training → ml scores → Lean build + checks + DP release + verified-sampler smoke → distill → tests → manifest → policy-filtered export → **one source-only bundle** in `dist/bundle/` |
| `make synth` | synthetic booking CSV from the local working tree (Lean generator from the local genspec → round-trip validation) into `synth/` |
| `make synth_from_bundle` | build the generator **from the bundle's Lean sources** (kernel re-checks every theorem), distill the bundle's release, generate into `synth_bundle/` — proves the bundle is self-contained |
| `make notebook-enclave` | serve the enclave EDA marimo notebook |
| `make notebook-synth` | serve the synth-vs-real comparison notebook |

**Lean toolchain policy**: `LEAN_HOME=/opt/data/lean` always. elan lives in
`$LEAN_HOME/elan`, the mathlib olean cache in `$LEAN_HOME/cache`, and the
pinned package store (mathlib, SampCert) is shared at `$LEAN_HOME/packages`
and symlinked into `lean/.lake/packages` — installed once, reused by every
build including `synth_from_bundle`. No per-project re-download, no reliance
on `~/.elan`.

**Deployment policy**: source-only. There is exactly ONE bundle format
(`dist/bundle/`): Lean sources verbatim + released artifacts + `distill.py`
(stdlib-only post-processing) + `build_from_source.sh` + manifest. No
compiled binaries ship; the consumer's Lean kernel re-checks every theorem
before anything runs. Binary deploy and alternative bundle layouts are
explicitly discarded. The only data that feeds synthesis is Lean-readable
JSON derived from the ε-DP release — never model weights, never the raw CSV.

**Privacy budget stance** (user decisions 2026-11-03/04: fidelity over
privacy, then an explicit ε REALLOCATION — privacy bounds may worsen; the
per-export cap is `dp_total_epsilon_max: 14/1` and the 1.2.0 ledger spends
**1315/100 = 13.15 ε**: country_transition raised to 4/100 (256 cells →
1024/100, the next-destination core), month-of-window replaces
month-of-year (the real observation ramp), a NEW repeat_struct release (the
A,A stay-extension anatomy), return_rate raised to 6/100, and the useless
dow release DROPPED. The bound is weak; state that plainly in every report
and treat every re-release as a budget event (fresh non-reproducible noise).
A **lifetime ledger** (`config/epsilon_ledger.jsonl`, append-only, keyed by
the source-data sha) tracks the cumulative spend of all exports of the same
data against `dp_lifetime_epsilon_max: 42/1` (~three exports at the current
ledger); the first dev export (release/20261103T224430Z, 1008/100,
attested: false) is recorded in it.

## Enclave Boundary

Keep the following inside the enclave by default:

- The raw CSV (`combined_set.csv`), user ids, trip ids, exact itineraries,
  and any per-user sequence or history.
- Individual reservations/legs/transitions, per-user trip structures
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Per-user embeddings and predictions; training checkpoints and fitted
  parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required, exact-
fraction ε budget; dev-host export requires `BOOKING_ALLOW_DEV_RELEASE=1`
and is marked `attested: false`). The three exit categories from
METHOD §6 apply:

1. **ε-DP histograms** over the public grids (reservations-per-user bins,
   hotel-country tiers, 16×16 tier transitions, nights bins, gap bins,
   months, dows, device cells, affiliate tiers, booker countries,
   return-rate bands), noised by a Lean executable using SampCert discrete
   Laplace, with the mechanism theorem proved and the ledger enforced in
   both Python and Lean.
2. **Suppression-thresholded aggregates** (no formal guarantee — declared
   as such): top-k cities and top-k ordered city pairs (≥ 50 distinct
   users AND ≥ 100 rows), gap moment summaries (≥ 200 gaps) — the
   next-destination vocabulary the generator renders from.
3. **Common-knowledge metadata**: the tier seeds (top-15 country names,
   top-8 affiliate ids), all class-bin edges, the observation window
   (2015-12-01 → 2017-03-01, epoch days 16770–17226), the CSV schema and
   `utrip_id = "<user>_<n>"` convention, and the declared public priors
   (return-boost diagonal, uniform dow, single-trip-per-user).

Suppression thresholds alone carry no formal guarantee; use them only for
artifacts explicitly marked non-DP (the top-k city/city-pair lists, gap
moments). DP noise randomness comes from a non-replayable source
(urandom-backed FFI); DP outputs are excluded from bitwise reproducibility
by design.

Require attestation and controlled key release, encrypted persistent
storage, restricted egress, and logs that redact user ids (`<user>`) and
city ids (`<city>`). CPU mini-models suffice at this scale (1.5M rows,
~64 features); GPU training stays outside the enclave threat model.

## Implementation Guidelines

### 0. Understand the data before modelling it — ✅ done (see `wiki/dataset-booking.md`)

Measured profile: 1,545,502 reservations, 259,159 users, 288,348 trips,
39,901 cities, 195 hotel countries, 3,611 affiliates, 5 booker countries,
window 2015-12-31 → 2017-02-27, zero nulls, zero bad dates. Legs per trip
median 5 (design: ≥4 legs except ~880 short trips); 99.11% contiguous
legs; nights median 1 (max 30); 38.07% of trips repeat a city; 9.51%
domestic; 1.11 trips per user; dow ~uniform. Every threshold above was
picked from these measured distributions, not defaults.

### 1. Define Concepts Before Fitting Models

Version every threshold (rc edges 4/5/6/7/9/13, nights edges 4/7, gap
edges 2/4, corridor 3 repeats / CV 4⁄5, tour 3 countries / 4 legs / 7-day
gaps, longstay 7 nights, rr band width 20%, top-k suppression 50/100,
gap-moment suppression 200, `dp_clip_total` 8, `dp_count_clip` 1, window
days 16770/17226) as dbt `vars:`, mirrored in `python/spec.py` and
`lean/EnclaveTrip/Params.lean`, locked by a sync test; bump `spec_version`
on any change. Thresholds as exact integers in Lean (days, not seconds).
The tier maps are part of the spec (dbt seeds, Python helpers, Lean
constants — one sync test over all three).

### 2. Start with Deterministic Baselines

SQL motif detection and empirical distributions first (trip structures,
transitions, repeat-city rate, nights/gap/month histograms). Mini-models
are added on top and gated: NextCountry vs **return-to-previous**,
TripLenClass vs the marginal rc histogram, StayNights vs the marginal
nights histogram, DeviceClass vs the marginal device mix. Gates are
data-dependent — read adoption flags from artifacts, not docs. Keep honest
negatives.

### 3. Separate Training from Inference

`make train` produces content-addressed immutable artifacts
(`sha256(name, data, hparams, code)`) under `models_out/`,
registry-activated (retrain only with `BOOKING_RETRAIN=1`); the inference
node `ml_trip_scores` materializes prediction tables. Rebuilding a mart
never retrains.

### 4. Preserve Uncertainty and Provenance

Content-hash data fingerprint (sha256 of the CSV), model version and
confidence on every prediction, quality flags, `untrained` placeholder
supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

Trips are (near-)complete chains by design; the residual censoring is the
challenge split (4-leg trips with a concealed last city in the test set —
not present in this training file). The month ramp (2016-08 peak → decay)
is observation censoring, not seasonality alone: released month cells are
calendar-month-of-year; the trend is a declared modelling gap. Flag, don't
drop.

### 6. Prevent Leakage

User-level 80/20 split (split on user id hash, fixed before fitting); with
259k users the split is comfortable; scalers fit on train only; labels
derived from the public tier taxonomy, never from evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

```text
P(next tier | prev tier, position)      — NextCountry teacher; distilled
                                          to the DP 16×16 transitions +
                                          return-boost prior + rr bands
P(rc bin    | first-leg features)        — TripLenClass / DP rc histogram
P(nights    | tier, month, device)       — StayNights / DP nights cells
P(device    | booker, aff tier, month)   — DeviceClass / DP device cells
                                          (χ² partner)
P(month     | —)                         — DP seasonality cells
```

### 9. Make Execution Reproducible

Data fingerprints, code versions, seeds, hyperparameters, split definitions
in `models_out/<v>/meta.json`; immutable artifact paths. DP outputs excluded
by design. SampCert pinned (`a2c53bb`); its toolchain dictates the
Lean/Mathlib versions; the toolchain and package store live in
`LEAN_HOME=/opt/data/lean` and are reused, never re-downloaded per project.

### 10. Keep Formal Claims Precise

Distinguish: proved in Lean (mechanism DP, sampler law, motif/trip
decidability, generator validity-by-construction) / checked on this
snapshot (`lake exe checks` SQL↔Lean agreement) / statistical estimates
(model metrics) / assumed (entropy source, I/O wrappers, and the SQL/Python
computing the clipped counts — DP holds for whatever counts are fed in;
their meaning is not proved). State trust boundary 2 explicitly.

### 11. Use Verified Randomness

DP noise and all generation-time draws only through SampCert-backed Lean
executables (`lake exe release`, `lake exe sample`, `lake exe generate`).
Never ad-hoc floating-point noise.

## Synthetic generation contract (planned)

- `python/distill.py` reads ONLY `dp_release.json` + released artifacts
  (post-processing theorem; `source.dp_release_sha256` recorded). Teachers
  map to spec constructs: NextCountry → DP tier transitions (+ declared
  return-boost diagonal prior + rr-band second-order return); TripLenClass
  → DP rc bins; StayNights → DP nights cells; DeviceClass → DP device
  cells. Public structural priors are declared `derived_from_data: false`.
- `lake exe generate <genspec> <n_users> <out>` renders CSV rows in the
  source schema: per user, draw booker country, rc bin (uniform inside the
  public range), start month (day-of-month uniform 1–28), then walk the
  trip: hotel tier from the DP transitions with return-boost, return-first
  second-order behavior from the rr band midpoint, city from the released
  top-k lists of the drawn tier (fallback synthetic ids), nights from DP
  cells (range 1–30), device/affiliate per leg from DP cells, dates
  contiguous by construction
  (checkin₁ < checkout₁ = checkin₂ ... inside the public window).
  **Validity by construction**: smart constructors with theorems —
  strictly increasing checkin days, nights ≥ 1, no overlap, window
  membership, `utrip_id = "<user>_1"`.
- Plant the rarest motifs in `plan.json` (user 1: repeat-city trip; user 2:
  4-leg 3-country contiguous tour; user 3: trip with a ≥ 7-night stay);
  round-trip re-ingestion through the production dbt pipeline must recall
  all three at 1.0 and `lake exe checks` must re-verify (exit 0 / 2 soft,
  1 hard); `scripts/synth.py certify` emits a kernel-checked certificate
  (`by decide`) over the plan.
- Per-column tests: every output column matches its declared distribution
  (GOF vs released weights), exhibits its declared property (contiguity,
  window, schema), or satisfies a structural invariant; gaps declared in
  `meta.json assumptions.fields_not_modelled` (initially: multi-trip users,
  within-tier city choice beyond the top-k lists, dow placement,
  sub-day time, affiliate choice beyond DP tier cells, cross-leg
  correlations beyond the released associations).
- The generator's ONLY inputs are the genspec JSON and the Lean sources —
  synthesis works identically from the working tree (`make synth`) and
  from the source-only bundle (`make synth_from_bundle`).

## Acceptance checklist (status as built, 2026-11-04)

| Area | Criterion | Status |
| --- | --- | --- |
| Data profiling | Measured profile in wiki; thresholds from measurements | ✅ `wiki/dataset-booking.md` (1.55M rows; contiguity 99.11%, repeat-city 38.07% measured before design) |
| Contract freeze | Cross-layer interfaces pinned before parallel work | ✅ `wiki/booking-contract.md` (grids, tables, JSON schemas, ledger, generator I/O) |
| Spec sync | dbt vars = spec.py = Params.lean (+ tier maps ×3), test green | ✅ spec 1.1.0: 31 vars × 3 mirrors (incl. rc_13p_draw_hi 24, synth_repeat_tol_pct 15, caps); sync tests green |
| dbt pipeline | staging/features/marts/ml build + schema tests green | ✅ 170/170 (28 table models + 139 tests + seeds); ml_trip_scores materializes from the active registry |
| Minimodels | Each gated vs frozen baseline; content-addressed; honest negatives kept | ✅ next_country 0.9186 = return-to-previous (tie: the model learned exactly the dominant stay-in-tier signal), device 0.8804 vs 0.6435 majority (adopted), trip_len 0.4148 vs 0.4142, stay_nights 0.5768 = majority (tie), profile_ae scoring-only; all recorded in meta.json |
| Lean mirrors | `lake exe checks` SQL↔Lean agreement on exported observations (motifs, χ², trips) | ✅ 143/143 agree (50 returns, 25 corridors, 25 tours, 25 longstays, 6 exact-ℚ χ², 12 trip checks); 20+ proved claims, no sorry/native_decide |
| Private release | Mechanism theorem on SampCert; exact-rational ledger enforced both sides; over-budget refusal tested | ✅ spec 1.3.0 ledger **1368/100 = 342/25 = 13.68 ≤ 14/1** (transitions 4/100, month-of-window, repeat_struct, return_rate 6/100, dow dropped); per-export gate refuses on dev host without BOOKING_ALLOW_DEV_RELEASE; LIFETIME ledger enforces dp_lifetime_epsilon_max 42/1 across exports, keyed by data sha, tested |
| Verified sampling | Sampler law proved; executable draws pass GOF | ✅ `sampleCellPMF_law`; 5000-sample smoke green; conformance tests green |
| Distillation | Weight-by-weight provenance test green; fidelity under a pre-declared criterion | ✅ genspec keyed to dp_release sha (provenance-skip); fidelity accepted: student 0.9101 ≥ return-to-previous 0.8672 − 0.02 (teacher 0.9186, 250,140 held-out transitions) |
| Generation | `lake exe generate` renders schema-valid CSV; planted recall 1.0 after re-ingestion; kernel-checked certificate | ✅ validate PASS (exit 0) at n = 2000 users under spec 1.3.0: recall 1.0 (repeat-city + tour + longstay), 0 overlap rows, contiguous-leg share 0.9898 (real 0.9911), TV vs genspec: res_count 0.021 / country 0.021 / nights 0.002 / gap 0.002, 0 Lean disagreements on re-ingested synth; **replicates PASS over 8 runs: repeat-city share 0.3957, res_count TV mean 0.0159**; certificate PROVED (`by decide`, kernel-only) over 120 planned users |
| Realism audit | enclave models grade the synthetic data | ✅ under spec 1.3.0: **trips-per-user share 0.096 vs real 0.0893** (was 0 — multi-trip rendering), **single-device trip share 0.7225 vs real 0.7067** (iid draws would render ~12%), consecutive same-city legs 0.0974 vs 0.0901, records/user mean 5.83 vs real 5.96, repeat-city share 0.394 vs 0.378, month-of-window follows the real ramp (TV 0.035), affiliate same-prev-tier 0.848 vs 0.754 (residual = the within-trip tier ORDER a count-only release cannot carry — declared); mix TVs ≤ 0.032; 0% of synth users above the real AE p99 |
| Simplified ops | the six make targets + `make help`; `LEAN_HOME` store reused; single source-only bundle; `synth_from_bundle` green | ✅ six flows all green incl. export (user-run dev release) + bundle + `synth_from_bundle`; synth flows now run validate + replicates (REPLICATES=8); 99 tests pass / 0 skipped; both notebooks export headless clean |
| Wiki discipline | Pages + append-only log updated on every ingest/incident | ✅ contract/dataset/migration/pipeline pages + log entries (incl. the longstay scalar-vs-list, tour gap-normalization and per-hop-return incidents) |

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible booking-trip encoder plus a
verified synthetic-reservation generator with an explicit output contract.

- Enclave execution protects processing within the selected platform's
  threat model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit (one user, add/remove their whole reservation
  history) and assumptions. The budget is deliberately loose (ε ≈ 10.08 of
  the 14/1 cap, user's explicit fidelity-over-privacy choice); that is
  stated, not hidden, and every re-release composes against it.
- Statistical evaluation establishes empirical usefulness for
  next-destination recommendation research.

None of these alone guarantees realism of future synthetic reservations.
Released non-DP artifacts (suppressed top-k city lists, gap moments) and
evaluation metrics remain outside the DP guarantee. Multi-trip users,
sub-day timing and cross-leg correlations beyond the released associations
are declared gaps until a ledger entry covers them.

# torch-lean-arch: Enclave Travel-Sequence Encoder and synthetic data generator

## Goal

Build a reproducible encoder that converts a private corpus of international
travel transfers (`country_transfer.tsv`: 84,964 rows, 29,640 users, 77
countries, 2012-04-04 → 2013-09-16, derived from Foursquare TIST2015
check-ins; one row = one user's country-to-country transfer with timestamps,
boundary venues/coordinates and per-country stay durations) into a versioned
travel-behavior representation: per-user travel-count distributions, travel
sequences and recurring motifs, inter-travel gap features with their
statistical moments, and fitted distributions — for mobility-analysis training
and downstream synthetic travel-sequence generation. Run the encoder inside an
enclave so raw travel histories and sensitive intermediate artifacts remain
within the protected processing boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

This is the third instantiation of the method proven on network-flow telemetry
and migrated once already (stock series): see `METHOD.md` (the
dataset-agnostic replication guide, especially §12's substitution map),
`wiki/migration-plan-foursquare.md` (the phased plan for THIS dataset) and
`wiki/dataset-foursquare.md` (the measured data profile). The five-layer
architecture, three release categories, spec-mirror discipline, Lean-only
generator, and every calibration lesson in METHOD §6 carry over unchanged.

**What is sensitive here:** each user's travel history — where they were,
when, how often, and in what order. A sequence of (country, timestamp) pairs
is a strong re-identification vector (a handful of known trips singles out a
user). The protected entity is the **user**; venue ids and raw coordinates
are structurally excluded from release; user ids never cross the boundary in
any form.

The encoder learns and measures data behavior from the input dataset,
producing an explicit specification that a later synthetic-data generator can
target.

### Dataset-specific emphases (this migration's brief)

1. **User-level travel-count distribution** — the heavy-tailed
   travels-per-user histogram (median 2, p90 5, p99 14, max 195) is a
   first-class release and the backbone of generation: synthetic users must
   reproduce it.
2. **Travel sequences** — recurring motifs, order structure, sequence
   minimodels and sequence features: region-transition matrices, return
   trips (72.1% of consecutive transfer pairs are A→B followed by B→A —
   the dominant structure in this data), commuter corridors, tours.
3. **Inter-travel date features** — gap classes, their statistical moments
   (measured: mean 40.7 d, sd 65.8 d, mean/median 3.16 — heavy right skew),
   and gap-based motifs (bursts, episodes) are retained through the DP
   release and reproduced by the generator.

### Outputs (planned)

- Per-transfer and per-user features: region classes for both endpoints,
  inter-travel gap (days) and gap class, stay-duration classes, hop distance
  (haversine between boundary venues) and distance class, return flag,
  episode (journey) grouping, calendar features (dow, month), quality flags.
- A **public region taxonomy** (one half of the DP cell grid, fixed in the
  spec, independent of the data): the ISO 3166 country list mapped to 8
  fixed world regions — `NA` (Northern America), `LA` (Latin America &
  Caribbean), `EU` (Europe incl. Russia), `ME` (Middle East & North Africa),
  `AF` (Sub-Saharan Africa), `SA` (Central & South Asia), `EA` (East &
  Southeast Asia), `OC` (Oceania). The map is a public constant (UN M49
  derived), versioned in the spec, total over all ISO codes.
- Public class bins (all fixed in the spec; edges chosen from measured
  distributions, then frozen as public):
  - **travel-count bins** (per user): 1, 2, 3–4, 5–8, 9–16, 17–32, 33+;
  - **gap classes** (days between consecutive transfers of one user):
    `hop` < 2, `trip` 2–7, `monthly` 7–30, `quarterly` 30–90, `rare` ≥ 90
    (measured shares 10.5 / 24.5 / 33.6 / 17.1 / 14.3%);
  - **stay-duration classes**: < 1 d, 1–7 d, 7–30 d, 30–90 d, ≥ 90 d
    (measured shares 13.7 / 33.3 / 16.2 / 12.2 / 24.5%);
  - **distance classes** (km): < 500, 500–2000, 2000–7000, ≥ 7000
    (measured shares 21.6 / 29.2 / 28.0 / 21.2% — an almost even public
    grid, by luck not design).
- Travel motifs with public thresholds (SQL and Lean mirrors):
  - **return trips** — consecutive A→B then B→A by the same user (the
    A→B→A analog; dominant: 39,879 of 55,324 consecutive pairs);
  - **commuter corridors** — the same ordered country pair repeated ≥ 5
    times by one user with gap CV ≤ 0.8 (the beacon analog: periodic
    re-contact; 761 user-corridors qualify on count, median CV of those is
    0.78);
  - **tours** — ≥ 3 consecutive transfers, every gap ≤ 7 d, spanning ≥ 3
    distinct countries (the rarest motif — 1,126 instances across 978
    users; the planted round-trip pattern);
  - **bursts** — ≥ 3 transfers within 14 days (7,196 windows);
  - **long stays** — stay duration ≥ 90 d (home-base/residency signal,
    24.5% of stays).
- **Episode (journey) structure**: consecutive transfers with gaps ≤ 14 d
  grouped into one journey; episode-length histogram over public bins
  (1, 2, 3, 4–5, 6+) — the session analog, and the unit the sequence
  models generate.
- Sequence analysis: the 8×8 region transition matrix with lift; sequences
  broken at gaps > 90 d (a `rare` gap never creates a false transition —
  the "breaks split transitions" lemma pattern).
- Conditional sequence and anomaly scores from versioned mini-models.
- Versioned mini-models, evaluation results, and candidate formal
  specifications.
- Differentially private releases of aggregate distributions, with the noise
  mechanism formally verified with
  [SampCert](https://github.com/leanprover/SampCert) (Lean 4, pinned commit
  `a2c53bb` and its toolchain, as in the reference implementation).
- A formally verified sampler for drawing from released distributions, built
  on SampCert's samplers — the entry point for the downstream generator.
- A standalone Lean generator (`lake exe generate`) that renders synthetic
  transfers **in the source TSV format** (`userid\tdt\tprev_country\t
  prev_last_venue_id\tprev_last_lat\tprev_last_long\tprev_duration\t
  next_country\tnext_first_venue_id\tnext_first_lat\tnext_first_long\t
  next_duration`) from the distilled genspec alone, validated by re-ingestion
  through the production pipeline, with kernel-checked certificates.
  **The only artifacts the generator consumes are Lean sources and the
  released JSON — no compiled binaries, no Python models, no raw data.**

## Processing Architecture (planned)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | DuckDB `read_csv` in dbt staging (`stg_transfers`) | Canonical transfer rows with stable `obs_id` and content-hash `obs_key`; typed timestamp/coords/durations |
| Cleaning and features | dbt + DuckDB (`feat_transfers`, `feat_users`, `feat_episodes`) | Region classes, gap days + class, duration classes, haversine distance + class, return flags, episode ids, dow/month, quality flags |
| Mini-model fitting | PyTorch (`python/train_minimodels.py`), CPU-deterministic mode supported | NextRegionGRU (next region from history; gate: return-to-previous baseline), GapClassGRU (next gap class; gate: marginal), EpisodeShapeModel (episode length/shape; gate: unigram), UserProfileAE (per-user anomaly/embedding); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_travel_scores`) | Per-transfer/per-user predictions with confidence and model version; never retrains on rebuild |
| marts | dbt SQL | `mart_user_stats`, `mart_motif_returns`, `mart_motif_corridors`, `mart_motif_tours`, `mart_motif_bursts`, `mart_episodes`, `mart_dp_*` input tables |
| Association analysis | dbt SQL (`mart_assoc_cells`, `mart_feature_assoc`) | χ²/Cramér's V between discretized features (gap class × distance class × return flag × region) on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Sequence analysis | dbt SQL (`mart_seq_transitions`) | 8×8 region transition matrix with lift, gap-aware (broken at `rare` gaps) |
| Formal specification | Lean 4 + Mathlib (`lean/EnclaveTravel/Motifs.lean`, `Params.lean`) plus `lake exe checks` | Decidable motif definitions, proof-status ledger, SQL↔Lean agreement on exported observations |
| Private release | Lean 4 + SampCert discrete Laplace over clipped per-user contributions (`lake exe release`) | User-level ε-DP histograms over the public grids, proved mechanism theorem, exact-rational budget ledger enforced in Lean AND Python |
| Verified sampling | SampCert `UniformSample` + bucketing map, law proved (`sampleCellPMF_law` pattern) (`lake exe sample`) | Samples from released distributions for generation and GOF checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: travel-count histogram, region transition rows, gap/duration/distance histograms, episode shapes, top-k corridors, schema constraints, public calendar window |
| Generation | `lake exe generate <genspec> <n_users> <out>` | Synthetic TSV transfers + `plan.json` ground truth (planted motifs) |
| Validation | re-ingest through production dbt + Lean checks; pytest per-column | Round-trip motif recall, GOF vs released weights, kernel-checked certificate |

Supporting infrastructure: the simplified `Makefile` (see **Build and run
contract** below), a marimo EDA notebook and a synth-vs-real comparison
notebook (validated headless), and the persistent **LLM-Wiki** under `wiki/`
(Karpathy-style: index, append-only log, schema, interlinked pages — updated
on every ingest and incident).

## Build and run contract (simplified — this migration's second brief)

Exactly six user-facing flows; everything else is an internal dependency:

| Target | What it does |
| --- | --- |
| `make clean` | remove ALL generated data (warehouse, models, Lean build outputs & exported data, release, bundle, synth, reports, caches); keeps source data, code and the shared Lean store |
| `make train` | the entire enclave flow: dbt build → minimodel training → ml scores → Lean build + checks + DP release + verified-sampler smoke → distill → policy-filtered export → **one source-only bundle** in `dist/bundle/` |
| `make synth` | synthetic TSV from the local working tree (Lean generator from the local genspec) into `synth/`, then round-trip validation |
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
(`dist/bundle/`): Lean sources verbatim + `dp_release.json` + `distill.py`
(stdlib-only post-processing) + `build_from_source.sh` + manifest. No
compiled binaries ship; the consumer's Lean kernel re-checks every theorem
before anything runs. Binary deploy and alternative bundle layouts are
explicitly discarded. The only data that feeds synthesis is Lean-readable
JSON derived from the ε-DP release — never model weights, never the raw TSV.

## Enclave Boundary

Keep the following inside the enclave by default:

- The raw TSV (`country_transfer.tsv`), user ids, venue ids, raw coordinates,
  exact timestamps, and any per-user sequence or history.
- Individual transfers, per-user features, motif hits, episode structures
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Per-user embeddings and predictions; training checkpoints and fitted
  parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required, exact-
fraction ε budget). The three exit categories from METHOD §6 apply:

1. **ε-DP histograms** over the public grids (travel-count bins, region mix,
   region transitions, gap classes, stay-duration classes, distance classes,
   month seasonality, hour-of-day, episode lengths), noised by a Lean
   executable using SampCert discrete Laplace, with the mechanism theorem
   proved and the ledger enforced in both Python and Lean.
2. **Suppression-thresholded aggregates** (no formal guarantee — declared as
   such): per-gap-class moment summaries (mean/sd of gap days within class)
   with min-count suppression, released to let the generator place gaps
   inside bins realistically.
3. **Common-knowledge metadata**: the ISO→region map, the class-bin edges,
   the observation calendar window (2012-04-04 → 2013-09-16), the TSV schema
   and venue-id shape (24-hex convention), country centroid coordinates
   (public geography), **top-k corridors** — ordered country pairs with
   ≥ 50 distinct users AND ≥ 100 transfers (283 corridors clear 50 users;
   MY→SG alone has 1,984 users) — and **top-k countries** — per-country
   arrival/departure volumes under the same floors (70 of 77 countries
   clear them, covering 99.7% of transfer appearances; ISO codes are a
   public taxonomy, and a country's aggregate traffic above the floor is
   population-level knowledge). Corridors and countries below threshold are
   never released; venue ids and user ids are structurally excluded
   regardless of popularity.

**Privacy unit (the critical design decision — METHOD §6 lessons applied):**

- One protected entity = **one user** (their entire travel history is one
  record in SampCert's add/remove-one `Neighbour` relation). The protected
  population is the user base: membership, trajectories, and timing.
- **n = 29,640 entities is comfortable** — the opposite regime from the
  stock migration's n = 8. METHOD's SNR estimate `share × n × ε_total / k`
  gives ≈ 139 on the 64-cell transition grid at ε_cell = 3/100 for a
  10%-share cell. The risk here is not SNR but the **heavy tail**: one user
  has 195 transfers, so unclipped contributions would let a single user
  dominate cells. Phase 6 still REQUIRES the pre-noise simulation (clipped
  counts vs raw distribution, SNR per cell) before the ledger is committed.
- **Bounded contributions** via proportional L1 allocation: each user's
  per-grid cell vector is allocated over a total budget `dp_clip_total = 8`
  (≈ p95 of travels/user; largest-remainder rounding in SQL), so per-cell
  sensitivity equals the total and the proved mechanism applies unchanged.
  Uniform per-cell clipping is known to flatten the mix (METHOD §6) — not
  used. The travels-per-user histogram needs no clipping: each user falls in
  exactly one bin, sensitivity 1.
- **Semantics**: equal per-user budgets estimate the per-user-average
  distribution, not the transfer-weighted one — and here they genuinely
  differ (median user: 2 travels; max: 195). This is the RIGHT semantics for
  this brief: the release describes the typical user's behavior, and the
  travels-per-user histogram carries the tail separately. Measure both
  (per-user-average vs row-weighted mix), document the gap as a declared
  property, and let the realism audit quantify it.
- **No rank labels needed for users** (nothing per-user is ever released);
  countries are a public taxonomy, so unlike tickers they appear by name —
  but only at region granularity under DP, or as top-k corridors under
  suppression. Synthetic users carry fresh ids (`1..n`); synthetic venue ids
  are generated 24-hex strings matching the public shape convention.
- **Coordinates**: raw venue coordinates never leave. Synthetic coordinates
  are public country centroids plus bounded jitter — declared
  `derived_from_data: false`; intra-country geography is in
  `assumptions.fields_not_modelled`.
- **Cross-user structure** (shared venues, co-travel): a protected relation,
  never released, declared out of scope for v1.

**ε ledger v2.1** (exact fractions; enforced in Lean AND Python; refusal
path tested; rebalanced 2026-11-03 to admit hour-of-day — each new width
displaced an old one, exactly as the saturation rule demands):
travel-count bins 7 × (4/100) = 28/100; region mix 8 × (3/100) = 24/100;
region transitions 64 × (3/100) = 192/100; gap classes 5 × (6/100) = 30/100;
duration classes 5 × (5/100) = 25/100; distance classes 4 × (5/100) = 20/100;
month seasonality 12 × (1/100) = 12/100; **hour of day 24 × (1/100) =
24/100**; episode lengths 5 × (5/100) = 25/100; return-rate bands 5 ×
(4/100) = 20/100.
Total 400/100 = 4/1 — EXACTLY saturated. `lake exe release` refuses exports
over budget (exit 2). Day-of-week stays OUT of the ledger (measured nearly
uniform, max/min 1.28 — a declared uniform public prior); hour of day is IN
(measured max/min 2.78, far from uniform — worth its ε). Worst-case SNR
after the rebalance: the smallest hod/month cells carry ≈ 3–6k clipped mass
against a noise scale of clip/ε = 800 — comfortably above noise.

Suppression thresholds alone carry no formal guarantee; use them only for
artifacts explicitly marked non-DP (the gap-moment summaries, the top-k
corridors). DP noise randomness comes from a non-replayable source
(urandom-backed FFI); DP outputs are excluded from bitwise reproducibility
by design.

Require attestation and controlled key release, encrypted persistent storage,
restricted egress, and logs that redact user ids (`<user>`) and venue ids
(`<venue>`). CPU mini-models suffice at this scale; GPU training stays
outside the enclave threat model.

## Implementation Guidelines

### 0. Understand the data before modelling it — ✅ done (see `wiki/dataset-foursquare.md`)

Measured profile: 84,964 transfers, 29,640 users (median 2 travels, p90 5,
p99 14, max 195; 23,528 users with ≥ 2), 77 countries, 2,791 ordered
corridors (MY↔SG dominant: 5,808 + 5,633 transfers), 72.1% consecutive-pair
return rate, gap mean 40.7 d / sd 65.8 d, durations 0 s – 527 d, no NULLs,
no self-transfers (prev ≠ next on every row), dow nearly uniform, strong
month seasonality (May 13,610 vs Sep 2,484 — the file starts Apr 2012 and
ends Sep 2013, so month counts mix seasonality with coverage; model months
within the public window, don't interpret as pure seasonality). Every
threshold above was picked from these measured distributions, not defaults.

### 1. Define Concepts Before Fitting Models

Version every threshold (region map, class-bin edges, corridor n ≥ 5 and
CV ≤ 0.8, tour 3/7d/3, burst 3/14d, long-stay 90 d, episode gap 14 d,
sequence break 90 d, top-k corridor suppression 50 users/100 transfers,
`dp_clip_total = 8`) as dbt `vars:`, mirrored in `python/spec.py` and
`lean/EnclaveTravel/Params.lean`, locked by a sync test; bump `spec_version`
on any change. Thresholds as exact rationals in Lean (CV ≤ 4/5, gap edges in
whole seconds: 172800, 604800, 2592000, 7776000). The ISO→region map is part
of the spec (a dbt seed, a Python dict, a Lean function — one sync test over
all three). Ingest needs no repairer (the TSV parses clean — measured);
record that as a fact, keep the verified-repairer pattern in reserve.

### 2. Start with Deterministic Baselines

SQL motif detection and empirical distributions first (returns, corridors,
tours, transition counts, gap/duration/distance histograms). Mini-models are
added on top and gated: NextRegionGRU vs **return-to-previous** (the honest
baseline here — it alone scores 72.1% on next-country direction),
GapClassGRU vs marginal class shares, EpisodeShapeModel vs unigram lengths.
Gates are data-dependent — read adoption flags from artifacts, not docs.
Keep honest negatives: with a 72% baseline, the GRU may well lose; that
result is a finding, not a failure.

### 3. Separate Training from Inference

`make train` produces content-addressed immutable artifacts
(`sha256(name, data, hparams, code)`) under `models_out/`,
registry-activated; the inference node `ml_travel_scores` materializes
prediction tables. Rebuilding a mart never retrains.

### 4. Preserve Uncertainty and Provenance

Stable `obs_id`, content-hash `obs_key`, data fingerprint (sha256 of the
TSV), model version and confidence on every prediction, quality flags,
`untrained` placeholder supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

Break sequences at gaps > 90 d; never let a `rare` gap create a false
transition. The observation window is finite (18 months): treat
first/last transfers per user as censored — a user's "first" country is
where observation began, not where they were born. Durations of 0 s exist
(single-checkin countries); flag, don't drop.

### 6. Prevent Leakage

User-level 80/20 split (split on user id hash, fixed before fitting — a
time split would leak seasonal structure across the boundary for the same
user); scalers fit on train only; weak labels derived from the public
taxonomy, never from evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

```text
P(next_region | region history)     — NextRegionGRU; and the empirical 8×8
                                      transition matrix with lift
P(gap class   | history, position)  — GapClassGRU
P(episode len | first-hop class)    — EpisodeShapeModel / empirical rows
P(distance    | region pair)        — empirical per-pair distance bins
P(return      | corridor, gap)      — empirical return-rate bands
```

### 9. Make Execution Reproducible

Data fingerprints, code versions, seeds, hyperparameters, split definitions
in `models_out/<v>/meta.json`; immutable artifact paths. DP outputs excluded
by design. SampCert pinned; its toolchain dictates the Lean/Mathlib versions;
the toolchain and package store live in `LEAN_HOME=/opt/data/lean` and are
reused, never re-downloaded per project.

### 10. Keep Formal Claims Precise

Distinguish: proved in Lean (mechanism DP, sampler law, motif decidability,
generator validity-by-construction) / checked on this snapshot (`lake exe
checks` SQL↔Lean agreement) / statistical estimates (model metrics) /
assumed (entropy source, I/O wrappers, and the SQL computing the clipped
counts — DP holds for whatever counts are fed in; their meaning is not
proved). State trust boundary 2 explicitly, as the reference implementation
does.

### 11. Use Verified Randomness

DP noise and all generation-time draws only through SampCert-backed Lean
executables (`lake exe release`, `lake exe sample`, `lake exe generate`).
Never ad-hoc floating-point noise.

## Synthetic generation contract (planned)

- `python/distill.py` reads ONLY `dp_release.json` + released artifacts
  (post-processing theorem; `source.dp_release_sha256` recorded). Teachers
  map to spec constructs: NextRegionGRU → 8×8 DP transition rows;
  GapClassGRU → DP gap-class histogram (+ suppressed within-class moments);
  EpisodeShapeModel → DP episode-length histogram; corridor knowledge →
  released top-k corridor list. Public structural priors (e.g. a
  return-transition boost on the matrix diagonal blocks, the uniform-dow
  prior) are declared `derived_from_data: false`.
- `lake exe generate <genspec> <n_users> <out>` renders TSV rows directly:
  per user, draw a travel count from the released travel-count histogram and
  a return-rate band; hops are **return-first** — with the band's probability
  the next country is the one the user came from, otherwise walk the region
  via the verified sampler over the DP transition rows and instantiate the
  country from the released **top-k country weights** (arrivals, restricted
  to the target region); draw gap classes then gap values; timestamps =
  cumulative gaps inside the public window with the arrival **hour of day
  drawn from the DP hod histogram** (minutes/seconds a declared uniform
  prior); durations by class, **capped at the gap to the next transfer** so
  stay intervals never overlap; coordinates = public centroid; venue ids =
  generated 24-hex. **Validity by construction**: `mkTravelRow` /
  `mkUserHistory` smart constructors with theorems — per-user timestamps
  strictly increasing, prev_country ≠ next_country on every row, durations
  ≥ 0, duration chaining (row i's next stay = row i+1's previous stay when
  consecutive), all dates inside the public window, and **no simultaneous
  presence** (a stay interval never extends past the next transfer —
  `mkUserHistory_no_overlap`); `times_sorted` via `List.Pairwise`. The same
  impossible-presence condition is flagged (`qf_overlap`) and removed in
  ingestion — the source data measures 0 such rows under UTC parsing (a
  local-time parse shows 13 DST false positives), so the flag is a standing
  guard and the production-side check the synthetic round-trip must pass.
- Plant the rarest motif (a tour: 3 hops ≤ 7 d apart across 3 countries)
  plus one commuter corridor in `plan.json`; round-trip re-ingestion through
  the production dbt pipeline must recall both at 1.0 and `lake exe checks`
  must re-verify (exit 0 / 2 soft, 1 hard).
- Per-column tests: every output column matches its declared distribution
  (χ² GOF vs released weights), exhibits its declared property, or satisfies
  a structural invariant; gaps declared in `meta.json
  assumptions.fields_not_modelled` (initially: intra-country geography,
  venue identity/popularity structure, cross-user relations, exact
  within-bin gap shapes beyond released moments, country-level transitions
  beyond top-k corridors).
- The generator's ONLY inputs are the genspec JSON and the Lean sources —
  synthesis works identically from the working tree (`make synth`) and from
  the source-only bundle (`make synth_from_bundle`).

## Acceptance checklist (status as built, updated 2026-11-03 — ledger v2.1)

| Area | Criterion | Status |
| --- | --- | --- |
| Data profiling | Measured profile in wiki; thresholds from measurements | ✅ `wiki/dataset-foursquare.md` |
| Spec sync | dbt vars = spec.py = Params.lean (+ region map ×3), test green | ✅ 8 sync tests green incl. the 77-row region map and data coverage |
| dbt pipeline | staging/features/marts/ml build + schema tests green | ✅ 127 PASS (26 models + 99 tests + seed) incl. mart_dp_hod, mart_topk_countries, qf_overlap removal; ml_travel_scores on the active artifact |
| Minimodels | Each gated vs frozen baseline; content-addressed; honest negatives kept | ✅ all four adopted on real data: NextRegionGRU 75.7% vs return-to-previous 72.0%, GapClassGRU 39.4% vs 29.1%, EpisodeShape NLL 0.733 vs 0.755, ProfileAE mse 0.029 |
| Lean mirrors | `lake exe checks` SQL↔Lean agreement on exported observations | ✅ 154/154 agree (returns/corridors/tours/bursts/longstays/χ²), 12/12 episodes; 24 claims proved + 3 assumed |
| Private release | Mechanism theorem on SampCert; exact-rational ledger 400/100 enforced both sides; over-budget refusal tested | ✅ spent exactly 400/100 = 4/1; Lean exit-2 refusal + Python Fraction cross-check both tested |
| Verified sampling | Sampler law proved; executable draws pass χ² GOF | ✅ `sampleCellPMF_law`; catsample χ² conformance green |
| Distillation | Weight-by-weight provenance test green; fidelity under the pre-declared criterion | ✅ provenance cell-exact (marginals = release; diagonal return_boost prior declared `derived_from_data: false`); fidelity **honest negative**: first-order student 56.5% < return-baseline 72% − 0.02 — as predicted, the matrix can't express second-order returns; the generator compensates via the released return_rate bands (declared) |
| Generation | `lake exe generate` renders schema-valid TSV; planted tour + corridor recall 1.0; kernel-checked certificate | ✅ validate exit 0 (recall 1.0 + 1.0, overlap 0, travel-count TV 0.023, region TV 0.031, hod TV 0.029, country TV 0.087 over all 70 released countries, in/out 0.058, 0 Lean disagreements); `certificate.json` kernel-checked; no-overlap proved (`mkUserHistory_no_overlap`) |
| Simplified ops | the six make targets; `LEAN_HOME=/opt/data/lean` reused; single source-only bundle; `synth_from_bundle` green | ✅ six flows wired; shared elan/cache/package store bootstrapped by `lean-store`; one 188K source-only bundle; bundle rebuild re-checks all theorems |
| Wiki discipline | Pages + append-only log updated on every ingest/incident | ✅ ongoing |

Open items (deliberate): the realism audit's former return-rate gap is FIXED
by the 2026-11-03 return-first redesign (synth 0.796 vs real 0.818, was
0.20); remaining declared gaps are intra-country geography, venue structure,
cross-user relations, within-bin gap shapes, the exact country-level
transition matrix, and sub-hour diurnal microstructure. Frozen-rebuild
reproduce script deferred, as in the stock build.

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible travel-sequence encoder plus a
verified synthetic-travel generator with an explicit output contract.

- Enclave execution protects processing within the selected platform's threat
  model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit (one user, add/remove their whole history) and
  assumptions. With n = 29,640 users, utility is expected to be good — but
  it is measured (pre-noise SNR simulation, released-vs-clipped comparison),
  not assumed.
- Statistical evaluation establishes empirical usefulness.

None of these alone guarantees realism of future synthetic travel data.
Released non-DP artifacts (suppressed moments, top-k corridors) and
evaluation metrics remain outside the DP guarantee. Intra-country geography,
venue structure and cross-user relations are declared gaps until a ledger
entry covers them.

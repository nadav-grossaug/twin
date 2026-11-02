# torch-lean-arch: Enclave Mobility-Trajectory Encoder and synthetic data generator

## Goal

Build a reproducible encoder that converts a private corpus of GPS mobility
trajectories (`geolife_data.tsv`: 5,440,384 points, 64 users, 4,464
trajectories, 2007-04-12 → 2011-12-31, the mode-labelled subset of Microsoft
GeoLife; one row = one GPS point with user id, trajectory id, timestamp,
lat/lon, altitude and a transport-mode label) into a versioned
mobility-behavior representation: staypoints and per-user anchors (home /
work / other), daily-routine motifs, anchor-to-anchor visit sequences, trip
kinematics (mode, speed, turning angles, trip distance), GPS-noise
statistical moments, and fitted distributions — for mobility-analysis
training and downstream synthetic trajectory generation. Run the encoder
inside an enclave so raw trajectories and sensitive intermediate artifacts
remain within the protected processing boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

This is the fourth instantiation of the method (pcap flows → stock series →
Foursquare travel → GeoLife trajectories), and it closes a loop: the
method's original reference implementation was built on GeoLife
(`wiki/geolife-reference.md`), so this migration both reuses that domain
logic and upgrades it to the current simplified contract (six flows,
LEAN_HOME store, ONE source-only bundle). See `METHOD.md` (the
dataset-agnostic replication guide, especially §12's substitution map),
`wiki/migration-plan-geolife.md` (the phased plan for THIS dataset) and
`wiki/dataset-geolife.md` (the measured data profile). The five-layer
architecture, three release categories, spec-mirror discipline, Lean-only
generator, and every calibration lesson in METHOD §6 carry over unchanged.

**What is sensitive here:** each user's movement history — where they live
and work, where they go, when, how often, and in what order. A GPS trace is
one of the strongest re-identification vectors known (home+work alone
identifies most people). The protected entity is the **user**; raw
coordinates, staypoint locations and anchor positions are structurally
excluded from release; user ids never cross the boundary in any form. All
released geometry is **anchor-relative** (distances, angles, dwell times) —
never absolute.

The encoder learns and measures data behavior from the input dataset,
producing an explicit specification that a later synthetic-data generator can
target.

### Dataset-specific emphases (this migration's brief)

1. **Staypoints and anchors as the behavioral backbone** — motifs follow
   users' daily patterns: staypoint detection (≥ 20 min within 200 m),
   per-user anchor discovery (home = dominant night cluster, work = dominant
   workday cluster), and daily-routine motif classes (home-day, commute,
   errand loop, tour, irregular). Synthetic users must reproduce the daily
   anchor rhythm, not just marginal distributions.
2. **Feature co-occurrence and correlation** — χ² statistical association
   analysis between discretized columns/features/motifs on full contingency
   grids (mode × speed class, mode × hour bin, anchor role × arrival hour,
   dwell class × role, motif × day type, distance class × mode), with
   Cramér's V and lift; Lean recomputes every χ² exactly over ℚ.
3. **Timely sequence flow** — input records follow time-ordered structure at
   two scales: anchor-visit transitions within a user-day (H→W→H, broken at
   observation gaps so a gap never creates a false transition) and
   within-trajectory mode-leg chains (walk→bus→walk).
4. **Trajectory realism** — trajectories carry GPS noise whose statistical
   moments are estimated in the enclave (staypoint-interior residuals: sd,
   MAD, p95, lag-1 autocorrelation) and reproduced by the generator; and
   synthetic paths follow a correlated-heading random walk — **no straight
   lines, no strict 90° grid turns** (validated: bounded collinear runs, no
   atoms at 0°/±90° in the turn-angle histogram).
5. **Juxtaposition manipulation** — a mandatory `manipulate` step in the
   synth flow relocates every synthetic user from the (Beijing-learned)
   anchor-relative frame onto **Tokyo** public geography. The generator
   itself emits a neutral local frame; **no Beijing lat/lon coordinates ever
   appear in the synthetic output** (bbox-asserted at validation).

### Outputs (planned)

- Per-point, per-leg, per-staypoint, per-visit and per-user-day features:
  step dt/distance/speed/heading/turn angle, leg kinematics and straightness,
  staypoint dwell + radius + residual stats, anchor role (home/work/other),
  visit arrival/departure hour bins, daily motif class, quality flags
  (coordinate range, duplicate timestamps, altitude sentinel −777).
- **Public class bins** (all fixed in the spec; edges chosen from measured
  distributions, then frozen as public):
  - **mode classes** (collapse of the 11 raw labels): `walk` (incl. run),
    `bike`, `bus`, `car` (car+taxi+motorcycle), `rail` (train+subway),
    `other` (airplane+boat);
  - **dwell classes**: < 1 h, 1–3 h, 3–8 h, 8–14 h, ≥ 14 h;
  - **departure-hour bins**: night ≤ 05, morning ≤ 11, afternoon ≤ 16,
    evening ≥ 17;
  - **trip-distance classes** (anchor-to-anchor path km): < 1, 1–3, 3–10,
    ≥ 10;
  - **speed classes** (m/s): < 0.5 (still), 0.5–2 (walk), 2–8 (city), 8–20
    (fast), ≥ 20 (very fast);
  - **staypoints-per-day bins**: 1, 2, 3, 4–5, 6+;
  - **turn-angle bins** (8 bins over (−180°, 180°], 45° wide).
- **Mobility motifs** with public thresholds (SQL and Lean mirrors):
  - **staypoint** — ≥ 1,200 s dwell within a 200 m radius (Li et al.
    parameters, fixed public);
  - **anchors** — 300 m grid clusters of staypoints; `home` = max night
    dwell (22–06), `work` = max workday dwell (09–17, weekdays, ≠ home);
  - **commute** — H→W and later W→H within one user-day (the dominant
    routine);
  - **errand loop** — H→O→H with dwell at O < 3 h;
  - **tour** — ≥ 3 distinct non-home anchors in one day-chain, consecutive
    visit gaps ≤ 2 h (the planted round-trip pattern);
  - **revisit beacon** — same non-home anchor visited on ≥ 5 distinct days
    within a 14-day window (periodic re-contact analog);
  - **long dwell** — ≥ 14 h at one staypoint (home-base signal).
- **Daily-routine motif classes** (per user-day, mutually exclusive,
  priority-ordered): `home_day`, `commute`, `errand_loop`, `tour`,
  `irregular` — the backbone of generation.
- **Sequence analysis**: the 3×3 anchor-role transition matrix (H/W/O) with
  lift, broken at observation gaps > 6 h and at day boundaries (the "breaks
  split transitions" lemma pattern); within-trajectory mode-leg chains.
- **GPS-noise moments**: per mode-class and at staypoints, residuals against
  the staypoint centroid / smoothed path — sd, MAD, p95, lag-1
  autocorrelation (category-2 release; reproduced by the generator's AR(1)
  noise).
- χ²/Cramér's V association tables on full contingency grids; Lean
  recomputes χ² exactly over ℚ on exported observations.
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
  points **in the source TSV format** (`user_id\tsequence_uuid\t
  trajectory_id\tdatetime\tlatitude\tlongitude\taltitude_ft\tlabel_start\t
  label_end\tmode`) from the distilled genspec alone, in a neutral local
  frame, followed by the Python `manipulate` step that places users on Tokyo
  geography; validated by re-ingestion through the production pipeline, with
  kernel-checked certificates. **The only artifacts the generator consumes
  are Lean sources and the released JSON — no compiled binaries, no Python
  models, no raw data.**

## Processing Architecture (planned)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | DuckDB `read_csv` in dbt staging (`stg_points`) | Canonical GPS points with stable `obs_id`, content-hash `obs_key`, typed timestamp/coords, quality flags (range, dup-ts, altitude sentinel) |
| Kinematic features | dbt SQL (`feat_steps`, `feat_legs`) | Per-point dt/dist/speed/heading/turn angle; per-mode-leg kinematics, straightness, speed/turn stats |
| Staypoints & anchors | dbt Python node (`feat_staypoints`) + SQL (`feat_anchors`, `feat_visits`, `feat_user_days`) | Li-et-al staypoints **plus implied inter-trajectory stays** (this labelled subset records movement, not dwells — the gap between one trajectory's end and the next one's start at the same place IS the dwell); per-user anchor clusters with home/work roles; anchor-visit sequences; user-day summaries + daily motif class |
| Noise estimation | dbt SQL (`feat_noise`) | Staypoint-interior residuals vs centroid; per-mode-class moments (sd, MAD, p95, lag-1 autocorr) |
| Route shape | dbt SQL (`feat_route_points/segments/legs`, `mart_route_shape`, `mart_seg_hist`, `mart_junction_hist`) | Legs resampled to 25 m; junction turns (≥ 30°), junction-to-junction segments, per-leg circuity and per-mode junction density — the road-likeness vocabulary the generator renders from (category-2 releases) |
| Mini-model fitting | PyTorch (`python/train_minimodels.py`), CPU-deterministic | NextAnchorGRU (next role from visit history; gate: return-to-previous), DwellClassModel (gate: marginal per role), ModeFromKinematics (gate: speed-rule baseline), UserDayAE (anomaly; gate: mean-recon), RouteShapeGRU (next route token — segment × junction class; gate: marginal), LegShapeAE (per-leg shape anomaly); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_geo_scores`) | Per-visit/per-day predictions with confidence and model version; never retrains on rebuild |
| marts | dbt SQL | `mart_user_stats`, `mart_motif_daily`, `mart_motif_revisit`, `mart_motif_tours`, `mart_staypoint_stats`, `mart_noise_moments`, `mart_dp_*` input tables |
| Association analysis | dbt SQL (`mart_assoc_cells`, `mart_feature_assoc`) | χ²/Cramér's V between discretized features on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Sequence analysis | dbt SQL (`mart_seq_transitions`, `mart_seq_chains`) | 3×3 anchor-role transition matrix with lift, gap-aware; mode-leg chains |
| Formal specification | Lean 4 + Mathlib (`lean/EnclaveGeo/Motifs.lean`, `Params.lean`, `Motion.lean`) plus `lake exe checks` | Decidable motif/staypoint definitions, proof-status ledger, SQL↔Lean agreement on exported observations, motion-constant linking theorems |
| Private release | Lean 4 + SampCert discrete Laplace over clipped per-user contributions (`lake exe release`) | User-level ε-DP histograms over the public grids, proved mechanism theorem, exact-rational budget ledger enforced in Lean AND Python |
| Verified sampling | SampCert `UniformSample` + bucketing map, law proved (`sampleCellPMF_law` pattern) (`lake exe sample`) | Samples from released distributions for generation and GOF checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: daily-motif histogram, role-transition rows, dwell/departure/distance/mode/staypoint-count histograms, noise + kinematic moments, schema constraints, public window |
| Generation | `lake exe generate <genspec> <n_users> <out>` | Synthetic TSV points in a neutral local frame + `plan.json` ground truth (planted motifs) |
| Manipulation | `scripts/synth.py manipulate` (mandatory in the synth flows) | Users juxtaposed onto Tokyo public geography; Beijing-bbox exclusion asserted |
| Validation | re-ingest through production dbt + Lean checks; pytest per-column | Round-trip motif recall, GOF vs released weights, noise-moment tolerance, straightness/turn-angle checks, kernel-checked certificate |

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
| `make train` | the entire enclave flow: dbt build → minimodel training → ml scores → Lean build + checks + DP release + verified-sampler smoke → distill → policy-filtered export → **one source-only bundle** in `dist/bundle/` |
| `make synth` | synthetic TSV from the local working tree (Lean generator from the local genspec → Tokyo manipulate → round-trip validation) into `synth/` |
| `make synth_from_bundle` | build the generator **from the bundle's Lean sources** (kernel re-checks every theorem), distill the bundle's release, generate + manipulate into `synth_bundle/` — proves the bundle is self-contained |
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

- The raw TSV (`geolife_data.tsv`), user ids, raw coordinates, exact
  timestamps, trajectory ids, and any per-user sequence or history.
- Individual points/legs/staypoints/visits, per-user anchors (home and work
  LOCATIONS are the crown jewels), motif hits, user-day structures
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Per-user embeddings and predictions; training checkpoints and fitted
  parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required, exact-
fraction ε budget). The three exit categories from METHOD §6 apply:

1. **ε-DP histograms** over the public grids (daily-motif classes,
   anchor-role transitions, departure-hour bins, dwell classes, mode-class
   mix, trip-distance classes, staypoints-per-day bins), noised by a Lean
   executable using SampCert discrete Laplace, with the mechanism theorem
   proved and the ledger enforced in both Python and Lean.
2. **Suppression-thresholded aggregates** (no formal guarantee — declared as
   such): GPS-noise moments per mode class (sd, MAD, p95, lag-1 autocorr),
   per-mode speed/sampling-interval summaries, the turn-angle histogram over
   public 45° bins, straightness-ratio moments, and within-dwell-class
   moment summaries — all with min-user (≥ 5) and min-count suppression,
   released so the generator can render kinematics realistically.
3. **Common-knowledge metadata**: the raw-label→mode-class map, all class-bin
   edges, staypoint/anchor parameters (200 m / 1,200 s / 300 m), the
   observation calendar window (2007-04-12 → 2012-01-01), the TSV schema and
   id-shape conventions (zero-padded user ids, `yyyymmddhhmmss` trajectory
   ids, uuid-shaped sequence ids, altitude sentinel −777), per-mode physical
   speed caps (public physics), and the **Tokyo target frame** (public
   district centroids + bboxes used by `manipulate`). Anchor locations, raw
   coordinates and user ids are structurally excluded regardless of
   popularity. **No absolute source geography is ever released** — the
   Beijing bbox appears only as an exclusion constant in validation.

**Privacy unit (the critical design decision — METHOD §6 lessons applied):**

- One protected entity = **one user** (their entire multi-year trace is one
  record in SampCert's add/remove-one `Neighbour` relation). The protected
  population is the user base: membership, locations, routines, and timing.
- **n = 64 entities is the hard regime** — closer to the stock migration's
  n = 8 than Foursquare's n = 29,640. METHOD's SNR estimate
  `share × n × ε_cell` gives ≈ 1.3 for a 10%-share cell at ε_cell = 20/100 —
  small cells are noise-dominated by construction. Consequences, all
  deliberate: grids stay coarse (≤ 9 cells; 37 DP cells total), budgets
  concentrate on the behavioral backbone (motifs, transitions), dominant
  cells (share ≥ 30%) reach SNR ≥ 4, and **public structural priors rescue
  what noise destroys** (declared `derived_from_data: false`). Phase 6
  REQUIRES the pre-noise simulation (clipped counts vs raw distribution, SNR
  per cell) before the ledger is committed.
- **Bounded contributions** via proportional L1 allocation (the Foursquare
  mechanism, unchanged): each user's per-grid cell vector is allocated over
  a total budget `dp_clip_total` (planned: 40 user-days / 60 visits ≈ p50 of
  the heavy-tailed per-user volumes; largest-remainder rounding in SQL), so
  per-cell sensitivity equals the total and the proved mechanism applies
  unchanged. Uniform per-cell clipping flattens the mix (METHOD §6) — not
  used. The user coverage spread is extreme (trajectories per user: median
  20, max 734), so equal per-user budgets are essential.
- **Semantics**: equal per-user budgets estimate the per-user-average
  distribution, not the point/visit-weighted one — and here they genuinely
  differ. This is the RIGHT semantics: the release describes the typical
  user's daily behavior. Measure both, document the gap as a declared
  property, let the realism audit quantify it.
- **Anchors**: roles (H/W/O) are released only as transition/motif
  statistics; anchor LOCATIONS never leave in any form (the reference
  implementation's `mart_region_context` denial carries over). Synthetic
  users carry fresh ids (`1..n`), fresh trajectory/sequence ids matching the
  public shape conventions.
- **Coordinates**: raw coordinates never leave. The generator emits a
  neutral local frame (meters-as-degrees around origin 0,0 — declared);
  `manipulate` places each user's frame at a Tokyo district centroid +
  bounded jitter — declared `derived_from_data: false`. Validation asserts
  **zero points in the Beijing bbox** and all points inside the Tokyo metro
  bbox. Intra-city source geography (street grid, transit lines) is in
  `assumptions.fields_not_modelled`.
- **Cross-user structure** (co-location, shared venues): a protected
  relation, never released, declared out of scope for v1.

**Committed ε ledger** (exact fractions; enforced in Lean AND Python; refusal
path tested; numbers fixed after the pre-noise inspection of the clipped
counts at n = 64 — scale = clip/ε_cell must stay comparable to the dominant
cell masses, which forced small clips (10) and a budget concentrated on the
behavioral backbone):
daily-motif classes 5 × (30/100) = 150/100; anchor-role transitions 9 ×
(15/100) = 135/100; dwell classes 5 × (13/100) = 65/100; departure-hour bins
4 × (5/100) = 20/100; mode-class mix 6 × (5/100) = 30/100.
Total 400/100 = 4/1 — EXACTLY saturated: a new query must displace an old
one. `lake exe release` refuses exports over budget (exit 2). Day-of-week,
staypoints-per-day and trip-displacement classes are NOT in the ledger: at
n = 64 their budget would buy only noise, so weekday/weekend structure enters
through the public calendar + the motif grid, staypoints-per-day through a
declared motif-template prior, and the displacement mix through a category-2
suppression release (`mart_dist_mix`, declared non-DP) — as are the
noise/kinematic/turn-angle summaries.

Suppression thresholds alone carry no formal guarantee; use them only for
artifacts explicitly marked non-DP (the noise/speed/turn-angle summaries).
DP noise randomness comes from a non-replayable source (urandom-backed FFI);
DP outputs are excluded from bitwise reproducibility by design.

Require attestation and controlled key release, encrypted persistent storage,
restricted egress, and logs that redact user ids (`<user>`) and coordinates
(`<coord>`). CPU mini-models suffice at this scale; GPU training stays
outside the enclave threat model.

## Implementation Guidelines

### 0. Understand the data before modelling it — ✅ done (see `wiki/dataset-geolife.md`)

Measured profile: 5,440,384 points, 64 users, 4,464 trajectories (points per
trajectory: median 487, p95 4,430, max 56,785; trajectories per user: median
20, p99 618, max 734), window 2007-04-12 → 2011-12-31, 100% mode-labelled
(walk 29%, bus 24%, bike 17%, train 10%, car 9%, subway 5%, taxi 4%, rest
< 1%), median sampling interval 2 s (p95 6 s), median step 4.9 m, median
step speed 3.1 m/s (p95 28 m/s), 83.8% of points inside the Beijing bbox
(the rest: other Chinese cities + international trips), 1 invalid
coordinate row (lat 400), 159,651 duplicate (user, trajectory, timestamp)
groups, altitude sentinel −777 on 55,478 rows. Every threshold above was
picked from these measured distributions, not defaults.

### 1. Define Concepts Before Fitting Models

Version every threshold (staypoint 200 m / 1,200 s, anchor cell 300 m, night
22–06, workday 09–17, visit-gap break 6 h, tour 3 anchors / 2 h, revisit
beacon 5 days / 14 d, long dwell 14 h, mode-class map, all bin edges,
`dp_clip_total`) as dbt `vars:`, mirrored in `python/spec.py` and
`lean/EnclaveGeo/Params.lean`, locked by a sync test; bump `spec_version` on
any change. Thresholds as exact rationals/integers in Lean (dwell edges in
whole seconds: 3600, 10800, 28800, 50400). The raw-label→mode-class map is
part of the spec (a dbt seed, a Python dict, a Lean function — one sync test
over all three). Ingest needs a light repairer: drop out-of-range
coordinates (counted), dedupe exact duplicate timestamps (counted), keep the
−777 altitude sentinel as a declared convention.

### 2. Start with Deterministic Baselines

SQL/Python-node motif detection and empirical distributions first
(staypoints, anchors, daily motifs, transition counts, dwell/departure/
distance/mode histograms, noise moments). Mini-models are added on top and
gated: NextAnchorGRU vs **return-to-previous-role**, DwellClassModel vs
per-role marginals, ModeFromKinematics vs a speed-threshold rule, UserDayAE
vs mean-reconstruction. Gates are data-dependent — read adoption flags from
artifacts, not docs. Keep honest negatives.

### 3. Separate Training from Inference

`make train` produces content-addressed immutable artifacts
(`sha256(name, data, hparams, code)`) under `models_out/`,
registry-activated; the inference node `ml_geo_scores` materializes
prediction tables. Rebuilding a mart never retrains.

### 4. Preserve Uncertainty and Provenance

Stable `obs_id`, content-hash `obs_key`, data fingerprint (sha256 of the
TSV), model version and confidence on every prediction, quality flags,
`untrained` placeholder supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

Break visit sequences at observation gaps > 6 h and at day boundaries; never
let a recording gap create a false transition. Recording is bursty (users
log some days, not others): user-days with no points are unobserved, not
home-days. First/last staypoints of a recording session are censored (an
arrival before recording started is not a departure from anywhere). Flag,
don't drop.

### 6. Prevent Leakage

User-level 80/20 split (split on user id hash, fixed before fitting); with
64 users accept the variance and document it; scalers fit on train only;
weak labels derived from the public taxonomy, never from evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

```text
P(next role    | visit history, hour)   — NextAnchorGRU; and the empirical 3×3
                                          transition matrix with lift
P(dwell class  | role, arrival bin)     — DwellClassModel / empirical rows
P(mode class   | leg kinematics)        — ModeFromKinematics (χ² partner)
P(motif        | day type)              — empirical (χ² tested)
P(noise        | mode class)            — released moment summaries (AR(1))
```

### 9. Make Execution Reproducible

Data fingerprints, code versions, seeds, hyperparameters, split definitions
in `models_out/<v>/meta.json`; immutable artifact paths. DP outputs excluded
by design. SampCert pinned (`a2c53bb`); its toolchain dictates the
Lean/Mathlib versions; the toolchain and package store live in
`LEAN_HOME=/opt/data/lean` and are reused, never re-downloaded per project.

### 10. Keep Formal Claims Precise

Distinguish: proved in Lean (mechanism DP, sampler law, motif/staypoint
decidability, generator validity-by-construction, motion linking theorems) /
checked on this snapshot (`lake exe checks` SQL↔Lean agreement) /
statistical estimates (model metrics, noise moments) / assumed (entropy
source, I/O wrappers, and the SQL/Python computing the clipped counts — DP
holds for whatever counts are fed in; their meaning is not proved). State
trust boundary 2 explicitly.

### 11. Use Verified Randomness

DP noise and all generation-time draws only through SampCert-backed Lean
executables (`lake exe release`, `lake exe sample`, `lake exe generate`).
Never ad-hoc floating-point noise. The `manipulate` step is deterministic
given its inputs (placement draws also come from the exported uniform pool).

## Synthetic generation contract (planned)

- `python/distill.py` reads ONLY `dp_release.json` + released artifacts
  (post-processing theorem; `source.dp_release_sha256` recorded). Teachers
  map to spec constructs: NextAnchorGRU → 3×3 DP transition rows;
  DwellClassModel → DP dwell histogram (+ suppressed within-class moments);
  daily structure → DP motif + staypoints-per-day histograms; kinematics →
  suppressed per-mode speed/turn/noise moments. Public structural priors
  (e.g. commute-hour shape, home-return evening boost) are declared
  `derived_from_data: false`.
- `lake exe generate <genspec> <n_users> <out>` renders TSV rows in a
  neutral local frame: per user, place a home anchor at the origin and
  work/other anchors by released trip-distance classes with non-grid
  bearings; per user-day, draw a daily motif class, instantiate the visit
  chain via the verified sampler over DP transition rows, draw departure
  bins and dwell classes; render each trip as a **road-like route polyline**
  — junction-to-junction segment lengths from the released segment
  histogram scaled by the mode's released junction density, junction angles
  from the released junction-angle histogram (in-class jitter: real turns,
  never on the 90° lattice — `jaRange_is_junction`), left/right from the
  released share with the cumulative deviation clamped to ±75° (forward
  progress), path budget from the released per-mode circuity, affine-fitted
  onto the anchor pair — plus **AR(1) GPS noise matched to the released
  moments** (moving noise attenuated 8×, a declared prior; stay wobble
  carries the full moments and is bounded so detection re-fires,
  `stayDrift_box_fits` pattern). **Validity by construction**:
  `mkPoint`/`mkDay` smart constructors with theorems —
  per-trajectory timestamps strictly increasing, all days inside the public
  window, speeds within mode caps, dwell ≥ staypoint threshold at planted
  staypoints (`mkDay_valid`, `times_sorted` via `List.Pairwise`).
- `scripts/synth.py manipulate` (mandatory in `make synth` /
  `synth_from_bundle`): map each user's local frame onto a Tokyo district
  centroid (public list) + bounded jitter, cos-corrected; emit final TSV.
  Validation asserts 0 points in the Beijing bbox and all points in the
  Tokyo metro bbox.
- Plant the rarest motif (a tour: ≥ 3 distinct non-home anchors, gaps ≤ 2 h)
  plus one revisit beacon in `plan.json`; round-trip re-ingestion through
  the production dbt pipeline must recall both at 1.0 (staypoint detection
  re-fires on noisy synthetic points — the real test) and `lake exe checks`
  must re-verify (exit 0 / 2 soft, 1 hard).
- Per-column tests: every output column matches its declared distribution
  (χ² GOF vs released weights), exhibits its declared property (noise sd
  within tolerance, turn-angle smoothness, no straight runs, Beijing
  exclusion), or satisfies a structural invariant; gaps declared in
  `meta.json assumptions.fields_not_modelled` (initially: street/transit
  geometry, altitude dynamics, label-window sub-structure, cross-user
  relations, within-bin dwell shapes beyond released moments).
- The generator's ONLY inputs are the genspec JSON and the Lean sources —
  synthesis works identically from the working tree (`make synth`) and from
  the source-only bundle (`make synth_from_bundle`).

## Acceptance checklist (status as built, 2026-11-02)

| Area | Criterion | Status |
| --- | --- | --- |
| Data profiling | Measured profile in wiki; thresholds from measurements | ✅ `wiki/dataset-geolife.md` (5.44M points, 64 users; implied-stay layer added after profiling showed the subset records movement, not dwells) |
| Spec sync | dbt vars = spec.py = Params.lean (+ mode map ×3), test green | ✅ all-integer spec (milli-m/s speeds, centidegree bboxes); 11-row mode map ×3; sync tests green |
| dbt pipeline | staging/features/marts/ml build + schema tests green | ✅ 29 models + 91 tests green; 574,708 dup rows + 1 invalid coord repaired and counted |
| Minimodels | Each gated vs frozen baseline; content-addressed; honest negatives kept | ✅ all four adopted: NextRoleGRU 0.697 vs return-to-previous 0.672 (majority 0.699 noted), DwellClassGRU 0.564 vs 0.480, ModeLegMLP 0.801 vs speed-rule 0.677, UserProfileAE scoring-only |
| Lean mirrors | `lake exe checks` SQL↔Lean agreement on exported observations (motifs, χ², transitions) | ✅ 16,975/16,975 agree (2,582 daily motifs re-decided, 1,955 revisit anchors, 10,000 class cells, 432 visit runs, 6 exact-ℚ χ²); 30 claims proved + 3 assumed; no sorry/native_decide |
| Private release | Mechanism theorem on SampCert; exact-rational ledger 400/100 enforced both sides; over-budget refusal tested; pre-noise SNR reviewed | ✅ spent exactly 400/100 = 4/1 (motif 150, transitions 135, dwell 65, departure 20, mode 30; clips 10); Lean exit-2 refusal + Python Fraction cross-check tested; staypoints/day + displacement dropped to priors/category-2 after the n = 64 SNR review |
| Verified sampling | Sampler law proved; executable draws pass χ² GOF | ✅ `sampleCellPMF_law`; catsample + samples.json conformance green |
| Distillation | Weight-by-weight provenance test green; fidelity under a pre-declared criterion | ✅ provenance cell-exact (declared priors: motif floor, home-return boost); fidelity **honest negative**: first-order role student 0.39 < return-to-previous 0.65 (teacher 0.70) — the generator compensates via motif templates, declared |
| Generation | `lake exe generate` renders schema-valid TSV; planted recall 1.0 after re-ingestion; noise in tolerance; no straight lines / 90° atoms; kernel-checked certificate | ✅ validate PASS (exit 0) at n = 64: recall 1.0 (commuter + tour + revisit beacon), mode-mix TV 0.004, motif-mix TV 0.30, collinear 0.0002, ±90° atom 0.0014, noise sd 30.3 m vs released 23.2 m (in band), 0 Lean disagreements on re-ingested data; certificate PROVED (`by decide`, kernel-only) |
| Manipulation | Tokyo juxtaposition applied; 0 points in Beijing bbox; all in Tokyo bbox | ✅ 308,130/308,130 points in the Tokyo box, 0 in the Beijing exclusion box (hard-asserted in manipulate AND validate AND pytest) |
| Simplified ops | the six make targets + `make help`; `LEAN_HOME=/opt/data/lean` reused; single source-only bundle; `synth_from_bundle` green | ✅ six flows wired + help; shared store reused; one 220K source-only bundle; pytest 110 passed / 0 skipped; both notebooks export headless with 0 tracebacks |
| Wiki discipline | Pages + append-only log updated on every ingest/incident | ✅ `wiki/geo-pipeline.md` + dataset/migration pages + log entries (implied-stay, negative-gap, ledger-rebalance incidents) |

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible mobility-trajectory encoder plus
a verified synthetic-trajectory generator with an explicit output contract.

- Enclave execution protects processing within the selected platform's threat
  model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit (one user, add/remove their whole trace) and
  assumptions. With n = 64 users, utility is the known hard constraint — it
  is measured (pre-noise SNR simulation, released-vs-clipped comparison) and
  compensated with declared public priors, never patched with private data.
- Statistical evaluation establishes empirical usefulness.

None of these alone guarantees realism of future synthetic trajectories.
Released non-DP artifacts (suppressed kinematic/noise moments) and
evaluation metrics remain outside the DP guarantee. Street geometry,
altitude dynamics and cross-user relations are declared gaps until a ledger
entry covers them.

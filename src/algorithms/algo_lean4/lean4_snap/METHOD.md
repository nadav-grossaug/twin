# METHOD — replicating the enclave → formal spec → synthetic data pipeline

This document describes the goal, plan, techniques and implementation notes of
the architecture built in this repository, written so that Claude Code (or any
engineer) can replicate the flow on an **entirely different dataset**. The
worked example here is network flow telemetry (Cisco Joy pcap exports), but
nothing in the method depends on packets: it applies to any sensitive,
event-like or record-like dataset that can be explored and modelled inside a
compute enclave — medical encounters, location traces, financial transactions,
industrial sensor logs, call records.

Read this together with `goal.md` (the original task framing), `wiki/`
(especially `architecture.md`, `dp-release.md`, `distillation.md`,
`practices.md` — the incident log that earned each rule), `HOWTO_run.md` and
the `Makefile`.

---

## 1. Goal

Build a pipeline where:

1. **Raw sensitive data never leaves an enclave.** All exploration, feature
   engineering, statistics and ML happen inside; the boundary is a default-deny
   release gate.
2. **Everything that crosses the boundary is one of three declared release
   categories** (§6): ε-DP histograms with a machine-checked mechanism,
   suppression-thresholded aggregates, or common-knowledge metadata — each
   listed in a versioned policy file, never ad-hoc.
3. **The formal layer is Lean 4 + Mathlib + SampCert**: domain predicates have
   decidable Lean mirrors re-checked on exported data; the DP mechanism's
   theorem is proved; random draws go through a verified sampler whose law
   (`P(i) = wᵢ/Σw`) is a theorem.
4. **PyTorch minimodels are distilled away**: private learning happens in the
   enclave, then each *generative* model is replaced by a RESTRICTED typed
   statistical specification (histograms, conditional rows, constraints — not
   code) built purely from released values.
5. **Synthetic data is generated entirely by a standalone Lean executable**
   from that spec — no PyTorch, no Python sampling — with record validity by
   construction, kernel-checked certificates, and a re-ingestion round-trip
   through the production pipeline as validation.

The end state: a consumer outside the enclave gets realistic-ish synthetic
data plus proofs/certificates/ledgers, and can audit exactly which facts about
real data informed it — weight by weight.

## 2. Architecture (five layers, two trust boundaries)

```
┌─ ENCLAVE ──────────────────────────────────────────────────────────┐
│ L1 ingest+repair → L2 dbt/DuckDB features·motifs·stats →           │
│ L3 PyTorch minimodels (scoring + teachers) → L4 Lean formal layer  │
│    (decidable mirrors, DP mechanism, verified sampler)             │
│         │  release gate: policy + attestation + ε ledger           │
└─────────┼──────────────────────────────────────────────────────────┘
          ▼  dp_release.json · topk artifacts · suppressed stats · lean claims
   L5 distiller (python/distill.py, post-processing only)
          ▼  genspec.json (typed, restricted, provenance-hashed)
   Lean generator (lake exe generate) → synthetic records + plan
          ▼
   validate: re-ingest through PRODUCTION dbt+Lean · GOF tests ·
   kernel-checked certificate · realism audit (enclave-side judges)
```

Trust boundary 1: enclave ↔ world (crossed only by the release gate).
Trust boundary 2: proved mechanism ↔ assumed I/O and SQL ("Lean proves
properties of the mechanism and the predicates; it does NOT prove the SQL that
computed its inputs" — state this explicitly, as `goal.md` guideline 10 does).

## 3. Stack

- **DuckDB + dbt-duckdb** — warehouse and transformation layer. One file
  database in `artifacts/`, models in `dbt/models/{staging,features,marts,ml}`.
- **PyTorch** — minimodels; MPS/CUDA when available, `FLOW_DEVICE=cpu` for a
  deterministic mode.
- **Lean 4 + Mathlib + AWS SampCert** (pin SampCert and its toolchain; it
  dictates the Lean version) — proofs, DP mechanism, verified sampler,
  generator executables. `lake exe cache get` for the Mathlib cache.
- **marimo** — EDA/visualization notebooks, validated headless via
  `marimo export html`.
- **pytest** — one test suite covering spec sync, SQL↔Lean agreement,
  release gate, DP conformance, distillation provenance, per-column synth
  quality.
- **Makefile** — the only entry point users need: `all`, `dbt`, `train`, `ml`,
  `lean`, `distill`, `synth`, `certify`, `test`, `manifest`, `clean*`.

## 4. Replication plan (phases, in order)

Phase order matters: each phase's tests lock the previous one before the next
builds on it.

0. **Understand the data before modelling it.** Profile the raw input first
   (counts, null rates, value distributions, per-entity concentration). Every
   threshold in this repo was picked from a measured distribution, not a
   default. Write the findings into the wiki as you go.
1. **Ingest + repair.** Parse raw records to a staging table. If the source
   serializer has known bugs, write a *verified repairer* (`make repair-data`
   style): touch only records that fail parsing, verify the re-parse, refuse
   to guess at anything else. Repair, don't drop — the broken records are
   often exactly the interesting ones. Quality problems become flags
   (`qf_*`, `is_*`) and a `q_data_integrity` model, never silent fixes.
2. **Concept spec, versioned, mirrored three ways.** Put every public
   threshold/constant in `dbt_project.yml vars:`; mirror them in
   `python/spec.py` (reader) and a Lean `Params.lean`; lock all three with a
   sync test that regex-parses the Lean source. Bump `spec_version` on any
   change. This is what lets SQL, Python and proofs share one number.
3. **dbt layers.** `staging` (typed, flagged), `features` (one row per
   record/entity with derived features), `marts` (statistics, motif/pattern
   detectors, association/χ² tables, sequence/session models, and the
   `mart_dp_*` DP-input tables), `ml` (model scores materialized back as a
   table). Add dbt schema tests for structural invariants.
4. **Minimodels.** Small PyTorch models per modelling question (§7). Each is
   content-addressed (`sha256(name, data, hparams, code)`), immutable,
   explicitly activated via a registry, and gated by a frozen non-ML baseline.
   Keep honest negatives. Scoring models will stay enclave-side forever;
   *generative* models (sequence, structure, name models) are teachers that
   phase 8 replaces.
5. **Lean formal layer.** For every SQL predicate that matters (motifs,
   well-formedness, session timeliness), write a decidable Lean mirror and an
   executable (`lake exe checks`) that re-decides it on exported data; export
   agreement counts. Model data for the proof, not the parser (e.g. names as
   `List String` labels so RFC bounds are arithmetic). Exact rationals for
   every comparison that matters (ε, thresholds like CV as num/den pairs).
6. **DP release** (§6 — read the calibration lessons before writing SQL).
7. **Verified sampling.** A sampler over released histograms built on
   SampCert's `UniformSample` with a proved law (`sampleCellPMF_law`), plus a
   χ²-GOF conformance test on its executable draws.
8. **Distillation** (§8): released facts → `genspec.json`, with a
   weight-by-weight provenance test and an enclave-only fidelity report.
9. **Lean generator** (§9): `lake exe generate <genspec> <n> <out>` renders
   synthetic records entirely in Lean; Python's generate becomes a thin
   adapter that invokes it and records provenance metadata.
10. **Validation loop** (§10): re-ingestion, per-column tests, certificates,
    realism audit, run manifests, notebooks, wiki.

## 5. The enclave side: features, stats, minimodels

- **Feature taxonomy is public by design.** Choose a fixed, data-independent
  classification of records (here: 7–17 service classes from protocol+port)
  early — it doubles as the DP cell grid later. Document that it is public.
- **Motifs** are domain patterns with *public thresholds* (here: beacons =
  periodic re-contact with CV ≤ 0.6 and n ≥ 8; fan-out; A→B→A returns;
  resolve→connect chains; sessions = gap-broken sequences). For another
  dataset: pick the 3–6 patterns a domain analyst would hunt for, define them
  with thresholds in the spec, implement in SQL *and* Lean.
- **Statistics**: per-class aggregates, χ²/Cramérs-V association tables
  between discretized features (export the contingency cells so Lean can
  recompute χ² exactly over ℚ), sequence transition matrices with lift.
- **Minimodels** (~5–15, each answering one question): an autoencoder for
  anomaly scores; classifiers for class-from-content (gate: majority
  baseline); sequence models for next-event (gate: repeat-last baseline);
  structure models (record-shape GRU; gate: marginal/unigram); a char-LM over
  names/strings (gate: unigram bits-per-char). Rules learned here:
  - **Gates are data-dependent**: a model not adopted on a small capture may
    be adopted on a larger one. Read flags from artifacts, not docs.
  - **Dedup + weight instead of brute scale**: train the char-LM on distinct
    strings weighted by frequency (same expected gradient, ~90× fewer steps);
    keep the frozen record-level eval so the gate stays meaningful.
  - Keep a `train_loop(..., sample_weights=None)` seam; retrains are no-ops
    unless inputs changed (content addressing makes this free).

## 6. The release gate (the heart of the method)

**Three exit categories, all declared in `config/release_policy.yaml`
(default deny, attestation required, exact-fraction ε budget):**

1. **ε-DP histograms** over PUBLIC cell grids, noised by a Lean executable
   using SampCert discrete Laplace (exact integer sampling — immune to
   float-Laplace attacks), with the mechanism theorem proved and an
   exact-rational ledger enforced in BOTH Python and Lean (the executable
   refuses to write output over budget, exit 2). Cells must be fixed
   taxonomies or RANK labels (`stack_1..5` with the rank→identity map kept
   inside); no raw string/hash ever reaches the mechanism.
2. **Suppression-thresholded aggregates** (no formal guarantee — say so):
   per-class stats with min-count suppression.
3. **Common-knowledge metadata**: top-k popular values (names, hostnames)
   with suppression thresholds (≥k entities, ≥n occurrences) and
   *structural exclusions* for value classes that are private by construction
   (here: `*.arpa` names encode internal IPs; single-label names are local
   hostnames — excluded regardless of popularity). Outside the ε ledger,
   declared as such.

**Calibration lessons (earned the hard way — read before choosing the privacy
unit and clipping):**

- **The privacy unit is a design decision the data can break.** "One source
  address" was fine on a small capture; on a large one, 2,433 external
  one-flow addresses outvoted the 15 monitored hosts and the released mix came
  out *inverted* (the dominant real class released as 0, noise-only classes
  released large). Define the protected population as the entities you
  actually monitor, and attribute each record to exactly one protected entity
  (here: the flow's internal endpoint). For your dataset: patients not
  providers, users not servers, households not meters — whichever side is the
  subject of protection.
- **Uniform per-cell clipping cannot express a mix** when entities are heavy:
  every major cell saturates at `n_entities × clip` and the histogram goes
  flat. Use **proportional L1 contribution**: each entity's cell vector is its
  own distribution allocated over a total budget (`dp_clip_total`,
  largest-remainder rounding in SQL). The vector sums to ≤ C so per-cell
  sensitivity is still C and the proved mechanism applies unchanged.
- **Know what the histogram estimates.** Equal per-entity budgets estimate the
  *per-entity average* distribution, not the record-weighted one. If a few
  infrastructure entities dominate the record counts (here: the DNS server),
  these differ. Measure both; if record-weighting matters, simulate
  flow-weighted budgets (`min(n/K, C)`) FIRST — here it collapsed SNR to ~1
  and was rejected. Document the chosen semantics as a declared property.
- **Budget arithmetic**: per-cell ε composes sequentially, so SNR on a k-cell
  grid with n entities ≈ `share × n × ε_total / k` — independent of C. Fewer,
  coarser public cells beat many fine ones. Spend the ledger where calibration
  matters (here: service mix 7×(2/5), shape grids 1/25, transitions 1/100,
  total 389/100 ≤ 4/1). Keep the ledger *nearly* saturated: a new query must
  visibly displace an old one at review time.
- **Audit before raising the budget, and fix the clip before the epsilon.**
  When a deliberate decision trades privacy for fidelity, measure first: per
  grid, the relative error vs the exact table and the number of cells below
  2× the Laplace scale (here the mixing grid: 36 of 45 cells, 34% error — the
  worst; the motif grid was noisier still but the generator never samples it,
  so it got nothing); and measure what the downstream structure actually
  needs (edge coverage by attribute category showed two huge groups, not the
  categories first guessed, carried 29 points of the 68→97% coverage). A
  per-entity clip that is too small is a BIAS no amount of epsilon removes (clip 2
  shrank memberships ~4× and halved the coverage). State the resulting bound
  plainly (ε ≈ 13.7 is no practical guarantee; the row-level unit and the node-DP
  gap weaken it further) and record that earlier draws of the same data compose.
- DP draws are **intentionally non-reproducible** (urandom-backed FFI).
  Anything comparing spec↔release must be hash-keyed to the release file and
  re-derived after each fresh release (see §8 provenance skip, and run
  `distill` AFTER `lean` in the `all` target).

## 7. Verified sampling

One Lean function (`sampleCellPMF` here) draws a cell index from a released
histogram via SampCert's uniform sampler and a bucketing map; its law
`P(i) = wᵢ/Σw` is a theorem. The generator and any sampling executable use
only this function. A pytest conformance test draws N samples via the compiled
executable and χ²-tests them against the released weights. Zero-sum rows fall
back to uniform (and that fallback is part of the proved law's statement).

## 8. Distillation: models → typed spec

`python/distill.py` (runs enclave-side, but reads ONLY released artifacts):

- Input: `dp_release.json` (+ aux histograms) and top-k artifacts. NEVER raw
  data, weights or embeddings — the spec inherits the release's privacy
  accounting by the post-processing theorem. Record
  `source.dp_release_sha256` in the spec.
- Output: `lean/out/genspec.json` — a RESTRICTED typed vocabulary, not code:
  weighted cells (`{name, w}`), conditional rows (`{given, cells}`), top-k
  name lists, schema constraints, and pointers to fixed public temporal
  constants. The generative teachers map to spec constructs:
  - sequence model → semi-Markov transition rows (DP 7×7) over the public
    class grid;
  - structure model → rank-mix over a FIXED canonical skeleton;
  - name model → shape histograms + common-knowledge top-k with a synthetic
    fallback (`*.test`-style gibberish matching released shape cells);
  - (graph datasets, lean4_snap) structure teacher (GraphSAGE) → the DP
    degree histogram; community embeddings → DP community-size histogram +
    rank-labeled mixing rows; link-prediction teacher → the DP
    wedge-closure conditional — and when the neural teacher LOSES to a
    classical baseline (VGAE 0.945 vs common-neighbor 0.990), distill the
    SQL-empirical grid anyway: the teacher's role was certifying WHICH
    conditional matters, not supplying weights; feature encoder →
    category/richness marginals + suppression-thresholded conditional rows.
- **Adopted ≠ distilled.** Adoption is a modelling verdict (the model beat
  its frozen baseline); distillation is a privacy event (a RELEASED
  construct takes over its generative role). Scoring models stay
  enclave-side forever. An adopted teacher with no ledger entry yet must
  surface as a NAMED ledger candidate plus a declared
  `fields_not_modelled` gap — never as silent fabrication. (Worked example:
  snap's affiliation models spent one release cycle in exactly that state —
  adopted, key-group artifacts denied, cohesion table recorded as the next
  ledger-review candidate — until the 2026-11-03 review DISPLACED budget
  from two informational grids (triangles 5→2/100, motifs 3→2/100) to fund
  a DP member-weighted key-group size histogram plus a suppression-released
  cohesion table; the key_cohesion_mlp teacher certified the (category ×
  size)→density conditional (MAE 0.058 vs mean-baseline 0.263 on the 19 cells it
  was trained on) and the released table became the distilled student (MAE
  0.003). The displacement
  path is the method: gaps close through the ledger, never around it. A
  SECOND review the same day measured that the displacement was too small
  (shared-key edge coverage 34% vs 68% real because clip 2 shrank memberships
  ~4×; locale/gender, 76%/55% of edges, were not modelled; the mixing grid had
  36 of 45 cells below 2× the noise scale) and, on an explicit user decision to
  favour fidelity over privacy, raised the cap 4/1 → 14/1. That is the method's
  other lever: audit noise per grid FIRST (relative error, cells below 2× the
  noise scale), spend the extra ε only where that audit says it buys fidelity,
  and state the resulting bound plainly instead of calling it private.)
- **Public structural priors rescue what DP noise destroys** — e.g. a
  self-transition diagonal boost equal to `max(1, row sum)`, declared
  `derived_from_data: false`. Never patch noise with private data.
- **Fidelity report** (enclave-only): distilled student vs the PyTorch teacher
  vs a frozen baseline on the same held-out windows, plus per-row TV between
  released and exact tables. Declare an acceptance criterion up front
  (here: student ≥ repeat-last − 0.02) and keep the report honest.
- **Provenance test**: every weight in the spec equals a released value (or
  released + declared public prior) — asserted cell-by-cell in pytest; skips
  with a clear message when the spec predates the current release.

## 9. The Lean generator

`lean/GenSpec.lean` + `lean/Generate.lean` (a `lean_exe` linking SampCert's
FFI):

- Parse the genspec JSON into typed structures (`Cell`, `Hist`, `CondHist`).
- All randomness through the verified sampler; Markov chains via
  fuel-bounded recursion (`genChain`) so termination is structural.
- **Validity by construction**: a `mkFlow`/`mkRecord` smart constructor whose
  output satisfies the schema predicate, with a theorem (`mkFlow_valid`); the
  emitted plan's temporal structure proved sorted/bounded
  (`stepTimes_sorted` via `List.Pairwise` — `List.Chain'` was not available in
  the pinned toolchain; keep proofs on whatever Mathlib the SampCert pin
  gives you).
- Render output records in the SOURCE format (here Joy JSONL, exact-decimal
  timestamps via `JsonNumber` mantissa/exponent) so the production pipeline
  ingests them unmodified. Note: Lean's `Json.compress` emits no space after
  colons — tests that string-match serialized JSON must use regexes.
- Also emit a `plan.json` (ground truth: planted patterns, per-entity session
  structure) for round-trip validation.
- Python's `generate` becomes an adapter: check spec + binary, invoke, write
  `meta.json` with full provenance (genspec sha, release sha, released-fact
  sources, and an `assumptions` dict with `fields_not_modelled` — declared
  gaps over silent fabrication, each tested by per-column tests).
- Seed-reproducibility is intentionally absent (consistent with
  non-reproducible DP draws) — document it as a caveat.
- **Every generator prior is declared AND measured.** A prior that is not in
  the genspec `priors` list (with `derived_from_data: false`) is a hidden
  assumption; one whose justification you wrote from memory is a bug waiting
  (a "p99 is well below 40" cap was false: mean 83, p99 623). Choose each
  knob from a dose-response (5–8 runs per setting, run-to-run noise measured
  first); expect it to be non-monotone and size-dependent (a member-window
  factor that is right for tiny groups overshoots mid-size ones).
- **Round discrete targets stochastically.** Deterministic rounding of a
  target count is biased on tiny units (a group of 2 has one possible pair,
  so any target ≥ 0.5 became a full pair); round up with probability equal
  to the fractional part, through the verified sampler.
- **Planted structures need a fallback search.** A strict construction (a
  broker grown with zero-common-neighbour partners) runs out on small dense
  graphs; fall back to the greedy minimum-violation choice and keep the
  decidable predicate as the final authority.
- **Declare the size window the tolerances were validated on** (public
  constants mirrored in the generator and the validator, sync-tested), warn on
  stderr outside it, and sweep the whole advertised range once — the sweep
  found a crash at the small end, tolerances unreachable below a few hundred
  nodes, and a decline above ten thousand.

## 10. Validation loop

- **Round-trip**: synthetic output re-ingested through the PRODUCTION dbt
  pipeline into a scratch database; the same marts/predicates must detect the
  planted patterns (beacon recall = 1.0 here) and the Lean checks must
  re-verify (`validate` exit codes: 0 pass / 2 soft tolerance / 1 hard
  schema or spec violation). Audits cut both ways — this found encoder bugs,
  not just generator bugs.
- **Per-column tests**: every output column either (a) matches its declared
  distribution (χ² GOF vs released weights or spec weights), (b) exhibits the
  declared statistical property, or (c) satisfies a structural invariant.
  Tests tolerate sampling (skip on zero-draw classes), never bugs.
- **Certificates**: synthetic structures kernel-checked as literals
  (`by decide` over the exported plan) — `certificate.json` records
  `proved: true, kernel_only: true` and the axiom list.
- **Realism audit**: the enclave's own scoring models judge the synthetic data
  (share above real anomaly percentiles, TV on public bins). It is a graded
  audit, not pass/fail — the generator targets the spec contract and the
  release, not realism; report the gap honestly.
- **Run manifests** fingerprint everything (data sha, spec, code, model
  versions, table hashes); a reproduce script rebuilds into scratch and
  compares within tolerances.
- **Judge stochastic properties over replicate runs, not one sample.** A
  generator is a random process; one scored sample makes the verdict luck
  (closure sat on the band edge and passed or failed by draw). A `replicates`
  command generates N graphs and applies a public rule to the distribution
  (here: ≥ 85% of runs in band AND mean ≥ 0.02 above the band edge AND mean
  group-density error ≤ 0.08). Per-run checks stay, but properties that a
  healthy run trips occasionally become warnings.
- **A validator must say what it does not judge.** Never `continue` past a
  cell that is too sparse to test — the sparse cells were exactly the wrong
  ones. Record unjudged cells in the verdict; widen the tolerance with the
  sample behind the cell (expected distance ~ √((k−1)/(2πn)), three standard
  errors when many categories are checked per run); scale a per-cell tolerance
  with the target (a flat 0.25 passed a 0.22 miss on a 0.24 target).
- **Measure a check's false-alarm rate on healthy runs** (a flat tolerance
  flagged 4 of 12 once nine categories were judged) and **prove every check by
  mutation**: feed it the old defect (a unit test with the pre-fix numbers,
  plus an end-to-end run with the defect reintroduced) and require a failure.
- **Notebooks** (marimo): one EDA notebook for the enclave data, one
  synth-vs-real comparison notebook (per-column TV scorecard with each
  column's fill source; declared-gap framing). Validate headless
  (`marimo export html` + grep for tracebacks/oversized cells). Charts carry
  aggregates, never per-row dataframes; keep a bounded
  `alt.data_transformers.enable("default", max_rows=20_000)`.

## 11. Testing and operations discipline

- Suite shape (~105–110 tests here): spec sync (3 mirrors), SQL↔Lean
  recompute agreement, SampCert conformance (GOF), release gate (attestation,
  egress, policy, exact-ε ledger, default-deny list), distillation
  (provenance/shape/fidelity), synth (meta assumptions, beacon recall,
  exit-code mapping, corruption → hard fail), per-column synth, minimodel
  gates. Every fixture that reads a produced artifact must `pytest.skip` with
  a "run `make X`" message when it is absent.
- `make all` ordering matters: `dbt train ml lean distill test manifest` —
  distill AFTER lean so the genspec always matches the fresh (non-reproducible)
  release.
- **Builds fail loudly or not at all.** `lake` is only on PATH inside the
  Makefile recipes; a bare `lake build | grep error` printed nothing for
  `command not found` and a whole series of experiments measured the previous
  binary. Use a wrapper (`set -e`, print the tail, BUILD FAILED), confirm the
  binary's mtime moved, and treat identical results across supposedly different
  variants as a harness bug first.
- **A/B on enough runs.** Measure the metric's run-to-run sd, compare at ≥ 8
  runs on separately built before/after binaries (checked by content), and
  quote a delta only if it exceeds ~2 sd — a favourable 4-run delta ("edges
  match the real count exactly") was noise.
- **Lists defined in several places are sync-tested** (a category list lived
  in spec.py, two SQL marts and the distiller).
- One pipeline driver at a time: concurrent `make` runs race over the DB write
  lock, the GPU and generated dirs that `clean` deletes mid-flight.
- Keep an **LLM wiki** (`wiki/`: index + concept pages + append-only log,
  per Karpathy's LLM-wiki gist): current-state claims on pages, history in the
  log, updated on every ingest/incident. It is the project's long-term memory
  and the source for documents like this one.

## 12. Adapting to a new dataset — substitution map

| This repo (pcap flows) | Your dataset: choose equivalents |
|---|---|
| flow record (Joy JSONL line) | one event/record (visit, transaction, trace point) |
| internal host (protected unit) | the protected entity (patient, user, household) |
| service class (public taxonomy) | a fixed public record classification (5–20 classes) |
| motifs: beacon/fan-out/session | 3–6 domain patterns with public thresholds |
| TLS/DNS structural sub-records | any nested structure worth modelling (prescription lines, GPS sub-tracks) — shape histograms over PUBLIC cells, canonical skeletons for rendering |
| top-k qnames / SNI | common-knowledge popular values (drug names, merchant categories) with suppression + structural exclusions for identifying value classes |
| FlowShape.lean constants | fixed public temporal/shape constants (session counts, dwell weights) |
| beacon recall round-trip | plant your rarest pattern in the plan; require recall 1.0 after re-ingestion |

Keep unchanged: the five-layer architecture, three release categories, the
spec-mirror discipline, content-addressed gated minimodels, the
distill-to-typed-spec rule (never generated code), the Lean-only generator
with validity by construction, the validation loop, and every calibration
lesson in §6.

### Graph datasets (lessons from the SNAP Facebook migration, lean4_snap)

The map above assumes event/record data. A static GRAPH needs three extra
substitutions, proven in the fifth instantiation:

- **Privacy unit**: removing a person perturbs their NEIGHBORS' derived rows
  too (degrees, triangles). Full node-DP costs a sensitivity factor of
  (1 + degree cap) — ruinous. Claim **row-level DP on the derived per-node
  table**, state the gap as a first-class assumption (ledger claim
  `row_level_privacy_unit: assumed`), and keep edge/wedge-derived grids on
  proportional L1 clipping as usual. Do not let the weaker unit hide: it
  belongs in goal.md, the release policy header and the claims ledger.
- **No temporal layer**: a snapshot graph has no sequences. Verify the
  absence (a quality model asserting time-looking attributes are flags, not
  dates) and substitute the TRIADIC-CLOSURE process as the "sequence"
  conditional: a released P(closed | wedge-center degree class) grid both
  replaces the sequence-model teacher (here VGAE — an honest negative vs the
  common-neighbor baseline at transitivity 0.52) and drives the generator's
  clustering toward the released level. Compensate degree targets for
  closure-pass additions using RELEASED closure weights only
  (post-processing, declared).
- **Generator invariants move from rows to the edge set**: smart constructor
  `mkEdge` (canonical src < dst, rejects exactly self-loops, proved), a fast
  tree-set during generation, then an O(m) PROVED output check
  (`edgesCanonical_sound`: lex-sorted ⇒ duplicate-free ⇒ well-formed) that
  the executable runs on its own output before writing — validity by
  construction where cheap, validity by decidable self-check where speed
  matters. The same trick plants motifs: grow a planted broker with
  zero-common-neighbor edges (its triangle count is invariant) until the
  DECIDABLE motif predicate accepts, then record it in the plan.

- **Attribute key-groups are the "feature identity" substitute.** People
  sharing one anonymized attribute value form groups far denser than random
  pairs. Release (i) a DP histogram of group sizes by (category × public size
  bin), MEMBER-weighted (one event per membership, proportional-L1 clip) and
  divided by public bin midpoints in distillation, and (ii) a
  suppression-thresholded cohesion table (integer-percent density). Choose the
  categories by measured EDGE coverage, not intuition; expect the biggest
  groups (here locale/gender, ≈ 213 members, ego-network containers) to carry
  most of the coverage while the "same workplace" cliques carry cohesion; set
  the clip near the p85 of per-node memberships. Real groups are nested
  (86% of an education group lies inside one locale group); the generator here
  does not model that, and a naive attempt made it worse.

## 13. Acceptance checklist

- [ ] Raw data parses or is verifiably repaired; quality flags counted.
- [ ] Spec in dbt vars = spec.py = Params.lean (sync test green).
- [ ] Every minimodel beats (or honestly fails) a frozen baseline; artifacts
      content-addressed and registry-activated.
- [ ] Every exported aggregate appears in `release_policy.yaml` under one of
      three categories; ledger exact-fraction, ≤ budget, enforced in Lean
      AND Python; denied list covers all row-level relations.
- [ ] Released mix sanity-checked against the enclave truth (pre-noise clipped
      counts vs raw distribution) — §6 lessons applied.
- [ ] genspec provenance test green; fidelity report `accepted: true` under a
      pre-declared criterion.
- [ ] `lake exe generate` renders schema-valid records; validate exit 0/2;
      planted patterns recalled at 1.0; certificate kernel-checked.
- [ ] Full suite green; run manifest written; both notebooks export headless
      with no tracebacks; wiki pages + log updated.
- [ ] Stochastic properties are judged over replicate runs (public pass rule);
      validators report unjudged cells, have a measured false-alarm rate, and
      each check fails on a reintroduced defect.
- [ ] The generator was swept across its advertised size range; the validated
      window is a public constant, tested for sync and warned on outside.
- [ ] Every generator prior appears in the genspec `priors` list with a measured
      justification; any privacy-for-fidelity trade is stated with its actual
      bound and its composition with earlier releases.

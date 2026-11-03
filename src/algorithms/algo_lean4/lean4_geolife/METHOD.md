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
   default. Write the findings into the wiki as you go. **Verify units and
   conventions from the data itself, not from documentation** — especially
   the time zone: GeoLife's timestamps are UTC (trajectories and label
   windows split at 16:00:00 = Beijing midnight; the point-hour profile is
   empty 16–21 h) while our own dataset page claimed "local", and every
   time-of-day concept (home anchor, work window, departure bins, day
   boundaries) ran 8 h off until fixed. Convert ONCE in staging through a
   spec var (`source_utc_offset_s`); generators plan in the converted frame
   and write the source's convention back.
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
- DP draws are **intentionally non-reproducible** (urandom-backed FFI).
  Anything comparing spec↔release must be hash-keyed to the release file and
  re-derived after each fresh release (see §8 provenance skip, and run
  `distill` AFTER `lean` in the `all` target).
- **Small-n regime (GeoLife, n = 64 users)**: since SNR ≈ `share × n ×
  ε_cell` is independent of the clip, no clip choice rescues a small cohort —
  only fewer histograms with bigger per-cell ε do. Rank the candidate
  releases by how much generation needs them, drop the tail to category-2
  suppression or declared public priors (here: staypoints-per-day and the
  displacement mix), and concentrate the ledger on the behavioral backbone
  (here: motif 5×30/100, transitions 9×15/100). Keep clips near the
  per-entity median so the pre-noise mass stays interpretable. **Measure,
  don't assume**: after every release compare the noised counts with the
  true clipped counts inside the enclave (TV per histogram). At n = 64 even
  the concentrated 4/1 ledger left departure and mode as noise (ε_cell 0.05:
  noise scale 200 vs ~640 user-mass; a release said "morning = 0"), and the
  generator faithfully reproduced the noise. The options are to drop the
  histogram or to raise ε_cell; the GeoLife owner chose fidelity
  (ε_cell = 1, 29/1 per export, released-vs-true TV ≤ 0.08) — a weak formal
  guarantee that must be stated as such, never called "private".
- **Pure DP composes across exports — keep a lifetime ledger.** The
  per-export budget check is not a guarantee: re-releasing after a pipeline
  fix spends ε again on the same people. Keep an append-only ledger keyed by
  the source data's sha256, stored outside every `make clean` target and
  under version control; reconcile it against every export manifest before
  exporting (no export can go unrecorded); record the spend BEFORE writing
  files; never refund on deletion; refuse exports past a declared lifetime
  cap (`dp_lifetime_epsilon_max`). Raising the cap is an explicit owner
  decision; an export that provably never left the enclave may be voided
  with an audited reason.
- **Single-source every ε.** The policy file is the only place ε values
  live; the bridge that feeds the Lean release reads them (a hand-kept
  mirror once drifted), and tests assert agreement with the policy rather
  than pinning constants.
- **Reconstruct implied events before choosing DP inputs.** When the source
  records only activity bursts (trip-logging GPS, call records, sessions),
  the dwell/idle structure lives BETWEEN records: derive implied stays/idles
  (end of burst → start of next burst at the same place), censor them when
  the endpoints disagree, and only then define the per-entity contributions —
  otherwise the released behavior describes the logger, not the person.
  Watch for overlaps between detected and implied events: an overlap is not a
  transition (negative gaps must break sequences in SQL exactly as the Nat
  ordering does in Lean, or the crosscheck will catch it — ours did).

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
    fallback (`*.test`-style gibberish matching released shape cells).
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
- **Distill the CURRENT release, not the latest export.** The build computes
  a fresh release before anything is exported, so "the newest `release/<id>/`"
  is usually the previous one — distilling it silently regenerated from a
  stale (here: pre-fix) release while tests skipped. Rule: an explicit
  release dir wins; inside the enclave a live release that differs from the
  latest export is distilled (category-2 rows from the policy-filtered
  marts, with a loud pre-export warning); off-enclave bundles use their
  export. Record the choice in the spec sidecar, and make every consumer
  (validate, tests) read the release the spec was built from. Bundles are
  built only from exports, so an unexported release never ships.

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
- **Render the source's record STRUCTURE and VOLUME, not just its fields**
  (each of these regressed silently in the GeoLife build):
  - *Cadence*: sample at the released per-category median interval, not a
    fixed points-per-record budget (a "~120 points per trip" rule produced
    5% of the source volume for the same 64 users).
  - *Record grouping*: one id per source grouping unit (GeoLife: one
    sequence per labelled window — sharing an id across a trajectory made
    the production leg builder fuse out-and-back trips).
  - *Conditional structure*: categories that depend on a released quantity
    must be drawn conditionally on it (trip mode ∝ released weight ×
    plausibility of the trip length around the mode's released median path —
    otherwise the generator invents 20 km walks and 2-minute rail hops).
  - *Per-entity volume*: heavy-tailed, from a declared public prior unless a
    per-entity histogram is released; every entity in the cohort has ≥ 1
    observed record (that is why it is in the data).
  - *Stationary periods* are sparse in movement-logging sources; full-rate
    "stay" rendering inflated one category to 71% of points.
- **Planted ground truth must be independent of DP noise and must have
  slack.** Planted entities get fixed public parameters (e.g. a 07–08 h
  departure) and geometry chosen well inside the detector's thresholds (a
  planted tour leg at the 2 h trip cap equalled the 2 h chaining bound and
  failed on detection jitter). State the slack as a decidable theorem
  where possible (`planted_departure_reaches_work_window`).

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
- **Every fidelity axis you care about needs a check against a released
  reference.** Volume (re-ingested cadence vs the released median interval),
  per-category volume share (vs released per-category counts) and time of
  day (vs the released departure mix) each regressed without anyone noticing
  until a soft check existed. Where no released reference exists (per-entity
  spread, the point hour profile), validate REPORTS the value and the
  enclave-only realism audit compares it with the real cohort (total volume
  ratio, per-entity spread, hour-of-day TV). Smoke-test each new check on the
  buggy output it is meant to catch.
- **Run manifests** fingerprint everything (data sha, spec, code, model
  versions, table hashes); a reproduce script rebuilds into scratch and
  compares within tolerances.
- **Notebooks** (marimo): one EDA notebook for the enclave data, one
  synth-vs-real comparison notebook (per-column TV scorecard with each
  column's fill source; declared-gap framing). Validate headless
  (`marimo export html` + grep for tracebacks/oversized cells). Charts carry
  aggregates, never per-row dataframes; keep a bounded
  `alt.data_transformers.enable("default", max_rows=20_000)`.

## 11. Testing and operations discipline

- Suite shape (~65–70 tests in the flows build; 144 in GeoLife): spec sync (3 mirrors), SQL↔Lean
  recompute agreement, SampCert conformance (GOF), release gate (attestation,
  egress, policy, exact-ε ledger, default-deny list), distillation
  (provenance/shape/fidelity), synth (meta assumptions, beacon recall,
  exit-code mapping, corruption → hard fail), per-column synth, minimodel
  gates. Every fixture that reads a produced artifact must `pytest.skip` with
  a "run `make X`" message when it is absent.
- `make all` ordering matters: `dbt train ml lean distill test manifest` —
  distill AFTER lean so the genspec always matches the fresh (non-reproducible)
  release.
- One pipeline driver at a time: concurrent `make` runs race over the DB write
  lock, the GPU and generated dirs that `clean` deletes mid-flight. When
  sibling builds share a machine, stop processes by PID or full project path —
  never by a bare name pattern (`make synth`, `bin/generate` exist in every
  sibling).
- A retrain after a spec change must be activated explicitly
  (`FLOW_RETRAIN=1`); content addressing otherwise keeps the old version
  active.
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

## 13. Acceptance checklist

- [ ] Raw data parses or is verifiably repaired; quality flags counted; time
      zone / units verified from the data and converted once in staging.
- [ ] Spec in dbt vars = spec.py = Params.lean (sync test green).
- [ ] Every minimodel beats (or honestly fails) a frozen baseline; artifacts
      content-addressed and registry-activated.
- [ ] Every exported aggregate appears in `release_policy.yaml` under one of
      three categories; ledger exact-fraction, ≤ budget, enforced in Lean
      AND Python; denied list covers all row-level relations; ε values
      single-sourced from the policy.
- [ ] Lifetime ε ledger in place (append-only, keyed by data sha, reconciled
      with every export, lifetime cap declared); the stated guarantee matches
      the cumulative spend.
- [ ] Released mix sanity-checked against the enclave truth (pre-noise clipped
      counts vs noised, TV per histogram) — §6 lessons applied.
- [ ] genspec built from the CURRENT release (recorded in its sidecar);
      provenance test green; fidelity report `accepted: true` under a
      pre-declared criterion (or an honest negative).
- [ ] `lake exe generate` renders schema-valid records; validate exit 0/2;
      planted patterns recalled at 1.0 with slack; certificate kernel-checked;
      volume, per-category share and time of day checked against released
      references.
- [ ] Full suite green; run manifest written; both notebooks export headless
      with no tracebacks; wiki pages + log updated.

---
title: Techniques & practices
type: synthesis
updated: 2026-11-03
sources: [log.md, ../tests, ../Makefile, architecture.md, geolife-reference.md]
---

# Techniques & practices

The working rules this repo converged on, with the incident or decision that
earned each one. Newest lessons first within each group.

## Data in

- **Repair, don't drop, known serializer bugs** — Joy's `},,{` empty
  cert-extension element breaks whole JSONL lines; `scripts/fix_joy_jsonl.py`
  (`make repair-data`) repairs in place, touches only lines that fail
  `json.loads`, verifies the re-parse, refuses to guess at anything else.
  Dropping would have lost exactly the TLS-rich flows this project studies
  ([[data-quality]]).
- **Profile raw blocks before modelling them** — the TLS/DNS layers were
  designed after counting what the capture actually holds (query-heavy DNS,
  8→33 stacks); thresholds picked from measured distributions, not defaults
  (e.g. top-k suppression moved ≥3→≥2 hosts after seeing 8,817 names sit at
  exactly 2 hosts).
- **Quality fixes leave flags, not silence** — negative durations, entropy>8,
  zero-byte flows are clamped/kept with `qf_*`/`is_*` flags and counted in
  `q_data_integrity` ([[data-quality]]).

## Spec & sync

- **One spec, three mirrors, test-locked** — dbt vars ↔ `python/spec.py` ↔
  `Params.lean`; `test_spec_sync.py` fails on drift. Generator constants live
  once in `FlowShape.lean` and are regex-read by Python — the simulation
  cannot drift from the proved constants.
- **Version the spec, not just the code** — threshold changes bump
  `spec_version`; artifacts and exports record it.

## Models

- **Content-addressed, immutable, explicitly activated artifacts** — version
  = `sha256(name, data, hparams, code)`; retrains are no-ops unless inputs
  changed; `FLOW_DEVICE=cpu` for deterministic mode.
- **Adoption gates with declared baselines** — every learned model records
  `adopted` vs a frozen non-ML baseline (majority, repeat-last, unigram).
  Honest negatives are kept and documented; the flag is **data-dependent**
  (TlsRecordGRU: not adopted on Nitroba n=112, adopted at 0.838 vs 0.450 on
  the large capture). Read `meta.json`, never a wiki page, for current flags.
- **Dedup + weight instead of brute scale** — DnsCharLM wedged at 213k
  per-record sequences on MPS; training on 9,209 distinct names weighted by
  query count is the same expected gradient at ~90× fewer steps, and the
  frozen record-level evaluation keeps the gate meaningful ([[minimodels]]).

- **Distill to data, not code** — teachers are replaced by a RESTRICTED
  typed spec (histograms, conditional rows, constraints), never generated
  Lean programs; provenance is test-asserted weight-by-weight
  ([[distillation]]). Public structural priors (e.g. the self-transition
  boost) are declared `derived_from_data: false` and rescue what DP noise
  destroys — measured in the fidelity report, never assumed.

## Formal layer

- **Decidable mirrors + agreement counts** — every SQL predicate has a
  decidable Lean twin re-decided on exported data; the agreement count is the
  cross-check that catches encoder bugs (the `least()` NULL bug was found
  this way, by the synthetic round-trip).
- **Model data for the proof, not the parser** — DNS names as `List String`
  labels makes RFC bounds pure arithmetic; accumulator-based structural
  recursion (`NoEarlyData`) buys decidability and plain-induction proofs.
- **Exact rationals at every ledger** — ε budgets compare as `Fraction`/Nat
  cross-multiplication in Python AND Lean; floats only as legacy fallback.
- **Certificates `by decide`** — synthetic structures are kernel-checked as
  literals (GenCert), not trusted from the generator.

## Privacy & release

- **Default deny; three explicit exit categories** — ε-DP histograms,
  suppression aggregates, common-knowledge metadata ([[architecture]]). A new
  release is a policy diff + ledger entry, never an ad-hoc file write.
- **Public cell grids only** — DP cells are fixed taxonomies or RANK labels
  (`stack_1..5`); the rank→fingerprint map stays inside. No qname, hash or
  cipher list reaches the mechanism.
- **The privacy unit is a design decision the data can break** — "one source
  host" was fine on Nitroba; on CIC-IDS, 2,433 external one-flow sources
  outvoted 15 monitored hosts and the released service mix came out inverted
  (dns→0 while ephemeral dominated). Unit = monitored host, flows attributed
  to their internal endpoint; and uniform per-cell clips saturate for heavy
  hosts — a PROPORTIONAL L1 allocation (`dp_clip_total`) is what preserves a
  mix ([[dp-release]], 2026-10-31).
- **Budget near-saturation is a feature** — 14/5 + 6/25 + 9/25 + 49/100 =
  389/100 of 4/1; a new query must fit in 11/100 or displace one of these,
  making the privacy cost of new features visible at review time.
- **Structurally-private classes are excluded, not thresholded** — `*.arpa`
  (encodes internal IPs) and single-label names (local hostnames) never enter
  top-k artifacts regardless of popularity.
- **Declared gaps over silent fabrication** — what the generator does not
  model is written in `synth/meta.json` (`fields_not_modelled`,
  `derived_from_data: False` per assumption) and asserted by per-column tests.

## Verification loop

- **Round-trip everything** — synth output re-ingested through the production
  dbt+Lean pipeline; audits cut both ways (found encoder bugs, not just
  generator bugs).
- **Three-way recompute tests** — pytest re-derives fingerprints/qnames from
  raw `flow.jsonl` in Python and compares to SQL and to Lean verdicts.
- **Notebooks are validated headless** — `marimo export html` + grep for
  Tracebacks and section headings before calling a notebook done.
- **Charts carry aggregates, never per-row dataframes** — Altair embeds its
  dataframe in the output, so `disable_max_rows()` + client-side binning on
  the 407k-flow capture produced a 442 MB cell marimo refused to show. Bin
  in SQL/numpy, chart the counts, and keep a bounded
  `max_rows=20_000` transformer so a regression fails loudly instead.
- **Tests tolerate sampling, not bugs** — a draw with zero dns flows skips
  the dns assertions (released share is tiny); it never weakens structural
  assertions on flows that do exist.

## Operations

- **One pipeline driver at a time** — concurrent `make` runs race over the
  DuckDB write lock, MPS, and `lean/data` (clean steps delete it mid-flight);
  the mid-run file-vanishing incident of 2026-10-28 came from exactly this.
- **ctx_shell blocks some tools permanently** (elan/dbt/pip/`python -c`) —
  use native Bash with `export PATH="$HOME/.elan/bin:$PATH"` for lake.
- **Run manifests fingerprint everything** — data sha256, spec, code, model
  version, table hashes (`scripts/run_manifest.py`); `reproduce_check.py`
  rebuilds into scratch and compares within tolerances.
- **The wiki is maintained on every ingest** — pages + index + append-only
  log ([[SCHEMA|SCHEMA.md]]); current-state claims live on pages, history in
  the log.

DuckDB gotchas collected so far: `least()` ignores NULLs;
`array_to_string([], sep)` is NULL; JSON columns need explicit `from_json`
schemas ([[data-quality]], [[cisco-joy-features]]).

## Rules earned on the stock migration (2026-10-31, [[stock-pipeline]])

- **Never release pair-level counts over protected entities.** A pairwise
  statistic has sensitivity n−1 per cell (removing one entity deletes all its
  pairs) and drowns at any affordable ε when n is small. Release per-entity
  summaries instead (each entity = one count in one bin, sensitivity 1):
  the co-movement histogram went from λ=0 (useless) to λ=70% this way.
- **Noise-dominated conditional rows need released-marginal anchors.** When
  aux DP histograms are below SNR 1 (unavoidable at n=8), sampling them raw
  drifts the generator's marginal away from the released mix. Anchor each
  conditional row to the released marginal (post-processing) and rescale
  secondary conditioners (dow rows → fixed small total) so they modulate
  rather than dominate; the TV to the released mix dropped 0.305 → 0.146.
- **An exactly saturated ledger catches arithmetic slips.** The Lean release
  executable's exit-2 refusal fired on a 402/100 mis-sum before anything
  left the enclave — budget enforcement as a safety net, not bookkeeping.


## Rules earned on the GeoLife fidelity pass (2026-11-03, [[geo-pipeline]], [[log]])

- **Verify the source's time zone from the data, not the docs.** GeoLife
  timestamps are UTC (labels split at 16:00:00 = Beijing midnight; the hour
  profile is empty 16–21 h). Our own dataset page said "local"; every
  time-of-day concept (home anchor at 03:00, work window, departure bins,
  day boundaries) was shifted 8 h. Convert ONCE in staging via a spec var;
  generators plan in local time and write the source's convention back.
- **Pure DP composes across exports — keep a lifetime ledger.** A per-export
  ε check is not a guarantee: a re-release after a pipeline fix spends again.
  Append-only ledger keyed by the data's sha256, outside `make clean`,
  reconciled against every export manifest, spend recorded BEFORE files are
  written, deleting an export never refunds; raising the cap is an explicit
  owner decision.
- **At small n, check SNR before trusting a release.** SNR ≈ share × n ×
  ε_cell; at n = 64 and ε_cell 0.05 the departure histogram was noise
  (morning = 0). Compare noised vs true clipped counts (TV) inside the
  enclave after every release; the generator reproduces whatever noise it
  is given.
- **Never distill "the latest export" blindly.** `make train` computes a new
  release before `make export`; distill must prefer the live release when it
  differs (inside the enclave), and every consumer (validate, tests) must
  read the release the genspec was built from — recorded in genspec_meta.
- **Planted ground truth must not depend on DP noise or zero-slack bounds.**
  Planted users get fixed public departure windows, separated anchors and
  legs with large slack under the detector's thresholds (a tour leg at the
  2 h trip cap = `tour_max_gap_s` broke recall on detection jitter).
- **Validate every axis you care about, against a released reference.**
  Volume (cadence vs released `dt`), per-mode point share (vs released
  `n_points`) and time of day (vs released departure mix) each regressed
  silently until a check existed; where no released reference exists, report
  in validate and compare with the real cohort in the enclave-only audit.
- **Render the source's record structure, not just its fields.** One
  sequence per label window (as in the source) — merging windows made the
  production leg builder fuse out-and-back trips. A cohort member always has
  ≥ 1 recorded day. Mode depends on trip length (released per-mode median
  path), or the generator invents 20 km walks.
- **Scope process kills to the project.** Sibling `lean4_*` builds run
  identically named processes (`make synth`, `bin/generate`); kill by PID or
  full project path, never a bare pattern.

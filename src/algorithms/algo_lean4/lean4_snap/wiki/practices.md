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


## Release what survives its own noise (lean4_snap, 2026-11-03)
The community-size histogram was first released as one-event-per-COMMUNITY
(15 events, sensitivity 2, ε_cell 4/100 → Laplace scale 50). The FIRST draw
happened to look plausible and sailed through; the SECOND draw released
(71, 0, 0, 0, 0, 0) — pure noise — and the generator built a degenerate
graph (closure 0.11). Two rules earned:
1. A lucky first draw is not calibration. Run the pre-noise SNR simulation
   (or at least re-release twice) before trusting any small-count grid —
   the stock-migration n=8 lesson applies PER GRID, not per dataset.
2. When a grid's natural unit is too small, re-express it in the privacy
   unit's scale: node-weighted community sizes (sensitivity 1, counts in
   thousands) release cleanly, and the community-count view is recovered in
   distillation by dividing by PUBLIC bin midpoints (post-processing).


## A validator that skips what it cannot judge hides the bug (lean4_snap, 2026-11-03)
The first affiliation generator shipped with three defects that passed
validation: member-weighted bin weights used as group-count weights, an
open-bin size cap justified by a claim ("p99 is well below 40") that nobody
had measured (real: mean 83, p99 623), and mid-bin density overshoot. The
validator had skipped every cell with < 20 synthetic groups — exactly the
cells that were wrong — and nothing compared group COUNTS. Rules earned:
1. A check must report what it does not judge (`affil_unjudged`), never
   `continue` past it; tolerances should scale with the sample behind the cell.
2. Compare the DISTRIBUTION of a generated structure (group-count mix by bin,
   TV) as well as its per-cell moments.
3. Prove a new check by mutation: feed it the old defect (unit test with the
   pre-fix numbers, plus an end-to-end run with the defect reintroduced) and
   require a failure.
4. Do not write a prior's justification from memory: measure the quantity
   (here, one query) before declaring a cap "well above" or "well below" it.
5. Scale a release's WEIGHT semantics to its sampling unit: a member-weighted
   histogram is divided by the public bin midpoint before it is used as
   group-count weights (same rule as node-weighted community sizes).


## Test the documented range, not the reference point (lean4_snap, 2026-11-03)
`generate` accepts 200-100,000 nodes; everything was tuned and verified at
4,039. The first sweep found a crash at the small end (exit 1 at n=200, a
latent bug independent of the feature being built), tolerances that cannot hold
below ~400 nodes, and a closure decline above ~10k. Rules earned:
1. Sweep the whole advertised range once (cheap: a loop and a metrics script)
   before calling a generator "done"; one size is one data point.
2. Attribute before fixing: re-run with the new feature switched OFF to
   separate "my change" from "the baseline" (the closure decline was baseline).
3. Revert a fix whose effect you could not measure (the hub-star hypothesis).
4. Declare a validated window in code (public constants, mirrored and
   sync-tested), warn outside it, and record what is not judged there.
5. Tolerances that ignore sample size misfire at the edges: scale them with n.


## A build that fails silently turns every later experiment into noise (lean4_snap, 2026-11-03)
`lake` is only on PATH inside the Makefile recipes. An ad-hoc `(cd lean && lake build
...) | grep error` printed nothing - `lake: command not found` contains no "error" -
so a whole series of "experiments" re-measured the previous binary and its
run-to-run noise, and were reported as falsified hypotheses. Rules earned:
1. Build through a wrapper that exits non-zero and prints BUILD FAILED (set -e, no
   filtering to a keyword), and confirm the binary's mtime moved before measuring.
2. Never filter build output down to a keyword; show the tail.
3. A result that is identical across supposedly different variants (here: 0.118,
   0.115, 0.104, 0.105 for four different generators) means the variants did not run -
   suspect the harness before the hypotheses.
4. Measure the noise floor first (5 runs of one setting); differences under it are
   not findings. Here it was +-0.012.
5. A null result you are about to publish ("X does not help") deserves one cheap
   check that X was actually in the build.

## Spend extra epsilon where an audit says it buys fidelity (lean4_snap, 2026-11-03)
When the budget is raised deliberately, audit first: relative error and the number of
cells below 2x the Laplace scale per grid (mixing: 36 of 45 cells, relative error 34%);
measure what the downstream structure actually needs (edge coverage by category showed
locale and gender, not the categories first guessed, carried 29 points of coverage);
and check the clip before the epsilon (a clip of 2 shrank memberships ~4x - a BIAS no
amount of epsilon removes). Then state the resulting bound plainly (eps ~13.7 is no
practical guarantee) and record that earlier draws compose.


## A/B on enough runs before quoting a delta (lean4_snap, 2026-11-03)
A 4-run comparison reported "edge count 85.0k -> 88.9k" after a generator change; an
8-run A/B on two separately built binaries showed 86.4k +- 3.1k vs 86.6k +- 1.5k - no
effect. Rules: (1) measure the run-to-run sd of the metric first; (2) A/B at >= 8 runs
and quote a delta only if it exceeds ~2 sd; (3) build the BEFORE and AFTER generators
into separate binaries and confirm they differ by content, so the comparison is not
against a stale build; (4) a favourable number that "matches the real value exactly"
(88.9k vs 88.2k) deserves the most suspicion, not the least.

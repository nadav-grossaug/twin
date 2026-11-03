---
title: snap as-built pipeline
type: component
updated: 2026-11-03
sources: [../Makefile, ../scripts/synth.py, ../lean/Generate.lean, ../python/distill.py]
---

# snap-pipeline — the as-built lean4_snap pipeline (fifth instantiation)

As built 2026-11-03. Layer map per [[architecture]]; plan was
[[migration-plan-snap]]; data profile [[dataset-snap]]; interfaces
[[snap-contract]].

## Layer map (as built)

- **Ingest**: `python/ingest_graphml.py` → `artifacts/ingest/*.tsv` (canonical
  edges, feature long table, circles, Louvain seed 7 relabeled by size,
  sha256 fingerprint). Re-used verbatim for synthetic re-ingestion.
- **dbt** (DuckDB `artifacts/enclave.duckdb`): 35 SQL models + the Python
  model `ml_graph_scores` + 105 schema tests + seed, all green (141 nodes in
  `dbt build`). SQL recomputes degree/triangles/wedges from edges (9.3M
  wedge join); `q_data_integrity` cross-checks networkx exactly (0
  mismatches); `q_temporal_fields` proves the static-snapshot claim (507
  time-looking keys, all binary flags). Motif counts: hub 215, leaf 75,
  broker 32, bridge 499, butterfly 302, core 368. Top associations:
  cc×degree Cramér's V 0.525, education×work 0.406.
- **Minimodels** (PyG 2.8.1, CPU-deterministic, content-addressed): nine
  models — eight adopted, one honest negative. Active artifact **e999a0ae31fd**
  (retrained 2026-11-03 on spec 1.2.0 in 68 s; supersedes 6c004f779b2e with all
  nine verdicts unchanged and deterministic models identical to 4 dp):
  degree_sage val acc 0.589 vs majority 0.192 ✓; community_gnn ARI 0.348 vs
  gate 0.30 ✓; **triad_vgae val AUC 0.945 LOST to the common-neighbor
  baseline 0.990** (expected at transitivity 0.52; kept, adopted=false);
  featcond_encoder mean AUC 0.674 ✓; ego_motif_pool macro-F1 0.422 vs 0 ✓;
  and the AFFILIATION set over shared-key communities (see
  [[affiliation-communities]]): affil_ae (951 keys → 16-dim, BCE 0.018 vs
  marginal 0.035) ✓, affil_link_mlp (AUC 0.951 vs the total-shared-keys
  ranker 0.935 — WHAT you share beats HOW MUCH) ✓, key_community_mlp
  (affiliations alone predict the Louvain community at 74.1% vs majority
  17.8%) ✓, key_cohesion_mlp (density MAE 0.126 vs mean-predictor 0.355) ✓.
  Key-level group marts stay enclave-side (denied list); only the DP size
  histogram and the suppression-released cohesion table cross (see below).
- **Lean formal layer** (SampCert a2c53bb / Lean v4.34.1, shared
  LEAN_HOME store): `lake exe checks` re-decided 24,234 motif + 20,195
  class + 8 exact-ℚ χ² checks — **0 disagreements**. 28-claim ledger
  (24 proved incl. histMech_DP, sampleCellPMF_law, edgesCanonical_sound,
  isBroker_not_isCore; 4 assumed incl. the DECLARED `row_level_privacy_unit`
  node-DP gap).
- **DP release**: discrete Laplace per cell, main degree grid + 10 aux
  grids, exact-fraction ledger **1369/100 ≤ 14/1** enforced in Lean (exit-2
  refusal) and Python (`Fraction`). **Fidelity-first budget (user decision
  2026-11-03)**: the cap was raised from 4/1, accepting a much weaker privacy
  bound (e^ε at ε ≈ 13.7 is no practical guarantee; the row-level unit and the
  declared node-DP gap weaken it further). The extra budget was spent where a
  per-grid noise audit showed it buys fidelity: the mixing grid (45 cells,
  2/100 → 10/100: TV vs exact 0.172 → 0.049, relative error 34% → 7%; 36 of 45
  cells had sat below 2× the noise scale) and the affiliation key-group size
  histogram (8 → 36 cells over NINE categories, clip 2 → 16, 18/100 per cell;
  clip 2 had shrunk everyone's memberships ~4× and halved the shared-key
  coverage). Triangles/motifs stay at 2/100 (informational: the generator
  never samples them). Per-node proportional L1 clip: 16 affiliation, 8
  category/mixing/closure, 4 motifs, 1 on one-cell-per-node grids. The
  cohesion rollup crosses as the suppression artifact `affil_cohesion_stats`
  (29 of 34 rows; 5 suppressed below 50 members). Earlier dev releases
  (`release/`) compose with this one.
- **Distill**: genspec keyed to the release sha; weight-by-weight provenance
  test (every genspec weight equals a released value or released + a declared
  public prior); fidelity accepted (released-vs-exact degree TV 0.023; released
  per-node closure level 0.604 vs exact transitivity 0.519 — the per-node
  equal-weighting semantics, declared). Affiliation fidelity (enclave-only,
  like-for-like on the 19 cells key_cohesion_mlp was trained on): teacher MAE
  0.058, mean-baseline 0.263, released-table student 0.003; the other four
  categories are student-only.
- **Generator** (`lake exe generate`, Lean-only): community sizes → degree
  targets (sorted per community) → **near-clique blocks** (size ≈ d/q+1 at
  density q = 5/3·released closure p, min-target pair wiring, homogeneity
  cap) → **affiliation key-groups** (nine categories; members drawn from a
  window of 3× / 16× / 8× the group size for sizes 2–4 / 5–16 / 17+ inside one
  community, sized by the released member-weighted histogram ÷ public bin
  midpoints, per-category open-bin caps; the target edge count is stochastically
  rounded and the fill prefers pairs that share a neighbour so added edges
  close wedges; shared feature instance ids — the same-school/same-workplace,
  same-locale, same-gender cliques; planted BEFORE the stubs so the
  deficit-adaptive stub pass absorbs the group edges) →
  mixing-row stubs (deficit-adaptive, receiver-capped; nodes in the open degree
  bin initiate their FULL deficit — otherwise hubs stop at ~half their target) →
  one light closure pass → planted hub(0)/broker(2)/butterfly(1) with DECIDABLE
  self-verification (the broker search falls back to the partner with the
  fewest common neighbours when strict zero-common-neighbour candidates run
  out, which happens on small dense graphs) → `edgesCanonical` proved output
  check → GraphML + plan.json. ~4 s for 4,039 nodes / ~86.6k edges.
- **Validation**: `make synth` AND `make synth_from_bundle` run validate plus a
  **replicate verdict** (`synth.py replicates`, 8 runs) that judges the
  distributions, not one sample: per-node closure (≥ 85% of runs inside the band
  around the released 0.604 and the mean ≥ 0.02 above its lower edge) AND the mean
  |synthetic − released| affiliation group density ≤ 0.08. Last scored: local
  closure mean 0.543 (min 0.534, margin 0.089), bundle 0.538 (margin 0.084);
  density error 0.047 / 0.048 (healthy 0.046–0.058; the single-window generator
  measured 0.095–0.119). Single-run validate checks planted recall 1.0 ×3, degree
  TV ≈ 0.08, group-count mix per category (sample-aware tolerance), per-cell
  density (tolerance clamp(0.6·released, 0.15, 0.30); a healthy run still trips a
  low-target cell about 1 time in 12, which is why per-cell density is a warning
  and the replicate verdict is the authority), **0 Lean disagreements**,
  certificate PROVED. Realism audit (graded; means ± sd over 8 runs of the final generator, real in
  parentheses): shared-key edge fraction 0.86 ± 0.02 (0.98; was 0.34), link-judge
  edge score ≈ 0.47–0.52 (0.70; was 0.14), edges 86.6k ± 1.5k (88.2k), transitivity
  0.531 ± 0.023 (0.519), modularity 0.840 ± 0.010 (0.835), core share 0.102 ±
  0.021 (0.091), average clustering 0.530 ± 0.008 (0.606), degree TV ≈ 0.08.
  Communities per run: 11–18, mean ≈ 14 (15) — an earlier "9 vs 15" was one run's
  draw, not a gap. Residuals: HUB TAIL — share of nodes with degree ≥ 100 is 0.105
  (0.122) but ≥ 150 is 0.002 (0.053) and the maximum degree 183 (1,045): the release
  has a single open "65+" degree cell and the declared prior draws uniformly on
  [65, 200], so the real heavy tail is not representable without a finer
  released degree grid; leaf share 0 (0.019); broker share ≈ 0.003 (0.008).
  Full suite 107 passed.
- **Size range** (measured 2026-11-03, re-measured on the nine-category
  generator): generation succeeds from 200 to 100,000 nodes. Cost is roughly
  linear (4.3 s at 4k, 9.2 s at 8k, 20.7 s at 16k); **100k takes ~5 min and
  ~3.1 GB for 2.2M edges** (the two-category generator needed 76 s). Fidelity
  tolerances are **validated for 500–10,000 nodes**: the replicate verdict at
  10k passes with closure margin 0.061 and density error 0.046; at 500 the
  degree TV is ~0.21 against the 0.25 tolerance (thin), and 1000-2000 sit at
  0.08–0.11. Below 500 the planted hub (degree ≥ 150) touches most of the graph
  and a few hundred draws carry sampling noise, so degree-mix and group-count
  tolerances cannot hold by construction (degree TV 0.25–0.30 at 200): validate
  records them as unjudged and the generator warns. Above 10k the older
  two-category measurements showed a closure decline (0.52 at 4k → 0.48 at 20k)
  caused by the rank-labelled mixing design (the "rest" bucket grows from 14%
  of nodes at 4k to 77% at 20k at lower closure); that was NOT re-measured at 20k
  with the better mixing grid, so treat >10k as unvalidated.
- **Ops**: six-flow Makefile + `make help`; single source-only bundle
  (`dist/bundle/`); `synth_from_bundle` rebuilds the generator from bundle
  sources via the shared store; marimo notebooks validated headless.

## Gotchas earned here (→ [[practices]])

- dbt-duckdb's Python-model ref extractor MISSES `dbt.ref(...)` calls whose
  statement continues with a backslash — keep each ref on its own simple
  statement.
- A lone leftover node at a community tail forced block size 2 past the
  array end (Lean panic → silent default in release builds): clamp block
  size to the remaining run, let singleton blocks be empty.
- Generator realism knobs that mattered, in order: block-clique structure
  (uniform wedge-closing alone plateaus at transitivity ≈ 0.24), min-target
  pair wiring (protects the degree histogram), receiver caps on deficit
  stubs (protects the LOW bins — the stub flood had wiped deg_1..deg_4),
  densification 5/3 (compensates stub dilution).
- Compare like with like: the released closure grid is per-node
  equal-weighted, so synth is judged on per-node mean closure, not
  edge-weighted transitivity (which sits lower by construction — same
  effect as real avg-clustering 0.6055 vs transitivity 0.5192).

- Every generator knob below was set from a measured dose-response, not a
  guess (5–8 runs per setting; run-to-run noise of the metric measured first):
  per-bin affiliation member windows 3× / 16× / 8× (a single factor of 8:
  mean density error 0.095–0.119; tight windows overshoot, wide ones undershoot
  because the degree-capped fill cannot add enough); STOCHASTIC rounding of a
  group's target edge count (deterministic rounding turned every size-2 group
  with released density ≥ 0.5 into a full pair); closure-biased density fill
  (weight 1 + 16·min(common neighbours, 3): closure back to the
  no-affiliation baseline). Tested and rejected: a larger closure-pass trial
  cap (no gain, +7k edges), choosing the sparsest of 8 candidate member sets
  (worse), anchoring groups to nest (worse), placing large groups first
  (neutral).
- Process gotchas (→ [[practices]]): a bare `lake build | grep error` fails
  silently outside the Makefile and silently invalidated a series of
  experiments; A/B on ≥ 8 runs before quoting a delta under ~2 sd; validators
  must report what they do not judge and be proven by mutation.

## Declared gaps (meta.json `fields_not_modelled`)

degree–community dependence; feature identity structure for the two
unreleased categories (first_name, other: ≤ 0.001 edge coverage); the NESTING
of smaller groups inside the locale/gender containers; the heavy degree tail
beyond the declared 200 cap (share ≥ 150: 0.002 vs 0.053 real — needs a finer
released degree grid); leaf share; circle rosters; inter-ego overlap; degree
assortativity. The realism audit quantifies the residuals: shared-key edge
fraction ≈ 0.86 vs 0.98 (unmodelled nesting; small cells zeroed by DP noise),
average clustering ≈ 0.53 vs 0.606, transitivity 0.531 vs 0.519 (no longer
below real by construction: the earlier undershoot was the half-deficit hubs).
Compare like with like: the released closure grid is per-node equal-weighted,
so closure is judged per node, not as edge-weighted transitivity.

# M v2 (outings): design, planted-truth recovery, real map matching, the transfer test

**This report is the 2026-11-02 step-8 redo** (PROCESS.md D28-D30). It replaces the earlier m-v2.md
(git history), whose numbers were measured at the 75 m snap radius with the pre-D24-D27 generator and the
29-table artifacts. Everything below is measured at snap 150 m (D17), with the regenerated 26-table
artifacts (`outings_v2_seed0-2`, eps 6.000) and the step-4/5 generator (AR(1) receiver noise, calibrated
outing process). The report name is kept (M is still v2; only its relation rule changed, see section 2) so
PLAN/PROCESS links stay valid.

Code: `src/geolife/eval/m/outings.py` (M), `src/geolife/eval/m/outings_transfer.py` (harness;
`python -m geolife.eval.m.outings_transfer`; stages `rules`, `dev`, `held`, `sens`, `split`, `real`,
`beijing`; real stages need `GEOLIFE_POINTS` / `GEOLIFE_PLACES`), `geolife/eval/match_real.py` (matcher on
real GPS), `geolife/eval/score.py` (per-variant scoring). Tests: `tests/geolife/eval/test_m2_*` (17 tests).
Raw numbers are transcribed from the harness JSONs (not committed); committed aggregates:
`geolife/eval/results/match_real.json`, `compare_outings_v2_seed0_d27.{csv,md}`.

## 1. Design

Pipeline: traces + map -> `tokenize.outings` (public parameters: gap 2 h, stop 200 m / 10 min, snap
**150 m** (D17), HMM matcher when a road graph exists) -> per-user place table -> places of interest,
common places/segments, co-visit graph, Louvain. Unchanged from the previous report except the relation
rule (section 2): map-resolution place merge (60 m leader clustering on the public map), POIs = places with
>= 2 distinct outings on >= 2 local days with an 8-bin visit-time profile, common places = >= k distinct
users (a user counts with >= 2 outings or >= 1 outing with a real stop), common segments = HMM-matched
segment ids (res-9 traversed cells where no road graph is given), relations = significantly shared places
of interest under a popularity-only null (binomial-matched Poisson-binomial tail), IDF edge weights with a
time-profile bonus, Louvain on top. No absolute coordinates after tokenization.

## 2. Relation significance in sparse regimes (rule re-chosen on dev only; D28)

The old rule (BH at FDR 0.2 over **all** n(n-1)/2 user pairs, >= 1 shared item) kept only 2 edges on the
old held-out draw although the p-value ranking carried signal (AUC 0.87). Cause: pairs that share nothing
have p = 1 and can never reject, yet still count as tests, so in sparse regimes BH demands p < alpha/~2,700
for the first edge. Fix: the BH family is now **the pairs that share >= 1 item** (`rel_universe="sharing"`;
the only pairs the statistic can test), with `rel_min_shared >= 3` to buy back the precision the smaller
family costs.

Chosen by a sweep (`--stages rules`) over {all, sharing} x FDR {0.05..0.5} x min_shared {1,2,3} on **two
dev draws only**: the dev draw (toy map seed 1, gen seed 11) and a new dev-sparse draw (map seed 2, gen
seed 13, 15 s sampling, 10 m noise, 6 gaps/h, 35% stops unlogged, ~28 outings/user) that is deliberately
different from the held-out draw (map seed 7, gen seed 4242, 20 s, 15 m, 8 gaps/h, 40%). The held-out draw
was not consulted. Selection criterion: precision >= 0.8 on both draws, then the best recall/ARI balance
within the sharing family (preferred on principle: its test count adapts to the number of testable pairs,
which is exactly what collapses in sparse regimes).

| rule (on dev / dev-sparse) | edges | precision | recall (group) | ARI grouped |
|---|---|---|---|---|
| old: all pairs, FDR 0.2, shared>=1 | 122 / 121 | 0.84 / 0.89 | 0.42 / 0.38 | 0.80 / 0.81 |
| **frozen: sharing pairs, FDR 0.2, shared>=3** | 173 / 175 | 0.80 / 0.86 | 0.57 / 0.52 | 0.81 / 0.80 |

Effect where it matters: on the held-out draw (never tuned on) the old rule now keeps 19 edges (recall
0.06, ARI 0.10), the frozen rule 58 (recall 0.14, ARI 0.28; section 3); on synthetic Beijing split halves the old rule's
edge reliability was 0/undefined in the previous report, the frozen rule gives edge Jaccard 0.39/0.30
(section 7). On dense data the two rules nearly coincide (full Beijing run: 1,459 vs 1,417 edges,
precision 0.666 vs 0.660).

## 3. Planted-truth recovery

Three sources of synthetic data, all scored against the planted truth at the same 60 m map resolution,
truth restricted to what the trace can show (observed stops, tokenized outings):

**(a) Toy harness draws** (GridCityProvider 40x40, 100 users generated, hand-set `fake_params_v2`, new
generator; same seeds as the old report, so the change vs the old numbers = new generator + 150 m snap +
new rule):

| draw | POI P/R | common places k=2 P/R | k=3 R | k=5 R | segments k=2 P/R | edges: precision / recall (group) | edge AUC | ARI grouped | n_edges |
|---|---|---|---|---|---|---|---|---|---|
| dev (seed 11, 10 s, noise 6 m; 70 kept) | 0.87/0.88 | 0.86/0.78 | 0.73 | 0.75 | 0.97/0.96 | 0.80 / 0.57 | 0.94 | 0.81 | 173 |
| dev, old rule | 0.87/0.88 | 0.86/0.78 | 0.73 | 0.75 | 0.97/0.96 | 0.84 / 0.42 | 0.94 | 0.80 | 122 |
| held-out (map seed 7, seed 4242, 20 s, 15 m, 8 gaps/h, 40% stops unlogged, fewer/shorter outings; 73 kept) | 0.77/0.80 | 0.83/0.71 | 0.67 | 0.61 | 0.96/0.92 | 0.47 / 0.14 | 0.86 | 0.28 | 58 |
| held-out, old rule | 0.77/0.80 | 0.83/0.71 | 0.67 | 0.61 | 0.96/0.92 | 0.58 / 0.06 | 0.86 | 0.10 | 19 |
| old report (75 m snap, old generator): dev | 0.88/0.84 | 0.87/0.72 | 0.69 | 0.68 | 0.96/0.97 | 0.87 / 0.57 | 0.96 | 0.84 | 154 |
| old report: held-out | 0.74/0.72 | 0.80/0.57 | 0.52 | 0.46 | 0.96/0.95 | 1.00 / 0.01 | 0.87 | undefined | 2 |

**(b) Artifact-driven grid draws** (generated from the committed `outings_v2_seed0` artifact, 100 users;
`generate_synthetic.py` + `score.py`; "dev-style" = map seed 1 / gen seed 0, "held-style" = map seed 7 /
gen seed 4242 — same M, nothing tuned on either):

| draw | users kept | POI P/R | common k=2 P/R | k=3 R | k=5 R | segments k=2 P/R | edges prec / recall | AUC | ARI grouped | n_edges |
|---|---|---|---|---|---|---|---|---|---|---|
| grid dev-style | 93 | 0.64/0.86 | 0.68/0.89 | 0.84 | 0.86 | 0.99/0.98 | 0.70 / 0.63 | 0.93 | 0.64 | 676 |
| grid held-style | 90 | 0.64/0.88 | 0.65/0.90 | 0.87 | 0.89 | 0.99/0.95 | 0.45 / 0.73 | 0.90 | 0.44 | 1009 |

**(c) Synthetic Beijing** (139 users generated into the source city from `outings_v2_seed0`, public
places table, cell proxy — the same data as the transfer test of section 6): 126 users kept, POI P/R
0.50/0.73, common places k=2 0.46/0.73 (k=3 R 0.73, k=5 R 0.67), common cells k=2 0.98/0.90, edges
precision 0.67 / recall 0.62, AUC 0.885, ARI grouped 0.73, 1,459 edges.

Honest reading: the artifact-driven data is much denser and hubbier than the hand-set draws (users-per-
place calibration, familiar-place starts), so **recall is higher and precision lower** than the old
numbers everywhere: POI precision 0.50-0.64 vs 0.88 before. Two known mechanisms, not new M failures:
(i) snap flicker at 150 m on dense place vocabularies (neighbouring map places within GPS noise; the 60 m
merge only partly absorbs it) creates predicted POIs the planted truth does not carry; (ii) the relation
truth is a proxy (same planted group / co-travel), and in hubby data non-group users genuinely share rare
places, which counts as false positives. Segments/cells stay excellent (P/R >= 0.90) in every regime.
The sparse-regime relation collapse is gone (58-1,459 edges per draw at AUC 0.86-0.94), but held-out
relations remain the weakest output: precision 0.47, recall 0.14, ARI 0.28.

## 4. Sensitivity (dev regime, 80 users, seed 21, new generator and rule)

| case | users kept | POI P/R | common k=2 P/R | segments P/R | edges prec / recall | AUC | ARI grouped | n_edges |
|---|---|---|---|---|---|---|---|---|
| baseline | 54 | 0.82/0.87 | 0.87/0.80 | 0.95/0.97 | 0.84/0.50 | 0.94 | 0.81 | 63 |
| noise 15 m | 54 | 0.80/0.86 | 0.86/0.79 | 0.93/0.98 | 0.82/0.50 | 0.94 | 0.73 | 65 |
| noise 30 m | 54 | 0.74/0.84 | 0.82/0.76 | 0.90/0.98 | 0.81/0.54 | 0.94 | 0.71 | 70 |
| noise 60 m | 54 | 0.60/0.74 | 0.68/0.74 | 0.89/0.97 | 0.74/0.47 | 0.92 | 0.58 | 68 |
| gaps 10/h | 55 | 0.80/0.81 | 0.87/0.66 | 0.97/0.88 | 0.86/0.32 | 0.94 | 0.60 | 43 |
| gaps 20/h | 51 | 0.81/0.72 | 0.94/0.48 | 0.98/0.77 | 0.91/0.11 | 0.86 | 0.34 | 11 |
| sampling 30 s | 55 | 0.82/0.87 | 0.85/0.77 | 0.97/0.94 | 0.71/0.46 | 0.95 | 0.66 | 75 |
| sampling 60 s | 53 | 0.81/0.83 | 0.88/0.77 | 0.97/0.91 | 0.86/0.57 | 0.96 | 0.83 | 72 |
| stops unlogged 50% | 54 | 0.76/0.88 | 0.78/0.77 | 0.96/0.96 | 0.83/0.52 | 0.93 | 0.75 | 66 |
| stops unlogged 75% | 54 | 0.70/0.89 | 0.69/0.76 | 0.96/0.96 | 0.88/0.48 | 0.90 | 0.82 | 58 |

Same shape as before: position noise hurts places first (POI precision 0.82 -> 0.60 at 60 m) while
segments barely move; dense signal loss hurts recall and relations (gaps 20/h: common-place recall 0.48,
11 edges — the new rule degrades gracefully there instead of vanishing, ARI 0.34 vs the old-generator 0.09).
Active-days window (median outings ~50, windows 10/20/40/80/160 d): edge recall 0.49/0.59/0.44/0.58/0.57
and ARI 0.59/0.64/0.55/0.58/0.71 — flatter and higher than the old rule's 0.41-0.25 slide; POI and common
places flat (P 0.81-0.84). Number of users (mean of 2 seeds, kept in brackets): 10 (8.5) still no edges;
20 (18) ARI 0.65 at precision 0.90; 40 (33.5) is a bad draw (ARI 0.40, 4.5 edges — single-seed noise);
80 (66) ARI 0.68, AUC 0.89. POI and common-place metrics do not depend on N. Share of users with few
outings (median 60/30/15/8, kept 63/54/44/34 of 80): POI stable 0.81-0.86; relations noisy (ARI
0.55/0.81/0.32/0.82) — the activity rule removes most sparse users before M sees them.

## 5. Map matching on real GPS (first time; `geolife/eval/match_real.py`)

Setup: the Matcher exactly as M configures it (thin 5 s, subsample 60 m, radius 50 m, no postprocess),
over the road graph built from the cached Overture Beijing extract the way the generator builds it
(subtype road, pedestrian classes excluded, largest component: 98,974 nodes / 131,346 edges / 87,190
segments); 8 real users with >= 50k Beijing-core points, 422 outings. Committed aggregate:
`geolife/eval/results/match_real.json`.

* **Every outing matched** (422/422 produce a segment path); 28 segments per outing on average (p50 13).
* **The matched path lies on the trace**: 98.7% (mean; p50 100%) of the matched path's res-9 cells fall
  inside the 1-ring-dilated traversed-cell proxy. The matcher does not hallucinate geometry.
* **Coverage of the cell proxy**: 72% of an outing's proxy cells are covered by the matched path on
  average (p50 80%). By mode: walk 0.81 (n=203), bike 0.64 (184), car 0.62 (12), bus 0.47 (20), train
  0.93 (3). The uncovered remainder is mostly off-graph travel: campus/park footpaths and alleys are
  excluded from the drivable graph, and bus/bike traces drift through blocks the graph does not cover.
* **Fast**: 0.09 s per outing mean (p90 0.24 s) on the full Beijing-core graph, graph+index build 34 s.
  All 18.5k real outings would match in ~30 min single-threaded.
* Matched-path length is shorter than the GPS path length (p50 ratio 0.34): the GPS length includes noise
  path, dwell jitter and off-graph segments; not a matcher defect but worth knowing before using matched
  lengths as a statistic.

**Verdict: M on real data can use the road graph now.** The matcher is reliable (on-trace, complete,
fast); the residual gap vs the cell proxy is graph coverage (walkways), not matching quality. For the
transfer test below the cell proxy is still used on BOTH sides, because the synthetic Beijing side renders
GPS along the same road graph that matching would use, which would make segment agreement circular; the
cell proxy is the harder, non-circular comparison. Running M-on-real with segments is now a switch, not a
project.

## 6. The transfer test: M unchanged on real vs synthetic Beijing (PLAN section 7 item 2)

Setup (`--stages beijing`): real = all GeoLife users, public Beijing places table (7.4k places), 139 users
pass the activity rule; synthetic = 139 users generated into Beijing from `outings_v2_seed0` (33.7M
points), 126 pass. Same tokenizer, same places table, same M configuration, cell proxy both sides.

**Tokenization counts, real side, vs the EDA** (`outings-eda.md`): 139 users (EDA 139), 18,555 real
outings (EDA 18,743; the small deficit is the outings of users tokenized here without label files),
25,536 stops = 1.376 stops/outing (EDA 1.38), 54.7% of stops snap to a place at 150 m (EDA: 48% of stops
within 150 m — the EDA counts all 182 users' stops, here only qualifying users, who skew central).
Synthetic side: 17,344 outings, 1.011 stops/outing, but **93% of synthetic stops snap** — the generator
plants stops at map places; real stops are often away from the 7.4k-place vocabulary. This one mismatch
drives most output-level gaps below.

**M output distributions** (TV on capped histograms; summary statistics real -> synthetic):

| M output | real | synthetic | TV | reading |
|---|---|---|---|---|
| places of interest per user (mean/p50/p90) | 33.7 / 16 / 92 | 54.7 / 31 / 118 | 0.33 | synthetic 1.6-1.9x too many (snap share + distinct-places gap) |
| POI visit-time profile (8 bins, pooled) | peaks morning-afternoon-evening weekday | same shape | **0.075** | transfers well |
| common places count k=2 / 3 / 5 | 2,165 / 1,060 / 433 | 2,649 / 1,609 / 686 | sizes TV **0.118** | counts 1.2-1.6x; size distribution transfers |
| common cells count k=2 (proxy) | 93,691 | 9,314 | sizes TV 0.38 | does NOT transfer: real traversal vocabulary ~10x wider (years of travel, off-road paths) |
| co-visit degree (mean/p50) | 17.1 / 11 | 23.2 / 18 | 0.31 | 1.4x denser (density 0.124 -> 0.185); share of users with a relation 0.87 -> 0.90 matches |
| communities >= 2 (sizes) | 8 (47,26,19,14,9,2,2,2) | 5 (42,29,20,13,9) | 0.63 (few counts) | large-community structure matches; real has extra 2-user pairs |
| users per place histogram | - | - | **0.157** | close (was 0.10 pre-D27) |

Underlying per-user statistics (`compare_beijing`): outings/user mean 133 -> 138 (median 45 -> 72),
active days 76 -> 68, revisit rate 0.40 -> 0.28, distinct places/user 119 -> 183, share of places with
>= 2 users 0.39 -> 0.55. The remaining generator gaps named in PROCESS.md steps 4-5 (distinct places ~2x,
revisit low, points/user 2.4x) map one-to-one onto the failing rows above.

## 7. Split-half reliability (random | time split)

| metric | real Beijing | synthetic Beijing | toy dev (cells) | toy held (cells) |
|---|---|---|---|---|
| POI Jaccard (median) | 0.40 \| 0.30 | 0.33 \| 0.27 | 0.43 \| 0.33 | 0.37 \| 0.26 |
| top POI of A in B's top 3 | 0.78 \| 0.73 | 0.78 \| 0.59 | 0.86 \| 0.71 | 0.70 \| 0.56 |
| B visits inside A's POIs (vs other user) | 0.20 (0.05) \| 0.17 (0.05) | 0.17 (0.03) \| 0.14 (0.03) | 0.26 (0.03) \| 0.21 (0.03) | 0.21 (0.04) \| 0.17 (0.04) |
| common places k=2 Jaccard | 0.41 \| 0.35 | 0.43 \| 0.42 | 0.44 \| 0.42 | 0.44 \| 0.43 |
| common cells k=2 Jaccard | 0.42 \| 0.39 | 0.78 \| 0.75 | 0.93 \| 0.91 | 0.85 \| 0.82 |
| relation edges (n_A, n_B) | 511, 556 \| 508, 475 | 632, 714 \| 687, 482 | 68, 48 \| 50, 73 | 1, 0 \| 0, 0 |
| edge Jaccard | **0.46 \| 0.31** | **0.39 \| 0.30** | 0.29 \| 0.30 | 0 \| undefined |
| edge lift over density | 11.4x \| 9.3x | 6.5x \| 6.4x | 19x \| 19x | undefined |
| community ARI A vs B | 0.48 \| 0.36 | 0.52 \| 0.47 | 0.55 \| 0.55 | undefined |
| AUC of A's p-values for B's edges | 0.91/0.94 \| 0.88/0.84 | 0.89/0.91 \| 0.88/0.87 | 0.96/0.97 \| 0.88/0.97 | undefined |
| new places in B from A-neighbours (vs random) | 0.38 (0.23) \| 0.35 (0.24) | 0.48 (0.31) \| 0.49 (0.29) | 0.34 (0.11) \| 0.26 (0.07) | - |

The headline of the old report — "edge reliability on synthetic data is zero/undefined, the real split-half
numbers cannot be matched" — is resolved **where it matters, the Beijing frame**: with the frozen rule the
synthetic side reproduces its co-visit graph at Jaccard 0.39 vs real 0.46 (random split), with the same
random-over-time ordering, matching ARIs and AUCs, and the same qualitative neighbour-prediction lift.
Real baselines moved vs the old report (0.46 vs 0.38 edge Jaccard) because the rule and the snap radius
changed on the real side too. On the sparse toy draws, halving a user's ~20-50 outings still starves the
statistic: dev halves keep ~50-70 edges (Jaccard 0.29), held-out halves keep ~0 — a sample-size limit of
the half-data regime, not of full-data M (held-out full data keeps 58 edges).

## 8. PROPOSED acceptance thresholds and verdict (NOT settled — awaiting user confirmation; D29)

The thresholds below are a **proposal**; PLAN.md section 11 keeps "acceptance thresholds" an open decision
for the user. Justification: TV thresholds are set at ~2x the TV noise floor seen between same-generator
draws / seeds on dev data (users-per-place TV 0.08-0.12); median-relative-error at 0.5 because the DP
artifact's own seed-to-seed fit spread is 0.13-0.52 (PROCESS.md step 4/D27) — transfer cannot be required
to be tighter than the artifact's own reproducibility; split-half bands at +-0.15 ~ the dev random-vs-time
spread; floors on held-out recovery at roughly 80% of the dev values.

| # | criterion (proposed) | threshold | measured | verdict |
|---|---|---|---|---|
| A1 | held-out planted recovery: POI P and R | >= 0.6 | 0.77 / 0.80 | PASS |
| A2 | held-out: segments/cells k=2 P and R | >= 0.9 | 0.96 / 0.92 | PASS |
| A3 | held-out: edge AUC | >= 0.85 | 0.86 | PASS |
| A4 | held-out: no relation collapse (edges >= 0.5 per kept user, precision >= 0.6) | - | 0.79/user, prec 0.47 | **FAIL** (collapse gone, precision short) |
| B1 | POI visit-time profile TV (real vs syn Beijing) | <= 0.2 | 0.075 | PASS |
| B2 | common-place size distribution TV | <= 0.2 | 0.118 | PASS |
| B3 | users-per-place TV | <= 0.2 | 0.157 | PASS |
| B4 | places of interest per user: TV / median rel. err. | <= 0.2 / <= 0.5 | 0.33 / 0.94 | **FAIL** |
| B5 | co-visit degree TV (and density rel. err.) | <= 0.2 / <= 0.5 | 0.31 / 0.50 | **FAIL** (borderline) |
| B6 | share of users with a relation (abs diff) | <= 0.10 | 0.03 | PASS |
| B7 | common cells (traversal proxy): count ratio in [1/3, 3] | - | 0.10 | **FAIL** |
| C1 | split-half POI Jaccard within +-0.15 of real | 0.40 +- 0.15 | 0.33 | PASS |
| C2 | split-half common places k=2 Jaccard within +-0.15 | 0.41 +- 0.15 | 0.43 | PASS |
| C3 | split-half edge Jaccard >= half of real | >= 0.23 | 0.39 | PASS |
| C4 | split-half community ARI within +-0.2 | 0.48 +- 0.2 | 0.52 | PASS |

**Proposed verdict: M transfers in kind, not yet in volume.** Every output M produces on synthetic data
appears on real data with the same shape (time profiles, size distributions, reliability, ranking quality:
A1-A3, B1-B3, B6, C1-C4 pass — 11 of 15), so a method developed on the synthetic data exercises the same
code paths, the same thresholds and the same failure modes it will meet on the real data. What does not
pass: the per-user volumes (B4, B5, B7 — synthetic users visit ~1.8x too many distinct places and revisit
too little; real traversal is an order of magnitude wider than synthetic), which are the residual
generator gaps already on the books (PROCESS.md steps 4-5 list; snap-flicker issue), and held-out relation
precision (A4: the collapse is fixed but 0.47 < 0.6 against the proxy truth). Awaiting the user: confirm
or move the thresholds, and decide whether B4/B5/B7/A4 block "M transfers" or are accepted as known
generator limits.

## 9. Caveats and unfinished items

* The relation rule was frozen on dev + dev-sparse only (section 2); held-out, Beijing and real results
  are unchanged-M. POI/common thresholds, merge radius 60 m and tokenizer parameters were NOT retuned at
  150 m snap; the old sweep values are kept as-is.
* Relation truth is a proxy (same planted group / co-travel); in hubby artifact-driven data, genuinely
  shared rare places between non-group users count as false positives, so the precision ~0.45-0.70 there
  is a lower bound.
* The cell-proxy common segments (B7) compare different physical things on the two sides (synthetic
  traces live on the road graph; real traces also traverse off-graph space). The matcher now works on
  real GPS (section 5), so the next iteration can compare matched road segments on both sides instead.
* Amsterdam cross-city draw not re-run (no Amsterdam cache on this machine); the old number (POI 0.72/0.71,
  cells 0.98/0.97) predates the new generator.
* Single seed everywhere (artifact seed 0, one split seed); the old report's window/user-count studies
  were re-run only as part of the sensitivity stage. Real-side numbers changed vs the old report partly
  because the snap radius (75 -> 150 m) and the rule changed on the real side too; like-for-like old-rule
  numbers are in the harness JSON (`eval_oldrule`).

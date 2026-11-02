# M v2 (outings): design, planted-truth recovery, real vs synthetic, split-half

Code: `src/geolife/eval/m/outings.py` (M), `src/geolife/eval/m/outings_transfer.py` (harness; `python -m geolife.eval.m.outings_transfer`, real stage
needs `GEOLIFE_POINTS` = points.parquet and `GEOLIFE_PLACES` = places parquet), tests: `tests/test_m2_units.py`, `tests/test_m2_synthetic.py`
(17 tests, about 20 s, synthetic only). v1 M (`m/core.py`, `m/transfer.py`) is untouched and still works (the harness reuses its toy map helpers).
Raw numbers come from one run of the harness (JSON not committed); everything below is transcribed from it.

## 1. Design

Pipeline: traces + map -> `tokenize.outings` (public parameters: gap 2 h, stop 200 m / 10 min, snap 75 m, HMM matcher when a road graph exists)
-> per-user place table -> places of interest, common places/segments, co-visit graph, Louvain.

* **Map-resolution place merge (`merge_m` = 60 m).** The first version failed because the toy city (and Beijing) has several map places per GPS-noise radius:
  only 63% of start, 37% of end and 77% of stop visits snapped to the planted place (the median wrong snap is a neighbour 54 m away). Places are
  merged by a data-independent leader clustering on the public map (visit places by map popularity, absorb places within 60 m). Only pairwise map
  distances are used (translation invariant, tested); truth is scored at the same resolution. POI precision/recall on dev: 0.69/0.67 without, 0.86/0.84 with.
* **(a) Places of interest** per user: place with >= 2 distinct outings on >= 2 local days (stops and outing start/end both count as visits),
  ranked by outings, then days, then dwell; each carries an 8-bin visit-time profile (`DAYTYPES x TOD_BINS`, local UTC+8).
* **(b) Common places**: >= k distinct users, where a user counts if it has >= 2 outings at the place or >= 1 outing with a real stop there
  (endpoint-only single visits are noisy: this rule moved k=2 precision/recall from 0.64/0.95 to 0.88/0.74). **Common segments**: HMM-matched segment ids used by
  >= k users when a graph is given, else res-9 traversed cells. Crossings (intersections) are not computed separately.
* **(c) Relations**: each user's set of places of interest is the item set. Null (popularity only, from the analysed data): user a holds item i with
  probability min(1, |A| m_i / S) (m_i users per item, S = total incidences). The number of shared items is Poisson-binomial under this null; it is matched
  in mean and variance by a binomial (a plain Poisson tail was tried first and wastes power because hubs are near-certain overlaps). Benjamini-Hochberg
  over all user pairs, FDR 0.2 (tuned on dev only). Edge weight = sum of IDF weights `log(N/m_i)` of shared places times `(1 + 0.5 cos)` of the two
  users' time profiles at that place. Louvain on top. An optional second channel (shared segments/cells, Fisher-combined) is off: on a city grid every
  pair shares roads (edge precision 0.29).
* No absolute coordinates after tokenization; no role features; time zone fixed to UTC+8 as decided (v1's night-window estimation is not needed).

Tried and dropped on dev: IDF-weighted overlap statistic (not better), time bonus on edge selection (it only changes weights; no effect on dev ARI because the generator has
no user-specific time preferences, so it is untestable on synthetic data).

## 2. Planted-truth recovery (generate_v2, toy GridCityProvider 40x40, 100 users generated, about 70 kept by the 10-outing/5-day rule)

Truth is restricted to what the trace can show: planted outings that M also segmented, stops with >= 2 trace points inside the dwell, ids at the
same 60 m map resolution. Relation truth: same planted group (pairs), co-travel pairs.

| draw | POI P/R | common places k=2 P/R | k=3 R | k=5 R | segments k=2 P/R | edges: precision (group or co-travel) / recall (group pairs) | edge AUC (group) | community ARI (grouped) |
|---|---|---|---|---|---|---|---|---|
| dev (seed 11, sensor 10 s, noise 6 m) | 0.88/0.84 | 0.87/0.72 | 0.69 | 0.68 | 0.96/0.97 | 0.87 / 0.57 (154 edges) | 0.96 | 0.84 |
| held-out (new map seed, seed 4242, 20 s, noise 15 m, 8 gaps/h, 40% stops unlogged, fewer/shorter outings, concentration 0.3) | 0.74/0.72 | 0.80/0.57 | 0.52 | 0.46 | 0.96/0.95 | 1.00 / 0.01 (2 edges) | 0.87 | undefined (no edges) |
| Amsterdam OvertureProvider (60 users, 44 kept, cells proxy; not tuned) | 0.72/0.71 | 0.80/0.70 | 0.62 | 0.57 | cells 0.98/0.97 | 0.60 / 0.21 (20 edges) | 0.89 | 0.40 |

Cells proxy on the toy city (dev draw): 0.99/0.98, but the proxy saturates there (small city, everybody shares cells).

Honest reading: places, common places and segments transfer with a moderate precision drop. **The thresholded relation graph does not transfer to the held-out
regime**: only 2 edges, because BH over about 2,700 pairs needs p < 1e-4 and the held-out users share on average 3.7 places per same-group pair versus 0.9 for other
pairs (dev: 5.4 vs 0.8). The ranking still carries signal (AUC 0.87), so the failure is the fixed FDR threshold, not the statistic. I did not retune on the held-out draw.

## 3. Sensitivity (dev regime, 80 users generated, seed 21 = not the tuning seeds)

| case | users kept | POI P/R | common k=2 P/R | segments P/R | edges prec / recall | AUC | ARI grouped |
|---|---|---|---|---|---|---|---|
| baseline | 55 | 0.87/0.84 | 0.87/0.68 | 0.97/0.97 | 0.88/0.64 | 0.95 | 0.73 |
| noise 15 m | 55 | 0.85/0.82 | 0.86/0.67 | 0.95/0.98 | 0.87/0.59 | 0.95 | 0.70 |
| noise 30 m | 55 | 0.77/0.76 | 0.80/0.65 | 0.93/0.99 | 0.89/0.50 | 0.94 | 0.72 |
| noise 60 m | 55 | 0.57/0.59 | 0.67/0.59 | 0.92/0.99 | 0.71/0.51 | 0.93 | 0.73 |
| gaps 10/h | 54 | 0.84/0.76 | 0.92/0.57 | 0.98/0.93 | 0.98/0.41 | 0.97 | 0.79 |
| gaps 20/h | 51 | 0.84/0.65 | 0.96/0.37 | 0.98/0.83 | 1.00/0.04 | 0.90 | 0.09 |
| sampling 30 s | 55 | 0.86/0.82 | 0.88/0.69 | 0.97/0.96 | 0.85/0.57 | 0.95 | 0.69 |
| sampling 60 s | 53 | 0.85/0.80 | 0.90/0.65 | 0.98/0.94 | 0.81/0.52 | 0.95 | 0.63 |
| stops unlogged 50% | 55 | 0.81/0.85 | 0.80/0.67 | 0.97/0.97 | 0.83/0.56 | 0.94 | 0.72 |
| stops unlogged 75% | 54 | 0.74/0.84 | 0.71/0.63 | 0.97/0.97 | 0.85/0.55 | 0.95 | 0.72 |

Active-days window (median outings about 50 in all): window 10/20/40/80/160 days gives edge recall 0.41/0.43/0.31/0.25/0.40 and ARI 0.74/0.53/0.51/0.47/0.62;
POI and common places are flat (POI P about 0.86-0.88, R about 0.83-0.86), so the window matters only for relations (noisy; one seed).
Number of users (mean of 2 seeds; kept users in brackets): 10 (8.5): no edges at all; 20 (17): ARI 0.35, edge precision 0.90, recall 0.27; 40 (33): ARI 0.44; 80 (65): ARI 0.43, AUC 0.90.
POI and common-place metrics do not depend on N (common k=5 recall drops to 0.30 at 8 users because few places reach 5 users).
Share of users with few outings (outings-per-user median 60/30/15/8; kept of 80 = 62/55/43/35): POI P/R stays 0.85-0.87/0.83-0.84; relation ARI 0.62/0.74/0.32/0.48, AUC
0.94/0.95/0.92/0.94. The minimum-activity rule removes most users with few outings before M sees them (23 to 45 of 80 dropped), so M itself degrades gently.
Position noise hurts places first (POI precision 0.87 to 0.57 at 60 m); segments stay fine because matching averages the noise.
Dense signal loss hurts recall (gaps 20/h: common-place recall 0.37, edges nearly vanish while AUC stays 0.90).

## 4. Real GeoLife (139 eligible users, places_beijing_big, no road graph so the cell proxy is used; label-free)

Tokenization is identical on both sides (the same function) and M ran with the same configuration, no real-data tuning.

| statistic | real | synthetic dev | synthetic held-out |
|---|---|---|---|
| users | 139 | 70 | 74 |
| outings per user: mean / median / p90 | 133 / 45 / 280 | 51 / 42.5 / 107 | 47 / 27 / 104 |
| stops per outing: mean (p90) | 1.38 (4) | 0.92 (3) | 0.57 (2) |
| outing length km: median / mean / p90 | 9.9 / 30.5 / 48 | 6.5 / 11.6 / 28 | 4.9 / 7.2 / 15 |
| outing duration min: median / p90 | 48 / 309 | 39 / 222 | 32 / 178 |
| distinct places per user: median / p90 | 56 / 271 | 54.5 / 136 | 40 / 141 |
| places of interest per user: median / p90 | 16 / 86 | 19.5 / 49 | 10.5 / 44 |
| stop revisit rate (mean) | 0.43 | 0.42 | 0.31 |
| active days per user: median / p90 | 35 / 167 | 23.5 / 46 | 17.5 / 53 |
| window length days: median / p90 | 107 / 530 | 79 / 275 | 54 / 198 |
| share of places with >= 2 users | 0.40 | 0.50 | 0.44 |
| TV distance, users-per-place histogram (cap 10) | - | 0.095 | 0.052 |
| TV distance, users-per-traversed-cell histogram | - | 0.69 | 0.66 |
| co-visit degree: mean / median / p90 | 20.3 / 13 / 53 | 4.4 / 4 / 9 | 0.05 / 0 / 0 |
| edge density (degree / (N-1)) | 0.147 | 0.064 | 0.0007 |
| share of users with a relation | 0.92 | 0.90 | 0.05 |

Where synthetic and real differ most, and what the generator would have to change:
1. **Heavy tails of user activity.** Real users have 3x more outings on average (p90 280 vs 107), 3x more active days and a p90 window of 530 days; real POIs per user have p90 86
   vs 49. The generator caps outings (`max_outings`) and its lognormals are too light-tailed for the top decile, which is the part that drives overlap. Widen the tails (or fit them from the released tables) and drop the cap.
2. **Outings are longer in reality** (mean 30 km vs 12 km; p90 duration 5 h): the generator's length target and its toy/Amsterdam extent are too small; long train/car outings with several stops are missing. Real stops per outing 1.38 vs 0.92: the generator plants 1.33, but 25% of stops are unlogged and short dwells are lost in the sensor model, so M sees fewer; the unlogged-stop rate looks too high or the dwell distribution too short.
3. **Co-visit density.** Real degree is 4.6x higher (mean 20 vs 4.4) and edge density 0.147 vs 0.064. Real users share places far more than the planted groups produce (real: 40% of places with >= 2 users but heavy hubs: 6 places with 51-67 users).
   The generator needs hub places and broader, overlapping (not disjoint) interest groups.
4. **Traversed-cell sharing** (TV 0.69): the toy city is small, so every user crosses every other user's cells; in Beijing 57% of cells have a single user. Needs a large city (Overture Beijing) with a realistic extent before the segment channel can be compared.
5. Matches: stop revisit rate (0.43 vs 0.42 on dev), users-per-place histogram (TV 0.05-0.10), share of users with a relation on dev (0.90 vs 0.92; this agreement is not informative since relation count depends on the FDR threshold and sample sizes).
   The held-out regime is a deliberately degraded sensor/activity draw and is further from real on every row.

## 5. Split-half reliability (each user's outings split in two halves, M run on each half separately; 'time' = first half vs second half)

POIs need >= 2 outings in a half, so the synthetic halves (about 20 outings each) are much sparser than real halves (about 22 median but heavier tail). "Other user" baselines are about 0 for all rows (Jaccard) or 0.02-0.06 (visit coverage).

| metric | real random | real time | synth dev random | synth dev time | synth held random | synth held time |
|---|---|---|---|---|---|---|
| POI Jaccard A vs B (median over users) | 0.43 | 0.30 | 0.33 | 0.21 | 0.22 | 0.19 |
| user's top POI of A in B's top 3 | 0.87 | 0.66 | 0.59 | 0.43 | 0.47 | 0.40 |
| B visits inside A's POIs (vs other user's) | 0.21 (0.06) | 0.18 (0.06) | 0.22 (0.03) | 0.18 (0.03) | 0.13 (0.02) | 0.12 (0.03) |
| common places k=2 Jaccard | 0.44 | 0.40 | 0.49 | 0.45 | 0.40 | 0.38 |
| common places k=3 Jaccard | 0.51 | 0.45 | 0.54 | 0.47 | 0.46 | 0.40 |
| common cells Jaccard (k=2) | 0.42 | 0.39 | 0.92 | 0.91 | 0.88 | 0.88 |
| relation edges (n in A, n in B) | 448, 493 | 526, 420 | 0, 32 | 35, 1 | 0, 0 | 0, 0 |
| edge Jaccard | 0.38 | 0.25 | 0 | 0 | undefined | undefined |
| P(edge in B given edge in A) / lift over density | 0.57 / 11x | 0.36 / 8x | undefined | 0 | undefined | undefined |
| AUC of A's p-values for B's edges | 0.87 / 0.93 | 0.86 / 0.80 | 0.93 (one direction) | 0.78 / 0.85 | undefined | undefined |
| community ARI A vs B | 0.40 | 0.29 | undefined | -0.00 | undefined | undefined |
| new places in B visited by A-graph neighbours (vs random users) | 0.37 (0.28) | 0.32 (0.26) | undefined | 0.24 (0.16) | undefined | undefined |

Findings:
* On real data M is reliable against itself: POIs reproduce (Jaccard 0.43 random, 0.30 time; much better than the EDA's raw res-8 cell sets, 0.21/0.14), common places reproduce at 0.44-0.51, and the co-visit graph
  reproduces at 11x chance (random split) and 8x (time split). Time splits are consistently weaker than random splits (drift), as in the EDA.
  The relation graph generalises to held-out visits modestly: new places in B are 1.3x more likely to have been visited by A-neighbours than by random users (0.37 vs 0.28).
* Edge reliability on synthetic data is undefined or zero because the half-sample statistics do not clear the FDR threshold; this is a synthetic-side weakness (planted structure is weaker and users have fewer outings), so the real
  split-half numbers can not yet be matched by the generator. The AUCs (0.78-0.93) show the ranking is reproducible even then.
* The time-profile bonus changes only edge weights: on real data community ARI between halves 0.38 with the bonus (1.0) vs 0.40 without. No evidence that it helps.

## 6. Caveats and unfinished items

* Tuned on dev only (seeds 11, 12 with 80 users; sweeps of merge radius, FDR level, POI/common thresholds). The sensitivity study uses seed 21 under the dev sensor; the held-out and Amsterdam draws were not used for tuning. Held-out relation results are bad as reported.
* Tokenization parameters (gap, stop rule, 75 m snap) are the public choices from the EDA (informed by real data, ledger caveat), not tuned here. The 60 m merge radius was chosen from the synthetic failure analysis on dev; on the sparse Beijing table (7.4k places) it merges little.
* Relation truth is "same planted group / co-travel", which is a proxy: a significant overlap of popular places between non-group users is counted as a false positive. Recall for group pairs is capped because members share few places.
* Amsterdam run has no road graph (cells proxy) and small N; Overture Beijing with a road graph was not run (real segments: cell proxy only). Real-data crossings are therefore not evaluated.
* The split-half statistics use one random split (seed 0) and a single dev/held draw; no confidence intervals. Single-seed noise is large for relation metrics (window study non-monotone).

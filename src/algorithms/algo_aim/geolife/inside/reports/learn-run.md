# Learner run on real GeoLife (agent TOK)

Command: `python -m geolife.inside.learn --out geolife/transfer/artifacts/stays_v0_eps6_seed0 --epsilon 6 --delta 1e-6` (data via `GEOLIFE_DATA_DIR`).
Artifact (gitignored) passes `load_artifact` checksums and `validate_no_raw_leakage`. Ledger: user-level adjacency,
rho = 0.5395 (epsilon = 6.000 at delta = 1e-6), 10 Gaussian tables with an equal rho split (0.054 each).
True values below come from a local noiseless re-run for evaluation only; they are not in the artifact.

## Setup

- Region: Beijing core box lat 39.75-40.10, lon 116.15-116.60 (public choice). Points thinned to one per 10 s, speed-filtered (70 m/s).
- Stays: 200 m / 20 min, adjacent stays at the same place merged if gap <= 30 min. Places: no Overture places parquet was ready, so
  the stand-in is H3-only places (res 8, 2,838 public cells covering the box, category `unknown`). The places path is configurable (`--places`).
- Caps per user: 100 stays, 100 trips, 20 distinct places; tables clip further (see `dp.build_count_table`).
- Result of tokenization: 156 users with stays (of 179 in the box), 7,156 stays (4,713 after caps; 40 users truncated), 1,890 trips
  (1,561 after caps). Median 15 stays per user, median dwell 0.54 h, median trip 1.8 km. Trip modes (labels for 68 users, else speed):
  car 666, bike 662, walk 472, train 58, bus 32. (The EDA's 16,481 stays was for all of China with 30 s thinning and no merging; here the box
  and the merging cut it.)
- Map matching was **off** in this run (24 M GPS points through a pure-Python HMM is not feasible and the Overture graph covers only a 4 x 4 km window). Trip length = GPS polyline length; the `detour` table is **circuity** (path / straight-line), not detour vs. shortest path.

## Which tables survive (sigma = Gaussian std; counts are users or events)

| table | sigma | keys (nonzero) | verdict | evidence |
|---|---|---|---|---|
| n_users | 3.0 | 1 | survives | 156 +/- 3 (2%) |
| role_users | 5.3 | 3 | survives | all keys > 3 sigma, L1 distance 0.013 |
| return_rate | 215 | 2 | survives (coarse) | new 1534 vs return 2526, 5% error |
| anchors_other | 3.0 | 11 | mostly survives | 4 of 11 bins > 3 sigma hold 74% of mass; tail bins are noise |
| dwell_hist | 32 | 24 (21) | partly | 10 keys > 3 sigma hold 87% of mass; L1 0.12 |
| trip_dist | 26 | 14 | partly | 5 keys hold 82%; L1 0.10 |
| detour (circuity) | 22 | 6 | partly | 5 keys hold 90%; L1 0.09 |
| role_trips | 27 | 9 | partly | 3 keys hold 61%; L1 0.14 (home-work-other structure visible only in the biggest cells) |
| arrival_hour | 45 | 72 | weak | 4 keys > 3 sigma hold 17% of mass; hourly shape is blurred, coarse day-part would survive; L1 0.27 |
| **place_role_users** | 8.6 | 8,514 (4,423) | **noise** | **0 keys above 3 sigma**; max true count 23 users; corr(top-10 true, noisy) = -0.06; L1 0.94 |

## The decisive finding: place-level overlap at user-level DP

`place_role_users` (distinct users per place and role) is the table that carries "shared anchors", and it does not survive at H3 res 8.
Only 41 of 756 visited cells have >= 10 distinct users and 8 have >= 25, but with at most 8 keys per user the noise is 8.6, so you need
roughly 26+ users per (place, role) to clear 3 sigma. Same data, same budget, coarser cells:

| H3 res | cells in vocab | keys above 3 sigma | share of true mass above 3 sigma | corr (top-10) | L1 distance |
|---|---|---|---|---|---|
| 8 | 2,838 | 0 | 0% | -0.06 | 0.94 |
| 7 | 415 | 3 | 13% | 0.57 | 0.76 |
| 6 | 56 | 7 | 43% | 0.90 | 0.41 |

So: with 156 users, the structure of **who shares which place** is recoverable only at ~30-100 km2 cells (res 6-7), i.e. district level, not at
POI/neighbourhood level. The behavioural tables (dwell, trips, distance, circuity, return rate, role structure) are the usable core; the
place-overlap table needs a coarse hierarchy (res 6 first, refine only above a noisy threshold via `dp.hierarchical_release`) or a larger rho for that table.
Note that the equal rho split is wasteful: n_users, role_users and return_rate are overpaid and `place_role_users` is underpaid. A weighted split is the first thing to try.

## v1: weighted rho split + hierarchical place tier (agent LEARN2)

Command: `GEOLIFE_DATA_DIR=... python -m geolife.inside.learn --out geolife/transfer/artifacts/stays_v1_eps6_seed0 --epsilon 6 --delta 1e-6 --places places_beijing_big.parquet`
(Overture-derived Beijing places, 7,401 in the box + 2,838 H3 res-8 cells = 10,239 public place ids). Same data, caps, tokenization and
privacy target as v0: user-level, rho = 0.5395, epsilon = 6.000 at delta = 1e-6. Artifact in `geolife/transfer/artifacts/stays_v1_eps6_seed0/` (gitignored); it
passes `load_artifact` checksums and `validate_no_raw_leakage`. `--flat` reproduces v0 (flat table, equal split).

**What changed**
- Rho split (`generate/measure.py: rho_weight, weighted_rho_split`): 40% of rho split equally (floor), 60% by the public weight
  `priority * (l2/clip)^2 * ln(vocab)` (equalises clip / detection-threshold across tables; sensitivity and vocabulary size only, never counts).
  Place tiers have declared priorities 6 / 3 / 1 (coarse to fine) and the shared-anchor histogram 6. A first attempt without the floor starved
  `n_users` (sigma 31) and `return_rate` (sigma 1562 vs total 4,050), so the floor was added after seeing that run; both are declared design choices.
- Place tier: fixed public hierarchy H3 res 6 -> 7 -> 8 -> Overture place (res 6 -> 8 taken from the brief before looking at v1 SNR). Res 6 (76 cells x 3 roles)
  is released fully; res 7 only under (cell, role) parents whose noisy count >= 2 sigma, the res-8 / place level only under kept res-7 parents
  (same logic as `dp.hierarchical_release`, chained over three levels). Max keys per user 6 / 8 / 8. Tiers: `place_role_users_r6`, `_r7`, and the
  legacy name `place_role_users` for the finest tier. `fit_behaviour` reads all tiers (new params `coarse`, `place_tiers`), and falls back
  to the coarse tier when no refined tier survives.
- New shared-anchor measurement `work_users_hist` (D2): number of res-6 cells having 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+ users with a work anchor there. L2 sensitivity sqrt(2).
  In the fitted params as `work_users_per_cell`.

**SNR, v0 (equal split, flat, H3-only; table above) vs v1** (sigma; keys above 3 sigma; share of true mass above 3 sigma; L1 = TV distance of normalised noisy vs true)

| table | v0 sigma | v1 sigma | v0 mass > 3 sigma | v1 mass > 3 sigma | v1 L1 |
|---|---|---|---|---|---|
| n_users | 3.0 | 5.4 | 1.00 | 1.00 | 0.00 |
| role_users | 5.3 | 9.1 | 1.00 | 1.00 | 0.02 |
| anchors_other | 3.0 | 5.3 | 0.74 | 0.71 | 0.11 |
| dwell_hist | 32 | 39 | 0.87 | 0.68 | 0.20 |
| trip_dist | 26 | 37 | 0.82 | 0.71 | 0.11 |
| role_trips | 27 | 39 | 0.61 | 0.51 | 0.17 |
| detour | 22 | 35 | 0.90 | 0.45 | 0.21 |
| arrival_hour | 45 | 41 | 0.17 | 0.34 | 0.28 |
| return_rate | 215 | 381 | 1.00 | 1.00 | 0.04 |
| place tier, coarse (v0: res-6 diagnostic at equal split; v1: `place_role_users_r6`) | 8.6 | 5.2 | 0.43 (7 keys) | **0.66 (13 keys)** | 0.37 (v0: 0.41) |
| place tier, mid (res 7) | 8.6 | 6.4 | 0.13 (3 keys) | **0.42 (8 keys)** | 0.36 (v0: 0.76) |
| place tier, finest (res 8 + places; v0: `place_role_users`) | 8.6 | 8.4 | 0 (0 keys) | 0 (0 keys) | 0.77 (v0: 0.94) |
| work_users_hist (new) | - | 5.9 | - | 0.70 | 0.22 |

(A same-data check with Overture places and the v0 recipe, `--flat`: finest tier sigma 8.6, 0 keys above 3 sigma, top-10 correlation 0.23, L1 0.97; the largest true place has 11 users.)

Place tier in v1: res 6 holds 13 keys above 3 sigma (top-10 correlation 0.97). In the fitted params the coarse tier shows 2 home cells (53 and 19 users), 1 work cell (40) and 9 other cells (93 ... 17 users);
about 41% of work-holders and 46% of all users sit in a significant district. The finest tier has 1,728 released keys (under kept res-7 parents), but its largest true count is 15 users and one
'other' key passes the multiple-testing cut; homes and work places are not individually recoverable.

## What is still noise (honest statement)

- **Place-level overlap (res 8 / Overture place) is still noise** at epsilon 6, user-level, 156 users: the largest true place count is 11-15 users, far below the ~3 sigma = 25 users needed.
  Hierarchy refinement does not change that, it only avoids paying noise on cells that cannot be seen. The shared-anchor structure that is recoverable is district-level (res 6-7, ~30-100 km2):
  who shares which district, plus the distribution of work-user counts per district (`work_users_hist`; its low bins are noisy at sigma 5.9 on 76 cells, only the bulk shape is trustworthy:
  about 64% of the cells have 0 work users after denoising).
- **Zipf popularity of "other" places** is fitted from at most a few significant keys, so `zipf_s` falls back to its default 1.0 on this run. Treat place-popularity exponents as prior, not measured.
- **Weighted split trades fidelity of the behaviour tables** (see table): sigma went up 20-60% on n_users, role, dwell, trip and detour tables and `return_rate` has ~9% relative noise. `arrival_hour` is
  still weak (corr(top-10) 0.02, 34% of mass above 3 sigma): use day-parts, not hours. Detour (circuity) lost half its above-3-sigma mass.
- Budget shares at v1 (rho): res 6 0.110, res 7 0.099, finest 0.056, histogram 0.029, arrival 0.066, dwell 0.035, other tables 0.017-0.027 each. About 0.056 on the finest tier buys no usable signal; a future run could drop that tier and give the rho to the coarse tiers.
- Shared-anchor ties are not measured below district level; M's "two users share a work place" will be reproduced at the synthetic level only through the planted structure plus the district-level counts.

## Matcher (synthetic noisy paths on the real Overture Haidian graph, 3,483 nodes / 4,904 edges, largest component)

Paths: random shortest paths >= 1.5 km, points every 25 m, 8 m Gaussian noise, 20 paths per row. "Recovered" = share of true segment ids matched; "extra" = matched ids not in the truth, per true segment (same convention as P0: 19/27 recovered, 33 extra, i.e. 70% / 122%).

| setting | recovered | extra | true length covered (15 m) | matched length correct (15 m) |
|---|---|---|---|---|
| old matcher, drivable-class paths | 76% | 68% | 98% | 67% |
| **new matcher, drivable paths, mode=car** | **72%** | **29%** | 97% | **83%** |
| old matcher, any-road paths (incl. footways) | 73% | 59% | 95% | 77% |
| new matcher, any-road paths, mode unknown | 53% | 38% | 78% | 81% |

Reading: for vehicles the new matcher halves the extras (68% -> 29%) at -4 points of recovery and raises geometric precision from 67% to 83%. For paths that run on footways with unknown mode the new matcher loses recall (53%) because it collapses footways into the parallel road by design; use `mode="walk"` for walkers (collapse off) and the label/speed mode otherwise. Grid over sigma/beta/subsample (car, drivable): best was sigma 8, beta 40, subsample 15-30 m; defaults are now sigma 10, beta 40, subsample 15 m. Caveat: tuned on synthetic noise (Gaussian, 8 m), real GeoLife errors are heavier-tailed. Remaining extras are mostly short intersection pieces at turns. One-way awareness works only when edges carry `oneway_from` (the stock graph does not; unit-tested on a grid).

## Honest caveats

- v1 (hierarchy, weighted split, priorities, floor) was designed after the v0 equal-split run on this same real data and tuned once more after a first weighted run (see v1 section); these data-dependent design choices are **not accounted** in epsilon.
- Hyperparameters (stay thresholds, caps, H3 res, matcher) were chosen from synthetic data and the EDA, but I looked at SNR on the real run and re-ran with res 7/6; that comparison is a local diagnostic, and choosing a resolution from it is data-dependent tuning that is **not accounted** in the epsilon (it should be fixed a priori or paid for).
- The vocabulary contains H3 cells of the source city, so the artifact is not geography-free; teleportation is provenance obfuscation, not DP.
- Stays are a deterministic function of one user's own points; the cap + clip bound the user-level sensitivity. Uneven users (max 1,217 stays) are cut hard by the caps (34% of stays dropped), which biases tables toward light users.
- Timestamps are assumed UTC; role/hour uses a fixed UTC+8 offset.

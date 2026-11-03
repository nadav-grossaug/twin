# DP learner v2 (outings): real-data SNR report (T2)

Command (T1 outings built in-process, since the T1 parquet export has no traversed cells):
`python -m geolife.inside.learn_v2 --outings/--points ... --out geolife/transfer/artifacts/outings_v2_eps6_seed0 --epsilon 6 --delta 1e-6`
(artifact is gitignored). Input: T1 `build_outings` on all 182 users, 139 qualifying users, 18,555 real outings, public places `places_beijing_big` (9 categories) and the 56 res-6 cells of the Beijing core box. User-level add/remove adjacency; rho = 0.5395, epsilon = 6.000 at delta = 1e-6; 24 tables, one seed. SNR is computed locally against the noiseless tables on the same released keys (never leaves the air-gap).

## Design (what changed vs v1)
- Per-user vectors with exact, unit-tested bounds (`tests/test_learn_v2_sensitivity.py`, property test over random user removal): presence tables sqrt(m); pmf tables (user-level mixture, counts L1-normalised) 1; user histograms one-hot, 1. Shape and time tables therefore cost far less noise than clipped event counts.
- Caps: 100 outings (seeded random subsample), 50 res-8 stop cells, 1000 res-8 traversed cells. Per-table key caps m: place 12/25/50 (res 6/7/8), traversal 20/100/300/300 (res 6/7/8/9).
- Hierarchical release (children only under parents with noisy value >= 2 sigma, r9 >= 3 sigma), weighted rho split with 40% equal floor, priorities declared before any SNR.

## Cap recheck on real data (139 users, uncapped, local)
| quantity | p50 | p90 | p99 | cap / m | users above |
|---|---|---|---|---|---|
| outings | 45 | 280 | 1367 | 100 | ~32% |
| res-8 stop cells | 35 | 172 | 719 | 50 | ~40% |
| traversed res-8 cells | 362 | 3981 | 17581 | 1000 | ~30% |
| keys per user, place r6 / r7 / r8 | 9 / 16 / 27 | 15 / 30 / 50 | | m 12 / 25 / 50 | 22% / 22% / 0% |
| keys per user, trav r6 / r7 / r8 / r9 (after 1000 cap) | 16 / 54 / 164 / 492 | 32 / 130 / 424 / 1291 | | m 20 / 100 / 300 / 300 | 38% / 22% / 25% / 65% |
The m values keep the median user whole at coarse levels and truncate the heavy tail; r9 is truncated for most users, but it is only released under busy r8 parents. No re-tuning was done after this check (caveat in the ledger).

## SNR per table at epsilon 6 (sigma, share of true mass above 3 sigma, top-10 correlation, TV distance of normalised noisy vs true)
| table | keys | sigma | true total | mass > 3 sigma | corr top10 | TV | verdict |
|---|---|---|---|---|---|---|---|
| n_users | 1 | 7.4 | 139 | 1.00 | - | 0.00 | survives |
| outing_stops | 11 | 7.4 | 139 | 0.76 | 0.96 | 0.12 | survives |
| outing_mode | 6 | 7.4 | 139 | 0.68 | 0.96 | 0.11 | survives (labelled subset, mostly unknown) |
| outing_duration | 9 | 7.4 | 139 | 0.57 | 0.89 | 0.10 | survives |
| outing_length | 9 | 7.4 | 139 | 0.54 | 0.81 | 0.16 | survives |
| outing_detour | 7 | 7.4 | 135 | 0.47 | 0.89 | 0.10 | survives (proxy) |
| stop_hop | 8 | 7.4 | 138 | 0.51 | 0.90 | 0.22 | survives |
| stop_dwell | 10 | 7.4 | 138 | 0.32 | 0.74 | 0.24 | partly |
| user_places_hist | 11 | 7.4 | 139 | 0.58 | 0.88 | 0.14 | survives |
| user_span_hist | 10 | 7.4 | 139 | 0.45 | 0.80 | 0.22 | survives |
| revisit_hist | 10 | 7.4 | 138 | 0.33 | 0.80 | 0.13 | survives |
| user_outings_hist | 11 | 7.4 | 139 | 0.00 | 0.84 | 0.19 | survives (flat, mass spread) |
| cat_time_visits | 80 | 7.4 | 139 | 0.16 | 0.81 | 0.52 | marginals only; category mix collapses to unknown |
| cat_transitions | 100 | 7.4 | 138 | 0.40 | 0.93 | 0.68 | weak |
| outing_start (24h x 2) | 48 | 7.4 | 139 | 0.00 | 0.52 | 0.49 | noise: fallback |
| place_time_users | 448 | 7.3 | 139 | 0.00 | 0.91 | 0.78 | noise |
| stop_transitions_r6 | 3136 | 7.3 | 138 | 0.30 | 0.79 | 0.91 | noise |
| place_users_r6 | 56 | 16.0 | 1129 | 0.52 (7 keys) | 0.95 | 0.25 | survives |
| place_users_r7 | 112 | 20.0 | 1868 | 0.26 (5) | 0.89 | 0.34 | partly |
| place_users_r8 | 105 | 29.5 | 1597 | 0.19 (3) | 0.57 | 0.37 | mostly noise, 3 hubs |
| trav_users_r6 | 56 | 21.8 | 2029 | 0.56 (12) | 0.84 | 0.21 | survives |
| trav_users_r7 | 175 | 27.0 | 6665 | 0.28 (18) | 0.76 | 0.23 | partly |
| trav_users_r8 | 413 | 34.3 | 13018 | 0.05 (5) | 0.23 | 0.29 | noise |
| trav_users_r9 | 154 | 42.5 | 6581 | 0.02 (1) | 0.04 | 0.31 | noise |

(The mass-above-3-sigma metric understates pmf tables: mass is spread across buckets of about 14 users each against sigma 7.4, so the shape is readable but per-bucket detail is coarse. Every pmf table got only the floor share, about 0.9% of rho.)

## What survives
Outing shape, user-level histograms and the coarse spatial tables (place r6, r7; traversal r6, r7). Fitted params for those come from released values. Place overlap at res 8 and traversal at res 8/9 do not clear the noise: a cell needs about 90-130 distinct users to pass the multiple-testing cut, while the largest has 107-131 of 139. Co-visit at place or segment level is therefore not recoverable from DP tables at this budget; only hubs and district structure are.

## Fell back to documented defaults (params['diagnostics']['fallbacks'])
- `outing_start` (24h x daytype): start_pmf stays uniform. The tod x daytype profile is not recoverable at 48 keys.
- `time.cat_profile`: global (uniform) profile for 8 of 10 categories; `places.cat_mix` collapses to {unknown: 1.0}.
- `places.users_per_place_hist`: default; only the tail is observed (floor 90 users), exposed as `places.users_per_place_tail`.
- `places.pool_size`: default; `zipf_s` is fitted on the res-6 tier (cells, not places) and flagged.
- `distance_decay` hit its grid bound 3.0 (lower bound only).
- `groups`: defaults (no co-visit table; the neighbour-capped degree histogram was not released because its sensitivity depends on other users).
- Sensor statistics: not measured (UserOutings has no raw points); generator defaults, flagged in the ledger.

## Compared with v1 (learn-run.md)
v1 coarse place tier (res 6, per role): sigma 5.2, 13 keys above 3 sigma, 66% of mass; finest tier: 0 keys, 0%. v2 res 6 has sigma 16 (m = 12 distinct cells per user instead of 6, and 24 tables share the budget) with 7 keys, 52%, so it is weaker per table; the finest tier now has 3 keys above the cut (v1: 0) and the new traversal tables give 12-18 coarse cells above 3 sigma. The useful new content is the outing-shape and user-level histograms, which have no v1 counterpart and are mostly readable.

## Findings and suggestions
1. Pure-noise tables cost budget: `place_time_users` (448 keys), `stop_transitions_r6` (3136), `outing_start` (48) and `trav_users_r8/r9` together take a large share. Dropping or coarsening them (e.g. outing start in the 8 tod x daytype bins instead of 48 keys, transitions at res 5 or by category only) and re-spending on place and shape tables is the obvious next step; I did not do it because that is tuning after seeing the SNR and would need its own ledger caveat.
2. A larger floor lowers pmf sigma only marginally; sigma for sensitivity-1 tables is about 7 for any single table at about 1% of rho. Fewer, larger-weight tables help more.
3. If T3 needs time-of-day structure, take it from public reasoning or a single 8-bin table with its own budget.

## Caveats (in the ledger as well)
Parameters informed by real data (gap, stop rule, real-outing rule, min activity, caps, bins, hierarchy, tod bins), bin edges and per-table key caps chosen after looking at the EDA, declared priorities and floor, floating-point Gaussian sampling, per-user determinism of tokenization assumed, detour is a proxy, mode labels cover a subset. The fallback defaults in `fit_v2.DEFAULTS` are round-number priors written after reading the EDA report. One run, one noise seed: SNR numbers are single draws.

---

# v2b: noise-only tables removed, coarse time design (T2b)

Same data, command and budget as above (139 users, 18,555 outings, epsilon 6, delta 1e-6, rho 0.5395, seed 0, one noise draw). Code: `generate/{schema_v2,measure_v2,fit_v2}.py`, `types.py` (v2 block: `TOD_COARSE`, `local_coarse_bin`, `tod_coarse`).

## What changed
- **Dropped (not released any more):** `place_time_users`, `stop_transitions_r6`, `trav_users_r8`, `trav_users_r9`. Rule: table with ~0 mass above 3 sigma and a "noise" verdict in the report above (mass > 3 sigma 0.00 / 0.30 with TV 0.91 / 0.05 / 0.02). Nothing in `fit_behaviour_v2` read them, so no parameter newly falls back. Traversal now stops at res 7 (24 -> 20 tables).
- **Coarse time:** `outing_start` goes from 24 h x 2 (48 keys) to am x pm x weekday/weekend (4 keys); `cat_time_visits` from 4 to 2 time bins (80 -> 40 keys). am = local [05,14), pm = local [14,05), so the 08h and 17-18h peaks fall in different bins.
- **Params stay on the 4 `TOD_BINS`:** each coarse bin's mass is split across its fine bins by a FIXED shape (`schema_v2.COARSE_TO_FINE`): proportional to the number of local hours of the coarse bin inside each fine bin (am: night 1/9, morning 6/9, afternoon 2/9; pm: night 5/15, afternoon 4/15, evening 6/15). This within-bin shape is an assumption, not measured; the released data cannot resolve it. Noted in `diagnostics.fallbacks.start_time_fine`.
- Freed rho re-spent by the unchanged weighted-split rule (public quantities, 40% floor). Total unchanged: epsilon 6.000 at delta 1e-6.
- Ledger caveat added: table list and coarse bins were chosen after looking at signal-to-noise of a previous real-data run, so the final mechanism's DP guarantee is conditional on a data-dependent design choice, outside the formal guarantee (test `test_data_dependent_design_caveats_present`).

## SNR before -> after (mass above 3 sigma / top-10 corr / sigma)
| table | before | after |
|---|---|---|
| outing_start | 0.00 / 0.52 / 7.4 (48 keys) | **0.89 / 0.94 / 6.8** (4 keys) |
| cat_time_visits | 0.16 / 0.81 / 7.4 | 0.42 / 0.93 / 6.7 (TV 0.52 -> 0.32) |
| cat_transitions | 0.40 / 0.93 / 7.4 | 0.40 / 0.95 / 6.6 (TV 0.68 -> 0.64, still weak) |
| outing_stops | 0.76 / 0.96 / 7.4 | 0.76 / 0.97 / 6.7 |
| outing_duration | 0.57 / 0.89 | 0.57 / 0.91 |
| outing_length | 0.54 / 0.81 | 0.70 / 0.84 |
| outing_mode | 0.68 / 0.96 | 0.86 / 0.95 |
| stop_dwell | 0.32 / 0.74 | 0.47 / 0.85 |
| stop_hop | 0.51 / 0.90 | 0.51 / 0.94 |
| user_outings_hist | 0.00 / 0.84 | 0.30 / 0.55 |
| user_places_hist | 0.58 / 0.88 | 0.58 / 0.79 |
| user_span_hist | 0.45 / 0.80 | 0.60 / 0.92 |
| place_users_r6 | 0.52 (7 keys) / 0.95 / 16.0 | **0.71 (12) / 0.94 / 11.3** |
| place_users_r7 | 0.26 (5) / 0.89 / 20.0 | 0.27 (6) / 0.93 / 13.7 |
| place_users_r8 | 0.19 (3) / 0.57 / 29.5 | 0.22 (5) / 0.84 / 20.4 |
| trav_users_r6 | 0.56 (12) / 0.84 / 21.8 | 0.62 (14) / 0.85 / 15.7 |
| trav_users_r7 | 0.28 (18) / 0.76 / 27.0 | **0.52 (42) / 0.47 / 17.7** |
Sigma for presence tables drops 25-45% (place r6 16.0 -> 11.3, r8 29.5 -> 20.4); pmf sigma only 7.4 -> 6.7 (the 40% floor dominates their share, as noted before). Single noise draw: differences of a few points (e.g. `user_outings_hist` corr 0.84 -> 0.55, `user_places_hist` 0.88 -> 0.79) are within draw-to-draw variation and should not be read as regressions or gains. `trav_users_r7` top-10 corr fell (0.76 -> 0.47) while mass above 3 sigma rose; its top cells are now many near-equal hubs.

## Do the time features now carry signal?
Partly. Coarse start times: true (am, pm) users-mass weekday 46.6 / 56.0, weekend 15.0 / 21.4; released 42.1 / 56.3 and 21.4 / 34.9 (sigma 6.8). So the am/pm split and the weekday/weekend split (true 73/27) are readable (3 of 4 cells above 3 sigma = 20), but the am share is close to the 37.5% that uniform-over-hours would give, and the weekend cells are within noise of each other. Fitted `start_pmf` (weekday): morning 0.285, afternoon 0.248, night 0.238, evening 0.229; weekend ~ uniform (0.25 each). That is, weekday morning gets a small lift and the 17-18h evening peak seen in the EDA is NOT recovered in the fine bins (the pm bin is flat by construction). The within-bin shape is the assumed one. `cat_time_visits` is still at signal 2.7 < 3 total mass in noise units, so `cat_mix` collapses to {unknown: 1.0} and `cat_profile` stays global/default: category x time is not recoverable at this budget, even at 40 keys.

## Still noise / still falling back to defaults
`cat_mix`, `time.cat_profile`, `cat_transitions` (TV 0.64), `users_per_place_hist` (res 8: 5-7 keys above floor of ~66 users, only the tail), `pool_size`, `zipf_s` (from res-7 cells, not places), `distance_decay` (grid bound 3.0), `groups`, sensor statistics, and the fine time-of-day shape. Place res 8 remains mostly noise except a handful of hubs; traversal beyond res 7 is no longer attempted.

## Integration (fit_behaviour_v2 on the new artifact -> generate_v2 on the toy grid city, 60 users)
Time-related planted checks pass on all three noise seeds (`start_tod_daytype_tv` 0.02-0.04 vs tol 0.1; `weekend_share_of_starts` within 0.01 of target), as do outings per user, stops per outing, `n_shared_places` and `group_visit_share`. The check as a whole does NOT pass: `stop_revisit_rate` (0.60-0.70 vs target 0.45-0.49, tol 0.12), `users_per_place_mean` (2.6-2.8 vs 1.78) and `share_places_ge2_users` (0.56-0.57 vs 0.33) fail on every seed (`distinct_places_per_user_median` fails on seed 0 only). The baseline (pre-change code, same seed) also fails `users_per_place_mean` plus two category checks that need a category profile; its other checks pass. So the failures in the place-overlap statistics are not caused by this change (a generator/default-prior issue on the toy city, to be looked at by T3), but fitted values vary a lot between noise draws (fitted stops-per-outing mean 1.2 to 2.1, outings-per-user median 37 to 67), because the shape tables get only floor-level budget.

## Caveats
Everything in the caveats above, plus: the table list and coarse bins were chosen after looking at this run's predecessor on the same real data (data-dependent design, outside the formal DP guarantee); this run is also the one reported here, so its SNR numbers are for the already-selected design. Within-coarse-bin shape is assumed. One seed for the SNR table (seeds 0-2 only for the integration check).

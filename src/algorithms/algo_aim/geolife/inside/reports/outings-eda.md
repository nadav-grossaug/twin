# Outings EDA on real GeoLife (T0)

Source: `geolife/inside/notebooks/02_outings_eda.ipynb` (built by `build_02_outings_eda.py`, helpers in `src/geolife/inside/eda_outings.py`, tests in `tests/geolife/test_eda_outings.py`). All 182 users, 24.9M raw points, 24.1M after cleaning (invalid coordinates, duplicate timestamps, isolated >70 m/s spikes). No raw data or real user ids are stored.

**Time.** GeoLife timestamps are GMT. All hour/weekday numbers use local time = UTC+8. The generator/artifact must declare which convention it uses (recommended: local hour in tables, UTC only in rendered timestamps).

**Ledger caveat.** Everything below that selects a parameter was chosen after looking at real data. Those choices are public-parameter tuning on the sensitive data and must be listed in the ledger caveats (marked "informed" in section 7).

## 1. Outing definition

- Gaps between consecutive files of a user: median 11.2 h, p10 38 min, p90 37 h; only 12% are under 1 h, 16% under 2 h, 21% under 4 h; 15% exceed one day.
- Gaps inside files (dt > 60 s): 6.5 per file, 2.1 over 5 min, 0.39 over 1 h, 0.22 over 2 h, 0.07 over 4 h. These are mostly signal loss, not separate trips.
- Outings by gap threshold (an outing merges files and splits files at the threshold):

| gap | outings | real (>=5 min, >=500 m) | median duration | median length | median points | users with >=10 real | top 10% users' share of real outings |
|---|---|---|---|---|---|---|---|
| 30 min | 28,862 | 26,014 | 31 min | 5.2 km | 396 | 147 | 60% |
| 1 h | 23,762 | 22,159 | 38 min | 7.0 km | 449 | 143 | 60% |
| 2 h | 19,746 | 18,743 | 45 min | 9.0 km | 509 | 139 | 59% |
| 4 h | 16,059 | 15,385 | 57 min | 11.4 km | 592 | 130 | 59% |

- At 2 h: p90 duration 5 h, p90 length 46 km; 9.5% of outings span more than one file.
- Mode (labelled subset only: 69 users, 20% of points; 4,197 outings with >=50% labelled points, 71% pure at >=80%): walk 1,158 outings (median 39 min, 4 km), bike 1,037 (22 min, 3.8 km), bus 834 (66 min, 12.9 km), car/taxi 794 (44 min, 18.6 km), rail 368 (44 min, 18.4 km).
- Per user (real outings, 2 h): median 30 (all users), p90 254, max 2,312; heavy tail, top 10% of users hold 59%. Active local days per user: median 22, p90 133, max 1,251; span median 72 days (active/span density median 0.52).
- Activity rule table (users kept): >=3 outings/>=2 days 174; >=10/>=5 139; >=20/>=5 109; >=30/>=5 91.
- **Recommendation: gap 2 h; real outing = >=5 min and >=500 m; user qualifies with >=10 real outings on >=5 local days (139 users, 99.0% of real outings).** Rationale: 1 h would cut an extra ~4k outings, mostly signal loss inside one visit; 4 h starts to glue separate errands (p90 duration 9 h).

## 2. Stops

Stay-point rule on 30 s-thinned points, segmented by outing. Per real outing of qualifying users:

| radius / dwell | stops | stops per outing | outings with >=1 / >=2 | median dwell | revisit r8 / r9 / r10 | places per user r8 / r9 |
|---|---|---|---|---|---|---|
| 100 m / 5 min | 38.8k | 2.05 | 55% / 34% | 11 min | 60% / 52% / 43% | 29 / 45 |
| 100 m / 10 min | 22.4k | 1.18 | 42% / 23% | 21 min | 59% / 51% / 42% | 22 / 30 |
| 100 m / 20 min | 12.7k | 0.67 | 31% / 15% | 39 min | 59% / 51% / 43% | 14 / 18 |
| 200 m / 5 min | 45.7k | 2.41 | 66% / 41% | 11 min | 62% / 53% / 42% | 35 / 54 |
| **200 m / 10 min** | 26.0k | 1.38 | 49% / 27% | 23 min | 59% / 50% / 39% | 25 / 34 |
| 200 m / 20 min | 15.5k | 0.82 | 37% / 19% | 42 min | 58% / 49% / 39% | 18 / 25 |
| 300 m / 10 min | 27.7k | 1.46 | 53% / 30% | 24 min | 60% / 51% / 38% | 27 / 39 |

(revisit = share of stops at a cell the same user visited in an earlier outing.) Stops per real outing for the chosen rule: p50 0, p90 4, p99 12. Median rms spread of points inside a stop is 64 m (p10 37, p90 97), an upper bound for GPS jitter because it includes walking around a site. Revisit is high (about 59% at res 8, 50% at res 9), so favourite places are real and the generator needs a revisit probability. Radius changes little beyond 200 m; dwell mostly trades yield for purity. **Recommendation: 200 m / 10 min**, plus start and end of each outing as visits (a destination where logging stops yields no dwell).

## 3. Time (local)

- Outing starts peak at 08h and 17-18h (commute-like), trough 03-04h; stop arrivals rise 09h and peak 18h. User-weighted and event-weighted profiles agree closely.
- Weekend share (2/7 = 29% if flat): outing starts 26%, stop arrivals 35%. Tod bins (night/morning/afternoon/evening): outing starts 9/36/28/28%, stops 8/24/40/29%.
- Top res-7 cells (112 to 38 users) have profiles distinct from the overall one: JSD 2 to 25 times the sampling-noise JSD for the same size, e.g. cell 1 is 36% evening, cell 8 is 49% weekend. By public category: school/office/food/shop 11x to 21x noise (see the heatmap in the notebook for the direction), leisure 5x, other 4x. Effects are real but modest in absolute terms (JSD 0.004 to 0.017 bits), so time profiles are usable as smooth features, not as sharp labels. Only 48% of stops snap to the 7.4k-place public table within 150 m.
- Finer bins vs counts (res-7 cell x bin, 139 users, 25.6k stops): 8 bins (4 tod x wd/we) give 3.1k stops/bin and 28% of user-keys in keys with >=10 users; 12 bins 25%; 24 h 21%; 48 bins (24 h x wd/we) 17.5%, 496 stops/bin. **Use 4 tod x weekday/weekend for place/category x time; reserve 24 h x 2 for global outing-start and stop tables.**

## 4. Overlap (139 qualifying users; stops at 200 m/10 min; "core" = 116.2-116.6E, 39.8-40.1N)

Share of user-cell incidences falling in cells with >=10 users:

| kind | res | all | core | cells >=10 users (all) |
|---|---|---|---|---|
| stops | 6 | 41% | 91% | 36 |
| stops | 7 | 41% | 78% | 81 |
| stops | 8 | 26% | 45% | 100 |
| stops | 9 | 15% | 25% | 102 |
| trajectory cells | 6 | 32% | 99% | 464 |
| trajectory cells | 7 | 32% | 97% | 1,289 |
| trajectory cells | 8 | 31% | 90% | 3,450 |
| trajectory cells | 9 | 30% | 80% | 9,038 |

Median users per cell is 1 for every "all" row; cells with >=2 users hold 64-78% of stop incidences and 78-82% of trajectory incidences. Restricting to the core multiplies overlap (trajectory res 8: 31% to 90%), so most of the signal is Beijing core; outside it stays thin.

Co-visit graph (edge = >=2 shared "rare" cells visited by 2 to 10 users; weight sum 1/log(1+n)):

| kind | res | scope | edges | density | users with a partner | degree p50 / p90 |
|---|---|---|---|---|---|---|
| stops | 7 | all | 427 | 4.5% | 66% | 2 / 20 |
| stops | 8 | all | 1,070 | 11% | 84% | 9 / 42 |
| stops | 9 | all | 1,216 | 13% | 89% | 11 / 45 |
| stops | 8 | core | 932 | 10% | 76% | 7 / 37 |
| traj | 8 | all | 2,021 | 21% | 86% | 23 / 68 |
| traj | 9 | all | 3,399 | 35% | 97% | 46 / 94 |

Sensitivity (stops r8): max popularity 5/10/20 with >=2 shared gives 394/1,070/1,888 edges (100/117/124 users with a partner). Coarse cells (res 7) in the core have almost no "rare" cells, so the rarity cap and the resolution must be chosen together. The traversal graph is too dense to discriminate (35% of all pairs at res 9).

## 5. Sampling regime (feeds the render)

- Within-file dt: median 2 s, p90 5 s, p99 20 s; 31% are 1 s, 33% 2 s, 5% 3 s, 25% 5 s; 96.4% <=5 s. Per-user median dt: 1 s (32 users), 2 s (50), 3 s (12), 4 s (5), 5 s (44), 11-16 s (5). So the render should draw a per-user sampling period from {1,2,3,5} s plus a few slow loggers.
- Gaps: 0.50% of steps exceed 60 s (5.4 gaps per logged hour); gap lengths p25/p50/p75/p90/p99 = 100 s / 210 s / 7 min / 29 min / 4.1 h.
- Step speed (m/s, labelled; p50 / p90 / p99): walk 1.1 / 2.7 / 11 (15% stationary), bike 3.1 / 5.2 / 11, bus 4.7 / 12.7 / 25, car/taxi 8.5 / 20.7 / 33, rail 19 / 37 / 57. Unlabelled-or-labelled overall p50/p90/p99: 3.8 / 20.8 / 38.9.
- Jitter proxy at stops: rms spread 64 m median (see section 2).

## 6. Split-half reliability (qualifying users, real outings split by time and at random)

Place sets (median over users with >=2 stop-outings in both halves; 124-126 users):

| res | split | Jaccard place sets | Jaccard top-3 | share of half-B stops in half-A set | same for another user's set |
|---|---|---|---|---|---|
| 7 | time / random | 0.20 / 0.28 | 0.5 / 0.5 | 55% / 64% | 30% / 32% |
| 8 | time / random | 0.14 / 0.21 | 0.2 / 0.5 | 43% / 51% | 15% / 17% |
| 9 | time / random | 0.11 / 0.14 | 0.2 / 0.2 | 32% / 41% | 8% / 8% |

Co-visit graph (each half built separately with the same rule; 132-139 users):

| kind | res | split | edge Jaccard | P(edge in B given edge in A) | density B | lift |
|---|---|---|---|---|---|---|
| stops | 7 | time / random | 0.28 / 0.29 | 0.47 / 0.42 | 3.8% / 3.5% | 12x |
| stops | 8 | time / random | 0.29 / 0.33 | 0.43 / 0.49 | 6.2% / 7.0% | 7x |
| traj | 8 | time / random | 0.36 / 0.44 | 0.51 / 0.63 | 16% / 18% | 3.3-3.6x |
| traj | 9 | time / random | 0.46 / 0.54 | 0.60 / 0.71 | 28% / 32% | 2.1-2.2x |

Reading: places reproduce well above chance (res 8: 43-51% of held-out stops fall in the other half's set vs 15-17% for a stranger) but the exact set is only moderately stable (Jaccard 0.14-0.21), and time splits are consistently less stable than random splits, i.e. there is drift. Stop co-visit edges reproduce at 7x chance at res 8 (about 43-49% of edges recur), which is a label-free reference level for M on real data (edge Jaccard of about 0.3 between halves is the self-consistency to beat or match). Note: each half has about half the data, so rarity counts shrink.

## 7. Recommended public parameters

"Informed" = chosen after seeing real data, so it is tuning for the ledger caveat.

| Parameter | Recommendation | Justification | Informed by real data |
|---|---|---|---|
| Local time | UTC+8; artifact states its hour convention | GMT timestamps; commute peaks at 08h/17-18h appear only in local time | no |
| Outing gap | 2 h (7,200 s), across file boundaries | 84% of consecutive files are >2 h apart; 0.22 in-file gaps per file exceed 2 h and are signal loss | yes |
| Real outing | >=5 min and >=500 m | removes 5% fragments | yes (mild) |
| Min activity | >=10 real outings on >=5 local days | 139 users, 99.0% of real outings; allows split-half | yes |
| Stop rule | 200 m / 10 min, plus outing start/end visits | 1.38 stops per real outing, median dwell 23 min; 100 m/5 min is mostly pauses | yes |
| Time bins | 4 time-of-day x weekday/weekend (8) for place/category tables; 24 h x 2 only for global outing and stop tables | 3.1k stops per bin; 24 h x 2 puts only 17% of user-keys in >=10-user keys | yes |
| Place hierarchy | H3 6 -> 7 -> 8 | cells with >=10 users saturate at ~100 by res 7-8 (36 / 81 / 100 / 102 for res 6-9) while incidence share falls 41% -> 26% -> 15% | yes |
| Segment hierarchy | H3 6 -> 7 -> 8, refine to 9 or matched segments under busy parents | traversed-cell incidence share in >=10-user cells is flat (32% to 30%) across res 6-9 | yes |
| Cap: outings per user | 100 | p50 45, p90 280; caps 32% of users, keeps 43% of outings | yes |
| Cap: distinct places per user | 50 res-8 stop cells | p50 25, p90 127; caps 26% of users, keeps 53% of incidences | yes |
| Cap: distinct segments per user | 1,000 traversed res-8 cells (recalibrate on matched segments in T2) | p50 362, p90 3,981; caps 30% of users, keeps 33% | yes |
| Co-visit edge | >=2 shared rare cells (2 to 10 users), IDF weight, stops at res 8; traversal graph only as a sparse weight, not as edges | 84% of users get a partner, 7x split-half lift; traj res 9 links 35% of all pairs | yes |

Caps trade sensitivity against retained signal: capping at the median for outings (45) would keep only 25% of the outing mass; the numbers above are a compromise, to be rechecked against the T2 SNR report (a further data-informed step).

## Caveats

Unlabelled majority (only 20% of points have mode labels, 69 users). Stay detection runs on 30 s-thinned points. 2008-09 dominate the data and Beijing core dominates space; 101 of 182 users have some points outside the Beijing box, which is why "all" overlap is so much lower than "core". Cell lookups round positions to about 11 m. Half-split graphs use the same rarity cap as the full graph, so their absolute edge counts are lower than a full graph would imply.

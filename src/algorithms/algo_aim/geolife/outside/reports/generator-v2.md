# Generator v2 (T3): outing process on the MapProvider protocol

Code: `src/geolife/outside/{population_v2,render_v2,pipeline_v2,fakedata_v2}.py`; tests: `tests/test_generate_v2_pipeline.py`.
Inputs are the `params` dict of `docs/geolife/OUTINGS_PARAMS.md` and a `MapProvider` (toy `GridCityProvider` or `OvertureProvider`);
output is traces plus `GroundTruthV2`. Only numpy/pandas/scipy (+ h3 for cells, already a repo dependency).

## API

```python
from geolife.outside.pipeline_v2 import generate_v2, planted_stats
from geolife.outside.fakedata_v2 import fake_params_v2, fake_generation_v2
g = generate_v2(params | artifact_dir, provider, n, seed, gen=None, grounding=None)   # -> GeneratedV2
g.result.traces        # {user: [GpsPoint]}   UTC timestamps
g.result.outings       # {user: [Outing]}     planted outings (visits with kind start/stop/end, cells, segments)
g.result.truth         # GroundTruthV2
planted_stats(g)       # planted-statistics check vs params
```
`sample_population_v2(params, n_users, seed, provider, gen)` and `render_v2(pop, mapping, provider, gen, seed)` are the two layers;
teleport reuses `teleport.ground_places` through `population_v2.latent_view` (a thin adapter, no change to v1 code).
With an artifact directory, `params` come from `fit_behaviour_v2(measurements, ledger)` wherever T2 puts it (tried in `generate.fit_v2`,
`generate.fit`, `learn`, `generate.measure`), and `gen` is layered over the artifact `generation` dict. `gen` overrides every sensor and
process knob (`DEFAULT_GEN_V2`, `SENSOR_DEFAULTS`) for sensitivity studies; the same planted world is rendered unchanged when only sensor
knobs change (tested).

`GroundTruthV2` fields as specified: `groups`, `favourite_places` (map ids; places with >= 3 visits), `shared_places` (>= 2 users),
`shared_segments`, `co_traveled` as `(leader, follower, leader_outing_index)`, `visits` (types.Visit as rendered). Extras: `co_traveled_detail`
`(leader, i_leader, follower, i_follower)`, `group_places` (map ids), `user_group`, `latent_to_map`.

## Model

1. Place pool: Zipf popularity (`zipf_s`), categories by `cat_mix` (restricted to categories the provider has), core-biased positions in an
   `extent_m` square (default: min side of the provider's place bbox, so latent metres are real metres). Pool size is calibrated (2 passes) so the
   mean users per visited place equals the mean of `users_per_place_hist`; capped at 70% of the provider's places.
2. User: activity centre and radius, Dirichlet taste vector over categories, latent group (Zipf-sized, 20% ungrouped) whose favoured places
   (a campus-like cluster of 14 places) take a `concentration` share of new-place draws. Outings per user ~ lognormal, window length ~ lognormal
   (at least outings/2.5 days), active days drawn inside the window with weekend weight from `start_pmf`, outing starts from the joint
   (daytype, tod) `start_pmf` with an hour shape peaking at 8h and 17-18h, minimum spacing 3 h.
3. Outing: mode from `mode_pmf`; stops from `stops_pmf` ("5+" gets a geometric tail); start place continues the previous outing's end with
   probability 0.5; the end place is the start with probability 0.15 after a stop. Every place draw: with probability `revisit_rate` a preferential
   revisit (count^0.8), else a new place, from the group set with probability `concentration`. Score = popularity x taste x activity region x
   P(daytype, tod | category) from `cat_profile` (a likelihood, not rescaled per category) x `(d+200)^-distance_decay` x a lognormal "length target"
   bump (target outing length from `length_km`, with a mode-dependent factor). Dwell ~ `stop_dwell_min` (>= 10 min). A per-user cap on distinct
   places (lognormal from `distinct_places`, correlated 0.8 with activity, x2) turns further draws into forced revisits. No home/work roles anywhere.
4. Co-travel: 20% of users are paired inside a group; the follower copies about 15% of the leader's outings (same places, time, mode, route,
   independent GPS noise), replacing overlapping own outings.
5. Render: each leg is a provider route with a lognormal detour (`1 + LN(route.detour_ratio)`, via-point search near the midpoint); mode speeds
   with per-outing/leg variation and stop-and-go; GPS only while an outing is under way (pre/post sit < 1 min). Sensor: per-user period from
   {1,2,3,4,5,13} s with weights from the EDA (median 2 s), jitter, signal-loss gaps (5/h, lognormal median 210 s, sigma 1.7, max 1 h), 25% of stops
   unlogged (indoors), stop jitter 25 m, noise 6 m, rare outliers. Outings are forced >= 2 h + margin apart after routing (own outings shift
   later; own outings overlapping a fixed co-travel copy are dropped). All hour/day logic is local UTC+8 (day 0 = Monday); timestamps are UTC.

## Planted-statistics check (`planted_stats`)

60 users, toy 40x40 grid city (6000 places), `fake_params_v2()`, outings capped at 120 per user for speed; Amsterdam: 60 users, Overture
(15k POIs + roads), cap 60 outings, 10 s sampling. "Got" is measured on the rendered truth (visits as placed on the provider).

| statistic | target (params) | toy grid | Amsterdam | tolerance |
|---|---|---|---|---|
| median outings per user | 30 | 38.5 | 25.0 | 35% rel (n small, heavy tail) |
| stops per outing (mean) | 1.34 | 1.33 | 1.33 | 15% rel |
| stop revisit rate | 0.55 | 0.59 | 0.58 | 0.12 |
| outing start (daytype x tod) TV distance | 0 | 0.038 | 0.044 | 0.10 |
| weekend share of starts | 0.26 | 0.28 | 0.26 | 0.08 |
| corr(weekend share per category, profile) | >= 0.5 | 0.99 | 0.99 | floor |
| corr(cell log-lift over all categories, profile) | >= 0.5 | 0.92 | 0.91 | floor |
| mean users per visited place | 2.14 | 2.36 | 2.45 | 30% rel |
| share of places with >= 2 users | 0.35 | 0.51 | 0.46 | 0.22 |
| median distinct places per user | 25 | 43.5 | 28 | 100% rel (informative) |
| visit share in group places | 0.35 | 0.40 | 0.44 | 0.20 |
| shared places (>= 2 users) | >= 1 | 791 | 417 | floor |

Timings (this machine): toy 60 users / 3.2k outings: population 3 s, teleport 0.2 s, render 3 s. Amsterdam 60 users / 1.9k outings /
406k points: population 1.7 s, teleport 1.4 s, render 17.5 s (route cache warm); `OvertureProvider` construction 10-18 s.
Tests run in about 25 s total, including a 12-user Amsterdam run (skipped if the cache is absent).

## Honest gaps

- Interpretation choices for ambiguous spec items: lognormal `{mu, sigma}` = mean/sd of ln(value) in days, minutes, km; `start_pmf` and
  `cat_profile` are joint over (daytype, tod) (per-daytype conditional pmfs are detected and weighted by a 26% weekend share); `n_groups` is
  capped at n_users/5; `detour_ratio` is the lognormal excess over the shortest route.
- `duration_min` is not a driver: durations come out of route length, mode speed and dwell, and are reported only (toy median 37 min vs 45).
  Outing length is steered by a soft target, but revisit draws and the grid detour make lengths 1.5-2x too long on the toy at `length_scale` 0.3;
  on Amsterdam the 7x9 km extent makes lengths about half the 9 km target. Treat both as city-dependent.
- Revisit rate and distinct places per user pull against each other: the revisit rate is honoured and the distinct-places cap is relaxed (x2,
  correlated with activity), so the median distinct places is above target on the toy (43 vs 25). Start/end visits also add distinct places.
- Users-per-place: only the mean is calibrated; the histogram shape is Zipf-driven, with too few single-user places (share with >= 2 users
  0.46-0.51 vs 0.35) and a thinner tail than the real hub structure. Where the pool hits the provider cap (70% of its places) the mean overshoots.
- Time-of-day profile is 4 bins; the within-bin hour shape is a fixed commute-peak curve, not fitted. Place-level time effects follow category only.
- Co-travel is whole-outing copying only (no partial shared routes), and groups share places but not times (no co-presence, by decision).
- Heavy users: outings per user is capped by `max_outings` (2500 default; tests use 120), so the heavy tail beyond that is truncated.
- Latent-to-map grounding keeps category and popularity percentile, so place popularity on the map follows provider weights only roughly;
  Overture has few residential POIs, so no residential category is used by the fake params' provider-restricted mix.
- Real-vs-synthetic transfer and recovery by M are not evaluated here (T4/T5). The fake params are hand-set from the EDA, not fitted.

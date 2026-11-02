# GeoLife interfaces: contract, `params`, planted truth, measurements

What the pieces of the pipeline hand to each other. Authoritative code: `src/geolife/transfer/types.py` (dataclasses),
`src/geolife/transfer/schema_v2.py` (released tables), `src/geolife/outside/fit_v2.py` (`params`). Plan: `PLAN.md`.

Contract types: `src/geolife/transfer/types.py` (v2 block: `Visit`, `Outing`, `UserOutings`, `TOD_BINS`, `DAYTYPES`).
Public parameters (from `geolife/inside/reports/outings-eda.md`, informed by real data -> ledger caveat): gap 2 h; real outing >= 5 min and >= 500 m;
min activity >= 10 real outings on >= 5 local days; stop rule 200 m / 10 min plus outing start/end as visits; time = 4 tod bins x weekday/weekend (UTC+8);
place hierarchy H3 6->7->8; traversal hierarchy H3 6->7->8 (->9 under busy parents); caps per user: 100 outings, 50 res-8 stop cells, 1000 res-8 traversed cells.

## Who hands what to whom

| Stage / piece | Consumes | Produces | Zone |
|---|---|---|---|
| Tokenization (`transfer/tokenize/outings.py`) | raw traces (+ public places table) | `list[UserOutings]` | transfer (shared pure code) |
| Learner v2 (`inside/learn_v2.py`, `measure_v2.py`) | `list[UserOutings]` | artifact: noisy tables + privacy ledger | inside |
| Fit (`outside/fit_v2.py`) | artifact tables | `params` dict (below) | outside |
| Generator v2 (`outside/*_v2.py`) | `params`, sensor knobs, a `MapProvider` | traces + `GroundTruthV2` | outside |
| Method M v2 (`eval/m/outings.py`) | traces + public map | places of interest, common places / segments, co-visit graph | eval |

## `params` (JSON-safe dict returned by `fit_behaviour_v2`, consumed by `sample_population_v2`)

```
params = {
  "version": 2,
  "places":  {"zipf_s": float, "pool_size": int, "cat_mix": {cat: share}, "users_per_place_hist": {k: n_places}},
  "users":   {"outings_per_user": {"mu": f, "sigma": f},           # lognormal
              "active_days": {"mu": f, "sigma": f},                 # lognormal, window length in days
              "distinct_places": {"mu": f, "sigma": f},
              "revisit_rate": float},
  "groups":  {"n_groups": int, "concentration": float},             # share of a user's visits in its group's favoured places
  "outing":  {"stops_pmf": {"0": p, "1": p, ...},                   # stops per outing, last key is "n+"
              "duration_min": {"mu": f, "sigma": f}, "length_km": {"mu": f, "sigma": f},
              "mode_pmf": {mode: p}, "stop_dwell_min": {"mu": f, "sigma": f},
              "distance_decay": float,                              # exponent on inter-stop distance
              "start_pmf": {daytype: {tod_bin: p}}},
  "time":    {"cat_profile": {cat: {daytype: {tod_bin: p}}}},       # when each category is visited
  "route":   {"detour_ratio": {"mu": f, "sigma": f}},               # lognormal of the EXCESS (ratio - 1)
}
```
Lognormal (mu, sigma) are MAP fits of the raw noisy histograms against the public DEFAULTS priors
(`fit_v2._lognormal`, PLAN step 4); `detour_ratio` is the excess, i.e. route target = 1 + LN(mu, sigma).
`sensor` (render): `params["sensor"]` (fitted by `outside/sensors.fit_sensors` from the released `sensor_*` tables; post-processing)
holds `hists` (binned pmfs: `interval`, `interval_user`, `gap_len`, `gap_rate`, `stop_jitter`, `heading`, `points_per_outing`,
`speed_by_mode`, `step_by_mode`), `speed_ms_by_mode`, `sensor_noise_m`, `gap_rate_per_hr_mean`, `signal` and `fallbacks`.
`render_v2` draws the per-user sampling period and gap rate, gap lengths and in-stop jitter from those histograms by
inverse-CDF sampling (`outside/sensors.HistSampler`); `pipeline_v2` injects `speed_ms_by_mode` / `sensor_noise_m` into `gen`.
Any table that is mostly noise falls back to the hand-set knobs, which still live in the `gen` dict / `render_v2.SENSOR_DEFAULTS`:
`sample_interval_s`, `gap_rate_per_hr`, `gap_mean_s`, `sensor_noise_m`, `dwell_jitter_m`, `speed_ms_by_mode` (same names as the
v1 generator uses where they exist). `use_sensor_hists: false` in `gen` disables every histogram draw (sensitivity studies).

## `GroundTruthV2` (T3 output, consumed by T4 evaluation)

`groups: dict[group_id, list[user]]`, `favourite_places: dict[user, list[place_id]]` (latent -> map ids after teleport),
`shared_places: dict[place_id, list[user]]` (visited by >= 2 users), `shared_segments: dict[segment_id, list[user]]`,
`co_traveled: list[(user, user, outing_index)]`, `visits: dict[user, list[Visit]]` (as rendered).

## Measurement names

Authoritative list, keys, caps and sensitivities: `src/geolife/transfer/schema_v2.py`. Released tables (about 20): hierarchical
distinct-user tables for stops (`place_users_r6|r7|r8`) and traversed cells (`trav_users_r6|r7`); `cat_time_visits` and `outing_start`
on the 3 time bins `types.TOD_3` (morning [05,11), midday [11,16), evening [16,05) local; x weekday/weekend); the fine `params` bins are an assumed hours-proportional split of them; outing-shape histograms (stops, duration, length, mode, detour, dwell, hop);
per-user histograms (`user_outings_hist`, `user_places_hist`, `user_span_hist`, `revisit_hist`, `n_users`); `cat_transitions`.
No longer released (pure noise at epsilon 6): `place_time_users`, `stop_transitions_r6`, `trav_users_r8`, `trav_users_r9`.
Derived statistics (users-per-place histogram, Zipf exponent) are post-processing of the distinct-user tables and cost no budget.
Sensor tables (since v2c, decision D20; measured from the raw points by `inside/measure_sensors.py`, 20% of the total rho):
`sensor_interval`, `sensor_interval_user`, `sensor_gap_len`, `sensor_gap_rate`, `sensor_speed` (by mode), `sensor_step` (by mode),
`sensor_heading`, `sensor_stop_jitter`, `sensor_points_per_outing` - user-level pmf/one-hot histograms with fixed public bin
edges and a capped per-user sample (`schema_v2.MAX_SENSOR_STEPS`), each with its own ledger entry. Tables that are mostly noise
are ignored by the generator, which falls back to the hand-set defaults (listed as a ledger caveat).

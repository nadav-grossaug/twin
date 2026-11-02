# Outings: interfaces between the parallel tasks (orchestrator-owned)

Contract types: `src/geolife/transfer/types.py` (v2 block: `Visit`, `Outing`, `UserOutings`, `TOD_BINS`, `DAYTYPES`).
Public parameters (from `geolife/inside/reports/outings-eda.md`, informed by real data -> ledger caveat): gap 2 h; real outing >= 5 min and >= 500 m;
min activity >= 10 real outings on >= 5 local days; stop rule 200 m / 10 min plus outing start/end as visits; time = 4 tod bins x weekday/weekend (UTC+8);
place hierarchy H3 6->7->8; traversal hierarchy H3 6->7->8 (->9 under busy parents); caps per user: 100 outings, 50 res-8 stop cells, 1000 res-8 traversed cells.

## Pipeline split

| Task | Consumes | Produces | Does not need |
|---|---|---|---|
| T1 tokenization | raw traces (+ places table) | `list[UserOutings]` | DP, generator |
| T2 learner v2 | `list[UserOutings]` | artifact (measurements + ledger) and `fit_behaviour_v2(measurements, ledger) -> params` | generator |
| T3 generator v2 | `params` (+ artifact `generation` dict for sensor), `MapProvider` | traces + `GroundTruthV2` | learner |
| T4 M v2 | traces + map | places of interest, common places/segments, relation graph | learner |

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
  "route":   {"detour_ratio": {"mu": f, "sigma": f}},
}
```
`sensor` (render) comes from the artifact `generation` dict: `sample_interval_s`, `gap_rate_per_hr`, `gap_mean_s`, `sensor_noise_m`, `dwell_jitter_m`,
`speed_ms_by_mode` (same names as the v1 generator uses where they exist).

## `GroundTruthV2` (T3 output, consumed by T4 evaluation)

`groups: dict[group_id, list[user]]`, `favourite_places: dict[user, list[place_id]]` (latent -> map ids after teleport),
`shared_places: dict[place_id, list[user]]` (visited by >= 2 users), `shared_segments: dict[segment_id, list[user]]`,
`co_traveled: list[(user, user, outing_index)]`, `visits: dict[user, list[Visit]]` (as rendered).

## Measurement names (T2 owns the exact keys; T2 must document them in `generate/schema_v2.py`)

Hierarchical distinct-user tables for stops (`place_users_r6|r7|r8`) and traversed cells (`trav_users_r6|r7|r8|r9`); `cat_time_visits` (category x tod x daytype);
`place_time_users` (res-6 cell x tod x daytype); outing shape histograms (stops, duration, length, mode, start time); `user_places_hist`, `user_outings_hist`,
`revisit_hist`; coarse `stop_transitions_r6` and `cat_transitions`. Derived statistics (users-per-place histogram, Zipf exponent) are post-processing of the
distinct-user tables and cost no budget. Co-visit degree histogram under a neighbour cap is a stretch goal; T2 decides whether its sensitivity is acceptable.

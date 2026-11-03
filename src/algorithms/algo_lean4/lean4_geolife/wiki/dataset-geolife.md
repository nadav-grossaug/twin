---
title: Dataset — GeoLife GPS trajectories (labelled subset)
type: entity
updated: 2026-11-03
sources: ["../geolife_data.tsv", "profiling run 2026-11-02 (DuckDB)"]
---

# dataset-geolife — `geolife_data.tsv`

**The CURRENT target.** The mode-labelled subset of Microsoft GeoLife GPS
trajectories, flattened to one TSV: one row = one GPS point.

## Schema (TSV, header row, tab-separated)

| column | type | notes |
| --- | --- | --- |
| `user_id` | varchar | zero-padded 3-digit (e.g. `010`); 64 distinct |
| `sequence_uuid` | varchar | uuid-shaped id of the labelled segment |
| `trajectory_id` | varchar | `yyyymmddhhmmss` of trajectory start (UTC) |
| `datetime` | timestamp | `YYYY-MM-DD HH:MM:SS`, **UTC** (GeoLife PLT convention; staging shifts +8 h to Beijing local via `source_utc_offset_s`) |
| `latitude` | double | WGS84 |
| `longitude` | double | WGS84 |
| `altitude_ft` | double | feet; **−777 = missing sentinel** (55,478 rows) |
| `label_start` | timestamp | mode-label window start (UTC) |
| `label_end` | timestamp | mode-label window end (UTC) |
| `mode` | varchar | transport-mode label, 100% populated |

## Measured profile (2026-11-02, full-file DuckDB pass)

- **5,440,384 points, 64 users, 4,464 trajectories**; window
  2007-04-12 10:21 → 2011-12-31 15:18.
- Points per trajectory: p5 28, median 487, p75 1,216, p95 4,430, p99
  11,082, max 56,785 (mean 1,219).
- Trajectories per user: p5 2, median 20, p90 170, p99 618, max 734 —
  **extreme heavy tail**; per-user DP clipping is mandatory.
- Mode distribution (points): walk 29.2%, bus 23.5%, bike 17.5%, train
  10.3%, car 9.5%, subway 5.3%, taxi 4.5%, airplane 0.17%, boat 0.07%,
  run 0.04%, motorcycle 0.006%. Public collapse map → 6 classes:
  walk(+run), bike, bus, car(+taxi+motorcycle), rail(train+subway),
  other(airplane+boat).
- Sampling interval (within trajectory): p25 1 s, median 2 s, p95 6 s,
  p99 25 s — high-frequency GPS.
- Step distance: median 4.9 m, p95 62 m, p99 204 m. Step speed: median
  3.1 m/s, p95 28 m/s, p99 44.6 m/s.
- Geography: **83.8% of points inside the Beijing bbox**
  (lat 39.4–41.1, lon 115.4–117.5); median point (39.975, 116.332).
  Remainder: other Chinese cities + international trips (lon spans the
  globe). Lat p01–p99: 30.48–42.90.
- Quality: 1 row with invalid latitude (400.17); 159,651 duplicate
  (user, trajectory, datetime) groups (multi-Hz logging collisions);
  no NULL coords beyond the 1 invalid row; labels 100% populated.

## Privacy shape

- Protected entity = **user**; n = 64 → the small-n hard regime (like the
  reference implementation, unlike Foursquare's 29,640). SNR ≈
  `share × 64 × ε_cell`: grids must stay coarse, dominant cells carry the
  signal, public priors rescue small cells.
- Crown jewels: home/work anchor LOCATIONS and any absolute coordinates.
  All released geometry must be anchor-relative; the Beijing bbox appears
  only as a validation exclusion constant.

## Consequences for thresholds (chosen from measurements)

- Staypoint 200 m / 1,200 s works at this sampling rate (median 2 s, step
  4.9 m ⇒ hundreds of points per staypoint).
- Speed-class edges 0.5 / 2 / 8 / 20 m/s separate still/walk/city/fast
  cleanly against the per-mode speed quantiles.
- Dedupe exact duplicate timestamps and drop out-of-range coords in
  staging, both counted as quality flags.
- Day boundaries + recording gaps > 6 h break visit sequences (recording
  is bursty; unobserved days are NOT home-days).

Links: [[migration-plan-geolife]], [[geolife-reference]], [[motifs]],
[[dp-release]], [[enclave-boundary]].

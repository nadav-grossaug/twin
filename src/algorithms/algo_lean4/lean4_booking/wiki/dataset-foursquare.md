---
title: Foursquare country-transfer dataset
type: entity
updated: 2026-11-01
sources: [../country_transfer.tsv, ../data_ingest/dataset_TIST2015/dataset_TIST2015_readme.txt]
---

# Foursquare country-transfer dataset — the CURRENT target

`country_transfer.tsv`: the raw source for the travel migration of the
pipeline (see [[migration-plan-foursquare]] and `../goal.md`). Derived from
the Foursquare TIST2015 global check-in corpus
(`../data_ingest/dataset_TIST2015/`): one row = one user's international
transfer — the moment their check-in stream moves from one country to
another. Tab-separated, 12 columns:

```
userid  dt  prev_country  prev_last_venue_id  prev_last_lat  prev_last_long
prev_duration  next_country  next_first_venue_id  next_first_lat
next_first_long  next_duration
```

`dt` = timestamp of the first check-in in the next country. `prev_duration`
/ `next_duration` = seconds between the user's first and last check-in in
that country-stay; consecutive rows chain (`next_duration` of row i =
`prev_duration` of row i+1 when the transfers are consecutive). Venue ids
are 24-hex Foursquare ids; coordinates are the boundary venues' positions.

## Measured profile (2026-11-01)

- **84,964 rows, 29,640 users, 77 countries, 2,791 ordered corridors**;
  window 2012-04-04 03:23 → 2013-09-16 21:30 (UTC, ~18 months).
- **No NULLs/NA in any column; no self-transfers** (prev ≠ next on every
  row); parses clean — no verified repairer needed (recorded; pattern kept
  in reserve).
- **Travels per user — heavy-tailed** (the migration's first focus):
  min 1, median 2, mean 2.87, p90 5, p99 14, **max 195**; 23,528 users
  (79%) have ≥ 2 transfers. Public log₂-ish bins chosen: 1, 2, 3–4, 5–8,
  9–16, 17–32, 33+.
- **Top countries** (prev side): MY 8,596 · SG 8,183 · US 7,972 · TR 5,144
  · GB 4,147 · ID 3,950 · RU 3,415 · TH 2,624 · JP 2,282 · CN 2,240 —
  Southeast-Asia-heavy corpus.
- **Corridors**: MY→SG 5,808 / SG→MY 5,633 dominate (1,984 / 1,952 distinct
  users); then ID↔SG, TR↔GR, MY↔ID. 283 ordered corridors have ≥ 50 users,
  544 have ≥ 20 — top-k release with suppression 50 users / 100 transfers
  is well populated.
- **Return structure (dominant motif)**: 39,879 of 55,324 consecutive
  transfer pairs (72.1%) are A→B followed by B→A. Any sequence model must
  beat a return-to-previous baseline, not repeat-last.
- **Commuter corridors** (beacon analog): 761 user-corridors with the same
  ordered pair ≥ 5 times (max 98 repeats); gap CV of those: p10 0.41,
  p50 0.78, p90 1.41 → threshold CV ≤ 0.8 keeps ~half.
- **Tours** (rarest motif → the planted round-trip pattern): ≥ 3 consecutive
  transfers with every gap ≤ 7 d spanning ≥ 3 countries — 1,126 instances,
  978 users.
- **Bursts**: 7,196 windows of 3 transfers within 14 d.
- **Inter-travel gaps** (second focus): p10 1.9 d, median 12.9 d, p90
  123 d, max 508 d; mean 40.7 d, sd 65.8 d, mean/median 3.16 (heavy right
  skew). Class shares over public bins <2 / 2–7 / 7–30 / 30–90 / ≥90 d:
  10.5 / 24.5 / 33.6 / 17.1 / 14.3%.
- **Stay durations**: 0 s (single-checkin stays — flagged, not dropped) to
  527 d. Class shares <1 / 1–7 / 7–30 / 30–90 / ≥90 d:
  13.7 / 33.3 / 16.2 / 12.2 / 24.5%.
- **Hop distance** (haversine between boundary venues): p50 1,891 km, max
  19,925 km. Class shares <500 / 500–2000 / 2000–7000 / ≥7000 km:
  21.6 / 29.2 / 28.0 / 21.2% — an almost even public 4-bin grid.
- **Calendar**: day-of-week nearly uniform (max/min = 13,659/10,656 = 1.28)
  → released as a declared uniform public prior, no ε spent. Month counts
  are strongly uneven (May 13,610 vs Sep 2,484) but the window starts in
  April 2012 and ends mid-September 2013 — month counts mix seasonality
  with coverage; model within the public window, don't read as pure
  seasonality.

## Sensitivity

A user's (country, timestamp) sequence is a strong re-identification vector.
Protected entity = **the user** (whole history = one `Neighbour` record).
User ids and venue ids never cross the boundary; coordinates leave only as
public country centroids; countries appear under DP only at region
granularity (8-region public map), by name only via suppressed top-k
corridors. See `../goal.md` Enclave Boundary for the planned 400/100 ledger.

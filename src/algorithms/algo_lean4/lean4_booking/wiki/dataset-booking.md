# dataset-booking — Booking.com multi-destination trips (combined_set.csv)

EDA run 2026-11-03 over the full 1,545,502-row `combined_set.csv` (DuckDB,
`/tmp/booking_eda.py`). This is the WSDM'21/SIGIR'21 Booking.com
multi-destination-trips challenge data (anonymized; fake country names:
Gondal, Elbonia, ...). Task: predict the next in-trip destination.

## Schema (9 columns, no nulls anywhere)

`user_id` (BIGINT), `checkin`/`checkout` (DATE, checkout > checkin always),
`city_id` (BIGINT), `device_class` (desktop/mobile/tablet),
`affiliate_id` (BIGINT), `booker_country` (VARCHAR, 5 values),
`hotel_country` (VARCHAR, 195 values), `utrip_id` ("<user_id>_<n>").

## Measured facts

- 259,159 users · 288,348 trips (utrips) · 39,901 cities · 195 hotel
  countries · 5 booker countries · 3,611 affiliates · 3 device classes.
- Window 2015-12-31 .. 2017-02-27; checkin month ramps up to 2016-08
  (301,806) then decays (seasonal + censoring).
- **Legs per trip: median 5, mean 5.36, max 48.** The challenge design
  shows: 130,373 trips have exactly 4 legs and only ~880 have <4 — the
  test split conceals the 4th leg of 4-leg trips.
- **Temporal flow: 99.11% of consecutive reservations inside a trip are
  CONTIGUOUS — next checkin == previous checkout** (gap 0..33 days).
  This is the sequence-flow logic of the dataset: a trip is an ordered
  chain of stays, checkout_i == checkin_{i+1} (nearly) always.
- Nights per stay (checkout−checkin): median 1, mean 1.74, max 30, min 1.
- 1.11 trips per user (median 1, max 27): a user ≈ one trip.
- **38.07% of trips revisit a city** already visited in the same trip
  (repeat-city motif — the return-trip structure the generator must carry).
- Only 9.51% of reservations are domestic (booker_country == hotel_country):
  overwhelmingly an international travel dataset.
- booker_country is concentrated: Gondal 46%, The Devilfire Empire 25%,
  Elbonia 20%; 99.8% of users book from ONE country only (529 users from 2).
- Top hotel countries: Cobra Island, Fook Island, Gondal, Glubbdubdrib,
  Elbonia ... (195 total); top city 47499 has 15,084 reservations.
- device: desktop 65%, mobile 28%, tablet 7%; 27.6% of trips use ≥2 device
  classes (device per leg is a real signal, not per-user constant).
- affiliate: 3,611 ids, heavy head (9924 24%, 359 15%, 384 7.6%).
- Day-of-week of checkin is ~uniform (min/max ratio 1.04) — a public
  uniform prior is honest here (unlike hour-of-day in other datasets,
  dates carry no time-of-day at all).
- utrip_id prefix always equals user_id (1545502/1545502).

## Modelling consequences (frozen in booking-contract.md)

1. The privacy unit is the USER; a synthetic record = one user's trip.
2. The generative backbone is the trip: draw legs-per-user, then walk
   hotel-country tiers with a DP transition matrix + return-first
   second-order behavior (38% repeat cities), then place contiguous dates,
   nights, month (seasonality), device/affiliate per leg, booker country
   per user.
3. Chi² associations worth releasing: device × booker country, device ×
   hotel tier, affiliate tier × device, affiliate tier × booker country,
   booker country × hotel tier (see `mart_feature_assoc`).
4. Temporal flow is enforced BY CONSTRUCTION in the generator
   (checkin_{i+1} = checkout_i + gap, gap ~ DP gap cells, 99% zero) and
   re-checked by `IsWellFormedTrip`/`IsContiguousTrip` in Lean and the
   qf_overlap flag in staging.
5. 39,901 cities cannot be DP-released individually: city ids come from
   the suppressed top-k lists (`mart_topk_cities`,
   `mart_topk_city_transitions`, ≥50 users / ≥100 rows) plus synthetic
   ids elsewhere (declared labelling device).

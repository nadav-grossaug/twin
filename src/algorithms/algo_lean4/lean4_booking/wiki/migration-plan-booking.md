---
title: migration-plan-booking
type: plan
updated: 2026-11-03
sources: [dataset-booking, booking-contract, migration-plan-foursquare, migration-plan-snap]
---

# migration-plan-booking — foursquare/travel pipeline → dataset-booking (Booking.com trips)

Sixth instantiation of METHOD.md (pcap → stock → foursquare → geolife →
snap → **booking**). The substitution base is the foursquare travel build
(same six-flow ops substrate, LEAN_HOME store, ONE source-only bundle),
with the snap build consulted for the χ²/assoc mart pattern and notebooks.
Cross-layer interfaces are FROZEN in [[booking-contract]] — read that
first; this page records the plan and progress.

## Substitution map (METHOD §12 applied)

| foursquare (base) | booking (this build) |
| --- | --- |
| transfer TSV (country-boundary hops) | reservation rows; utrip = ordered chain of stays |
| privacy unit: user | privacy unit: user (whole reservation history) |
| region map seed (8 ISO regions) | country_tiers seed (top-15 hotel countries + ct_other) |
| travel-count bins tc_* | reservations-per-user bins rc_* (edges 4/5/6/7/9/13) |
| region/region_transition 8+64 cells | country/country_transition 16+256 cells (ε 3/100 each) |
| gap classes in seconds (5) | gap classes in DAYS (4: g_0 contiguous / g_1 / g_2_3 / g_4p) |
| duration classes (5) | nights bins (5: n_1..n_7p, range 1–30) |
| month + hod histograms | month (12) + dow (7, measured ~uniform → honest uniform prior) |
| episode_len bins | — (dropped; trip = the episode; rc bins carry it) |
| return-rate bands | return-rate bands (repeat-city trip share, 38% motif) |
| venue/coordinates/centroid table | — (no geometry; cities are opaque ids, top-k released) |
| mart_topk_corridors / topk_countries | mart_topk_city_transitions / mart_topk_cities (suppression ≥50 users / ≥100 rows) |
| motifs: returns/corridors/tours/bursts/longstays | motifs: repeat-city returns / corridors (≥3 repeats, CV ≤ 4/5) / tours (≥3 countries, ≥4 legs, gaps ≤ 7d) / longstays (nights ≥ 7) — bursts dropped (contiguity makes them degenerate) |
| hod (24 cells) | dropped — dates carry no time-of-day |
| distance classes | dropped — no coordinates in the challenge data |

## Phases

1. **Profile + contract** — ✅ done 2026-11-03: [[dataset-booking]] EDA;
   [[booking-contract]] frozen (grids, ledger 1008/100 ≤ 14/1, tables, JSON
   schemas, generator I/O, planted motifs).
2. **dbt layer** — ✅ done 2026-11-04: 170/170 (28 models + 139 tests).
3. **PyTorch minimodels** — ✅ done 2026-11-04: five gated models
   (`05863414d8a7`), honest ties recorded (see [[booking-pipeline]]).
4. **Lean layer** — ✅ done 2026-11-04: full build green, 143/143 checks
   agree, release 252/25 ≤ 14/1, days (not seconds) as the time unit.
5. **Distillation + generator** — ✅ done 2026-11-04: fidelity accepted
   (0.9101 vs 0.8672−0.02); contiguity by construction; planted
   repeat-city/tour/longstay certified `by decide`.
6. **Ops + validation** — ✅ 93/96 tests, both notebooks headless-clean;
   ⏳ export + bundle + synth_from_bundle pending the user-run dev-release
   export.

## Learning targets (from the challenge brief)

- **Next destination in real time**: P(next hotel tier | prev tier) is the
  distilled core (16×16 DP transitions + return-boost + rr-band
  second-order return); city level via suppressed top-k lists.
- χ² co-occurrence: device × booker country, device × hotel tier,
  affiliate tier × device, affiliate tier × booker country,
  booker × hotel tier, month × hotel tier.
- Timely flow: leg ordering, checkout→checkin chaining (99.11%),
  seasonality (month cells), contiguity as a generator invariant.

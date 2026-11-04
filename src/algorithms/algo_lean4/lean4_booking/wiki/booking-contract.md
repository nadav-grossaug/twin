# booking-contract — frozen cross-layer interfaces (v1.4.0)

FROZEN 2026-11-03 before parallel implementation; bumped **1.0.0 → 1.1.0 →
1.2.0 → 1.3.0 → 1.4.0**. The 1.3.0 changes (second fidelity pass — new
minimodels/motifs/features): three new STRUCTURE releases the per-leg
marginals hide — trips_per_user (tp_1/tp_2/tp_3p; real 91.1/7.4/1.5%, the
single_trip prior rendered 1.00), device_persist (dv_1/dv_2/dv_3; 70.7% of
real trips are single-device, iid draws would render ~12%),
affiliate_persist (ap_1..ap_4p; 75.4% of legs keep the previous leg's
tier); three new gated minimodels (trip_count, device_seq, aff_seq); two
new decidable motifs (IsConstantRun — device/affiliate-constant trips,
re-decided by lake exe checks); multi-trip rendering with balanced splits
and a median-matched inter-trip-gap prior (uniform [1,94]d); per-segment
synthetic ids for the suppressed affiliate tail. Ledger 1368/100 = 13.68. The **1.4.0 changes** (four measured-gap
fixes): NEW releases countries_per_trip (cpt_1..cpt_4p — real trips are
72.2% single-country, mean 1.41; the unconstrained walk rendered 2.26),
device_plan (1:*/2:*/3:all — real 1-device trips are 70.9% desktop vs the
64.6% marginal), domestic_start (per-booker first-leg-domestic counts, real
18.7% Gondal → 2.1% Devilfire; calibrated by domestic_stay_scale_pct 57 so
the ALL-LEG international rate matches); top-k suppression thresholds
50/100 → 3/5 (release now 18,204 cities / 164 countries / 32,119 corridors;
a synth sample covers cities ∝ its size — 20k users → ~12k cities); the
IsSingleCountryTrip motif; budget-raised ledger **1440/100 = 14.4 ≤ 16/1**. The 1.1.0 changes: rc_13p DRAW range narrowed 64 → 24
(`rc_13p_draw_hi`), `synth_repeat_tol_pct` added, visited-city exclusion in
the generator's city draw, hop-1 returns render a consecutive same-city
stay (23.7% of real trips contain one). The **1.2.0 changes are the
owner-ordered ε REALLOCATION** (fidelity over privacy, bounds may worsen;
released STRUCTURE is the fidelity lever): month-of-year (12 cells) →
**month-of-window** (15 cells `mw00..mw14`, the real observation ramp);
NEW **repeat_struct** (6 cells `r_ext1/r_ext2/r_ext3p/r_ret1/r_ret2p/
r_both` — the consecutive-extension vs revisit anatomy of repeat trips);
country_transition ε 3/100 → **4/100**; return_rate 4/100 → **6/100**;
**dow DROPPED** (measured uniform, never enforced). Ledger:
**1315/100 = 13.15 ≤ 14/1**. The DP table shapes and all other grids are
unchanged. Every layer implements EXACTLY these names; changes need a
contract bump and a `wiki/log.md` note.

## Identity

- dbt project & profile: `enclave_trip`; DuckDB: `artifacts/enclave.duckdb`
- input env var: `BOOKING_CSV` (default `../combined_set.csv`), columns
  `user_id,checkin,checkout,city_id,device_class,affiliate_id,booker_country,hotel_country,utrip_id`
- Lean namespace: `EnclaveTrip`; lean dir `lean/` (toolchain v4.34.1, mathlib +
  SampCert pin copied from ../lean4_foursquare/lean)
- spec_version `1.0.0`; genspec_version `1.0.0`; policy `release-1.0.0-trip`
- dev-release env var: `BOOKING_ALLOW_DEV_RELEASE=1`
- privacy unit: ONE USER (all their reservations are one record)
- time unit: DAYS (epoch day numbers, DATE 1970-01-01 = 0);
  public window `window_start_day = 16770` (2015-12-01),
  `window_end_day = 17226` (2017-03-01, exclusive)

## Public DP grids (constants; every cell always emitted, zero included)

    RC_BINS  (main, per-user total reservations; clip 1, ε 4/100):
      ["rc_1_3","rc_4","rc_5","rc_6","rc_7_8","rc_9_12","rc_13p"]
      ranges: [1,3],[4,4],[5,5],[6,6],[7,8],[9,12],[13,64]
    CT_CELLS (hotel-country tiers; clip 8, ε 3/100): the 15 named countries
      + "ct_other":
      ["Gondal","Cobra Island","Fook Island","Glubbdubdrib","Elbonia",
       "Borginia","Kangan","Bozatta","Alvonia","Holy Britannian Empire",
       "Axphain","Santa Prisca","Patusan","Rolisica","Kasnia","ct_other"]
      seed dbt/seeds/country_tiers.csv: country,tier (15 rows; unmapped -> ct_other)
    TRANS_CELLS: "{a}__{b}" for a,b in CT_CELLS  (256 cells; clip 8, ε 3/100)
    NIGHTS_CELLS ["n_1","n_2","n_3","n_4_6","n_7p"] (clip 8, ε 6/100)
      ranges [1,1],[2,2],[3,3],[4,6],[7,30]
    GAP_CELLS ["g_0","g_1","g_2_3","g_4p"] (clip 8, ε 6/100)
      ranges [0,0],[1,1],[2,3],[4,33]  (gap_days = next checkin - prev checkout)
    MONTH_CELLS ["m01".."m12"] (clip 8, ε 1/100; calendar month of checkin)
    DOW_CELLS ["d0".."d6"] (clip 8, ε 1/100)
    DEVICE_CELLS ["desktop","mobile","tablet"] (clip 8, ε 5/100)
    AFF_CELLS ["a9924","a359","a384","a9452","a4541","a8436","a10332",
               "a3631","aff_other"] (clip 8, ε 4/100)
      seed dbt/seeds/affiliate_tiers.csv: affiliate_id,aff_tier (8 rows; else aff_other)
    BOOKER_CELLS ["Gondal","The Devilfire Empire","Elbonia","Tcherkistan",
                  "Bartovia"] (clip 1 — one cell per user; ε 4/100)
    RR_BANDS ["rr_0_19","rr_20_39","rr_40_59","rr_60_79","rr_80_100"]
      (per-user share of trips with a repeat city; clip 1, ε 4/100)

## ε ledger (per-cell ε = EPS1/100; total must stay <= 14/1 — the user
explicitly chose fidelity over privacy AND ordered the 1.2.0 reallocation;
state the bound is weak, ε ≈ 13.15)

    7*(4/100) rc  + 16*(3/100) country + 256*(4/100) country_transition
  + 5*(6/100) nights + 4*(6/100) gap + 15*(2/100) month_window
  + 3*(5/100) device + 9*(4/100) affiliate + 5*(4/100) booker
  + 5*(6/100) return_rate + 6*(5/100) repeat_struct
  = 28+48+1024+30+24+30+15+36+20+30+30 = 1315/100 (v1.2.0)
  + 3*(6/100) trips_per_user + 3*(5/100) device_persist
  + 4*(5/100) affiliate_persist = 1368/100 = 13.68 <= 14/1 (v1.3.0)

Aux histogram names in dp_counts.json / dp_release.json (ORDERED):
`res_count` is the MAIN histogram; aux = ["country", "country_transition",
"nights", "gap", "month_window", "device", "affiliate", "booker",
"return_rate", "repeat_struct", "trips_per_user", "device_persist",
"affiliate_persist"] (dow was dropped in 1.2.0; month-of-year replaced by
month-of-window).

## dbt relations (enclave_trip)

- staging: `stg_reservations` — userid, utrip, leg_idx (1-based by checkin),
  checkin (DATE), checkout (DATE), nights (INT), city_id, device_class,
  affiliate_id, booker_country, hotel_country, hotel_ct, aff_tier,
  month_cell, dow_cell, nights_cell, rc? no. flags:
  qf_bad_dates (checkout<=checkin), qf_overlap (checkin_next < checkout_prev)
- features:
  - `feat_reservations` — stg + per-leg prev_* (prev_city_id, prev_country,
    prev_ct, gap_days, is_contiguous (gap_days=0), city_return (city equals
    any EARLIER city of the same utrip))
  - `feat_transitions` — one row per consecutive leg pair: userid, utrip,
    from_leg, prev_city_id, next_city_id, prev_country, next_country,
    prev_ct, next_ct, gap_days, is_contiguous, city_return, month_cell,
    dow_cell, nights_next
  - `feat_trips` — per utrip: userid, n_legs, n_cities, n_countries,
    start_day, end_day, span_days, total_nights, has_repeat_city, is_tour,
    all_contiguous, nights_max, has_longstay
  - `feat_users` — per userid: n_res, n_trips, n_cities, rc_bin, booker_country,
    rr (repeat-city trip share), rr_band
- marts (DP inputs emit EXACTLY their public grid; columns
  `cell, clipped_count, n_contributing_users, clip`):
  mart_dp_res_count, mart_dp_country, mart_dp_country_trans,
  mart_dp_nights, mart_dp_gap, mart_dp_month, mart_dp_dow, mart_dp_device,
  mart_dp_affiliate, mart_dp_booker, mart_dp_return_rate
- association (chi2 / Cramér's V over reservation-level feature pairs,
  candidates: device_class x booker_country, device_class x hotel_ct,
  aff_tier x device_class, aff_tier x booker_country,
  booker_country x hotel_ct, month_cell x hotel_ct):
  `mart_feature_assoc` (feat_a, feat_b, chi2, dof, cramers_v),
  `mart_assoc_cells` (feat_a, feat_b, val_a, val_b, observed)
- sequence/motif marts (exact, unnoised, release-DENIED):
  `mart_seq_transitions` (next_ct given prev_ct counts),
  `mart_motif_returns` (repeat-city trips: userid, utrip, cities, days),
  `mart_motif_corridors` (userid, prev_country, next_country, gaps_days,
    n_repeats, is_corridor),
  `mart_motif_tours` (userid, utrip, days, countries, is_tour),
  `mart_motif_longstays` (userid, utrip, leg, nights, is_longstay),
  `mart_user_stats` (denied), `mart_trip_stats` (denied aggregates)
- suppression-only release artifacts:
  `mart_topk_cities` (city_id, hotel_country, hotel_ct, n_users, n_res;
    suppress >= 50 users AND >= 100 res),
  `mart_topk_city_transitions` (prev_city, next_city, hotel_ct, n_users,
    n_trans; suppress >= 50 users AND >= 100 trans),
  `mart_gap_moments` (gap_class, n_gaps, mean_days, median_days, sd_days;
    suppress n_gaps >= 200)
- quality: `q_data_integrity` (nulls, qf_bad_dates=0, unique (utrip,leg_idx),
  window membership)
- ml: `ml_trip_scores` (dbt python model; per-trip minimodel scores)

## Lean layer

- `lean/data/` (written by python/export_lean_data.py):
  - `dp_counts.json` — foursquare schema: main `cells` = RC_BINS; `aux`
    list with the 10 names above, each {name, clip, eps1, eps2, cells}
  - `motifs.json` — {spec_version, params, returns, corridors, tours, longstays}
  - `assoc.json` — {pairs: [feat_a, feat_b, sql_chi2, dof, cramers_v, rows,
    cols, observed]} (top 6 by cramers_v; Lean recomputes chi2 exactly)
  - `trips.json` — {seqbreak_gap_d, trips: [{user, utrip, days: [checkin
    epoch days], nights, cities, countries, sql_all_contiguous}]} (top trips;
    Lean re-decides IsWellFormedTrip / IsContiguousTrip)
- executables: `lake exe checks|release|sample <N>|generate <genspec> <n_users> <out_dir>`
- `lean/out/`: dp_release.json, samples.json, lean_results.json, genspec.json
- `lean_results.json` keys: claims, agreement_counts, returns, corridors,
  tours, longstays, assoc, trips
- Motif predicates (EnclaveTrip.Motifs; names used by GenCert):
  `HasRepeatCity (cities : List String)` (some city occurs twice),
  `IsCorridor (min_repeats : Nat) (gaps : List Nat)` (>= min_repeats gaps,
    CV <= 4/5 — port of the foursquare corridor),
  `IsTour (days : List Nat) (countries : List String)`
    (>= 3 distinct countries, >= 4 legs, every consecutive gap <= 7 days),
  `IsLongStay (nights : List Nat)` (some nights >= 7),
  `IsWellFormedTrip` / `IsContiguousTrip` (Sequences: strictly increasing
    checkin days, no overlap; contiguous variant gap=0 on every pair)
- DPRelease: discrete-Laplace over each grid via SampCert; per-user clip
  as in the JSON; proves the composed bound (histMech_DP pattern).

## Minimodels (PyTorch; registry models_out/minimodels/<hash12>/)

files: `next_country.pt`, `trip_len_class.pt`, `stay_nights.pt`,
`device_class.pt`, `profile_ae.pt` (+ meta.json, registry.json with
`active` pointer; deterministic user-hash train split `is_train_user(uid)`).
- next_country: teacher for the next hotel-country tier; distilled ->
  DP country_transition + return_boost diagonal prior
- trip_len_class: per-user res-count bin; distilled -> main RC bins
- stay_nights: per-leg nights cell; distilled -> DP nights
- device_class: per-leg device cell; distilled -> DP device
- profile_ae: scoring-only (realism audit), never distilled
`models_out/fidelity.json`: student(first-order CT argmax) vs teacher vs
return-to-previous baseline on held-out transitions, ACCEPT_MARGIN 0.02.

## Genspec (lean/out/genspec.json; written by python/distill.py)

keys: genspec_version, kind, spec_version, source{dp_release_sha256,
privacy}, n_users_hint, res_count{cells}, country_mix{cells},
transitions{rows:[{given,cells:[{name,w}]}]}, nights{cells}, gap{cells},
month{cells}, dow{cells}, device{cells}, affiliate{cells}, booker{cells},
return_rate{cells}, cities[{city_id,country,ct,w}],
corridors[{a_city,b_city,ct,n_users,n_trans}], gap_moments[{gap_class,
mean_days,sd_days}], priors[return_boost, uniform_dow, single_trip],
window{start_day,end_day}, rc_bin_ranges, gap_ranges_d, nights_ranges.
Priors (declared derived_from_data:false):
- return_boost: transition row r: diagonal += max(1, row sum); return-first:
  per user draw rr band, p = band midpoint, next city = previous city with
  prob p (second-order return).
- uniform_dow: dow histogram released but not enforced; start day-of-month
  uniform in 1..28 of the drawn month.
- single_trip: one synthetic utrip per user (multi-trip users declared in
  fields_not_modelled).

## Generator I/O (`lake exe generate`)

- writes `<out>/booking_synth.csv` with EXACTLY the input schema header
  `user_id,checkin,checkout,city_id,device_class,affiliate_id,booker_country,hotel_country,utrip_id`
  (utrip_id = "<user>_1"), dates ISO from epoch days, contiguous by
  construction (checkin_{i+1} = checkout_i + drawn gap, mostly 0), nights in
  [1,30], and `<out>/plan.json`:
  {"users":[{"user":i,"n_res":m}],
   "planted":{"return":{"user":1,"cities":[...],"days":[...]},
              "tour":{"user":2,"countries":[...],"days":[...]},
              "longstay":{"user":3,"nights":[...]}}}
  user 1 = trip with a repeat city, user 2 = 4-leg 3-country contiguous
  tour, user 3 = trip containing a >=7-night stay.
- `scripts/synth.py generate|validate|certify` (validate re-ingests the CSV
  through the production dbt pipeline into a scratch DB; GOF TV vs genspec
  for res_count/country_mix/nights/gap mixes; planted-motif recall must be
  1.0; no qf_overlap / qf_bad_dates rows; Lean checks re-run on the
  re-exported observations).

## Release policy (config/release_policy.yaml)

policy release-1.0.0-trip; require_attestation true;
allow_dev_platform_env BOOKING_ALLOW_DEV_RELEASE; dp_total_epsilon_max "14/1";
dp_releases keys: res_count_histogram, country_histogram,
country_transition_histogram, nights_histogram, gap_histogram,
month_histogram, dow_histogram, device_histogram, affiliate_histogram,
booker_histogram, return_rate_histogram; artifacts: topk_cities,
topk_city_transitions, gap_moment_stats, lean_claims; model_weights
deny_unless_reviewed; denied: stg_reservations, feat_*, mart_seq_transitions,
mart_user_stats, mart_motif_*, ml_trip_scores, mart_assoc_cells.

## Makefile (six flows, snap/foursquare shape)

clean, train (= dbt train-models ml lean distill test manifest export
bundle), synth, synth_from_bundle, notebook-enclave, notebook-synth, plus
help/certify/realism-audit/notebook-html/test; N_USERS ?= 2000,
CERT_N ?= 120, REPLICATES ?= 8; LEAN_HOME policy as in the other builds.

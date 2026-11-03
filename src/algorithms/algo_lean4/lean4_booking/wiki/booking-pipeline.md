---
title: booking-pipeline
type: component
updated: 2026-11-03
sources: [booking-contract, migration-plan-booking, dataset-booking]
---

# booking-pipeline — the as-built lean4_booking pipeline (status: in progress)

Pipeline: `combined_set.csv` → dbt/DuckDB (`enclave_trip`) → PyTorch
minimodels → Lean 4 `EnclaveTrip` (+SampCert) → ε-DP release → distill →
`lake exe generate` → synthetic `booking_synth.csv` (source schema),
re-ingested and re-decided. Interfaces frozen in [[booking-contract]].

## Layer map

- **dbt** — `stg_reservations` (leg_idx, epoch days, tier joins, month/dow/
  nights cells, qf flags) → `feat_reservations/transitions/trips/users` →
  11 grid-complete `mart_dp_*` inputs, `mart_assoc_cells` +
  `mart_feature_assoc` (χ²/Cramér's V), motif + sequence marts (exact,
  denied), `mart_topk_cities` / `mart_topk_city_transitions` /
  `mart_gap_moments` (suppression-only releases), `q_data_integrity`,
  `ml_trip_scores`.
- **PyTorch** — next_country (teacher; gate: return-to-previous),
  trip_len_class, stay_nights, device_class, profile_ae (scoring-only);
  content-addressed under `models_out/minimodels/<hash12>/`, deterministic
  user-hash split, `BOOKING_RETRAIN=1` retrain gate.
- **Lean** — `EnclaveTrip/{Params,Stats,Motifs,Sequences,Assoc,DPRelease,
  GenSpec,Sampler,TripShape}` + checks/release/sample/generate; time unit =
  epoch days; SampCert discrete Laplace; ledger 1368/100 = 13.68 ≤ 14/1
  (fidelity-over-privacy choice, bound weak, re-releases compose).
- **Generator** — trips per user from the tp release (1.11 real; was
  single_trip), contiguous dates by construction, repeat anatomy from the
  rr bands + repeat_struct, tier walk from the DP transitions +
  return-boost diagonal, per-trip device/affiliate plans from the
  persist releases, cities from suppressed top-k lists (synthetic ids
  elsewhere), planted repeat-city / tour / longstay motifs certified
  `by decide`.

## Results (spec 1.3.0 run, 2026-11-04 — minimodels/motifs/features pass)

- **dbt**: 191/191 incl. five structure-release marts (month_window,
  repeat_struct, trips, device_persist, aff_persist); ml mart from the
  active registry.
- **Lean**: full build green; `lake exe checks` 143/143 SQL↔Lean agreement;
  ε ledger **1368/100 = 342/25 = 13.68 ≤ 14/1** per export (1.2.0
  reallocation + the 1.3.0 trips/device/affiliate structure releases);
  lifetime ledger enforced at 42/1.
- **Distill**: genspec 1.3.0 (spec-version-keyed staleness guard); fidelity
  accepted; 5 gated minimodels + 3 new (trip_count, device_seq, aff_seq).
- **Synth** (2,000 users): validate PASS — planted recall 1.0 ×3, 0
  overlaps, contiguity 0.9898 (real 0.9911), TVs ≤ 0.021, 0 Lean
  disagreements; **replicates PASS ×8** (repeat share 0.3957, rc TV
  0.0159); certificate PROVED.
- **Realism audit (the 1.3.0 payoff)**: trips-per-user share **0.096 vs
  real 0.0893** (was 0.0000), single-device trips **0.7225 vs 0.7067**
  (iid draws would render ~12%), consecutive same-city legs 0.0974 vs
  0.0901, affiliate same-prev-tier 0.848 vs 0.754 (residual = within-trip
  tier ORDER — count-only release, declared gap), records/user 5.83 vs
  5.96, repeat share 0.394 vs 0.378, month-window TV 0.035; 0% above the
  real AE p99.
- **Tests**: 102/102; both notebooks export headless clean (4 new
  scorecard rows: trips/user, single-device, tier persistence,
  consecutive legs).

## Open items

- **Bundle is three specs behind** (`dist/bundle` = the 1.0.0 export).
  The next `BOOKING_ALLOW_DEV_RELEASE=1 make export && make bundle`
  refreshes it under 1.3.0 and records ledger entry #2 (cumulative
  1008/100 + 1368/100 = 2376/100 = 23.76 ≤ 42/1).
- Minimodel ties: next_country = return-to-previous, trip_len/stay_nights ≈
  majority — the teachers add nothing beyond the release (recorded
  honestly; distill replaces them by design).
- Corridor motif is near-degenerate on this data (1.11 trips/user) — kept
  for method parity; candidate for a within-trip redefinition in a future
  spec bump.
- Declared gaps (synth meta `fields_not_modelled`): within-tier city
  choice beyond the top-k lists, dow placement, sub-day time,
  within-trip affiliate ORDER beyond the per-trip tier-count plans,
  cross-leg correlations beyond the released associations. The 1.2.0/1.3.0
  passes CLOSED the former month-of-year, consecutive-stay, multi-trip,
  device-switching and within-bin gaps.

---
title: GeoLife pipeline (as built)
type: component
updated: 2026-11-02
sources: ["../goal.md", "[[migration-plan-geolife]]", "make train/synth runs 2026-11-02"]
---

# geo-pipeline — the as-built lean4_geolife pipeline

Fourth instantiation (flows → stock → travel → GeoLife), closing the loop to
the method's reference implementation. Port base: `../lean4_foursquare`
([[travel-pipeline]]); domain logic re-derived from [[geolife-reference]].
Working checklist: [[migration-plan-geolife]]; data: [[dataset-geolife]].

## Layer map

| Layer | As built |
| --- | --- |
| dbt (29 models, 91 tests) | `stg_points` (4.87M kept / 5.44M raw; 574,708 dup rows collapsed, 1 invalid coord dropped, all counted) → `feat_steps`/`feat_legs` → `feat_staypoints` (Python node: Li-et-al staypoints + **implied inter-trajectory stays**, dep-censored on place mismatch) → `feat_anchors` (58/61 home, 37 work) → `feat_visits` → `feat_user_days` → motif/assoc/seq/noise/kinematics/dist marts → 5 DP-input marts (proportional-L1 clip macro, clip 10) |
| Minimodels (all ADOPTED) | NextRoleGRU 0.697 vs return-to-previous 0.672; DwellClassGRU 0.564 vs majority 0.480; ModeLegMLP 0.801 vs speed-rule 0.677; UserProfileAE (scoring-only); content-addressed, registry-activated |
| Lean formal layer | `EnclaveGeo`: daily-motif classifier as the spec (priority tour > commute > errand_loop > home_day > irregular; totality theorem `motifClassOfDay_mem`), revisit beacon, visit-sequence chain lemmas, χ² over ℚ, SampCert mechanism (`histMech_DP`), verified sampler (`sampleCellPMF_law`), generator plan theorems (`mkStays_times_sorted` / `mkStays_isStay` / `mkStays_visitSeq`), Motion linking theorems (`wobble_box_fits`, `trip_interval_le_gap`, `genTrip_chains`, `render_speed_within_cap`, `modeRep_roundtrip`, `excl_tgt_disjoint`). 30 proved + 3 assumed claims; no sorry, no native_decide |
| SQL↔Lean checks | `lake exe checks`: 2,582 user-day motifs re-decided + 1,955 revisit anchors + 10,000 class-ladder cells + 2,000 long-dwells + 432 visit runs + 6 exact-ℚ χ² — **0 disagreements** (16,975 checks) |
| ε-DP release | ledger EXACTLY 400/100 = 4/1: motif 5×(30/100), transitions 9×(15/100), dwell 5×(13/100), departure 4×(5/100), mode 6×(5/100); clips 10 (proportional L1); SampCert discrete Laplace in `lake exe release`; refusal path tested; staypoints/day + displacement deliberately DROPPED to priors/category-2 (n = 64: SNR = share×n×ε_cell) |
| Category-2 (suppression, declared) | noise moments (tight staypoints: sd 23.2 m, MAD 14.2 m, lag-1 0.78), per-mode kinematics, 45° turn histogram, displacement mix, **route shape** (per-mode circuity walk 1.99 → rail 1.27, junction density walk 11.1/km → rail 0.2/km, left share 0.50; segment + junction-angle histograms over public bins: junctions ≥ 30° on a 25 m-resampled stream, segments median 50 m) |
| Distillation | genspec from the release ONLY; declared priors: motif floor-at-1, home-return boost max(1, row_sum//2); fidelity HONEST NEGATIVE (first-order role student 0.39 < return-to-previous 0.65; teacher 0.70) — the generator compensates with motif templates, not private data |
| Generator (Lean-only) | `lake exe generate`: neutral local frame (0°N 0°E isotropic), per-user anchor frames at golden-angle bearings, daily motif templates + DP role walks, **road-like trip polylines** (segment lengths from the released segment histogram scaled by the mode's released junction density; junction angles from the released junction-angle histogram, jittered off the 90° lattice; left/right from the released share with deviation clamped ±75°; path budget from released per-mode circuity; affine-fitted onto the anchor pair; endpoint-snapped), AR(1) GPS noise matched to released moments (moving noise attenuated 8×, declared prior), wobble clamped into the detection box; all randomness through the verified sampler |
| Manipulate (mandatory) | `synth.py manipulate`: per-user deterministic placement on 12 public Tokyo district centroids (cos-corrected); hard asserts: **0 points in the Beijing bbox**, all in the Tokyo bbox |
| Validation | production-dbt re-ingest; planted recall 1.0 (commuter 001 / tour + revisit beacon 002); TV GOF vs released weights; structural + path-realism checks (collinear share ~1e-4, ±90° atoms ~2e-3); kernel-checked `GenCert.lean` (`by decide`, 4 theorems) |
| Ops | six make flows + `make help`; `LEAN_HOME=/opt/data/lean` shared elan/cache/package store; ONE source-only bundle (240K) with `build_from_source.sh` (kernel re-checks every theorem); pytest 130 tests; run manifest (bit_xor row hashes); policy export (default-deny, attested or `GEO_ALLOW_DEV_RELEASE=1` marked unattested) |

## Flow results (2026-11-02, clean-slate)

- `make train`: green end-to-end — dbt build, 4/4 minimodels adopted, Lean
  checks 0 disagreements, release ε spent exactly 4/1, distill (+ honest
  fidelity negative), pytest green, manifest, export (8 files), 220K bundle.
- `make synth` (n = 64): 308,130 points → Tokyo manipulate → validate PASS
  (exit 0): planted recall 1.0, mode TV 0.004, motif TV 0.30, collinear
  2e-4, no ±90° atom, noise in band, 0 Beijing points, 0 re-ingested Lean
  disagreements. `make certify`: PROVED (kernel `decide`).
- `make synth_from_bundle`: bundle kernel re-check (7,806 jobs), bundle-mode
  distill (no db), 316,104 points, validate PASS. pytest final: 110 passed,
  0 skipped; both notebooks export headless with 0 tracebacks.

## Declared gaps (meta.json `assumptions.fields_not_modelled`)

street/transit geometry; altitude dynamics (sentinel −777 only); label-window
sub-structure; cross-user relations; within-bin dwell shapes beyond released
moments; home-day device-off behaviour (an unobserved day is unobserved in
synth too — parity with the source logger).

Links: [[dp-release]], [[motifs]], [[sequence-analysis]],
[[feature-associations]], [[synthetic-generation]], [[enclave-boundary]],
[[practices]].

---
title: GeoLife pipeline (as built)
type: component
updated: 2026-11-03
sources: ["../goal.md", "[[migration-plan-geolife]]", "make train/synth runs 2026-11-02", "spec 4.1.0 / release-3.0.0 runs 2026-11-03", "[[log]]"]
---

# geo-pipeline — the as-built lean4_geolife pipeline

Fourth instantiation (flows → stock → travel → GeoLife), closing the loop to
the method's reference implementation. Port base: `../lean4_foursquare`
([[travel-pipeline]]); domain logic re-derived from [[geolife-reference]].
Working checklist: [[migration-plan-geolife]]; data: [[dataset-geolife]].
Current spec **4.1.0**, release policy **release-3.0.0-geolife**.

## Layer map

| Layer | As built |
| --- | --- |
| dbt (29 models, 91 tests) | `stg_points` (4.87M kept / 5.44M raw; 574,708 dup rows collapsed, 1 invalid coord dropped, all counted; **source timestamps are UTC — shifted once by `source_utc_offset_s` = 28800 to Beijing local, so every hour/day/weekday feature downstream is local**) → `feat_steps`/`feat_legs` → `feat_staypoints` (Python node: Li-et-al staypoints + **implied inter-trajectory stays**, dep-censored on place mismatch) → `feat_anchors` (58 home, 44 work) → `feat_visits` → `feat_user_days` (2,566 days) → motif/assoc/seq/noise/kinematics/dist marts → 5 DP-input marts (proportional-L1 clip macro, clip 10) |
| Minimodels (all ADOPTED) | retrained on spec 4.1.0 (active `29e1c569aa77`): NextRoleGRU 0.627 vs return-to-previous 0.593; DwellClassGRU, ModeLegMLP, UserProfileAE (scoring-only) adopted against their frozen baselines; content-addressed, registry-activated (`FLOW_RETRAIN=1` to activate a retrain) |
| Lean formal layer | `EnclaveGeo`: daily-motif classifier as the spec (priority tour > commute > errand_loop > home_day > irregular; totality `motifClassOfDay_mem`), revisit beacon, visit-sequence chain lemmas, χ² over ℚ, SampCert mechanism (`histMech_DP`), verified sampler (`sampleCellPMF_law`), generator plan theorems (`mkStays_*`), Motion linking theorems (`wobble_box_fits`, `trip_interval_le_gap`, `genTrip_chains`, `render_speed_within_cap`, `modeRep_roundtrip`, `excl_tgt_disjoint`, and 2026-11-03: `activeDayClasses_valid`, `planted_slice_covers_beacon`, `planted_departure_reaches_work_window`). 35 exported claims; no sorry, no native_decide |
| SQL↔Lean checks | `lake exe checks`: 2,566 user-day motifs + 1,955 revisit anchors + 16,000 class-ladder cells + 2,000 long-dwells + 432 visit runs + 7 exact-ℚ χ² — **0 disagreements** |
| ε-DP release | **release-3.0.0 (owner decision 2026-11-03: fidelity over privacy):** ε_cell = 1/1 in every histogram — motif 5, transitions 9, dwell 5, departure 4, mode 6 cells → **29/1 per export**, noise scale = one user's clip (10). Released vs true clipped TV: departure 0.027, dwell 0.028, motif 0.057, transitions 0.069, mode 0.079 (release-2.0.0's ε_cell 0.05–0.30 made departure/mode mostly noise). A **weak** formal guarantee, chosen knowingly. ε values and the budget are read from `config/release_policy.yaml` by `export_lean_data.py` (single source) |
| Lifetime ε ledger | `config/epsilon_ledger.jsonl` (append-only, git-tracked, outside every `make clean`): `20261102T125506Z` ε 4 (release-2.0.0) + `20261103T130405Z` ε 29 = **33/1 = `dp_lifetime_epsilon_max`** — this dataset's budget is FULLY SPENT; any further export is refused until the owner raises the cap again (or voids an export that provably never left the enclave: `release.py --void`) |
| Category-2 (suppression, declared) | noise moments (tight staypoints: sd 23.2 m, MAD 14.2 m, lag-1 0.78), per-mode kinematics (speed, sampling interval, **median path**, point counts), 45° turn histogram, displacement mix, route shape (per-mode circuity walk 1.99 → rail 1.27, junction density walk 11.1/km → rail 0.2/km, left share 0.50; segment + junction-angle histograms) |
| Distillation | genspec from the release ONLY; **release selection**: `$RELEASE_DIR`, else inside the enclave the LIVE `lean/out` release when it differs from the latest export (category-2 rows from the policy-filtered marts), else the latest export; `genspec_meta.release_source` records which. Declared priors: motif floor-at-1, home-return boost max(1, row_sum//2); fidelity HONEST NEGATIVE (first-order role student < return-to-previous) |
| Generator (Lean-only) | `lake exe generate`: neutral local frame (0°N 0°E), per-user anchor frames at golden-angle bearings, daily motif templates + DP role walks, road-like trip polylines from the released route shape; AR(1) GPS noise matched to released moments. 2026-11-03: **trip cadence = released per-mode median dt** (`dt_ds`); stays rendered sparsely (`genStayDtS` = 30 s); **trip mode conditioned on trip length** (released mode weight × log-normal σ=1 plausibility around the released per-mode median path); **recorded days per user from the heavy-tailed public prior `activeDayClasses`** (median ~12, p90 ~96, mean ~25 days; every user forced ≥ 1 observed day); **one `sequence_uuid` per label window** (as in the source); rows/labels/ids written in **UTC** (planning is local). Planted users: 001 commuter, 002 revisit beacon + tour, fixed 24-day slice, fixed public 07–08 h departure; the planted tour drives (car legs) between stops 1.2–5 km apart / from home. All randomness through the verified sampler |
| Manipulate (mandatory) | `synth.py manipulate`: per-user deterministic placement on 12 public Tokyo district centroids; hard asserts: **0 points in the Beijing bbox**, all in the Tokyo bbox |
| Validation | production-dbt re-ingest; planted recall (hard); TV GOF vs released motif / mode / **departure** mixes; **per-mode sampling cadence** within [0.5×, 2×] of release (guards volume); **per-mode point share** vs released kinematics `n_points`; route-shape straightness band; structural + path-realism checks; Lean deciders on the re-export; reports per-user volume spread and the local hour-of-day profile. Kernel-checked `GenCert.lean` (`by decide`, 4 theorems) |
| Realism audit (enclave-only) | real vs synth TVs (mode/speed/motif/dwell/circuity), ModeLegMLP agreement, **total volume ratio, points/trajectories-per-user spread, local hour-of-day TV** |
| Ops | six make flows + `make help`; `LEAN_HOME=/opt/data/lean` shared store; source-only bundle (256K) built ONLY from an export (`release/<id>/`); pytest 144; run manifest; policy export (default-deny, attested or `GEO_ALLOW_DEV_RELEASE=1` marked unattested — run by the owner) |

## Flow results

**2026-11-03 (spec 4.1.0, release-3.0.0, export `20261103T130405Z`)** — latest:
- `make synth` (n = 64): 2,885,038 points, validate **soft tolerance**
  (exit 2): planted recall **1.0**; mode mix TV 0.046, motif 0.138,
  departure 0.212, per-mode point share 0.224; soft failures = walk / other
  route straightness 0.04 vs released 0.50 / 0.96. `make certify` PROVED;
  pytest **144/144**; bundle 256K.
- Realism audit vs the real cohort: total volume **0.59×**, per-user points
  cv 1.16 (real 1.92), local hour-of-day TV 0.27, mode point share TV 0.22.

2026-11-02 (spec 4.0.0, release-2.0.0 — superseded): `make synth` 308,130
points (~5% of the source — the volume bug), validate PASS; pytest 130.
Those numbers were measured with the UTC/local bug and the noisy release.

## Open issues (2026-11-03, ranked)

1. **Volume 0.59× real** — `activeDayClasses` was calibrated before the
   mode/stay fixes removed padded walk points; recalibrate the mean days
   against the released total point count (kinematics `n_points` sum).
2. **Walk / other straightness ≈ 0.04** — hypothesis (unverified):
   same-anchor "trips" (consecutive home stays in the irregular role walk)
   have zero net displacement and now all map to walk, or to `other` when
   every weight underflows and `draw` falls back to uniform.
3. **Car points over-represented** (37% vs 15% released), **rail under**
   (14% vs 17%; 5.5% on the 8-user trial).
4. **Per-user spread** cv 1.16 vs real 1.92 — the day prior is public; a
   released days-per-user histogram would need ε (the budget is spent).
5. Departure mix TV 0.21 — only the first departure of a day is drawn from
   the release; later departures follow from dwell times.

## Declared gaps (meta.json `assumptions.fields_not_modelled`)

street/transit geometry; altitude dynamics (sentinel −777 only); label-window
sub-structure; cross-user relations; within-bin dwell shapes beyond released
moments; home-day device-off behaviour; international trips are also read
at UTC+8.

Links: [[dp-release]], [[motifs]], [[sequence-analysis]],
[[feature-associations]], [[synthetic-generation]], [[enclave-boundary]],
[[practices]], [[log]].

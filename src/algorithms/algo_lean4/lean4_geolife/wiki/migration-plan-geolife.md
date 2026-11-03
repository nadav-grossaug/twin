---
title: Migration plan (travel/flows → GeoLife trajectories)
type: synthesis
updated: 2026-11-03
sources: ["../METHOD.md", "[[dataset-geolife]]", "../goal.md", "[[migration-plan-foursquare]]", "[[geolife-reference]]"]
---

# Migration plan — pipeline → [[dataset-geolife]]

Status 2026-11-03: **implemented; fidelity pass ongoing** — all phases
landed 2026-11-02; on 2026-11-03 a fidelity pass fixed the UTC/local time
bug (spec 4.1.0), the ~5% volume bug, the stale-release distill trap, added
the lifetime ε ledger and re-released at ε 29 (release-3.0.0, owner
decision). As-built state and ranked open issues: [[geo-pipeline]].
Original status 2026-11-02: **in progress** — plan written, data profiled, goal.md
rewritten. `../goal.md` is the full task framing; this page is the working
checklist. METHOD.md §12's substitution map instantiated, then the phases in
METHOD §4 order. The code base being ported is `../lean4_foursquare`
([[travel-pipeline]] — the newest build with the six-flow ops contract);
domain logic (staypoints, mobility physics, manipulate step) is additionally
sourced from the original GeoLife reference ([[geolife-reference]],
`~/tmp/test_torch_lean_arch_01`) — this migration closes the loop back to
the method's birthplace, upgraded to the current contract.

## Substitution map (METHOD §12 instantiated)

| Method concept | GeoLife instantiation |
| --- | --- |
| one record | one GPS point; modeling units: mode-leg, staypoint, anchor visit, user-day |
| protected entity | the user (n = 64 — small-n regime; whole multi-year trace = one record) |
| public taxonomy | 6 mode classes (collapse of 11 labels); anchor roles H/W/O |
| motifs | staypoint, anchors (home/work), commute H→W→H, errand loop, tour, revisit beacon, long dwell |
| nested structure | trips as point sub-tracks: correlated-heading walk + AR(1) GPS noise over public moment summaries |
| top-k common knowledge | none from data; Tokyo target frame + mode taxonomy + bin edges are the public constants |
| shape constants | Motion.lean: per-mode speed caps, turn-angle bins, max collinear run, staypoint drift box |
| rarest-pattern round-trip | plant a tour + a revisit beacon; staypoint detection must re-fire on noisy synth points; recall 1.0 |

## This migration's extra briefs (beyond METHOD)

1. **χ² association analysis** between columns/features/motifs
   (mode × speed, mode × hour, role × arrival bin, dwell × role,
   motif × day type, distance × mode) — full contingency grids, Cramér's V,
   lift; Lean recomputes χ² over ℚ ([[feature-associations]] pattern).
2. **Timely sequence flow**: anchor-visit transitions broken at gaps > 6 h
   and day boundaries ("breaks split transitions" lemma); within-trajectory
   mode-leg chains ([[sequence-analysis]] pattern).
3. **Noise realism**: estimate GPS-noise statistical moments in-enclave
   (staypoint-interior residuals: sd, MAD, p95, lag-1 autocorr, per mode
   class); category-2 release; generator reproduces them (AR(1)).
4. **Path realism**: no straight lines, no strict 90° turns — correlated
   heading random walk; validation checks collinear-run bound and turn-angle
   histogram smoothness (no atoms at 0°/±90°).
5. **Daily-behavior motifs**: staypoints + anchors drive the motif taxonomy
   (home-day / commute / errand loop / tour / irregular per user-day).
6. **Tokyo juxtaposition**: mandatory `synth.py manipulate` step relocates
   each synthetic user's local frame onto Tokyo public geography; **zero
   Beijing lat/lon in synth output** (bbox-asserted).
7. **Simplified ops** (carried over): six make flows + `make help`;
   `LEAN_HOME=/opt/data/lean`; ONE source-only bundle; synth consumes only
   Lean sources + released JSON.

## Phases (each phase's tests lock the previous one)

1. **Profile the data** — ✅ done, [[dataset-geolife]] (64 users, 5.44M
   points; heavy tail: max 734 trajectories/user).
2. **Spec**: dbt `vars:` + `python/spec.py` + `lean/EnclaveGeo/Params.lean`
   + mode-map seed; sync tests. Staypoint 200 m/1,200 s, anchor cell 300 m,
   night 22–06, workday 09–17, gap break 6 h, bins per goal.md.
3. **dbt pipeline**: `stg_points` (quality flags, dedupe) → `feat_steps`,
   `feat_legs` (SQL) → `feat_staypoints`, `feat_anchors`, `feat_visits`,
   `feat_user_days` (Python nodes, Li-et-al staypoints) → `feat_noise` →
   marts (motifs, assoc, seq, noise moments, dp inputs).
4. **Minimodels** (gated): NextAnchorGRU vs return-to-previous,
   DwellClassModel vs per-role marginal, ModeFromKinematics vs speed rule,
   UserDayAE vs mean-recon; `ml_geo_scores` inference node.
5. **Lean mirrors**: Params, Motifs (staypoint/anchor-role/daily motifs,
   decidable, iff-lemmas), Sequences (gap-break lemma), Assoc (χ² over ℚ),
   Stats; `lake exe checks` SQL↔Lean agreement.
6. **DP release** (as planned; superseded 2026-11-03 by release-3.0.0 — ε_cell 1, 29/1 per export, lifetime ledger 33/1, see [[geo-pipeline]]): ledger 400/100 exactly (motif 100, transitions 90,
   departure 40, dwell 50, mode 30, distance 40, staypoints/day 50);
   proportional-L1 clip (`dp_clip_total`); **pre-noise SNR simulation
   REQUIRED before committing** (n = 64!); category-2 noise/kinematics
   with min-5-users suppression; refusal path tested.
7. **Verified sampling**: catsample + law; χ² GOF conformance.
8. **Distill**: genspec from release only; provenance test weight-by-weight;
   declared public priors (`derived_from_data: false`).
9. **Generate**: Lean-only, neutral local frame, validity-by-construction
   theorems (times sorted, speeds within caps, staypoint drift box);
   Motion.lean linking theorems.
10. **Manipulate + validate**: Tokyo placement; bbox assertions; round-trip
    re-ingestion recall 1.0; per-column tests; noise-moment tolerance;
    certificate (`by decide`).
11. **Ops + docs**: six flows green end-to-end; bundle self-contained;
    notebooks headless; wiki/METHOD/HOWTO updated.

## Known risks (from prior builds + this data)

- **n = 64**: noise-dominated small cells — keep grids coarse, declare
  priors, simulate pre-noise (METHOD §6; [[dp-release]]).
- **Staypoint detection on synth**: round-trip recall depends on noise sd ≪
  staypoint radius; the `stayDrift_box_fits` linking theorem guards this.
- **Python dbt nodes** (staypoints) sit on trust boundary 2: DP holds for
  whatever counts are fed in; their meaning is checked by tests, not proved.
- **Duplicate timestamps** (159k groups) must be deduped before step
  features, else dt = 0 ⇒ infinite speeds.
- **Heavy per-user tail** (734 vs median 20 trajectories): proportional-L1
  clip, equal per-user budgets; per-user-average semantics declared.

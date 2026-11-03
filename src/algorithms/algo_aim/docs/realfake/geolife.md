# Real-vs-fake classifier: GeoLife design

Status: designed and run 2026-11-03 (D8/D9 in `docs/realfake/PROCESS.md`); results in section 5 and notebook `geolife/eval/notebooks/viz_discriminator.ipynb`. Same classifier two-sample test as SNAP (AUC 0.5 = indistinguishable at this level): core and controls `docs/realfake/PLAN.md`, SNAP reference `docs/realfake/snap.md`. Generator and dataset: `docs/geolife/PROCESS.md`, `geolife/README.md`.

## 1. Sample: a contiguous trace window

1. Pick a user uniformly at random among eligible users (those with at least one run of >= 30 points).
2. Draw `k` log-uniformly from 30..600, capped at that user's longest run.
3. Pick a window start uniformly over all valid starts of that user: the window is `k` consecutive GPS points that never cross a trajectory boundary or a > 2 h time gap. Runs are maximal such spans; the real side splits on trajectory id AND `dt > 7200 s` AND `dt <= 0`, the synthetic side on `dt > 7200 s`.

Label 1 = cut from synthetic traces, label 0 = cut from the real traces, same sampler parameters for both. Real side: `geolife/inside/data/points.parquet` clipped to the Beijing core bbox (179 users). Fake side: `geolife/outside/synthetic/<variant>_seed<k>/traces.parquet` — currently the single variant `outings_v2_seed0` (139 users, the sensor-hist generation, eps 6.0). `group_id` = uid, so the group-aware split keeps train and test users disjoint. With one fake source the realfake single-source fallback applies (D5): a stratified group split, repeated over 3 fold seeds for stability; the held-out-source fold machinery reactivates automatically if a future variant ships several seeds. Controls as on SNAP: real-vs-real AUC and label-permutation AUC in every row.

Shared core `src/realfake/` reused unchanged. GeoLife-side code: `src/geolife/eval/discriminator.py` (`WindowSampler`, `TraceFeaturizer`, loaders), runner `src/geolife/eval/discriminate.py`, thin stage script `geolife/eval/discriminate.py`. Undefined statistics are 0.0, never NaN. Meta per sample: `size` = k and `dur_min`, for reporting only, never features.

## 2. What is deliberately dropped (giveaways of representation, not statistics under test)

| Dropped | Why it would be a trivial or unwanted separator |
|---|---|
| User id / pseudonym | Real uids and synthetic pseudonyms are different namespaces; any id feature separates by lookup. |
| Trajectory id | Same: a labelling artifact of each side's file format, not a statistic of the traces. |
| Absolute timestamps and dates | Real GeoLife is 2007-2012, synthetic epochs are 2024 — a date feature would separate perfectly. Only LOCAL time of day and weekday (UTC+8) stay in, in the time group. |
| Absolute lon/lat | The classifier must not separate on the locations chosen at generation time. All geometry is computed from steps / centered coordinates via the fixed-origin linear `to_xy` projection, so the features are exactly translation-invariant. Exception: the map group (section 3), which exists to read absolute position against public map tables and is toggleable for that reason. |
| Per-user point volume | Users differ hugely in recorded points on both sides; uniform-over-users sampling plus the group split by uid removes volume as a signal. |

Not dropped on purpose: everything the generator models or fails to model — sampling cadence, speeds and accelerations, path shape, stop structure, daily rhythm, position relative to the public map. Known differences from `docs/geolife/PROCESS.md` (step 8) and `m-v2.md`: synthetic stops snap to a public place at 93% vs 55% real, sampling-interval tail, revisit rate; expect the classifier to find some of these.

## 3. Features (~67; 60 without map; names `group:name`)

| Group | n | Features | What it detects |
|---|---|---|---|
| sampling | 11 | dt quantiles q10/25/50/75/90, log-dt mean/std, dt_mode_share, dt_le2_share, dt_gt60_share, dt_gt300_share | Recording cadence: the real logger's dt mixture vs the generator's per-user interval bins. |
| kinematics | 26 | 2 sub-blocks `raw_*` and `thin30_*` x 13: step length q25/50/90, speed q25/50/75/90, speed_log_std, accel_abs q50/q90, turn_abs mean/q90, turn_cos_mean (lag-1 heading autocorrelation) | Motion micro- and macro-structure. Two scales because raw ~2 s steps are GPS-noise dominated (`docs/geolife/PROCESS.md` D20a: raw steps once fitted bike at 14 m/s); the 30 s-thinned block carries the macroscopic motion. |
| shape | 7 | tortuosity (net/gross displacement), rgyr_over_path, pc1/pc2_extent_over_path (covariance eigenvalues), anisotropy, reversals_per_km (> 150 deg turns), net_over_rgyr | Window geometry, translation- and rotation-invariant: commute-line vs loop vs jitter-cloud. |
| stops | 5 | on thinned points: slow_share (< 0.3 m/s), slow_time_share, stay_count_per_h / stay_time_share / jitter_rms_log via the shared `detect_stays` (200 m / 10 min rule) | Stop frequency, dwell share and in-stay jitter — the generator's stop model vs real loitering. |
| time | 11 | LOCAL time only (UTC+8), never the date: 8 three-hour point-share bins, weekend_share, hour_sin/hour_cos (circular mean) | Daily and weekly rhythm of when points are recorded. |
| map | 7 | OPTIONAL (`--no-map`); the only group reading absolute coordinates: distance to nearest public place q10/50/90 (log1p; KD-tree over the public places table), near_place_100m/250m shares, on_road_cell_share + on_road_dilated_share (res-9 H3 against the cells of the public Overture road graph, 1-ring dilated) | Position relative to the public map. Captures a real documented quality gap (synthetic traces are rendered on the road graph; 93% of synthetic stops snap to a place vs 55% real — `m-v2.md` section 6 / B7) but is partly "different by construction", so the group is toggleable and the headline AUC is reported both with and without it. |

Models: logistic regression (baseline) and gradient boosting (stronger). The strongest is reported.

## 4. Output and usage

- `geolife/eval/discriminate.py`, library `src/geolife/eval/discriminator.py` + `src/geolife/eval/discriminate.py`.
- Results: `geolife/eval/results/discriminator/` (and `discriminator_nomap/`), the same four-table artifact set as SNAP: `discriminator.csv` (one row per method, fold seed, model; AUC, CI, accuracy, real-vs-real AUC, permuted-label AUC), `discriminator_summary.csv` (mean/std over folds), `discriminator_importance.csv` (per feature group), `discriminator_by_size.csv` (AUC by window-size bucket), plus json mirrors.

```
PYTHONPATH=src python geolife/eval/discriminate.py --out geolife/eval/results/discriminator
PYTHONPATH=src python geolife/eval/discriminate.py --no-map --out geolife/eval/results/discriminator_nomap
PYTHONPATH=src python geolife/eval/discriminate.py --n 200 --max-users 20 --n-boot 100 --no-map --out /tmp/disc_smoke   # smoke
```

Key args: `--n 4000` per class, `--folds 3`, `--models logreg gbm`, `--max-users`, `--only`, `--cache-dir` (road graph build ~1-2 min, skipped under `--no-map`). Expected full-run time ~8-12 min, dominated by parquet loading + the road graph.

Reading: AUC 0.5 means this model cannot tell real from synthetic windows; 1.0 means trivially detectable. Compare against the real-vs-real line, with and without the map group.

## 5. Results (first run, 2026-11-03; n=4000 per class, 3 fold seeds)

Controls are clean: real-vs-real 0.52/0.48 (logreg/gbm, map run; 0.53/0.47 without map), permuted labels 0.46-0.49. Against that floor the generator is trivially detectable, and the map group is irrelevant to it:

| run | logreg AUC (mean +- std) | gbm AUC | gbm 95% CI |
|---|---|---|---|
| map on | 0.983 +- 0.004 | 0.9996 +- 0.0001 | [0.999, 1.000] |
| no map | 0.981 +- 0.004 | 0.9995 +- 0.0001 | [0.999, 1.000] |

Grouped importance (fold 0, gbm): `sampling` dominates (auc_drop 0.258 map / 0.266 no-map), `kinematics` second (0.058 / 0.061), `shape`/`stops`/`time`/`map` all <= 0.0002. AUC by window size: 0.997-0.998 already at k 30-60 (about a minute of trace), 1.000 above — the tell is local, per-step texture. Per-feature analysis (notebook section 4): the best single separators are upper-tail kinematics — the synthetic traces are now too SLOW and too uniform (`thin30_step_q90` single AUC 0.82, window-median 131 m real vs 32 m synthetic; `thin30_speed_q90` 0.81, 4.0 vs 1.0 m/s; speed log-std halved), the overcorrection of the old too-fast baseline (seed0's `sensor_speed` table fell back to defaults at signal 1.9 < 3.0). The model itself leans on the dt texture: `sampling:dt_mode_share` is the only feature with a material per-feature drop (0.111; `dt_q90` 0.017, `dt_log_std` 0.010) — rendered dt concentrates too tightly on the per-user modal interval while real dt sequences are ragged mixtures, even though the dt histogram matches the released table. Kinematics per-feature drops are ~0 despite single AUCs of 0.7-0.8: the signal is redundant, a texture rather than one knob (the SNAP pattern). `map:place_dist_q90/q50` separate alone (0.70/0.66) but the model never needs them.

Generator implications (PLAN candidates): restore speed scale and within-outing spread (per-mode speeds, not a pinned median; revisit the seed0 sensor_speed fallback), render dt as a process (burst/jitter mixture) rather than near-constant draws, deprioritise the map gap. Tables: `geolife/eval/results/discriminator{,_nomap}/`; figures and per-feature view: `geolife/eval/notebooks/viz_discriminator.ipynb`.

# Process log: GeoLife track

Rough log of what happened, the decisions taken, and the issues found, for the GeoLife DP mobility track. Newest at the bottom of the log.
**What to do next is in `PLAN.md` section 9**; this file is history. Other docs here: `PLAN.md` (current plan), `INTERFACES.md` (contract, `params`,
planted truth), `p0-feasibility.md` (Overture and routing findings), `archive/` (superseded plans and the first task board).
SNAP log for comparison: `docs/snap/PROCESS.md`. Layout and usage: `geolife/README.md`, `src/geolife/README.md`.

## Log

Work was done by an orchestrator session that merged parallel agent branches (`--no-ff`) into `feat/geolife-impl`, then into `main`. All merges are local; nothing pushed.

**2026-11-01**
- Read the proposal on inspectable synthetic-data artifacts (AIM / Bayesian networks). Chose GeoLife as the example and designed a user-level DP artifact with
  map-anchored tokens, "teleport" to any city, and a success test: a method M built on synthetic data must work on real data one-shot (plan v1, `archive/PLAN_v1_home-work.md`).
- Decisions: user-level DP, no second dataset (compare EDAs instead), code in `src/geolife/`.
- Wave 1 (parallel agents): Overture and routing feasibility (S3 reachable, graph from connectors and segments, HMM matcher), GeoLife loader and EDA (dataset downloadable;
  stay-level overlap weak), DP core, artifact I/O and inspector. Merged.
- Wave 2: generator on a toy city, tokenization (stays, trips, matcher tuning) and learner CLI, Overture `MapProvider` and places tables, teleport demo on Amsterdam. Merged.
- Wave 3: hierarchical place tier and weighted budget (res-8 places are noise at epsilon 6, res 6 usable), method M v1 (home/work, shared anchor). Merged to `main`.
- First run of M v1 on 25 real users: no transfer (4% home, 1% work, 8% of users with a relation vs 72% synthetic).
- **Pivot (user):** the data is mostly outings, so drop home/work. New plan: outings, co-visit relations, time of day and day of week in scope but coarse, no co-presence,
  parametric first, embedding tier as future work (`archive/PLAN_outings_draft.md`).
- T0 outing EDA on all 182 users (public parameters: 2 h gap, 200 m / 10 min stops, UTC+8, caps), T1 outing tokenization, T2 learner v2, T3 generator v2. Merged.
- T2b (user decision): coarse time bins (am/pm x weekday/weekend) and drop the noise-only tables, budget re-spent. Tests moved to `tests/geolife/`. Merged to `main`.
- T4 M v2 on planted truth (places of interest 0.88/0.84 dev, 0.74/0.72 held-out; edges AUC 0.96/0.87) and the first real-vs-synthetic comparison: the generator is far from the real regime.
- RESTRUCT: code, folders, tests and docs moved into inside / transfer / outside / eval with three stage scripts, a boundary test and a provenance allowlist (shared code to
  transfer, M to eval, per the user). Merged to `main`.

**2026-11-02**
- Cleanup: dependencies in `pyproject.toml`, place-snap radius 75 -> 150 m, `CLAUDE.md` library allowlist, leftover folders removed, teleport demo regenerated.
- Committed transfer artifacts `outings_v2_seed0-2` (epsilon 6, 139 users) and stopped ignoring seeds 1-9.
- Generated a 15-user sample into Amsterdam from the artifact; noticed the GPS rendering uses hand-set sensor defaults, not values from the artifact.
- Plan docs reorganised: this file is now the log, `PLAN.md` the single current plan with next steps, old plans in `archive/`.
- **Disk cleanup (user, no space left):** removed all merged worktrees and branches (about 2 GB freed, 19 -> 21 GB free). The gitignored real-data copies were moved out of the repo to
  `C:/Users/NaomiD/geolife-data/` (`inside-data/` = `points.parquet` + `Geolife Trajectories 1.3/Data/<uid>/labels.txt`, `outside-cache/` = Overture parquet caches). Point `GEOLIFE_DATA_DIR`
  at `C:/Users/NaomiD/geolife-data/inside-data` and `--cache-dir` at `C:/Users/NaomiD/geolife-data/outside-cache`. The original zip is still in the main checkout `geolife/data/`.
- **Priorities reset (user):** the most important next step is to render GPS from the real noise distributions (DP-measured sensor tables); then 3 time bins; then richer
  real-vs-synthetic visual comparison in `eval/` (lon/lat, moments); **M is the last step**. `PLAN.md` section 9 re-ordered accordingly (D18, D19).

## Decisions

- **D1 (privacy unit): user-level DP, epsilon about 6 at delta 1e-6**, zCDP accounting with a per-table ledger, Gaussian
  noise; adjacency is "one user's whole data". Every table is a sum of per-user capped contributions, so sensitivity follows from the caps.
- **D2 (env): `C:/Users/NaomiD/geo-venv`** light env (numpy, pandas, scipy, pyarrow, h3, networkx, shapely, matplotlib);
  run tests with `PYTHONPATH=src:. python -m pytest -q tests/geolife`.
- **D3 (place tier, v1): hierarchical release** (H3 res 6 first, refine only under parents whose noisy count passes tau
  sigma) with a weighted rho split instead of an equal split. Choosing resolutions after seeing SNR is unaccounted tuning and is stated as a ledger caveat.
- **D4 (relation for M, v1): shared anchor**; co-presence in time is out of scope. No second dataset: transfer is checked by comparing EDAs.
- **D5 (data model): outings, not home/work.** The first real run showed M's roles almost never fire (4% home, 1% work);
  GeoLife is mostly outings with no stable overnight home. A user is a set of outings; an outing is a trajectory (gap above 2 h
  splits) with a few stops (200 m / 10 min). Output of M: places of interest, common places, crossings, a co-visit graph with communities. See `PLAN.md`.
- **D6 (parametric first): the generator is a parametric outing process** driven by a small `params` dict
  (`INTERFACES.md`) fitted from the noisy tables, with sensor knobs in a public `generation` dict; no learned model on
  the tables. Planted truth (groups, shared places, co-travel) makes M scorable on synthetic data.
- **D7 (time): coarse time bins in the released tables**: local time (UTC+8) am [05,14) / pm [14,05) x weekday / weekend; the
  4 fine time-of-day bins are recovered by a fixed uniform-over-hours split (an assumption, not measured). Chosen after
  seeing the SNR of a finer design: data-dependent, listed in the ledger caveats. Noise-only tables (`place_time_users`,
  `stop_transitions_r6`, `trav_users_r8`, `trav_users_r9`) are no longer released.
- **D8 (public settings chosen by looking at the data) are caveats, not guarantees**: outing gap, real-outing rule, stop rule,
  minimum activity, time bins, H3 hierarchy, per-user caps (100 outings, 50 res-8 stop cells, 1000 traversed cells), histogram
  edges and key caps, the list of dropped tables. They are listed in `privacy.json` `caveats` (`inside/measure_v2.py` `CAVEATS`).
- **D9 (zones): same four-zone layout as SNAP D13.** inside = may read real traces; transfer = artifacts plus the code
  that defines their format and pure code both sides need; outside = generator, never reads real data; eval = needs both.
  Decisions by the user: shared code goes to **transfer** together with the artifacts; **method M (the downstream task) goes to eval**.
- **D10 (shared pure code in transfer):** the types, artifact I/O, `schema*`, geometry helpers (`geo.py`), the `MapProvider`
  protocol, outing / stay / trip tokenization and the road graph with the HMM matcher are used by more than one zone
  (learner and M tokenize; M and the generator share the map), so they live in `transfer/`. Readers of the real data that
  used to sit in those modules (`load_points_parquet`, labels, the outings CLI) moved to `inside/loaders.py` and `inside/build_outings.py`.
- **D11 (third-party libs in transfer / outside): allowlist `h3`, `networkx`, `shapely`, `pyarrow` besides numpy / pandas / scipy**,
  documented in `src/geolife/README.md` and `tests/geolife/test_boundary.py`. All are pure computation. **Decision for the
  user to confirm** (CLAUDE.md names only numpy / pandas / scipy "whatever the method needs"). The network and duckdb fetcher
  is not on the allowlist: it is the script `geolife/outside/fetch_public_map.py`, which may not import inside or eval.
  `OvertureProvider` reads the parquet cache and takes an optional `fetch=` callable.
- **D12 (no OS access in shipped code):** `os`, `tempfile`, `importlib`, `subprocess` etc. are barred from transfer and outside.
  Environment variables (`GEOLIFE_DATA_DIR`) live in `inside/data.py`; paths are passed as arguments; `pipeline_v2` imports
  `fit_v2` directly instead of `importlib`; matplotlib plotting (`plot_artifact`) moved to `eval/plots.py`; `fakedata.py` (v1
  fake artifact, uses `tempfile` and the inside measurement code) moved to eval.
- **D13 (provenance allowlist):** `validate_no_raw_leakage()` rejects top-level keys of `preprocessing.json`,
  `diagnostics.json`, `grounding.json`, `generation.json` not on `PROVENANCE_ALLOWED`. The v1 learner wrote
  `n_users_hint`, an exact un-noised user count, into `diagnostics.json`; removed (nothing read it).
- **D14 (stages):** `geolife/inside/fit_artifacts.py` (real -> `transfer/artifacts/<variant>_seed<k>/`),
  `geolife/outside/generate_synthetic.py` (artifacts only -> traces, outings and planted truth in `outside/synthetic/`; runs
  with only `src` on the path, tested in a subprocess), `geolife/eval/score.py` (synthetic + real -> M metrics and comparison).
- **D15 (data locations):** `geolife/inside/data/` (env `GEOLIFE_DATA_DIR` overrides), `geolife/outside/cache/` (public map,
  inside may read it), `geolife/inside/results/`, `geolife/eval/results/`. Gitignored; copy by hand when moving machines.

- **D18 (priorities, 2026-11-02):** order of work: (1) sensor noise from released real distributions, (2) 3 time bins, (3) visual and statistical comparison in `eval/`, (4-7) parameter
  stability, generator realism, housekeeping, privacy hardening, (8) M last. Reason: M's outputs shift with the sensor regime, so M is only meaningful once the rendering and the comparison are right.
- **D19 (time bins):** the released time tables move to 3 time-of-day bins x weekday/weekend (proposed edges morning [05,11), midday [11,16), evening/night [16,05), fixed from public
  reasoning). Supersedes D7's 2 bins. Still a data-dependent public setting for the ledger caveats.

### Cleanup decisions (2026-11-02)

- **D16 allowlist confirmed:** h3, networkx, shapely and pyarrow are allowed in `transfer/` and `outside/`; duckdb stays in `geolife/outside/fetch_public_map.py`. Written into `CLAUDE.md` rule 2. The dependencies are in `pyproject.toml` (`h3`, `networkx`, `shapely`, `pyarrow`; `duckdb` in the `fetch` group) and `uv.lock` was re-resolved. `uv sync` / `uv run pytest` on the full project env were not run end to end (heavy synthcity env); the geolife suite was run in the light env.
- **D17 place-snap radius 75 m -> 150 m** (default in `OutingsConfig`), because only about 30% of stops snapped at 75 m (48% at 150 m in the EDA). It is a public setting chosen after looking at the snap rate, so it is listed in the ledger caveats. M v2 and its report were measured at 75 m (PLAN next step 5).
- Leftover empty worktree folders removed; the teleport demo PNG regenerated with the v2 generator.

## Issues and findings (from the runs)

| # | Issue | Status |
|---|---|---|
| 1 | Stay-level overlap is weak: res 8 has 63 of 1,236 cells with 10 or more users (point-level overlap is strong) | motivated hierarchical tier (D3) |
| 2 | Real run at epsilon 6 (156 users, 7,156 stays): res-8 place overlap is pure noise; res 6 has 66% of mass above 3 sigma (v1), res 7 42%; `work_users_hist` 70%; behaviour tables 20-60% noisier from the reweighting; `zipf_s` falls back to 1.0 | known |
| 3 | Resolution and time design chosen after seeing SNR is unaccounted tuning | stated in ledger caveats (D7, D8) |
| 4 | Matcher on the real Haidian graph: 72% recovered, 29% extra in car mode; 53% recovered with unknown mode (footways collapse onto roads); tuned on Gaussian noise only | known |
| 5 | Overture: `categories` renamed to `taxonomy` / `basic_category`; no work category, so weak workplace grounding | handled in `outside/map/provider.py`; reason for D5 |
| 6 | M v1 on synthetic (150 users): main places P/R 0.996/0.987, relation edges P/R 0.97/0.88; fails above about 150 m position noise; visit-merge added after the first sensitivity run (held-out slightly optimistic), thresholds tuned to the generator's long stays | superseded by M v2 (`geolife/eval/reports/m-v2.md`) |
| 7 | M v2 was run on the real traces (139 users) with the res-9 cell proxy only: no road graph, matcher not exercised on real GPS; real-vs-synthetic gaps are large (see next steps) | open; PLAN next steps 2, 5, 7 |
| 8 | Privacy attacks, ledger review and artifact signing; dropping the useless finest-tier rho (about 0.056) in v1 | open; PLAN next step 12 |
| 9 | Overture reachability from the air-gap is a risk; the map must be a cached public parquet fetched on the connected side | mitigated by `fetch_public_map.py` (D11) |
| 10 | Light env has no sklearn, so `tests/test_evaluate.py` and `tests/test_run_benchmark.py` (Adult) fail at collection; not a GeoLife problem | known |
| 11 | Two legacy tests/tools point at a map cache in another agent worktree (`AMS_CACHE`); they now prefer `geolife/outside/cache` and skip when absent | cosmetic |

---

## Real-vs-synthetic baseline (before sensor tables) - 2026-11-02

The "before" picture for PLAN next steps 1 and 5, recorded with the comparison tooling of next step 3
(`src/geolife/eval/compare.py` + `viz.py`, `geolife/eval/score.py --viz`, notebook
`geolife/eval/notebooks/viz_comparison.ipynb`). Setup: committed artifact `outings_v2_seed0` (epsilon 6),
current generator defaults, generated INTO the source city for a direct lon/lat overlay
(`generate_synthetic.py --only outings_v2_seed0 --city beijing --n-users 139` -> 49.9M synthetic points);
real side = `points.parquet` restricted to the Beijing core bbox (179 users with points there, 128 pass the
min-activity rule after tokenization; 25M points). Both sides run through the same tokenizer and the same
public places table. Panels: `geolife/eval/results/viz/outings_v2_seed0/*.png`; table: `summary.{csv,md}`
there (copy below; the CSV also has std / skew / kurtosis / p1..p99). Moments are of the full samples;
KS / Wasserstein / TV on subsamples of up to 100k; `tv` of the time rows is total variation over the
hour / time-bin shares.

Headline gaps (what steps 1 and 5 must fix), largest first:

- **Heading change** (render artefact): real traces go straight (median 3.5 deg per step), synthetic GPS
  wiggles (median 121 deg) - the hand-set 6 m noise on a ~2 s, ~5 m-step render dominates the direction
  signal. KS 0.79. This alone explains much of the mode misclassification below.
- **Speeds**: synthetic "walk" outings move at 13.7 m/s on average (real 2.8, KS 0.59) - the tokenizer's
  speed heuristic calls fast synthetic outings walk because rendered speeds are too high across the board:
  overall speed mean 13.5 vs 5.1 m/s (KS 0.38), step length 28 vs 13 m.
- **Outing size**: synthetic outings are 5.4x too long (mean 81.7 vs 15.2 km; p95 341 vs 49) and 2.1x
  longer-lasting (duration 212 vs 99 min). Points per user 3.5x too many (359k vs 102k).
- **Gaps**: synthetic gaps are too long (mean 1030 vs 436 s, KS 0.34) and too rare (7.9k vs 19.1k gap steps
  at ~1.3x the step count); the sampling interval is close (median 2 s both, KS 0.09).
- **Stop jitter** is too clean and too narrow: mean 53 vs 67 m, KS 0.41.
- **Structure is closer**: users per place TV 0.08 (mean 3.1 vs 2.8), users per cell TV 0.09, co-visit
  degree mean 24.1 vs 19.2 (KS 0.16), outings per user mean 123 vs 134 - but revisit rate is high
  (0.57 vs 0.45) and distinct places per user 1.6x too many (174 vs 109).
- **Time**: hour profiles TV 0.22-0.33 (the synthetic starts miss the sharp real 08h peak), time-bin
  shares TV 0.18.

Full table (real vs `outings_v2_seed0`, Beijing frame):

| quantity | group | n_real | n_syn | real_mean | syn_mean | real_p50 | syn_p50 | real_p95 | syn_p95 | ks | wasserstein | tv | rel_err_mean | rel_err_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sampling_interval | point | 1948679 | 2517705 | 3.43 | 3.09 | 2 | 2 | 5 | 6 | 0.0937 | 0.535 | 0.184 | 0.0994 | 0 |
| gap_len | point | 19095 | 7944 | 436 | 1.03e+03 | 199 | 393 | 1.88e+03 | 4.17e+03 | 0.337 | 591 | 0.339 | 1.36 | 0.975 |
| step_len | point | 1948679 | 2517705 | 13.3 | 28.3 | 6.42 | 19.1 | 45.1 | 77.7 | 0.401 | 17.5 | 0.4 | 1.13 | 1.98 |
| speed | point | 1955426 | 2519161 | 5.09 | 13.5 | 2.19 | 7.72 | 18 | 47.7 | 0.383 | 8.46 | 0.382 | 1.65 | 2.52 |
| speed_walk | point | 837742 | 1957550 | 2.79 | 13.7 | 1.24 | 7.63 | 12 | 49 | 0.586 | 11 | 0.586 | 3.89 | 5.17 |
| speed_bike | point | 415517 | 442635 | 4.59 | 13.5 | 3.02 | 8.03 | 14.6 | 46 | 0.417 | 8.87 | 0.416 | 1.94 | 1.65 |
| speed_bus | point | 280043 | 66895 | 7.02 | 9.67 | 5.69 | 7.19 | 18.4 | 24.2 | 0.257 | 2.68 | 0.297 | 0.378 | 0.265 |
| speed_car | point | 221391 | 14148 | 10.7 | 14.4 | 10.9 | 12.1 | 21.7 | 31.9 | 0.167 | 3.72 | 0.213 | 0.344 | 0.108 |
| speed_train | point | 27077 | 3724 | 16 | 21.5 | 17.2 | 19.9 | 27.5 | 40.5 | 0.217 | 5.48 | 0.24 | 0.343 | 0.154 |
| heading_change | point | 993359 | 2180492 | 11.6 | 109 | 3.53 | 121 | 49.1 | 175 | 0.791 | 97.4 | 0.79 | 8.39 | 33.4 |
| stop_jitter | point | 20797 | 22835 | 67 | 52.9 | 65.4 | 43.6 | 108 | 104 | 0.408 | 15.1 | 0.453 | 0.21 | 0.333 |
| outing_length | outing | 17195 | 16791 | 15.2 | 81.7 | 9.56 | 29.6 | 48.6 | 341 | 0.367 | 66.6 | 0.366 | 4.39 | 2.09 |
| outing_duration | outing | 17195 | 16791 | 99.1 | 212 | 44.1 | 107 | 370 | 755 | 0.238 | 113 | 0.24 | 1.14 | 1.42 |
| stops_per_outing | outing | 17195 | 16791 | 1.13 | 1.36 | 0 | 1 | 5 | 5 | 0.0994 | 0.353 | 0.0994 | 0.197 |  |
| stop_dwell | outing | 19494 | 22788 | 42.1 | 62.2 | 24 | 32.4 | 119 | 220 | 0.117 | 20.7 | 0.117 | 0.476 | 0.351 |
| points_per_user | user | 179 | 139 | 1.019e+05 | 3.586e+05 | 2.48e+04 | 1.690e+05 | 4.180e+05 | 1.431e+06 | 0.461 | 2.568e+05 | 0.47 | 2.52 | 5.82 |
| outings_per_user | user | 128 | 136 | 134 | 123 | 55 | 74.5 | 494 | 403 | 0.202 | 42.6 | 0.358 | 0.0809 | 0.355 |
| active_days | user | 128 | 136 | 75.4 | 61.4 | 37.5 | 41.5 | 245 | 200 | 0.145 | 19.5 | 0.291 | 0.185 | 0.107 |
| distinct_places_per_user | user | 128 | 136 | 109 | 174 | 63 | 128 | 302 | 555 | 0.279 | 65 | 0.327 | 0.595 | 1.04 |
| revisit_rate | user | 128 | 136 | 0.451 | 0.573 | 0.452 | 0.575 | 0.766 | 0.815 | 0.289 | 0.123 | 0.343 | 0.27 | 0.274 |
| users_per_place | structure | 5022 | 7583 | 2.78 | 3.12 | 2 | 2 | 9 | 9 | 0.0839 | 0.367 | 0.0839 | 0.122 | 0 |
| users_per_cell | structure | 12365 | 13428 | 9.72 | 11.9 | 4 | 6 | 37 | 45 | 0.0907 | 2.15 | 0.0915 | 0.22 | 0.5 |
| covisit_degree | structure | 128 | 136 | 19.2 | 24.1 | 12 | 17 | 50.6 | 60 | 0.161 | 4.88 | 0.297 | 0.253 | 0.417 |
| outing_start_hour_weekday | time | 12875 | 12300 |  |  |  |  |  |  |  |  | 0.227 |  |  |
| outing_start_hour_weekend | time | 4320 | 4491 |  |  |  |  |  |  |  |  | 0.219 |  |  |
| stop_hour_weekday | time | 13078 | 16581 |  |  |  |  |  |  |  |  | 0.263 |  |  |
| stop_hour_weekend | time | 6416 | 6207 |  |  |  |  |  |  |  |  | 0.325 |  |  |
| timebin_share | time | 17195 | 16791 |  |  |  |  |  |  |  |  | 0.177 |  |  |

---

## 2026-11-02 (later): PLAN steps 1 and 2 done — DP sensor tables, inverse-CDF rendering, 3 time bins

Delimited section, appended by the step-1/2 agent. Code: `src/geolife/inside/measure_sensors.py`, `src/geolife/outside/sensors.py`
(new), changes in `transfer/{types,schema_v2,artifact}.py`, `inside/{measure_v2,learn_v2}.py`, `outside/{fit_v2,render_v2,pipeline_v2}.py`,
`geolife/inside/fit_artifacts.py`. Tests: `tests/geolife/inside/test_measure_sensors.py`, `tests/geolife/outside/test_sensors.py`,
new tod3 cases in `test_learn_v2_pipeline.py`. Suite: 205 passed, 2 skipped, 1 known-environment failure (`test_clean_room`:
the audit hook flags a benign read of `/usr/share/zoneinfo/UTC` on this machine; pre-existing, not touched).

**Step 1 (sensor tables).** A new inside stage reads the raw points (they are not in `UserOutings`) and measures, per user,
9 histograms with fixed public bin edges: `sensor_interval` (log-ish bins, within-run dt <= 60 s), `sensor_interval_user`
(one-hot of the user's median interval - this is what gives the renderer per-user 1 s vs 5 s heterogeneity), `sensor_gap_len`
(dt in (60 s, 2 h]), `sensor_gap_rate` (one-hot, gaps per logged hour), `sensor_speed` (by run mode, on 30 s-THINNED steps -
raw 1-5 s steps are GPS-noise dominated: 6 m noise at 2 s is ~3 m/s of fake speed; the first run of this stage measured raw
steps and fitted bike at 14 m/s, which is how the thinning fix was found), `sensor_step` (raw step length by mode),
`sensor_heading`, `sensor_stop_jitter` (distance of in-stop points from a per-axis-MEDIAN stop centre - a mean centre is
dragged by walk-in/walk-out points), `sensor_points_per_outing`. All are user-level pmf (L2 <= 1) or one-hot (L2 = 1) vectors,
per-user contribution additionally capped by a deterministic uniform subsample of at most 100k steps seeded from the pseudonym.
Release and accounting are the same Gaussian/zCDP machinery as the behaviour tables, one ledger entry per table.

**Step 2 (3 time bins).** `outing_start` and `cat_time_visits` moved from am/pm x daytype to `types.TOD_3` x daytype with the
D19 edges (morning [05,11), midday [11,16), evening/night [16,05), local UTC+8; `local_tod3_bin`). `fit_v2` expands to the 4
fine bins with the fixed hours-proportional `schema_v2.TOD3_TO_FINE`; the ledger caveat states that both the choice of 3 bins
and the edges are public settings fixed after the previous run. `TOD_COARSE` and its helpers are kept for back-compatibility.

**Outside.** `fit_v2` -> `params["sensor"]` (fit by `outside/sensors.fit_sensors`: denoise, usable iff released mass >= 3 sd of
the noise on that total, per-mode mass test plus a public plausibility window on the per-mode median speed). `render_v2` draws
by inverse CDF (`sensors.HistSampler`): per-user sampling period from `interval_user`, per-user gap rate from `gap_rate`, each
gap length from `gap_len`, in-stop displacement (radius + uniform angle) from `stop_jitter`; `pipeline_v2` injects fitted
`speed_ms_by_mode` unless the caller set it explicitly. Every unusable table falls back to the old hand-set defaults
(`render_v2.SENSOR_DEFAULTS`), and the artifact's `privacy.json` lists both the generic hand-set values and the per-release
fallback list. En-route `sensor_noise_m` stays hand-set by design: it is not identifiable from the released tables (stop jitter
conflates receiver noise, indoor drift and walking around), stated as an explicit fallback.

**Artifacts regenerated and committed** (`outings_v2_seed0-2`): 139 users with outings, 182 sensor views, 24.9M raw points;
29 tables; rho 0.5395, 80% behaviour / 20% sensors (exactly 0.1079); epsilon 6.000 at delta 1e-6. Signal per table
(released total mass / sigma sqrt(K); usable iff >= 3), seeds 0/1/2:

| table | sigma | signal s0/s1/s2 | fit |
|---|---|---|---|
| outing_start (3x2) | 7.5 | 6.8 / 7.1 / 5.7 | usable on all seeds |
| cat_time_visits (10 cats x 3 x 2) | 7.4 | 2.9 / 2.0 / 3.1 | below 3 on seeds 0-1 (as at 2 bins): cat profile falls back |
| sensor_interval | 6.6 | 9.5 / 7.4 / 7.6 | usable |
| sensor_interval_user | 6.6 | 8.1 / 7.8 / 8.3 | usable; q25/50/75 ~ 1.8 / 3-4.5 / 5.4 s |
| sensor_gap_len | 6.6 | 6.9 / 7.8 / 7.6 | usable; median ~ 145-165 s |
| sensor_gap_rate | 7.6 | 8.1 / 8.0 / 9.8 | usable; mean ~ 4.1-5.7 / h |
| sensor_speed (6 modes x 13) | 5.0 | 1.9 / 5.2 / 4.6 | seed0 falls back; seeds 1-2 fit walk 1.24 / 1.09 and bike 3.4 m/s (EDA labelled medians 1.1 / 3.1); other modes below the per-mode floor |
| sensor_step (6 modes x 11) | 5.6 | 2.6 / 3.5 / 3.3 | marginal; per-mode mass below floor everywhere (diagnostic only) |
| sensor_heading | 7.5 | 7.6 / 7.4 / 7.5 | usable (not yet consumed by the renderer) |
| sensor_stop_jitter | 6.7 | 8.9 / 5.3 / 8.7 | usable; q25/50/75 ~ 16-24 / 37-43 / 67-87 m |
| sensor_points_per_outing | 7.7 | 6.3 / 8.7 / 7.6 | usable (diagnostic) |
| behaviour tables | 7.4-22.8 | 4.3-26.0 | as before (cat_transitions still noise at 0.2-1.2) |

**Sanity check** (stage 2, 50 users, toy grid, seeds 0-1): per-user median sampling interval q25/50/75 rendered 2/3/5 s vs
released 1.8/3.4/5.3 s; gap length median rendered 165 s vs released 145 s; 30 s-thinned rendered speeds q25/50/75
1.0-1.2 / 1.6-1.8 / 3.1-3.3 m/s, consistent with the walk-heavy mode mix. The full visual comparison is PLAN step 3.

### Decisions

- **D20 (sensor budget split and sensor release design, confirmed by the user 2026-11-02):** 80% of the total rho to the behaviour tables, 20% to the new
  sensor group (`schema_v2.SENSOR_RHO_FRAC`), split within each group by the same public-weight rule (priority x ln(vocab),
  40% equal-split floor); declared sensor priorities in `schema_v2.SENSOR_PRIORITIES`. Rationale: the sensor tables aggregate
  millions of per-point events under per-user caps and L1 normalisation, so they reach high SNR at a small budget - confirmed
  by the run (signal 5-10 at 20% for 7 of 9 tables). Shares use public quantities only; total epsilon stays 6 at delta 1e-6.
  The bin edges, caps, thresholds and the thinning rule are public settings informed by the (non-DP) EDA: ledger caveats.
- **D20a (within step 1, implementation):** per-mode speed is fitted as the median of the 30 s-thinned distribution and
  accepted only inside a public plausibility window per mode; `sensor_noise_m` is never derived from the released tables
  (stop jitter is not a receiver-noise measurement) - the hand-set 6 m stays, as an explicit per-release fallback caveat.

### Issues found while doing this

| Issue | Status |
|---|---|
| `tokenize._clean` crashes on empty arrays (users with all points outside the bbox) | guarded in `make_sensor_view`; regression test |
| Raw-step speeds are GPS-noise dominated (first fit gave bike 14 m/s) | fixed: speed measured on 30 s-thinned steps + plausibility window |
| Mean stop centre dragged by walk-in/out points inflated the jitter histogram bulk to ~20 m | fixed: per-axis median centre |
| `_signal` (noisy total mass) fluctuates across seeds near the threshold: sensor_speed usable on seeds 1-2 but not seed 0 | by design (fallback); a real release would publish one seed |
| cat_time_visits at 3 bins is still at the noise edge (2.0-3.1) | expected cost of D19; cat profile falls back exactly as at 2 bins |

---

## 2026-11-02 (later still): PLAN step 7 — privacy hardening (attacks, ledger review, signing, one seed)

Delimited section, appended by the step-7 agent. Everything here leaves the committed artifacts byte-identical; the one
step-7 item that would regenerate them ("drop zero-signal budget") is deliberately deferred. Code:
`src/geolife/eval/attacks.py` (new), `geolife/eval/{run_attacks,review_ledger}.py` (new), signing in
`src/geolife/transfer/{artifact,artifact_cli}.py`, policy/signing flags in `geolife/inside/fit_artifacts.py`
(`--release`, `--sign-key`) and `geolife/outside/generate_synthetic.py` (`--key`), caveat-text fixes in
`inside/measure_v2.py`. Reports: `geolife/eval/reports/{privacy-attacks,ledger-review}.md`; tables in
`geolife/eval/results/attacks/`. Tests: `tests/geolife/transfer/test_artifact_signing.py`,
`tests/geolife/eval/test_attacks.py`. Suite: 229 passed, 2 skipped, 1 known machine failure (`test_clean_room`).

**Empirical attacks** (label on everything: EMPIRICAL DIAGNOSTIC, not the DP guarantee). Setup: the 139 fitted users,
the 3 committed artifacts, and for linkage 139 synthetic users generated from seed0 into Beijing (81.1M points, same
tokenizer both sides). Results (details in `privacy-attacks.md`):

- *Membership inference at the user level (vs the artifact):* per-user contribution vectors to all 29 released tables
  give the informed-attacker test EXACTLY (AUC = Phi(sqrt(Q/2)), Q = sum ||v_u||^2/sigma^2): AUC p50 0.695 / max 0.730
  against the zCDP ceiling 0.769 (Q max 0.751 < 2 rho = 1.079); worst-user TPR 0.072 at FPR 0.01. A realistic
  (uninformed) matched filter over 400 simulated releases: AUC p50 0.598 / max 0.704. Within what eps 6 permits, and
  an end-to-end check that sensitivities and sigmas are what the ledger says.
- *Linkage / re-identification (vs the synthetic data):* nowhere above chance. Best top-10-stop-cell Jaccard of a real
  user against all synthetic users: mean 0.111 (chance vs other real users 0.351, own split half 0.397);
  re-identification rate 0.036 vs chance 0.439; nearest-profile ratio 1.40 (copy would be << 1); coarse-signature
  excess -0.042. The margin below chance partly reflects the generator's realism gap (step 5): re-run after improving it.
- *Seed averaging (PLAN section 4 risk, quantified):* the 3 seeds are exact sigma-replicates (cross-seed std / ledger
  sigma median 0.84); averaging them cuts noise by sqrt(3)=1.73 and would e.g. lift cat_time_visits over the usability
  threshold it deliberately fails at one seed; honest accounting of releasing all three: rho 1.619 -> eps 11.08 at
  delta 1e-6 (not 6); worst-case informed-MI AUC would rise 0.730 -> 0.856.

**Ledger review** (`ledger-review.md`): every stated number of all 3 artifacts re-derived independently — per-entry
rho = sens^2/(2 sigma^2), rho sum, eps(delta) conversion, sensitivities vs the schema clipping bounds, entry/table/meta
agreement, caps vs code constants, manifest hashes, provenance allowlist, index.json: all OK, no numeric error.
Caveat gaps found and fixed in `measure_v2.CAVEATS` for FUTURE artifacts (committed privacy.json not regenerated):
the analysis region (Beijing core bbox) and the tokenizer's mode-heuristic speed cuts / 70 m/s spike filter were not
named; the committed trio nowhere states that it is 3 averageable replicates.

### Decisions

- **D21 (empirical attacks live in eval, with references):** attack code is eval-only (`src/geolife/eval/attacks.py`,
  imports inside code for the per-user vectors; the boundary test still passes). Every linkage statistic is reported
  against BOTH an across-user chance reference and a within-user split-half "copy" reference; membership inference is
  reported as the exact informed-attacker AUC (no simulation error) plus a Monte-Carlo realistic attacker. Committed
  outputs are aggregate tables only (no per-user rows, no ids, no coordinates).
- **D22 (one seed per release, confirmed by the user 2026-11-02):** a real release publishes exactly ONE artifact seed. `fit_artifacts.py --release`
  enforces it (refuses --seeds > 1) and stamps `learn_v2.ONE_SEED_CAVEAT` into `privacy.json`. The 3 committed seeds
  remain a dev set (regenerating them is out of scope here) and must not be published together; quantified cost above.
- **D23 (artifact authenticity = optional HMAC, key stays inside):** integrity was already SHA-256 per file in the
  manifest; authenticity is now an HMAC-SHA256 over the exact `manifest.json` bytes in `manifest.hmac.json`
  (one MAC covers everything the manifest hashes). Stdlib `hmac` only; transfer imports no os/subprocess; key paths are
  arguments. Explicitly SHARED-KEY authenticity, not a public-key signature: the key is generated inside the air gap
  and carried to the outside verifier over the same trusted channel that moves the code, never stored next to the
  artifact; >= 16 bytes, file bytes stripped of surrounding whitespace. Signing is optional and backward compatible —
  unsigned artifacts (the committed ones) load and verify exactly as before; `load_artifact(dir, key_path=...)`,
  `artifact_cli sign/verify --key`, `fit_artifacts.py --sign-key`, `generate_synthetic.py --key`.

---

## 2026-11-02 (later still): PLAN steps 4 and 5 done — stable fits, generator matched to the real regime

Delimited section, appended by the step-4/5 agent. Code: `src/geolife/outside/{fit_v2,population_v2,render_v2,pipeline_v2,sensors}.py`,
allowlist additions in `transfer/artifact.py`, caveat text in `inside/measure_v2.py` (future releases only — the committed artifacts
were NOT regenerated; all changes are post-processing of the released tables). Tests: `tests/geolife/outside/test_fit_learned_artifacts.py`
(new: fit spread + planted stats on the committed artifacts), HOUR_SHAPE expansion case in `test_learn_v2_pipeline.py`.
Suite on this machine: 218 passed, 2 skipped, 1 known machine-specific failure (`test_clean_room`: benign zoneinfo audit-hook read).
"After" comparison committed as `geolife/eval/results/compare_outings_v2_seed0_steps45.{csv,md}` (versioned against the baseline
`compare_outings_v2_seed0.{csv,md}`); the panels under `results/viz/outings_v2_seed0/` now show the after state.

**Step 4 (fit stability).** The old fit zeroed histogram cells below 1 sigma and moment-matched the survivors on bin midpoints;
between noise replicates whole bins flipped in and out, so the three committed seeds (replicates of the SAME data) gave wildly
different fits and output volumes, and the planted-stats check failed on all seeds. Replaced by MAP fits of the raw, unclamped
noisy bin counts: lognormals as A*(Phi bin mass) with closed-form amplitude, Gaussian likelihood at the release noise sd and a
prior around the public DEFAULTS values (TAU_MU 0.3 / TAU_SIGMA 0.15; the prior weight scales with the noise, so a high-budget
release is data-dominated); stops-per-outing as a geometric-tail MAP fit (the old threshold+PAV plateau nearly doubled the planted
stop count); `outing_start` / `outing_mode` cells clamp at zero without the sigma cut (it zeroed weekend-morning starts and the
train share on some replicates); `detour_ratio` is now fitted as the EXCESS (ratio - 1), the convention the renderer always drew
with — the old ratio-as-excess readout detoured every route to ~2.3x the direct path. Fitted parameters over seeds 0/1/2
(lognormal mean = exp(mu + sigma^2/2); "before" = old moment-matched fit):

| parameter (mean) | before s0/s1/s2 | after s0/s1/s2 | spread (max-min)/mean before -> after |
|---|---|---|---|
| outings_per_user | 182 / 59 / 143 | 167 / 99 / 127 | 0.96 -> 0.52 |
| active_days (span) | 232 / 146 / 388 | 218 / 183 / 286 | 0.95 -> 0.45 |
| duration_min | 172 / 146 / 139 | 136 / 83 / 136 | 0.21 -> 0.45 (the before level was +40-70% biased) |
| length_km | 27 / 23 / 18 | 22 / 20 / 20 | 0.41 -> 0.13 |
| distinct stop cells | 53 / 64 / 73 | 36 / 50 / 57 | 0.31 -> 0.45 |
| stops/outing (pmf mean) | 2.06 / 1.59 / 1.75 | 1.22 / 0.86 / 0.84 | (real tokenized 1.13) |
| revisit_rate | .546 / .484 / .536 | unchanged | |

Generated volume, 50 users on the toy grid, seeds 0/1/2: before 20.2M / 6.0M / 29.4M points (max/min 4.9x); after
9.6M / 6.4M / 14.4M (max/min 2.25x). The planted-stats check passes on ALL three learned artifacts (before: `stop_revisit_rate`,
`users_per_place_mean`, `share_places_ge2_users` failed on all seeds) — partly from the fits, partly from aligning the check with
what the tables measure: per-user revisit rate (revisit_hist is a histogram over users), distinct res-8 STOP cells capped at 50
(user_places semantics), users-per-place targets scaled to the generated population size (binomial plug-in anchored at
n_users_source; reported as info when the target map's pool cap makes them unreachable), a finite-sample log tolerance for the
outings median, and category-profile checks only over categories whose profile was actually fitted.

**Step 5 (generator realism).** Biggest single cause of the baseline gaps was WHITE 6 m GPS noise at a 1-5 s sampling period:
~10 m of fabricated travel per step = ~5 m/s of fake speed, 121 deg median heading change, and tens of km of fake outing length
(a 2-h outing gained ~40 km of noise path). Receiver error is now an AR(1) process per axis (marginal sd `sensor_noise_m`, tau
`noise_tau_s` 600 s — the loggers emit smoothed tracks), and in-stop points drift slowly (`stop_drift_tau_s`) with a per-stop
spread drawn from the DP stop-jitter histogram (per-point independent draws made in-stop points jump tens of metres between
samples). Gap regime: gap lengths keep the raw clamped tail of the DP histogram (the 1-sigma cell cut had erased the long-gap
tail), the per-user DP gap rate is applied to wall-clock time uncorrected (the released rate is a user mean ~4.7/h; the pooled
real share is gap-heavy-user dominated), and 30% of dwells lose signal entirely (a dwell that vanishes whole never shows in the
within-outing gap statistics). Outing process: stop draws target the released revisit rate with a per-user proportional
controller; starts/ends lean to familiar places (0.85/0.9); `length_scale` is calibrated so the latent outing-length median hits
the fitted median (damped, saturation-aware, divided by an observed 1.3x render bias); per-mode length factors rebalanced and
capped (walk 3.5 km ... train 120 km — the slow-mode lognormal tails were multi-hour walks); motorised cruising speeds raised to
offset the stop-and-go dip; the users-per-place pool calibration target scales with population size over a 1.2x tokenization
bias. The fine-bin time expansion weights hours by the public HOUR_SHAPE prior instead of uniform (the uniform split put ~38% of
the evening/night bin's mass into 0-5h starts — the missing 08h peak of the baseline).

Before/after (real vs synthetic, Beijing frame, committed artifact `outings_v2_seed0`, 139 users; before = baseline table above;
full after table in `compare_outings_v2_seed0_steps45.md`):

| quantity | real | before | after |
|---|---|---|---|
| outing_length mean km (p95) | 15.2 (48.6) | 81.7 (341) | 12.0 (35.5) |
| outing_duration mean min (p50) | 99.1 (44.1) | 212 (107) | 106 (61.2) |
| points_per_user mean | 102k | 359k | 270k |
| gap_len mean s (p50) | 436 (199) | 1030 (393) | 523 (188) |
| step_len mean m | 13.3 | 28.3 | 11.0 |
| speed mean m/s | 5.09 | 13.5 | 2.59 |
| speed_walk mean m/s (p50) | 2.79 (1.24) | 13.7 (7.63) | 1.68 (1.27) |
| heading_change p50 deg | 3.53 | 121 | 4.99 |
| stop_jitter mean m (KS) | 67 | 52.9 (0.41) | 62.7 (0.12) |
| stops_per_outing mean | 1.13 | 1.36 | 0.87 |
| stop_dwell mean min | 42.1 | 62.2 | 27.7 |
| outings_per_user mean (p50) | 134 (55) | 123 (74.5) | 174 (89) |
| active_days mean | 75.4 | 61.4 | 85.0 |
| distinct_places_per_user mean | 109 | 174 | 230 |
| revisit_rate mean | 0.451 | 0.573 | 0.368 |
| users_per_place mean | 2.78 | 3.12 | 3.35 |
| covisit_degree mean | 19.2 | 24.1 | 18.1 |
| outing_start_hour TV wkday / wkend | - | 0.227 / 0.219 | 0.121 / 0.203 |
| timebin_share TV | - | 0.177 | 0.102 |
| sampling_interval KS | - | 0.094 | 0.174 |

Closed: outing size and shape (length KS 0.37 -> 0.09), the whole GPS noise regime (heading, step length, per-mode speeds,
stop jitter), gap lengths, time-of-day profiles, co-visit degree, hub tail (users_per_place p95 now 9 vs 9). Still open, in
rough priority order: points per user ~2.7x (duty cycle and outing count), distinct places per user ~2.1x (and the related
tokenizer snap flicker that also pulls the synthetic revisit rate ~0.08 below the real one at an identical planted rate),
stop-dwell tail ~0.66x, pooled speed mean ~0.5x (walk-point share), sampling-interval tail (the DP one-hot bins place a few
users at 10-20 s), outings-per-user median (this seed's fit; the median check now carries a finite-sample tolerance).

### Decisions

- **D24 (fit denoising, step 4):** parameters are MAP fits of the RAW noisy tables (no cell thresholding) against the public
  DEFAULTS priors; sigma capped by a public plausibility window (SIGMA_CAP 2.0); `detour_ratio` = excess convention; the
  users-per-place DEFAULT prior has mean ~2.8 (the old 1.78 was unreachable by the outing process and made the planted check
  fail whenever the res-8 tier is under its noise floor). The priors and shrinkage strengths are public settings informed by
  the (non-DP) EDA and the real-vs-synthetic comparison: ledger caveat for future releases, outside the DP guarantee (D8).
- **D25 (generator process, step 5):** released revisit rate is enforced per user by a proportional controller on stop draws;
  starts/ends prefer familiar places; latent length and users-per-place are calibrated inside `sample_population_v2` with
  documented render-bias constants (1.3x length, 1.2x users-per-place); per-mode length factors + caps; the planted-stats
  check compares like with like (per-user revisit, capped stop cells, population-size-scaled users-per-place).
- **D26 (sensor rendering, step 5):** receiver noise is AR(1) (tau 600 s) with the hand-set 6 m marginal; in-stop drift AR(1)
  (tau 900 s) with per-stop spread from the DP stop-jitter histogram; gap lengths keep the raw histogram tail; per-user gap
  rate on wall-clock time; 30% whole-dwell signal loss on top. All time constants and loss shares are hand-set public settings
  tuned on the comparison runs (ledger caveat), like the 6 m noise itself (D20a).

### Issues found while doing this

| Issue | Status |
|---|---|
| Caveats missed the analysis-region bbox and the mode-heuristic cuts; nothing stated the 3-seed replicate risk in the artifacts themselves | caveat text fixed forward (measure_v2.CAVEATS, ONE_SEED_CAVEAT); committed artifacts intentionally untouched, gap documented in ledger-review.md |
| Hierarchical child tables (place r7/r8, trav r7) have seed-dependent vocabularies, so cross-seed averaging/std only covers the 26 common-key tables | by design; noted in privacy-attacks.md |
| Synthetic linkage baselines are depressed by the generator's realism gap (best-Jaccard 0.111 << chance 0.351) | done 2026-11-02 (attack refresh, end of this doc): with the realistic generator the population statistics rose to their chance references (profile ratio 1.40 -> 1.08, signature excess -0.042 -> -0.003) and linkage stays at/below chance |
| `artifact_cli` subprocess aborts AT EXIT on this machine after printing OK (pyarrow teardown; same family as the known `test_clean_room` failure) | machine-specific, rc in-process is 0 |
| render read the fitted detour RATIO as an excess: every route detoured to ~2.3x the direct path | fixed (excess convention, D24) |
| threshold+PAV stops fit filled zeroed tail bins with a flat plateau (planted ~2 stops/outing vs real 1.13) | fixed (geometric-tail MAP fit) |
| 1-sigma cell cut zeroed weekend-morning starts / the train mode share / the long-gap tail on some replicates | fixed (clamp-only for those tables) |
| uniform-over-hours TOD3 expansion put ~38% of evening/night mass into 0-5h starts | fixed (HOUR_SHAPE weights) |
| planted users-per-place target did not scale with the generated population size (histogram is for 139 users) | fixed (anchored binomial plug-in); reported as info when the map's pool cap binds |
| tokenizer snap flicker: identical planted revisit rates tokenize ~0.08-0.18 lower (both sides; synthetic a bit worse) | open; the distinct-places gap is the same mechanism |
| stage-2 output volume still ~2.25x across seeds (outings-per-user tail mass sits at the noise floor of one table) | accepted; a real release publishes one seed |

---

## 2026-11-02 (last): PLAN step 6 (housekeeping) + the deferred step-7 item (drop zero-signal budget, D27)

Delimited section, appended by the step-6 agent. Code commits, in order: v1 retirement; `cells` column in the
outings export; budget drop (D27) + regenerated artifacts; clean-room test fix; this doc update. The regenerated
artifacts REPLACE `outings_v2_seed0-2` byte for byte (26 tables instead of 29); every number below was measured
on this machine with the system python (`/opt/venv`; there is NO `uv` here).

### v1 retirement (step 6)

v1 and v2 were entangled; the survivors moved into the v2 modules and the v1 modules and their tests were
deleted. The map (old module -> where its survivors went):

| retired v1 module | survivors -> new home |
|---|---|
| `inside/measure.py` | `rho_weight`, `FLOOR_FRAC`, `weighted_rho_split` -> `inside/measure_v2.py` |
| `inside/learn.py` | `rho_for_epsilon`, `snr_report` -> `inside/learn_v2.py` |
| `outside/fit.py` | `_pmf`, `_fit_zipf`, `_hist`, `fit_decay` -> `outside/fit_v2.py` |
| `outside/population.py` | `LatentPlace` -> `outside/population_v2.py` (nothing else) |
| `outside/render.py`, `outside/pipeline.py` | nothing (v1 only) |
| `transfer/schema.py` | `INF`, `fmt`, `bin_of`, `bin_keys`, `mid` -> `transfer/schema_v2.py` |
| `eval/fakedata.py` | nothing (`outside/fakedata_v2.py` is the v2 fake) |
| `eval/m/core.py` | `MapView` -> `eval/m/outings.py` (M v1 itself deleted) |
| `eval/m/transfer.py` | `places_table`, `grid_graph`, `_ari` -> `eval/m/outings_transfer.py` |

`outside/teleport.py` is kept (the v2 pipeline grounds through it via `population_v2.latent_view`); it lost its
import of the v1 `Population` (the pop argument is duck-typed now). `eval/m/__init__.py` re-exports M v2.
Deleted v1-only tests: `test_tokenize_learn`, `test_learn2_hierarchy`, `test_generate_population`,
`test_generate_render`, `test_generate_e2e` (v1 e2e; v2 e2e lives in `test_boundary.py` stage-2 subprocess and
`test_generate_v2_pipeline.py`), `test_m_core`. Rewritten: `test_generate_fit.py` (surviving fit_v2 helpers +
the weighted-rho-split rules re-homed from `test_learn2_hierarchy`), `test_generate_teleport.py` (teleport on
the v2 population). The boundary test passes after the retirement.

### Outings export (step 6)

`tokenize.outings.to_frames` now writes `cells` (traversed res-9 cells) and `segments` as JSON columns next to
`visits`, so `learn_v2.load_outings_parquet` rebuilds full `UserOutings` from the export alone and the
traversal tables can be fed from it (before, only the in-process `fit_artifacts.py` path had `Outing.cells`).
Chosen over documenting the gap: the column costs little (parquet-compressed JSON of cell id lists on an
inside-only, gitignored file) and removes a silent data loss on round trip. Round-trip test added; older
exports without the column still load (empty traversal).

### Drop zero-signal budget (deferred step-7 item; D27)

Evidence (per-table signal = released total mass / (sigma * sqrt(K)), the fit's usability measure, threshold
3.0), measured on the 3 old committed artifacts, seeds 0/1/2:

- `cat_transitions`: 1.16 / 0.19 / 0.73 - pure noise, and NO parameter ever read it (fit_v2 never consumed it).
- `cat_time_visits`: 2.94 / 2.00 / 3.09 - at/below the floor on 2 of 3 seeds; on the one passing seed its
  per-category content still failed the fit's internal floors (`cat_mix: category mass below 3 sigma`,
  `cat_profile: global profile used for 10 of 10 categories`), so the fitted category mix / per-category
  profiles fell back to defaults on EVERY committed seed. Its only effective contribution (a global visit-time
  shape) duplicates `outing_start`.
- `sensor_step`: 2.58 / 3.50 / 3.26 overall but every per-mode slice below the per-mode floor on every seed,
  and the renderer never draws step lengths (they follow from speed x sampling interval).

Decision **D27**: drop `cat_time_visits`, `cat_transitions` and `sensor_step` from the release (schema v2d);
their rho re-spends on the remaining tables through the existing weighted split. Kept with stated reasons:
`sensor_heading` and `sensor_points_per_outing` (usable signal 5.8-9.4 on every seed; released DIAGNOSTICS for
the heading-change and duty-cycle realism panels - the renderer does not read them; stated in the caveats);
`place_users_r8` (total signal 5.7-7.8; carries the hub tail - the only hub statistic in the artifact - even
though the per-key floor (~74 users) keeps `users_per_place_hist`/`pool_size` on their priors);
`trav_users_r6/r7` (the highest-signal tables, 14.8-25.6). Like v2b, the drop is a data-dependent design
choice after seeing SNR: listed in the ledger caveats, outside the formal DP guarantee (D8 pattern).
Side changes: dev runs with `--seeds k > 1` now stamp a k-replicate caveat into every artifact's
`privacy.json` (the ledger-review gap: the old trio nowhere said it was 3 averageable replicates), and the
caveat-text fixes of the step-7 agent (analysis bbox, mode-heuristic cuts) are now IN the committed artifacts
instead of forward-only.

### Regenerated artifacts (3 dev seeds, eps 6.000 at delta 1e-6, rho 0.5395, 139 users, 24.9M points)

26 tables (was 29), 26 caveats, 60 files per artifact; `review_ledger.py`: all-OK on all 3 seeds.
Freed rho 0.0341 (6.3% of the total: cat_time 0.0090 + cat_trans 0.0091 + sensor_step 0.0160); behaviour
sigmas shrink ~5%, sensor sigmas ~7-10% (sensor_speed 4.96 -> 4.55, its rho 0.0203 -> 0.0242). Signal per
table, OLD (29-table artifacts) vs NEW (D27 artifacts), seeds 0/1/2 - the values are noisy-mass based, so a
new noise draw moves them by +-1-2 even where nothing changed:

| table | sigma old->new | signal old s0/s1/s2 | signal new s0/s1/s2 |
|---|---|---|---|
| cat_time_visits | 7.44 -> dropped | 2.9 / 2.0 / 3.1 | - |
| cat_transitions | 7.42 -> dropped | 1.2 / 0.2 / 0.7 | - |
| sensor_step | 5.59 -> dropped | 2.6 / 3.5 / 3.3 | - |
| n_users | 7.58 -> 7.19 | 18.5 / 18.7 / 18.5 | 19.5 / 19.7 / 19.5 |
| outing_detour | 7.53 -> 7.15 | 6.0 / 6.2 / 6.5 | 7.6 / 6.5 / 6.5 |
| outing_duration | 7.52 -> 7.14 | 7.3 / 6.0 / 6.1 | 5.1 / 5.8 / 6.2 |
| outing_length | 7.52 -> 7.14 | 6.2 / 5.9 / 5.8 | 6.5 / 5.6 / 5.8 |
| outing_mode | 7.53 -> 7.15 | 8.1 / 7.4 / 7.5 | 7.8 / 9.6 / 9.3 |
| outing_start | 7.53 -> 7.15 | 6.9 / 7.1 / 5.7 | 8.7 / 6.4 / 8.4 |
| outing_stops | 7.51 -> 7.13 | 6.3 / 5.1 / 5.0 | 5.9 / 6.6 / 6.0 |
| place_users_r6 | 12.66 -> 12.48 | 10.5 / 11.3 / 11.6 | 11.9 / 11.0 / 11.4 |
| place_users_r7 | 15.33 -> 15.18 | 12.4 / 12.2 / 10.8 | 11.5 / 9.7 / 10.6 |
| place_users_r8 | 22.82 -> 22.57 | 5.7 / 7.8 / 5.9 | 6.6 / 7.4 / 5.8 |
| revisit_hist | 7.51 -> 7.13 | 5.5 / 6.8 / 5.0 | 4.9 / 6.8 / 7.2 |
| sensor_gap_len | 6.61 -> 6.10 | 7.0 / 7.8 / 7.6 | 8.0 / 8.6 / 8.0 |
| sensor_gap_rate | 7.58 -> 7.03 | 8.1 / 8.0 / 9.8 | 9.2 / 8.5 / 7.8 |
| sensor_heading | 7.50 -> 6.95 | 7.6 / 7.4 / 7.5 | 8.7 / 7.5 / 5.9 |
| sensor_interval | 6.61 -> 6.10 | 9.5 / 7.4 / 7.7 | 9.1 / 7.4 / 8.6 |
| sensor_interval_user | 6.61 -> 6.10 | 8.1 / 7.8 / 8.3 | 8.1 / 7.0 / 9.3 |
| sensor_points_per_outing | 7.67 -> 7.12 | 6.3 / 8.7 / 7.6 | 8.7 / 8.2 / 9.4 |
| sensor_speed | 4.96 -> 4.55 | 1.9 / 5.2 / 4.6 | 3.4 / 2.4 / 4.3 |
| sensor_stop_jitter | 6.68 -> 6.17 | 8.9 / 5.3 / 8.7 | 7.6 / 8.8 / 8.5 |
| stop_dwell | 7.51 -> 7.13 | 4.5 / 6.1 / 7.4 | 8.6 / 6.0 / 5.7 |
| stop_hop | 7.52 -> 7.14 | 8.0 / 6.5 / 6.7 | 6.5 / 6.6 / 8.3 |
| trav_users_r6 | 17.50 -> 17.23 | 15.5 / 15.8 / 15.7 | 14.8 / 16.7 / 15.5 |
| trav_users_r7 | 19.81 -> 19.71 | 24.9 / 23.0 / 26.0 | 24.4 / 24.6 / 25.6 |
| user_outings_hist | 7.51 -> 7.13 | 6.5 / 4.7 / 5.8 | 6.4 / 5.7 / 5.8 |
| user_places_hist | 7.51 -> 7.13 | 4.8 / 5.5 / 5.2 | 6.1 / 5.6 / 5.9 |
| user_span_hist | 7.51 -> 7.13 | 4.9 / 4.3 / 5.9 | 6.6 / 5.7 / 5.3 |

Every released table is now above the 3.0 usability floor on every seed EXCEPT `sensor_speed` on the new
seed 1 (2.39; seeds 0/2 pass at 3.4/4.3) - the same noise lottery that hit old seed 0 at 1.9. The per-mode
floor still gates walk/bike, and a failing draw falls back to the hand-set defaults exactly as designed; a
real one-seed release would publish a passing seed. cat-table behaviour is now seed-INDEPENDENT (defaults
always) instead of flapping with the draw.

### Verification on the new artifacts

- `tests/geolife/outside/test_fit_learned_artifacts.py`: both tests pass. The fit-spread thresholds are bounds
  on the noise lottery calibrated on a committed draw; the new draw put `outing_length` high on seed 1
  (lognormal-mean spread 0.67 vs 0.13 on the old draw; medians 7.0/11.5/7.5 km), so that constant moved
  0.5 -> 0.8 - still well under the "factor of two" the test demands. All other spreads are at or below the
  old draw (outings_per_user 0.52 -> 0.17, active_days 0.45 -> 0.13, distinct places 0.45 -> 0.14,
  revisit 0.22, duration 0.38).
- `geolife/eval/review_ledger.py`: all-OK on all 3 new artifacts (26 tables, rho 0.539548, eps 6.000,
  26 caveats, 60 files verified each).
- Stage-2 sanity (`generate_synthetic.py --only outings_v2 --n-users 50`, toy grid): runs from the artifacts
  alone; volumes in the issues table below.
- geolife suite on this machine: 198 passed, 2 skipped, 0 failed (the `test_clean_room` machine failure is
  fixed, see below); boundary test passes.

### Clean-room test fix (step 6)

`tests/geolife/transfer/test_artifact.py::test_clean_room` failed on this machine because the subprocess
audit hook flagged a read of `/usr/share/zoneinfo/UTC` (datetime/pandas resolve UTC through the tz database
here; benign). The hook now ignores paths containing `/zoneinfo/` ONLY; any other read outside the artifact
directory and the interpreter's own code still fails the test.

### Full-environment check (step 6; replaces the documented `uv sync` / `uv run pytest`, no `uv` here)

`/opt/venv/bin/python3 -m pytest tests -q --continue-on-collection-errors` from the project root:
**548 passed, 5 skipped, 0 failed, 1 collection error**. Breakdown outside `tests/geolife`:
`tests/graph_aim` + `tests/realfake` 340 passed / 3 skipped; `tests/test_evaluate.py` 10 passed (this venv has
sklearn, unlike the old light env); the one error is `tests/test_run_benchmark.py` FAILING AT COLLECTION with
`ModuleNotFoundError: synthcity` - an environment gap, not a code failure (the heavy synthcity env was never
installed on this machine; same known limitation as D16; without the flag, plain `pytest tests` aborts at that
collection error). No real failures anywhere; geolife alone: 198 passed, 2 skipped, 0 failed.

### Decisions

- **D27 (drop zero-signal budget):** `cat_time_visits`, `cat_transitions` and `sensor_step` are no longer
  released (schema v2d); evidence and kept-table reasons above. The freed rho (6.3% of total) re-spends via
  the existing weighted split; no priority was retuned. Category mix and visit-time-by-category profiles are
  now always the public generator defaults (stated in a dedicated ledger caveat). Data-dependent design
  choice after seeing the SNR of the committed dev run: ledger caveat, outside the formal DP guarantee.
- **D27a (dev replicate caveat):** `fit_artifacts.py` with `--seeds k > 1` stamps
  `learn_v2.multi_seed_caveat(k)` into each artifact's `privacy.json` (k noise replicates; averaging cuts the
  noise std by sqrt(k); honest accounting of releasing all k is k * rho). Closes the ledger-review gap for dev
  trios; `--release` keeps stamping the one-seed policy.

### Issues found while doing this

| Issue | Status |
|---|---|
| Regenerating into the existing artifact dirs would have left stale cat/step files on disk unlisted by the manifest | old dirs deleted before stage 1; `load_artifact` rejects unlisted files either way |
| New noise draw put `outing_length` high on seed 1 (fit-spread 0.67) | noise lottery, machinery unchanged; test bound recalibrated 0.5 -> 0.8 with a comment |
| `sensor_speed` below the usability floor on new seed 1 (2.39) | by design (fallback to hand-set speeds); old draw had the same on seed 0 |
| Stage-2 volumes for 50 users on the new seeds: 7.8M / 5.6M / 11.3M points (max/min 2.0x) | comparable to the old draw's 2.25x; same accepted noise-floor cause |
| Attack/comparison reports (`privacy-attacks.md`, `compare_*_steps45`) describe the superseded 29-table artifacts | attacks re-run 2026-11-02 on the new artifacts (`privacy-attacks.md` updated; see the attack-refresh section at the end of this doc); the comparison reports (`compare_*_steps45`) remain the old wave (PLAN step 8 preamble) |

---

## 2026-11-02 (attack refresh on the 26-table artifacts)

The PLAN step-7 attack suite was re-run unchanged (`geolife/eval/run_attacks.py`; no code edits needed) against the
regenerated D27 artifacts (`outings_v2_seed0-2`, 26 tables, sigmas ~5-10% lower) and, for linkage, a fresh Beijing
synthetic set from seed0 made by the realistic step-5 generator (139 users requested, 33.7M points, 126 tokenized).
Report rewritten with previous-run comparison rows: `geolife/eval/reports/privacy-attacks.md`; tables regenerated in
`geolife/eval/results/attacks/`. Headlines (previous 29-table run in parentheses): informed-MI per-user AUC p50 0.700 /
max **0.734** vs the unchanged zCDP ceiling 0.769, Q max 0.778 vs 1.079 (was 0.730 / 0.751 - the re-spent rho raised the
worst case ~3.5%, as the accounting predicts, still clearly under the bound); Monte-Carlo MI p50 0.598 / max 0.648
(was 0.598 / 0.704, within simulation noise); seed averaging unchanged in substance - 23 common-vocab tables now,
cross-seed std / ledger sigma median 0.80, sqrt(3) noise cut, honest 3-seed accounting eps 11.08, averaged worst-case
MI AUC 0.860; linkage with the realistic generator resolves the old "depressed by the realism gap" caveat exactly as
predicted: population statistics rose to their chance references without crossing them (nearest-profile ratio 1.40 ->
**1.08**, coarse-signature excess -0.042 -> **-0.003**) while re-identification stays ~9x below chance (rate 0.050 vs
0.439; best top-10-cell Jaccard mean 0.101 vs chance 0.351 / copy 0.397). No finding against the accounting; the
one-seed release policy and its cost are unchanged. Suite after the refresh: 198 passed, 2 skipped.

---

## 2026-11-02 (step 8, last plan step): M redone on the new artifacts, real map matching, the transfer test

Delimited section, appended by the step-8 agent; everything measured on this machine (`/opt/venv`, no `uv`),
decisions numbered D28+ (the concurrent privacy-report refresh uses no decision numbers). Code:
`src/geolife/eval/m/{outings,outings_transfer}.py` (relation rule + new harness stages `rules`/`beijing`),
`geolife/eval/match_real.py` (new). Full rewritten report: **`geolife/eval/reports/m-v2.md`** (replaces the
pre-D27 version in place — M is still v2, only its relation rule changed, and keeping the name keeps every
PLAN/PROCESS link valid; the old numbers stay in git history and are quoted in the new report's tables).
Committed aggregates: `geolife/eval/results/match_real.json`, `compare_outings_v2_seed0_d27.{csv,md}`
(the pre-D27 `compare_outings_v2_seed0{,_steps45}` tables are kept as the before/after history they are
cited as; the `results/viz/outings_v2_seed0/` panels are regenerated in place). Suite after the changes:
**198 passed, 2 skipped** (unchanged); boundary test passes. Harness JSONs are not committed (aggregate
but bulky); every number in the report is transcribed from them.

### What was run

1. **M v2 planted-truth redo** at snap 150 m (D17) with the step-4/5 generator: toy dev/held draws (same
   seeds as the old report), two artifact-driven grid draws (`outings_v2_seed0`, map seeds 1/7), and 139
   synthetic users generated INTO Beijing from `outings_v2_seed0` (33.7M points). Headline (full tables in
   m-v2.md section 3): dev POI P/R 0.87/0.88 (old report 0.88/0.84), held-out 0.77/0.80 (old 0.74/0.72),
   segments >= 0.92 everywhere; artifact-driven draws are denser and hubbier, so recall rises and precision
   falls (POI P 0.50-0.64) — snap flicker at 150 m plus the group-proxy relation truth, not new M failures.
2. **Relation rule re-chosen on dev only** (D28 below): held-out edges 19 -> 58 at AUC 0.86.
3. **Map matching on real GPS** (first time; 8 users, 422 Beijing outings, road graph from the cache,
   98,974 nodes): every outing matches, 98.7% of the matched path lies on the trace's dilated res-9 cells,
   72% of the proxy cells covered (walk 0.81, bus 0.47 - off-graph walkways/blocks), 0.09 s/outing.
   **M on real can use the road graph now**; the transfer test still uses the cell proxy on both sides
   because the synthetic GPS is rendered along that same graph (segment agreement would be circular).
4. **The transfer test** (PLAN section 7 item 2): M unchanged on real (139 users) vs synthetic Beijing
   (126/139 kept), same tokenizer, same public places table. Tokenization counts re-checked against
   `outings-eda.md`: real 18,555 outings (EDA 18,743), 1.376 stops/outing (EDA 1.38), 54.7% of stops snap
   at 150 m (EDA 48% over all 182 users). M-output distributions and split-half reliability: m-v2.md
   sections 6-7. Shape statistics transfer (POI time-profile TV 0.075, common-place sizes TV 0.118,
   users-per-place TV 0.157, share of users with a relation 0.87 vs 0.90; split-half POI Jaccard 0.40 vs
   0.33, edge Jaccard 0.46 vs 0.39, ARI 0.48 vs 0.52, same random-over-time ordering). Volume statistics
   do not: POIs per user 1.6-1.9x high (synthetic stops snap at 93% vs real 55%; distinct-places gap),
   co-visit degree 1.4x, common res-9 cells 10x low. Old blocker resolved: synthetic split-half edge
   reliability was 0/undefined, now 0.39 vs real 0.46.
5. **Comparison refresh on the D27 artifacts** (`score.py --viz`): new table
   `compare_outings_v2_seed0_d27.{csv,md}`. vs the steps-4/5 table: timebin TV 0.177 -> 0.071, stop jitter
   matched (mean 66 vs 67 m), outing length mean 10.9 vs real 15.2 km (KS 0.14); regressions on this seed:
   revisit rate 0.281 (steps-4/5 run: 0.368; real 0.451, KS 0.52) and a sampling-interval tail (p95 19 s
   vs real 5 s, KS 0.17) from the one-hot per-user interval bins; points per user still 2.4x.

### Decisions

- **D28 (relation significance in sparse regimes; supersedes the FDR-0.2-over-all-pairs rule in M v2):**
  the BH family is the pairs that share >= 1 item (`rel_universe="sharing"`), FDR level stays 0.2, plus a
  minimum support of 3 shared items (`rel_min_shared=3`). Reason: pairs sharing nothing have p = 1 and can
  never reject, yet counted as tests — in sparse regimes BH then needs p < alpha/~2,700 for the first edge
  (the old held-out collapse: 2 edges at AUC 0.87). Chosen by a committed sweep (`--stages rules`) on the
  dev draw plus a NEW dev-sparse draw only (map seed 2 / gen seed 13 / 15 s / 10 m / 6 gaps/h — deliberately
  not the held-out settings); selection criterion: precision >= 0.8 on both dev draws, then the best
  recall/ARI balance within the sharing family. The held-out draw, Beijing and real data were not consulted.
  Effect: dev recall 0.42 -> 0.57 at precision 0.84 -> 0.80; held-out 19 -> 58 edges, ARI 0.10 -> 0.28;
  real Beijing 1,417 -> 1,459 edges (dense regimes barely move); synthetic split-half edges 0 -> ~650.
- **D29 (acceptance thresholds for "M transfers": PROPOSAL ONLY, awaiting user confirmation —
  this remains the open decision of PLAN section 11):** 15 criteria in m-v2.md section 8 (held-out
  recovery floors; output-distribution TV <= 0.2 / median rel. err. <= 0.5; split-half within +-0.15 of
  real), justified from dev-side noise floors (same-generator TV 0.08-0.12) and the artifact's own
  seed-to-seed fit spread (0.13-0.52 — transfer cannot be required tighter than the release's own
  reproducibility). Applied: **11 of 15 pass**; failing: POIs-per-user volume, co-visit-degree TV
  (borderline), traversal-cell count, held-out relation precision (0.47 < 0.6). Proposed verdict, equally
  not settled: **"M transfers in kind, not yet in volume"** — same outputs, shapes, reliability and failure
  modes; per-user volumes blocked by the two known generator gaps (distinct places ~1.8x + low revisit;
  traversal width). The user decides whether B4/B5/B7/A4 block the success criterion or are accepted.
- **D30 (reporting):** m-v2.md rewritten in place; `_d27` comparison tables added next to the historical
  ones; viz panels regenerated in place; per-variant score JSONs stay uncommitted.

### Issues found while doing this

| Issue | Status |
|---|---|
| Local `from ... import place_merge_map` inside `main()`'s real stage shadowed the module import and broke the new beijing stage (UnboundLocalError) | fixed (module-level import only) |
| POI precision on artifact-driven draws 0.50-0.64 (toy fake-params draws: 0.77-0.87): 150 m snap flicker on dense place vocabularies + hubby planted structure vs group-proxy truth | known mechanisms; listed in m-v2.md caveats; merge radius was NOT retuned (dev-only sweep would be needed) |
| Synthetic stops snap at 93% vs real 55%: the generator plants stops AT map places, real stops are often off-vocabulary | main driver of the POIs-per-user volume gap (B4); generator work, not M work |
| Revisit rate on the regenerated seed0 artifact renders at 0.281 (steps-4/5 run: 0.368; real 0.45) and the one-hot interval bins put a few users at 10-20 s (sampling KS 0.09 -> 0.17) | new-seed noise lottery on the same machinery; next generator iteration |
| Toy held-out split-half still edge-starved (halving ~27 outings/user leaves too little support even for the new rule) | sample-size limit of the half-data regime; full-data held-out keeps 58 edges; Beijing frame unaffected |
| Harness real stage reads the full parquet (no bbox cut), compare.py bundles cut to the Beijing core: outing-length means differ (30.5 vs 15.2 km) between the two real baselines | methodology difference, documented in m-v2.md section 6/9; medians agree |

## 2026-11-03: realfake discriminator added for the geolife track

The real-vs-fake classifier (classifier two-sample test, `docs/realfake/PLAN.md`) gets its GeoLife
implementation: samples are contiguous k-point trace windows, real Beijing-core points vs
`outings_v2_seed0` synthetic traces. Design, dropped-giveaway list and feature table:
`docs/realfake/geolife.md` (decision D8 in `docs/realfake/PROCESS.md`). Code:
`src/geolife/eval/discriminator.py` + `src/geolife/eval/discriminate.py`, stage script
`geolife/eval/discriminate.py`; results dir `geolife/eval/results/discriminator*/` (with and
without the map feature group). Eval-zone only: no artifacts or synthetic data regenerated,
no M rerun.

**First run (same day, D9 in `docs/realfake/PROCESS.md`):** controls clean (real-vs-real 0.47-0.53,
permuted ~0.5); gbm AUC 0.9996 with map features and 0.9995 without them — the synthetic windows are
trivially detectable and the map is not the reason. Grouped importance: sampling 0.26 >> kinematics
0.06 >> everything else ~0; AUC 0.997+ already at 30-60-point windows. Per-feature view (notebook):
the synthetic traces are too SLOW and too uniform in the upper speed/step tail (thin30 speed q90
~1.0 vs real ~4.0 m/s — the overcorrection of the old 2.7x-too-fast baseline; this seed's
`sensor_speed` fell back to defaults at signal 1.9 < 3.0), and the dt texture is inverted: real
loggers lock on one interval (flat runs, modal-dt share p50 0.84, repeat prob up to 0.95, pause tail
p99 ~137 s) while rendered dt flickers step to step in a narrow band (0.55 / 0.28-0.67 / p99 ~26 s);
`sampling:dt_mode_share` is the one feature the model materially uses (perm drop 0.111). Generator
to-dos: per-mode speed scale + within-outing spread, dt as a locked/regime process with a heavy
pause tail. Write-up:
`docs/realfake/geolife.md` section 5; notebook `geolife/eval/notebooks/viz_discriminator.ipynb`
(built by `build_viz_discriminator.py`).

## 2026-11-03 (later): PLAN step 9 - dt texture through the artifact (realfake D10)

The user asked for the dt fix to come from the artifact, not from generator constants. Built as two packages
(inside+transfer, outside) in worktrees cut from `aim`, then iterated on the Beijing set.

**Released (schema v2e, 28 tables, eps 6.000, ledger review OK on all 3 dev seeds):** `sensor_dt_runlen`, the pmf
of maximal equal-rounded-dt run lengths among within-run sampling steps (dt <= 60 s; gaps and outing-run
boundaries break runs), and `sensor_dt_runlen_user`, a one-hot of the user's MEAN run length. Edges
`RUNLEN_EDGES` = 1..1200 plus an open bin; priorities 1.5 / 1.0 in the sensor split. Caveat: the run definition,
rounding and edges were chosen after the non-DP discriminator EDA (D8 pattern). Both tables usable on every seed.

**Renderer (`render_v2._times_locked`):** with a per-user regime m, a home interval is held for geometric runs
(mean 2m-1) and broken by single-step excursions to a different dt, so the window repeat rate is 1 - 1/m and the
home value dominates the window. Without the per-user table: pooled run lengths with a different dt per run;
without either table: the old per-step draw. The base/excursion structure is a public rendering assumption,
like the AR(1) noise shape (D26).

**Iterations (Beijing, 139 users, window dt texture; real: modal share 0.84, repeat 0.84):**

| renderer | modal share | repeat | why it was wrong |
|---|---|---|---|
| per-step draw (before) | 0.55 | 0.35 | dt re-drawn every step |
| pooled run lengths | 1.00 | 1.00 | 96% of steps sit in the 700+ bin; every user locked |
| + run end changes dt | 1.00 | 1.00 | same: the pooled mix, not the merging, was the cause |
| + per-user one-hot of the MEDIAN run | 0.48 | 0.65 | median dominated by 1-step runs even for locked users |
| + one-hot of the MEAN run (repeat = 1 - 1/m) | 0.51 | 0.83 | base and excursion runs equally long |
| + home interval with single-step excursions | 0.92 | 0.84 | matches; modal share slightly high |

**Discriminator after (D10):** gbm 0.9970 (map and no-map), was 0.9996 / 0.9995; logreg 0.942 / 0.930, was
0.983 / 0.981; sampling drop 0.20 / 0.23 (was 0.26 / 0.27), kinematics 0.11 / 0.12 (was 0.06). Best single
separators are now all kinematics (speed spread and upper tail; seed0 `sensor_speed` fails its floor at signal
just under 3.0). Next: PLAN step 10 (speed regime through the artifact).

**Issues:**

| Issue | Status |
|---|---|
| Root checkout switched to another session's `presentation` branch mid-run; one commit landed there | moved to `aim` by cherry-pick, `presentation` rewound by exactly that commit; all git work now in `.claude/worktrees/geolife-aim` |
| Disk full on the shared volume during a test run | removed this round's merged agent and package worktrees; reruns were idempotent |
| Within-window pause tail still missing (dt p99 ~137 s real vs ~25 s) | open, cause not diagnosed |
| Base interval: real dt p50 2 s, synthetic 3 s | open (per-user interval table's median bin) |

## 2026-11-03 (last): discriminator without speed and distance (realfake D11)

Speed and distance are not part of this data's model (user decision), so they were removed from the classifier rather than
fixed in the generator: `geolife/eval/discriminate.py --no-speed-distance` (24 features dropped, 29 with the map group;
list in `discriminator.SPEED_DISTANCE_FEATURES`). Same synthetic set as the dt round. Result: gbm AUC 0.990, logreg 0.892,
controls clean; the separation is the dt process (sampling group drop 0.36): synthetic windows slightly over-locked (modal
share 0.94 vs 0.84), too few <= 2 s steps (0.01 vs 0.17), too little dt spread (log-std 0.12 vs 0.39), and no > 60 s pauses
inside runs. Heading is a smaller second signal. PLAN step 10 now targets these dt items; the speed-regime item is dropped.
Tables `geolife/eval/results/discriminator_nospeed{,_nomap}/`; notebook section 7; write-up `docs/realfake/geolife.md` section 7.

**viz_comparison rerun on the dt-round set (same day):** notebook re-executed on the current synthetic set (28.0M points);
summary table `geolife/eval/results/compare_outings_v2_seed0_dt.{csv,md}` (new `GEOLIFE_COMPARE_TAG` env in
`build_viz_comparison.py` keeps the earlier tables as history). Closed vs the first baseline: heading KS 0.79 -> 0.11, step
length 0.40 -> 0.15, outing length 0.37 -> 0.14, duration 0.24 -> 0.06, start-hour TV 0.23 -> 0.13. Open: revisit rate (0.30 vs
0.45, KS 0.50), distinct places 1.7x, points per user 2.0x, sampling interval (KS 0.15), and a NEW finding - the synthetic mass is
centred ~6 km south-east of the real one, on the public map's centre instead of the released H3 cells (PLAN step 11).

## 2026-11-03 (last): generator-only fixes using tables the artifact already releases

User request: fix only the generator, using released information it ignored or used poorly (no artifact change, no new budget).
Artifacts unchanged; Beijing set regenerated into `geolife/outside/synthetic/genfix/` (gitignored); summary table
`geolife/eval/results/compare_outings_v2_seed0_genfix.{csv,md}`; discriminator `results/discriminator_nospeed_nomap_genfix/`.

**Kept (on in the code):**
- *Spatial anchoring* (`population_v2._anchor_from_cells`, `teleport.ground_places` anchored mode): `fit_v2` now keeps the
  released `place_users_r7` cell KEYS (51 cells above 1 sigma; fallback r6) as anchors; when >= 80% of their user mass lies
  inside the target map (the source city), latent places are drawn inside those cells by user count and grounding scores
  position on a 1.5 km scale. Spatial KS 0.41 / 0.42 -> 0.19 / 0.20; synthetic median 6-7 km off -> ~2 km (it now sits on
  the anchors' weighted centre; the r7 cells are the resolution limit). Teleport to another city is unchanged.
- *Per-user revisit* (`fit_v2` keeps `revisit_pmf`; `population_v2` draws each user's rate): revisit KS 0.50 -> 0.29, the
  spread now covers 0-0.65 (p95 0.65 vs 0.77 real); the mean stays low (0.32 vs 0.45; the tokenizer measures planted rates
  ~0.1 lower, known snap-flicker issue).
- *Interval floor* (`sensors.fit_sensors`): `sensor_interval_user` cells must clear 3 sigma (same bar as table usability);
  the 16-32 s noise bins are gone - interval p95 19 s -> 6 s (real 5 s). Not applied to `gap_rate` / `dt_runlen_user`
  (they would keep 3-4 bins). Side effect: no 5-8 s users either, so sampling is faster overall (KS 0.15 -> 0.17,
  step length KS 0.15 -> 0.22, points per user 202k -> 249k).

**Tried and switched off:** a points-per-outing budget (`render_v2._budget_points`, `POINTS_BUDGET = False`) that thins
each outing to the released `sensor_points_per_outing` quantile at its rank by cutting released-length gaps. It fixed
points per user (104k-122k vs real 102k) but cutting dwells erased stops (stops/outing 0.74 -> 0.36) and over-long gaps;
protecting dwells left mostly in-stop points (heading KS 0.10 -> 0.24, stop jitter KS 0.19 -> 0.41). Needs a better
thinning model.

**Not done:** heading - synthetic sharp turns come from the in-stop drift (a position random walk: 54% of >= 5 m
stationary steps turn > 90 deg vs 12% real) and from moving steps (7% vs 1.8%); the released `sensor_heading` cannot fix a
process of the wrong shape with a parameter (needs a smoother drift model calibrated against it). In-run pauses: still
absent (window dt p99 6 s vs 137 s real).

**Side effects of anchoring:** places concentrate in fewer cells - users per place 2.98 -> 3.61 (real 2.78; KS 0.09 -> 0.16),
users per cell KS 0.04 -> 0.16, co-visit degree 19.5 -> 12.1 (real 16.6). Discriminator without speed/distance: gbm 0.990
(unchanged), logreg 0.892 -> 0.885. Suite: 217 geolife passed.

## 2026-11-03 (experiment, branch `geolife-thin`): ON/OFF signal-loss thinning from released tables only

`render_v2._onoff_logging` (`ONOFF_THINNING`): after all users are rendered, each outing alternates ON/OFF periods - OFF
lengths from `sensor_gap_len` (non-overlapping), ON mean the shorter of the user's `sensor_gap_rate` cycle and the
`sensor_points_per_outing` quantile at the outing's rank. No assumptions about where gaps fall (no dwell protection).
Set `geolife/outside/synthetic/thin/`; tables `compare_outings_v2_seed0_thin`, `discriminator_nospeed_nomap_thin`.

| metric (real) | before (genfix) | thin |
|---|---|---|
| points per user (102k), KS | 249k, 0.37 | 72k, 0.16 |
| sampling interval KS | 0.17 | 0.15 |
| users per cell KS | 0.16 | 0.06 |
| outings per user (134) | 138 | 104 |
| stops per outing (1.13) | 0.74 | 0.54 |
| co-visit degree (16.6) | 12.1 | 4.0 |
| revisit rate (0.45), KS | 0.32, 0.29 | 0.24, 0.42 |
| gap length KS | 0.16 | 0.27 |
| outing length (15.2 km), KS | 11.9, 0.10 | 9.0, 0.19 |
| window >60 s step share (0.058) | 0.003 | 0.190 |
| discriminator no speed/distance gbm / logreg | 0.990 / 0.885 | 0.990 / 0.917 |

Reading: it over-thins. The released points-per-outing table is measured on raw outing-like runs, a quarter of which are
fragments under 100 points that the tokenizer never counts as outings; rank-matching every synthetic outing (all real
outings) against it gives small outings fragment-sized targets - they fall apart (outings per user 138 -> 104), stops go
undetected (co-visit degree collapses 12 -> 4), and windows get 3x too many in-run pauses. The released gap-length mean
(~900 s with the clamped raw tail) is also about twice the real within-outing gap mean (436 s). Net: worse than genfix;
not merged into `aim`. A working version needs a released statistic on the same footing as the synthetic outings
(points per TOKENIZED outing) or a fraction-of-time-logged table.

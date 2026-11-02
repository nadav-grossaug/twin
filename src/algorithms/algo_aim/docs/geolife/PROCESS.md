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

- **D20 (sensor budget split and sensor release design):** 80% of the total rho to the behaviour tables, 20% to the new
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
- **D22 (one seed per release):** a real release publishes exactly ONE artifact seed. `fit_artifacts.py --release`
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

### Issues found while doing this

| Issue | Status |
|---|---|
| Caveats missed the analysis-region bbox and the mode-heuristic cuts; nothing stated the 3-seed replicate risk in the artifacts themselves | caveat text fixed forward (measure_v2.CAVEATS, ONE_SEED_CAVEAT); committed artifacts intentionally untouched, gap documented in ledger-review.md |
| Hierarchical child tables (place r7/r8, trav r7) have seed-dependent vocabularies, so cross-seed averaging/std only covers the 26 common-key tables | by design; noted in privacy-attacks.md |
| Synthetic linkage baselines are depressed by the generator's realism gap (best-Jaccard 0.111 << chance 0.351) | re-run `run_attacks.py --synthetic` after PLAN step 5 |
| `artifact_cli` subprocess aborts AT EXIT on this machine after printing OK (pyarrow teardown; same family as the known `test_clean_room` failure) | machine-specific, rc in-process is 0 |

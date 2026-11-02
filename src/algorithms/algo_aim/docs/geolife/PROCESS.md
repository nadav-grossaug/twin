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

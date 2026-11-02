# Process log: GeoLife track

Log for the GeoLife DP mobility track (user-level DP, map-anchored artifact, teleport to a public map). Related docs in
this folder: `PLAN.md` (v1 plan, home/work), `PLAN_OUTINGS.md` (v2 plan, outings), `OUTINGS_PARAMS.md` (the `params` dict),
`TASKS.md` (the original task board), `p0-feasibility.md` (Overture and routing). SNAP log for comparison: `docs/snap/PROCESS.md`.
Layout and usage: `geolife/README.md`, `src/geolife/README.md`.

## Phases (2026-11-01)

| Phase | Task | Branch / owner | Status |
|---|---|---|---|
| 0 | P0 Overture and routing feasibility, P1 loader and EDA | wave 1 | done, merged |
| 1 | DP core (`dp.py`), artifact I/O and inspector | wave 1 | done, merged |
| 2 | GEN (fit, population, teleport, render), TOK (stays, trips, matcher, learner CLI), MAP (Overture provider, places) | wave 2 | done, merged |
| 3 | LEARN2 (weighted budget, hierarchical place tier), M v1 (home/work, shared anchor) and its transfer harness | wave 3 | done, merged |
| 4 | Outings pivot: T0 outing EDA on all 182 users, T1 tokenization, T2 learner v2 (T2b coarse time, drop noise-only tables), T3 generator v2, T4 M v2 and real-vs-synthetic comparison | `feat/geolife-impl` | done, merged |
| 5 | Restructure into inside / transfer / outside / eval (this commit series), three stage scripts, boundary test, provenance allowlist | agent RESTRUCT | done |
| 6 | Run M v2 on the real GeoLife traces and compare with synthetic (stage 3 with `--real-points`), P5 calibration of render noise from released stats | open | open |
| 7 | P6 hardening: empirical privacy attacks in `eval/`, artifact signing | open | open |

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
  splits) with a few stops (200 m / 10 min). Output of M: places of interest, common places, crossings, a co-visit graph with communities. See `PLAN_OUTINGS.md`.
- **D6 (parametric first): the generator is a parametric outing process** driven by a small `params` dict
  (`OUTINGS_PARAMS.md`) fitted from the noisy tables, with sensor knobs in a public `generation` dict; no learned model on
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

## Issues and findings (from the runs; CHECKPOINTS section 3)

| # | Issue | Status |
|---|---|---|
| 1 | Stay-level overlap is weak: res 8 has 63 of 1,236 cells with 10 or more users (point-level overlap is strong) | motivated hierarchical tier (D3) |
| 2 | Real run at epsilon 6 (156 users, 7,156 stays): res-8 place overlap is pure noise; res 6 has 66% of mass above 3 sigma (v1), res 7 42%; `work_users_hist` 70%; behaviour tables 20-60% noisier from the reweighting; `zipf_s` falls back to 1.0 | known |
| 3 | Resolution and time design chosen after seeing SNR is unaccounted tuning | stated in ledger caveats (D7, D8) |
| 4 | Matcher on the real Haidian graph: 72% recovered, 29% extra in car mode; 53% recovered with unknown mode (footways collapse onto roads); tuned on Gaussian noise only | known |
| 5 | Overture: `categories` renamed to `taxonomy` / `basic_category`; no work category, so weak workplace grounding | handled in `outside/map/provider.py`; reason for D5 |
| 6 | M v1 on synthetic (150 users): main places P/R 0.996/0.987, relation edges P/R 0.97/0.88; fails above about 150 m position noise; visit-merge added after the first sensitivity run (held-out slightly optimistic), thresholds tuned to the generator's long stays | superseded by M v2 (`geolife/eval/reports/m-v2.md`) |
| 7 | M v2 was run on the real traces (139 users) with the res-9 cell proxy only: no road graph, matcher not exercised on real GPS; real-vs-synthetic gaps are large (see next steps) | partly done; see next steps 5-7 |
| 8 | Privacy attacks and artifact signing (P6); dropping the useless finest-tier rho (about 0.056) in v1 | open (phase 7) |
| 9 | Overture reachability from the air-gap is a risk; the map must be a cached public parquet fetched on the connected side | mitigated by `fetch_public_map.py` (D11) |
| 10 | Light env has no sklearn, so `tests/test_evaluate.py` and `tests/test_run_benchmark.py` (Adult) fail at collection; not a GeoLife problem | known |
| 11 | Two legacy tests/tools point at a map cache in another agent worktree (`AMS_CACHE`); they now prefer `geolife/outside/cache` and skip when absent | cosmetic |

## Cleanup decisions (2026-11-02)

- **D16 allowlist confirmed:** h3, networkx, shapely and pyarrow are allowed in `transfer/` and `outside/`; duckdb stays in `geolife/outside/fetch_public_map.py`. Written into `CLAUDE.md` rule 2. The dependencies are in `pyproject.toml` (`h3`, `networkx`, `shapely`, `pyarrow`; `duckdb` in the `fetch` group) and `uv.lock` was re-resolved. `uv sync` / `uv run pytest` on the full project env were not run end to end (heavy synthcity env); the geolife suite was run in the light env.
- **D17 place-snap radius 75 m -> 150 m** (default in `OutingsConfig`), because only about 30% of stops snapped at 75 m (48% at 150 m in the EDA). It is a public setting chosen after looking at the snap rate, so it is listed in the ledger caveats. M v2 and its report were measured at 75 m (see next step 2).
- Leftover empty worktree folders removed; the teleport demo PNG regenerated with the v2 generator.

## Next steps (known problems, not new features)

| # | Problem to fix | Where | Notes |
|---|---|---|---|
| 1 | Retire v1 code. v1 and v2 are entangled: `outside/fit_v2` imports `outside/fit`, `population_v2` imports `population`, M v2 imports `eval/m/core`, M v2 uses `transfer/tokenize/stays`, and `eval/m/transfer.py`, `eval/fakedata.py`, `inside/learn.py`, `inside/measure.py` and their tests are v1-only | `src/geolife`, `tests/geolife` | First move the shared helpers into v2 modules, then delete the v1 modules and tests; boundary test must still pass |
| 2 | Re-run M v2 and `geolife/eval/reports/m-v2.md` at the new 150 m snap radius (numbers were produced at 75 m) | `eval/m`, reports | also re-check the tokenization counts against `outings-eda.md` |
| 3 | Relation significance is too strict in sparse regimes (FDR 0.2 gives 2 edges on the held-out draw; AUC is still 0.87) | `eval/m/outings.py` | choose the rule on dev only, report on held-out |
| 4 | Learned (noisy) params fail the planted-stats check on `stop_revisit_rate`, `users_per_place_mean`, `share_places_ge2_users` on all seeds; fitted values vary a lot between noise draws | `outside/fit_v2.py` | denoise/shrink fits, report variance over several noise draws |
| 5 | Generator does not match the real regime: outings per user mean 51 vs 133 real, active-days p90 46 vs 167, outing length 12 km vs 30 km, co-visit degree 4.4 vs 20, no hub places (6 real places have 51-67 users), distinct places per user 43 vs target 25, users-per-place histogram shape uncalibrated | `outside/*_v2.py` | the real-vs-synthetic comparison in `eval/m/outings_transfer.py` is the check |
| 6 | Map matching is untested on real GPS (tuned on synthetic Gaussian noise); M v2 on real used the cell proxy | `transfer/map/match.py`, `eval/` | run on a real sample with the Beijing road graph, measure segment agreement |
| 7 | The outings parquet export has no `cells` column, so it cannot feed the traversal tables | `inside/build_outings.py` | add the column or document the in-process path |
| 8 | Privacy side of P6: attacks in `eval/`, ledger review, artifact signing, drop zero-signal budget | `eval/`, `inside/` | still open |
| 9 | Fine-bin time profile is an assumption: the 17-18h peak is not recovered in the 4 fine bins (the DP tables are coarse am/pm) | `outside/fit_v2.py` | decide whether to measure 3 bins or keep the assumption |
| 10 | Verify `uv sync` / `uv run pytest` on the full project env with the new dependencies | repo root | not run end to end |


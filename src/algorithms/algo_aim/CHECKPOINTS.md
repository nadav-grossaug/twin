# Checkpoints: hackaton-2026 (snapshot 2026-11-01 ~13:45; updated after outings pipeline merge (Geo); SNAP section updated after the v1.1 attribute-coupling round, 2026-11-02)

Written by the read-only "spokesperson" session and tracked in git from `chore/housekeeping`.
Use it to resume after moving environments. Verify each state with the commands at the bottom before trusting it.

## 0. Read this first

- **`main` contains SNAP (latest merge `be560fe`, restructured into inside/transfer/outside) and all Geo work including agent M (`b4762e4`).** Merges were local `--no-ff`; nothing was pushed.
- **The repo has no git remote.** All work exists only on this machine (branches, worktrees, gitignored data). To move envs, either copy the whole directory including `.git` and `.claude/worktrees/`, or push the branches to GitLab (`gitlab.dv2cloud.net`, via `glab` auth; no force-push, MR merge is a human action).
- **Not in git, must be copied by hand:**
  - GeoLife raw data: `geolife/data/` (gitignored) in worktree `.claude/worktrees/agent-a04900ec477b059b7`.
  - All `*/data/` folders (`adult-census/data`, `geolife/data`) are gitignored from `chore/housekeeping` on; copy them by hand. Exception by owner request (2026-11-02): `snap/inside/data/` and the full SNAP artifact/synthetic sets (all seeds, both rounds) are committed on `aim`, so the SNAP track is fully self-contained in git.
  - Python envs: `~/snapcheck` (WSL, SNAP), `C:/Users/NaomiD/geo-venv` (Windows, Geo), project `.venv` (Adult). All are rebuildable.
- **Stray files, do not commit:** `adult-census/data/adult.csv:sec.endpointdlp` (DLP marker), `C:\` directory inside worktree `agent-aa94e004e7c0f95b0`.
- Worktrees under `/home/naomid/projects/...` are stale WSL paths (marked prunable). Safe to `git worktree prune`.

## 1. Branch map

| Track | Branch | Worktree | Ahead of `main` | Last known commit |
|---|---|---|---|---|
| SNAP / graph-AIM | `feat/snap-facebook-eda`, `refactor/inside-transfer-outside`, `docs/inside-outside-readmes` | none (branches only) | 0 (all merged into `main`) | `be560fe` Merge docs/inside-outside-readmes |
| Geo / GeoLife | `feat/geolife-impl` | `.claude/worktrees/geolife-orch` | 0 (merged into `main`) | `7d7f59a` Merge M v2 + restructure |
| Adult demo | `main` / `chore/windows-env` | `.claude/worktrees/adult-win` | 0 | `8862c26` win32 env |

Agent branches are `worktree-agent-*`; their work is merged `--no-ff` into the track branch, so the track branch is the source of truth.
Check for unmerged agent work before discarding any worktree: `git log <track-branch>..worktree-agent-<id>`.

## 2. SNAP / graph-AIM (edge-DP + attribute-row DP, ego-Facebook)

**Status: v1.1 attribute-coupling round done (2026-11-02) on branch `aim-coupling` (worktree `.claude/worktrees/aim-coupling`), not yet merged to `main`.** Commits on the branch: packages A/B/C in the 2026-11-02 baseline snapshot (`d6640c4`), generality test `a7166ff`, package D `2d04584`, decision point + shipped config `31da504`, package E tried/dropped `171462e`, v1.1 outputs in separate dirs `e802ff7`, plus the docs round-up. v1 itself is merged to `main` (`be560fe`).

Layout (see `snap/README.md`, `src/graph_aim/README.md`, `docs/snap/PROCESS.md` D13-D14): `src/graph_aim/{inside,transfer,outside,eval}/` and `snap/{inside,transfer,outside,eval}/`. Three stages, each a script:
1. inside: `PYTHONPATH=src:snap/inside python snap/inside/fit_artifacts.py --seeds 3 --out snap/transfer/artifacts_v1_1`
2. outside: `PYTHONPATH=src python snap/outside/generate_synthetic.py --artifacts snap/transfer/artifacts_v1_1 --out snap/outside/synthetic_v1_1` (only `src` importable)
3. eval: `PYTHONPATH=src python snap/eval/score.py --seeds 3 --synthetic snap/outside/synthetic_v1_1 --out snap/eval/results_v1_1` (then open `snap/eval/notebooks/viz_comparison.ipynb`)

The v1.1 round writes to the `*_v1_1` dirs above; without the flags the stages use `artifacts/`, `synthetic/`, `results/`, which hold the frozen v1 round for side-by-side comparison.
Full run takes about 5 minutes (fit about 1.5 min, generate about 1.5 min, score about 2 min).

- Code: `src/graph_aim/inside/` (capping, privacy, measure, fit), `transfer/` (types, artifact, inspect, graph_io), `outside/generate.py`, `eval/` (evaluate, viz, baselines, attacks, utility). Data prep `snap/inside/prepare.py`, config `snap/inside/config/graph_aim_v1.json`, frozen attribute policy `snap/inside/attribute_policy.json`.
- Outputs in git: seed 0 of 7 variants in `snap/transfer/artifacts/`, two synthetic graphs in `snap/outside/synthetic/`, metrics in `snap/eval/results/` (all three: the frozen v1 round) and the v1.1 round next to them, same convention (`snap/transfer/artifacts_v1_1/` seed 0 of 7 variants, `snap/outside/synthetic_v1_1/` two graphs, `snap/eval/results_v1_1/` incl. `coupling_reference.csv`), notebooks `snap/eval/notebooks/viz_comparison.ipynb` (real vs synthetic plots) and `02_graph_aim_results.ipynb`, report `snap/reports/graph_aim_v1.md` (v1 and v1.1 sections). Seeds 1-2 are gitignored and regenerated by the stages.
- Docs: `docs/snap/PROCESS.md` (decisions D7-D18 and the "Attribute coupling round (v1.1)" section), plans (see note below), `research/graph-data-synthetic-generation.md`.
- Tests: 335+ passing (`tests/graph_aim/{inside,transfer,outside,eval}/` plus `test_boundary.py` and `test_generality.py`; the notebook round adds a few). Run `PYTHONPATH=src:snap/inside:snap/eval python -m pytest tests/graph_aim -q`, or `uv run pytest` where the project env works.
- Env: WSL `~/snapcheck/bin/python` (py 3.14; code must stay 3.11-compatible). `snap/inside/data/` (raw SNAP files and `processed/attributed_graph.npz`) is committed on `aim` since 2026-11-02 (owner request; `git add -f` over the gitignore rule), so nothing needs hand-copying for SNAP. Stage 1 and the real-data tests need it; stage 2 does not.

Done: G1-G5 (prep, fit, generator, artifact I/O, eval), R1 sampler rewiring, R2 measurement redesign (uncapped measurements, per-measurement caps), R3 link-prediction utility, R4 hub-tail pooling. Then an audit of what crosses the boundary:
- Found and removed an exact, un-noised capped-edge count in `provenance.json`; `validate()` now enforces a provenance key allowlist.
- Removed the `run_clean_room` test helper from `transfer/` (shipped code must not spawn processes); `test_boundary.py` now also forbids `os`, `subprocess`, `socket`, `shutil`, `tempfile`, `urllib`, `http`, `pickle`, `importlib`, etc. in `transfer/` and `outside/`, and runs stage 2 in a subprocess with only `src` importable.
- `score.py` aligns synthetic attribute columns to the real graph by name (artifacts store them sorted).

Headline results (3 seeds, eps per adjacency, error vs real): at eps 4 degree KS 0.013, triangles 0.112 (BTER 0.109), 2-way attribute TVD 0.071 (BTER 0.079), homophily 0.257 (BTER 0.306), assortativity 0.122 (BTER 0.269, but per-seed 0.10/0.24/0.02), transitivity 0.054 (BTER 0.020), clustering 0.043 (BTER 0.016), degeneracy 0.235 (BTER 0.200). Full tables: `snap/reports/graph_aim_v1.md`.

Open / known gaps:
- Per-seed spread remains (hub count is noisy); clustering, transitivity and degeneracy (sampler overshoots, about 140 vs 115) still weak; homophily improved in v1.1 but plateaus at gap about 0.4-0.5 for school/hometown (generator capacity, see PROCESS.md D17/D18).
- Link-prediction utility does not discriminate (0.99 everywhere); use structure metrics.
- Public settings chosen by looking at the real graph (attribute set and value labels, max degree, caps, bins, budget shares) are outside the DP guarantee; listed as ledger caveats. DP top-K is v2.
- The boundary is import rules plus a subprocess test, not an OS sandbox. Next step would be a container with no network and only `transfer/artifacts` mounted.
- The `eval/` zone now exists (as planned in `CLAUDE.md`); nothing outside it may import it (checked by `test_boundary.py`). Findings from `viz_comparison` (see `docs/snap/PROCESS.md` D14): synthetic graphs are more compact (paths 3 hops vs 4), degree mixing sits on the diagonal where the real graph links low-degree nodes to hubs, homophily is weaker (42% vs 59% same-group edges), neighbourhood colours are mixed where real ones are coherent. Not privacy-noise effects (the non-private graph shows them too).
- v2 ideas: adaptive AIM selection, learned encoders, node-DP; attribute-aware blocks were tried in v1.1 (package E, dropped) -- degree-consistent blocks within groups are the follow-up.

**Real-vs-fake classifier track (2026-11-02, built on the parallel `realfake` branch, merged into `aim` the same day):** a classifier two-sample test as a stricter eval (`docs/realfake/{PLAN,snap,PROCESS}.md`). Code: dataset-independent core `src/realfake/` (dataset, group-aware split with held-out fake seeds, numpy logreg + sklearn gbm, metrics, controls, grouped permutation importance, report), SNAP walk sampler/featurizer `src/graph_aim/eval/discriminator.py`, stage-3 CLI `snap/eval/discriminate.py` (`PYTHONPATH=src python snap/eval/discriminate.py`, ~15 min, needs the real graph). Tests `tests/realfake/` + `tests/graph_aim/eval/test_discriminat{or,e}.py`; boundary test also forbids `realfake` outside eval. First runs (results committed in `snap/eval/results/discriminator*.csv` for the v1 graphs and `snap/eval/results_v1_1/discriminator*.csv` for v1.1, notebooks `viz_discriminator.ipynb` / `viz_discriminator_v1_1.ipynb`, report `snap/reports/discriminator_v1.md`): controls clean, but every graph_aim variant incl. non-private is detectable at AUC ~0.9999 (gbm); attributes group dominates importance, falling 0.43 -> 0.24 from eps 0.5 to inf. Open: split the attributes group per feature (`none`/`other` artifact suspicion), `docs/realfake/PROCESS.md` issue 1; the composition gap is stated generator-agnostically in `docs/snap/attribute-composition-gap.md`. Log entry: `docs/snap/PROCESS.md` D19.

Plan files: `docs/snap/graph-aim-plan.md` (v1 method; moved there from `docs/graph-aim-plan.md`, references updated) and `docs/snap/attribute-coupling-plan.md` (the v1.1 round: degree x attribute, edge x attribute; implemented, see below).

**v1.1 attribute-coupling round: done (2026-11-02)**, implementing `docs/snap/attribute-coupling-plan.md`. Packages A (eval coupling metrics, `cpl_*` columns in `score.py`, `snap/eval/coupling_report.py`), B (measurement kind `attr_degree`, sensitivity (2 sqrt(2), sqrt(2)) brute-force tested), C (degrees assigned conditional on attributes; old artifacts byte-identical fallback) and D (three `attr_edge_pair` groups school/hometown/location, multi-channel rewiring) shipped; E (attribute-aware triangle blocks) tried and dropped (structure broke; the `within_group_triangle_share` metric is kept). Shipped config: `shares.attr_degree` 0.35, 4 degree bins, `homophily_groups` school/hometown/location, `epsilon_attr` = `epsilon_edge`. Dataset-independence held: no SNAP attribute name in `src/graph_aim` (`tests/graph_aim/test_generality.py`).
Headline v1.1 (3-seed means): at eps=4 the gap closed vs the permuted null (eta2 / deg-by-value TVD) is school .68/.56, birthday .54/.53, hometown .69/.34, location .99/.47 (all over the 0.3 bar; locale and employer fail deg_tvd -- null floors about .04); mean degree by value recovered (birthday value 5: real 99, v1 44, v1.1 84 at inf); hometown lift 1.00 -> 1.75, location 1.07 -> 1.61 (real 2.81 / 2.24); homophily_error .257 -> .233; other structure within the v1 seed spread. Known cost: eps=inf assortativity error .155 -> .176-.185 (hub concentration in the conditioned classes). Details, decisions D16-D18 and the open-items list: `docs/snap/PROCESS.md` "Attribute coupling round (v1.1)"; report section: `snap/reports/graph_aim_v1.md` "v1.1". Still open from before: DP selection of attribute domains and caps closes the biggest privacy gap.

## 3. Geo / GeoLife (user-level DP map-anchored mobility artifact)

**Status: M v2 and the zone restructure are merged to `main` locally (`7d7f59a`). No agents running. Nothing pushed.** GeoLife now follows the same inside / transfer / outside / eval layout as `snap/`.

Layout (see `geolife/README.md`, `src/geolife/README.md`, `docs/geolife/PROCESS.md` D1-D15):
- `src/geolife/inside/`: `data` (raw loader), `dp`, `learn`, `learn_v2`, `measure(_v2)`, `eda_outings`, `loaders`, `build_outings`.
- `src/geolife/transfer/`: `types` (contract), `artifact`, `inspect`, `artifact_cli`, `schema(_v2)`, `maps` (MapProvider protocol), `geo`, `tokenize/*` (stays, trips, outings), `map/{graph,match}` (shared code lives with the artifacts, per the user's decision).
- `src/geolife/outside/`: generator (`population*`, `render*`, `teleport`, `pipeline*`, `fakedata_v2`, `grid_city`), `fit(_v2)` (artifact tables -> params), `map/{provider,overture}`, `providers`, `synthetic_io`.
- `src/geolife/eval/`: method M (`m/*`: v1 home/work, v2 outings, transfer harnesses), `score`, `plots`, `match_tune`, `fakedata` (v1).
- Stage scripts: (1) inside: `geolife/inside/fit_artifacts.py` real traces -> `geolife/transfer/artifacts/`; (2) outside: `geolife/outside/generate_synthetic.py` artifacts only -> `geolife/outside/synthetic/`; (3) eval: `geolife/eval/score.py`. Public-map fetch (duckdb/network) is a separate script, `geolife/outside/fetch_public_map.py`; the provider reads its parquet cache.
- Tests: `tests/geolife/{inside,transfer,outside,eval}/` plus `tests/geolife/test_boundary.py` (183 passed, 1 skipped = Amsterdam cache test). Run `PYTHONPATH=src:. C:/Users/NaomiD/geo-venv/Scripts/python.exe -m pytest -q tests/geolife`. The light env has no sklearn, so Adult tests fail at collection there.
- Docs/reports: `docs/geolife/` (PLAN, PLAN_OUTINGS, OUTINGS_PARAMS, TASKS, p0-feasibility, PROCESS), reports under `geolife/{inside,outside,eval}/reports/`, notebooks `geolife/inside/notebooks/`.

Decided: the third-party allowlist in transfer/outside (h3, networkx, shapely, pyarrow; duckdb only in `fetch_public_map.py`) is confirmed and written into `CLAUDE.md` rule 2. Dependencies added to `pyproject.toml`/`uv.lock`. Place-snap radius raised 75 -> 150 m (public setting, ledger caveat).
- `inside.learn_v2` imports `outside.fit_v2` (post-processing of already-noisy tables; inside->outside is allowed).

Decisions (user): user-level DP, epsilon about 6, delta 1e-6. Data is mostly outings: no home/work, unit = outing. Relation = co-visit; co-presence out of scope. Time of day / day of week in scope but coarse (DP tables use am/pm x weekday/weekend). Parametric first; embedding tier is future work. No second dataset: transfer is checked by comparing real vs synthetic distributions plus split-half reliability on real data. Noise-only DP tables dropped and budget re-spent (data-dependent design choice listed in ledger caveats). Shared code -> transfer, method M -> eval.

Public parameters (informed by the real data, so ledger caveats): 2 h outing gap; real outing >= 5 min and >= 500 m; users with >= 10 real outings on >= 5 local days (139 of 182); stops 200 m / 10 min plus outing start/end; local time UTC+8 (GeoLife timestamps are GMT); place and traversal hierarchy H3 6->7->8; caps 100 outings, 50 res-8 stop cells, 1000 res-8 traversed cells per user.

Findings:
- Real data: 182 users, 18.7k real outings, 1.38 stops per outing, revisit rate 59% at res 8, starts peak 08h and 17-18h local, stops skew weekend. Split-half reliability on real data: co-visit edges Jaccard 0.38 (11x chance), places of interest 0.43 random / 0.30 time halves.
- Learner v2 at epsilon 6: readable = outing shape tables, per-user histograms, coarse place and traversal tables (res 6/7), coarse outing-start (mass above 3 sigma 0.89). Noise: place tier at res 8 (3 hub cells), `cat_time_visits`; category mix/profile fall back to defaults; fine-bin start profile nearly flat (within-bin split is an assumption).
- Generator v2 matches its params on toy city and Amsterdam but NOT real: real outings per user mean 133 vs 51 synthetic, active-days p90 167 vs 46, outing length 30 km vs 12 km, co-visit degree 20 vs 4.4 (real has hub places: 6 places with 51-67 users). Revisit rate matches (0.43 vs 0.42). On learned noisy params the planted-stats check fails on `stop_revisit_rate`, `users_per_place_mean`, `share_places_ge2_users` (fitted values vary a lot between noise draws).
- M v2 (synthetic, planted truth; P/R): places of interest 0.88/0.84 dev, 0.74/0.72 held-out; common places k=2 0.87/0.72, 0.80/0.57; segments k=2 0.96/0.97, 0.96/0.95; edge AUC 0.96/0.87; ARI 0.84 dev. Held-out relation edges collapse to 2 (FDR 0.2 too strict there; ranking still fine). Amsterdam untuned: places 0.72/0.71, edge precision 0.60, ARI 0.40. Real GeoLife used the cell proxy only (no road graph).
- M v1 (home/work) does not transfer to real (superseded).
- Tokenization: about 30% of stops snap to an Overture place at 75 m (48% at 150 m); rest fall back to H3 cells. Map matching exists, untested on real data. Overture: S3 reachable; `categories` replaced by `taxonomy`/`basic_category`; no work category.

Not in git (copy by hand when moving env): `geolife/inside/data/` (raw zip and `raw/` currently still under the main checkout's `geolife/data/`; `points.parquet` cache in worktree `agent-a04900ec477b059b7`), public-map caches `geolife/outside/cache/` (currently `geolife/cache/` in worktree `agent-a63109956b0e67a7c`; legacy absolute Amsterdam paths remain as fallbacks), older results/artifacts v0/v1/v2 in the agent worktrees. The current artifacts (`outings_v2_seed0-2`, epsilon 6, 150 m snap radius) are committed in `geolife/transfer/artifacts/`. Rebuild with `fetch_public_map.py`, `inside/build_outings.py`, `inside/learn_v2.py`. Env: `C:/Users/NaomiD/geo-venv` (light, rebuildable).

Next (fix list, full table in `docs/geolife/PROCESS.md` "Next steps"): (1) retire v1 code (entangled with v2: extract shared helpers first); (2) re-run M v2 and its report at the new 150 m snap radius; (3) relation significance too strict in sparse regimes; (4) learned params fail the planted-stats check (noise robustness); (5) generator vs real regime (heavy tails, longer outings, hub places, users-per-place shape); (6) map matching untested on real GPS; (7) outings export lacks `cells`; (8) privacy attacks, ledger review, signing; (9) fine-bin time profile is an assumption; (10) verify `uv sync` / `uv run pytest` with the new dependencies.
Overture reachability from the air-gap is still a risk.

## 4. Adult synthesis demo (Synthcity showcase, 12 plugins)

**Status: framework and docs done; benchmark not run. No agent was active.**

- On `main`: `src/synth_bench/` (config, data, datasets, methods, generators, evaluate, `run_benchmark` CLI, tests), EDA notebook `adult-census/notebooks/01_eda.ipynb`, 12 method reports in `docs/methods/`, log in `docs/adult-census/PROCESS.md`.
- Env: project `.venv` with D1 (`opacus>=1.4,<1.5.3`) and D2 (local monai stub `stubs/monai-stub`). Windows env re-resolved in `8862c26`. Check `uv sync` works on the new machine.
- Scope: GReaT and GOGGLE excluded. Budget about 10k train rows, reduced epochs, single seed.

Remaining (PROCESS.md phase board):
1. Phase 2: run the benchmark -> `adult-census/results/`.
2. Check the method reports' "Adult usage" sections are filled in (not verified).
3. Phase 3: results notebook + fill report Results sections.
4. Phase 4: review, re-execute notebooks, close out.

D6 eval checks the benchmark must report: constraint violations, target-rate preservation, rare-category coverage, association-matrix difference, TSTR/TRTR AUC, DCR with holdout baseline.

## 5. Cross-cutting rules (org policy, apply on the new env)

- Dependencies only via `dv2cloud.jfrog.io`; Python via `uv`, no `pip`.
- No force-push; check remote state before rebasing. Branch names `<prefix>/<JIRA-ID>-<desc>`; Conventional Commits.
- Merging a GitLab MR is a human action.
- Repo hygiene flags still open: no `catalog-info.yaml`, no `CLAUDE.md`.
- gitleaks pre-commit; `prek install` if `.pre-commit-config.yaml` exists.

## 6. Verify state on the new env

```bash
git worktree prune && git worktree list
git branch -vv --sort=-committerdate | head
for b in feat/snap-facebook-eda refactor/inside-transfer-outside docs/inside-outside-readmes feat/geolife-impl; do echo $b $(git rev-list --count main..$b); done   # both 0 now
git log --all --since="2 hours ago" --format='%h %ad %d %s' --date=format:'%H:%M'
# dirty worktrees = in-flight agent work that would be lost if not copied
git worktree list | awk '{print $1}' | while read w; do n=$(git -C "$w" status --short | wc -l); [ "$n" != 0 ] && echo "dirty $n $w"; done
```

Before moving: wait for or commit in-flight agents (none for Geo now; any SNAP diagnostic worktree with uncommitted `snap/eval/diagnose_*.py`). Uncommitted worktree files only exist on disk.

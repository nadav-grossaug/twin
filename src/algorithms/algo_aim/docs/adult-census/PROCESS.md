# Process log

Orchestrated build of the Synthcity showcase on Adult Census. The orchestrator (main session) owns this file; agents work in git worktrees on their own branches and report back, and the orchestrator merges into `main`.

Plan: 12 Synthcity plugins (`marginal_distributions`, `bayesian_network`, `ctgan`, `tvae`, `nflow`, `arf`, `ddpm`, `adsgan`, `privbayes`, `dpgan`, `pategan`, `aim`), quick budget (~10k train rows, reduced epochs, single seed).

## Phase board

| Phase | Task | Owner / branch | Status |
|---|---|---|---|
| 0 | Env fix, git init, scaffolding | orchestrator / `main` | done |
| 1 | EDA notebook `adult-census/notebooks/01_eda.ipynb` | agent A / `worktree-agent-a47771135d3b332fd` | done, merged |
| 1 | Framework `src/synth_bench/` | agent B | done, merged (config, data, datasets, methods, generators, evaluate, `run_benchmark` CLI, tests) |
| 2 | Benchmark run → `adult-census/results/` | agent C | pending, not started (no `adult-census/results/`) |
| 2 | Method reports: marginal_distributions, bayesian_network, arf, nflow | agent D1 / `worktree-agent-a7c8dd48e7d141fa4` | theory merged; Adult usage part not written yet (B is merged, so unblocked) |
| 2 | Method reports: ctgan, tvae, ddpm, adsgan | agent D2 / `d2-method-reports-gan-vae-ddpm` | theory merged; Adult usage part not written yet (B is merged, so unblocked) |
| 2 | Method reports: privbayes, dpgan, pategan, aim | agent D3 / `worktree-agent-a93b8b37f274fea62` | theory merged; Adult usage part not written yet (B is merged, so unblocked) |
| 3 | Results notebook + fill report Results sections | agent E | pending |
| 4 | Review, re-execute notebooks, close out | orchestrator | pending |

## Decisions

- **D1 (env): pin `opacus>=1.4,<1.5.3`.** opacus 1.5.3+ registers a grad sampler for `nn.RMSNorm`, which doesn't exist in torch 2.2 (synthcity pins `torch<2.3`). Importing synthcity crashed.
- **D2 (env): local `monai` stub (`stubs/monai-stub`).** The JFrog proxy blocks monai. Synthcity imports it eagerly (`metrics/eval_detection.py` → `plugins/core/models/convnet.py`), but only image plugins use it. The stub provides the four imported names and raises if they are instantiated.
- **D3 (scope): GReaT and GOGGLE are excluded.** They fail to load (`dgl` is missing, and GReaT is LLM-based and too slow on CPU). This matches the user's "core + DP" choice.
- **D4 (structure): multi-dataset layout.** Shared code, method theory, and process live at the root. Each dataset owns `data/`, `notebooks/`, `reports/`, and `results/`.
- **D5 (git): worktree per agent.** Branches are merged `--no-ff` by the orchestrator. Worktrees share `.venv` from the main checkout (`PYTHONPATH=<worktree>/src`).

- **D6 (eval): EDA-driven checks.** The EDA found informative "?" missingness, zero-inflated and capped numerics, a 1:1 education↔education.num mapping, near-deterministic relationship↔sex/marital rules, rare categories (15 of the 42 countries have fewer than 30 rows), and 24 real duplicates. Agent B was asked to add: logical-constraint violation rates (generic `DatasetSpec.constraints` hook), target-rate preservation, rare-category coverage, association-matrix differences, TSTR/TRTR ROC-AUC, and DCR with a holdout baseline. `education.num` is kept and its consistency measured through a constraint.

## Issues

| # | Issue | Resolution |
|---|---|---|
| 1 | `import synthcity.plugins` → `AttributeError: torch.nn has no RMSNorm` | D1 |
| 2 | `ModuleNotFoundError: monai` | D2 |

## Log

- 2026-10-30: Phase 0 done. Env fixed; all 12 target plugins and the metric suite import.
- 2026-10-30: Phase 1 launched (A, B). D1–D3 started early on the dataset-agnostic `docs/methods/*.md` (no dependency on B); their Adult usage reports follow once B is merged.
- 2026-10-30: A merged (53-cell EDA, 0 errors). EDA implications forwarded to B.
- 2026-11-01: Session resumed. Agents B and D1 had been interrupted with uncommitted work (worktrees' gitfiles point to WSL paths; the Windows checkout is only a mirror, work happens in WSL). Committed and merged D1 theory and B's partial framework; relaunched B (evaluate + CLI) and D3 (theory). Smoke test: fit/generate with `marginal_distributions` works.
- 2026-11-01: **Moved off WSL to native Windows.** `pyproject.toml` now resolves for Linux and win32 (`uv.lock` regenerated, adds pytest); `uv sync --group dev` works on Windows and the 15 tests pass. WSL `main` and D3 (privbayes/dpgan/pategan/aim reports) were brought over via git bundle and merged. The Windows checkout had other work (SNAP/graph branches), so `main` was fast-forwarded without touching the checked-out branch. New agents work in Windows worktrees under `.claude/worktrees/`.
- D3 findings to respect in the benchmark: privbayes noise is ~n times too small and numerics are never binned; dpgan defaults give noise multiplier ~22 and the DP critic step looks buggy; pategan leaks real rows to the student and overshoots epsilon; aim runs its mechanism inside `generate()` and mishandles numeric columns with >=15 levels (pre-bin to small integer codes). Details in `docs/methods/`.

## Where we stopped (2026-11-01) - resume here

Adult was paused (low priority) while the SNAP and GeoLife tracks were built; both are merged into `main`. No agent is running on Adult and nothing is half-finished.

State:
- Done and on `main`: env fixes (D1, D2), EDA notebook, `src/synth_bench/` framework with `evaluate` and the `run_benchmark` CLI, all 12 theory reports in `docs/methods/`.
- Not done: Phase 2 benchmark run, the "Adult usage" and "Results" sections of the method reports, `02_synthcity_results.ipynb`, Phase 4 review.
- Not checked since the move to Windows: the `synth_bench` tests (`tests/test_evaluate.py`, `tests/test_run_benchmark.py`) need the project env (sklearn, synthcity); the light Geo venv cannot run them. Last known: 15 tests passed after the Windows move.
- Data: `adult-census/data/adult.csv` is no longer tracked (gitignored). Copy it into place on a new machine.

Resume steps:
1. `uv sync --group dev`, then `uv run pytest tests/test_evaluate.py tests/test_run_benchmark.py` to confirm the framework still works.
2. Smoke test: `uv run python -m synth_bench.run_benchmark --dataset adult-census --methods marginal_distributions,bayesian_network,arf --n-train 2000 --results-dir /tmp/adult_smoke`.
3. Full run (Phase 2): `--methods all`, 10k rows, single seed, `--resume`; output to `adult-census/results/`. Respect the D3 findings below (privbayes, dpgan, pategan, aim need care) and the D6 evaluation checks.
4. Write the Adult usage and Results sections of each method report, then the results notebook (Phase 3), then review and re-execute notebooks (Phase 4).

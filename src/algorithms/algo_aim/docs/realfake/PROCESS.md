# Process log: real-vs-fake classifier track

## Phases and decisions (started 2026-11-02)

- **D1 (scope): shared core plus per-dataset implementation.** `src/realfake/` holds the dataset-independent parts (split, models, metrics, controls, importance, report); each dataset supplies a sampler and a featurizer. SNAP first. Eval-only: lives in the inside-only eval zone, imported by nothing in inside, transfer or outside.
- **D2 (sample): random-walk subgraph of random size**, label = real or synthetic source graph; ids, ordering and column order dropped. See `docs/realfake/snap.md`.
- **D3 (honesty): every result is reported with controls**: real-vs-real, permuted labels, held-out fake seed, bootstrap CI, baselines for scale. See `docs/realfake/PLAN.md` section 4.
- **D4 (implementation, 2026-11-02): all five plan steps done on branch `realfake`.** Core `src/realfake/` (dataset, split, models, metrics, controls, importance, report; numpy logreg + sklearn gradient boosting with uniform log-odds `scores`), SNAP `src/graph_aim/eval/discriminator.py` (walk sampler + featurizer), stage-3 CLI `snap/eval/discriminate.py`, tests `tests/realfake/` + `tests/graph_aim/eval/test_discriminator.py` + `test_discriminate.py`, boundary test extended to forbid `realfake` outside eval. Built by three subagents (core, SNAP featurizer, CLI+run), each integrated and re-tested by the orchestrator; full suite green.
- **D5 (fold design): one dataset per method, folds rotate the held-out fake seed.** A method = a variant with its 3 generation seeds as fake sources; fold `s` holds seed `s` out for test, real rows are group-split by start node. A single-fake-source method falls back to a stratified group split.
- **D6 (first SNAP result): every graph_aim variant is near-perfectly detectable at walk level; the epsilon gradient lives in the attribute importance.** Controls clean (real-vs-real 0.515/0.528 logreg/gbm, permuted ~0.5, ER and Chung-Lu at 1.0000). Gradient boosting reaches AUC 0.9994-0.9999 for every variant, eps 0.5 through inf and nocap_nonprivate alike; logreg alone 0.993-0.997. Permutation importance: the attributes group dominates everywhere and falls monotonically with privacy budget (auc_drop 0.43 at eps 0.5 -> 0.24 at inf); degree second (0.001-0.043), structure and size negligible. AUC >= 0.995 in every walk-size bucket. This is a fidelity finding the scalar metrics and the 0.99-everywhere link-prediction utility (D12 in `docs/snap/PROCESS.md`) hid; it is not a privacy statement. Results: `snap/eval/results/discriminator*.csv`, notebook `snap/eval/notebooks/viz_discriminator.ipynb`, report `snap/reports/discriminator_v1.md`.

Status: implemented and run on SNAP (branch `realfake`, not merged). Next: issue 1 below, then Adult / GeoLife samplers.

## Issues

| # | Issue | Status |
|---|---|---|
| 1 | The attributes group mixes genuine statistics (marginal shares, same-value edge rates, pair co-occurrence) with a candidate generation artifact (exact `none`/`other` code shares, the open point in `snap.md` section 2). "Attributes dominate" must not yet be read as "homophily is the cause". | open: split the group per feature and re-run importance |
| 2 | The worktrees given to the first two subagents branched from a pre-baseline HEAD, so the project files were missing there; both agents copied the tree in and committed only their new files. Integration was done by copying files back and re-testing in the main checkout. | worked around |
| 3 | An external checkpointing process was seen committing and then resetting staged changes in the main checkout; every commit on `realfake` was verified on HEAD after committing. | worked around |

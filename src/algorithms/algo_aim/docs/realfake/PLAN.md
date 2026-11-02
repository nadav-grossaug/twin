# Real-vs-fake classifier: plan

Status: planned (2026-11-02), nothing implemented. SNAP is the first dataset; SNAP-specific design is in `docs/realfake/snap.md`, the log in `docs/realfake/PROCESS.md`.

## 1. Why

The current SNAP evaluation compares aggregate metrics (degree, clustering, homophily) and a link-prediction utility task that does not discriminate (0.99 everywhere, `docs/snap/PROCESS.md` D12). A real-vs-fake classifier (a classifier two-sample test) is a stricter check that is the same idea for every data type: train a model to tell samples cut from the real data from samples cut from the synthetic data. If it cannot (ROC AUC about 0.5), the synthetic data is faithful at whatever level that model can see. If it can, the feature importance shows what gave it away.

It is eval-only: it needs real and synthetic data side by side. It lives in the inside-only eval zone and changes no artifact, generator or privacy accounting. A failed discriminator is evidence of fidelity, not of privacy, and a weak classifier proves little, so every result is reported with controls (section 4).

## 2. Architecture: shared core plus one implementation per data type

New package `src/realfake/`, dataset-independent. It is eval-zone code: `graph_aim.eval` (and later `geolife` / `synth_bench` eval code) may import it; nothing in inside, transfer or outside may. `tests/graph_aim/test_boundary.py` `test_nothing_outside_eval_imports_eval` is extended to cover `realfake`.

```
src/realfake/
  types.py       Sample (payload, label, group_id, meta); Sampler and Featurizer protocols
  dataset.py     build_dataset(real_source, fake_sources, sampler, featurizer, n, seed) -> X, y, groups, feature_names
  split.py       group-aware train/test split (no group on both sides); held-out fake seeds
  models.py      logistic regression (numpy/scipy L-BFGS, same style as graph_aim.eval.utility) and sklearn gradient boosting if importable
  metrics.py     ROC AUC, accuracy, bootstrap CI, AUC by meta bucket (for example sample size)
  controls.py    real-vs-real control, label-permutation control
  importance.py  permutation importance per feature group
  report.py      results -> csv / json in the layout of snap/eval/results
```

Contract a data type implements:
- `Sampler(source, rng) -> Sample`: cut one unit (SNAP: random-walk subgraph) from a real or synthetic source. Same parameters for both labels, so unit size and selection bias match by construction.
- `Featurizer(sample) -> vector` plus `feature_names` and `feature_groups`: must be invariant to representation details that differ between real and synthetic by construction (ids, ordering, column order). Anything the method intentionally models stays in.

Per-dataset code lives next to the dataset's other eval code: SNAP is `src/graph_aim/eval/discriminator.py` and `snap/eval/discriminate.py`. Later datasets (Adult, GeoLife) add only a sampler and a featurizer.

## 3. Steps

1. `realfake/types.py`, `split.py`, `metrics.py`, `controls.py` with tests on toy data (one separable, one not).
2. `realfake/models.py` (numpy logistic regression; sklearn gradient boosting guarded by import), `importance.py`, `report.py`, `dataset.py`.
3. `graph_aim/eval/discriminator.py`: walk sampler, relabelling, featurizer (see `snap.md`).
4. `snap/eval/discriminate.py` (stage 3, CLI like `score.py`: `--config --synthetic --out --seeds --only`); run over `snap/outside/synthetic/index.json` and the baselines in `eval/baselines.py` (`erdos_renyi`, `chung_lu`, `dc_sbm`, `bter`).
5. Notebook `snap/eval/notebooks/viz_discriminator.ipynb`, summary in `snap/reports/`, decision in `docs/snap/PROCESS.md` (next number after D14), `CHECKPOINTS.md` update.

## 4. Controls (apply to every dataset)

- Group-aware split: samples that share a start unit or overlap are never on both sides of the split. For SNAP the real graph's nodes are split into two halves and walks are assigned by start node.
- Real-vs-real control: classifier trained on real half A against real half B. Must give AUC about 0.5; otherwise the pipeline leaks and no other number can be trusted. This is the reference for "indistinguishable".
- Held-out fake seed: train on fake seeds 0 and 1, test on seed 2.
- Label-permutation control: shuffled labels give AUC about 0.5.
- Bootstrap confidence interval on AUC; AUC by sample-size bucket.
- Scale: baselines (Erdos-Renyi and the like) give the "easy to detect" end, real-vs-real the "cannot detect" end.
- Report the strongest model, not the weakest.

## 5. Tests (offline)

| Test | Checks |
|---|---|
| `tests/realfake/test_split.py` | no group on both sides; fake seed held out |
| `tests/realfake/test_metrics.py` | AUC against a reference implementation on known scores; bootstrap CI covers the true value on toy data |
| `tests/realfake/test_controls.py` | permuted-label and real-vs-real controls give AUC within CI of 0.5; a separable toy gives AUC above 0.9 |
| `tests/realfake/test_models.py` | numpy logistic regression separates a linear toy; sklearn path skipped when sklearn is missing |
| `tests/graph_aim/eval/test_discriminator.py` | sampler determinism, size range, k distinct nodes; features invariant to node relabelling and column order; Erdos-Renyi vs a clustered graph is detected |
| `tests/graph_aim/test_boundary.py` | nothing outside eval imports `realfake` or `graph_aim.eval` |

Run: `PYTHONPATH=src:snap/inside:snap/eval python -m pytest tests/realfake tests/graph_aim -q`.
End-to-end: `uv run python snap/eval/discriminate.py --only graph_aim_eps4`; check real-vs-real AUC about 0.5, Erdos-Renyi AUC high, result files written.

## 6. Risks

| Risk | Handling |
|---|---|
| Overlapping samples inflate or deflate AUC | group split plus the real-vs-real control |
| sklearn missing in some environments (WSL snapcheck, light Geo env) | numpy logistic regression is the fallback; gradient boosting optional |
| The real graph is one graph, so few independent samples | many walks, bootstrap, state the limit in the report |
| A giveaway feature we did not anticipate | permutation importance per feature group; log each finding in PROCESS.md |
| A weak classifier gives a flattering result | strongest model reported; baselines show it can detect bad graphs |

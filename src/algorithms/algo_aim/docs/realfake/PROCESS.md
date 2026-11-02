# Process log: real-vs-fake classifier track

## Phases and decisions (started 2026-11-02)

- **D1 (scope): shared core plus per-dataset implementation.** `src/realfake/` holds the dataset-independent parts (split, models, metrics, controls, importance, report); each dataset supplies a sampler and a featurizer. SNAP first. Eval-only: lives in the inside-only eval zone, imported by nothing in inside, transfer or outside.
- **D2 (sample): random-walk subgraph of random size**, label = real or synthetic source graph; ids, ordering and column order dropped. See `docs/realfake/snap.md`.
- **D3 (honesty): every result is reported with controls**: real-vs-real, permuted labels, held-out fake seed, bootstrap CI, baselines for scale. See `docs/realfake/PLAN.md` section 4.

Status: plan only, no code yet. Next: step 1 of `PLAN.md`.

## Issues

| # | Issue | Status |
|---|---|---|

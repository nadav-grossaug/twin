# OUTSIDE: generation without the real graph

```bash
PYTHONPATH=src python snap/outside/generate_synthetic.py [--only graph_aim_eps4] [--artifacts DIR] [--out DIR]
```

Reads artifact folders from `../transfer/artifacts/` (hash-verified; folders that are not present are skipped) and writes
`synthetic/<variant>_seed<k>.npz`. Needs only Python >= 3.11 with numpy, pandas, scipy, plus `src/graph_aim/transfer/` and
`src/graph_aim/outside/`. It imports nothing from `inside`; `tests/graph_aim/test_boundary.py` enforces that and runs this
script with only `src` importable.

One artifact from Python, and how to read the result: see "Generate synthetic data outside" in `../README.md`.
`synthetic/` holds the generated graphs (only the eps = 4 and non-private no-cap graphs of seed 0 are committed).
Generator code: `src/graph_aim/outside/generate.py`.

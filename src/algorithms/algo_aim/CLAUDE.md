# CLAUDE.md

Synthetic-data experiments (Synthcity benchmark on Adult Census, DP graph synthesis on SNAP ego-Facebook, DP mobility artifact on GeoLife). Log and decisions per dataset: `docs/<dataset>/PROCESS.md`. Environment and org rules: the managed policy; Python through `uv`.

## Project structure: simulate the air gap

The real use case is an air-gapped network that holds the real data. Only a small, inspectable, noise-protected **artifact** may leave it. A generator outside builds synthetic data from the artifact alone. The code is laid out so you can see which side every file is on. The reference implementation is `snap/` (see `snap/README.md`, `docs/snap/PROCESS.md` D13); new datasets and methods follow the same layout.

```
INSIDE (real data lives here)     TRANSFER (the only thing that moves)     OUTSIDE (never sees real data)
<dataset>/inside/                 <dataset>/transfer/                      <dataset>/outside/
src/<pkg>/inside/                 src/<pkg>/transfer/                      src/<pkg>/outside/
```

| Zone | May read real data? | What lives here |
|---|---|---|
| **inside** | yes | raw and processed data, preparation, fitting (measure, noise, privacy accounting), EDA notebooks of the real data |
| **transfer** | no | the artifacts (`transfer/artifacts/<variant>_seed<k>/`) and the code that defines their format: shared types, save / load / validate / inspect. Readable by both sides |
| **outside** | no, enforced | the generator, the synthetic data it writes (`outside/synthetic/`), and the reports about the method |
| **eval** | yes, it needs both | code that compares synthetic data with real data: metrics, baselines, empirical attacks, comparison plots and notebooks (`viz_comparison`), result tables. Inside-only zone: may import every zone, none of them may import it |

Rules:

1. **Transfer is the only folder both sides access.** Inside writes artifacts to it; outside reads them from it. Nothing else crosses.
2. **Outside and transfer code may import only** the standard library, numpy/pandas/scipy (whatever the method needs), and `transfer` / `outside` modules. GeoLife additionally uses the pure-computation libraries h3, networkx, shapely and pyarrow (allowlisted in `tests/geolife/test_boundary.py`); libraries that fetch from the network (duckdb) stay in the separate script `geolife/outside/fetch_public_map.py`. They must never import inside code or eval code (fitting, preparation, evaluation). Run outside stages with only `src` on the path.
3. **The generator must work from a copied `transfer/artifacts/` alone.** The artifact holds noisy tables and metadata, never records, edges or raw coordinates. Artifact files carry a SHA-256 in `manifest.json`; provenance keys are allowlisted so an un-noised statistic cannot be released by accident.
4. **Three stages**, each a script: (1) inside: real data -> artifacts; (2) outside: artifacts -> synthetic data; (3) eval: synthetic + real -> metrics. Synthetic data goes back inside only for scoring.
5. **Public settings chosen by looking at the real data** (attribute sets, caps, bins) are outside the DP guarantee. List them as `caveats` in the artifact's `privacy.json`.
6. The boundary is checked in tests (`tests/graph_aim/test_boundary.py`: parses imports, runs stage 2 in a subprocess). It is not an OS sandbox.

### The eval folder

Add `eval/` next to `inside/`, `transfer/` and `outside/` (`<dataset>/eval/` for scripts and results, `src/<pkg>/eval/` for library code). It holds everything that needs the real data **and** the synthetic data side by side: fidelity and utility metrics, baselines used as reference points, privacy attacks on the synthetic data, comparison notebooks and result tables. It is an inside-only zone: it may import inside, transfer and outside code, and nothing in transfer or outside may import it.

In `snap/` this is done: `snap/eval/` (`score.py`, `notebooks/viz_comparison.ipynb`, `results/`, `diagnose_*.py`) and `src/graph_aim/eval/` (`evaluate`, `viz`, `baselines`, `attacks`, `utility`). `tests/graph_aim/test_boundary.py` enforces that nothing outside eval imports it. New datasets follow the same layout; do not move code without checking the boundary test still passes.

## Layout of shared code and docs

| Path | What |
|---|---|
| `src/synth_bench/` | Synthcity benchmark framework for tabular data (Adult). Not yet split into zones |
| `src/graph_aim/` | DP synthetic attributed graphs (SNAP), split into zones |
| `src/geolife/` | DP mobility artifact (GeoLife), split into zones. Contract in `src/geolife/transfer/types.py` |
| `docs/<dataset>/PROCESS.md` | per-dataset phases, decisions, issues, resume notes |
| `docs/methods/` | method theory, independent of dataset |
| `tools/` | throwaway diagnostics, not part of any pipeline |
| `CHECKPOINTS.md` | state of every project, for resuming after moving environments |

## Working rules

- Datasets (`*/data/`) are gitignored. Do not commit them. Copy by hand when moving machines.
- Tests: `uv run pytest` (pythonpath already includes `src` and `snap/inside`). Keep tests offline.
- Keep saved artifacts byte-exact (`.gitattributes`: no EOL conversion).
- Update `docs/<dataset>/PROCESS.md` when a decision is made or a stage changes.

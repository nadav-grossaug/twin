# hackaton-2026: synthetic data experiments

Three datasets, each with a way to synthesize data and check how good it is:

| Dataset | What | Method | Start here |
|---|---|---|---|
| **Adult Census** (`adult-census/`) | tabular | 12 [Synthcity](https://github.com/vanderschaarlab/synthcity) generators, benchmarked | `docs/adult-census/PROCESS.md` |
| **SNAP ego-Facebook** (`snap/`) | attributed graph | differentially private "graph-AIM" | `snap/README.md` |
| **GeoLife** (`geolife/`) | GPS traces | differentially private mobility artifact | `geolife/README.md` |

## The idea (SNAP and GeoLife)

The real data is assumed to live in an air-gapped network. Only a small, readable, noise-protected **artifact** may leave it;
synthetic data is generated from the artifact alone. The folders mirror this:

```
<dataset>/inside/     real data, preparation, fitting                       (STAGE 1: real -> artifact)
<dataset>/transfer/   the artifacts: the only thing that moves
<dataset>/outside/    generator and synthetic data, never sees real data    (STAGE 2: artifact -> synthetic)
<dataset>/eval/       compares synthetic with real                          (STAGE 3: synthetic + real -> metrics)
```

Library code has the same split under `src/graph_aim/` and `src/geolife/`. Tabular Adult (`src/synth_bench/`) is not split yet.

## Setup

```bash
uv sync --group dev
```

Datasets are not in git (`*/data/` is gitignored). Put the raw files in place first; see the `data/README.md` next to each dataset or the dataset README.

## Run

```bash
# SNAP: the three stages, from the repo root (options in snap/README.md)
PYTHONPATH=src:snap/inside python snap/inside/fit_artifacts.py --seeds 3     # real -> snap/transfer/artifacts/
PYTHONPATH=src            python snap/outside/generate_synthetic.py          # artifacts -> snap/outside/synthetic/
PYTHONPATH=src            python snap/eval/score.py --seeds 3                # compare -> snap/eval/results/

# GeoLife: same three stages with geolife/inside/fit_artifacts.py, geolife/outside/generate_synthetic.py,
# geolife/eval/score.py (extra arguments in geolife/README.md)

# Adult: benchmark the Synthcity generators
uv run python -m synth_bench.run_benchmark --dataset adult-census --methods all
```

On Windows, separate `PYTHONPATH` entries with `;` instead of `:`.

## Check

```bash
uv run pytest
```

`tests/graph_aim/test_boundary.py` checks that the outside code cannot import anything from inside.

## Where to read more

- `CLAUDE.md`: the zone rules (what may import what), in detail
- `docs/<dataset>/PROCESS.md`: phases, decisions, open issues. `CHECKPOINTS.md`: current state of every project
- `docs/methods/`: how each Synthcity method works. `research/`: background notes

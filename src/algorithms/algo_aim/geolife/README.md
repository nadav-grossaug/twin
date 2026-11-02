# GeoLife: inside / transfer / outside / eval

Fit on the real GPS traces **inside**, generate synthetic traces **outside**, move only readable, noise-protected
artifacts between the two, and compare real with synthetic in **eval**. The folders show which side a file is on.
Same layout as `snap/` (see `snap/README.md`, `docs/snap/PROCESS.md` D13).

```
 INSIDE  (may read real traces)            TRANSFER (what moves)          OUTSIDE (never sees real data)
 ──────────────────────────────            ─────────────────────          ──────────────────────────────
 data/            raw GeoLife, points      transfer/artifacts/            outside/generate_synthetic.py
 fit_artifacts.py (STAGE 1) outings,        <variant>_seed<k>/  ────────►  (STAGE 2) artifacts -> traces +
   caps, DP noise ───────────────────────►  manifest, schema, privacy,     planted truth in outside/synthetic/
                                            model/behaviour/*              outside/cache/ public map (Overture)
 EVAL  (needs real AND synthetic; inside-only zone)                                   │ synthetic traces
 ────────────────────────────────────────────────────                                 │ come back only
 eval/score.py (STAGE 3), method M, split-half, real-vs-synthetic  ◄──────────────────┘ to be scored
```

| Zone | Folders | May read real traces? | Contains |
|---|---|---|---|
| **inside** | `geolife/inside/`, `src/geolife/inside/` | yes | raw data, loaders, outing extraction, caps, DP measurement and accounting, EDA notebooks and reports |
| **transfer** | `geolife/transfer/`, `src/geolife/transfer/` | no | the artifacts, the contract types, the artifact format (save / load / validate / inspect), shared pure code: tokenization, map graph and matcher, geometry, `MapProvider` protocol |
| **outside** | `geolife/outside/`, `src/geolife/outside/` | **no, enforced** | the generator, stage-2 script, synthetic traces, public-map cache and fetcher, generator reports |
| **eval** | `geolife/eval/`, `src/geolife/eval/` | yes (needs both) | method M (the downstream task), its harnesses, split-half reliability, real-vs-synthetic comparison, results, reports |

## Generate synthetic data outside

You need: Python >= 3.11 with numpy, pandas, scipy, h3, networkx, shapely, pyarrow; this code; an artifact folder.

1. **Copy over** (and nothing else): `src/geolife/__init__.py`, `src/geolife/transfer/`, `src/geolife/outside/`,
   `geolife/outside/generate_synthetic.py`, and the artifact folders, e.g. `geolife/transfer/artifacts/outings_v2_seed0/`.
2. **Generate** (only `src` on the path, so nothing from `inside` is importable):
   ```bash
   PYTHONPATH=src python geolife/outside/generate_synthetic.py [--only outings_v2] [--n-users 200] \
       [--map grid | --map overture --bbox 116.25 39.85 116.45 40.05 --cache-dir geolife/outside/cache]
   ```
   Output: `geolife/outside/synthetic/<variant>_seed<k>/{traces.parquet,outings.json,truth.json,meta.json}` and `index.json`.
   Loading an artifact verifies its SHA-256 hashes. `--map overture` needs a map cache fetched beforehand (public data,
   needs network and duckdb, run anywhere): `PYTHONPATH=src python geolife/outside/fetch_public_map.py --name beijing 116.25 39.85 116.45 40.05 --all`.
3. **Or from Python** (no index file needed):
   ```python
   from geolife.outside.pipeline_v2 import generate_v2
   from geolife.outside.providers import make_provider
   g = generate_v2("geolife/transfer/artifacts/outings_v2_seed0", make_provider({"kind": "grid"}), n=100, seed=0)
   g.result.traces   # {uid: [GpsPoint(t, lat, lon)]}      g.result.outings, g.result.truth  planted outings and structure
   ```
4. **Look at an artifact** (budgets, noise, caveats): `PYTHONPATH=src python -m geolife.transfer.artifact_cli inspect <dir>`;
   check it: `... artifact_cli verify <dir>`; plots: `python -m geolife.eval.plots <dir> <png dir>`.

Other seeds give other traces from the same artifact; the artifact's privacy guarantee does not change.

## Run the whole pipeline (repo root)

```bash
PYTHONPATH=src python geolife/inside/fit_artifacts.py --places geolife/outside/cache/places_beijing_big.parquet --seeds 3   # STAGE 1 inside:  real -> geolife/transfer/artifacts/
PYTHONPATH=src python geolife/outside/generate_synthetic.py                                                               # STAGE 2 outside: artifacts -> geolife/outside/synthetic/
PYTHONPATH=src python geolife/eval/score.py --real-points geolife/inside/data/points.parquet --places geolife/outside/cache/places_beijing_big.parquet   # STAGE 3 eval -> geolife/eval/results/
# real-vs-synthetic panels + distance table (geolife/eval/results/viz/<variant>/): add --viz; for the direct lon/lat
# overlay first generate into the source city: geolife/outside/generate_synthetic.py --city beijing --n-users 139
```

Stage 1 reads `geolife/inside/data/` (env `GEOLIFE_DATA_DIR` overrides). Data, the map cache, results and non-seed-0
artifacts are gitignored: copy them by hand when moving machines (`geolife/inside/data/README.md`).

## Where things are

| What | Where |
|---|---|
| Plan, params, task board, process log | `docs/geolife/` (`PROCESS.md` has phases, decisions D1.., issues) |
| EDA notebooks, data access, learner reports | `geolife/inside/notebooks/`, `geolife/inside/reports/` |
| Generator and teleport reports | `geolife/outside/reports/` |
| Method M reports (`m-transfer.md`, `m-v2.md`) | `geolife/eval/reports/` |
| Code | `src/geolife/README.md` |

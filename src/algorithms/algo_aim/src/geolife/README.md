# geolife

Differentially private, map-anchored **mobility artifact** from GPS traces, and a generator that turns the artifact back
into synthetic traces on any public map. This is the dataset-specific code for GeoLife; the data, run scripts and
results live in `geolife/` (see `geolife/README.md`).

```
real traces ──► inside: tokenize, cap, measure, add noise ──► Artifact (noisy tables, on disk) ──► outside: fit + generate ──► synthetic traces
                                                                                                          │
                          eval: method M on real and synthetic traces, planted-truth scoring  ◄───────────┘
```

Four zones, same layout as `src/graph_aim/`. The split shows what may touch the real traces, what crosses, what runs
without ever seeing them, and what needs both.

| Zone | May read real traces? | Modules |
|---|---|---|
| **inside/** | yes | `data.py` raw PLT/label loader and pseudonyms; `loaders.py` points.parquet / labels readers; `dp.py` caps, zCDP accountant, Gaussian measure; `measure_v2.py` (outings tables), `measure.py` (v1 stay/trip tables); `learn_v2.py`, `learn.py` fit to artifact; `build_outings.py`; `eda_outings.py` EDA helpers |
| **transfer/** | no | `types.py` contract; `artifact.py` save / load (SHA-256) / `validate_no_raw_leakage` with the provenance allowlist; `inspect.py`, `artifact_cli.py`; `schema.py`, `schema_v2.py` measurement names, keys, bins; `geo.py` pure geometry and cleaning helpers; `maps.py` the `MapProvider` protocol; `tokenize/` stays, trips, outings; `map/` road graph and HMM matcher |
| **outside/** | **no, enforced** | `fit_v2.py`, `fit.py` artifact tables to generator params; `population_v2.py`, `render_v2.py`, `teleport.py`, `pipeline_v2.py` (and the v1 `population.py`, `render.py`, `pipeline.py`); `fakedata_v2.py` fake params; `grid_city.py` toy map; `map/provider.py` Overture provider (reads cached parquet), `map/overture.py`; `providers.py` map spec; `synthetic_io.py` stage-2 output format |
| **eval/** | yes (needs both) | `m/core.py`, `m/outings.py` method M; `m/transfer.py`, `m/outings_transfer.py` transfer harnesses, split-half reliability, real-vs-synthetic comparison; `score.py` stage 3; `plots.py`; `match_tune.py`; `fakedata.py` (v1 fake artifact, uses inside) |

## Rules (checked by `tests/geolife/test_boundary.py`)

- `outside` may import the standard library, numpy/pandas/scipy, the **pure-computation allowlist** `h3`, `networkx`,
  `shapely`, `pyarrow`, and `geolife.transfer` / `geolife.outside`. `transfer` may import the same, minus `outside`.
  `inside` may import `transfer` (and `outside.fit_v2`: post-processing of already-noisy tables). `eval` may import every zone.
- The allowlist is a decision to confirm: CLAUDE.md names numpy/pandas/scipy "whatever the method needs". h3 (cells),
  networkx + shapely (road graph, map matching) and pyarrow (parquet tables of the artifact) are pure computation; none
  opens a socket or a process. Network and duckdb live only in the script `geolife/outside/fetch_public_map.py`.
- **Nothing in `inside`, `transfer` or `outside` may import `eval`.** Nothing in `transfer` or `outside` may import `inside`.
- `transfer` and `outside` may not import `os`, `subprocess`, `socket`, `shutil`, `tempfile`, `urllib`, `http`, `ctypes`,
  `multiprocessing`, `pickle`, `importlib` and similar: paths and environment variables are handled by the stage scripts
  and by `inside`; shipped code gets paths as arguments.
- Nothing un-noised may leave the inside: artifacts hold only noisy tables, and `validate_no_raw_leakage()` rejects
  top-level keys of `preprocessing.json`, `diagnostics.json`, `grounding.json`, `generation.json` that are not on
  `PROVENANCE_ALLOWED`, coordinate-like values, real user ids and inconsistent privacy accounting.
- No module sits outside the four zones. Imports are absolute (`geolife.transfer.types`).

## What ships to an outside machine

`geolife/__init__.py`, `geolife/transfer/`, `geolife/outside/` (the package), `geolife/outside/generate_synthetic.py` and an
artifact folder under `geolife/transfer/artifacts/`. Python >= 3.11 with numpy, pandas, scipy, h3, networkx, shapely, pyarrow.
Nothing from `inside/` or `eval/`, no dataset files. Usage: "Generate synthetic data outside" in `geolife/README.md`.

## Contract in one paragraph

Learning returns an `Artifact`: a public `schema` and `preprocessing` record, a list of `Measurement`s (each a noisy,
clipped, user-level count table over public-vocabulary keys with its sigma, L2 sensitivity and rho), a `PrivacyLedger`
(zCDP, user-level adjacency, epsilon at delta, caps, and `caveats` listing what the guarantee does not cover), fitted
summaries, and a `generation` dict of public sensor knobs. `fit_v2.fit_behaviour_v2` turns the tables into generator
params; `pipeline_v2.generate_v2` is the only consumer. Table names and keys: `transfer/schema_v2.py`; v2 doc:
`docs/geolife/OUTINGS_PARAMS.md`.

## Changing it

Tests: `tests/geolife/{inside,transfer,outside,eval}/` mirror the zones, plus `tests/geolife/test_boundary.py`. Run
`PYTHONPATH=src:. python -m pytest -q tests/geolife`. A new table: `inside/measure_v2.py` + `transfer/schema_v2.py`, read it in
`outside/fit_v2.py`, add the sensitivity test in `tests/geolife/inside/test_learn_v2_sensitivity.py`. Anything that needs
real data next to synthetic data goes to `eval/`.

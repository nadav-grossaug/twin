# GeoLife data (not in git)

Gitignored: raw data and everything derived from it. Copy by hand when moving machines.

| Path | What | How to get it |
|---|---|---|
| `Geolife Trajectories 1.3/Data/<uid>/Trajectory/*.plt` (+ `labels.txt`) | raw GeoLife 1.3 (182 users, 24.9M points) | Microsoft Research download (id 52367), extract here; see `../reports/data-access.md` |
| `points.parquet` | flat cache `uid, traj, t, lat, lon` built from the raw tree by the EDA notebook (`../notebooks/build_01_eda.py`) | rebuild, or copy |

Location changes when moving from the pre-restructure layout: `geolife/data/` -> `geolife/inside/data/`,
`geolife/cache/` -> `geolife/outside/cache/` (public map parquet files), `geolife/results/artifact_*` ->
`geolife/transfer/artifacts/<variant>_seed<k>/` (or regenerate with `geolife/inside/fit_artifacts.py`).
`GEOLIFE_DATA_DIR` overrides the default location (`geolife/inside/data`).

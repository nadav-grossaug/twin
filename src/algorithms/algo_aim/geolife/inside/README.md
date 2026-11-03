# geolife/inside

May read the real traces. Contents: `data/` (raw GeoLife and `points.parquet`, gitignored, see `data/README.md`),
`fit_artifacts.py` (STAGE 1: real traces + public places -> `../transfer/artifacts/`), `notebooks/` (EDA of the real data,
`01_eda.ipynb`, `02_outings_eda.ipynb`, builders and runner), `reports/` (data access, EDA summaries, learner runs and
signal-to-noise: `learn-run.md`, `learn-v2.md`), `results/` (local intermediates, gitignored; SNR files must not leave).

Code: `src/geolife/inside/`. Everything an artifact is built from (outing extraction, caps, noise, accounting) is here;
what the other zones need from it is the artifact only.

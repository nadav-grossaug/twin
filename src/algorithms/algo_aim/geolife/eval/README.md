# geolife/eval

Needs real and synthetic data side by side: an inside-only zone. It may import every zone; none of them may import it.

- `score.py` (STAGE 3): `../outside/synthetic/*` (+ optionally the real points) -> `results/<variant>.json`, `summary.csv`.
  Runs method M v2 (places of interest, common places and segments, co-visit relations, communities) and scores it
  against the planted truth; with `--real-points` + `--places` it also runs M on the real traces and compares the
  distribution statistics. Code: `src/geolife/eval/score.py`.
- Method M (the downstream task) and its harnesses: `src/geolife/eval/m/` (`core`, `outings`, `transfer`, `outings_transfer`;
  split-half reliability and dev / held-out / sensitivity studies are in `outings_transfer`; run
  `python -m geolife.eval.m.outings_transfer`, `python -m geolife.eval.m.transfer`).
- `results/` (parquet / pickles ignored), `notebooks/` (comparison notebooks, none yet), `reports/` (`m-transfer.md`, `m-v2.md`).
- There are no empirical privacy attacks on GeoLife yet (P6 in `docs/geolife/PROCESS.md`).

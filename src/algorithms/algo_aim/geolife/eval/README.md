# geolife/eval

Needs real and synthetic data side by side: an inside-only zone. It may import every zone; none of them may import it.

- `score.py` (STAGE 3): `../outside/synthetic/*` (+ optionally the real points) -> `results/<variant>.json`, `summary.csv`.
  Runs method M v2 (places of interest, common places and segments, co-visit relations, communities) and scores it
  against the planted truth; with `--real-points` + `--places` it also runs M on the real traces and compares the
  distribution statistics. `--viz [--viz-only]` additionally writes the real-vs-synthetic panels and the summary
  distance table to `results/viz/<variant>/` (code: `src/geolife/eval/viz.py`, `compare.py`). Code: `src/geolife/eval/score.py`.
- Method M (the downstream task) and its harnesses: `src/geolife/eval/m/` (`core`, `outings`, `transfer`, `outings_transfer`;
  split-half reliability and dev / held-out / sensitivity studies are in `outings_transfer`; run
  `python -m geolife.eval.m.outings_transfer`, `python -m geolife.eval.m.transfer`).
- `notebooks/viz_comparison.ipynb` (built by `notebooks/build_viz_comparison.py`, executed with
  `geolife/inside/notebooks/run_nb.py`): the real-vs-synthetic visual comparison - space (density + difference maps,
  marginals with KS, example trajectories), moments/shape (ECDF + QQ for every quantity in `compare.QUANTITIES`),
  time profiles, and social/place structure incl. split halves. Direct lon/lat overlay needs a variant generated with
  `generate_synthetic.py --city beijing`; teleported cities are drawn in centred metric frames.
- `results/` (parquet / pickles ignored; `compare_<variant>.{csv,md}` and `viz/` panels are committed),
  `reports/` (`m-transfer.md`, `m-v2.md`, `privacy-attacks.md`, `ledger-review.md`).
- Privacy (PLAN step 7): `run_attacks.py` (user-level membership inference against the artifact, linkage /
  re-identification against the synthetic data, seed-averaging quantification; library `src/geolife/eval/attacks.py`;
  aggregate tables only to `results/attacks/`) and `review_ledger.py` (independent re-derivation of every privacy.json
  number). Write-ups: `reports/privacy-attacks.md`, `reports/ledger-review.md`. These are EMPIRICAL diagnostics, not
  the DP guarantee.

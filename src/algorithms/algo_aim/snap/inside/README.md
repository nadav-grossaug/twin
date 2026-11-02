# INSIDE: everything that reads the real graph to produce the artifact

| File | Role |
|---|---|
| `data/`, `prepare.py`, `snap_loader.py` | raw SNAP files, and turning them into the processed attributed graph |
| `attribute_policy.json`, `reports/attributes.md` | which attributes are kept and how (frozen, data-dependent) |
| `config/graph_aim_v1.json` | fitting configuration: caps, bins, privacy budget, budget shares, homophily attribute |
| `fit_artifacts.py` | STAGE 1: fit + noise + privacy accounting, writes `../transfer/artifacts/` |
| `notebooks/01_eda.ipynb` | exploration of the raw graph |

Library code: `src/graph_aim/inside/` (capping, privacy, measure, fit). Nothing here leaves the inside except through
`fit_artifacts.py`. Comparing with synthetic data (scoring, baselines, plots) is in `../eval/`, not here.

To add a node attribute or a new statistic, follow "Add features inside" in `../README.md`.

# EVAL: real and synthetic side by side

Everything that needs the real graph **and** the synthetic graphs at the same time: metrics, baselines used as reference
points, empirical privacy checks, comparison notebooks, result tables. It may import inside, transfer and outside code and
read inside data and outside output. Nothing in inside, transfer or outside may import it (`tests/graph_aim/test_boundary.py`).

| File | What it is |
|---|---|
| `notebooks/viz_comparison.ipynb` | **start here:** plots of real against synthetic behaviour (degrees, clustering, degree mixing, paths and cores, attributes, homophily) plus small examples (same-degree neighbourhoods and a connected patch, side by side) |
| `notebooks/02_graph_aim_results.ipynb` | privacy-utility curves and tables from the multi-seed sweep |
| `score.py` | STAGE 3: scores every synthetic graph and the non-private baselines against the real graph -> `results/` |
| `results/` | `sweep*.csv`, `baselines*.csv`, `attacks.json` (empirical diagnostics, not the DP guarantee) |
| `diagnose_*.py` | investigation scripts for the sampler and the private histograms (fit, generate and compare in one process) |

```bash
PYTHONPATH=src python snap/eval/score.py --seeds 3        # needs stage 1 and 2 to have run
jupyter notebook snap/eval/notebooks/viz_comparison.ipynb  # or run it headless; it reads snap/outside/synthetic/
```

The plotting code is `src/graph_aim/eval/viz.py` (library, tested in `tests/graph_aim/eval/test_viz.py`). To compare another
synthetic graph, add its file stem from `snap/outside/synthetic/` to the `SYNTHETIC` dict at the top of the notebook.

# Graph-AIM v1 on ego-Facebook

Differentially private synthetic attributed graph, generated from an inspectable artifact of noisy statistics. Plan: `docs/snap/graph-aim-plan.md`. Research: `research/graph-data-synthetic-generation.md`. Figures and tables: `snap/eval/notebooks/viz_comparison.ipynb` (plots) and `snap/eval/notebooks/02_graph_aim_results.ipynb`. Raw numbers: `snap/eval/results/`. Layout and how to reproduce (three stages): `snap/README.md`.

## Method

1. **Prepare** (`snap/inside/prepare.py`): 4,039 nodes, 88,234 edges, 8 categorical attributes compiled from the anonymized features (gender, locale, birthday, school, employer, location, hometown, language). Name fields are dropped. Homophily attribute: school.
2. **Fit** (`graph_aim.inside.fit`): Gaussian-noise measurements with zCDP accounting: degree histogram (uncapped), clustering by degree, joint degree, all 1-way and 2-way attribute marginals, and edge counts between school groups.
3. **Artifact** (`graph_aim.transfer.artifact`): JSON + `.npz`, no pickle, SHA-256 per file, `python -m graph_aim.transfer.inspect <dir>` (run with `PYTHONPATH=src`) prints the budgets, noise scales, sensitivities and caveats.
4. **Generate** (`graph_aim.outside.generate`): attributes from a Chow-Liu tree, a BTER-style block phase plus Chung-Lu phase, and degree-preserving rewiring toward the measured class-pair counts. It reads only the artifact and runs as its own stage (`snap/outside/generate_synthetic.py`) whose imports are limited to `graph_aim.transfer` and `graph_aim.outside`; `tests/graph_aim/test_boundary.py` enforces this and runs the stage with only `src` importable.

## Privacy statement

(eps, delta)-DP, delta = 1e-6, holds **separately** for edge adjacency (one friendship) and attribute-row adjacency (one person's attribute vector), each at the stated epsilon. Nothing is claimed for simultaneous changes. The node count is public. Node-DP is not provided.

**Not covered by the guarantee** (recorded in each artifact's `privacy.json`): the choice of attributes and their top values, the maximum degree, degree caps, bin edges, the histogram grid, and the budget shares were all chosen by looking at the real graph. Sensitivity bounds are derived in `measure.py` and checked by brute force on small graphs, but have not been independently audited.

## Results (mean of 3 seeds, eps per adjacency)

| Metric (error vs real, lower is better) | eps = 1 | eps = 4 | non-private | BTER (non-private) |
|---|---|---|---|---|
| Degree KS | 0.030 | 0.013 | 0.006 | 0.014 |
| Triangles (relative) | 0.288 | 0.112 | 0.058 | 0.109 |
| Transitivity | 0.050 | 0.054 | 0.002 | 0.020 |
| Clustering | 0.113 | 0.043 | 0.005 | 0.016 |
| Degeneracy (relative) | 0.214 | 0.235 | 0.229 | 0.200 |
| Assortativity | 0.085 | 0.122 | 0.155 | 0.269 |
| Attribute 2-way TVD | 0.161 | 0.071 | 0.045 | 0.079 |
| Homophily | 0.277 | 0.257 | 0.236 | 0.306 |

- Degree, triangles and attributes reach the non-private level at eps of about 4 to 8, and at eps = 4 match non-private BTER on degree KS and triangles and beat it on attributes and homophily.
- Transitivity and clustering trail BTER; transitivity does not converge to its non-private level by eps = 8.
- Assortativity depends on about 7 hub nodes whose count is released through a noisy pooled tail, so it varies a lot across seeds (per-seed errors at eps = 4: 0.10, 0.24, 0.02).
- Degeneracy overshoots (about 140 vs 115) even without noise: a sampler limit.
- Empirical privacy diagnostics (`attacks.json`) flag no variant, but they are heuristics and say nothing about the DP statement.
- Link-prediction utility does not discriminate between methods (0.99 of the real-trained AUC for every graph_aim row).

## What changed during development

See `docs/snap/PROCESS.md` issues 3 to 9. The first end-to-end run was poor (assortativity error 0.72, 21% of edges lost). Causes found by experiment: the degree cap itself made the graph assortative; clustering and joint-degree sensitivities were so large that those measurements were noise; and the generator's thresholding deleted the hub tail of the noisy degree histogram. Each was fixed in post-processing or measurement design, with the privacy tests re-run.

## Limitations and next steps

- One graph, ego-network structure (10 egos) may bias statistics.
- Parameters chosen on the real data (above). A DP selection of attribute domains and caps is the main privacy gap.
- v1 uses a fixed workload; AIM-style adaptive selection, node-DP via projection, attribute-aware blocks (homophily is the weakest metric), and learned encoders under DP-SGD are the planned next passes.

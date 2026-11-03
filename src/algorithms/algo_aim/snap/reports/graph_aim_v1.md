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

## v1.1: attribute coupling (2026-11-02)

v1 kept the structure of the graph but not who is connected to whom, or who has many connections: measured with the new
coupling metrics, every attribute's homophily sat at chance except school, and degree was independent of the attributes
(real mean degree 25 to 99 by value, synthetic flat at about 45) -- at eps=inf too, so a model gap, not a noise cost.
Plan and packages: `docs/snap/attribute-coupling-plan.md`; decisions D16-D18 in `docs/snap/PROCESS.md`.

**What changed.**

- **Measure** (package B): a new kind `attr_degree:<attr>`, the K x degree-bin count table per attribute (shipped: 4
  bins, share 0.35). Sensitivity (2 sqrt(2), sqrt(2)), derived in `measure.py` and checked by brute force.
- **Generate** (package C): degrees are now assigned conditional on attributes -- the released tables are raked (IPF) to
  the sampled margins, a maximum-entropy node-level fit combines the per-attribute tables, and bins are handed out with
  exact capacity, so the synthetic degree histogram still equals the released one exactly. Artifacts without the kind
  fall back byte-identically to v1.
- **Homophily** (package D): three `attr_edge_pair` instances (school, hometown, location) instead of one, and
  multi-channel degree-preserving rewiring. `epsilon_attr` stays equal to `epsilon_edge`; the kind's share splits
  equally over the instances.
- **Tried and dropped** (package E): sorting phase-one triangle blocks by (group, degree) closed school's homophily gap
  completely but broke structure (degree KS .006 -> .080 at eps=inf, transitivity 21x) -- see PROCESS.md D18. The
  `within_group_triangle_share` eval metric is kept.
- The method stays dataset-independent: no SNAP attribute name appears in `src/graph_aim`
  (`tests/graph_aim/test_generality.py` runs the pipeline on a toy dataset with different names and an ordered attribute).

**Acceptance** (3-seed means; gap closed = 0 at the permuted-attribute null, 1 at the real graph; bars: >= 0.3 at
eps=4, >= 0.5 at eps=inf):

| attribute | gap_eta2 eps4 | gap_deg_tvd eps4 | gap_eta2 inf | gap_deg_tvd inf |
|---|---|---|---|---|
| school | .68 | .56 | .61 | .67 |
| birthday | .54 | .53 | .69 | .67 |
| hometown | .69 | .34 | .60 | .56 |
| location | .99 | .47 | .59 | .58 |
| employer | 2.79 | -.02 | .39 | .52 |
| locale | .85 | .12 | .60 | .57 |

Four of six pass both eps=4 bars. locale and employer fail deg_tvd at eps=4 because their null floors (about .04) leave
no room over the noise; employer, the weakest real coupling (eta2 .012), also misses its eps=inf eta2 bar.

**Before / after headlines** (before = same artifacts with the `attr_degree` tables stripped, i.e. the v1 generator):

| | real | v1 (before, eps=inf) | v1.1 (eps=inf) |
|---|---|---|---|
| mean degree, birthday value 5 | 99 | 44 | 84 |
| mean degree, school 538 | 66 | 42 | 63 |
| mean degree, school 50 | 25 | 40 | 30 |
| lift hometown | 2.81 | 0.99 | 1.75 |
| lift location | 2.24 | 1.09 | 1.61 |
| lift school | 2.13 | 1.55 | 1.46 |
| homophily error (eps=4) | - | .257 | .233 |

Homophily for the configured groups improves but plateaus (gap about 0.4-0.5 even without noise): rewiring moves only
phase-two edges, a generator-capacity limit, not a budget one (PROCESS.md D17/D18). Non-configured attributes stay at
chance, as intended.

**No regression**, with one caveat. At eps=4 the structure metrics (degree KS .013, triangles .125, clustering .048,
transitivity .061, attribute 2-way TVD .069) stay within the v1 seed spread, and homophily error improves. The caveat:
at eps=inf the degree-assortativity error rises 0.155 -> 0.176 (0.185 before package D; 0.210 at 6 degree bins) --
conditioning degrees on attributes concentrates hubs in the same classes and phase two over-wires hub-hub pairs.
Documented, not fixed; a fix would go through the phase-two `joint_degree` steering.

**Where.** Outputs of this round are in separate directories, the v1 ones above stay frozen:
`snap/transfer/artifacts_v1_1/`, `snap/outside/synthetic_v1_1/`, `snap/eval/results_v1_1/` (per-attribute `cpl_*`
columns in `sweep_summary.csv`; reference rows in `coupling_reference.csv`). Plots: the executed
`snap/eval/notebooks/viz_comparison.ipynb` (coupling section) and `02_graph_aim_results.ipynb`. Ad-hoc check of any
graph: `snap/eval/coupling_report.py`.

**Privacy.** The guarantee statement is unchanged: (eps, delta)-DP per adjacency, edge-DP and attribute-row DP
separately, no node-DP. One kind was added (`attr_degree`, sensitivities above, brute-force tested); the ledger entries
come from the existing accounting. New caveats in each artifact's `privacy.json`: the `attr_degree` bin edges and the
choice of `homophily_groups` (and its count) were chosen by looking at the real graph, so they sit outside the DP
guarantee like the other public settings. A degree-by-attribute table can reveal that a small group has unusual
degrees; that is what the noise bounds, and it is also where the noise is largest, so small cells are unreliable by design.

# Real-vs-fake classifier: SNAP design

Status: planned (2026-11-02). Core and controls: `docs/realfake/PLAN.md`. Graph format and generator: `snap/README.md`, `docs/snap/PROCESS.md` D13-D14.

## 1. Sample: a random-walk subgraph

1. Pick a start node uniformly at random.
2. Walk (simple random walk, jump back to the start node when stuck) until `k` distinct nodes have been visited. `k` is drawn per sample from a fixed distribution, default log-uniform on 10..200, configurable.
3. The sample is the subgraph induced on the visited nodes, with their attribute rows and the node's degree in the parent graph.

Label 1 = built from a synthetic graph, label 0 = built from the real graph. Real and synthetic use the same sampler parameters. Real graph: `snap/inside/data/processed/attributed_graph.npz`; synthetic graphs: `snap/outside/synthetic/index.json`; loader `graph_aim.transfer.graph_io.load_attributed_graph`. For the baselines use `graph_aim.eval.baselines`.

## 2. What is dropped (giveaways of representation, not statistics under test)

| Dropped | Why |
|---|---|
| Node ids, edge order | Real ids keep the SNAP ego-network order (node 0 has degree 347, attribute rows correlate with id). Synthetic ids are exchangeable. Nodes are relabelled by visit order for both classes. |
| Attribute column order | Synthetic columns come back alphabetical, real ones in policy order. `graph_aim.eval.evaluate.align_columns(syn, real)` aligns by name and checks the domain labels. |
| Whole-graph constants (n, m) | Not a property of a sample. |

Not dropped on purpose: everything the method models or fails to model (noisy degree histogram, attribute marginals, homophily, clustering). These are what the classifier is meant to detect. Known differences from `docs/snap/PROCESS.md` D14: synthetic graphs have shorter paths, degree mixing on the diagonal, 42% same-group edges against 59%; expect the classifier to find some of these. Open point: after the first run, check the importance table for any feature that is a pure artifact of generation and not a statistic (candidate: exact `none` / `other` code frequencies). Log the outcome in PROCESS.md.

## 3. Features (grouped, for permutation importance)

| Group | Features |
|---|---|
| size | k, edges in sample, density |
| degree | within-sample degree mean, std, max, quantiles; log parent-graph degree stats; ratio of within-sample to parent degree |
| structure | triangles, mean local clustering, transitivity, degree assortativity |
| attributes | per-attribute value shares, share of `none` / `other`, same-value edge rate per attribute in `config["homophily_attrs"]`, edge co-occurrence of attribute pairs |

Models: logistic regression (baseline) and gradient boosting (stronger). The strongest is reported.

## 4. Output (eval folder)

- `snap/eval/discriminate.py`, library `src/graph_aim/eval/discriminator.py`.
- `snap/eval/results/discriminator.csv`: one row per method, epsilon, seed, model; columns AUC, CI low and high, accuracy, real-vs-real AUC, permuted-label AUC.
- `discriminator_summary.csv` (mean and std over seeds), `discriminator_importance.csv` (per feature group).
- `snap/eval/notebooks/viz_discriminator.ipynb`: AUC against epsilon with baselines, non-private variant and the real-vs-real line; importance by feature group; AUC by sample size.

Reading: AUC 0.5 means this model cannot tell real from synthetic; 1.0 means trivially detectable. A good method sits near the real-vs-real line and well below the baselines.

## 5. Results

To be filled in after the first run.

| Method | eps | AUC (95% CI) | Top feature group |
|---|---|---|---|
| real vs real (control) | - | | |
| graph_aim | inf | | |
| graph_aim | 4 | | |
| erdos_renyi | - | | |
| bter | - | | |

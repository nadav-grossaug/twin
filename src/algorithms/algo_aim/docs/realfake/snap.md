# Real-vs-fake classifier: SNAP design

Status: implemented and run (2026-11-02, branch `realfake`); results in section 5. Core and controls: `docs/realfake/PLAN.md`. Graph format and generator: `snap/README.md`, `docs/snap/PROCESS.md` D13-D14.

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

First run 2026-11-02 (n = 2000 per label, 3 folds rotating the held-out fake seed, strongest model = gradient boosting everywhere; full tables in `snap/eval/results/discriminator*.csv`, plots in `snap/eval/notebooks/viz_discriminator.ipynb`, summary in `snap/reports/discriminator_v1.md`):

| Method | eps | AUC mean±std (gbm) | Top feature group (auc_drop) |
|---|---|---|---|
| real vs real (control) | - | 0.515 logreg / 0.528 gbm | - |
| graph_aim | 0.5 | 0.9999±0.0002 | attributes (0.43) |
| graph_aim | 4 | 0.9998±0.0001 | attributes (0.31) |
| graph_aim | inf | 0.9999±0.0001 | attributes (0.24) |
| graph_aim_nocap_nonprivate | inf | 0.9999±0.0000 | attributes (0.26) |
| erdos_renyi | - | 1.0000 (logreg) | degree |
| bter | - | 0.9993±0.0010 | attributes (0.32) |

Reading: the pipeline is sound (controls at ~0.5, baselines at the easy end) and the verdict is bad for walk-level fidelity: every variant, including non-private, is near-perfectly detectable, and even plain logistic regression reaches 0.993-0.997. The epsilon gradient the AUC cannot show (it is saturated) appears in the attributes-group importance, falling monotonically 0.43 -> 0.24 from eps 0.5 to inf; degree is a distant second, structure and size negligible. AUC ≥ 0.995 in every walk-size bucket (lowest at k 10-20).

Open point from section 2, first-run outcome: resolved by the v1.1 diagnostics (see below) — `none`/`other` shares carry 7e-5 (v1) / 2e-5 (v1.1) of AUC; the artifact suspicion is refuted.

### v1.1 (branch `aim-coupling`, pinned `e802ff70`): no change in detectability

Rerun of the identical pipeline on the attribute-coupling revision (three homophily `attr_edge_pair` groups, multi-channel rewiring, attr_degree 0.35/4 bins): gbm AUC 0.996–0.9999 at every epsilon (unchanged), attributes importance 0.43 → 0.23 across eps (unchanged within noise). Splitting the attributes group (`snap/eval/results_v1_1/discriminator_importance_{perfeature,subgroups}_epsinf.csv`, eps=inf, fold 0): no single feature exceeds auc_drop 0.0012 — the signal is a distributed fingerprint; by statistic type, per-walk value shares dominate (`type:share` 0.086 v1 / 0.099 v1.1), `same_edge_rate` ≈ 0, `pair_edge_match` ≈ 0.003, `none_other` 7e-5 / 2e-5; strongest single column hometown (0.0164 → 0.0075, the one visible improvement). Reading: the classifier detects per-walk attribute composition — locally concentrated values in real communities against globally mixed values in synthetic walks — which global homophily edge-count matching does not fix; attributes need community-coherent placement. Results `snap/eval/results_v1_1/`, notebook `snap/eval/notebooks/viz_discriminator_v1_1.ipynb`.

### v2 (branch `aim-v2`, composition round): composition matched, detectability unchanged

Rerun on the v2 graphs (`snap/eval/results_v2/discriminator*.csv`, 7 variants x 3 seeds; plan
`docs/snap/composition-plan.md`, log `docs/snap/PROCESS.md` D20-D24). The generator now matches released
neighbourhood-composition tables (`attr_neigh_simpson`, the per-degree-bin Simpson index of every node's
neighbourhood) with a label-swap chain, and plants latent regions in the structure when the released spectrum
supports them (eps = inf only on SNAP). gbm AUC stays at .996-.9999 for every epsilon, logreg .989-.996. The
attributes-group importance falls from .43 / .31 / .23 (v1.1, eps .5 / 4 / inf) to .45 / .28 / .18 and the structure
group rises from .005 / .009 to .019 / .038 (eps 4 / inf): composition is no longer the only binding gap, but a
classifier on per-walk value shares still separates the graphs almost perfectly. The chain-alone graphs (no
regions) reached .14-.16 at inf; the regions improve paths, degeneracy and the community profile (modularity .71 ->
.73-.77, conductance .20 -> .08-.16) without lowering detectability. Reading: 1-hop concentration plus edge-pair
mixing, matched to within noise, does not pin down the 10-200-node walk composition the features measure; the
next measurements are multi-scale (radius-2 composition or the walk-share variance itself).

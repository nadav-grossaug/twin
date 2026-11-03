# Graph-AIM v1: implementation plan

Spec for the first method on the SNAP ego-Facebook graph. Background and alternatives: `research/graph-data-synthetic-generation.md`. Contract types: `src/graph_aim/types.py` (orchestrator-owned).

Goal: a **faithful synthetic attributed graph**, released through an **inspectable artifact** of noisy statistics, under **edge-DP on structure plus attribute-row DP on attributes**.

v1 is a *fixed-workload* measure-then-reconstruct method in the spirit of AIM. The adaptive query selection of AIM proper is v2.

## 1. Pipeline

```
SNAP files --prepare--> AttributedGraph --fit (cap, measure, noise, account)--> Artifact (on disk)
                                                                                    |
Artifact --load/inspect/validate--> generate (no access to the real graph) --> synthetic AttributedGraph --evaluate
```

The generator sees only the `Artifact`. A clean-room test must prove it (section 7).

## 2. Data preparation (`snap/inside/prepare.py`)

- Build a global node-by-feature matrix from the 10 ego networks. Column order is ego-local, but feature **names** are global (`field;anonymized feature N`, 1,406 distinct); merge the 118 nodes that appear in several egos with OR. Structure comes from `facebook_combined.txt.gz` (88,234 edges).
- **Drop direct identifiers**: fields `first_name`, `middle_name`, `last_name`, `name`.
- Compile at most **8 categorical attributes** from the remaining fields. Each attribute: top-K most frequent features of one field (K <= 8 including `other`/`none`; a node with several features of the field takes the most frequent one). Write the policy to `snap/inside/attribute_policy.json` (fields kept, K, which 1 to 2 attributes are the `homophily_attrs`, with at most 6 groups in total).
- **Known limitation, recorded as a ledger caveat**: choosing attributes and top-K values from the real data is data-dependent and outside the DP guarantee in v1. A DP top-K is v2.
- Output: `AttributedGraph` plus `snap/inside/data/processed/attributed_graph.npz`, and a short `snap/inside/reports/attributes.md` describing the result.

## 3. Privacy model

- **Public**: `n` (node set is fixed under both adjacencies), degree cap `D`, bin edges, hash salt, attribute domains (see limitation above), query workload.
- **Edge adjacency**: graphs differ by one edge. **Attribute-row adjacency**: graphs differ by one node's attribute vector.
- **Stable degree cap `cap_edges(graph, D, salt)`**: keep edge `{u,v}` iff its hash rank is within the `D` smallest among the edges incident to `u` **and** within the `D` smallest at `v`. Hash = fixed public function of the sorted endpoint pair and salt, so ranks depend only on the edge set. Removing or adding one edge changes the capped graph by at most 3 edges (the edge itself plus at most one newly admitted edge per endpoint). Attribute changes do not alter the capped graph. Default `D = 100`.
- **Mechanism**: Gaussian noise, zCDP accounting. A query with noise `sigma` and L2 sensitivity `s` costs `s^2 / (2 sigma^2)`. Each query has two sensitivities (edge, attribute-row) and charges both ledgers (`PrivacyLedger.rho_edge`, `rho_attr`). The released guarantee is stated **per adjacency**: (eps, delta)-DP for each, with delta = 1e-6, by the standard zCDP conversion. No claim is made for simultaneous change of both.
- **Sensitivities must be derived per query and checked by brute force in tests** (add or remove every edge, change every node's attribute, on small random graphs; the measured L2 change must not exceed the proved bound). Use the bound that is proved, not the empirical maximum.
- Post-processing is free: clamp to non-negative, rescale, round, enforce symmetry and 1-way/2-way consistency.

### Workload and sensitivities (agent derives and proves the exact numbers)

| Measurement `kind` | What | Edge-adj. sensitivity | Attr-row sensitivity |
|---|---|---|---|
| `degree_hist` | degree histogram of the capped graph, 0..D | up to 3 edge changes, each moving 2 nodes between 2 bins | 0 |
| `clustering_by_degree` | per degree bin: sum of local clustering coefficients (clipped) and node count | depends on common-neighbour counts; clip per-node contribution | 0 |
| `joint_degree` | edge counts between degree bins (about 12 log-spaced bins) | an endpoint's degree change can move up to `D` incident edges | 0 |
| `attr_1way` | all attributes | 0 | 2 cells change by 1 |
| `attr_2way` | all attribute pairs (at most 28) | 0 | 2 cells change by 1 |
| `attr_edge_pair` | capped edges between homophily groups | 3 edges | up to `2D` cell moves |

Budget split is a config (`snap/inside/config/graph_aim_v1.json`): shares across the six `kind`s and a total rho per adjacency. Target epsilons for the sweep: 0.5, 1, 2, 4, 8, plus a non-private run (no noise) as the upper bound.

## 4. Artifact (`src/graph_aim/artifact.py`)

On-disk layout (JSON plus `.npz` for arrays; **no pickle**):

```
artifact/
  manifest.json      # format version, method "graph_aim_v1", created, sha256 of every file
  schema.json        # n, domains, degree_cap, cap_salt, homophily_attrs
  privacy.json       # delta, rho_edge, rho_attr, derived epsilons, adjacency definitions, per-measurement entries, caveats
  measurements/      # one JSON (metadata, sigma, sensitivities, params) + one .npz (values) per measurement
  provenance.json    # library versions, seed policy, config hash
```

Required functions: `save(artifact, path)`, `load(path)` (verifies hashes), `validate(artifact)` (shapes match the contract in `types.py`, non-negative, symmetric, ledger consistent with entries), `inspect_report(path) -> str` (markdown: schema, budgets, per-measurement noise-to-signal, caveats), and a CLI `python -m graph_aim.transfer.inspect <path>`.

## 5. Fit (`src/graph_aim/capping.py`, `privacy.py`, `measure.py`, `fit.py`)

`fit(graph: AttributedGraph, config: dict, seed: int) -> Artifact`. Applies the cap, computes each measurement, adds Gaussian noise from a seeded `numpy` generator, post-processes, fills the ledger. `config["noise"] = False` gives the non-private artifact (sigma 0, ledger rho = inf).

## 6. Generate (`src/graph_aim/generate.py`)

`generate(artifact, n=None, seed=0) -> AttributedGraph`, using only the artifact.

1. **Attributes**: fit a Chow-Liu tree over the attributes using mutual information from the (post-processed) `attr_2way`, with conditional tables from the same marginals; ancestral sampling for `n` nodes.
2. **Degrees**: sample a target degree sequence from the post-processed `degree_hist`; make the sum even.
3. **Structure**: BTER-style two-phase sampler. Phase one builds dense affinity blocks of nodes with similar degree, with within-block density derived from `clustering_by_degree`. Phase two adds the remaining expected degree with a Chung-Lu sampler. The edge probability between nodes is modulated by an attribute-group affinity matrix fitted (iterative proportional fitting) to `attr_edge_pair`, and degree-bin mixing is steered by `joint_degree`. Output a simple graph with no self loops or duplicates.
4. Node labels are fresh; there is no alignment with real nodes.

Acceptable to simplify phase details if documented; the contract is: given **noise-free** measurements, the sampler must reproduce degree CCDF, average clustering, and the attribute-pair edge matrix reasonably (this is the fidelity ceiling).

## 7. Evaluation (`src/graph_aim/evaluate.py`, `baselines.py`, `attacks.py`)

- `structure_metrics(g)`: degree stats and CCDF, global and by-degree clustering, triangles, largest-component share, sampled path lengths, assortativity, k-core profile, top eigenvalues.
- `attribute_metrics(real, syn)`: 1-way and 2-way total variation distance; homophily matrix error.
- `compare(real, syn) -> dict` of scalar distances (KS and earth mover's distance on degree, absolute errors on the rest).
- Baselines, each with `fit_sample(real, seed) -> AttributedGraph`: Erdos-Renyi, Chung-Lu, degree-corrected SBM (non-DP, networkx Louvain blocks), non-DP BTER. Attributes for baselines: independent marginals.
- Empirical privacy (clearly labelled as empirical, not the DP statement): degree-and-attribute signature linkage rate, and a closest-node distance on (attributes, degree, clustering, neighbour-degree summary) from synthetic to real nodes, against the same measure from a held-out half of the real nodes.
- Utility task: link prediction on real graph edges with a simple classifier trained on the synthetic graph (features: common neighbours, degree product, attribute match) and tested on real, versus trained on real train edges.

## 8. Tests and environment

- `pytest`, tests under `tests/graph_aim/`, runnable offline. Tests must be deterministic (seeded).
- Python runs in WSL with a verified environment (the project `.venv` is broken under WSL, see Decision D7). From a worktree `W` (Windows path `C:\...\W`):
  `wsl -e bash -c 'cd /mnt/c/.../W && PYTHONPATH=src ~/snapcheck/bin/python -m pytest tests/graph_aim -q'`
  That interpreter is Python 3.14 with numpy, pandas, scipy, networkx, matplotlib, pytest. Write code compatible with Python 3.11 (project target). Do not use `-I` for your own code (it ignores `PYTHONPATH`).
- Add **no new dependencies**.

## 9. Work packages

| ID | Package | Owner files | Depends on |
|---|---|---|---|
| G1 | Data preparation | `snap/inside/prepare.py`, `snap/inside/attribute_policy.json`, `snap/inside/reports/attributes.md`, `tests/graph_aim/test_prepare.py` | `types.py` |
| G2 | Capping, privacy accountant, measurements, fit, sensitivity tests | `src/graph_aim/{capping,privacy,measure,fit}.py`, `snap/inside/config/graph_aim_v1.json`, `tests/graph_aim/test_{capping,privacy,measure,fit}.py` | `types.py` |
| G3 | Generator | `src/graph_aim/generate.py`, `tests/graph_aim/test_generate.py` | `types.py` (test with hand-built or noise-free artifacts) |
| G4 | Artifact I/O, validate, inspect, CLI, clean-room test | `src/graph_aim/{artifact,inspect}.py`, `tests/graph_aim/test_artifact.py` | `types.py` |
| G5 | Evaluation, baselines, attacks | `src/graph_aim/{evaluate,baselines,attacks}.py`, `tests/graph_aim/test_{evaluate,baselines,attacks}.py` | `types.py` |
| G6 | Integration run: epsilon sweep, `snap/inside/notebooks/02_graph_aim_results.ipynb`, `snap/reports/graph_aim_v1.md`, `snap/inside/results/` | orchestrator | G1 to G5 |

Agents work in git worktrees on their own branches, do not edit `types.py`, `README.md`, or `docs/snap/PROCESS.md`, and do not touch files owned by another package. If the contract in `types.py` is insufficient, report it instead of changing it.

## 10. Out of scope for v1

Node-DP; adaptive AIM query selection; DP selection of attribute domains; learned or embedding encoders (v3, see the research note); directed or temporal graphs.

## 11. Round 2: fixing the first end-to-end results

Evidence: `snap/inside/results/` (3 seeds) and issues 6 to 9 in `docs/snap/PROCESS.md`. Non-private v1 on ego-Facebook: degree KS 0.118, transitivity error 0.110, relative triangle error 0.655, assortativity error 0.718 (real 0.064), relative degeneracy error 0.441; non-private BTER gets 0.015 / 0.020 / 0.108 / 0.264 / 0.209. About 21% of real edges are lost by the degree cap at D=100. At eps=4, `clustering_by_degree` and `joint_degree` are mostly noise (noise/signal 3.5 and 1.6).

Work packages (disjoint file ownership; agents do not edit `types.py`, `README.md`, `docs/snap/PROCESS.md`):

| ID | Package | Owner files |
|---|---|---|
| R1 | **Sampler fidelity (non-private ceiling)**: make `generate` reproduce the real graph's structure from noise-free measurements. Diagnose the assortativity blow-up (suspects: degree-sorted phase-1 blocks, joint-degree steering), triangles, degeneracy. Targets on the real graph with an uncapped, noise-free artifact: assortativity error < 0.15, relative triangle error < 0.3, relative degeneracy error < 0.25, degree KS < 0.05, no regression on attribute TVD and homophily error. May use MCMC rewiring toward the measured targets, using only artifact numbers. Report honest ceilings if a target is infeasible. | `src/graph_aim/generate.py`, `tests/graph_aim/test_generate.py`, `snap/inside/diagnose_sampler.py` |
| R2 | **Measurement redesign (privacy-critical)**: (a) `degree_hist` on the uncapped graph with a public `max_degree` and a proved, brute-force-tested sensitivity (an edge change moves two nodes between bins; no 3x cap factor); (b) per-measurement caps (`params["cap"]`) chosen to minimise noise where a cap is still needed; (c) cheaper clustering and joint-degree measurements (smaller cap, fewer bins, or another statistic with a smaller proved sensitivity, e.g. global clustering plus a coarse degree-mixing statistic) while keeping the shapes in `types.py` readable by the generator, or add new kinds without removing old ones; (d) re-tune budget shares by analytic noise-to-signal, and record in the ledger caveats that choosing caps and shares looked at the real data. Keep all brute-force sensitivity tests, extend them to every changed measurement, and update `artifact.validate` only as far as needed for the changed shapes. | `src/graph_aim/{capping,privacy,measure,fit}.py`, `snap/inside/config/graph_aim_v1.json`, `src/graph_aim/artifact.py` (validation only), their tests |
| R3 | **Utility and evaluation extension**: link-prediction utility task (train a simple classifier on the synthetic graph, test on real edges, versus train on real train edges; features: common neighbours, degree product, attribute match; no new dependencies); extend `snap/run_graph_aim.py` with a non-private **no-cap reference variant** (`D` at or above the max degree, `noise` off) and the utility columns, and make the output tables include per-method means and standard deviations. | `src/graph_aim/utility.py`, `tests/graph_aim/test_utility.py`, `snap/run_graph_aim.py` |

After merging, the orchestrator re-runs the sweep and writes the results notebook and report.

## 12. Layout change: inside / transfer / outside (2026-11-01)

Sections 1 to 11 use the original flat module names (`graph_aim.fit`, `graph_aim.generate`, ...). The code is now split into three zones so the boundary between fitting on real data and generating synthetic data is visible and enforced; see `snap/README.md`.

| Old | New |
|---|---|
| `capping`, `privacy`, `measure`, `fit`, `evaluate`, `baselines`, `attacks`, `utility` | `graph_aim.inside.*` |
| `types`, `artifact`, `inspect`, and the graph file format (new `graph_io`) | `graph_aim.transfer.*` |
| `generate` | `graph_aim.outside.generate` |
| `snap/run_graph_aim.py` (one driver) | stage 1 `snap/inside/fit_artifacts.py`, stage 2 `snap/outside/generate_synthetic.py`, stage 3 `snap/inside/score.py` |

Enforced by `tests/graph_aim/test_boundary.py`: outside and transfer code may import only the standard library, numpy/pandas/scipy, and `graph_aim.transfer`/`graph_aim.outside`; the stage-2 script runs in a subprocess with only `src` importable. Artifact provenance keys are restricted to an allowlist (`PROVENANCE_ALLOWED`), after an exact capped-edge count was found in `provenance.json`.

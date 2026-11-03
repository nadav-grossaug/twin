# graph-aim-walks: a non-private generator that matches random-walk statistics

Plan for a **separate, non-private** version of the SNAP graph generator (2026-11-03). It does not replace or change
graph-AIM (v1 / v1.1 / v2 stay as they are, byte for byte). Problem: `docs/snap/attribute-composition-gap.md`. What v2
learned (`docs/snap/composition-plan.md`, `docs/snap/PROCESS.md` D20-D24): matching one-hop neighbourhood tables
fixed composition and homophily, but the real-vs-fake classifier still separates walks of 10-200 nodes at AUC
~0.999 — what it reads is multi-scale, and the synthetic structure has too few, too weak communities.

**Question this version answers:** with privacy set aside, can a generator that only sees readable tables of
random-walk statistics produce graphs whose walks look real? If yes, the gap is a measurement and budget problem
for the private versions; if no, the generator class is the limit.

## 1. Rules

- **Not private, said everywhere.** The artifact holds exact statistics of the real graph. Its `privacy.json` says
  `"private": false` and the manifest's `method` is `graph_aim_walks`. It must never be presented as a DP release.
- **Same air-gap layout** (`CLAUDE.md`): a new package `src/graph_aim_walks/{inside,transfer,outside,eval}/`,
  stage scripts under `snap/{inside,outside}/`, outputs in their own directories. The artifact is still readable tables;
  nothing learned (no embedding, no decoder) crosses.
- **Nothing in `graph_aim` changes behaviour.** `graph_aim_walks` may import `graph_aim.transfer` (and outside:
  `graph_aim.outside`); `graph_aim` imports nothing from `graph_aim_walks`. Eval code in `graph_aim/eval` may gain
  plotting functions and generic method naming, with the existing outputs unchanged.

## 2. The walk statistics (contract: `src/graph_aim_walks/transfer/walkstats.py`, written first)

One module, imported by inside, outside and eval, so the real and the synthetic graph are measured by the same code.
Walk process: `P` simple random walks of `T = 32` steps, starts uniform over nodes of degree >= 1, each walk driven by
its own uniforms (next node = `neighbours(x)[floor(U * d)]`, neighbours sorted). Common random numbers let the
generator keep a walk pool fixed while it changes the graph and re-simulate only the walks through a changed node.

| Table | Shape | What it captures |
|---|---|---|
| `return` | (T,) | P(walk is back at its start after t steps): small dense regions trap walks |
| `coverage` | (T,) | distinct nodes visited after t steps: how fast walks escape, i.e. path length and mixing |
| `degree_lag` | (8 lags, 6, 6) | degree bin at step t vs step t+k: who links to whom by degree, at walk scale |
| `attr_lag:<g>` | (8 lags, G, G) | attribute group at t vs t+k, for k = 1..16: how composition decays with distance |
| `share:<a>` | (3 windows, K, 10) | distribution over walks of each value's share among distinct nodes in windows of 8 / 16 / 32 steps |
| `dominance:<a>\|<b>` | (3 windows, 5, 5) | joint dominant-value share of two attributes: regions coherent in both at once |
| `transition_spectrum` | (32,) | top eigenvalues of the walk matrix `D^-1 A`: number and strength of near-separate regions |

Groups: school, hometown, location, hometown+location; share for all eight attributes; dominance pairs
hometown|location, hometown|school, school|locale. Real tables from 200,000 walks (exact up to sampling error, which
the generator's energy accounts for).

**Held out for evaluation (never matched):** walk windows above 32 (the classifier's samples reach 200 distinct
nodes and use a different walk rule), Louvain community statistics, path lengths, k-cores, and the discriminator.

## 3. Package W1: artifact and stage 1 (inside + transfer)

- `transfer/types.py`: `WalksArtifact` = the base graph-AIM artifact (exact, the v2 config with `noise = false`,
  saved with `graph_aim.transfer.artifact.save` under `base/`) + `spec` (`WalkSpec` JSON) + `tables` (name ->
  array) + `n_walks` + `provenance`. `transfer/artifact.py`: save / load / validate (SHA-256 manifest over every file,
  `method = graph_aim_walks`, `privacy.json` with `"private": false` and a plain-language warning, provenance
  allowlist, table shapes checked against the spec and the schema domains). `transfer/inspect.py`: readable report.
- `inside/fit.py`: `fit_walks(graph, config, seed)` -> `WalksArtifact` (base via `graph_aim.inside.fit.fit` on the
  v2 config with noise off, tables via `walkstats.walk_tables` with `n_walks`, spectrum via
  `walkstats.transition_spectrum`).
- `snap/inside/config/graph_aim_walks.json`: `base_config` path (`graph_aim_v2.json`), `n_walks` 200000, the spec.
- `snap/inside/fit_walks.py` (stage 1): writes `snap/transfer/artifacts_walks/graph_aim_walks_seed<k>/` (+ index).
- Tests (`tests/graph_aim_walks/`): walkstats (additivity over walk subsets, resimulation identity, hand-computed
  tables on a 6-node graph, share = 1 in the last bin, isolated nodes), artifact round trip / tamper detection /
  shape errors, a boundary test (`test_boundary.py` in the style of `tests/graph_aim/test_boundary.py`: transfer may
  import numpy/scipy + `graph_aim.transfer` + `graph_aim_walks.transfer`; outside additionally
  `graph_aim.outside` + `graph_aim_walks.outside`; nothing outside eval imports eval; stage 2 runs in a subprocess
  with only `src` on the path).

## 4. Package W2: the generator (outside)

`outside/generate.py`: `generate_walks(artifact, seed, **options) -> (AttributedGraph, info)`.

1. **Initialize** with `graph_aim.outside.generate_v2.generate_v2(artifact.base, seed=seed)` (degrees, blocks,
   regions, label chain against the base tables).
2. **Walk pool** on the synthetic graph: `P_s` walks (default 40,000) from `walkstats.draw_pool`, paths from
   `walkstats.simulate`; summed counts per table from `walkstats.walk_counts`. A node -> walks index (and each
   walk's first visit position of every node) for incremental updates.
3. **Energy**: per table, `sum (m_syn - m_real)^2 / (var_syn + var_real)` with `m` the normalised table and the
   variances the sampling variances of the two pools (binomial / multinomial per cell, so the energy is in noise
   units and a table is matched when its energy is about its cell count); plus the base graph-AIM terms the chains
   must keep: `clustering_by_degree` and `joint_degree` (structure chain), `attr_edge_pair` / `attr_2way` (label
   chain), all exact in the base artifact.
4. **Structure chain**: degree-preserving double-edge swaps over all edges. Per proposal: the walks that visit any
   of the four endpoints are re-simulated from their first visit (`walkstats.resimulate_from`), their old
   contribution subtracted and the new added; accept if the energy falls (optionally Metropolis at temperature
   `tau`). Batches of proposals whose affected walk sets are disjoint are evaluated together (vectorized), the rest
   deferred. The triangle change of a swap is computed exactly from sparse row intersections for the clustering term.
5. **Label chain**: attribute-value swaps within `attr_degree` strata (as `graph_aim.outside.refine`). Paths do not
   change; only the walks containing the two nodes are re-counted for the attribute tables.
6. **Schedule**: alternate the chains (structure `S` sweeps, label `L` sweeps, `C` cycles). Every cycle the walk
   pool is redrawn (fresh `(starts, U)`), so the graph cannot fit one pool. Stop when every table's energy is within
   2x its cell count, or at the cycle limit. `info` reports per-table energy per cycle, acceptance rates, time.
7. Budget: under 15 minutes per graph at SNAP scale (4,039 nodes, 88k edges). If the full walk re-simulation per
   proposal is too slow, propose swaps in batches and re-simulate only once per accepted batch (with a
   rejection of the batch tail if the exact energy rose), as `refine.py` does with its prefix search.
8. `snap/outside/generate_walks.py` (stage 2): reads `artifacts_walks/`, writes
   `snap/outside/synthetic_walks/graph_aim_walks_seed<k>.npz` and an index with `variant = graph_aim_walks`,
   `epsilon = null`, `method = graph_aim_walks`.
9. Tests: on a planted-partition toy graph whose real tables come from `fit_walks`, the generator's tables move
   toward the targets from the v2 initialization (energy falls by a large factor), degrees exactly preserved, the
   edge set stays simple, the label chain keeps 1-way and attr_degree exact, determinism per seed, incremental counts
   equal a from-scratch recount after every cycle.

## 5. Package W3: evaluation and visual comparison (eval)

- `graph_aim/eval/viz.py`: `community_structure(real, syns)` — Louvain (networkx, seed 0) modularity, node-weighted
  part-size distribution (CCDF), per-part conductance distribution, share of nodes in parts of >= 100 nodes; also
  label-propagation modularity. Plus `walk_profiles(real, syns)` (return, coverage, attr-lag decay, share variance by
  window **including held-out windows 64 and 128**) — implemented in `graph_aim_walks/eval/` and re-exported for the
  notebook if needed. `graph_aim/eval/evaluate.py` or a new `graph_aim_walks/eval/metrics.py`: `community_metrics`
  and `walk_metrics` returning scalars for score rows (`com_*`, `wlk_*` columns).
- `snap/eval/score.py` and `snap/eval/discriminate.py`: the method name comes from the index entry's `method` when
  present (else today's rule), so `graph_aim_walks` appears as its own method; `score.py` adds the `com_*` columns
  for every method (existing columns unchanged).
- `snap/eval/notebooks/viz_comparison.ipynb`: a new section "9. Communities" (the `community_structure` figure for
  v1.1 as today) — the notebook's other sections unchanged. A new notebook `viz_comparison_walks.ipynb`: the same
  sections for real vs v2 non-private vs graph-aim-walks, plus "Communities" and "Walk statistics (matched and
  held out)". A new `viz_discriminator_walks.ipynb`: AUC by method (real-vs-real, baselines, v1.1, v2, walks),
  feature-group importance, AUC by walk size, attributes subgroups.
- Tests for every new function (figure smoke tests, metrics on tiny graphs with known answers).

## 6. Order of work

1. Orchestrator: this plan and `walkstats.py` (done before dispatch).
2. W1, W2, W3 in parallel in worktrees cut from `aim-walks`; W2 builds its test artifacts with its own helper until
   W1 lands (the artifact type is specified above).
3. Orchestrator: merge, full suite, stage 1 (1 artifact x 3 seeds is not needed: the artifact is exact; one artifact,
   three generation seeds), stage 2, score, discriminator (`graph_aim_walks` plus v2 non-private and v1.1 non-private
   for reference), execute the notebooks, write the "What we see" cells, report.
4. Docs: `docs/snap/PROCESS.md` (W-decisions), `snap/README.md` (how to run it), `CHECKPOINTS.md`.

## 7. Acceptance (read against the held-out checks, not the matched tables)

1. Matched tables within ~2 sampling-noise units per cell (otherwise: which table could not be reached, and why).
2. Held-out: share variance at windows 64 / 128 and Louvain modularity / conductance / part sizes move toward real
   relative to v2 non-private.
3. Discriminator: gbm and logreg AUC below v2 non-private; attributes and structure importances both fall. AUC
   below the structure baselines (dc_sbm .9999, bter .9993) would be the first time any generator here gets there.
4. No regression of degree distribution (exact by construction), clustering by degree, 1-way / 2-way attributes.
5. Existing suites pass; graph-AIM outputs byte-identical.

## 8. Risks

- **Memorization is not the risk here, but overfitting the pool is:** the redrawn pool per cycle and the held-out
  windows guard it.
- **Mixing time:** structure moves change walks globally; if the chain stalls, initialize regions more strongly
  (pin `regions_R`, `within_region_frac` of v2) and let the walk chain refine.
- **Conflicting targets** (clustering vs return probability vs degree lag): the energy reports which table loses.
- **Cost:** the re-simulation cost grows with the walks through hubs; cap the pool's start nodes per hub or batch.

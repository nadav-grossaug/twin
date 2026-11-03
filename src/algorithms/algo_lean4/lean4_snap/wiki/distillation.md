---
title: Distillation (teacher → genspec)
type: concept
updated: 2026-11-03
sources: [../python/distill.py, ../lean/EnclaveGraph/GenSpec.lean, ../lean/Generate.lean]
---

# Distillation (teacher → genspec) — snap graph build

The generator is a **standalone Lean executable** (`lake exe generate`)
driven by one restricted, typed specification — `lean/out/genspec.json` —
distilled from RELEASED artifacts by `python/distill.py` (`make distill`).
PyTorch remains a training and discovery tool inside the enclave; the
generator has no Python, no PyTorch, no neural weights. Previous builds'
mappings: [[travel-pipeline]], [[stock-pipeline]], flows (page history).

**Adopted ≠ distilled.** A model is *adopted* when it beats its frozen
baseline; it is *distilled* only when a RELEASED artifact (ε-DP histogram or
suppressed table) replaces its generative role. Scoring/insight models stay
enclave-side forever; a teacher whose released replacement has no ledger
entry yet is a **declared gap**, not a silent one.

| Teacher ([[minimodels]]) | Status | Distilled executable replacement in Lean generation |
|---|---|---|
| degree_sage | adopted | **DP degree histogram** (8 bins, ε 40/100, clip 1) → `genspec.degree.cells` → each node's degree class drawn by the verified sampler (`sampleCellPMF_law`), value grid-uniform in the PUBLIC bin range; degree targets size the near-clique blocks and stub budgets. The SAGE itself certified that degree classes are structurally learnable; only the marginal crosses the boundary. |
| community_gnn | adopted | **DP community-size histogram** (node-weighted, sensitivity 1; community counts recovered by PUBLIC bin midpoints — declared post-processing) + **DP 45-cell rank-pair mixing matrix** (ε 90/100, clip 8) → `community_sizes.cells` + `mixing.rows` (symmetric expansion) → the community plan (sizes drawn until n nodes, ranks c1..c8/rest) and the routing of every cross/mixing stub. |
| triad_vgae | **honest negative** (AUC 0.946 < common-neighbor 0.990) | Never adopted, so never a teacher: the **DP wedge-closure grid** (16 cells, ε 48/100, clip 8) is distilled directly from SQL → `closure.rows` (per degree class {closed, open}) → block density q = 5/3·p (declared densification prior) and the light closure pass, where every single closure decision is a verified 2-cell draw at the RELEASED probability. The VGAE's job ended at certifying which conditional matters. |
| featcond_encoder | adopted | **DP richness histogram** (9 bins) + **DP category-presence histogram** (11 cells, clip 8; `cat_` prefix strip declared) → `richness.cells` + `category.cells` → per node: richness class → that many DISTINCT categories via `pickDistinct` (duplicate-free by `addPick_nodup`). The degree-conditioned rows the encoder learned exist as a SUPPRESSED release (`mart_cond_categories`, min-50) but the generator currently samples marginals — independence beyond marginals is a declared assumption. |
| ego_motif_pool | adopted | **DP motif-presence histogram** (6 cells, clip 4) → `motifs.cells`, carried as the VALIDATION anchor: generated motifs arise structurally (blocks, stubs, closure) plus three PLANTED instances (hub/broker/butterfly) self-verified with the DECIDABLE Lean predicates before plan.json is written; the released rates calibrate the realism audit, not a sampling step. |
| affil_ae | adopted | **Not distilled — scoring model.** Compresses affiliation profiles for anomaly/embedding use inside the enclave; feeds key_community_mlp. No ledger entry; feature-identity structure is in `assumptions.fields_not_modelled`. |
| affil_link_mlp | adopted | **Not distilled — ledger candidate.** Its releasable replacement (shared-key homophily lift per category) would need a new ε entry; the key-group marts sit on the DENIED list. The generator's stand-in is the `community_scoped_features` declared prior (features scoped to the community's pseudo-ego). |
| key_community_mlp | adopted | **Not distilled — evidence for a prior.** Its 73.4%-vs-17.8% result is the measured justification that scoping synthetic features BY community (the declared prior above) points the right way; the quantitative feature↔community coupling remains a declared gap. |
| key_cohesion_mlp | adopted | **Not distilled — named ledger candidate.** The cohesion-by-category×size table it predicts is the top candidate for the next ledger review ([[affiliation-communities]]); until a ledger entry exists, synthetic key-groups inherit no cohesion target. |

**The spec is data, never code**: categorical weights, conditional rows,
public ranges and metadata (release sha256). Lean types + parser in
`GenSpec.lean`; semantics via the verified sampler. Proved reusable
properties: `mkEdge_ok` / `edgesCanonical_sound` / `edges_nodup` (edge-set
validity), `addPick_nodup` (category assignment), `genChain` fuel-bounded
(termination).

**Privacy**: every genspec weight equals a released value or released +
declared public prior (clamp-at-zero, bin-interior-uniform, csize midpoint
division, mixing symmetric expansion, cat_-prefix strip) — asserted
cell-by-cell by `tests/test_distill.py`. Ledger 385/100 = 77/20 ≤ 4/1.

**Fidelity** (enclave-only `models_out/fidelity.json`): released-vs-exact
degree TV 0.016, accepted under the pre-declared rule (TV ≤ 0.15); the
released closure level ≈ 0.57 vs exact transitivity 0.519 reflects the
per-node equal-weighting semantics, declared.

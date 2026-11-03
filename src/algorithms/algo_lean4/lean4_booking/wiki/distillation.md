---
title: Distillation (teacher → genspec)
type: concept
updated: 2026-11-03
sources: [../python/distill.py, ../lean/EnclaveFlows/GenSpec.lean, ../lean/EnclaveGraph/GenSpec.lean, ../lean/Generate.lean]
---

# Distillation (teacher → genspec)

Merged page. Common idea in every build: the generator is a **standalone Lean executable** driven by one typed spec (`genspec.json`) distilled from RELEASED artifacts only; PyTorch teachers stay enclave-side. **Adopted ≠ distilled.** Sections: flow/travel/stock/GeoLife builds first, then the snap graph build.

## Flow / travel / stock / GeoLife builds (lean4_pcap, lean4_foursquare, lean4_stockpredict, lean4_geolife)

The big rewrite (2026-10-30): the generator is a **standalone Lean
executable** (`lake exe generate`) driven by one restricted, typed
specification — `lean/out/genspec.json` — distilled from released artifacts
by `python/distill.py` (`make distill`). PyTorch remains a training and
discovery tool inside the enclave; the generator has no Python, no PyTorch,
no neural weights.

| Teacher ([[minimodels]]) | Distilled executable replacement |
|---|---|
| NextServiceGRU | probabilistic state machine: DP-released 7×7 transition histogram (`mart_dp_transitions`, ε 49/50) blended with a PUBLIC self-transition diagonal prior; dwell/gaps from FlowShape |
| TlsRecordGRU | ordered conditional sampler: DP stack-rank mix over the fixed canonical-handshake skeleton |
| DnsCharLM | DP shape histogram + common-knowledge top-k names |
| ByteDistClassifier, FlowAutoencoder | scoring models — not generative; stay enclave-side for the realism audit |

**The spec is data, never code**: schema constraints, categorical weights,
conditional rows, temporal pointers and metadata (`genspec_version`,
release sha256). Lean types + parser in `GenSpec.lean`; semantics via the
verified sampler (`sampleCellPMF_law`). Proved reusable properties:
`stepTimes_sorted` (event ordering), `mkFlow_valid` (record validity by
construction), `genChain` fuel-bounded (termination). Statistical fidelity
and privacy are SEPARATE guarantees, accounted elsewhere.

**Privacy**: every genspec weight is a released value (post-processing) —
asserted cell-by-cell by `tests/test_distill.py::test_genspec_provenance`.
The ε rebudget: service 7@2/5 + transitions 49@1/100 + tls 6@1/25 +
dns 9@1/25 = **389/100 ≤ 4/1** (policy release-1.3.0).

**Fidelity** (enclave-only `artifacts/reports/distill_report.json`): the raw
DP transition rows are too noisy to predict with on this capture (clip 4,
~12 hosts, scale-200 noise → student argmax 0.469); with the public
self-transition prior the student matches repeat-last (0.881, teacher GRU
0.885) → **accepted** under the declared criterion, while the DP rows shape
the off-diagonal sampling mass. Per-row TV released-vs-exact is recorded.
This is the [[practices]] honest-negative pattern: the fidelity cost of ε is
measured and written down, not hidden.

### Which release is distilled (2026-11-03)

`make train` computes a fresh release in `lean/out/` before `make export`,
so "the latest export" is often the PREVIOUS release. `distill._select_release`:
explicit `$RELEASE_DIR` wins; inside the enclave a live release that differs
from the latest export is distilled (category-2 rows from the
policy-filtered marts, a loud pre-export warning); the bundle (no warehouse)
uses its export. `genspec_meta.source.release_source` records the choice and
validate / tests read the same release. GeoLife genspec also carries
per-mode `path_m` (released median leg path) for length-conditioned mode
choice. See [[geo-pipeline]], [[practices]].

## Snap graph build (lean4_snap)

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
| triad_vgae | **honest negative** (AUC 0.945 < common-neighbor 0.990) | Never adopted, so never a teacher: the **DP wedge-closure grid** (16 cells, ε 48/100, clip 8) is distilled directly from SQL → `closure.rows` (per degree class {closed, open}) → block density q = 5/3·p (declared densification prior) and the light closure pass, where every single closure decision is a verified 2-cell draw at the RELEASED probability. The VGAE's job ended at certifying which conditional matters. |
| featcond_encoder | adopted | **DP richness histogram** (9 bins) + **DP category-presence histogram** (11 cells, clip 8; `cat_` prefix strip declared) → `richness.cells` + `category.cells` → per node: richness class → that many DISTINCT categories via `pickDistinct` (duplicate-free by `addPick_nodup`). The degree-conditioned rows the encoder learned exist as a SUPPRESSED release (`mart_cond_categories`, min-50) but the generator currently samples marginals — independence beyond marginals is a declared assumption. |
| ego_motif_pool | adopted | **DP motif-presence histogram** (6 cells, clip 4) → `motifs.cells`, carried as the VALIDATION anchor: generated motifs arise structurally (blocks, stubs, closure) plus three PLANTED instances (hub/broker/butterfly) self-verified with the DECIDABLE Lean predicates before plan.json is written; the released rates calibrate the realism audit, not a sampling step. |
| affil_ae | adopted | **Not distilled — scoring model.** Compresses affiliation profiles for anomaly/embedding use inside the enclave; feeds key_community_mlp. No ledger entry; feature-identity structure for NON-affiliation categories is in `assumptions.fields_not_modelled`. |
| affil_link_mlp | adopted | **Not distilled — enclave-side realism judge.** It scores edge vs non-edge pairs on real and synthetic graphs in the realism audit (edge score 0.70 real vs ≈0.14 synthetic). A releasable replacement for the categories NOT yet modelled (hometown, location, languages, …) would need a new ε entry; the key-group marts stay on the DENIED list. |
| key_community_mlp | adopted | **Not distilled — evidence for two priors.** Its 74.1%-vs-17.8% result justifies scoping synthetic features BY community (`community_scoped_features`) and planting affiliation key-groups INSIDE one community (`affil_group_sample`); the quantitative feature↔community coupling remains a declared gap. |
| key_cohesion_mlp | adopted | **Distilled (2026-11-03).** The cohesion-by-category×size conditional it predicts was funded by two ledger reviews (displacement, then the fidelity-first cap raise): a DP member-weighted size histogram (`affil_sizes`, 36 cells over nine categories, clip 16) + the suppression-released cohesion table (`affil_cohesion_stats`, integer percent) → `affiliation.size_rows` (÷ public bin midpoint, prior `affil_member_weighted`) + `affiliation.density_rows`. The teacher certifies the conditional on the 19 cells of the five categories it was trained on (MAE 0.058 vs mean-baseline 0.263); the released table is the student (MAE 0.003); the other four released categories are student-only. The Lean generator plants key-groups for all nine categories from them ([[affiliation-communities]]). |

**The spec is data, never code**: categorical weights, conditional rows,
public ranges and metadata (release sha256). Lean types + parser in
`GenSpec.lean`; semantics via the verified sampler. Proved reusable
properties: `mkEdge_ok` / `edgesCanonical_sound` / `edges_nodup` (edge-set
validity), `addPick_nodup` (category assignment), `genChain` fuel-bounded
(termination).

**Privacy**: every genspec weight equals a released value or released +
declared public prior (clamp-at-zero, bin-interior-uniform, csize midpoint
division, affiliation midpoint division + window, mixing symmetric expansion,
cat_-prefix strip) — asserted cell-by-cell by `tests/test_distill.py`. Ledger 1369/100 ≤ 14/1 (fidelity-first cap).

**Fidelity** (enclave-only `models_out/fidelity.json`): released-vs-exact
degree TV 0.016, accepted under the pre-declared rule (TV ≤ 0.15); the
released closure level ≈ 0.57 vs exact transitivity 0.519 reflects the
per-node equal-weighting semantics, declared.

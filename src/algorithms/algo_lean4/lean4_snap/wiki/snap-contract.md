# snap-contract — frozen cross-layer interfaces for the lean4_snap port

Authoritative for the 2026-11-02 build. Any change here bumps `spec_version`.
Pattern templates: the corresponding `../lean4_foursquare` files — keep names,
shapes and column conventions IDENTICAL except the substitutions below.

## 1. Spec constants (dbt `vars:` = `python/spec.py` = `lean/EnclaveGraph/Params.lean`)

dbt project name `enclave_graph`, profile `enclave_graph`, `spec_version: "1.0.0"`.
Flat scalars only (spec.py regex-parses the block). Values:

```yaml
spec_version: "1.0.0"
ingest_dir: "{{ env_var('GRAPH_INGEST_DIR', '../artifacts/ingest') }}"
# degree bins 1,2,3-4,5-8,9-16,17-32,33-64,65+ (lower edges)
deg_e2: 2
deg_e3: 3
deg_e5: 5
deg_e9: 9
deg_e17: 17
deg_e33: 33
deg_e65: 65
# local clustering classes: 0,(0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1),1
# exact integer comparisons on cc = 2*tri / (d*(d-1)) via cross-multiplication
cc_den: 5
# triangle-count bins 0,1-10,11-100,101-1000,1001-10000,10001+ (lower edges)
tri_e1: 1
tri_e11: 11
tri_e101: 101
tri_e1001: 1001
tri_e10001: 10001
# feature-richness bins 0..7 singletons + 8+
rich_e8: 8
# circle-membership bins 0,1,2,3,4+
circ_e4: 4
# community-size bins <25,25-50,51-100,101-200,201-400,>400 (lower edges)
csize_e25: 25
csize_e51: 51
csize_e101: 101
csize_e201: 201
csize_e401: 401
# community rank labels c1..c8 by size, then 'rest'
n_community_ranks: 8
# motif thresholds (public; from measured distributions)
hub_degree_min: 150
leaf_degree: 1
broker_degree_min: 17
broker_cc_max_num: 1
broker_cc_max_den: 5
bridge_communities_min: 3
butterfly_circles_min: 3
core_degree_min: 65
core_cc_min_num: 3
core_cc_min_den: 5
# suppression (non-DP artifacts)
suppress_min_nodes: 50
# DP release parameters (public)
dp_clip_total: 8
dp_count_clip: 1
dp_csize_clip: 1
dp_clip_motifs: 4
# ingest determinism
louvain_seed: 7
# synth validation band: |synth closure - released closure| <= tol_pct/100
synth_closure_tol_pct: 15
```

Seed `dbt/seeds/category_map.csv` (`raw_category,category`): birthday,
education, first_name, gender, hometown, languages, last_name, locale,
location, work map to themselves; middle_name, name, political, religion →
other. Public category order (fixed): birthday, education, first_name,
gender, hometown, languages, last_name, locale, location, work, other.

## 2. Ingest (`python/ingest_graphml.py` → `artifacts/ingest/`)

CLI: `ingest_graphml.py [--graphml PATH] [--out DIR]` (defaults
`facebook_full_features.graphml`, `artifacts/ingest`). networkx + stdlib.
Tab-separated with header row:

- `nodes.tsv`: `node_id` (int), `is_ego` (0/1), `nx_degree`, `nx_triangles`,
  `community_id` (Louvain, seed 7, sizes-desc re-labeled 0..k-1), `n_circles`
  (total memberships across egos), `n_egos` (ego networks containing node).
- `edges.tsv`: `src`, `dst` (ints, canonical src < dst, no dups/self-loops).
- `node_features.tsv`: `node_id`, `feature_key` (full GraphML attr name),
  `raw_category` (first `;`-token convention → from key: segment after
  `fN_` up to the category word; derive from the manifest-free key by
  matching known raw category prefixes), `value` (always 1).
- `circles.tsv`: `node_id`, `ego_id`, `circle_label`.
- `ingest_meta.json`: `graphml_sha256`, `n_nodes`, `n_edges`, `n_communities`,
  `modularity`, `louvain_seed`, `generated_at`, `source_path`.

Re-ingestion of SYNTHETIC GraphML must work through the same script
(`--graphml synth/synth_graph.graphml --out <scratch>/ingest`).

## 3. dbt (DuckDB at `artifacts/enclave.duckdb`, profile dir `dbt/`)

staging: `stg_nodes`, `stg_edges`, `stg_node_features` (joins seed →
`category`), `stg_circles` + `schema.yml` tests.
quality: `q_data_integrity` (no self-loops, src<dst, dup edges 0, feature →
node referential integrity, SQL degree/triangles == nx columns),
`q_temporal_fields` (every feature value = 1; no attr parses as a date —
static-snapshot assertion).
features:
- `feat_nodes`: `node_id, degree, degree_class, tri_count, wedge_count,
  cc_num, cc_den, cc_class, tri_class, community_id, community_rank,
  richness, richness_class, n_circles, circle_class, max_circles_one_ego,
  n_egos, neighbor_communities, is_hub, is_leaf, is_broker, is_bridge,
  is_butterfly, is_core, is_ego, qf_zero_features`.
  SQL computes degree/tri/wedges from `stg_edges` (wedge join); `cc_num =
  2*tri_count`, `cc_den = degree*(degree-1)`; classes by integer
  cross-multiplication (`cc_den=0 → cc_zero`). `community_rank`: `c1..c8`
  by size then `rest`.
- `feat_edges`: `src, dst, src_class, dst_class, same_community, pair_cell,
  common_neighbors` (`pair_cell` = `<ra>__<rb>` with ranks ordered
  c1<c2<…<c8<rest).
marts (non-DP): `mart_degree_stats`, `mart_clustering`,
`mart_triangle_stats`, `mart_community_sizes`, `mart_community_mixing`,
`mart_wedge_closure` (per degree_class: n_nodes, wedges, closed,
closure_rate; closed(v)=tri_count(v), open=wedge−tri), `mart_feature_categories`,
`mart_circle_stats`, `mart_motif_summary`, `mart_cond_categories`
(degree_class × category node counts, cells with n < suppress_min_nodes
suppressed), `mart_assoc_cells` (grid_name, row_label, col_label, n for
grids `deg_cc`, `deg_rich`, `motif_comm`, plus 2×2 grids
`cat__<a>__<b>` for all unordered category pairs over the 11 public
categories), `mart_feature_assoc` (per grid: chi2, dof, cramers_v, n).
marts (DP inputs; exact shape of foursquare `mart_dp_region.sql` —
columns `cell, clipped_count, n_contributing_nodes, clip`):

| mart | cells (fixed order) | clip |
| --- | --- | --- |
| `mart_dp_degree` | deg_1, deg_2, deg_3_4, deg_5_8, deg_9_16, deg_17_32, deg_33_64, deg_65p | 1 |
| `mart_dp_clustering` | cc_zero, cc_0_20, cc_20_40, cc_40_60, cc_60_80, cc_80_99, cc_one | 1 |
| `mart_dp_triangles` | tri_0, tri_1_10, tri_11_100, tri_101_1000, tri_1001_10000, tri_10001p | 1 |
| `mart_dp_richness` | rich_0 … rich_7, rich_8p | 1 |
| `mart_dp_categories` | cat_birthday, cat_education, cat_first_name, cat_gender, cat_hometown, cat_languages, cat_last_name, cat_locale, cat_location, cat_work, cat_other | 8 (proportional L1) |
| `mart_dp_circles` | circ_0, circ_1, circ_2, circ_3, circ_4p | 1 |
| `mart_dp_community_sizes` | csize_lt25, csize_25_50, csize_51_100, csize_101_200, csize_201_400, csize_gt400 (NODE-weighted: one event per node in its community's size bin; distill divides by public midpoints) | 1 |
| `mart_dp_mixing` | `<ra>__<rb>` for ra ≤ rb over c1..c8,rest — 45 cells, order row-major upper triangle | 8 (proportional L1 over each node's incident edges) |
| `mart_dp_closure` | `clo_<degree_class>__closed`, `clo_<degree_class>__open` ×8 classes = 16, degree-class order | 8 (proportional L1 over each node's centered wedges) |
| `mart_dp_motifs` | motif_hub, motif_leaf, motif_broker, motif_bridge, motif_butterfly, motif_core | 4 (proportional L1 over presence flags) |

ml: `ml_graph_scores` (dbt Python node; reads `models_out/registry.json`,
never retrains): `node_id, anomaly_score, pred_degree_class,
community_agree, model_version, confidence`.

## 4. ε ledger (exact fractions; Python `release.py` + Lean both enforce)

degree 8×(5/100) + clustering 7×(5/100) + triangles 6×(5/100) +
richness 9×(4/100) + categories 11×(4/100) + circles 5×(4/100) +
community sizes 6×(4/100) + mixing 45×(2/100) + closure 16×(3/100) +
motifs 6×(3/100) = **385/100 ≤ 4/1**.
Env overrides mirror foursquare (`DP_DEGREE_EPS1` etc., default EPS2=100).

## 5. Lean layer (`lean/`, namespace `EnclaveGraph`, lib name `EnclaveGraph`)

Same lakefile/toolchain/manifest as foursquare (SampCert a2c53bb, Lean
v4.34.1; `List.Pairwise` not `Chain'`). Executables: `checks`, `release`,
`sample`, `catsample`, `generate`. Files: `Params.lean`, `GraphShape.lean`
(bin maps, total), `Motifs.lean` (decidable isHub/isLeaf/isBroker/isBridge/
isButterfly/isCore over exported per-node records), `Assoc.lean` (exact ℚ
χ²), `Stats.lean`, `DPRelease.lean`, `Sampler.lean`, `GenSpec.lean`,
`Generate.lean` (GraphML renderer).
`lean/data/` inputs (written by `python/export_lean_data.py`):
`dp_counts.json` (grids: name, eps1, eps2, clip, cells `[{cell, n}]`),
`nodes.json` (per-node: node_id, degree, tri, wedges, community_rank,
neighbor_communities, max_circles_one_ego, n_circles, richness, categories,
sql motif flags + classes — for SQL↔Lean agreement), `assoc.json`
(contingency cells + SQL chi2 as num/den per grid).
`lean/out/` outputs: `dp_release.json`, `genspec.json` (distill),
`sample_hist.json`, `checks_report.json`.

## 6. Distill / genspec / generator

`python/distill.py` reads ONLY `lean/out/dp_release.json` (+ suppressed
`mart_cond_categories` export in `release/` if present) → `lean/out/genspec.json`:
`{spec_version, source:{dp_release_sha256}, degree_hist, clustering_hist,
richness_hist, category_hist, circle_hist, community_sizes, mixing_rows,
closure_rows (per degree class: closed_w/open_w), motif_rates, priors:[…],
schema:{n_nodes_default: 4039, feature_key_shape, ego_ids: []}}`.
Weights = released values clamped at 0 (clamp declared as public prior).
`lake exe generate <genspec> <n_nodes> <out_dir>` → `synth_graph.graphml` +
`plan.json` (planted: one hub, one broker, one butterfly; community size
draws; closure-pass stats). Validation: `scripts/synth.py validate` re-ingests
via `ingest_graphml.py` + dbt scratch build (`GRAPH_INGEST_DIR` +
`GRAPH_DB` overrides), checks planted recall 1.0, degree-hist χ² GOF vs
released weights, |synth global closure − released| ≤ 15/100, `lake exe
checks` agreement on the synth export.

## 7. Minimodels (`python/train_minimodels.py`, PyG, CPU-deterministic via `FLOW_DEVICE=cpu`)

Content-addressed artifacts in `models_out/<version>/`, registry-activated
(foursquare `artifacts.py` pattern). Node split 80/20 by id hash. Models +
frozen gates:

| name | model | gate (adopt iff) |
| --- | --- | --- |
| degree_sage | 2-layer GraphSAGE → degree-class head | val acc > majority-class share + 0.02 |
| community_gnn | GCN/SAGE embeddings + KMeans(k=15) | ARI vs Louvain labels > 0.30 |
| triad_vgae | PyG VGAE link prediction | val AUC > common-neighbor-count AUC + 0.01 |
| featcond_encoder | degree_class + embedding → 11-dim category presence | mean per-category val AUC > 0.55 |
| ego_motif_pool | mean-pooled 1-hop ego features → 6-dim motif presence | macro-F1 > marginal-rate baseline |

Honest negatives are kept (triad_vgae is EXPECTED to lose to the
common-neighbor baseline at transitivity 0.52).

## 8. Ops

Makefile = foursquare's verbatim modulo: `ingest` step before dbt (`$(PY)
python/ingest_graphml.py`), wording, `N_NODES ?= 4039` replacing `N_USERS`,
same six flows + `help`, `LEAN_HOME=/opt/data/lean`. Release policy env
gate: `SNAP_ALLOW_DEV_RELEASE=1` (named in `config/release_policy.yaml`).
Bundle: `dist/bundle/` source-only (make_bundle.sh pattern).

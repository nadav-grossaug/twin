# affiliation-communities — key-groups and the four affiliation minimodels

Added 2026-11-03 on top of the as-built [[snap-pipeline]]. A **key-group** is
the set of people sharing ONE anonymized feature instance — the same
education value (school/degree/year), the same employer, the same hometown
or language entry. These are affiliation communities, orthogonal to the
structural Louvain communities, and they are person-identifying: everything
on this page stays ENCLAVE-SIDE (all three marts are on the policy denied
list) pending a ledger review. Note the crawl scoping: feature instances are
ego-scoped, so a key-group never spans ego networks — "affiliation" here
means within-crawl-scope affiliation.

## Warehouse (dbt marts, enclave-only)

- `mart_edge_shared` — per friendship edge: shared-key count per public
  category (wide, 11 columns) + total. Measured: mean ≈ 3.5 shared keys per
  edge; only 1,970 of 88,234 edges share nothing.
- `mart_key_groups` — one row per key with ≥ 2 holders (2,280 groups; 1,137
  with ≥ 5): category, group size, internal friendship edges, density.
- `mart_affil_cohesion` — category × size-bin cohesion rollup with a
  suppression display flag.

## Measured cohesion (vs the random-pair baseline 88,234 / C(4039,2) ≈ 0.0108)

| category | n_groups | mean density | × baseline |
| --- | --- | --- | --- |
| hometown | 90 | 0.613 | 57× |
| last_name | 194 | 0.600 | 56× |
| education | 925 | 0.505 | 47× |
| location | 132 | 0.485 | 45× |
| work | 620 | 0.459 | 43× |
| birthday | 92 | 0.348 | 32× |
| languages | 81 | 0.223 | 21× |
| locale | 29 | 0.205 (mean size 142) | 19× |
| gender | 20 | 0.104 (mean size 205) | 10× |

Cohesion decays with size but stays far above baseline: education groups of
17+ still sit at density 0.166 (15×), work 17+ at 0.116 (11×). Every 2-person
affiliation group in the crawl is an actual friendship (density 1.0 across
education/work/hometown g_2 bins — an ego-scoping artifact worth remembering).

## The four minimodels (artifact version 6c004f779b2e; all gates honest)

| model | result | baseline | adopted |
| --- | --- | --- | --- |
| `affil_ae` (951 affiliation keys → 16-dim code) | val BCE 0.0170 | marginal-rate BCE 0.0350 | ✅ profiles compress 2× below marginals |
| `affil_link_mlp` (13-dim shared-category profile → P(edge)) | val AUC 0.9507 | total-shared-count ranker 0.9347 | ✅ WHAT you share beats HOW MUCH (+0.016, over the +0.01 gate) — and the scalar baseline alone at 0.93 shows affiliation is a strong friendship signal |
| `key_community_mlp` (affil_ae embedding → community rank) | val acc 0.734 | majority rank 0.178 | ✅ affiliations largely determine the structural community — the two community notions are strongly aligned |
| `key_cohesion_mlp` (category ⊕ log₂ size → density) | val MAE 0.123 | train-mean predictor 0.355 | ✅ cohesion is predictable from size + category (release-ledger candidate) |

The original five models retrain unchanged (triad_vgae still the honest
negative: AUC 0.945 vs common-neighbor 0.990). Full suite: 70 passed.

## Release stance

Nothing here crosses the boundary yet. The natural candidates for a future
ledger entry are the `mart_affil_cohesion` aggregates (category × size-bin
mean densities under per-node contribution clipping) and the
`key_cohesion_mlp` teacher's distilled rows; a ledger entry must displace an
existing one (the budget sits at 385/100 of 400/100). Until that review,
key-group artifacts are denied-by-policy, and the synthetic generator does
NOT model affiliation structure (already declared in
`assumptions.fields_not_modelled` as "feature identity structure").

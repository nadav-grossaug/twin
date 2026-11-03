---
title: Affiliation communities (key-groups)
type: concept
updated: 2026-11-03
sources: [../dbt/models/marts/mart_key_groups.sql, ../dbt/models/marts/mart_dp_affil_sizes.sql, ../dbt/models/marts/mart_affil_release.sql, ../models_out]
---

# affiliation-communities — key-groups and the four affiliation minimodels

Added 2026-11-03 on top of the as-built [[snap-pipeline]]. A **key-group** is
the set of people sharing ONE anonymized feature instance — the same
education value (school/degree/year), the same employer, the same hometown
or language entry. These are affiliation communities, orthogonal to the
structural Louvain communities, and they are person-identifying: the key-level
marts below (`mart_edge_shared`, `mart_key_groups`, the full
`mart_affil_cohesion`) stay ENCLAVE-SIDE on the policy denied list. Two
AGGREGATES derived from them cross the gate since the 2026-11-03 ledger
reviews — a DP size histogram and a suppression-released cohesion table (see
"Release stance" below). Note the crawl scoping: feature instances are
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

## Edge coverage by category — what actually carries "same attribute ⇒ connected"

Share of the 88,234 real friendships whose endpoints share ≥ 1 key of the
category (`mart_edge_shared`, measured 2026-11-03):

| category | edges sharing it | | category | edges sharing it |
| --- | --- | --- | --- | --- |
| locale | 0.762 | | hometown | 0.127 |
| education | 0.680 | | location | 0.090 |
| gender | 0.545 | | work | 0.048 |
| birthday | 0.139 | | languages / last_name | 0.037 / 0.025 |

Union coverage: education+work 0.681; + locale 0.946; + gender 0.974; all
eleven categories 0.978. So the huge locale and gender groups (mean size ≈ 213,
density ≈ 0.12) carry most of the coverage, while work (the "same workplace
clique") carries cohesion, not coverage. This is what chose the nine released
categories ([[snap-pipeline]], Release stance below); first_name / other add
≤ 0.001.

**Nesting.** Groups of different categories are NESTED: 86% of an education
group's members (median) sit inside one locale group, 88% for work, 73% for
hometown, 71% for birthday; locale and gender groups overlap heavily (median
containment 0.63, Jaccard 0.42). Locale/gender groups behave like the
ego-network containers. The generator does not model this (anchoring groups to
a shared grid so they nest was tested and made density fidelity worse).

## The four minimodels (artifact e999a0ae31fd, retrained on spec 1.2.0 — supersedes 6c004f779b2e with identical verdicts; all gates honest)

| model | result | baseline | adopted |
| --- | --- | --- | --- |
| `affil_ae` (951 affiliation keys → 16-dim code) | val BCE 0.0181 | marginal-rate BCE 0.0350 | ✅ profiles compress 2× below marginals |
| `affil_link_mlp` (13-dim shared-category profile → P(edge)) | val AUC 0.9509 | total-shared-count ranker 0.9347 | ✅ WHAT you share beats HOW MUCH (+0.016, over the +0.01 gate) — and the scalar baseline alone at 0.93 shows affiliation is a strong friendship signal |
| `key_community_mlp` (affil_ae embedding → community rank) | val acc 0.741 | majority rank 0.178 | ✅ affiliations largely determine the structural community — the two community notions are strongly aligned |
| `key_cohesion_mlp` (category ⊕ log₂ size → density) | val MAE 0.126 | train-mean predictor 0.355 | ✅ cohesion is predictable from size + category (DISTILLED: certifying teacher for the released cohesion table) |

The original five models retrain unchanged (triad_vgae still the honest
negative: AUC 0.945 vs common-neighbor 0.990).

## Release stance (2026-11-03, second review: fidelity-first budget)

**First review (displacement).** Triangles (5→2/100) and motifs (3→2/100), both
informational, funded an 8-cell DP histogram over education and work and the
suppression-released cohesion table; ledger 393/100 ≤ 4/1.

**Second review (user decision: fidelity over privacy).** Two measurements
showed that review was too small. (1) Edge coverage: locale is shared by 76% of
real edges, education 68%, gender 55%, birthday 14%, hometown 13%, location 9%,
work 5%, languages 4%, last_name 3% — locale and gender (groups of ~213, the
ego-network containers) were never modelled, and education+work alone gave only
34% synthetic coverage against 68% real because the clip of 2 shrank everyone's
memberships ~4× (release held ≈6,000 member-slots vs ≈23,900 exact). (2) A
per-grid noise audit: the mixing grid was the noisiest (36 of 45 cells below 2×
the noise scale, TV vs exact 0.172). So the cap went 4/1 → 14/1, spent as:
mixing 2/100 → 10/100 per cell (TV 0.049, relative error 7%) and the
affiliation size histogram → **36 cells over nine categories** (birthday,
education, gender, hometown, languages, last_name, locale, location, work) with
per-node clip 16 (≈85% of nodes unscaled), 18/100 per cell: ledger **1369/100
≤ 14/1**. The bound is weak (e^ε at ε ≈ 13.7; the row-level unit and the
declared node-DP gap weaken it further). The cohesion rollup still crosses as the
suppression artifact `affil_cohesion_stats` (29 of 34 rows released; 5 suppressed
below 50 members). Key-level artifacts (`mart_key_groups`, `mart_edge_shared`,
the full `mart_affil_cohesion`) stay denied.

Downstream (see [[snap-pipeline]]): distill builds the genspec `affiliation`
section from ONLY those two releases (size cells ÷ public bin midpoints, with a
per-category cap on the open bin so each category's uniform mean matches its
measured mean group size); `key_cohesion_mlp` certifies the conditional for the
five categories it was trained on (education, work, hometown, location,
languages) while the other four are student-only; the Lean generator plants the
key-groups inside communities and deficit-wires them to the released density.
`affil_link_mlp` is the enclave-side realism judge.

**Measured effect.** Shared-key edge fraction 0.34 → 0.91 (real 0.98);
link-judge edge score 0.14 → 0.52 (real 0.70); group counts match real closely
(education 387/177/97/166 vs 449/176/120/180 for the 2–4/5–8/9–16/17+ bins);
mean per-cell density error 0.095–0.119 → 0.047. Residuals: small cells are
noise-limited (a bin whose released count falls below the Laplace scale can be
zeroed, e.g. location 2–4), mid-bin density of low-target categories (birthday,
languages) runs +0.1–0.2 high, and the 9-vs-15 community count and
transitivity/modularity overshoot are unrelated to affiliation.

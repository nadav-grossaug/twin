# torch-lean-arch: Enclave Social-Graph Encoder and synthetic graph generator

## Goal

Build a reproducible encoder that converts a private social graph
(`facebook_full_features.graphml`: 4,039 people, 88,234 mutual friendships,
one connected component, 10 ego networks, 2,280 ego-scoped anonymized binary
profile features in 14 categories, friendship circles per ego — the SNAP
"Social circles: Facebook" dataset used for social-media mining) into a
versioned graph-behavior representation: the degree distribution, local
clustering / triadic-closure structure, community sizes and mixing, node
feature-category distributions, and ego-network motifs — for graph-mining
training and downstream synthetic graph generation. Run the encoder inside an
enclave so the raw friendship lists and per-person profile features remain
within the protected processing boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

This is the **fifth instantiation** of the method proven on network-flow
telemetry and migrated four times (stock series, Foursquare travel, pcap
flows, GeoLife trajectories): see `METHOD.md` (the dataset-agnostic
replication guide, especially §12's substitution map),
`wiki/migration-plan-snap.md` (the phased plan for THIS dataset) and
`wiki/dataset-snap.md` (the measured data profile). The five-layer
architecture, three release categories, spec-mirror discipline, Lean-only
generator, and every calibration lesson in METHOD §6 carry over unchanged.
The reference implementation for the ops substrate is `../lean4_foursquare`
(six-flow Makefile, `LEAN_HOME` store, single source-only bundle).

**What is sensitive here:** a person's friendship list (who they know), their
profile-feature fingerprint (which anonymized education / employer / hometown
/ language values they carry), and their circle memberships. Degree plus a
handful of known friendships is a strong re-identification vector in social
graphs (narayanan-shmatikov-style attacks); the feature fingerprint narrows
it further. The protected entity is the **person (node)**; raw node ids,
individual edges, feature-instance ids and circle rosters never cross the
boundary in any form.

The encoder learns and measures graph behavior from the input dataset,
producing an explicit specification that a later synthetic-graph generator
can target.

### Dataset-specific emphases (this migration's brief)

Five learning targets, each with a PyTorch(-Geometric) model inside the
enclave and a distilled, released output:

| Learning target | PyG approach | Distilled output |
| --- | --- | --- |
| Degree distribution | GraphSAGE embeddings + statistics | DP degree histogram over public bins |
| Community structure | GNN embeddings + clustering | DP community-size histogram and rank-labeled mixing matrix |
| Triadic closure | VGAE / edge prediction | Conditional wedge-closing probabilities by degree class |
| Node features | Feature-conditioned encoder | Category-presence distributions (+ degree-conditioned rows under suppression) |
| Ego-network motifs | Graph-level pooling + motif statistics | DP motif-presence histogram and conditional rules |

A graph VAE (PyG `VGAE`) is the starting point for edge structure, but VGAE
learns a reconstruction-oriented latent and cannot be translated losslessly
into a compact Lean rule set — a **dedicated distillation stage** replaces
every generative model with released histograms/rows (METHOD §8).

Further emphases:

1. **Strong community structure** — the dataset's published average
   clustering coefficient is ≈ 0.6055; our ingest reproduces it exactly
   (0.6055 measured). Louvain finds 15 communities at modularity 0.835 with
   96.15% of edges within communities. The synthetic graph must reproduce
   the degree histogram, a clustering coefficient near 0.6, and the
   community size/mixing structure.
2. **χ² association analysis** — co-occurrence and correlation between
   columns, features and motifs: contingency grids between discretized node
   features (degree class × clustering class, degree class × feature
   richness, category × category co-occurrence, motif × community rank),
   exported as cells so Lean recomputes χ² exactly over ℚ.
3. **Temporal flow logic — measured absence.** The graph is a static
   snapshot: no timestamps exist anywhere in the file. Attribute names that
   look temporal (`birthday;…`, `work;end_date;…`) are anonymized binary
   presence flags, not dates; node ids are per-ego anonymization order, not
   arrival order. A quality model (`q_temporal_fields`) asserts every
   time-named attribute is a 0/1 flag, and the finding is recorded in
   `wiki/dataset-snap.md`. The "sequence" analog in this build is the
   **triadic-closure process** (wedge → triangle), modelled conditionally.

### Outputs (planned)

- Per-node features: degree + degree class, triangle count + class, local
  clustering coefficient + class, community label (Louvain, seeded) and
  community rank, feature-category presence vector and richness class,
  circle-membership count + class, ego-network memberships, quality flags.
- Per-edge features: canonical (src < dst) undirected edges, common-neighbor
  count, same-community flag, endpoint degree classes.
- A **public feature-category taxonomy** (one half of the DP cell grids,
  fixed in the spec, independent of this dataset's values): the SNAP
  featname convention's category prefixes — `birthday`, `education`,
  `first_name`, `gender`, `hometown`, `languages`, `last_name`, `locale`,
  `location`, `work`, `other` (11 categories; `middle_name`, `name`,
  `political`, `religion` fold into `other`).
- Public class bins (all fixed in the spec; edges chosen from measured
  distributions, then frozen as public):
  - **degree bins**: 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, 65+
    (measured 75 / 98 / 192 / 400 / 712 / 869 / 804 / 889);
  - **local clustering classes**: 0, (0,0.2), [0.2,0.4), [0.4,0.6),
    [0.6,0.8), [0.8,1), 1 (measured 76 / 32 / 498 / 1413 / 1213 / 540 / 267);
  - **triangle-count bins** (per node): 0, 1–10, 11–100, 101–1000,
    1001–10000, 10001+;
  - **feature-richness bins** (categories present per node): 0, 1, 2, 3, 4,
    5, 6, 7, 8+ (measured 8 / 26 / 553 / 539 / 687 / 813 / 721 / 435 / 257);
  - **circle-membership bins** (circles per node across egos): 0, 1, 2, 3,
    4+ (measured 1155 / 2124 / 439 / 183 / 138);
  - **community-size bins**: <25, 25–50, 51–100, 101–200, 201–400, >400.
- Graph motifs with public thresholds (SQL and Lean mirrors):
  - **triangle** — a closed wedge, the unit of triadic closure (1,612,010
    triangles; 9,314,849 wedges; global closure 0.5192);
  - **hub** — degree ≥ 150 (≈ p95; the high-degree re-identification class);
  - **leaf** — degree 1 (periphery; 75 nodes);
  - **broker** — degree ≥ 17 and local clustering < 0.2 (structural-hole
    spanner: well-connected, unclosed neighborhood);
  - **bridge** — neighbors in ≥ 3 distinct communities;
  - **social-butterfly** — member of ≥ 3 circles within one ego network.
- **Wedge-closure grid**: P(closed | wedge-center degree class) — the
  triadic-closure conditional the generator uses to reach clustering ≈ 0.6.
- Association analysis: χ²/Cramér's V on full contingency grids
  (degree × clustering, degree × richness, category × category,
  motif × community rank); Lean recomputes χ² exactly over ℚ.
- Conditional scores and embeddings from versioned mini-models.
- Versioned mini-models, evaluation results, candidate formal specifications.
- Differentially private releases of aggregate distributions, with the noise
  mechanism formally verified with
  [SampCert](https://github.com/leanprover/SampCert) (Lean 4, pinned commit
  `a2c53bb` and its toolchain, as in the reference implementation).
- A formally verified sampler for drawing from released distributions, built
  on SampCert's samplers — the entry point for the downstream generator.
- A standalone Lean generator (`lake exe generate`) that renders a synthetic
  graph **in the source GraphML format** (node elements with ego-scoped
  sparse feature keys and circle strings, canonical undirected edge
  elements) from the distilled genspec alone, validated by re-ingestion
  through the production pipeline, with kernel-checked certificates.
  **The only artifacts the generator consumes are Lean sources and the
  released JSON — no compiled binaries, no Python models, no raw data.**

## Processing Architecture (planned)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | `python/ingest_graphml.py` → TSV/parquet, then DuckDB `read_csv` in dbt staging | `stg_nodes`, `stg_edges` (canonical src<dst), `stg_node_features` (long), `stg_circles`; stable `node_id`, content-hash fingerprint; Louvain labels (seeded) and networkx triangle/clustering cross-check columns |
| Cleaning and features | dbt + DuckDB (`feat_nodes`, `feat_edges`) | Degree/triangle/clustering classes, community rank, richness, circle counts, quality flags; SQL recomputes triangles/wedges and a quality model cross-checks networkx |
| Mini-model fitting | PyTorch + PyG (`python/train_minimodels.py`), CPU-deterministic mode | DegreeSAGE (degree class from structure; gate: majority class), CommunityGNN (embeddings+KMeans vs Louvain; gate: ARI vs random), TriadVGAE (edge prediction; gate: common-neighbor baseline AUC), FeatCondEncoder (category presence; gate: marginal rates), EgoMotifPool (motif vector from pooled ego subgraph; gate: marginal); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_graph_scores`) | Per-node predictions/scores with confidence and model version; never retrains on rebuild |
| marts | dbt SQL | `mart_degree_stats`, `mart_clustering`, `mart_motif_*` (hubs/leaves/brokers/bridges/butterflies), `mart_wedge_closure`, `mart_community_sizes`, `mart_community_mixing`, `mart_feature_categories`, `mart_circle_stats`, `mart_dp_*` input tables |
| Association analysis | dbt SQL (`mart_assoc_cells`, `mart_feature_assoc`) | χ²/Cramér's V between discretized features on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Closure analysis | dbt SQL (`mart_wedge_closure`) | P(closed) by wedge-center degree class — the sequence-analog conditional |
| Formal specification | Lean 4 + Mathlib (`lean/EnclaveGraph/Motifs.lean`, `Params.lean`) plus `lake exe checks` | Decidable motif definitions, proof-status ledger, SQL↔Lean agreement on exported observations |
| Private release | Lean 4 + SampCert discrete Laplace over clipped per-node contributions (`lake exe release`) | Node-level ε-DP histograms over the public grids, proved mechanism theorem, exact-rational budget ledger enforced in Lean AND Python |
| Verified sampling | SampCert `UniformSample` + bucketing map, law proved (`sampleCellPMF_law` pattern) (`lake exe sample`) | Samples from released distributions for generation and GOF checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: degree histogram, community sizes + mixing rows, closure probabilities, category/richness/circle histograms, motif rates, schema constraints |
| Generation | `lake exe generate` renders schema-valid GraphML; planted motif recall 1.0; kernel-checked certificate | ✅ validate PASS locally AND from the bundle (recall 1.0 ×3, degree TV ≈ 0.08, 0 Lean disagreements) plus a REPLICATE verdict over 8 runs: per-node closure mean ≈ 0.54 (margin ≈ 0.08 above the band edge) and affiliation group density mean error ≈ 0.046 (limit 0.08); `certificate.json` kernel-checked (`by decide`); generator runs 200–100,000 nodes, fidelity tolerances validated for 500–10,000 |
| Validation | re-ingest through production dbt + Lean checks; pytest per-column | Round-trip motif recall, GOF vs released weights, clustering-coefficient band, kernel-checked certificate |

Supporting infrastructure: the simplified `Makefile` (see **Build and run
contract** below), a marimo EDA notebook and a synth-vs-real comparison
notebook (validated headless), and the persistent **LLM-Wiki** under `wiki/`
(Karpathy-style: index, append-only log, schema, interlinked pages — updated
on every ingest and incident).

## Build and run contract (simplified)

Exactly six user-facing flows; everything else is an internal dependency:

| Target | What it does |
| --- | --- |
| `make clean` | remove ALL generated data (warehouse, models, Lean build outputs & exported data, release, bundle, synth, reports, caches); keeps source data, code and the shared Lean store |
| `make train` | the entire enclave flow: ingest + dbt build → minimodel training → ml scores → Lean build + checks + DP release + verified-sampler smoke → distill → policy-filtered export → **one source-only bundle** in `dist/bundle/` |
| `make synth` | synthetic GraphML from the local working tree (Lean generator from the local genspec) into `synth/`, then round-trip validation |
| `make synth_from_bundle` | build the generator **from the bundle's Lean sources** (kernel re-checks every theorem), distill the bundle's release, generate into `synth_bundle/` — proves the bundle is self-contained |
| `make notebook-enclave` | serve the enclave EDA marimo notebook |
| `make notebook-synth` | serve the synth-vs-real comparison notebook |

`make help` prints this flow summary.

**Lean toolchain policy**: `LEAN_HOME=/opt/data/lean` always. elan lives in
`$LEAN_HOME/elan`, the mathlib olean cache in `$LEAN_HOME/cache`, and the
pinned package store (mathlib, SampCert) is shared at `$LEAN_HOME/packages`
and symlinked into `lean/.lake/packages` — installed once, reused by every
build including `synth_from_bundle`. No per-project re-download, no reliance
on `~/.elan`.

**Deployment policy**: source-only. There is exactly ONE bundle format
(`dist/bundle/`): Lean sources verbatim + `dp_release.json` + `distill.py`
(stdlib-only post-processing) + `build_from_source.sh` + manifest. No
compiled binaries ship; the consumer's Lean kernel re-checks every theorem
before anything runs. Binary deploy and alternative bundle layouts are
explicitly discarded. The only data that feeds synthesis is Lean-readable
JSON derived from the ε-DP release — never model weights, never the raw
GraphML.

## Enclave Boundary

Keep the following inside the enclave by default:

- The raw GraphML (`facebook_full_features.graphml`), node ids, individual
  edges (who is friends with whom), feature-instance ids, circle rosters,
  and any per-node adjacency list.
- Per-node derived rows, motif hits, community labels
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Per-node embeddings and predictions; training checkpoints and fitted
  parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required, exact-
fraction ε budget). The three exit categories from METHOD §6 apply:

1. **ε-DP histograms** over the public grids (degree bins, clustering
   classes, triangle bins, richness bins, category presence, circle bins,
   community sizes, community mixing, wedge closure, motif presence),
   noised by a Lean executable using SampCert discrete Laplace, with the
   mechanism theorem proved and the ledger enforced in both Python and Lean.
2. **Suppression-thresholded aggregates** (no formal guarantee — declared as
   such): degree-conditioned category-presence rows and per-motif summary
   stats with min-count suppression (≥ 50 nodes per cell), released so the
   generator can condition features on degree realistically.
3. **Common-knowledge metadata**: the category taxonomy, the class-bin
   edges, the GraphML schema and feature-key shape convention
   (`ego_<e>__f<i>_<slug>`), the published dataset statistics (4,039 nodes,
   88,234 edges, avg clustering ≈ 0.6055, 10 ego networks — all public on
   the SNAP site), and the ego ids (published as file names in the public
   dataset). Feature-instance ids, circle rosters and node ids are
   structurally excluded regardless of popularity.

**Privacy unit (the critical design decision — METHOD §6 lessons applied,
plus the graph-specific subtlety):**

- One protected entity = **one person (node)**: their derived record — degree
  class, clustering class, triangle class, feature categories, circle
  counts, community cell vector, wedge vector — is one record in SampCert's
  add/remove-one `Neighbour` relation.
- **The claim is row-level DP on the per-node derived table, stated
  precisely.** In a graph, removing a person also perturbs their neighbors'
  derived rows (degrees drop, triangles vanish). Full node-DP would need the
  mechanism's sensitivity scaled by (1 + degree cap) — with a measured max
  degree of 1,045 that destroys utility. We therefore claim ε-DP w.r.t.
  add/remove of ONE ROW of the derived table (the person's own
  contributions), and declare the neighbor-induced effect as an assumption
  (trust boundary 2: "DP holds for whatever counts are fed in"). This is
  documented, tested for the stated sensitivity, and listed as the main
  formal caveat — not hidden.
- **n = 4,039 entities** is workable but 7× smaller than foursquare's: SNR
  `share × n × ε_total / k` ≈ 25 for a 10%-share cell on the 45-cell mixing
  grid at ε_cell = 2/100. The heavy tail (one node touches 1,045 edges,
  30,025 triangles) makes **clipping mandatory**: per-node cell vectors over
  edge- and wedge-derived grids are allocated by proportional L1 over a
  total budget `dp_clip_total = 8` (largest-remainder rounding in SQL), so
  per-cell sensitivity equals the total and the proved mechanism applies
  unchanged. Pure per-node grids (degree bin, clustering class, richness,
  circle bins) need no clipping: one node = one cell, sensitivity 1.
  Community-size histogram: released NODE-WEIGHTED (each row contributes its
  community's size bin once — sensitivity 1; community counts are recovered
  in distillation via public bin midpoints). A community-count histogram —
  15 events against Laplace scale 50 — was measured to be pure noise and
  replaced (the stock-migration small-n lesson, re-learned). Phase 6
  REQUIRES the pre-noise simulation
  (clipped counts vs raw distribution, SNR per cell) before the ledger is
  committed.
- **Semantics**: equal per-node budgets estimate the per-person-average
  structure, not the edge-weighted one — they differ here (median degree 25,
  max 1,045). This is the RIGHT semantics: the release describes the typical
  person's neighborhood; the degree histogram carries the tail separately.
- **Rank labels for communities**: communities are data-derived, so they are
  released only as rank labels (`community_1..8` by size + `rest`); the
  rank→member map never leaves. Synthetic nodes carry fresh ids (`0..n-1`);
  synthetic feature instances are generated ids matching the public shape
  convention.
- **Cross-node microstructure** (who is friends with whom, shared-feature
  identity, circle rosters): protected relations, never released, declared
  out of scope for v1. The release carries only grid-level aggregates.

**Planned ε ledger** (exact fractions; enforced in Lean AND Python; refusal
path tested; final numbers fixed by the phase-6 pre-noise simulation):
degree bins 8 × (5/100) = 40/100; clustering classes 7 × (5/100) = 35/100;
triangle bins 6 × (2/100) = 12/100; richness bins 9 × (4/100) = 36/100;
category presence 11 × (4/100) = 44/100; circle bins 5 × (4/100) = 20/100;
community sizes 6 × (4/100) = 24/100; community mixing 45 × (10/100) =
450/100 (mixing was 2/100 per cell, 90/100, until the 2026-11-03 fidelity-first
review); wedge closure 16 × (3/100) = 48/100; motif presence 6 × (2/100) =
12/100; affiliation key-group sizes over nine categories 36 × (18/100) =
648/100. Total 1369/100 ≤ 14/1 (cap raised from 4/1 by explicit user decision:
fidelity over privacy; the bound is weak — see wiki/snap-contract §4), nearly
saturated: a new query must displace an old one at review time. `lake exe release` refuses exports over budget
(exit 2). The global clustering coefficient is NOT a separate ledger entry:
it is derivable from the released closure grid (post-processing).

Suppression thresholds alone carry no formal guarantee; use them only for
artifacts explicitly marked non-DP (degree-conditioned category rows, motif
summaries). DP noise randomness comes from a non-replayable source
(urandom-backed FFI); DP outputs are excluded from bitwise reproducibility
by design.

Require attestation and controlled key release, encrypted persistent storage,
restricted egress, and logs that redact node ids (`<node>`) and feature
instances (`<feat>`). CPU mini-models suffice at this scale (4k nodes); GPU
training stays outside the enclave threat model.

## Implementation Guidelines

### 0. Understand the data before modelling it — ✅ done (see `wiki/dataset-snap.md`)

Measured profile: 4,039 nodes, 88,234 edges, connected (1 component); degree
median 25, mean 43.7, p90 113, p95 154, p99 200, max 1,045; avg clustering
0.6055 (matches the published value), transitivity 0.5192, 1,612,010
triangles over 9,314,849 wedges; Louvain (seed 7): 15 communities,
modularity 0.835, sizes 548…19, within-community edge fraction 0.9615;
2,280 feature keys in 14 raw categories (locale on 3,981 nodes, gender
3,955, education 3,013, location 1,659, birthday 1,577, work 1,461 …
religion 2); median 9 set features and 4–5 categories per node, 8 nodes
with none; circle memberships 0–7+ per node (1,155 with none); 118 nodes
belong to ≥ 2 ego networks; **no temporal fields** (time-looking attrs are
binary anonymized flags). Every threshold above was picked from these
measured distributions, not defaults.

### 1. Define Concepts Before Fitting Models

Version every threshold (category taxonomy, class-bin edges, hub ≥ 150,
broker degree ≥ 17 ∧ clustering < 1/5, bridge ≥ 3 communities, butterfly ≥ 3
circles, suppression ≥ 50 nodes, `dp_clip_total = 8`, Louvain seed) as dbt
`vars:`, mirrored in `python/spec.py` and `lean/EnclaveGraph/Params.lean`,
locked by a sync test; bump `spec_version` on any change. Thresholds as
exact rationals in Lean (clustering class edges 1/5, 2/5, 3/5, 4/5; broker
cut 1/5). The category taxonomy is part of the spec (a dbt seed, a Python
dict, a Lean function — one sync test over all three). Ingest needed one
repair already (isolated alters present in `.feat` but absent from `.edges`
— fixed in the converter, kept as a quality flag); `q_data_integrity`
asserts no self-loops, canonical src<dst, symmetric closure, and
feature-node referential integrity.

### 2. Start with Deterministic Baselines

SQL motif detection and empirical distributions first (triangles via wedge
joins, closure grid, community mixing, category histograms) — cross-checked
against networkx values computed at ingest (a measured SQL↔python agreement,
then SQL↔Lean at export). Mini-models are added on top and gated: DegreeSAGE
vs majority class, CommunityGNN vs seeded-Louvain ARI baseline, TriadVGAE vs
common-neighbor-count AUC, FeatCondEncoder vs marginal rates, EgoMotifPool
vs marginal motif rates. Gates are data-dependent — read adoption flags from
artifacts, not docs. Keep honest negatives: with transitivity 0.52, the
common-neighbor baseline for link prediction is strong; VGAE may well lose.

### 3. Separate Training from Inference

`make train` produces content-addressed immutable artifacts
(`sha256(name, data, hparams, code)`) under `models_out/`,
registry-activated; the inference node `ml_graph_scores` materializes
prediction tables. Rebuilding a mart never retrains.

### 4. Preserve Uncertainty and Provenance

Stable `node_id`, content-hash fingerprint (sha256 of the GraphML), model
version and confidence on every prediction, quality flags, `untrained`
placeholder supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

The 10 ego networks cover only part of each person's true friendship list:
the graph is a UNION of ego neighborhoods, so "degree" means degree within
the published crawl, not real-world friend count — a censoring statement
recorded in the release metadata. 8 nodes carry zero features and 1,155
belong to no circle; flag, don't drop. Feature absence means "not observed
in this crawl", never "false" (the sparse-encoding convention from ingest).

### 6. Prevent Leakage

Node-level 80/20 split (split on node id hash, fixed before fitting); edge
split for VGAE leaks no test edges into message passing (PyG
`RandomLinkSplit` discipline); scalers fit on train only; weak labels from
the public taxonomy, never from evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

```text
P(degree class | neighborhood structure)   — DegreeSAGE; released: marginal histogram
P(community pair | edge)                   — empirical mixing matrix (DP, rank-labeled)
P(closed | wedge-center degree class)      — TriadVGAE teacher; released closure grid
P(category present | degree class)         — FeatCondEncoder; suppressed conditional rows
P(motif | node)                            — EgoMotifPool; released motif rates
```

### 9. Make Execution Reproducible

Data fingerprints, code versions, seeds, hyperparameters, split definitions
in `models_out/<v>/meta.json`; immutable artifact paths. DP outputs excluded
by design. SampCert pinned at `a2c53bb`; its toolchain dictates the
Lean/Mathlib versions; the toolchain and package store live in
`LEAN_HOME=/opt/data/lean` and are reused, never re-downloaded per project.

### 10. Keep Formal Claims Precise

Distinguish: proved in Lean (mechanism DP under the stated row-level unit,
sampler law, motif decidability, generator validity-by-construction) /
checked on this snapshot (`lake exe checks` SQL↔Lean agreement) /
statistical estimates (model metrics) / assumed (entropy source, I/O
wrappers, the SQL computing the clipped counts, and the row-level ↔ node-DP
gap — stated in the Enclave Boundary section). State trust boundary 2
explicitly, as the reference implementation does.

### 11. Use Verified Randomness

DP noise and all generation-time draws only through SampCert-backed Lean
executables (`lake exe release`, `lake exe sample`, `lake exe generate`).
Never ad-hoc floating-point noise.

## Synthetic generation contract (planned)

- `python/distill.py` reads ONLY `dp_release.json` + released artifacts
  (post-processing theorem; `source.dp_release_sha256` recorded). Teachers
  map to spec constructs: DegreeSAGE → DP degree histogram; CommunityGNN →
  DP community sizes + mixing rows; TriadVGAE → DP closure grid;
  FeatCondEncoder → DP category/richness histograms (+ suppressed
  degree-conditioned rows); EgoMotifPool → DP motif rates. Public structural
  priors (e.g. zeroing negative noised counts, a minimum-1 diagonal on the
  mixing matrix) are declared `derived_from_data: false`.
- `lake exe generate <genspec> <n_nodes> <out>` renders GraphML directly:
  draw community sizes from the released histogram until ≈ n nodes; per
  node, draw a target degree class then a degree within it; place edges by
  degree-weighted stub matching routed within/between communities via the
  verified sampler over the DP mixing rows; run a triadic-closure pass that
  closes sampled wedges with the released per-degree-class probabilities
  (driving clustering toward the released level); draw feature-richness
  then categories per node and instantiate fresh feature ids matching the
  public shape; assign circles inside communities (declared structural
  prior). **Validity by construction**: a `mkEdge` / `mkNode` smart
  constructor with theorems — canonical src < dst, no self-loops, no
  duplicate edges, degrees ≥ 1, feature keys well-formed
  (`mkGraph_valid`, edge-list sortedness via `List.Pairwise`).
- Plant the rarest motifs (one broker, one social-butterfly) plus a hub in
  `plan.json`; round-trip re-ingestion through the production dbt pipeline
  must recall all planted motifs at 1.0 and `lake exe checks` must
  re-verify (exit 0 / 2 soft, 1 hard).
- Per-column tests: every output property matches its declared distribution
  (χ² GOF vs released weights), exhibits its declared property (clustering
  coefficient within a declared band of the released closure level), or
  satisfies a structural invariant; gaps declared in `meta.json
  assumptions.fields_not_modelled` (initially: degree–community dependence,
  feature–feature identity correlation beyond category level for non-affiliation
  categories, circle roster
  structure, inter-ego overlap, degree assortativity).
- The generator's ONLY inputs are the genspec JSON and the Lean sources —
  synthesis works identically from the working tree (`make synth`) and from
  the source-only bundle (`make synth_from_bundle`).

## Acceptance checklist (status as built, 2026-11-03)

| Area | Criterion | Status |
| --- | --- | --- |
| Data profiling | Measured profile in wiki; thresholds from measurements | ✅ `wiki/dataset-snap.md`; published clustering 0.6055 reproduced exactly |
| Spec sync | dbt vars = spec.py = Params.lean (+ category taxonomy ×3), test green | ✅ 38 vars + 11-category taxonomy + rank labels locked across all three mirrors |
| dbt pipeline | staging/features/marts/ml build + schema tests green | ✅ 141 PASS (35 SQL models + 105 tests + seed) + `ml_graph_scores` on the active artifact; SQL↔networkx cross-checks exact |
| Minimodels | Each gated vs frozen baseline; content-addressed; honest negatives kept | ✅ nine models: degree_sage 58.9% vs majority 19.2%; community_gnn ARI 0.348; featcond mean AUC 0.674; ego_motif_pool macro-F1 0.422; **triad_vgae honest negative** (AUC 0.945 < common-neighbor 0.990, kept unadopted); affiliation set (2026-11-03): affil_ae BCE 0.018 vs marginal 0.035, affil_link_mlp AUC 0.951 vs total-shared ranker 0.935, key_community_mlp 74.1% vs majority 17.8%, key_cohesion_mlp MAE 0.126 vs 0.355 | (artifact e999a0ae31fd, retrained on spec 1.2.0; verdicts unchanged)
| Lean mirrors | `lake exe checks` SQL↔Lean agreement on exported observations | ✅ 24,234 motif + 20,195 class + 8 exact-ℚ χ² checks, **0 disagreements**; 24 claims proved + 4 assumed (incl. the declared row-level unit) |
| Private release | Mechanism theorem on SampCert; exact-rational ledger ≤ 4/1 enforced both sides; over-budget refusal tested | ✅ spent 1369/100 ≤ 14/1 (fidelity-first cap, user decision 2026-11-03; mixing at 5× and nine-category affiliation funded); Lean exit-2 refusal + Python Fraction cross-check both tested; community-size grid re-based to node-weighted sensitivity 1 after the pure-noise incident (see wiki/practices) |
| Verified sampling | Sampler law proved; executable draws pass χ² GOF | ✅ `sampleCellPMF_law`; catsample χ² conformance green |
| Distillation | Weight-by-weight provenance test green; fidelity under the pre-declared criterion | ✅ provenance cell-exact (marginals = release; cat_-prefix strip, mixing symmetric expansion and csize and affiliation public-midpoint division all declared `derived_from_data: false`); fidelity accepted (released-vs-exact degree TV 0.016) |
| Generation | `lake exe generate` renders schema-valid GraphML; planted motif recall 1.0; kernel-checked certificate | ✅ validate PASS exit 0 locally AND from the bundle (recall 1.0/1.0/1.0 ×2, degree TV 0.087/0.093, per-node closure 0.516/0.503 vs released 0.567, 0 Lean disagreements); `certificate.json` kernel-checked (`by decide`, no native_decide) |
| Simplified ops | the six make targets + `make help`; `LEAN_HOME=/opt/data/lean` reused; single source-only bundle; `synth_from_bundle` green | ✅ six flows wired; shared elan/cache/package store reused; one 172K source-only bundle; bundle rebuild re-checks all theorems |
| Wiki discipline | Pages + append-only log updated on every ingest/incident | ✅ dataset-snap, migration-plan-snap, snap-contract, snap-pipeline, practices incident, log entries |

Full pytest suite: 107 passed, 0 skipped, 0 failed. Realism audit (graded,
enclave-only; means over 8 runs, real in parentheses): shared-key edge fraction
0.86 (0.98; was 0.34 before the fidelity-first release), transitivity 0.531
(0.519), modularity 0.840 (0.835), average clustering 0.530 (0.606), edges 86.6k
(88.2k), degree TV ≈ 0.08; honest residuals: heavy degree tail beyond the
released 65+ cell (share ≥ 150: 0.002 vs 0.053), leaf share, group nesting,
degree assortativity.

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible social-graph encoder plus a
verified synthetic-graph generator with an explicit output contract.

- Enclave execution protects processing within the selected platform's threat
  model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit (one person's derived row; the row-level ↔ node-DP gap
  is declared, not hidden) and assumptions. With n = 4,039 nodes, utility is
  measured (pre-noise SNR simulation, released-vs-clipped comparison), not
  assumed.
- Statistical evaluation establishes empirical usefulness.

None of these alone guarantees realism of future synthetic graphs. Released
non-DP artifacts (suppressed conditional rows, motif summaries) and
evaluation metrics remain outside the DP guarantee. Degree–community
dependence, feature identity structure for non-affiliation categories, circle
rosters and assortativity are declared gaps until a ledger entry covers them
(nine affiliation categories are covered since 2026-11-03; first_name/other are not).

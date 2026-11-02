# Synthetic graph data: encodings, privacy, and how it fits the inspectable-artifact idea

Date: 2026-11-01. Builds on `inspectable_synthetic_data_artifact_proposal.docx` (§4.3 graphs) and `synthetic-data-existing-solutions.md`.

Verification: references below are cited from memory and were **not** re-checked against the papers. Verify before relying on any claim, and never treat this note as a privacy audit.

Test bed: SNAP ego-Facebook (`snap/`): 4,039 nodes, 88,234 undirected edges, mean degree 43.7, max 1,045; average clustering about 0.61; one connected component; 1,406 anonymized binary node features (see `snap/inside/notebooks/01_eda.ipynb`).

## 1. The problem

Goal: release a **faithful synthetic attributed graph** (structure plus node attributes) from a graph that contains personal relationships, without releasing the original, and with a stated privacy guarantee.

The proposal's central idea carries over: split **learn** from **generate**, and make the learned representation a portable, inspectable **artifact**. For graphs the artifact must describe a graph-generating process. A table of adjacency entries is the dataset itself and defeats the purpose.

**Design rule: the artifact holds only aggregate parameters, nothing per node.** Anything keyed by node (adjacency, per-node embeddings, per-node community labels, a generator that replays training edges) is the original graph in another form. Node identities are re-created at generation time.

## 2. Why removing identifiers and features is not enough

SNAP already relabeled node ids and replaced feature names with "anonymized feature N". That helps but is not a guarantee:

- Backstrom, Dwork, Kleinberg (2007), "Wherefore art thou R3579X?": active and passive attacks re-identify nodes in anonymized social graphs from structure.
- Narayanan and Shmatikov (2009), "De-anonymizing social networks": structural matching re-identified users across Twitter and Flickr.
- In this dataset, node 107 has degree 1,045; degree alone is a fingerprint.

So dropping direct identifiers (names) is a necessary hygiene step, and the guarantee must come from differential privacy on what is measured.

## 3. Privacy definitions for graphs

| Unit | Neighbouring graphs differ by | Notes |
|---|---|---|
| **Edge-DP** | one edge | Practical. Degree histograms have small sensitivity. Protects individual friendships, not a person's presence. |
| **Node-DP** | one node and all its incident edges/attributes | Much stronger, much harder: one node can change up to max-degree edges and many triangles. Needs projection (truncation) or clipping. Related work: Kasiviswanathan et al. 2013; Day, Li, Lyu 2016 on degree distributions. |
| **Attribute-row DP** | one node's attribute vector | Needed if attributes are released too. Pure edge-DP treats attributes as public. |

Chosen for the first pass: **edge-DP on structure plus attribute-row DP on attributes**, composed (see `docs/snap/graph-aim-plan.md` §3). Node-DP is a later pass.

Notes:
- Cross queries (edges between attribute groups) depend on both edges and attributes. Their sensitivity under attribute-row change grows with a node's degree, so a degree cap is required.
- Any data-dependent choice (which attributes to keep, bin edges, degree cap) is part of the mechanism. If it is chosen by looking at the private data, it must be inside the privacy budget or taken from public information.

## 4. Candidate encodings

### 4.1 Inspectable (fit the artifact contract)

| Encoding | Artifact contents | DP precedent | Strengths | Weaknesses |
|---|---|---|---|---|
| **dK-series** (dK-1 degree histogram, dK-2 joint degree matrix) | histograms | Sala et al. 2011 (DP dK-2) | Very readable; direct match to marginal-measurement style | Weak clustering; JDM sensitivity grows with max degree |
| **Chung-Lu** | expected degree sequence | degree histograms under edge-DP (Hay et al. 2009) | Trivial | No clustering |
| **BTER** (Seshadhri, Kolda, Pinar 2012) | degree histogram plus clustering by degree | none standard, but its inputs are simple noisy statistics | Reproduces degree and clustering; readable | Community structure only implicit |
| **(Degree-corrected) SBM** (Karrer and Newman 2011) | block sizes, k by k connection matrix, degree parameters | several DP-SBM works | Very readable | Block labels are per node, so they must not be released; overlapping circles fit poorly |
| **Kronecker** (Leskovec et al. 2010) | small initiator matrix | DP fitting exists | Tiny artifact | Too coarse |
| **Graph-AIM** (this project) | noisy graph statistics plus fitted sampler | (new) | Matches the AIM workload idea; extends naturally to attributes | Needs a sound sensitivity analysis per query |

### 4.2 Less explainable (future passes)

| Encoding | Notes |
|---|---|
| **Latent-space / RDPG / spectral embeddings** (Young and Scheinerman 2007) | Per-node positions. Faithful, but an embedding table is the graph. Only acceptable under a DP release of the embedding (for example DP-SGD training of an encoder, or noisy spectral methods) and a generator that does not expose per-node vectors. |
| **Random-walk generators** (NetGAN, Bojchevski et al. 2018; CELL) | Memorize training edges; high privacy risk without DP. |
| **Autoregressive / VAE / diffusion** (GraphRNN, You et al. 2018; GraphVAE; DiGress, Vignac et al. 2023) | Good fidelity on small graphs; scale poorly; DP-SGD is awkward for structured outputs. |
| **Graph neural encoders + DP-SGD (Opacus)** | The "DP encoder/decoder" idea from the existing-solutions note applied to graphs. |

The user is open to these in later passes. They should be evaluated with the same harness so results are comparable (§6).

## 5. Attributes

Findings on ego-Facebook features:

- 1,406 distinct binary features; each ego network uses its own column order but names are global, so a global node-by-feature matrix can be built (118 nodes appear in more than one ego network; merge with OR).
- Fields: education (925 features), work (620), last_name (194), location (132), birthday (92), hometown (90), first_name (82), languages (81), locale (29), gender (20), middle_name (10), name (3), religion (1), political (1).
- `first_name`, `middle_name`, `last_name`, `name` are direct identifiers even when the values are anonymized: **drop them**. Treat birthday, hometown and location as quasi-identifiers: coarsen or keep only frequent values.

Representation for generation: a small set of **categorical attributes**, each with a domain of the top-K frequent values plus `other` and `none`. Attribute model: noisy 1-way and a few 2-way marginals, fitted to a joint distribution (PrivBayes/AIM-style). Attribute-structure coupling comes from **edge counts between attribute-group pairs** (homophily), measured with noise.

## 6. Evaluation

- **Structure**: degree CCDF (Kolmogorov-Smirnov, earth mover's distance), clustering (global and by degree), triangle count, largest-component share, shortest-path distribution, assortativity, k-core profile, spectrum (top eigenvalues).
- **Attributes**: 1-way and 2-way marginal error; edge-count homophily matrices between attribute groups.
- **Utility**: downstream tasks on real versus synthetic graphs (for example link prediction and attribute prediction by a simple classifier trained on synthetic, tested on real).
- **Privacy (empirical, never a substitute for the DP statement)**: edge-membership inference, degree-based re-identification, nearest-neighbour edge overlap (fraction of synthetic edges present in the real graph, against a size-matched baseline).
- **Baselines**: Erdos-Renyi, Chung-Lu, SBM, non-DP BTER, and the non-private version of our own method. Plot a **privacy-utility curve over epsilon**.

## 7. Open questions

- Sound sensitivity bounds for triangle- and clustering-based queries under edge adjacency (common neighbours per edge can be large; needs clipping or smooth sensitivity).
- A stable degree cap that keeps edge-neighbouring relations intact (naive truncation is not stable; ordering incident edges by a fixed hash is one candidate).
- How to choose attribute domains (top-K values) without leaking: public configuration, a DP top-K, or a share of the budget.
- Best sampler given noisy statistics: BTER-style, a dK-2 rewiring, or a fitted exponential random graph model.
- Whether ego-network structure (10 egos around 4,039 nodes) biases the global statistics we try to preserve.
- Node-DP: which projection and which queries remain useful at max degree 1,045.
- Moving from a purely inspectable artifact to a learned encoder: what can still be exported and audited?

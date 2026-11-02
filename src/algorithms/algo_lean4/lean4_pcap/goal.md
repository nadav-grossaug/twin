# torch-lean-arch: Enclave Flow Encoder and synthetic data generator

## Goal

Build a reproducible encoder that converts private network flow records (Cisco
[Joy](https://github.com/cisco/joy) features extracted from pcap, used for
cyber-defense training, here `flow.jsonl`: 406,938 flows) into a versioned
traffic representation: per-flow features, per-channel repetition statistics,
motifs, and fitted distributions — for cyber-defense training and downstream
synthetic traffic generation. Run the encoder inside an enclave so raw flows and
sensitive intermediate artifacts remain within the protected processing
boundary.

Here, **enclave** means an attested trusted execution environment. The
particular enclave platform has not yet been selected.

The encoder learns and measures data behavior from the input dataset,
producing an explicit specification that a later synthetic-data generator can
target.

This is the third instantiation of the dataset-agnostic method (see
`METHOD.md`, the replication guide — especially §12's substitution map; first
migration was the stock series, second the Foursquare travel corpus: see
`../lean4_foursquare` and `wiki/migration-plan.md`). The five-layer
architecture, three release categories, spec-mirror discipline, Lean-only
generator and source-only bundle discipline are unchanged; the domain layer
substitutes flows for transfers.

### Outputs

- Per-flow and temporal features (volumes, rates, durations, entropy, service
  taxonomy, TTLs, quality flags).
- Observed channels (sa, da, dp) and their repetition/timing statistics.
- Traffic motifs: **beacons** (periodic re-contact), **fan-out/scan** shapes
  (including labeled false positives), **return-trip motifs** `A → B → A`
  over destination sequences, and **TLS motifs**: non-canonical handshake
  shapes and rare client stacks.
- TLS packet structure and fingerprinting: per-flow JA3-style client-stack
  fingerprints (hello version + cipher suites + extensions) and the full TLS
  record sequence (type/direction/size/timing — the shape that survives
  encryption), with population stats per stack and per handshake shape.
- DNS name structure: per-query qname wire shape (labels, RFC 1035 limits,
  character makeup), registered-domain populations with the subdomain-fan
  motif (DNS tunneling/DGA/CDN shape), and a char-LM name-shape anomaly score.
- **χ² co-occurrence analysis** (`feat_flow_cats`, `mart_assoc_cells`,
  `mart_feature_assoc`): pairwise χ²/Cramér's V between discretized features
  and motif flags on full contingency grids — which features co-occur with
  which motifs; Lean recomputes χ² exactly over ℚ.
- **Sequence analysis** (`mart_seq_transitions`, `mart_seq_chains`,
  `mart_seq_sessions`): timely flow logic — the service transition matrix
  with lift, resolve→connect chains, and maximal timely sessions broken at
  recording gaps.
- Conditional service and anomaly scores from versioned mini-models.
- Versioned mini-models, evaluation results, and candidate formal
  specifications.
- Differentially private releases of aggregate distributions. The noise
  mechanism is formally verified with
  [SampCert](https://github.com/leanprover/SampCert), AWS's Lean 4 library of
  verified samplers and differential-privacy (DP) proofs.
- A formally verified sampler for drawing from released distributions, built
  on SampCert's samplers. It is the entry point for the downstream generator.

Synthetic generation is in scope: `lake exe generate` renders Joy-schema flows
from the distilled genspec alone (every draw through the verified sampler),
validates them by re-ingestion through the production pipeline and the Lean
deciders, and emits a kernel-checked certificate (`make synth`, `make
certify`).

## Processing Architecture (implemented)

| Stage | Implementation | Output |
| --- | --- | --- |
| Ingestion | DuckDB `read_json_auto` in dbt staging (`stg_flows`) | Canonical flow records with stable `flow_id` and provenance `flow_key`; Joy config header excluded |
| Cleaning and features | dbt + DuckDB (`feat_flows`, `feat_pairs`, `feat_tls`, `feat_dns`) | Durations (clamped), volumes, rates, capped entropy, internal/external, public service taxonomy, per-channel gap statistics; TLS ClientHello fingerprints and record-structure features; DNS qname wire-shape features |
| Enrichment | Offline IP→country (geoip2fast) in the notebook | Country context and MapLibre map (visualization only; approximate for 2008 IPs) |
| Mini-model fitting | PyTorch (`python/train_minimodels.py`), CUDA/MPS/CPU | FlowAutoencoder (anomaly/embedding), ByteDistClassifier (service from byte histogram), NextServiceGRU (next service from flow history), TlsRecordGRU (client stack from TLS record structure) and DnsCharLM (query-name perplexity); versioned artifacts in `models_out/` |
| Inference | Separate dbt Python node (`ml_flow_scores`) | Per-flow predictions with confidence and model version; never retrains on rebuild |
| marts | dbt SQL | `mart_flow_stats`, `mart_motif_beacon`, `mart_motif_fanout`, `mart_motif_return`, `mart_dns`, `mart_dns_domains`, `mart_dp_counts`, `mart_dp_dns`, `mart_dp_tls`, `mart_dp_transitions`, `mart_tls_fingerprints`, `mart_tls_records`, `mart_tls_sni_topk` |
| Association analysis | dbt SQL (`feat_flow_cats`, `mart_assoc_cells`, `mart_feature_assoc`) | Pairwise χ²/Cramér's V between features and motif flags on full contingency grids; Lean recomputes χ² exactly over ℚ |
| Sequence analysis | dbt SQL (`mart_seq_transitions`, `mart_seq_chains`, `mart_seq_sessions`) | Service transition matrix with lift, resolve→connect chains, maximal timely sessions |
| Distribution fitting | Empirical estimators (per-service stats, gap distributions) | Conditional summaries; PyTorch reserved for where it beats them |
| Formal specification and evaluation | Lean 4 + Mathlib (`lean/EnclaveFlows/Motifs.lean`, `Tls.lean`, `Dns.lean`) plus the `lake exe checks` executable | Decidable motif definitions (incl. TLS canonical handshake/rare stack and DNS RFC shape/subdomain fan), proof status ledger, SQL↔Lean agreement on exported observations |
| Private release | Lean 4 + SampCert: discrete Laplace mechanism (`privNoisedQueryPure` instantiated as `privClippedCell`) over clipped per-host counts, executed through SampCert's FFI (`lake exe release`) | Host-level ε-DP histograms over the public service grid + aux (svc_transition, tls_stack, dns_shape), with the proved theorem `privClippedCell_PureDP` and an enforced budget ledger |
| Verified sampling | Lean 4 + SampCert: `UniformSample` composed with a bucketing map; exact law proved (`sampleCellPMF_law`) (`lake exe sample`) | Samples from the released distribution for downstream generation and goodness-of-fit checks |
| Distillation | `python/distill.py` reading ONLY released artifacts | `genspec.json`: restricted typed spec — services, transitions (with a public self-transition prior), TLS stacks, DNS shape cells, top-k qnames/SNI, schema constraints |
| Generation | `lake exe generate` (Lean, genspec-driven) | `flow_synth.jsonl` in the Joy schema + `plan.json` ground truth + `meta.json` provenance; validity by construction (`mkFlow_valid`) |

Supporting infrastructure: the `Makefile` (six user-facing flows, see
`HOWTO_run.md`), the marimo notebooks `visualize_enclave_mo.py` /
`visualize_synth_mo.py` (EDA, features, stats, motifs, Lean results,
IP→country MapLibre map; synth-vs-real scorecard), and a persistent
**LLM-Wiki** under `wiki/` (Karpathy-style: index, append-only log, schema,
interlinked pages).

## Build and run contract (simplified)

Exactly six user-facing flows; everything else is an internal dependency:

| Flow | What it does |
| --- | --- |
| `make clean` | remove ALL generated data (warehouse, models, Lean build outputs & exported data, release, bundle, synth, reports, caches); keeps source data, code and the shared Lean store |
| `make train` | the entire enclave flow: dbt build → minimodel training → ml scores → Lean build + checks + DP release + verified-sampler smoke → distill → tests → manifest → policy-filtered export → **one source-only bundle** in `dist/bundle/` |
| `make synth` | synthetic Joy JSONL from the local working tree (Lean generator from the local genspec) into `synth/`, then round-trip validation (`SYNTH_HOSTS=20`) |
| `make synth_from_bundle` | build the generator **from the bundle's Lean sources** (kernel re-checks every theorem), distill the bundle's release, generate into `synth_bundle/` — proves the bundle is self-contained |
| `make notebook-enclave` | serve the enclave EDA marimo notebook |
| `make notebook-synth` | serve the synth-vs-real scorecard notebook |

Ops briefs (unchanged from the previous migration): `LEAN_HOME=/opt/data/lean`
shared toolchain/package store (elan + mathlib cache + pinned packages,
symlinked into `lean/.lake/packages` — no per-project re-download, no reliance
on `~/.elan`); **source-only synth and deploy** — no compiled binaries leave
the enclave, exactly one bundle (`dist/bundle/`: Lean sources verbatim +
`dp_release.json` + top-k JSONs + `distill.py` (stdlib-only post-processing)
+ `build_from_source.sh` + manifest); synth consumes **Lean sources + released
JSON only**. `make help` prints the summary.

## Enclave Boundary

Keep the following inside the enclave by default:

- Raw records (`flow.jsonl`) and host identifiers.
- Individual features, protocols, motifs, connections
  (`artifacts/enclave.duckdb`, `lean/data/*.json`).
- Host-level embeddings and predictions.
- Training checkpoints and fitted parameters until reviewed for release.

Only export artifacts allowed by an explicit release policy
(`config/release_policy.yaml`: default deny, attestation required,
exact-fraction ε ledger). Three exit categories: (1) ε-DP histograms with the
machine-checked mechanism, (2) suppression-thresholded aggregates (top-k
qnames/SNI — no formal guarantee, declared non-DP), (3) common-knowledge
metadata (public service taxonomy, RFC 1035 limits, address-class structure).

Release aggregate distributions through a differentially private mechanism
whose privacy is proved in Lean on top of SampCert:

- **Privacy unit.** One user = one MONITORED (internal) host: each flow is
  attributed to exactly one internal endpoint (sa when internal, else da), so
  external responders are counted through the monitored host they talked to,
  never as users. Each user is one record in SampCert's add/remove-one
  `Neighbour` relation.
- **Bounded contributions.** PROPORTIONAL (L1): each host's cell vector is
  its own traffic mix allocated over a total budget `dp_clip_total = 128`
  flows (largest-remainder rounding); per-cell sensitivity is proved
  (`clippedSum_sensitivity`). Rationale: uniform per-cell clipping at 4
  saturated every major class for the 15 internal hosts while 2,433 external
  one-flow sources dominated the tail — the released mix came out inverted;
  proportional allocation preserves each host's mix shape.
- **Public grid.** The 7-class generator service taxonomy (genServices) is
  fixed in SQL, independent of the data — plus aux grids: 6 TLS client-stack
  RANKS (the rank→fingerprint map stays in-enclave), 9 DNS name-shape cells
  (no qname string reaches the mechanism) and the 49-cell `svc_transition`
  grid.
- **Budget.** Exact-rational ε ledger, enforced in Lean AND Python: service
  + tls_stack + dns_shape + svc_transition compose to 389/100 ≤ 4/1 per
  release (measured spend 3.89); `lake exe release` refuses over-budget
  exports (exit 2; tested both sides).

Suppression thresholds alone carry no formal guarantee. Use them only for
artifacts that are explicitly marked as non-DP. Randomness for DP noise must
come from a non-replayable source inside the enclave (SampCert's urandom FFI).
DP outputs are therefore excluded from bitwise reproducibility.

Require attestation and controlled key release, encrypted persistent storage,
restricted egress, and logs that exclude sensitive observations. Confirm that
the selected platform supports the required Python stack, memory footprint,
and execution model. Start with CPU mini-models; do not assume GPU processing
remains inside the protected boundary (MPS/CUDA are used for development
training outside the enclave threat model).

## Implementation Guidelines

### 1. Define Concepts Before Fitting Models

Specify what counts as a motif, sequence, connection, and host feature;
version thresholds, units, timestamp interpretation, conventions, and
missing-data handling. Implemented: public thresholds as dbt `vars:`
(spec_version 1.3.0), mirrored in `python/spec.py` and
`lean/EnclaveFlows/Params.lean`, locked by a sync test; Joy quirks documented
and flagged (negative durations clamped, entropy capped at 8, zero-byte and
unanswered flows flagged — see `wiki/data-quality.md`).

### 2. Start with Deterministic Baselines

Implemented: SQL motif detection (beacon CV rule, fan-out counts, return-trip
window), transition counts, and empirical distributions. Mini-models are added
on top and compared against baselines (classifier vs majority baseline).

### 3. Separate Training from Inference

Training (`make train`) produces content-addressed immutable artifacts under
`models_out/<version>/` with a registry (FLOW_RETRAIN gating). The inference
node `ml_flow_scores` materializes prediction tables that downstream SQL
references through `dbt.ref()`. Rebuilding a mart does not retrain any model.

### 4. Preserve Uncertainty and Provenance

Implemented: stable `flow_id`, content-hash `flow_key`, data fingerprint
(sha256 of `flow.jsonl`), model version and confidence on every prediction,
quality flags, and an `untrained` placeholder supporting `unknown` outcomes.

### 5. Respect Incomplete Observations

Break destination sequences at recording gaps (session_gap_s = 600). Treat
unanswered flows as censored, not erroneous. Distinguish per-flow sampling
(packets array covers first 50 packets) from elapsed time.

### 6. Prevent Leakage

Time-ordered 80/20 split defined before fitting; scaler fit on train only;
weak labels derived from the public taxonomy, not from evaluation data.

### 8. Fit Dependencies, Not Just Marginal Distributions

Target relationships such as:

```text
P(service_class | byte_dist)            — implemented (ByteDistClassifier)
P(next_service  | history, gaps)        — implemented (NextServiceGRU; and the
                                          empirical transition matrix with lift
                                          in mart_seq_transitions)
P(duration      | service, hour)        — empirical per-service stats
```

### 9. Make Execution Reproducible

Record data fingerprints, code versions, dependency versions, seeds,
hyperparameters, and split definitions (all in `models_out/<v>/meta.json`).
Use immutable artifact paths and explicit retraining rules. Attestation alone
does not establish reproducibility. SampCert is pinned to commit `a2c53bb`
(and its Lean v4.34.1 + Mathlib); the toolchain and package store live in
`LEAN_HOME=/opt/data/lean`, shared across builds.

### 10. Keep Formal Claims Precise

Distinguish:

- Mathematical properties proved in Lean (see ledger below).
- Checks performed on this particular dataset (`lake exe checks` output).
- Statistical estimates and their assumptions (model metrics).
- Learned semantic interpretations (weak labels).

Importing a Python-computed statistic into Lean does not prove that it was
correctly calculated from the raw records.

For privacy and sampling, exactly what is mechanised here:

- **Proved.** `clippedSum_sensitivity` (sensitivity = clip under `Neighbour`);
  `privClippedCell_PureDP` (per-cell pure (ε₁/ε₂)-DP, via SampCert's
  `privNoisedQueryPure_DP`); `sampleCellPMF_law` (the released-histogram
  sampler draws cell i with probability exactly wᵢ/Σw, via SampCert's
  `UniformSample_apply'`); motif decidability and sanity/monotonicity
  theorems; generator contracts (`genBeacon_isBeacon`, `genSession_isTimely`,
  `mkFlow_valid`, `stepTimes_sorted` via `List.Pairwise`).
- **Assumed.** The entropy source (SampCert's FFI reads `/dev/urandom`), the
  unverified I/O wrappers in the executables, and the SQL that computes
  per-user clipped counts. DP holds for whatever counts are fed in, but the
  counts' meaning is not proved.
- **Post-processing.** Clamping noisy counts at 0 (and any downstream
  normalisation) reads only the noisy output, so the DP guarantee is
  preserved. The genspec is post-processing of the release; the bundle
  carries only sources + the release.
- **Utility is empirical.** With 15 monitored internal hosts, dp_clip_total
  128 and 71 composed cells, noise of scale ~20 dominates small cells (chat,
  ntp, other release as 0). Utility is measured and reported in the notebook;
  coarsen grids rather than overstate precision.

### 11. Use Verified Randomness

Draw DP noise and samples for downstream generation only through
SampCert-backed Lean executables (`lake exe release`, `lake exe sample`,
`lake exe generate`). Never use ad-hoc floating-point noise. SampCert's
samplers are exact and discrete, which avoids the floating-point attacks on
naive Laplace implementations. SampCert is pinned to commit
`a2c53bb5ff790e729e252958060699b7eff898b9`.

## Synthetic generation contract (implemented)

- `python/distill.py` reads ONLY `dp_release.json` (+ aux svc_transition /
  tls_stack / dns_shape histograms) and the common-knowledge top-k artifacts;
  NEVER raw data, weights or embeddings; the genspec is hash-keyed to the
  release (post-processing theorem). Teachers map to spec constructs:
  NextServiceGRU → DP transition rows blended with a PUBLIC self-transition
  prior (diagonal boost = max(1, row mass) — bursty traffic is
  self-transitioning, `derived_from_data: false`); TlsRecordGRU → stack-rank
  mix over the canonical-handshake skeleton; DnsCharLM → shape histograms +
  top-k names with a gibberish `.test` shape-fallback.
- `lake exe generate <genspec> <n_hosts> <out>` renders Joy-schema JSONL
  directly: service mix from the released histogram, sessions/beacons with
  proved timely/periodic structure (`genBeacon_isBeacon`,
  `genSession_isTimely`), structural TLS/DNS renders; every draw through
  `sampleCellPMF` (law proved).
- Validation: re-ingestion through the PRODUCTION dbt pipeline (schema
  tests), Lean deciders re-decide the detected structures, planned beacons
  must recall at 1.0 (a theorem), TV distance of the detected clipped service
  histogram vs the released mix within tolerance (exit 0 / 2 soft / 1 hard).
- `plan.json` is the latent ground truth (not the product); `meta.json`
  records every ASSUMPTION vs released fact, including
  `fields_not_modelled` (packets/byte_dist/entropy are null — no payload
  model; IPs from documentation ranges).

## Acceptance Criteria — status (2026-11-02)

| Area | Required evidence | Status |
| --- | --- | --- |
| Data integrity | Parsing, units, ordering, duplicates, gaps, and exclusions documented and checked | ✅ dbt tests green; Joy quirks flagged (`wiki/data-quality.md`); verified repairer for the X509 `},,{` serializer bug (`make repair-data`) |
| Feature extraction | Comparison against a rule baseline and a reviewed sample | ✅ classifier vs majority baseline (0.945 vs 0.521); EDA review |
| Motif extraction | Known sequences produce expected motifs; gaps do not create false transitions | ✅ Lean sanity theorems + SQL↔Lean agreement checks; session-gap breaking |
| Association analysis | χ² co-occurrence between features and motifs; exact Lean recomputation | ✅ `mart_assoc_cells` / `mart_feature_assoc`; Lean χ² over ℚ, SQL↔Lean exact agreement |
| Sequence analysis | Timely flow logic: transitions, chains, sessions | ✅ `mart_seq_*`; `IsTimelySession` decided in Lean; NextServiceGRU vs repeat-last |
| Distribution modeling | Held-out fit outperforms or justifies empirical baselines | ✅ time-ordered held-out metrics in `meta.json` |
| Robustness | Controlled noise/missingness produce expected responses | ✅ perturbation suite with declared expectations, frozen artifacts |
| Formal layer | Lean files compile; verified claims and unproved obligations separately listed | ✅ no `sorry`; ledger in `lean/out/lean_results.json` |
| Enclave operation | Attestation, artifact access, logging, permitted exports tested | ✅ abstraction + SimulatedTEE unit-tested (nonce replay, wrong key/measurement, egress escapes); policy-gated `make export`; platform still unselected |
| Private release | DP theorem compiles on SampCert; budget enforcement tested; executable noise matches the discrete Laplace law; utility reported | ✅ theorem + budget refusal path; utility shown in notebook |
| Verified sampling | Sampler law proved in Lean; executable samples pass goodness-of-fit against the target | ✅ `sampleCellPMF_law`; χ² check in notebook |
| Reproducibility | Same pinned inputs and configuration reproduce results within declared tolerances | ✅ frozen rebuild hash-identical; CPU retrain bit-stable; content-addressed artifacts; DP outputs excluded by design |
| Six ops flows | clean / train / synth / synth_from_bundle / notebook-enclave / notebook-synth | ✅ this migration; `make help` |
| Source-only bundle | No binaries; kernel re-checks proofs at deploy; synth from bundle == synth from local dir | ✅ `dist/bundle/` sources + `build_from_source.sh`; `make synth_from_bundle` |

## Deliverable and Guarantee Boundaries

The deliverable is a protected, reproducible network-flow encoder with an
explicit output contract.

- Enclave execution protects processing within the selected platform's threat
  model.
- Lean supports precisely scoped correctness claims.
- SampCert-based proofs establish ε-DP of the release mechanism under the
  stated privacy unit and assumptions. They do not establish that the released
  numbers are useful.
- Statistical evaluation establishes empirical usefulness.

None of these alone guarantees realism of future synthetic traffic. Released
non-DP artifacts, model weights and evaluation metrics remain outside the DP
guarantee.
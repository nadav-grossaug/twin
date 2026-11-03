---
title: Distillation (teacher → genspec)
type: concept
updated: 2026-11-03
sources: [../python/distill.py, ../lean/EnclaveFlows/GenSpec.lean, ../lean/Generate.lean]
---

# Distillation (teacher → genspec)

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

## Which release is distilled (2026-11-03)

`make train` computes a fresh release in `lean/out/` before `make export`,
so "the latest export" is often the PREVIOUS release. `distill._select_release`:
explicit `$RELEASE_DIR` wins; inside the enclave a live release that differs
from the latest export is distilled (category-2 rows from the
policy-filtered marts, a loud pre-export warning); the bundle (no warehouse)
uses its export. `genspec_meta.source.release_source` records the choice and
validate / tests read the same release. GeoLife genspec also carries
per-mode `path_m` (released median leg path) for length-conditioned mode
choice. See [[geo-pipeline]], [[practices]].

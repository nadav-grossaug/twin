---
title: System architecture
type: component
updated: 2026-10-30
sources: [../goal.md, ../Makefile, ../dbt/dbt_project.yml, ../lean/EnclaveFlows.lean, ../config/release_policy.yaml]
---

# System architecture

The end-to-end shape of the enclave flow encoder: five layers, one direction
of data flow, and a single gated exit. `make all` runs the first four;
`make synth` lives entirely on the public side of the gate.

```
 flow.jsonl (Cisco Joy; repair-data fixes known serializer bugs in place)
    │
 1. dbt + DuckDB            staging → features → marts (+ data tests, unit test)
    │                       artifacts/enclave.duckdb — ALL enclave-only
 2. PyTorch minimodels      content-addressed immutable artifacts, adoption gates
    │                       scores re-enter dbt as the Python node ml_flow_scores
 3. Lean 4 formal layer     re-decides every motif/predicate on exported data;
    │                       proves mechanism theorems (SampCert); 27 claims
 4. DP release gate         lake exe release: discrete Laplace + exact-ε ledger
    │                       + policy artifacts (suppression / common-knowledge)
 5. Distiller + generator   distill.py packages released facts into the TYPED
                            genspec; `lake exe generate` runs ENTIRELY in Lean
                            (verified sampler in-process, records via mkFlow)
```

## Layer responsibilities

- **dbt/DuckDB** ([[dbt-pipeline]]): one staging model over the raw JSONL,
  feature tables (`feat_flows`, `feat_pairs`, `feat_tls`, `feat_dns`,
  `feat_flow_cats`), stats/motif/association/sequence marts, DP input marts
  (`mart_dp_counts`, `mart_dp_tls`, `mart_dp_dns`), common-knowledge top-k
  marts (`mart_dns_topk`, `mart_tls_sni_topk`), and the data-integrity ledger
  `q_data_integrity`. The dbt `vars` block is the **single source of
  thresholds** (the concept spec, versioned — currently 1.2.0).
- **PyTorch** ([[minimodels]]): fifteen deliberately small models trained from
  the warehouse; artifacts are immutable and content-addressed
  (`sha256(name, data, hparams, code)[:12]`), activated explicitly, each with
  a frozen evaluation definition and an `adopted` flag vs a declared non-ML
  baseline. Inference is a dbt Python node, so scores are just another mart.
- **Lean 4 + SampCert** ([[lean-formal-layer]]): mirrors every SQL predicate
  as a decidable definition and re-decides it on exported observations
  (`lake exe checks` — agreement counts are the cross-check, e.g. 60,796/60,796
  TLS canonical verdicts); proves the DP mechanism (`histMech_DP`), the
  sampler law (`sampleCellPMF_law`, [[verified-sampling]]) and the generator
  contracts (`FlowShape.lean`); kernel-checks synthetic-structure certificates
  (`GenCert.lean`, `by decide`).
- **Release gate** ([[dp-release]], [[enclave-boundary]]): `lake exe release`
  noises the clipped histograms (service 7 cells at ε 2/5, proportional L1
  clip 128 per monitored host; aux `tls_stack` 6 and `dns_shape` 9 cells at
  ε 1/25, `svc_transition` 49 cells at ε 1/100) and enforces the exact-rational
  ledger 14/5 + 6/25 + 9/25 + 49/100 = 389/100 ≤ 4/1 — it REFUSES output over budget.
  `python/release.py` applies `config/release_policy.yaml` (default DENY,
  attestation required) and exports the three artifact categories below.
- **Distiller + generator** ([[distillation]], [[synthetic-generation]]):
  `make distill` packages released facts into the restricted typed genspec
  (the minimodels' generative behavior as DP histograms + a semi-Markov state
  machine); `lake exe generate` renders Joy JSONL entirely in Lean — verified
  sampler in-process, `mkFlow_valid` record validity, `stepTimes_sorted`
  ordering, fuel-bounded chains; TLS blocks satisfy `IsCanonicalHandshake` by
  construction; validation re-ingests the output through the production
  pipeline (round-trip audit). PyTorch never ships in the generator.

## The three release categories

Everything that leaves the enclave is one of:

1. **ε-DP histograms** — SampCert discrete Laplace, proved mechanism, exact
   ledger. The only category with a formal privacy guarantee.
2. **Suppression-thresholded aggregates** — `flow_stats`; reduces but does
   not eliminate risk, no formal guarantee, declared as such.
3. **Common-knowledge metadata** — `dns_topk`, `tls_sni_topk`: popular names
   seen from ≥2 hosts, with structurally-private classes excluded outright
   (`*.arpa` encodes internal IPs; single-label names are local hostnames).
   Treated like the public service taxonomy; costs no ε by declaration,
   gated by policy entry + suppression, never by silence.

Everything else is in the policy's `denied` list by name.

## Trust boundaries (what is proved vs assumed)

Proved: mechanism DP bounds, sampler law, motif predicates' properties,
generator contracts, kernel certificates. Assumed (goal.md guideline 10): the
SQL computed the inputs correctly (mitigated by the Lean↔SQL agreement
counts and the pytest Python↔SQL recomputes), SampCert's `/dev/urandom` FFI,
the unverified I/O wrappers, and the weak port-taxonomy labels.

## Sync discipline

One spec, three mirrors: dbt `vars` ↔ `python/spec.py` ↔
`lean/EnclaveFlows/Params.lean`, locked by `tests/test_spec_sync.py`.
Generator constants live once, in `FlowShape.lean`, regex-read by `synth.py`.
Changing a threshold means bumping `spec_version`.

See [[practices]] for the techniques this architecture is built with.

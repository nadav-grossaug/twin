---
title: GeoLife reference implementation (torch-lean-arch 01)
type: reference
updated: 2026-10-26
sources: ["~/tmp/test_torch_lean_arch_01"]
---

# GeoLife reference implementation

Sibling project at `~/tmp/test_torch_lean_arch_01`: the same torch-lean-arch
(dbt+DuckDB → PyTorch minimodels → Lean4+SampCert) on **Microsoft GeoLife GPS
trajectories**, carried through to **synthetic data generation**. Same pins as
ours (Lean v4.34.1, SampCert `a2c53bb`, mathlib transitively via SampCert).
It is substantially more mature; techniques below are the migration source
for [[migration-plan]].

## Technique inventory (by layer)

**Spec discipline** — dbt `vars:` is the single versioned concept spec
(`spec_version` stamped into mart rows); Python reads it (`encoder/spec.py`),
Lean mirrors it (`Params.lean`), and `test_spec_sync.py` regex-parses the Lean
`def`s and asserts equality via an explicit name map. For generator params the
direction flips: `Motion.lean` is master and `synth.py` regex-reads its defs
at import, so simulation can't drift from the proved constants.

**Provenance** — content-addressed immutable artifacts: version =
`sha256(name, data_fingerprint, hparams, code_fingerprint)[:12]`, write-once +
chmod read-only, `registry.json` active-version map, activation only when
empty or `GEOLIFE_RETRAIN=1`; `train_or_reuse` makes dbt train nodes
idempotent. Per-run manifest with per-table `bit_xor(hash(row))` fingerprints
(floats rounded to 6dp). Two-phase `reproduce_check`: frozen rebuild must
hash-match exactly (model floats ≤1e-6); retrain must land on the *same
candidate version id* and metrics ≤1e-6.

**Release gate** — default-deny `config/release_policy.yaml` (relation/column
allowlists, suppression thresholds, audited `denied:` list, DP releases with
fixed public grids and ε as exact fraction strings, `model_weights:
deny_unless_reviewed`); `release.py` enforces attestation (TEE abstraction
with HMAC-signed `SimulatedTEE` test double, nonce anti-replay, measurement =
code fingerprint), EgressGuard path containment, read-only DB, Fraction-exact
budget cap, per-file sha256 manifest. Dev override is an env var *named in
the policy* and permanently marked `attested: false`.

**Lean layer** — beyond our theorems: composed multi-cell DP via `privCompose`
+ `privPostProcess` induction (**`histMech_DP : PureDP … (K·ε₁/ε₂)`** — a
proved total budget, where ours is per-cell + informal ledger); the 8-line
`neighbour_map` transport lemma reusing SampCert sensitivity per projection;
extractor correctness as soundness **and completeness** iff-lemmas over
Mathlib infix (`mem_abaMotifs : … ↔ [a,b,a] <:+: l ∧ a ≠ b`,
`transitions_append_brk` = "gaps create no false transitions");
hand-written `Decidable` instances framed as "the checker *is* the spec; only
the TSV wrapper is unverified"; linking theorems proving generator parameters
imply encoder acceptance (`stayDrift_box_fits`, `trip_interval_le_gap`).

**Synthesis** — `synth.py generate/manipulate/validate/certify`: a latent plan
sampled from the **DP release only** through the verified sampler (every draw
via `catsample`), rendered with physics whose parameters are the Lean motion
defs; `validate` re-ingests through the production parser + dbt-compiled SQL
replay + Lean deciders with tiered exit codes (1 = spec violation, 2 =
declared tolerance miss, 0 = pass); `certify` emits **GenCert.lean** with
`theorem sample_valid : ∀ u ∈ sample, ValidUser u := by decide` and records
the axiom footprint (`kernel_only`) in `certificate.json` — a kernel-checked
data certificate. `realism_audit` (enclave-only) judges synthetic data with
the production models (mode-F1 gap, share above real anomaly p99) + TV
distances; `remote-bundle` ships ~215 KB (release + Lean spec + generator) to
run generation/validation/certification on an untrusted host.

**Testing & ops** — pytest suite (attestation failure modes, Lean↔SQL
multiset crosscheck on live marts, SampCert conformance: ε→∞ gives exact
sums, zero-user noise moments, χ² on catsample; artifact immutability;
spec sync; no-sorry grep); robustness harness of declared
perturbation→expectation CASES rebuilt with frozen artifacts; censoring-aware
actuarial estimators; `eval_models` adopts a torch model only if it beats its
baseline (paired-bootstrap CIs) with frozen `EVAL_DEFINITIONS`; redacting log
filter + log-scan test; ingest with explicit reject reasons + source
manifest; review_sample for human labels; data-integrity ledger mart;
Makefile with `help`/`test`/`robustness`/`reproduce`/`manifest`/`export`/
`synth`/`realism-audit`/`remote-bundle`/granular cleans/`distclean`
(deps-preserving `clean_lean`, unlike ours).

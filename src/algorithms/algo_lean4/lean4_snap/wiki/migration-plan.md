---
title: Migration plan (from GeoLife reference)
type: synthesis
updated: 2026-10-26
sources: ["[[geolife-reference]]", ../Makefile, ../goal.md]
---

# Migration plan — what to adopt from [[geolife-reference]]

Status 2026-10-27: **implemented** (all five phases; see [[log]] and the
per-item notes below — original plan text kept for provenance). Residual
deltas vs the reference: no remote-bundle target yet (deferred, low value for
this repo), review_sample deferred, and the notebook was extended rather than
split into a second remote notebook.

Ordered by value/effort; each phase leaves the repo consistent.

## Phase 1 — spec & test hygiene (small, do first) — ✅ done

1. **Spec sync**: our motif/DP thresholds are tripled (dbt vars,
   `python/export_lean_data.py` constants, `Checks.lean` defaults) with only a
   comment keeping them equal. Adopt: `python/spec.py` reading the dbt `vars:`
   block as single source; add `lean/EnclaveFlows/Params.lean` mirroring them;
   `tests/test_spec_sync.py` regex-locks Lean ↔ dbt ↔ Python; stamp
   `spec_version` into mart rows.
2. **pytest suite + `make test`**: Lean↔SQL crosscheck as a test (run
   `lake exe checks`, assert all `agree`), SampCert conformance tests
   (ε→huge ⇒ release equals clipped counts; empty input ⇒ discrete-Laplace
   moments; χ² on the sampler), no-`sorry` grep, artifact immutability.
3. **Makefile gaps**: add `help` (self-documenting), `test`, `manifest`,
   `notebook-html`, granular cleans (`clean_db`, `clean_reports`,
   `clean_release`, `clean_py`), and **split `clean_lean` from `distclean`**
   — ours currently deletes `lean/.lake` (≈5 GB re-download) on every clean.
4. **Claims ledger doc**: graded proved / proved-on-SampCert /
   checked-on-this-snapshot / empirical / assumed table in README or goal.md
   (we have the JSON ledger; add the human layer).

## Phase 2 — provenance & reproducibility — ✅ done (reproduce PASSED; robustness 5/5 with one declared tolerance)

5. **Content-addressed immutable artifacts**: version =
   hash(data fingerprint, hparams, code fingerprint); write-once, chmod
   read-only; `registry.json` + `RETRAIN=1` activation; `train_or_reuse` so
   `make train` is idempotent (replaces our timestamp versions + latest.txt).
6. **Run manifest + reproduce check**: per-table
   `bit_xor(hash(row))` fingerprints (floats rounded 6dp); two-phase check —
   frozen rebuild hash-identical, retrain reproduces the same artifact
   version and metrics ≤1e-6. Fills our goal.md "Reproducibility" row with
   evidence instead of recorded inputs only.
7. **Robustness harness** (`make robustness`): declared CASES for flows —
   shuffle rows (identical motifs), duplicate 10% (identical), start-time
   jitter ±200 ms (beacon recall ≥ 0.85), drop 30-min holes (no beacon spans
   a hole; session counts stable), thin half (fan-out flags stable), injected
   RST storm (portscan-shape flag fires, beacon set unchanged) — frozen
   artifacts, scratch DB, report.json. Clears the ◻ robustness criterion.

## Phase 3 — release discipline — ✅ done (histMech_DP proved; exact-rational budget; policy export verified)

8. **`config/release_policy.yaml` + export gate**: default-deny allowlists,
   suppression thresholds for non-DP artifacts, audited `denied:` list
   (feat_flows, ml_flow_scores, mart_dns…), ε as exact fractions, budget cap
   checked with rational arithmetic (our `Release.lean` compares in Float),
   `model_weights: deny_unless_reviewed`; `make export` writing
   `release/<id>/` with per-file sha256 manifest through an EgressGuard.
9. **Attestation abstraction**: enclave.py-style interfaces + SimulatedTEE
   (HMAC over code-fingerprint measurement + nonce) so all failure modes are
   unit-testable pre-platform; dev override named in policy, marked
   `attested: false`.
10. **Composed DP theorem**: upgrade our per-cell `privClippedCell_PureDP` +
    informal ledger to a proved whole-histogram bound via `neighbour_map` +
    `privCompose`/`privPostProcess` induction (`histMech_DP`-style:
    `PureDP … (K·ε₁/ε₂)`), and move clipping inside the verified query
    (SampCert `exactBoundedSum`).
11. **Safe logging**: redacting filter (IPs → `<ip>`, MACs/hosts) + log-scan
    test; logs inside the boundary.

## Phase 4 — synthetic flow generation (the deferred goal) — ✅ done (FlowShape.lean contracts proved; synth generate/validate/certify + realism audit)

12. **`scripts/synth.py` analog**: latent plan per synthetic host = service
    mix + session structure sampled from the DP release *only*, every draw
    through the verified sampler; renderer produces Joy-schema `flow.jsonl`
    using parameters declared as Lean `def`s in a new `FlowShape.lean`
    (gap grids per service, byte/packet size bins, beacon periods+jitter) that
    the generator regex-reads — Lean as master for generator params.
13. **Linking theorems**: prove the generator's parameters imply the encoder's
    predicates (beacon jitter bound ⇒ `IsBeacon`; session step bound ⇒
    `IsTimelySession`; hole spacing ⇒ no false transitions) — the
    `stayDrift_box_fits` pattern.
14. **validate / certify**: re-ingest synthetic JSONL through stg/feat SQL
    (dbt-compiled SQL replay) + `lake exe checks`; tiered exit codes (1 spec,
    2 TV/round-trip tolerance); `certify` generates `GenCert.lean` with
    `theorem sample_valid : … := by decide`, records axiom footprint
    (`kernel_only`) in certificate.json.
15. **realism_audit** (enclave-only): judge synthetic flows with our trained
    models (classifier accuracy gap vs real test; share of windows above the
    real anomaly p99) + TV on byte/duration histograms over public grids.
16. **remote-bundle**: ship release + Lean spec + generator (+ compiled stg
    SQL) with a sha256 MANIFEST for off-enclave generation.

## Phase 5 — model & data polish — ✅ done except review_sample (deferred)

17. **eval + adoption**: per-model baseline comparisons with paired-bootstrap
    CIs and frozen `EVAL_DEFINITIONS`; adopt a model only if it beats its
    baseline (our GRU vs repeat-last is exactly such a decision).
18. **Censoring**: treat `expire_type='a'` (active timeout) flows as
    right-censored in duration stats (actuarial estimator pattern).
19. **Completeness lemmas**: add infix-based iff characterizations for the
    return-trip extractor (`mem_abaMotifs` pattern) and a
    "breaks split transitions" homomorphism lemma; share fixtures between
    Lean `example := by decide`, dbt unit tests, and pytest.
20. **Ingest hardening**: explicit reject reasons + `source_manifest.json`
    (our staging silently filters the Joy header; keep a rejects relation and
    a data-integrity ledger mart).

## Deliberately NOT migrated

- GeoLife's geo-specific marts/macros (haversine, regions) — no flow analog.
- The dbt tag-based multi-stage `make train` ordering — our single train
  script is adequate at this scale; revisit with Phase 5.
- marimo layouts/slides and the second remote notebook — optional polish.

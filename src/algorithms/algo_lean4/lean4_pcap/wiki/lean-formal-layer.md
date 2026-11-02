---
title: Lean formal layer
type: component
updated: 2026-10-30
sources: [../lean, "https://github.com/leanprover/SampCert"]
---

# Lean formal layer

Lean 4 **v4.34.1** + mathlib v4.34.1 — the versions pinned by **SampCert**
(commit `a2c53bb`, AWS's verified DP library, used by AWS Clean Rooms DP).
Everything compiles without `sorry`.

**Proved** (`lean/EnclaveFlows/`):
- `clippedSum_sensitivity` — the clipped histogram-cell query is clip-sensitive
  under add/remove-one `Neighbour` (Stats.lean).
- `privClippedCell_PureDP` — the per-cell release is pure (ε₁/ε₂)-DP, by
  SampCert's `privNoisedQueryPure_DP` + our sensitivity proof (DPRelease.lean).
- `sampleCellPMF_law` — the released-histogram sampler hits cell i with
  probability exactly wᵢ/Σw, via SampCert's `UniformSample_apply'`
  (Sampler.lean; see [[verified-sampling]]).
- Motif theorems: `isBeacon_replicate`, `isBeacon_mono_cv`, `isFanout_mono`,
  `hasReturnTrip_cons_iff` + decidability of all three predicates (Motifs.lean,
  [[motifs]]).
- Association theorems: `chiSq_nonneg`, `chiSq_eq_zero_of_indep`,
  `assocAtLeast_mono` — exact ℚ chi-square (Assoc.lean,
  [[feature-associations]]).
- Sequence theorems: `isTimelySession_mono`, `isTimelySession_ordered`,
  `isTimelySession_tail` + decidability of `IsTimelySession`
  (Sequences.lean, [[sequence-analysis]]).
- `histMech_DP` — the WHOLE released histogram as one composed mechanism:
  K cells cost K·ε₁/ε₂ (DPRelease.lean); the executable's exact-rational
  ledger (now spanning the service + aux tls_stack/dns_shape histograms,
  total 4/1) enforces what the theorem bounds.
- TLS theorems: `canonicalHandshake_data_has_ccs`,
  `keyChangeBeforeData_head_not_data`, `fpCounts_partition`,
  `isRareStack_mono` + decidability via accumulator recursion (Tls.lean,
  [[tls-fingerprinting]]).
- DNS theorems: `wellFormed_labels_le` (≤127 labels for any legal name),
  `isSubdomainFan_mono`, `isSubdomainFan_le_length`; names modelled as label
  lists so proofs are arithmetic (Dns.lean, [[dns-structure]]).
- Generator contracts: `genBeacon_isBeacon`, `genSession_isTimely`,
  `sessionStep_lt_chainGap`, `interSessionGap_breaks` (FlowShape.lean,
  [[synthetic-generation]]); synthetic structures kernel-certified `by
  decide` in generated GenCert.lean.

**27 claims** in the ledger (`out/lean_results.json`); `lake exe checks`
re-decides every predicate on exported data and records Lean↔SQL agreement
counts (beacons, fanout, χ², sessions, TLS canonical/rare, DNS names/fans).

**Assumed** (stated in source comments and in `out/lean_results.json`):
SQL correctness of exported counts/gaps; /dev/urandom via SampCert's FFI;
the executables' JSON I/O.

**Executables**: `lake exe checks` (all predicates on data),
`lake exe release` ([[dp-release]]; aux histograms + exact ledger),
`lake exe sample` ([[verified-sampling]]), `lake exe catsample` (the
generator's only randomness). Outputs land in `lean/out/` and feed the
notebook and [[synthetic-generation]].

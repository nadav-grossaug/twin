---
title: TLS packet structure and fingerprinting
type: concept
updated: 2026-10-27
sources: [../dbt/models/features/feat_tls.sql, ../dbt/models/marts/mart_tls_fingerprints.sql, ../dbt/models/marts/mart_tls_records.sql, ../lean/EnclaveFlows/Tls.lean]
---

# TLS packet structure and fingerprinting

112 of the 5,397 flows carry joy's `tls` block ([[cisco-joy-features]]): hello
versions (joy coding 3 = TLS 1.0, 2 = SSLv3 — 2008 capture), offered cipher
suites, extensions, SNI, and `srlt` — the full **TLS record sequence**
(content type 20 CCS / 21 alert / 22 handshake / 23 appdata, direction, bytes,
inter-arrival ms). Two layers are extracted (`feat_tls`):

- **Client-stack fingerprint** (JA3-style): `md5('v{c_version},{cs joined},
  {sorted ext names}')[:8]`. 8 distinct stacks; `e2c75c72` (28 suites,
  3 extensions — the household's main browser) carries 87 flows,
  `6048d62e` 16; six stacks are **rare** (≤ `tls_rare_max_flows` = 3 flows,
  spec 1.1.0) — the rare-stack motif. `mart_tls_fingerprints` holds the
  population (hosts, selected suites, SNI diversity, share).
- **Record structure**: per-flow counts by type, per-direction TLS bytes,
  handshake setup latency, and the shape signature (first 8 `dir+type`
  tokens). `mart_tls_records` groups flows by shape — 8 shapes, all but one
  flow **canonical** (opens with a client handshake record; no appdata
  before the CCS in its direction). Flow 852 is the single violation: it
  starts mid-stream with appdata (truncated capture).

Formal mirror (`EnclaveFlows/Tls.lean`): `IsCanonicalHandshake` (decidable,
via the accumulator scan `NoEarlyData`) with soundness theorem
`canonicalHandshake_data_has_ccs`, and `IsRareStack` with
`fpCounts_partition` (Σ stack counts = N) and threshold monotonicity.
`lake exe checks` re-decides both on every exported flow: 112/112 canonical
verdicts and 8/8 stack counts agree with SQL ([[lean-formal-layer]]).

χ² wiring: `tls_version` and `tls_stack` are categoricals in `feat_flow_cats`,
so [[feature-associations]] covers them (service_class × tls_version
Cramér's V ≈ 0.81 — https vs the rest, as expected). Single-valued features
(e.g. all-`no_tls` synthetic data) are excluded from `mart_feature_assoc` by
a `having` guard — Cramér's V would divide by zero.

ML: **TlsRecordGRU** ([[minimodels]]) predicts the stack from record
structure alone (dir/type/log-size/log-gap, no payload) — honest negative
result at n=112: val accuracy 0.739 < 0.787 majority, `adopted: false`;
rare stacks appear only late in the capture, so the time-ordered split has
unlearnable classes. Predictions still materialize as `pred_tls_stack` in
`ml_flow_scores`.

Gotcha for re-implementers: DuckDB `array_to_string([])` returns NULL, not
`''` — fingerprints of extension-less hellos need `coalesce`
([[data-quality]] family: `least()` ignores NULLs).

Synthetic data (since 2026-10-30): every https flow carries a `tls` block —
six synthetic stacks over a fixed canonical-handshake skeleton, stack mix
from the DP-released `tls_stack` rank histogram (ε 6/25 of the 4/1 ledger),
`sni` from the policy-released common-knowledge top-k SNI histogram
(`mart_tls_sni_topk`, suppression-thresholded, outside the ledger). Cipher
realism stays a declared gap ([[synthetic-generation]], [[dp-release]]).

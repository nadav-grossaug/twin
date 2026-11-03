---
title: DP release
type: concept
updated: 2026-11-03
sources: [../dbt/models/marts/mart_dp_counts.sql, ../lean/Release.lean]
---

# DP release

The ONLY aggregate allowed out of the [[enclave-boundary]] with a formal
guarantee. Design (reworked 2026-10-31 for the large capture — see log):

- **Privacy unit**: one MONITORED (internal) host = one user = one list
  element in SampCert's add/remove-one `Neighbour` relation. Each flow is
  attributed to exactly one internal endpoint (sa if internal, else da);
  external responders are counted through the monitored host they talked to,
  never as users (2,433 external one-flow "users" previously inverted the
  mix).
- **Public grid**: the 7-class generator service taxonomy (genServices) —
  fixed in SQL, independent of the data — plus (2026-10-30) two AUX grids:
  6 TLS client-stack RANKS (`mart_dp_tls`; the rank→fingerprint map stays
  in-enclave), 9 DNS name-shape cells (`mart_dp_dns`: kind, label-count
  band, length band — no qname string reaches the mechanism) and the 49-cell
  `svc_transition` grid.
- **Contribution bounding**: PROPORTIONAL (L1) for the service mix — each
  host's cell vector is its own traffic mix allocated over a total budget of
  `dp_clip_total` = 128 flows (largest-remainder rounding, `mart_dp_counts`).
  The vector sums to ≤ 128, so per-cell sensitivity = 128 — the proved bound
  (`clippedSum_sensitivity`, [[lean-formal-layer]]) holds with clip = 128.
  Uniform per-cell clipping (kept at 4 for the aux grids) saturates for heavy
  hosts and cannot express a mix — the 2026-10-31 inversion incident.
- **Mechanism**: discrete Laplace via SampCert (`lake exe release`), exact
  integer sampling — immune to the floating-point attacks on naive Laplace.
- **Budget ledger**: per-cell ε = 2/5 on the service grid, 1/25 on the TLS/DNS
  aux grids, 1/100 on transitions; sequential composition → total ε =
  14/5 + 6/25 + 9/25 + 49/100 = **389/100 ≤ 4/1**, summed as an exact
  fraction in Release.lean (policy release-1.3.0 mirrors it). The executable
  REFUSES to write output when the ledger exceeds budget (exit 2).
- **Post-processing**: clamping negatives to 0 (and any normalization
  downstream) reads only the noised output, so DP is preserved.

Utility is empirical, not guaranteed: with 15 monitored hosts the noise scale
is 320 on the service grid against a ≤ 1,920 signal — major classes (dns,
https, http) carry SNR ≈ 1–3, tiny classes (chat, ntp) drown — report, don't
overstate. DP draws are intentionally non-reproducible ([[enclave-boundary]]).

## GeoLife (current target) — release-3.0.0 and the lifetime ledger

The content above describes the flow-era release. For [[dataset-geolife]]
(see [[geo-pipeline]]): privacy unit = one user (n = 64); five DP
histograms (motif 5, transitions 9, dwell 5, departure 4, mode 6 cells),
proportional-L1 clip 10. Under release-2.0.0 (ε_cell 0.05–0.30, total 4/1)
departure and mode were mostly noise; the owner chose fidelity over privacy
(2026-11-03): **ε_cell = 1 → 29/1 per export**, released-vs-true TV ≤ 0.08
on every histogram. Pure DP composes across exports, so `release.py` keeps a
**lifetime ledger** (`config/epsilon_ledger.jsonl`, cap
`dp_lifetime_epsilon_max` = 33/1 = 4 + 29): the dataset's budget is now
fully spent. ε values live only in `config/release_policy.yaml`
(`export_lean_data.py` reads them; sync test).

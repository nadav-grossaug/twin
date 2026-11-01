---
title: Enclave boundary
type: concept
updated: 2026-10-26
sources: [../goal.md]
---

# Enclave boundary

The encoder is designed to run inside an attested TEE (platform not yet
selected). What that means operationally here:

**Stays inside** (by default): `flow.jsonl` and the pcap; `artifacts/
enclave.duckdb` (all staging/features/marts — they contain raw IPs, ports,
DNS names); `models_out/` weights until reviewed; `lean/data/*.json` (the
clipped counts are NOT yet private — clipping is preparation, not protection);
per-flow predictions of [[minimodels]].

**May leave**: `lean/out/dp_release.json` (ε-DP, [[dp-release]]),
`lean/out/samples.json` (post-processing of the release),
`lean/out/lean_results.json` (proof statuses + motif agreement booleans —
reviewed release; note it names internal IPs, so in a real deployment it
would go through the same review gate).

Rules carried from goal.md: aggregate ≠ safe (DP or explicit review only);
suppression thresholds carry no formal guarantee; DP randomness must be
non-replayable (so DP outputs are excluded from bitwise reproducibility);
logs must exclude sensitive observations.

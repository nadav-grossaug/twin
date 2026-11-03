---
title: Dataset — CIC-IDS-style Thursday capture (flow.jsonl)
type: entity
updated: 2026-11-02
sources: [../flow.jsonl, ../artifacts/enclave.duckdb, wiki log 2026-10-28]
---

# Dataset — CIC-IDS-style Thursday capture (`flow.jsonl`)

**The CURRENT target** of the `lean4_pcap` build. A Cisco
[Joy](https://github.com/cisco/joy) export of a cyber-defense training pcap
(CIC-IDS-style, Thursday working hours, July 2017 — the capture era of the
`Thursday-WorkingHours.pcap` family). 1.3 GB, one JSON object per bidirectional
flow; the first line is a Joy config header (filtered by `stg_flows`).
Byte-identical to the file the original flow pipeline
(`~/tmp/test_torch_lean_arch_02`) was calibrated on, so all measured
thresholds carry over.

## Measured profile (2026-11-02, from `artifacts/enclave.duckdb`)

| Measure | Value |
| --- | --- |
| Flows | 406,938 |
| Time window | 2017-07-06, ≈ 8 h 05 min (1499342357 → 1499371475 epoch) |
| Source hosts (`sa`) | 2,448 — but only **15 internal** (192.168.10.0/24); 2,433 external sources are one-flow responders |
| Distinct destinations | 8,462 |
| Service mix (data) | dns 181,145 (44.5%) · https 141,604 (34.8%) · http 74,920 (18.4%) · ephemeral 5,754 · other 3,482 · icmp 31 · ard 1 · chat 1 |
| TLS flows | 60,796 (all canonical-checked in Lean); 33 client-stack fingerprints |
| DNS rows | 333,965 queries; 9,972 qnames shape-checked, 12 subdomain-fan hosts |
| Unanswered (`bytes_in=0`) | 108,328 (26.6%) |
| Zero-byte TCP | 72,018 (17.7% of TCP) |
| Duration | p50 36 ms · p90 20.4 s · max 60.3 s |
| Timely sessions | 21,783 (gap-broken at 600 s) |
| Association cells / pairs | 2,123 χ² cells · 78 Cramér's-V pairs |
| Beacon candidates | 9,192 channels; top true periodic: 192.168.10.25→142.0.160.53:443 (n=1722, CV 0.225); top false positive: 172.16.0.1→192.168.10.50:80 (n=2270, CV 16.3 — scan-like burst, not periodic) |

## Privacy-relevant shape

The 15-host population is the opposite regime from the travel build's 29,640
users: per-cell DP noise (scale ≈ 20) dominates small cells, so the public
grid is coarse (7 gen-services + ranks/shape bands, no strings) and
contribution bounding is proportional-L1 (`dp_clip_total = 128`), see
[[dp-release]]. External one-flow sources must NOT be treated as users —
they previously inverted the released mix.

Related: [[cisco-joy-features]] (feature schema; its coverage table is
Nitroba-era — this capture is 14.9% TLS, 44.5% DNS), [[data-quality]]
(Joy quirks; `make repair-data`), [[findings]] (Nitroba-era answers — re-read
with care), [[dataset-nitroba]] (the original 5,397-flow capture).
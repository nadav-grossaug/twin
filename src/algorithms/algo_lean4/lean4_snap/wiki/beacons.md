---
title: Beacon motif
type: concept
updated: 2026-10-26
sources: [../artifacts/enclave.duckdb, ../lean/EnclaveFlows/Motifs.lean]
---

# Beacon motif

A (sa, da, dp) channel re-contacted ≥ 8 times whose inter-start gaps have
coefficient of variation ≤ 0.6 and mean period ≥ 1 s. Lean states it without
square roots: `ssd ≤ maxCvSq · mean² · (n−1)` ([[lean-formal-layer]]).

Best candidates in [[dataset-nitroba]] (by CV):

1. 209.85.201.189 → 192.168.15.4:35828 — 8 flows, **27.2 s period, CV 0.029**.
   Google (GTalk-era) server-side keepalive; cleanest beacon in the capture.
2. 205.188.9.131 → 192.168.1.64:38458 — AOL/AIM keepalive, ~57–60 s, CV 0.106.
3. 192.168.15.4 → Yahoo/Google :80 — 9–11 flows at ~60–64 s, CV 0.17–0.24;
   webmail auto-refresh, the textbook HTTP-polling beacon shape.
4. 192.168.1.64 → 205.155.65.231:4500 — IPsec NAT-T keepalive ~40 s, CV 0.216.

Negative exemplars (high count, NOT periodic): SSDP multicast bursts (CV 5.4)
and bursty CDN fetches (CV 12). CV is what separates beaconing from volume —
see [[hosts-of-interest]].

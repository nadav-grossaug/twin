---
title: Fan-out / scan motif
type: concept
updated: 2026-10-26
sources: [../dbt/models/marts/mart_motif_fanout.sql]
---

# Fan-out / scan motif

One source touching many distinct destinations (host-scan shape) or many
distinct ports (port-scan shape). Threshold: ≥ 20 distinct destinations.

Key training lesson from [[dataset-nitroba]]: **the top "port scanner" is the
gateway 192.168.1.254** — 373 distinct dst ports toward 192.168.15.4 — but it
is NAT/RST reflection to ephemeral client ports, not probing. The separating
feature is `zero_byte_frac` (its flows carry no payload). The mart therefore
distinguishes `is_fanout` (destination count) from `is_portscan_shape`
(port count AND > 50% zero-byte), keeping the false positive labeled as such.

Genuine fan-out sources are the browsers: 192.168.15.4 (313 distinct dsts) and
192.168.1.64 (88) — "noisy host" exemplars, see [[hosts-of-interest]].
Lean's `IsFanout` (dedup-count ≥ k) is checked against exported destination
lists by `lake exe checks` ([[lean-formal-layer]]).

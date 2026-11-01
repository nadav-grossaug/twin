---
title: Cisco Joy feature schema
type: concept
updated: 2026-10-26
sources: [../flow.jsonl, "https://github.com/cisco/joy"]
---

# Cisco Joy feature schema

Joy condenses each (bidirectional) flow into one JSON object. Coverage in
[[dataset-nitroba]] (% of 5,397 flows):

| field | coverage | meaning |
|---|---|---|
| sa, da, pr, sp, dp | 100% | 5-tuple |
| bytes_out/num_pkts_out | 100% | initiator→responder volume |
| bytes_in/num_pkts_in | 65.7% | responder→initiator (34% unanswered) |
| time_start/time_end | 100% | epoch seconds (float) |
| packets[] {b, dir, ipt} | 100% | first-50-packet sizes/dirs/inter-arrivals |
| byte_dist[256] | 100% (66.6% non-zero) | payload byte histogram → [[minimodels]] input |
| entropy | 66.6% | payload entropy; >8 is invalid ([[data-quality]]) |
| ip.out/in.ttl | 100% | TTLs (OS/topology signal) |
| expire_type | 86% | 'i' inactive / 'a' active timeout |
| dns[] | 15.1% | queries+answers (qn = name) — see [[dns-structure]] |
| tls | 2.1% | versions, ciphersuites, SNI on 106 flows; record sequences — see [[tls-fingerprinting]] |
| ike | 0.2% | matches the udp/4500 IPsec NAT-T channel |

Parsed into DuckDB by [[dbt-pipeline]] `stg_flows` (header line filtered,
`dns` arrives typed as JSON — unnest via `from_json`).

---
title: Nitroba flow dataset
type: entity
updated: 2026-10-26
sources: [../flow.jsonl, ../nitroba.pcap]
---

# Nitroba flow dataset

`flow.jsonl`: 5,397 bidirectional flows extracted by Cisco Joy
([[cisco-joy-features]]) from the Nitroba University Harassment Case pcap —
a classic forensics teaching capture. First line is the Joy config header.

- **Window**: 2008-07-22 01:51:07 → 06:13:47 UTC (4.38 h), epoch 1216691467–1216707227.
- **Hosts**: 156 src IPs, 376 dst IPs; 16 RFC1918/link-local, 422 public.
  Two NAT'd segments: 192.168.15.x (dorm, behind NAT at .1) under 192.168.1.x
  (gateway .254). See [[hosts-of-interest]].
- **Protocols**: TCP 4,099 (76%), UDP 1,274 (24%), IGMP 21, ICMP 3.
- **Services**: http 3,626 flows; ssdp 227; https 162; ntp 92; dns 815 flows
  carry joy dns records (551 distinct names, akamai/weather.com/idisk.mac.com top).
- **Shape**: download-heavy browsing (median bytes_out 260 vs p95 bytes_in 22,100);
  33% of flows are zero-byte TCP scaffolding ([[data-quality]]).
- Narrative context: the harassment email in the case is sent near 06:09 UTC;
  the upload-shaped flows around then are listed in [[findings]].

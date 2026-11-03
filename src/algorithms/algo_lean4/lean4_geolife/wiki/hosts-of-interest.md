---
title: Hosts of interest
type: entity
updated: 2026-10-26
sources: [../artifacts/enclave.duckdb, "[[dataset-nitroba]]"]
---

# Hosts of interest

- **192.168.15.4** — the dominant (suspect) host: 3,575 flows (66%), 19.7 MB out,
  313 distinct destinations. Web/webmail/eBay/weather browsing; source of 3 of
  the 4 exfil-shaped uploads ([[findings]]). Drives the browsing-type
  [[beacons]] (60 s HTTP polling).
- **192.168.15.1** — dorm NAT router; SSDP chatter (213 bursty flows to
  239.255.255.250:1900 — high CV, correctly NOT a beacon).
- **192.168.1.254** — upstream gateway: 746 flows, 373 distinct dst ports toward
  192.168.15.4 — the canonical false-positive port scan ([[fanout-scan]]):
  zero-byte RST/NAT reflection, not probing.
- **192.168.1.64** — a Mac (idisk.mac.com, Bonjour mDNS): AIM keepalive beacon,
  IPsec NAT-T (udp/4500) VPN with ~40 s keepalive, receiver of the
  22.5 kB one-way external push that tops the anomaly ranking ([[minimodels]]).

---
title: Findings (synthesis)
type: synthesis
updated: 2026-10-26
sources: ["[[beacons]]", "[[fanout-scan]]", "[[minimodels]]", "[[dataset-nitroba]]"]
---

# Findings — current best answers

- **Cleanest beacon**: 209.85.201.189 → 192.168.15.4:35828, 27.2 s period,
  CV 0.029 (Google keepalive). Full ranking in [[beacons]].
- **Best scan false positive**: gateway 192.168.1.254's 373-port "sweep" —
  separable by `zero_byte_frac` ([[fanout-scan]]). Use as a negative training
  exemplar for port-count heuristics.
- **Exfil-shaped flows** (bytes_out ≥ 10 kB and ≥ 5× bytes_in): only 4.
  Notable: 192.168.15.4 → 65.55.13.158:443 (48.6 kB up, 05:58 UTC) and
  192.168.15.4 → 209.191.110.214:80 (36.8× ratio, ~06:09 UTC — the time of
  the Nitroba harassment email).
- **Top anomaly by autoencoder** ([[minimodels]]): 65.212.118.21 →
  192.168.1.64:33410, a 22.5 kB external one-way push with zero reply —
  also the EDA's #2 exfil-shaped flow; model and heuristic agree.
- **Byte histograms identify services**: 94.5% val accuracy from payload
  byte_dist alone — supports port-agnostic service classification.
- **Noise floor**: 33% zero-byte TCP; any volume-based detector must be
  robust to it ([[data-quality]]).

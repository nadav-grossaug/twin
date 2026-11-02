---
title: PyTorch minimodels
type: component
updated: 2026-10-30
sources: [../python/train_minimodels.py, ../models_out]
---

# PyTorch minimodels

Fifteen deliberately small models (`make train`, device = CUDA > MPS > CPU; this
machine trains on **MPS**). Versioned artifacts under `models_out/<version>/`
with `meta.json` (data fingerprint, seed, scaler, metrics); inference is the
separate dbt Python node `ml_flow_scores` ([[dbt-pipeline]]).

Current version **96a949312e59** (seed 1216691467, time-ordered 80/20 split,
scaler fit on train only — guideline 6), trained on the 2026-10-28 large
capture (406,938 flows; the previous five-model version was 036101d45d95).

## The original five

- **FlowAutoencoder** 14→16→8→3→8→16→14 on standardized numeric features.
  Reconstruction MSE = anomaly score. Val MSE mean 0.17, p95 0.79.
  (On Nitroba it surfaced the 22.5 kB one-way external push — [[findings]].)
  The 3-d bottleneck doubles as an embedding for visualization.
- **ByteDistClassifier** 256→64→C on normalized `byte_dist`
  ([[cisco-joy-features]]), weak labels = port-derived service class.
  **Val accuracy 0.951 vs 0.536 majority baseline** (6 classes on the large
  capture) — payload byte histograms alone nearly recover the service, a
  useful signal for port-agnostic service identification.
- **NextServiceGRU** (emb 8 → GRU 32, window K=5 services + log-gaps):
  P(next service | history) — the learned side of [[sequence-analysis]].
  **Val accuracy 0.885 vs 0.882 repeat-last and 0.458 marginal** baselines —
  adopted, but the margin over repeat-last is thin; windows never cross a
  600 s session break. Inference adds
  `pred_next_service`/`pred_next_confidence` per flow in `ml_flow_scores`.
- **TlsRecordGRU** (7-dim records → GRU 16, K=12 records): client-stack
  fingerprint from TLS record structure alone — direction, type, log-size,
  log-gap; no payload ([[tls-fingerprinting]]). On the large capture (48,636
  TLS flows, 24 stacks): **val accuracy 0.838 vs 0.450 majority baseline →
  adopted**. (On Nitroba it was an honest negative: 0.739 vs 0.787 at n=112,
  rare stacks only in the validation window — the gate flips with the data.)
- **DnsCharLM** (emb 16 → GRU 32 char LM over query names,
  [[dns-structure]]): per-name bits/char = name-shape anomaly, the
  DGA-detector shape. **Val 2.448 bits/char vs 4.605 unigram baseline →
  adopted.** Flow-level `dns_name_score` (max over the flow's names) and
  `dns_top_qname` in `ml_flow_scores`. **Scaling note (2026-10-28):** trains
  on *distinct* names weighted by query count (same record-level objective,
  identical expected gradient) — per-record training at 267k query records /
  9,209 names made epochs hours-long on MPS; evaluation stays record-level
  so the adoption gate is unchanged.

## The ten 2026-10-30 additions

Same conventions: a frozen entry in `EVAL_DEFINITIONS`, a declared non-ML
baseline (majority / per-group train median / repeat-last), time-ordered
80/20 split, `adopted` recorded honestly either way. Flow-level scorers add
columns to `ml_flow_scores` (`bd_recon_score`, `pred_service_vol` +
confidence, `bytes_in_abs_err`, `pred_unanswered_conf`); the channel/
session/TLS/DNS-record-level models evaluate in `meta.json` only.

- **ByteDistAE** 256→32→256 on the normalized `byte_dist` of payload-bearing
  flows (334,920): unsupervised payload-shape anomaly, recon MSE, **no
  adoption gate** (like FlowAutoencoder). Val MSE mean 1.27e-05, p95 4.19e-05.
- **VolumeServiceMLP** 14→32→6 — service class from the standardized numeric
  volume/timing features ONLY (no ports). **Val acc 0.893 vs 0.443 majority →
  adopted**: traffic shape alone nearly recovers the service.
- **BytesInMLP** svc-onehot+2→32→1 (L1) — log1p(bytes_in) from service +
  log bytes_out + log duration. **Val MAE 0.439 vs 1.784 per-service train
  median → adopted** (response volume is highly predictable from request
  volume).
- **DurationBandMLP** svc-onehot+2→32→5 — duration band
  (≤0.02/≤0.09/≤0.4/≤2 s/longer) from service + log bytes. **Val acc 0.791
  vs 0.451 majority → adopted**.
- **PairGapGRU** (GRU 16, K=5) — next log1p inter-flow gap on (sa,da,dp)
  channels with ≥8 flows (139,595 train windows). **Val MAE 1.093 vs 1.659
  repeat-last → adopted** — the learned side of [[beacons]]' periodicity.
- **SessionLenMLP** 14→32→5 — session flow-count bucket {2,3,4–5,6–8,9+}
  (FlowShape sessionFlowsChoices) from the session's FIRST flow's features
  ([[sequence-analysis]]). **Val acc 0.469 vs 0.407 marginal majority →
  adopted**, but weakly — the first flow says little about session length.
- **UnansweredMLP** 2+svc→16→2 — is_unanswered from OUTBOUND-only features.
  **Val acc 0.993 vs 0.992 majority → adopted** on the thinnest of margins
  (unanswered flows are 0.8% of traffic); read the gate, not the headline.
  Adds `pred_unanswered_conf` per flow.
- **TlsSetupBandMLP** fp-onehot+2→32→4 — hs_setup_ms band
  (≤50/≤200/≤1000 ms/longer) from the TLS fingerprint + n_records + log
  tls_bytes ([[tls-fingerprinting]]). **Val acc 0.601 vs 0.496 majority →
  adopted** — the client stack carries real setup-time signal.
- **DnsKindClassifier** 5→16→3 — query kind (ordinary / reverse_lookup /
  service_discovery) from numeric name-shape features only (qname_len,
  n_labels, max/min label len, digit_frac; [[dns-structure]]). **Val acc
  0.999 vs 0.988 majority → adopted** — near-perfect as expected: the flags
  are themselves name-shape rules.
- **BeaconChannelMLP** 5→16→2 — is_beacon from mart_motif_beacon channel
  stats (log n_flows, cv_gap, gap stats; [[beacons]]). Expected an honest
  negative at n=9,192 channels, but **val acc 0.977 vs 0.956 majority →
  adopted** — unsurprising, since is_beacon is a threshold rule over these
  same statistics; it is a consistency probe, not a discovery.

Limitations: weak labels inherit port-taxonomy errors; anomaly scores are
unsupervised (no ground truth); the GRU's margin over repeat-last is small
because traffic is self-transitioning bursts ([[sequence-analysis]]); each
model's `adopted` flag is data-dependent — always read it from the active
artifact's meta.json, not from this page; all stay inside
[[enclave-boundary]] until reviewed.

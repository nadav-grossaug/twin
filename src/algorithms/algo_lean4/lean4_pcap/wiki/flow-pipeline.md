---
title: Flow pipeline (lean4_pcap)
type: component
updated: 2026-11-02
sources: [../Makefile, [[migration-plan-pcap]], [[dataset-cicids]], ../HOWTO_run.md]
---

# Flow pipeline (lean4_pcap) — as built

The flow domain on the six-flow ops substrate ([[migration-plan-pcap]]).
Domain layer identical to the arch_02 build (same `flow.jsonl`, md5 match);
ops: `clean / train / synth / synth_from_bundle / notebook-enclave /
notebook-synth` + `make help`, `LEAN_HOME=/opt/data/lean` store, ONE
source-only bundle.

## Layer map and results (2026-11-02 run)

| Layer | Evidence |
| --- | --- |
| dbt/DuckDB | 36/36 models+tests PASS (stg → feat → marts incl. χ² assoc, seq, DP inputs, quality ledger) |
| Minimodels | 14 models, **all ADOPTED** vs frozen baselines (content-addressed `96a949312e59`, MPS, time-ordered 80/20): ByteDist 0.951 vs 0.536 · NextServiceGRU 0.885 vs repeat-last 0.882 (honest near-tie; bursty self-transition prior) · TlsRecordGRU 0.838 vs 0.450 · DnsCharLM 2.448 vs 4.605 bits/char · FlowAE MSE 0.170 · ByteDistAE 1.3e-5 · VolumeService 0.893 · BytesIn MAE 0.439 · DurationBand 0.791 · PairGap 1.093 vs 1.659 · SessionLen 0.469 · Unanswered 0.993 (near-tie) · TlsSetupBand 0.601 · DnsKind 0.999 · BeaconChannel 0.977 |
| Lean checks | SQL↔Lean agreement: 25 beacons, 15 fanout, 10 returns, 6 χ², 12 sessions, 60,796 TLS-canonical, 33 rare stacks, 9,972 DNS names, 12 subdomain-fans; 27 claims, no `sorry` |
| DP release | ε = 1215625/312500 = **3.89 ≤ 4/1** exactly-enforced ledger; 7 service cells + aux 49 transitions + 6 stacks + 9 DNS shapes; verified sampler χ²-consistent on 5,000 draws |
| Distill | genspec: 7 services, 7×7 transitions, 6 stacks, 9 DNS cells, 32 qnames, 32 SNI; fidelity **accepted**: student 0.8816 vs repeat-last 0.8816 vs teacher GRU 0.8854 |
| Export | `release/20261102T085618Z/` — DP release + flow_stats (2 suppressed) + top-k artifacts + 27 lean claims, `attested: false` (dev host) |
| Bundle | `dist/bundle` **176 KB source-only** (Lean sources verbatim + release + top-k + stdlib distill + build_from_source.sh) — no binaries |
| Synth | **10,000 hosts default** (SYNTH_HOSTS; max 10,000; generator addressing generalized past a /24): 154k flows / 5,002 beacons; re-ingested through the PRODUCTION dbt; **beacon recall 1.0** (genBeacon_isBeacon is a theorem); all 23,958 sessions timely; TV service-mix 0.269 < 0.35 (better at scale — LLN over hosts); exit 0 |
| synth_from_bundle | build_from_source re-checked all jobs from the bundle's sources (shared store), distill (enclave-free, `accepted=None` — correct), generate 154k flows, **recall 1.0**, TV 0.268 → **exit 0** |
| Certificate | `GenCert.lean` **PROVED**: 12 beacons + 43 sessions `by decide`, kernel-only, 11.3 s |
| Realism audit | declared gaps confirmed: autoencoder flags 100% of synth above real p95 (payload/byte-dist fields unmodelled — `fields_not_modelled` in meta); TV duration 0.79 / bytes 0.54 (discrete FlowShape choices vs continuous reality) |
| Notebooks | both marimo notebooks export headless (`visualize_enclave.html`, `visualize_synth.html` into artifacts/reports/) |
| pytest | **68 passed, 0 skipped** after `make synth` (48/20 skipped before — synth fixtures regenerate) |

## Notes

- The realism audit's "100% above p95" is NOT a pipeline failure: the
  generator deliberately renders `packets/byte_dist/entropy: null`
  (`fields_not_modelled`), so the anomaly scorer flags the missing payload
  shape. Continuous volume/duration realism is likewise declared-not-modelled
  (discrete public FlowShape choices).
- `synth_from_bundle` re-runs the same validation against the generator
  rebuilt from `dist/bundle` sources only — results in the log entry for
  2026-11-02.
- ε ledger 389/100 per release, privacy unit = monitored internal host
  ([[dp-release]]); ledger arithmetic is exact-fraction in Lean AND Python.
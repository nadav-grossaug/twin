---
title: Synthetic flow generation
type: component
updated: 2026-10-30
sources: [../scripts/synth.py, ../lean/EnclaveFlows/FlowShape.lean]
---

# Synthetic flow generation

`make synth` / `make certify` / `make realism-audit` — the downstream
generator the encoder goal deferred, now implemented on the GeoLife pattern
([[geolife-reference]]):

- **Inputs**: ONLY the [[dp-release]] (service mix + the 2026-10-30 aux
  histograms: `tls_stack` rank mix and `dns_shape` kind/label/length mix;
  exact ledger 17/5 + 6/25 + 9/25 = 4/1, the budget saturated) plus generator
  shape parameters declared as Lean `def`s in `FlowShape.lean` (gap grids, byte
  bins, beacon periods/jitter, session steps) — regex-read by `synth.py`, so
  the simulation cannot drift from the proved constants. Every random draw
  goes through the **verified `catsample` binary** (law = `sampleCellPMF_law`);
  no Python RNG exists in the generator.
- **TLS/DNS structure** (2026-10-30): every https flow carries a `tls` block
  — six synthetic client stacks over a fixed canonical-handshake `srlt`
  skeleton, so `IsCanonicalHandshake` holds by construction; every dns flow
  carries one `{qn, rc:0}` query — kind ~ the released `dns_shape` cells,
  ordinary/DNS-SD names ~ the policy-released **common-knowledge top-k qname
  histogram** (suppression-thresholded, outside the ε ledger; `.test`
  gibberish fallback), reverse lookups from the synthetic `da`; each https
  flow's `sni` ~ the matching **top-k SNI histogram** (`svc.synth.test`
  fallback). Only mixes and multi-host popular names are data-derived; no
  fingerprint hash, cipher list or private name crosses the gate
  ([[tls-fingerprinting]], [[dns-structure]]). Cipher realism stays a
  declared gap.
- **Linking theorems** (`FlowShape.lean`, all proved): `genBeacon_isBeacon`
  (gaps drawn as period−jitter+catsample(2j+1) satisfy [[beacons]]' IsBeacon),
  `genSession_isTimely` / `isTimely_cumTimes` (bounded steps form timely
  [[sequence-analysis]] sessions), `sessionStep_lt_chainGap`,
  `interSessionGap_breaks` — "the encoder accepts the plan" is a theorem.
- **validate**: re-ingests the Joy-schema output through the production dbt
  pipeline + the Lean `checks` binary; tiered exits (1 = spec violation,
  2 = tolerance). Current run (12 hosts, 226 flows): dbt 25/25, 8/8 planned
  beacons detected (recall 1.0), 29/29 sessions timely, per-host service
  counts exact, TV(service mix vs release) 0.096 ≤ 0.20.
- **certify**: emits `lean/GenCert.lean` with the detected structures as
  literals and `by decide` theorems; kernel-checked in 2.4 s,
  `kernel_only=true` (axioms: propext, Classical.choice, Quot.sound). Note:
  needs `unseal Rat.add ...` because core ℚ arithmetic is `@[irreducible]`.
- **realism audit** (enclave-only, not exportable): the production
  autoencoder scores synthetic flows (20.3% above real p95, 0% above p99);
  TV of duration/bytes histograms ~0.45–0.64 — synthetic traffic is
  spec-conformant but visibly simpler than real traffic, as expected from
  assumption-driven rendering.
- The round-trip audit **found a real encoder bug** (the `least()` NULL
  entropy issue, see [[data-quality]]) — validation cuts both ways.
- **Per-column assurance**: `tests/test_synth_columns.py` tests every column's
  fill (χ² GOF vs declared weights/released mix, uniformity of steps and
  jitter, structural invariants); `visualize_synth_mo.py`
  (`make notebook-synth`) shows synthetic EDA, declared-vs-observed GOF, and
  a full column-by-column enclave-vs-synth comparison (TV distances, NULL
  rates, fill source) with per-column drilldown.

## Topology (2026-11-05)
Who-talks-to-whom is a set of **declared assumptions** (GenSpec.lean `Topology`
block, not data-derived): ~40 busy internal hosts (10.99.x.y) fan OUT to 9 popular
externals (198.51.100.0/24, 2 per host) and an 8,000-address long tail
(198.18.0.0/15); 12% of flows go to 2 internal server hosts. Sized for ~8 h at
~14 flows/s. Cost: the dp_clip=4 per-host clipping saturates, so `validate`'s
service-mix TV is soft-failing (exit 2). See [[log]] 2026-11-05.

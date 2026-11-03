---
title: Migration plan — pcap flows (lean4_pcap)
type: synthesis
updated: 2026-11-02
sources: [../goal.md, ../METHOD.md, [[travel-pipeline]], [[migration-plan-foursquare]]
---

# Migration plan — pcap flows (lean4_pcap)

**Status: IMPLEMENTED 2026-11-02** (all six ops flows green). This page records
the third migration of the dataset-agnostic method ([[migration-plan-stock]],
[[migration-plan-foursquare]] before it): the target dataset is the ~407k-flow
Cisco Joy capture ([[dataset-cicids]]) — the same `flow.jsonl` the original
flow pipeline (`~/tmp/test_torch_lean_arch_02`, dataset-era 2026-10-26/27) was
built on, byte-identical (md5 match), so every measured threshold, motif
count and DP number carries over unchanged.

## What moved, what stayed

| Aspect | Source | Decision |
| --- | --- | --- |
| Domain layer (dbt features/motifs/assoc/seq marts, `EnclaveFlows` Lean modules, minimodels, distill, synth, notebooks, tests) | `test_torch_lean_arch_02` (the flow-era build) | **copied verbatim** — the data is identical, so the calibration is identical (checks reproduce: 25 beacons, 15 fanout, 6 χ², 12 sessions, 60,796 TLS-canonical) |
| Ops substrate (six flows, `LEAN_HOME` store, source-only bundle, `synth_from_bundle`, `make help`) | `../lean4_foursquare` (the travel build's simplified contract) | **ported**: new Makefile, `make_bundle.sh` rewritten source-only (binary + dylib-patching deleted), `--bundle` flag added to `scripts/synth.py` |
| Binary deploy / multiple bundles / deploy_dir | arch_02 era | **discarded** per ops brief: exactly ONE source-only bundle; synth consumes Lean sources + released JSON only |

## The six flows (the contract)

`clean` / `train` / `synth` / `synth_from_bundle` / `notebook-enclave` /
`notebook-synth`, plus `make help`. `LEAN_HOME=/opt/data/lean` is the only
Lean toolchain source (elan + mathlib cache + pinned packages, symlinked into
`lean/.lake/packages`; `lean-store` bootstraps it). See `../HOWTO_run.md`.

## Substitution-map notes (METHOD §12, flow column)

- Protected entity: one source host (`sa`) — NOT one flow (flows would
  under-count repeat structure) and not one (sa,da) channel.
- Public grids: 7-cell gen-service mix, 7×7 svc_transition, 6 TLS stacks,
  9 DNS shape cells. Exact-fraction ε ledger: **389/100 ≤ 4/1 per release**
  (service + tls_stack + dns_shape + svc_transition; measured 3.89).
- Common-knowledge top-k: DNS qnames and TLS SNI (suppression-thresholded,
  declared non-DP, outside the ledger).
- Structural exclusions: `*.arpa` / single-label names can never be released
  as top-k regardless of popularity ([[dns-structure]]).
- Motifs: beacon (CV rule), fan-out/scan, return-trip, TLS canonical/rare
  stack, DNS subdomain-fan — all decidable, re-decided by `lake exe checks` on
  exported observations.
- Sequence layer: timely sessions (gap-broken), resolve→connect chains,
  service transition matrix with lift — the "timely flow logic" analysis.

## Incidents earned during this migration

1. Direct `lake` invocation from a shell without the Makefile's exported
   `ELAN_HOME`/`PATH` fails with 127 — always go through `make lean-build`
   (or export the store's elan first). Recorded in
   [[lean4-pipeline-env]]-style memory.
2. `scripts/synth.py --bundle` must default `--release` to the BUNDLE's own
   `lean/out/dp_release.json` (else it silently proves self-containment
   against the repo's release); the Makefile passes it explicitly anyway.
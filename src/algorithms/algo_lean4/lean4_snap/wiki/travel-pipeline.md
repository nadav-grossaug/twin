---
title: Travel pipeline (as built)
type: component
updated: 2026-11-02
sources: ["../goal.md", "[[migration-plan-foursquare]]", "../artifacts/enclave.duckdb", "../lean/out/", "../models_out/"]
---

# Travel pipeline — the as-built lean4_foursquare build

Status 2026-11-02: **implemented and green end-to-end.** All six ops flows
work: `clean` / `train` / `synth` / `synth_from_bundle` / `notebook-enclave`
/ `notebook-synth`. 65 pytest green; dbt 112/112; `lake build` clean on the
pinned toolchain (Lean v4.34.1, SampCert a2c53bb) with **no `sorry`, no
`native_decide`**.

## Layer map (what runs where)

- **dbt/DuckDB**: `stg_transfers` → `feat_transfers`/`feat_users`/
  `feat_episodes` → 11 analytic marts + 9 DP-input marts (proportional-L1
  largest-remainder clipping at `dp_clip_total = 8`; travel-count and
  return-rate at sensitivity 1). 86 schema tests.
- **Minimodels** (content-addressed, registry-activated, user-hash 80/20
  split): all four adopted on real data — NextRegionGRU **75.7%** vs
  return-to-previous 72.0% (the brutal baseline, beaten), GapClassGRU 39.4%
  vs majority 29.1%, EpisodeShape NLL 0.733 vs unigram 0.755, ProfileAE
  val mse 0.029.
- **Lean formal layer** (`EnclaveTravel`): decidable motif mirrors
  (IsReturnPair / IsCorridor with the 25·k·s2 ≤ 41·s1² exact-CV algebra /
  IsTour / IsBurst / IsLongStay / IsEpisode), χ² over ℚ, SampCert DP
  mechanism (privClippedCell_PureDP, histMech_DP), sampler law
  (sampleCellPMF_law), generator contracts (times sorted, prev≠next,
  duration chaining by construction, window containment). `lake exe checks`:
  **154/154 agreements**, 12/12 episodes; 24 claims proved, 3 assumed
  (trust boundary 2 explicit).
- **ε-DP release**: ledger spent **exactly 400/100 = 4/1** (travel-count 28
  + region mix 32 + transitions 192 + gaps 30 + durations 30 + distance 24
  + months 24 + episodes 20 + return-rate 20); over-budget refusal exit-2
  tested on both sides; dow released as a 0ε uniform public prior.
- **Distillation**: genspec provenance cell-exact; the only prior is the
  declared diagonal `return_boost`. **Fidelity: honest negative** — the
  first-order matrix student scores 56.5% vs the 72% second-order
  return-to-previous baseline (`accepted: false`, exactly the watch-list
  prediction); the generator compensates with per-user return-rate draws
  from the released bands.
- **Generator** (`lake exe generate`, source-only): per-user travel count →
  episode/gap tempo → region walk → corridor-based country instantiation →
  centroid coordinates → 24-hex venues. Plants corridor (user 1) + tour
  (user 2). validate exit 0: **recall 1.0 + 1.0**, travel-count TV 0.012,
  region-mix TV 0.239, 0 Lean disagreements on re-ingestion; certificate
  kernel-checked.
- **Bundle**: ONE source-only bundle (188K). `synth_from_bundle` rebuilt all
  7,817 Lean jobs from bundle sources via the shared store and its output
  validated (recall 1.0/1.0, TV 0.033/0.230).

## Ops substrate

`LEAN_HOME=/opt/data/lean` holds elan, the mathlib cache and the pinned
package store; `make lean-store` bootstraps and symlinks
`lean/.lake/packages`. Seeded once from the stock build (2026-11-01); both
the repo build and the bundle rebuild reuse it — no per-project downloads.

## Known gaps (declared, measured)

- **Return-rate realism**: synth per-user return share 0.20 vs real 0.82.
  Cause: the second-order return draw only fires when the first-order
  region walk routes back, so released return_rate bands under-apply. Top
  candidate fix: make the generator return-first (draw return vs new-hop
  BEFORE the region walk), still sourced from the same released bands —
  post-processing only, no new ε.
- `fields_not_modelled`: intra-country geography (centroid coords), venue
  identity structure, cross-user relations, within-bin gap shapes beyond
  suppressed moments, country choice beyond the 178 released corridors.
- Frozen-rebuild reproduce script deferred (as in [[stock-pipeline]]).

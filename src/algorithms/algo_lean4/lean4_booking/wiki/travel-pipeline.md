---
title: Travel pipeline (as built)
type: component
updated: 2026-11-03
sources: ["../goal.md", "[[migration-plan-foursquare]]", "../artifacts/enclave.duckdb", "../lean/out/", "../models_out/"]
---

# Travel pipeline — the as-built lean4_foursquare build

Status 2026-11-03: **implemented and green end-to-end, ledger v2.1.** All six
ops flows work: `clean` / `train` / `synth` / `synth_from_bundle` /
`notebook-enclave` / `notebook-synth`. dbt 127/127; `lake build` clean on the
pinned toolchain (Lean v4.34.1, SampCert a2c53bb) with **no `sorry`, no
`native_decide`**. The 2026-11-03 quality pass (see [[log]]) fixed overlap /
hour-of-day / countries / in-out: synth validate now reports overlap 0,
hod TV 0.029, country TV 0.087 over all 70 released countries, in/out
imbalance 0.058 (real 0.021), region-mix TV 0.031, return rate 0.796 vs
real 0.818, recall 1.0 + 1.0, 0 Lean disagreements.

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
- **ε-DP release (ledger v2.1)**: spent **exactly 400/100 = 4/1** —
  travel-count 28 + region mix 24 + transitions 192 + gaps 30 + durations 25
  + distance 20 + months 12 + **hour-of-day 24** + episodes 25 + return-rate
  20; hod displaced width elsewhere per the saturation rule; over-budget
  refusal exit-2 tested on both sides; dow stays a 0ε uniform public prior
  (measured max/min 1.28) while hod earns its ε (max/min 2.78).
- **Distillation**: genspec provenance cell-exact; the only prior is the
  declared diagonal `return_boost`. **Fidelity: honest negative** — the
  first-order matrix student scores 56.5% vs the 72% second-order
  return-to-previous baseline (`accepted: false`, exactly the watch-list
  prediction); the generator compensates with per-user return-rate draws
  from the released bands.
- **Generator** (`lake exe generate`, source-only): per-user travel count →
  return-rate band → **return-first hops** (return to the previous country
  with the band's probability, else region walk + country from the released
  per-region w_in weights) → gap tempo → timestamps with **DP hour-of-day**
  (min/sec uniform prior) → durations **capped at the next gap** (no
  simultaneous presence — theorem `mkUserHistory_no_overlap`) → centroid
  coordinates → 24-hex venues. Plants corridor (user 1) + tour (user 2).
  validate exit 0: **recall 1.0 + 1.0**, overlap 0, travel-count TV 0.023,
  region TV 0.031, hod TV 0.029, country TV 0.087 (all 70 released
  countries), in/out 0.058, 0 Lean disagreements; certificate
  kernel-checked.
- **Suppressed releases**: gap moments, 178 corridors, and (new) 70
  countries with per-country arrival/departure volumes.
- **Bundle**: ONE source-only bundle (188K). `synth_from_bundle` rebuilt all
  7,817 Lean jobs from bundle sources via the shared store and its output
  validated (recall 1.0/1.0, TV 0.033/0.230).

## Ops substrate

`LEAN_HOME=/opt/data/lean` holds elan, the mathlib cache and the pinned
package store; `make lean-store` bootstraps and symlinks
`lean/.lake/packages`. Seeded once from the stock build (2026-11-01); both
the repo build and the bundle rebuild reuse it — no per-project downloads.

## Known gaps (declared, measured)

- ~~Return-rate realism~~ FIXED 2026-11-03 by the return-first redesign:
  synth 0.796 vs real 0.818 (was 0.20).
- `fields_not_modelled`: intra-country geography (centroid coords), venue
  identity structure, cross-user relations, within-bin gap shapes beyond
  suppressed moments, exact country-level transition matrix (only region
  transitions + returns + country marginals are released), minute/second
  diurnal microstructure beyond the hourly DP histogram.
- Frozen-rebuild reproduce script deferred (as in [[stock-pipeline]]).

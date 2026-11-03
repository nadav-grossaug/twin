---
title: Verified sampling
type: concept
updated: 2026-10-26
sources: [../lean/EnclaveFlows/Sampler.lean, ../lean/Sample.lean]
---

# Verified sampling

Entry point for the downstream synthetic-data generator: draw service classes
from the [[dp-release]] histogram with a sampler whose law is **proved**, not
assumed.

Construction (`sampleCellPMF`): a SampCert `UniformSample` over the total
released weight, mapped through a cumulative `bucketOf`. Built from `SPMF`
combinators, so normalization holds by construction.

Proved chain ([[lean-formal-layer]]):
1. `UniformSample_apply'` — SampCert: uniform law of the base draw.
2. `uniformBucket_apply` — bucketed draw hits i with P = |preimage|/total.
3. `card_bucketOf` — the preimage of cell i has exactly weight wᵢ positions.
4. ⇒ `sampleCellPMF_law`: **P(cell i) = wᵢ / Σw**, exactly, in ENNReal.

`lake exe sample N` draws N times through SampCert's /dev/urandom FFI and
writes draw counts to `lean/out/samples.json`; the notebook runs a χ²
goodness-of-fit of draws against the released weights (an empirical check of
the proved law end-to-end through the unverified I/O).

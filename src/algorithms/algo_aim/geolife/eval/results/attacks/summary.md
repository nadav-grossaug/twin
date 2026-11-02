# Attack results (EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee)

## Membership inference (user level, against the artifact)
- informed (worst-case, exact): per-user AUC p50 0.695, p90 0.723, max 0.730 (zCDP bound for a cap-saturating user: 0.769)
- Q (LR separation): p50 0.519, max 0.751, bound 2*rho = 1.079
- worst user TPR at FPR: 0.01: 0.072, 0.1: 0.339
- Monte Carlo (uninformed matched filter, 400 worlds): per-user AUC p50 0.598, p90 0.634, max 0.704
- tables skipped in the vectors: none

## Seed averaging
- 3 seeds; averaging divides the noise std by 1.73
- honest accounting of releasing all seeds: rho 1.619 -> epsilon 11.08 (single seed: 6.00)
- per-table cross-seed std vs ledger sigma and signal gain: `seed_averaging.csv`


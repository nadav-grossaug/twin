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

## Linkage / re-identification (against the synthetic data, same frame)
- nearest-user profile distance: real->syn median 2.920 vs real-holdout median 2.090 (ratio 1.40; too_close=False)
- top-10 cell Jaccard: best vs syn mean 0.111 (max 0.250); chance (other real users) mean 0.351; copy reference (own split half) mean 0.397
- re-identification rate (best syn match >= own split half): 0.036 vs chance 0.439
- Hungarian matching mean Jaccard 0.060 vs random 0.009
- coarse signature linkage: rate vs syn 0.077, holdout 0.119, excess -0.042


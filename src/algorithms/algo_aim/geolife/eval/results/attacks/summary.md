# Attack results (EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee)

Run 2026-11-02 on the 26-table D27 artifacts (`outings_v2_seed0-2`); linkage vs a fresh
Beijing synthetic set from seed0 (realistic generator, D24-D26; 126/139 users tokenized).
Previous numbers (29-table artifacts, old generator): `geolife/eval/reports/privacy-attacks.md`.

## Membership inference (user level, against the artifact)
- informed (worst-case, exact): per-user AUC p50 0.700, p90 0.729, max 0.734 (zCDP bound for a cap-saturating user: 0.769)
- Q (LR separation): p50 0.548, max 0.778, bound 2*rho = 1.079
- worst user TPR at FPR: 0.01: 0.074, 0.1: 0.345
- Monte Carlo (uninformed matched filter, 400 worlds): per-user AUC p50 0.598, p90 0.635, max 0.648
- tables skipped in the vectors: none

## Seed averaging
- 3 seeds; averaging divides the noise std by 1.73
- honest accounting of releasing all seeds: rho 1.619 -> epsilon 11.08 (single seed: 6.00)
- per-table cross-seed std vs ledger sigma and signal gain: `seed_averaging.csv`

## Linkage / re-identification (against the synthetic data, same frame)
- nearest-user profile distance: real->syn median 2.247 vs real-holdout median 2.090 (ratio 1.08; too_close=False)
- top-10 cell Jaccard: best vs syn mean 0.101 (max 0.250); chance (other real users) mean 0.351; copy reference (own split half) mean 0.397
- re-identification rate (best syn match >= own split half): 0.050 vs chance 0.439
- Hungarian matching mean Jaccard 0.065 vs random 0.010
- coarse signature linkage: rate vs syn 0.115, holdout 0.119, excess -0.003


# Empirical privacy attacks on the GeoLife artifact and synthetic data (2026-11-02, refreshed on the 26-table artifacts)

**EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee.** The guarantee is the zCDP
ledger in each artifact's `privacy.json` (user-level adjacency, rho 0.5395, epsilon 6.000 at
delta 1e-6); everything here asks what concrete attackers achieve against this release, as a
sanity check of that accounting and of the generator. Code: `src/geolife/eval/attacks.py`
(design notes in its docstring), runner `geolife/eval/run_attacks.py`, machine-readable results
`geolife/eval/results/attacks/{attacks.json,seed_averaging.csv,summary.md}`. PLAN step 7.

This is the second run. The first (earlier on 2026-11-02) attacked the superseded 29-table
artifacts with the pre-step-5 generator; those numbers are kept below as "previous (29-table)"
comparison rows. This run attacks the committed D27 artifacts (26 tables - `cat_time_visits`,
`cat_transitions`, `sensor_step` dropped, their rho re-spent so every sigma is ~5-10% lower;
same rho_total/epsilon) and, for linkage, a fresh synthetic set from the step-5 realistic
generator (D24-D26) - the old linkage baselines were flagged as depressed by the generator's
realism gap, so this run is the honest version of that test.

Setup: the 139 real users the artifacts were fitted on (same tokenizer, pseudonyms and caps as
stage 1, sensor views included, so the attack vectors are exactly the vectors the mechanism
summed); the 3 committed artifacts `outings_v2_seed0-2`; synthetic side for linkage = 139 users
generated from `outings_v2_seed0` into the source city (`--city beijing`, 33.7M points), so real
and synthetic share the map frame.

## 1. Membership inference at the user level (the DP unit), against the artifact

The attacker holds the released noisy tables and a target user's complete data and must decide
whether the target was measured.

- **Informed attacker (worst case, exact).** If the attacker also knows every other user's data,
  the optimal test faces a Gaussian location problem with separation
  Q_u = sum_t ||v_{u,t}||^2 / sigma_t^2, so the per-user AUC is Phi(sqrt(Q_u/2)) with no
  simulation error. Over the 139 real users: **Q p50 0.548, p90 0.742, max 0.778**, against the
  zCDP ceiling 2*rho = 1.079 (a user would have to saturate every cap to reach it). Per-user AUC:
  **p50 0.700, p90 0.729, max 0.734** (ceiling 0.769). Previous (29-table): Q p50 0.519 / max
  0.751, AUC p50 0.695 / max 0.730 - re-spending the freed rho on fewer tables (lower sigmas)
  raised the worst-case separation by ~3.5%, exactly as the accounting predicts, and the maximum
  stays clearly under the analytic bound. The worst user's TPR is **0.074 at FPR 0.01** and
  0.345 at FPR 0.1 (previous: 0.072 / 0.339): even the strongest adversary against the most
  exposed user gets a weak test. The presence tables dominate the worst user's Q
  (trav_users_r7 0.25, place_users_r7 0.11, place_users_r6 0.08, trav_users_r6 0.07,
  place_users_r8 0.06); every pmf/one-hot table contributes < 0.027.
- **Uninformed attacker (realistic).** Matched-filter score <y, v_u>/sigma^2 over 400 simulated
  releases of the real mechanism (random half of the users in, ledger sigmas), without knowledge
  of the other users: per-user AUC **p50 0.598, p90 0.635, max 0.648** - clearly below the
  informed ceiling, because the other users' mass acts as additional noise. Previous
  (29-table): p50 0.598 / p90 0.634 / max 0.704; the p50/p90 are unchanged and the max moved
  within the +-0.03 simulation noise of a fresh 400-world draw plus the new vectors.
- Reading: AUC 0.5 is chance; the observed levels are what epsilon 6 is expected to allow and
  nothing above it. The empirical maximum staying under the analytic bound (0.734 < 0.769, with
  Q_max 0.778 < 1.079) is an end-to-end consistency check of sensitivities and sigmas - it held
  on both the 29-table and the 26-table releases.

## 2. Seed averaging (why a real release is ONE seed)

The 3 committed artifacts are noise replicates of one fit. Measured on the 23 tables whose
released vocabulary is identical across seeds (the hierarchical child tiers place r7/r8 and
trav r7 differ by seed and are excluded; the previous run had 26 common tables of 29):

- The cross-seed per-key std matches the ledger sigma (median ratio 0.80 over the 23 tables,
  previous run 0.84; 3 samples per key, so individual ratios scatter, 0.63-1.11 away from the
  single-key n_users table): the seeds are exactly sigma-noise replicates.
- Averaging the 3 seeds therefore cuts the effective noise std by **sqrt(3) = 1.73** on every
  table (e.g. outing_start: sigma 7.15 -> 4.13; sensor_speed: 4.55 -> 2.63), and lifts every
  per-table signal score by the same factor - e.g. sensor_speed on the new seed 1 sits below
  the fit threshold 3 (2.39, "falls back") but averaging lifts it to ~4.1 (usable; seed 0's
  score 3.4 -> 5.8): averaging recovers tables the single-seed design deliberately left in the
  noise. (The previous run's example, cat_time_visits 2.7 -> 4.6, is moot: D27 dropped it.)
- The honest accounting of releasing all three is 3 * rho = 1.619, i.e. **epsilon 11.08** at
  delta 1e-6, not 6. The informed-MI separation Q scales the same way (worst-case per-user AUC
  Phi(sqrt(3Q/2)): max 0.734 -> 0.860).
- Per-table numbers: `geolife/eval/results/attacks/seed_averaging.csv`.

Consequence, already implemented: `fit_artifacts.py --release` emits exactly one seed and stamps
the policy into `privacy.json` (`learn_v2.ONE_SEED_CAVEAT`); since D27a the committed dev trio
also carries the 3-replicate caveat in each `privacy.json`. The trio stays as a DEV set and must
never be published together (also recorded in `geolife/transfer/README.md` and the ledger
review).

## 3. Linkage / re-identification against the synthetic data

Synthetic users are fresh samples, not transformed real users, so there is no true
correspondence to recover; the attack asks whether any synthetic user is suspiciously CLOSE to a
real user (a memorised copy). Every statistic is read against two references: the same statistic
with the synthetic side replaced by OTHER real users (chance - different people share hubs), and
the within-user split-half value (what finding a true copy of the same person would look like).
Both sides in the Beijing frame, same tokenizer. Unlike the previous run, the synthetic side now
comes from the step-5 realistic generator (D24-D26), so chance-level statistics are meaningful
rather than depressed by a realism gap.

Setup detail: 126 of the 139 synthetic users pass the tokenizer's min-activity rule (previous
run: 136 - the realistic generator produces more low-activity users, like the real tail).

| statistic | new (26-table, realistic gen) | previous (29-table, old gen) | chance ref | copy ref |
|---|---|---|---|---|
| nearest-profile distance ratio (syn/holdout medians) | **1.08** (2.25 / 2.09) | 1.40 (2.92 / 2.09) | 1.0 | << 1 |
| best top-10-cell Jaccard vs syn, mean (max) | **0.101** (0.250) | 0.111 (0.250) | 0.351 | 0.397 |
| re-identification rate | **0.050** | 0.036 | 0.439 | - |
| Hungarian mean Jaccard (vs random matching) | **0.065** (0.010) | 0.060 (0.009) | - | - |
| coarse-signature match rate (excess over holdout) | **0.115** (-0.003) | 0.077 (-0.042) | 0.119 | - |

- **Nearest-user profile distance** (log activity sizes, revisit rate, median outing shape,
  6 time-bin shares, standardised): real -> nearest synthetic median 2.25 vs real-holdout
  (user half A -> half B) median 2.09, ratio **1.08** (too_close = False). The ratio fell from
  1.40 to almost exactly 1: synthetic users now look like an independent sample of comparable
  people - the faithful-generator target - while still not sitting closer than real strangers
  do; a copy would drive the ratio towards 0 (min distance 0.92 vs holdout min 0.0).
- **Top-10 visited-cell Jaccard** (res-8 stop cells, the most identifying map-relative feature):
  best match against the synthetic users mean **0.101** (max 0.250), chance level against other
  real users mean 0.351, copy reference (own split half) mean 0.397. **Re-identification rate**
  (a synthetic user resembles the target at least as much as the target's own other half):
  **0.050** vs chance 0.439 - still ~9x below chance. This statistic stays low even with the
  realistic generator because synthetic placement samples hubs from the released aggregate
  tables, not from any individual's cell set.
- **Coarse signature linkage** (the released histogram buckets: outings, distinct places, span,
  revisit): real users unique by signature whose signature occurs among the synthetic users:
  rate 0.115 vs holdout 0.119 (excess **-0.003**, previously -0.042). The rate rose to the
  holdout level, which is expected of a faithful generator; only a large positive excess would
  be suspicious.

Reading: the attack still does not beat chance anywhere. The previous run's caveat - that the
sub-chance margins partly reflected the generator's realism gap - is now resolved empirically:
with the realistic generator the population-level statistics (profile distance, signature rate)
rose to their chance/holdout references as predicted, and none crossed towards the split-half
copy line (re-identification 0.050 vs chance 0.439; nearest-profile ratio 1.08, not << 1;
signature excess -0.003). The one-to-one Hungarian matching still does better than a random
pairing (0.065 vs 0.010) because synthetic users visit the same popular hubs - popularity
structure, not identity. The artifact-mediated pipeline gives no channel for memorisation short
of the tables themselves leaking, which section 1 bounds.

## 4. Caveats of the attacks themselves

- These attacks certify nothing: a stronger attack can exist. The DP statement is the ledger.
- The membership-inference vectors take the released (post-selection) vocabularies as given,
  which matches the attacker's knowledge but ignores the tiny extra signal in WHICH child keys
  were released (that selection is post-processing of noisy parents, so it is covered by the
  accounting).
- The linkage references are estimated from 139 users (126 synthetic after tokenization), so
  rates carry sampling noise of a few hundredths; the conclusions rest on order-of-magnitude
  gaps (0.05 vs 0.44), not on the third decimal.
- Monte Carlo MI uses 400 worlds; per-user AUCs carry +-0.03 of simulation noise.

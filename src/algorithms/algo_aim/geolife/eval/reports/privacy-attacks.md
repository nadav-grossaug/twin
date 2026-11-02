# Empirical privacy attacks on the GeoLife artifact and synthetic data (2026-11-02)

**EMPIRICAL DIAGNOSTIC - not the differential privacy guarantee.** The guarantee is the zCDP
ledger in each artifact's `privacy.json` (user-level adjacency, rho 0.5395, epsilon 6.000 at
delta 1e-6); everything here asks what concrete attackers achieve against this release, as a
sanity check of that accounting and of the generator. Code: `src/geolife/eval/attacks.py`
(design notes in its docstring), runner `geolife/eval/run_attacks.py`, machine-readable results
`geolife/eval/results/attacks/{attacks.json,seed_averaging.csv,summary.md}`. PLAN step 7.

Setup: the 139 real users the artifacts were fitted on (same tokenizer, pseudonyms and caps as
stage 1, sensor views included, so the attack vectors are exactly the vectors the mechanism
summed); the 3 committed artifacts `outings_v2_seed0-2`; synthetic side for linkage = 139 users
generated from `outings_v2_seed0` into the source city (`--city beijing`, 81.1M points), so real
and synthetic share the map frame.

## 1. Membership inference at the user level (the DP unit), against the artifact

The attacker holds the released noisy tables and a target user's complete data and must decide
whether the target was measured.

- **Informed attacker (worst case, exact).** If the attacker also knows every other user's data,
  the optimal test faces a Gaussian location problem with separation
  Q_u = sum_t ||v_{u,t}||^2 / sigma_t^2, so the per-user AUC is Phi(sqrt(Q_u/2)) with no
  simulation error. Over the 139 real users: **Q p50 0.519, p90 0.698, max 0.751**, against the
  zCDP ceiling 2*rho = 1.079 (a user would have to saturate every cap to reach it). Per-user AUC:
  **p50 0.695, p90 0.723, max 0.730** (ceiling 0.769). The worst user's TPR is **0.072 at FPR
  0.01** and 0.339 at FPR 0.1: even the strongest adversary against the most exposed user gets a
  weak test. The presence tables dominate the worst user's Q (trav_users_r7 0.24,
  place_users_r7 0.11, place_users_r6 0.07); every pmf/one-hot table contributes < 0.025.
- **Uninformed attacker (realistic).** Matched-filter score <y, v_u>/sigma^2 over 400 simulated
  releases of the real mechanism (random half of the users in, ledger sigmas), without knowledge
  of the other users: per-user AUC **p50 0.598, p90 0.634, max 0.704** - clearly below the
  informed ceiling, because the other users' mass acts as additional noise.
- Reading: AUC 0.5 is chance; the observed levels are what epsilon 6 is expected to allow and
  nothing above it. The empirical maximum staying under the analytic bound (0.730 < 0.769, with
  Q_max 0.751 < 1.079) is an end-to-end consistency check of sensitivities and sigmas.

## 2. Seed averaging (why a real release is ONE seed)

The 3 committed artifacts are noise replicates of one fit. Measured on the 26 tables whose
released vocabulary is identical across seeds (the hierarchical child tiers r7/r8 differ by
seed and are excluded):

- The cross-seed per-key std matches the ledger sigma (median ratio 0.84 over the 26 tables; 3
  samples per key, so individual ratios scatter, 0.62-1.01 away from the single-key n_users
  table): the seeds are exactly sigma-noise replicates.
- Averaging the 3 seeds therefore cuts the effective noise std by **sqrt(3) = 1.73** on every
  table (e.g. outing_start: sigma 7.53 -> 4.35; sensor_interval: 6.61 -> 3.82), and lifts every
  per-table signal score by the same factor - e.g. cat_time_visits goes from 2.7 (below the
  fit threshold 3, "falls back") to 4.6 (usable): averaging recovers tables the single-seed
  design deliberately left in the noise.
- The honest accounting of releasing all three is 3 * rho = 1.619, i.e. **epsilon 11.08** at
  delta 1e-6, not 6. The informed-MI separation Q scales the same way (worst-case per-user AUC
  Phi(sqrt(3Q/2)): max 0.730 -> 0.856).
- Per-table numbers: `geolife/eval/results/attacks/seed_averaging.csv`.

Consequence, now implemented: `fit_artifacts.py --release` emits exactly one seed and stamps the
policy into `privacy.json` (`learn_v2.ONE_SEED_CAVEAT`). The committed trio stays as a DEV set
and must never be published together (also recorded in `geolife/transfer/README.md` and the
ledger review).

## 3. Linkage / re-identification against the synthetic data

Synthetic users are fresh samples, not transformed real users, so there is no true
correspondence to recover; the attack asks whether any synthetic user is suspiciously CLOSE to a
real user (a memorised copy). Every statistic is read against two references: the same statistic
with the synthetic side replaced by OTHER real users (chance - different people share hubs), and
the within-user split-half value (what finding a true copy of the same person would look like).
Both sides in the Beijing frame, same tokenizer.

Setup detail: 136 of the 139 synthetic users pass the tokenizer's min-activity rule.

- **Nearest-user profile distance** (log activity sizes, revisit rate, median outing shape,
  6 time-bin shares, standardised): real -> nearest synthetic median 2.92 vs real-holdout
  (user half A -> half B) median 2.09, ratio **1.40** (too_close = False). A ratio at or
  above 1 means no synthetic user sits closer to a real user than an independent real person
  would; a copy would drive it towards 0.
- **Top-10 visited-cell Jaccard** (res-8 stop cells, the most identifying map-relative feature):
  best match against the synthetic users mean **0.111** (max 0.250), chance level
  against other real users mean 0.351, copy reference (own split half) mean 0.397.
  **Re-identification rate** (a synthetic user resembles the target at least as much as the
  target's own other half): **0.036** vs chance 0.439. Hungarian one-to-one matching
  mean Jaccard 0.060 vs random matching 0.009.
- **Coarse signature linkage** (the released histogram buckets: outings, distinct places, span,
  revisit): real users unique by signature whose signature occurs among the synthetic users:
  rate 0.077 vs holdout 0.119 (excess **-0.042**). Matching a coarse bucket is
  expected of a faithful generator; only a large positive excess would be suspicious.

Reading: the attack does not beat chance anywhere - every synthetic-side statistic sits BELOW
both references. A real user's best synthetic match (Jaccard 0.111) is far less similar than an
arbitrary OTHER real person (0.351), the re-identification rate is 12x below its chance level,
and the signature excess is negative. Two honest footnotes: (a) the one-to-one Hungarian matching
does better than a random pairing (0.060 vs 0.009) because synthetic users visit the same popular
hubs - popularity structure, not identity; (b) the gap BELOW chance partly reflects the
generator's realism deficit (PLAN step 5: too many distinct places, hub structure not matched),
so these numbers must be re-run when the generator improves - with a faithful generator the
synthetic statistics should approach the across-user chance level, and only EXCEEDING it
(approaching the split-half copy line) would signal memorisation. The artifact-mediated pipeline
gives no channel for that short of the tables themselves leaking, which section 1 bounds.

## 4. Caveats of the attacks themselves

- These attacks certify nothing: a stronger attack can exist. The DP statement is the ledger.
- The membership-inference vectors take the released (post-selection) vocabularies as given,
  which matches the attacker's knowledge but ignores the tiny extra signal in WHICH child keys
  were released (that selection is post-processing of noisy parents, so it is covered by the
  accounting).
- The linkage attack is limited by the generator's current realism gap (PLAN section 8): a
  generator that matched the real marginals better would raise the chance baselines, not the
  copy risk, as long as users are sampled from released aggregates only.
- Monte Carlo MI uses 400 worlds; per-user AUCs carry +-0.03 of simulation noise.

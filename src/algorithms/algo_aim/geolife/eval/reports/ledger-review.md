# Ledger review of the committed GeoLife artifacts (2026-11-02)

Scope: the three committed artifacts `geolife/transfer/artifacts/outings_v2_seed0-2` (29 tables
each, learner `geolife-outings-v2`). Reviewer tooling: `geolife/eval/review_ledger.py` (re-derives
every stated number independently of the built-in validator) plus a manual read of the caveats
against the code. Machine table: `geolife/eval/results/attacks/ledger_review.md`. PLAN step 7.

## What was checked, and the result

| Check | How | Result |
|---|---|---|
| Manifest hashes | `artifact_cli verify` + `load_artifact` (SHA-256 per file, unlisted files rejected) | **OK**, 66 files per artifact, all three seeds |
| Every released table has a ledger entry | entry set vs table set, per seed | **OK**, 29 = 29 |
| Entry sigma / sensitivity / rho match the table meta | per-field, rel tol 1e-12 | **OK** |
| rho internally consistent | recompute rho = sens^2 / (2 sigma^2) per table | **OK** |
| Stated sensitivity equals the schema clipping bound | `schema_v2.l2_bound` (presence sqrt(m), pmf/one-hot 1) | **OK** |
| rho sums to the stated total | sum of entries vs `rho_total` 0.539548 | **OK** |
| epsilon(delta) conversion | eps = rho + 2 sqrt(rho ln(1/delta)) = 6.000 at delta 1e-6 | **OK** |
| Budget split sanity | behaviour 0.431638 + sensor 0.107910 = 80% / 20% (D20) | **OK** (exact) |
| Caps block vs the code constants | `MAX_OUTINGS`, `MAX_STOP_CELLS`, `MAX_TRAV_CELLS`, per-tier m, `MAX_SENSOR_STEPS` | **OK** |
| Leakage guard + provenance allowlist | `validate_no_raw_leakage` (coordinates, uid patterns, allowlisted json keys) | **OK**, no flags |
| `index.json` agreement | epsilon per row vs each ledger | **OK** |
| Adjacency | stated `user` everywhere | **OK** |

No numeric error was found in any of the three ledgers.

## Caveat completeness (public settings actually used in the code)

Covered by the committed caveats (20 per artifact): outing gap, real-outing rule, minimum
activity, stop rule, snap radius 150 m, time-bin design (D19), H3 hierarchy, per-user caps,
histogram edges and per-table key caps, declared priorities / tau / floor, per-user determinism,
hierarchical child release, pmf weighting, detour proxy, dropped-table list, TOD3->fine split
assumption, co-visit non-release, all sensor-table settings (edges, cap, gap threshold, thinning,
80/20 split), per-release sensor fallbacks, floating-point Gaussian.

**Gaps found** (grep of `measure_v2.py`, `tokenize/outings.py`, `learn_v2.py` for hard-coded
data-dependent choices; fixed in `measure_v2.CAVEATS` for future artifacts, the three committed
privacy.json files are NOT regenerated):

1. **Analysis region.** The Beijing core bbox (`BEIJING_CORE`, in `preprocessing.json` as
   `region`) fixes the public H3 vocabulary and which points/users contribute, and was chosen by
   looking at where the data is. It was not named in the caveats.
2. **Mode heuristic and spike filter.** The tokenizer's median-speed mode cuts (walk<2 / bike<4 /
   bus<7 / car<15 m/s, label vote 0.5) and the 70 m/s speed-spike point filter shape
   `outing_mode` and the per-mode sensor tables; EDA-informed, previously only alluded to.
3. **Multi-seed release not stated.** The three committed artifacts are noise replicates of one
   fit; nothing in their privacy.json says that averaging them cuts the noise std by sqrt(3), or
   that the honest accounting of all three is 3 * rho = 1.619 (epsilon 11.06 at delta 1e-6, not
   6). Quantified in `geolife/eval/reports/privacy-attacks.md`. Going forward
   `fit_artifacts.py --release` emits exactly one seed and stamps the policy caveat
   (`learn_v2.ONE_SEED_CAVEAT`); the committed trio stays as a DEV set and must not be published
   together.

## Authenticity (not a ledger error, a hardening note)

The committed artifacts carry integrity (SHA-256 manifest) but no authenticity: anyone who can
write to the transfer folder can rewrite tables *and* manifest. Optional HMAC-SHA256 signing over
the manifest was added (`artifact_cli sign/verify --key`, `fit_artifacts.py --sign-key`,
`generate_synthetic.py --key`); the committed artifacts remain valid unsigned. HMAC is shared-key
authenticity, not a public-key signature: the key is generated inside the air gap and carried to
the outside verifier over the same trusted channel that moves the code, never stored next to the
artifact.

## Code-level findings (no artifact change needed)

- `validate_no_raw_leakage` already re-checks rho consistency, the rho sum, the epsilon
  conversion and entry/table agreement at load time; the review script confirms them
  independently (different formulas path). No discrepancy.
- `n_keys` in each ledger entry equals the released vocabulary length for all 29 tables,
  including the hierarchical child tiers (whose vocabulary depends only on noisy parents).
- Machine-specific: the `artifact_cli` subprocess aborts AT EXIT on this machine (pyarrow
  teardown, after printing `OK`); verification itself returns rc 0 in-process. Pre-existing,
  also seen in `test_clean_room`.

# W2 generator: what was tried on the real-scale problem (2026-11-03)

Throwaway dev notes for `src/graph_aim_walks/outside/` (plan `docs/snap/walks-plan.md` section 4). Real-scale input:
the W1 artifact `snap/transfer/artifacts_walks/graph_aim_walks_seed0` (SNAP, 4,039 nodes, 88,234 edges, tables from
200,000 walks of 32 steps); scripts here: `run_real.py` (full run, prints the energy table), `debug_batch.py`
(screen vs validation of single batches), `init_grid.py` (walk energy of v2 inits with pinned regions),
`make_input.py` (pre-W1 input builder, superseded by the W1 artifact). Caches in `cache/` (not committed).
Energies below are the walk-table energy in sampling-noise units on a FRESH 40k pool (12,651 informative cells; a
matched graph would sit near that number).

## Starting point

The v2 initializer (`generate_v2_with_info`, 25 s, R = 4, q = 0.9) has walk energy **1.51M**, ~120 noise units per
cell: syn walks return far more often (dense phase-one blocks: return at t=32 0.021 vs 0.002 real), cover less, and
lose attribute coherence with distance (same-school probability at lag 16: 0.35 vs 0.44 real; mean squared locale
share in 32-step windows 0.72 vs 0.78). Pinning other regions (R in 8..64, q 0.75-0.9) changes the init energy by
<10% (`init_grid.py`): not the lever.

## What did not work

1. **Per-proposal screen on the walk pool** (plan 4.4 as written: re-simulate the walks through a swap, score
   their change with the energy gradient, accept negatives, exact check on the same pool). The pool energy falls,
   the fresh-pool energy does not: a swap re-routes ~30 walks (unsorted rows) and the sign of its score is set by
   the re-drawn continuations, not by the graph; selecting on that noise fits the pool. Predicted -47k per batch,
   exact on the selection pool -27k, on an independent pool -8k, then positive after three batches.
2. **Sorted neighbour rows** (the `walkstats.csr` layout): a swap shifts every entry between the old and new
   neighbour, so every walk VISITING one of the four endpoints changes (~3,000 per swap at SNAP scale instead of
   ~30). Kept as `WalkPool(sorted_rows=True)` (tested, exact), not used by the generator; the unsorted layout
   is the same Markov chain with a permuted map from `U` to the neighbour.
3. **Validating whole screened batches** (~800 swaps at once, halving on rejection): after two cycles almost
   every batch was rejected and the structure chain stalled. Chunks of 150 (most promising first) fixed it.
4. **Share surrogate without a floor on the concentration gap**: the scale `2 E / gap` reached -1.1e8 for locale
   (gap ~0) and drove the structure chain to destroy coherence; every chunk was rejected. Fixed with
   `SURROGATE_MIN_GAP = 0.02` (attributes whose concentration already matches get no surrogate).
5. **Base terms at their noise units** (`(C - T)^2 / (T + 1)`): the walk tables outbid them. After 16 cycles at
   weight 1 the mean clustering of every degree bin had fallen from ~0.6 to ~0.2-0.3 (clustering energy 2.7 ->
   845, only 9 cells), and `attr_edge_pair` rose 4-8x. A uniform weight of 5 or 20 did not protect clustering
   (982 / 1157) and slowed the walk tables a lot (weight 20: 0.83M after 14 cycles instead of 0.57M). Per-kind
   weights (`BASE_WEIGHTS`) instead; clustering 300 / jd 1 / edge pairs 3 still let clustering fall to 266 and
   the edge pairs rise 1.5-3x; the shipped clustering 1000 / jd 3 / edge pairs and 2-way 10 hold the edge pairs
   and 2-way at or below their init and roughly halve the joint-degree energy, at ~20% less walk-table gain.
   Clustering still drifts: mean local clustering per degree bin ends 0.1-0.2 below the real values (energy 104
   in its own units, 9 cells). Next step if that matters: a stricter unit for clustering (per-node mean) or a
   hard band.
6. `pool_screen = 1` on top of the analytic screen (the noisy pool score for return / coverage / share /
   dominance): overfits the selection pool again (selection -16k vs validation -1.5k per chunk). Off by default.

## What works (shipped defaults)

- **Two pools per cycle**: selection pool A (gradients) and validation pool B (exact accept / reject of chunks);
  the energies reported per cycle are measured on a freshly drawn B before the cycle touches the graph.
- **Closed-form screen for the lag tables** (`analytic.LagGradient`): exact gradient of the EXPECTED
  `degree_lag` / `attr_lag` tables in the transition matrix (finite-difference check: corr 0.98; expected vs a
  200k-walk pool: max abs diff 5e-4). Validation then becomes consistent (selection and validation pool agree).
- **Share surrogate**: same-value probability at every lag for share attributes whose concentration is off.
- **Proposals**: 50% class-preserving partners (d of the class of b: degree bin x hometown+location), 25%
  two-hop partners (closes a triangle), 25% uniform; batches of 6,000 proposals, chunks of 150 (structure) / 60
  (labels).
- Label chain: exact base deltas + first-order expected `attr_lag` change + pool scores of `share` / `dominance`
  (paths fixed, so no re-simulation noise).

## Real-scale trajectories (fresh-pool walk energy, seed 0, cached v2 init)

| run | settings | after 5 cycles | final | time |
|---|---|---|---|---|
| r1 | pool screen only, 1 sweep each | 0.84M (stalling) | 0.80M (5 cycles) | 4.5 min |
| r3a | analytic screen + chunks, struct 2 / label 1 | 0.77M | 0.74M (6 cycles; structure stalled: surrogate bug) | 4 min |
| r4a | + surrogate floor, base weight 1 | 0.70M | 0.44M (16 cycles), clustering broken | 9.7 min |
| r4b | struct 1 / label 2 | 0.69M (cycle 3) | ~0.47M (10 cycles) | 9.5 min |
| r5a/b | uniform base weight 5 / 20 | 0.79M / 1.08M | 0.57M / 0.83M (14 cycles) | 7.7 min |
| r6a | per-kind weights clu 300 / jd 1 / ep 3, 18 cycles | 0.65M | 0.43M; clustering 263, jd 39k | 9.5 min |
| r6b | clu 1000 / jd 1 / ep 3 | 0.67M | 0.45M; clustering 84, jd 66k | 9 min |
| r7 | clu 300 / jd 3 / ep 10 / 2way 10, 20 cycles | 0.79M (c4) | 0.49M; clustering 340, jd 15k, ep at init | 10.3 min |

## Shipped defaults, real init (no cache), 20 cycles: `final2_seed0` / `final2_seed1`

Walk energy on a fresh 40k pool, v2 init -> final (seed 0): **1.51M -> 0.52M** (seed 1: 1.47M -> 0.38M); time per
graph **626 s / 621 s** including the 26 s initializer (two runs side by side, 8 threads each).

| table | cells | v2 init | final |
|---|---|---|---|
| return | 31 | 63,372 | 3,566 |
| coverage | 31 | 65,921 | 6,721 |
| degree_lag | 281 | 172,404 | 62,582 |
| attr_lag:school | 288 | 38,283 | 23,640 |
| attr_lag:hometown | 288 | 50,496 | 26,564 |
| attr_lag:location | 392 | 15,136 | 11,757 |
| attr_lag:hometown+location | 9,985 | 249,180 | 154,502 |
| share:gender | 86 | 26,872 | 1,544 |
| share:locale | 152 | 237,133 | 26,788 |
| share:birthday | 178 | 141,272 | 9,140 |
| share:school | 176 | 112,075 | 58,486 |
| share:employer | 157 | 27,660 | 3,411 |
| share:location | 190 | 47,407 | 28,927 |
| share:hometown | 162 | 164,816 | 70,947 |
| share:language | 111 | 17,477 | 869 |
| dominance:hometown\|location | 48 | 14,130 | 3,938 |
| dominance:hometown\|school | 48 | 34,452 | 10,686 |
| dominance:school\|locale | 47 | 35,564 | 11,419 |
| **walk total** | 12,651 | 1,513,650 | 515,485 |
| clustering_by_degree (base) | 9 | 2.7 | 104.5 |
| joint_degree (base) | 100 | 89,473 | 39,169 |
| attr_edge_pair school / hometown / location | 36 / 36 / 49 | 1,360 / 4,268 / 1,624 | 760 / 4,896 / 1,923 |
| attr_2way (28 tables) | | 5,338 | 3,844 |
| transition spectrum, sum sq gap | 32 | 0.192 | 0.045 |

Not matched: the walk tables end ~40 noise units per cell (acceptance asks ~2). Progress per cycle is still 2-3%
at the end of the budget; the largest remaining terms are the joint hometown+location lags, the hometown /
school / location shares and degree_lag.

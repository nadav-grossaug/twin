# Process log: SNAP / graph-AIM track

Orchestrator-owned log for the SNAP ego-Facebook graph track. Adult Census log: `docs/adult-census/PROCESS.md`.

## Phases and decisions (started 2026-11-01)

Plan: `docs/snap/graph-aim-plan.md`. Research: `research/graph-data-synthetic-generation.md`. Branch: `feat/snap-facebook-eda`.

| Phase | Task | Owner / branch | Status |
|---|---|---|---|
| 0 | ego-Facebook data, loader, EDA notebook | orchestrator | done, committed `0f78787` |
| 0 | Research note, plan, contract `src/graph_aim/types.py` | orchestrator | done |
| 1 | G1 data preparation | agent G1 | done, merged |
| 1 | G2 capping, accountant, measurements, fit | agent G2 | done, merged |
| 1 | G3 generator | agent G3 | done, merged (fixed after orchestrator test run) |
| 1 | G4 artifact I/O, inspect, CLI | agent G4 | done, merged |
| 1 | G5 evaluation, baselines, attacks | agent G5 | done, merged |
| 2 | G6 integration, sweep, `02_graph_aim_results.ipynb`, `snap/reports/graph_aim_v1.md` | orchestrator | done |

Decisions:

- **D7 (env): run Python through WSL with `~/snapcheck/bin/python`.** The project `.venv` was built on Linux and its `python` is a 0-byte file when seen from WSL, and its Windows `Scripts/python` fails on import. `~/snapcheck` is a throwaway venv (Python 3.14, numpy, pandas, scipy, networkx, matplotlib, nbclient, pytest) created with `uv` through the JFrog proxy. Code must stay Python 3.11 compatible.
- **D8 (privacy unit): edge-DP on structure plus attribute-row DP on attributes**, two zCDP ledgers, guarantees stated per adjacency. Node-DP is out of scope.
- **D9 (method): v1 is a fixed-workload measure-then-reconstruct method** (degree histogram, clustering by degree, joint degree, attribute marginals, attribute-group edge counts), a BTER-style sampler, and an inspectable artifact. Adaptive AIM selection and learned encoders come later.
- **D10 (attributes): names are dropped, at most 8 categorical attributes are kept.** Attribute and top-K selection use the real data and sit outside the DP guarantee in v1; the ledger lists this as a caveat.

SNAP track issues (from the first end-to-end run, 1 seed, 2026-11-01):

| # | Issue | Status |
|---|---|---|
| 3 | Sandboxed agents cannot run `wsl`; G2 tested on a Windows venv, G3/G4 could not test. Orchestrator re-ran everything in WSL. | worked around: `wsl env PYTHONPATH=src '~/snapcheck/bin/python' -m pytest` was accepted by some sandboxes |
| 4 | G3 emitted unordered edges (`u >= v`) and overshot clustering. Found only by the orchestrator's test run. | fixed (c048bd6) |
| 5 | Measurement names like `attr_1way:gender` were rejected by `save` as file names. Cross-package contract gap. | fixed (18c6fbb) |
| 6 | Degree cap D=100 removed about 21% of real edges and distorted structure. | fixed in round 2 (R2): degree histogram and clustering/joint-degree measured uncapped with proved sensitivities |
| 7 | Generated graph far too assortative (error 0.72). Root cause: the cap itself (capped real graph is 0.53 assortative) and, for private runs, the hub tail of the degree histogram being zeroed. | fixed (R1 rewiring, R2 uncapped measurements, R4 tail pooling); per-seed spread remains (hub count is noisy) |
| 8 | clustering_by_degree and joint_degree measurements were mostly noise at eps=4. | fixed (R2): noise/signal 3.7 to 0.24 and 1.6 to 0.24 |
| 9 | Non-private graph_aim trailed non-private BTER on structure. | partly: now on par for degree and triangles; clustering, transitivity, degeneracy (sampler overshoots, about 140 vs 115) and homophily weakness remain open |

Round 2 and 3 (2026-11-01): R1 sampler rewiring, R2 measurement redesign, R3 link-prediction utility and driver, R4 hub-tail pooling in the degree histogram; all merged and re-tested by the orchestrator (163 tests). Final 3-seed sweep in `snap/eval/results/`, notebook `snap/eval/notebooks/02_graph_aim_results.ipynb`, report `snap/reports/graph_aim_v1.md`.

- **D11 (privacy): measurements use per-measurement caps.** `degree_hist`, `clustering_by_degree`, `joint_degree` are measured uncapped with proved sensitivities (cap only for `attr_edge_pair`, 150). Data-dependent choices (Dmax, caps, bins, min degree, grid, shares) are listed as ledger caveats.
- **D12 (evaluation): link-prediction utility does not discriminate** (0.99 for every decent method); structure metrics are the comparison.
- Process note: sandboxed agents often could not run `wsl`; every merge was re-tested by the orchestrator, which found two real bugs (unordered edges in the generator, unsafe file names in `save`).

Restructure (2026-11-01, branch `refactor/inside-transfer-outside`): after an audit of what crosses from fitting to generation, the code was split into `inside` / `transfer` / `outside` zones and the single driver into three stages (see `snap/README.md`). Audit findings: the generator never read real data, but `provenance.json` carried an exact, un-noised capped-edge count (`capped_edges_by_cap`); removed and made impossible by an allowlist checked in `validate()`. The earlier "clean-room" subprocess only proved the generator runs from a copied artifact; the new stage 2 plus `test_boundary.py` additionally enforce the import boundary. Still not an OS-level sandbox.

- **D13 (structure): three zones (later four, see D14).** inside = may read the real graph; transfer = artifact and graph formats; outside = generator. Outside and transfer may import only `graph_aim.transfer`/`graph_aim.outside` plus numpy/pandas/scipy.

- **D14 (structure): fourth zone `eval/`** (`snap/eval/`, `src/graph_aim/eval/`), as planned in `CLAUDE.md`. It holds everything that needs the real and the synthetic graph side by side: `evaluate`, `baselines`, `attacks`, `utility` (moved out of `inside/`), stage 3 `score.py`, the sweep results, the diagnose scripts, the results notebook, and the new `viz_comparison.ipynb` with `viz.py` (real-against-synthetic plots, ego neighbourhood and connected-patch examples). Eval may import every zone; `test_boundary.py` checks nothing outside eval imports it. `fit_artifacts.py` no longer imports the evaluation module (it needed only the maximum degree).
- First reading of `viz_comparison` (eps 4 and non-private, seed 0): degree distribution, attribute shares and the non-private clustering curve match well; the synthetic graphs are more compact (typical path 3 hops against 4), degree mixing is concentrated on the diagonal where the real graph links low-degree people to hubs, homophily is weaker (42% same-group edges against 59%), and neighbourhood colours are mixed where real ones are coherent. These are new findings the scalar metrics hid; none is a privacy-noise effect, since the non-private graph shows them too.

- **D15 (package D of `docs/snap/attribute-coupling-plan.md`, 2026-11-02): several `attr_edge_pair` instances plus multi-channel rewiring.** Config `homophily_groups` = list of attribute groups, one measurement `attr_edge_pair:<a>` / `attr_edge_pair:<a>+<b>` per group (`homophily_attrs` stays the single-group alias with the legacy bare name; `Schema.homophily_attrs` keeps the primary group); hard limit `max_homophily_groups` (default 3) enforced in `fit.plan`. Sensitivities per instance unchanged (edge 3, attr sqrt(2) Dp); the kind's budget share is split equally among instances by the existing `privacy.calibrate`, so each extra group makes all of them noisier (ledger caveat added). Generator selects instances by kind, never by name; the primary class channel is the instance with the highest lift computed from the released table itself, and `_rewire` now takes a list of channels (per-channel `G_k x G_k` counts; swap cost = sum of per-channel linearised chi-square, each normalised by its target mass and rescaled by channel 0's mass, so a single channel is BIT-identical to before -- pinned by the fallback digests in `test_generate_attr_degree.py`). `MAX_GROUPS` = 64 moved to `transfer/artifact.py` and enforced by `validate()`; old single-instance artifacts load and validate unchanged (`FORMAT_VERSION` 1). The config stayed single-group (`homophily_groups: null`) until the decision point; the new non-private candidate ranking in `snap/inside/reports/attributes.md` (lift excluding missing labels -- the old same-group table counts none-none matches and understates homophily) gives hometown 2.70, location 2.26, school 2.08, birthday 1.81, rest <= 1.4. The decision point (plan section 11 step 5) was taken the same day: three groups shipped, budget policy in D17.

- **D16 (attribute coupling, packages B/C levers, 2026-11-02): `attr_degree` share 0.35 with 4 coarse bins (`attr_degree_bin_edges` [0, 1, 11, 113, 1201]).** At the starting config (share 0.20, 6 bins) the plan's eps=4 bar (section 10, gap closed >= 0.3 on `deg_by_value_tvd`) was met by school (.51) and birthday (.42) only -- 2 of the 6 attributes with real coupling. Sweep over share 0.35 / 4 bins / both (`tools/sweep/`): the bin count is the decisive lever (4 bins at share 0.20 already brings hometown to .34 and location to .34; share 0.35 at 6 bins does not, hometown .26, location .12). Shipped `share35_bins4`: school .54, birthday .53, hometown .40, location .40 at eps=4 -- 4 of 6 -- with no structure regression against the committed v1 baseline (`tools/regression_check.csv`; the only eps <= 8 metric beyond the v1 seed spread is homophily_error, which improved). Trade-off: the coarser table caps what can be recovered without noise -- at eps=inf the `deg_by_value_tvd` gaps drop from about 0.8 (6 bins) to 0.57-0.67, still over the 0.5 inf bar -- and employer (weakest real coupling, eta2 .012) fails its inf eta2 bar at .38. Config committed in `31da504`.
- **D17 (package D enabled, 2026-11-02): three homophily groups `[["school"], ["hometown"], ["location"]]`; `epsilon_attr` stays equal to `epsilon_edge`.** The plan section 6 budget decision: no extra attribute-row budget; the `attr_edge_pair` share splits equally over the three instances, so each instance's sigma grows by sqrt(3). Measured in `tools/sweep_d/` (X one group, Z two, Y three; Y shipped). At eps=inf, lift: hometown 1.18 -> 1.75 (mix_tvd .484 -> .372), location 1.19 -> 1.61 (.253 -> .155), school unchanged at about 1.46. Non-configured attributes stay at chance, exactly as the plan expects (locale 1.01; birthday unchanged at 1.31 -- its above-chance level comes from the attribute tree plus the new degree coupling, identical in the control). The plan section 10 bar for D (gap >= 0.5 on lift and assort at eps=inf) is met only by location (.50/.55); school plateaus at .40/.37 and hometown at .42/.38 even with full budget and zero noise -- a generator-capacity limit (rewiring moves only phase-two edges), not a budget one; confirmed by the D18 experiment. homophily_error improves .257 (v1) -> .233 at eps=4.

- **D18 (package E, plan section 7) tried and dropped (2026-11-02): attribute-aware phase-one triangle blocks.** Experiment in `tools/exp_e/` (artifacts reused from `tools/sweep_d/Y_three`, byte-identical to the shipped config; baseline regeneration matched the sweep outputs exactly). Sorting phase-one blocks by (primary group, then degree) instead of (degree, then group tie-break) — primary from the released tables only; under the shipped 3-group config the primary is **school** on all six artifacts, because the released-table lift includes the `none` cell, which dilutes hometown (eps=inf lifts: school 2.08, hometown 1.27, location 1.27) — moved the within-group triangle share of school from 0.16 to 0.62 (real 0.55, null 0.06) and closed the school homophily gap completely (gap_lift 0.40 -> 1.02 at eps=inf; hometown 0.42 -> 0.52, location 0.50 -> 0.60). But blocks at group boundaries mix degrees, so the block size (first node's degree + 1) no longer fits its members: at eps=inf degree KS 0.006 -> 0.080, clustering_abs 0.004 -> 0.025, transitivity_abs 0.003 -> 0.070, triangles_rel 0.062 -> 0.185, edge count overshoots by ~5% — all far beyond the seed spread, so the plan section 7 keep-rule fails and the generator change was reverted. Kept: the eval metric `within_group_triangle_share` in `graph_aim.eval.coupling` (+ tests). The result does confirm the hypothesis that phase-one blocks are why school/hometown homophily plateaus at ~0.4; a refined E (degree-consistent blocks within groups) is the obvious follow-up.

- **Planned (2026-11-02): real-vs-fake classifier track.** A discriminator trained on random-walk subgraphs of the real and synthetic graphs, as a stricter fidelity check than the scalar metrics and the non-discriminating link-prediction utility (D12). Shared core plus SNAP implementation; plan in `docs/realfake/PLAN.md` and `docs/realfake/snap.md`, log in `docs/realfake/PROCESS.md`. Not implemented yet.

## Attribute coupling round (v1.1, 2026-11-02)

Plan: `docs/snap/attribute-coupling-plan.md`. Branch `aim-coupling` (worktree `.claude/worktrees/aim-coupling`), agents per package, merges re-tested by the orchestrator; 335 tests pass, including the boundary test and the new generality test. The v1.1 outputs live in their own directories next to the v1 ones, which stay frozen for side-by-side comparison: `snap/transfer/artifacts_v1_1`, `snap/outside/synthetic_v1_1`, `snap/eval/results_v1_1` (`e802ff7`). Decisions of the round: D16, D17, D18 above (D15 is the round's package-D design).

| Package | Task | Status |
|---|---|---|
| A | coupling metrics in eval: lift, Newman assortativity, mixing TVD, degree-by-value TVD, eta2, gap closed vs a permuted-attribute null (`src/graph_aim/eval/coupling.py`); wired into `score.py` as `cpl_*` columns plus `coupling_reference.csv`; CLI `snap/eval/coupling_report.py` | done |
| B | measurement kind `attr_degree` (K x degree-bin table per attribute), sensitivity (2 sqrt(2), sqrt(2)) derived in `measure.py` and brute-force tested | done |
| C | generator: rake the released tables to the sampled margins -> maximum-entropy node fit -> exact-capacity degree assignment; artifacts without the kind fall back byte-identically | done |
| D | several `attr_edge_pair` instances + multi-channel rewiring (D15), enabled with three groups (D17) | done (`2d04584`, `31da504`) |
| E | attribute-aware phase-one blocks | tried and dropped (D18, `171462e`); `within_group_triangle_share` metric kept |
| - | generality test `tests/graph_aim/test_generality.py`: toy dataset with different attribute names, an ordered attribute, a custom missing label; no SNAP name appears in `src/graph_aim` | done (`a7166ff`) |
| - | v1.1 three-stage rerun into the separate dirs; docs round-up | done (`e802ff7`) |

### Before and after (the plan section 3 baseline table)

**Before** = the v1 generator, measured by ablation: the shipped artifacts regenerated with their `attr_degree` tables stripped (3-seed means at eps=4; per-seed values in `tools/coupling_before.csv`). Every attribute's coupling sits at the null level: gap_eta2 between -0.03 and 0.01, synthetic eta2 about .001 everywhere, mean degree flat at 40-46 across attribute values, lift at chance except school (about 1.5, the one steered attribute). The eps=inf columns look the same -- a model gap, not a noise cost.

| attribute | real lift | lift before | real eta2 | eta2 before | gap_eta2 | gap_deg_tvd |
|---|---|---|---|---|---|---|
| school | 2.13 | 1.50 | .112 | .001 | .01 | .03 |
| birthday | 2.10 | 1.02 | .102 | .002 | .01 | .10 |
| hometown | 2.81 | 1.00 | .096 | .001 | -.01 | -.20 |
| location | 2.24 | 1.07 | .057 | .001 | .01 | .02 |
| employer | 1.20 | 0.98 | .012 | .001 | -.03 | .14 |
| locale | 1.17 | 1.00 | .038 | .001 | .00 | .03 |
| gender | 1.08 | 1.00 | .002 | .000 | n/a | .18 |
| language | 1.04 | 0.98 | .005 | .001 | n/a | -.06 |

**After** = the shipped v1.1 config (3-seed means, `snap/eval/results_v1_1/sweep_summary.csv`; bars from plan section 10: gap closed >= 0.3 at eps=4, >= 0.5 at eps=inf, on eta2 and deg_by_value_tvd):

| attribute | gap_eta2 eps4 | gap_deg_tvd eps4 | gap_eta2 inf | gap_deg_tvd inf | lift at inf (real) |
|---|---|---|---|---|---|
| school | .68 | .56 | .61 | .67 | 1.46 (2.13) |
| birthday | .54 | .53 | .69 | .67 | 1.32 (2.10) |
| hometown | .69 | .34 | .60 | .56 | 1.75 (2.81) |
| location | .99 | .47 | .59 | .58 | 1.61 (2.24) |
| employer | 2.79 | -.02 | .39 | .52 | 1.04 (1.20) |
| locale | .85 | .12 | .60 | .57 | 1.01 (1.17) |

Four of the six coupled attributes pass both eps=4 bars; locale and employer fail deg_tvd there because their null floors (about .04) leave no room over the noise (employer's eps=4 gap_eta2 above 1 is the same instability: its real eta2, .012, is close to the null). At eps=inf everything passes except employer's eta2 (.39). Headline mean degrees (real / v1 / v1.1 at inf): birthday value 5: 99 / 44 / 84; school 538: 66 / 42 / 63; school 50: 25 / 40 / 30.

Structure at eps=4 (3 seeds): degree KS .013, triangles_rel .125, clustering_abs .048, transitivity_abs .061, assortativity_abs .132, attribute 2-way TVD .069, homophily_error .233 -- all within the v1 seed spread except homophily_error, which improved (v1: .257).

**Known cost of C (documented, not fixed): eps=inf degree assortativity error 0.155 (v1) -> 0.185** (B/C at the 4-bin config; 0.210 at 6 bins; 0.176 in the shipped three-group run). Diagnosis: v1 was already over-assortative; conditioning degrees on attributes concentrates hubs in the same (group x degree-bin) classes and phase two then over-wires hub-hub pairs. The before-ablation restores 0.155 exactly, so the cause is the conditional assignment; 4 bins partially mitigate it. A fix would target the phase-two `joint_degree` steering, not `attr_degree`. At eps=4 the error (.132) sits within v1's wide per-seed spread (0.10 / 0.24 / 0.02).

Open items:

- locale and employer fail the eps=4 deg_by_value_tvd bar: null floors about .04 leave no headroom over the noise; more budget alone will not fix it.
- school and hometown homophily plateau at gap about 0.4 (D17, D18): try degree-consistent blocks within groups -- a refined package E.
- assortativity at eps=inf (the known cost above): steer phase two by the released `joint_degree` table.
- untried eps=4 levers: an `attr_degree_attrs` restriction, and a sweep of 5-bin midpoints.
- DP selection of `homophily_groups` (exponential mechanism over lift) stays a later option; the current choice is a ledger caveat.

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

- **Planned (2026-11-02): real-vs-fake classifier track.** A discriminator trained on random-walk subgraphs of the real and synthetic graphs, as a stricter fidelity check than the scalar metrics and the non-discriminating link-prediction utility (D12). Shared core plus SNAP implementation; plan in `docs/realfake/PLAN.md` and `docs/realfake/snap.md`, log in `docs/realfake/PROCESS.md`. Implemented the same day, see D15.

- **D15 (evaluation, 2026-11-02): real-vs-fake classifier implemented and run; every graph_aim variant is near-perfectly detectable at walk level.** Core `src/realfake/`, SNAP sampler/featurizer `src/graph_aim/eval/discriminator.py`, stage-3 CLI `snap/eval/discriminate.py` (branch `realfake`). Controls clean (real-vs-real AUC 0.515/0.528 logreg/gbm, permuted labels ~0.5, Erdos-Renyi and Chung-Lu detected at 1.0000), but gradient boosting separates real from synthetic walks at AUC 0.9994-0.9999 for every variant from eps 0.5 to inf and nocap_nonprivate; logreg alone reaches 0.993-0.997. The saturated AUC hides the privacy sweep; the permutation importance shows it instead: the attributes feature group dominates everywhere and falls monotonically with budget (auc_drop 0.43 at eps 0.5 -> 0.24 at inf), degree second, structure and size negligible. Consistent with the D14 homophily/mixing gaps, but the attributes group also contains the exact `none`/`other` share features, a candidate generation artifact — splitting the group per feature is the open follow-up before assigning the cause (`docs/realfake/PROCESS.md` issue 1). Results `snap/eval/results/discriminator*.csv`, notebook `snap/eval/notebooks/viz_discriminator.ipynb`, report `snap/reports/discriminator_v1.md`. The boundary test now also forbids importing `realfake` outside the eval zone.

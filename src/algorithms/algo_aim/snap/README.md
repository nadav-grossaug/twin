# SNAP ego-Facebook: inside / transfer / outside / eval

Fit on the real graph **inside**, generate synthetic data **outside**, move only readable, noise-protected artifacts
between the two, and compare real with synthetic in **eval**. The folders show which side a file is on.

```
 INSIDE  (may read the real graph)       TRANSFER (what moves)        OUTSIDE (never sees real data)
 ───────────────────────────────         ─────────────────────        ──────────────────────────────
 data/            raw SNAP files          transfer/artifacts/          outside/generate_synthetic.py
 prepare.py       real -> clean graph      <variant>_seed<k>/  ──────►  (STAGE 2) reads artifacts, writes
 fit_artifacts.py (STAGE 1) fit + noise ──► manifest, schema, privacy,  outside/synthetic/*.npz
                                            measurements/                       │
 EVAL  (needs real AND synthetic; inside-only zone)                             │ synthetic graphs
 ────────────────────────────────────────────────────                           │ come back only
 score.py (STAGE 3), notebooks/viz_comparison.ipynb, results/  ◄────────────────┘ to be compared
```

| Zone | Folders | May read the real graph? | Contains |
|---|---|---|---|
| **inside** | `snap/inside/`, `src/graph_aim/inside/` | yes | data, preparation, fitting (measure + noise + privacy accounting) |
| **transfer** | `snap/transfer/`, `src/graph_aim/transfer/` | no | the artifact files, and the code defining their format (types, artifact save/load/validate/inspect, graph file format) |
| **outside** | `snap/outside/`, `src/graph_aim/outside/` | **no, enforced** | the generator, the stage-2 script, the synthetic graphs it writes |
| **eval** | `snap/eval/`, `src/graph_aim/eval/` | yes (needs both graphs) | metrics, baselines, empirical attacks, utility, comparison plots and notebooks (`viz_comparison`), result tables. May import the other zones; none of them may import it |

## Generate synthetic data outside

You need: Python >= 3.11 with numpy, pandas, scipy (nothing else), this code, and an artifact folder.

1. **Copy over** (and nothing else): `src/graph_aim/__init__.py`, `src/graph_aim/transfer/`, `src/graph_aim/outside/`,
   `snap/outside/generate_synthetic.py`, and the artifact folders you want, e.g. `snap/transfer/artifacts/graph_aim_eps4_seed0/`
   (plus `snap/transfer/artifacts/index.json` if you use the script).
2. **Generate** (only `src` on the path, so nothing from `inside` is importable):
   ```bash
   PYTHONPATH=src python snap/outside/generate_synthetic.py [--method v1|v2] [--only graph_aim_eps4] [--artifacts DIR] [--out DIR]
   ```
   Output: `snap/outside/synthetic/<variant>_seed<k>.npz`. Loading an artifact verifies its SHA-256 hashes.
   `--method v2` (composition round, `docs/snap/composition-plan.md`) runs the v1 generator and then the label chain of
   `graph_aim.outside.refine` against the artifact's `attr_neigh_simpson` tables (`generate_v2` in Python); on an
   artifact without them it is byte-identical to v1.
3. **Or from Python, for one artifact** (no index file needed):
   ```python
   from graph_aim.transfer.artifact import load
   from graph_aim.outside.generate import generate
   from graph_aim.transfer.graph_io import save_attributed_graph   # load_attributed_graph reads it back

   g = generate(load("snap/transfer/artifacts/graph_aim_eps4_seed0"), n=None, seed=0)  # n=None: the artifact's node count
   save_attributed_graph(g, "synthetic.npz")
   g.edges   # (m, 2) int array, u < v        g.attrs  # DataFrame of integer codes
   g.domains # {column: [labels]}; label of a code: g.domains[c][code]   (columns are alphabetical)
   ```
4. **Look at an artifact** (budgets, noise, caveats): `PYTHONPATH=src python -m graph_aim.transfer.inspect <artifact dir>`

Other seeds give other graphs from the same artifact (`seed=`); the artifact's own privacy guarantee does not change.

## Run the whole pipeline (repo root)

Each round writes to its own directories; `artifacts/`, `synthetic/` and `results/` hold the frozen v1 round, the
`*_v1_1` directories the attribute-coupling round, and the `*_v2` directories the composition round (its own config
`inside/config/graph_aim_v2.json` and generator method). Without the flags the stages use the v1 directories.

```bash
# v2 (composition round)
PYTHONPATH=src:snap/inside python snap/inside/fit_artifacts.py --seeds 3 --config snap/inside/config/graph_aim_v2.json --out snap/transfer/artifacts_v2   # STAGE 1 inside
PYTHONPATH=src            python snap/outside/generate_synthetic.py --method v2 --artifacts snap/transfer/artifacts_v2 --out snap/outside/synthetic_v2     # STAGE 2 outside
PYTHONPATH=src:snap/inside:snap/eval python snap/eval/score.py --seeds 3 --config snap/inside/config/graph_aim_v2.json --synthetic snap/outside/synthetic_v2 --out snap/eval/results_v2   # STAGE 3 eval
PYTHONPATH=src:snap/inside:snap/eval python snap/eval/discriminate.py --config snap/inside/config/graph_aim_v2.json --synthetic snap/outside/synthetic_v2 --out snap/eval/results_v2     # real-vs-fake
# v1.1 (attribute coupling)
PYTHONPATH=src:snap/inside python snap/inside/fit_artifacts.py --seeds 3 --out snap/transfer/artifacts_v1_1                                  # STAGE 1 inside
PYTHONPATH=src            python snap/outside/generate_synthetic.py --artifacts snap/transfer/artifacts_v1_1 --out snap/outside/synthetic_v1_1  # STAGE 2 outside
PYTHONPATH=src            python snap/eval/score.py --seeds 3 --synthetic snap/outside/synthetic_v1_1 --out snap/eval/results_v1_1              # STAGE 3 eval
```

Then open `snap/eval/notebooks/viz_comparison.ipynb` for the real-against-synthetic plots.

## Add features inside

Rule of thumb: code that reads the real graph goes in `inside/`; the file formats go in `transfer/`; code that only
turns an artifact into a graph goes in `outside/`. Anything un-noised must not leave: only `measurements/` carry data,
and `validate()` rejects unlisted provenance keys.

**A new node attribute** (e.g. another profile field):
1. Add `(name, feature-name prefix, K)` to `DEFAULT_SPECS` in `snap/inside/prepare.py` (at most 8 attributes, each domain
   at most 8 values including `other`/`none`), or edit `inside/attribute_policy.json` and use `--keep-policy`. To drop a
   sensitive field add it to `DROPPED_FIELDS`.
2. `PYTHONPATH=src:snap/inside python snap/inside/prepare.py` (rewrites the policy, `inside/reports/attributes.md` and the
   processed graph), then re-run the three stages.
3. Nothing else changes: every attribute gets 1-way and pairwise tables automatically (pairs grow quadratically and
   share the budget). Use it for homophily by setting `homophily_attrs` in `inside/config/graph_aim_v1.json` (keep the group
   count small, the noise grows with it), or several groups at once with `homophily_groups` (one `attr_edge_pair:<a>+<b>`
   instance per group, at most `max_homophily_groups`; the kind's budget share is split equally among them, so each extra
   group makes all of them noisier). Candidate ranking: `inside/reports/attributes.md`.

**A new statistic** (a new kind of table in the artifact):
1. `src/graph_aim/inside/measure.py`: add the kind to `KINDS`, emit a `Spec` in `workload()`, compute it from the graph in
   `exact_values()`, and return its L2 sensitivity under edge and attribute-row changes in `sensitivity()` (write the
   proof in the docstring).
2. `inside/config/graph_aim_v1.json` (or the round's config, e.g. `graph_aim_v2.json`): give the kind a weight under
   `shares`. A kind emitted only when its share is positive, and appended after the existing kinds, keeps older
   configs byte-identical (`tests/graph_aim/inside/test_fit.py` pins the v1 config's fitted values).
3. `src/graph_aim/transfer/types.py` (document shape and meaning in the `Measurement` docstring) and
   `transfer/artifact.py` (`_KINDS` and the shape check in `_validate_measurement`).
4. `src/graph_aim/outside/generate.py` (or, for a statistic the v2 chain should match, an energy term with a change
   statistic in `outside/refine.py`): use it; otherwise it is released but ignored. The two v2 kinds are worked examples:
   `attr_neigh_simpson` (per degree bin, the sum of the neighbourhood Simpson index on the graph capped at its own `cap`;
   sensitivity `(6 sqrt(2), 2 cap / min_degree)`) and `spectrum` (top-k adjacency eigenvalues, `(sqrt(2), 0)` by Mirsky's
   inequality).
5. Tests: add it to the brute-force sensitivity tests in `tests/graph_aim/inside/test_measure.py` (every edge toggle and
   attribute change must stay within the declared bound). This test is the evidence behind the privacy claim.
6. Re-run the three stages; `tests/graph_aim/test_boundary.py` checks the outside code still imports only `transfer`/`outside`.

Run all tests: `PYTHONPATH=src:snap/inside:snap/eval python -m pytest tests/graph_aim -q`

## Boundary rules and what they do not cover

1. **Outside** code (`src/graph_aim/outside/`, `snap/outside/*.py`) may import only the standard library,
   numpy/pandas/scipy, `graph_aim.transfer`, `graph_aim.outside` (and no `os`, `subprocess`, `socket`, `pickle`, ...). **Transfer** code: the same, minus `outside`.
2. The loader verifies a SHA-256 per file; `validate()` rejects any provenance key not on an allowlist.
3. **Eval** may import every other zone; nothing in inside, transfer or outside may import eval.
4. `tests/graph_aim/test_boundary.py` checks 1 to 3 and runs the real stage 2 in a subprocess with only `src` importable.

This is not an OS sandbox: a process of the same user could still open the real data by path. Also not protected by the
noise: the **public settings** chosen by looking at the real graph (attribute set and labels, maximum degree, bin edges,
caps, budget shares); each artifact lists them under `caveats`. Synthetic graphs return to eval only to be compared with the real one.
The non-private baselines (Erdos-Renyi, Chung-Lu, DC-SBM, BTER) are fitted to the real graph in `score.py` as reference points.

## Where to look

| I want to see | Open |
|---|---|
| budget and caveats of an artifact | `transfer/artifacts/<variant>_seed0/privacy.json`, or `graph_aim.transfer.inspect` |
| the noisy statistics | `transfer/artifacts/<variant>_seed0/measurements/` |
| a generated graph | `outside/synthetic_v1_1/<variant>_seed<k>.npz` (v1.1), `outside/synthetic/` (frozen v1) |
| how real and synthetic compare (plots) | `eval/notebooks/viz_comparison.ipynb` |
| how good it is (numbers) | `eval/notebooks/02_graph_aim_results.ipynb`, `reports/graph_aim_v1.md`, `eval/results_v1_1/*.csv` (v1.1), `eval/results/*.csv` (frozen v1) |
| attribute coupling (degree x attribute, homophily) | `cpl_*` columns in `eval/results_v1_1/sweep_summary.csv`; reference rows (real, permuted null, non-private) in `eval/results_v1_1/coupling_reference.csv` |
| coupling of one graph, ad hoc | `PYTHONPATH=src python snap/eval/coupling_report.py --real PATH --syn PATH [--out CSV]` |
| raw graph and its preparation | `inside/data/`, `inside/prepare.py`, `inside/reports/attributes.md` |
| development history | `docs/snap/PROCESS.md`, `docs/snap/graph-aim-plan.md` |

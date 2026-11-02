# graph_aim

Differentially private synthetic **attributed graphs** from an inspectable artifact. This is the dataset-independent
method; a dataset's data, config, run scripts and results live in its own folder (for SNAP: `snap/`, see `snap/README.md`).

```
AttributedGraph (real) ──► inside: fit ──► Artifact (noisy tables, on disk) ──► outside: generate ──► AttributedGraph (synthetic)
                                                                                                            │
                                    eval: compares the real and the synthetic graph side by side  ◄─────────┘
```

The code is split into four zones. The split is the point: it shows what may touch the real graph, what crosses to the
other side, what runs without ever seeing it, and what needs both.

| Zone | May read the real graph? | Module | What it does |
|---|---|---|---|
| **inside/** | yes | `capping.py` | stable degree cap (a bounded-influence projection of the edge set) |
| | | `privacy.py` | Gaussian noise, zCDP accounting, budget calibration, privacy ledger |
| | | `measure.py` | the statistics that get released, their proved sensitivities, noise and post-processing |
| | | `fit.py` | graph + config -> `Artifact` |
| **transfer/** | no | `types.py` | the contract: `AttributedGraph`, `Measurement`, `PrivacyLedger`, `Schema`, `Artifact` |
| | | `artifact.py` | save / load (SHA-256 verified) / validate the artifact; provenance allowlist |
| | | `inspect.py` | readable report and CLI: `python -m graph_aim.transfer.inspect <artifact dir>` |
| | | `graph_io.py` | `.npz` file format for an `AttributedGraph` |
| **outside/** | **no, enforced** | `generate.py` | `generate(artifact, n=None, seed=0)` -> synthetic `AttributedGraph`, from the artifact only: attributes (Chow-Liu tree), degrees assigned conditional on attributes from the `attr_degree` tables (raked to the sampled margins, maximum-entropy fit, exact-capacity assignment; without the tables, the old unconditional fallback), structure with one degree-preserving rewiring channel per `attr_edge_pair` group |
| **eval/** | yes (needs both) | `evaluate.py` | structure and attribute metrics, `compare`, `align_columns` |
| | | `coupling.py` | attribute coupling: lift, Newman assortativity, mixing TVD, degree-by-value TVD, eta2, within-group triangle share, gap closed vs a permuted-attribute null |
| | | `viz.py` | real-against-synthetic plots: degree, clustering, degree mixing, paths and cores, attributes, homophily, small ego/patch examples |
| | | `baselines.py` | non-private reference generators (Erdos-Renyi, Chung-Lu, DC-SBM, BTER) |
| | | `attacks.py`, `utility.py` | empirical privacy diagnostics, link-prediction utility |

## Rules (checked by `tests/graph_aim/test_boundary.py`)

- `outside` may import the standard library, numpy/pandas/scipy, `graph_aim.transfer` and `graph_aim.outside`.
  `transfer` may import the same, minus `outside`. `inside` may import `transfer` and anything third-party.
  `eval` may import every zone.
- **Nothing in `inside`, `transfer` or `outside` may import `eval`.** (Evaluation needs both graphs, so it depends on the
  others, never the other way round.)
- `transfer` and `outside` may not import `os`, `subprocess`, `socket`, `shutil`, `tempfile`, `urllib`, `http`, `ctypes`,
  `multiprocessing`, `pickle`, `importlib` and similar: shipped code reads artifact files and writes graphs, nothing more.
- Nothing un-noised may leave the inside: artifacts hold only noisy tables, and `validate()` rejects provenance keys that
  are not on an allowlist. No module sits outside the four zones.

## What ships to an outside machine

`graph_aim/__init__.py`, `graph_aim/transfer/`, `graph_aim/outside/` and an artifact folder. Python >= 3.11 with numpy,
pandas, scipy. Nothing from `inside/` or `eval/`, and no dataset files. Usage: "Generate synthetic data outside" in `snap/README.md`.

## Contract in one paragraph

`fit` returns an `Artifact`: a `Schema` (public node count, attribute value labels, degree cap), a dict of `Measurement`s
(each a noisy, post-processed table with its noise scale and proved L2 sensitivities under edge and attribute-row
adjacency), a `PrivacyLedger` (zCDP per adjacency, plus caveats listing what the guarantee does not cover), and
allowlisted provenance. The shapes and meaning of each measurement kind are documented in the `Measurement` docstring in
`transfer/types.py`; `artifact.validate` checks them; `outside/generate.py` is the only consumer.

## Changing it

Add a node attribute or a new statistic: "Add features inside" in `snap/README.md`. Add a comparison plot: a function in
`eval/viz.py` that takes `(real, syns)` and returns a figure, a smoke test in `tests/graph_aim/eval/test_viz.py`, and a cell
in `snap/eval/notebooks/viz_comparison.ipynb`. Tests: `tests/graph_aim/{inside,transfer,outside,eval}/` mirror the zones
(run `PYTHONPATH=src:snap/inside:snap/eval python -m pytest tests/graph_aim -q`). The sensitivity tests in
`inside/test_measure.py` are the evidence behind the privacy claim, so every new released statistic needs one.

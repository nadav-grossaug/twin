# Attribute coupling: plan (degree x attribute, optional edge x attribute)

Spec for the next round of Graph-AIM on the SNAP track (2026-11-02). Builds on `docs/snap/graph-aim-plan.md` (v1 method, sections 3 and 11 for the privacy model and the round-2 measurement design) and `docs/snap/PROCESS.md`. Code layout and boundary rules: `CLAUDE.md`, `snap/README.md`.

Goal: the synthetic graph must keep **which kinds of people are connected, and which kinds of people have many connections**, not only the structure of the graph. The method must stay **dataset-independent**: nothing may name `school`, `birthday` or any other SNAP attribute in code. Everything that depends on the dataset lives in the attribute policy and the config.

## 1. Why: what the v1 generator loses

Measured on 2026-11-02 with an ad-hoc diagnostic (not in the repo yet; work package A turns it into tests and eval code). Real graph vs the saved synthetic graphs in `snap/outside/synthetic/`. **Lift** = share of edges whose two ends have the same value, among edges where both ends are observed (value is not `none`), divided by the degree-weighted chance level (`sum_i a_i^2`, `a_i` = share of edge endpoints in value `i`).

| attribute | real lift | synthetic eps=4 (2 seeds) | synthetic eps=inf (no noise) |
|---|---|---|---|
| hometown | 2.81 | 0.98 to 1.01 | 0.97 |
| location | 2.24 | 1.06 to 1.08 | 1.08 |
| school | 2.13 | 1.45 to 1.55 | 1.58 |
| birthday | 2.10 | 1.01 to 1.02 | 0.99 |
| employer | 1.20 | 0.98 to 0.99 | 1.06 |
| locale | 1.17 | 1.00 | 1.01 |
| gender | 1.08 | 1.00 | 1.00 |
| language | 1.04 | 0.99 | 0.97 |

Mean degree by attribute value (real | synthetic eps=4, seeds 0 and 1):

| value | real | synthetic |
|---|---|---|
| birthday `anonymized feature 5` | 99 | 45, 46 |
| birthday `none` | 36 | 45, 44 |
| school `538` | 66 | 41, 44 |
| school `50` | 25 | 48, 45 |
| school `none` | 27.5 | 47, 44 |
| school `other` | 53 | 45, 46 |

Findings:

1. **Only school has homophily**, the single attribute the `attr_edge_pair` measurement steers, and it is under-fitted even without noise. Every other attribute is at chance level (about 1.0).
2. **Degree is independent of attributes in the synthetic graph.** `generate._sample_degrees` draws degrees from `degree_hist` and shuffles them over nodes (listed under "Known simplifications" in `outside/generate.py`). Real degree depends strongly on the attribute values, including on whether a value is missing.
3. **The eps=inf row equals the eps=4 row.** This is a model limitation, not a privacy cost. The v1 homophily metric covered school only, so it hid this.
4. Caveats on the evidence: three synthetic graphs, one ad-hoc script. Real mean degrees may be inflated by the 10 ego networks and a handful of hubs (max degree 1,045). The `birthday` and other values are anonymized feature ids with **no order**: "similar age" cannot be modelled on this dataset, only "same value". Gender is weak here (lift 1.08).

## 2. Scope and design principles

| Package | Status | What |
|---|---|---|
| A | **must** | Coupling metrics in the eval zone, generic over the schema |
| B | **must** | New measurement kind `attr_degree` (degree x attribute table), one per attribute like the 1-way and 2-way tables |
| C | **must** | Generator adaptation: assign degrees to nodes conditional on their attributes |
| D | optional | Edge x attribute homophily for a small, configured set of attribute groups, plus multi-channel rewiring in the generator |
| E | optional experiment | Attribute-aware triangle blocks in the generator |

Generalization rules (every package must satisfy them):

- **Tables are generated from the schema by rule**, like `attr_1way` and `attr_2way` today: one `attr_degree:<attr>` for every attribute in `Schema.domains` (restrictable by config, default all). No attribute is named in code.
- **Domain sizes, number of attributes, and number of degree bins are parameters**, bounded only by the existing limits (at most 8 attributes, at most 8 values each).
- **Dataset-specific knowledge goes in the policy/config**, never in code: which label means "missing" (`missing_label`, default `none`), whether an attribute is ordered (`ordered`, default false; if true the domain order is the value order), which attribute groups get edge tables (`homophily_groups`), the public degree bins.
- **Nothing in the artifact is dataset-specific beyond the tables** (the existing `Schema` carries the domains). The generator derives everything from table shapes and `params`.
- **A second dataset must be able to run unchanged**: package tests include a toy graph with different attribute names, counts and domain sizes (section 9).

## 3. Package A: eval metrics (must)

Where: new `src/graph_aim/eval/coupling.py` (library), called from `snap/eval/score.py`; plots in `snap/eval/notebooks/viz_comparison.ipynb`; rows added to `snap/eval/results/sweep*.csv`. Inside/eval zone only: it reads real and synthetic graphs, so nothing in `transfer` or `outside` may import it.

API sketch: `coupling_metrics(real, syn, *, degree_bin_edges, missing=None, ordered=None, exclude_hubs=0) -> dict`, where `missing` maps attribute to its missing label (default `none` if present in the domain, else no missing label) and `ordered` maps attribute to a bool. These come from `attribute_policy.json` (new optional keys `missing_label` and `ordered` per attribute, written by `prepare.py`). Defaults keep today's policies valid.

For every attribute `k` in the schema (not only the released ones):

**Edge x attribute (homophily)**, over edges whose two ends are both observed:

- `mix_tvd_k`: total variation distance between the real and synthetic normalised `V x V` mixing matrices (symmetric, diagonal counted once, as `evaluate.homophily_matrix`).
- `assort_k`: Newman's categorical assortativity `r = (sum_i e_ii - sum_i a_i^2) / (1 - sum_i a_i^2)`, reported for real, synthetic and absolute difference. `lift_k = sum_i e_ii / sum_i a_i^2` reported alongside because it is easy to read.
- `observed_pair_share_k`: share of edges with both ends observed (real vs synthetic). It catches missingness being wired at random.
- If `ordered_k`: `mean_abs_bin_distance_k` over edges, real vs synthetic and vs the random-pairing expectation. This is what answers "do similar ages connect"; it is skipped when the attribute is unordered.

**Degree x attribute**:

- `deg_by_value_tvd_k = sum_v w_v * TVD(P_real(bin | v), P_syn(bin | v))`, `w_v` = real share of value `v`, `bin` = the same public degree bins as the `attr_degree` tables.
- `eta2_k`: correlation ratio (share of the variance of `log(1 + degree)` explained by the attribute), real vs synthetic. One comparable scalar per attribute across datasets.
- Long-form table `mean_degree_by_value` (attribute, value, n real, mean real, mean synthetic) for the notebook.
- Hub robustness: every degree metric is also reported with the `exclude_hubs` highest-degree nodes removed (default 10), because a few ego hubs can drive a value's mean.

**Reference rows** so every number has a floor and a ceiling:

- floor: the real graph with its attributes randomly permuted (attribute-blind null: lift 1, `eta2` about 0);
- ceiling: the non-private run (`epsilon = inf`);
- the existing structure baselines (Erdos-Renyi, Chung-Lu, DC-SBM, BTER) carry no attributes: report `n/a`, not zero.

**Headline summary per metric**: gap closed `= (m_null - m_syn) / (m_null - m_real)`, so 0 means as bad as the null and 1 means as good as the real graph. It is comparable across attributes and datasets. The existing `homophily_error` stays for continuity.

Deliverables: `coupling.py`, tests with hand-computed answers, score.py wiring, notebook section, a **baseline table of today's numbers** recorded in `docs/snap/PROCESS.md` (the "before" column for B, C and D).

## 4. Package B: `attr_degree` measurement (must)

A new kind in the existing measure-then-reconstruct workload.

**What it is.** For attribute `a` with domain size `K_a` and public degree bins `b_0 = 0 < b_1 < ... < b_B` (last bin open-ended, degrees clamped to `Dmax` exactly as `degree_hist`): the `K_a x B` table counting nodes with attribute value `v` and degree in bin `j`. One `Measurement` per attribute, named `attr_degree:<attr>`, `params = {"attrs": [a], "cap": None, "bin_edges": [...]}`. The degree is the **uncapped** degree (same as `degree_hist`).

**Sensitivity (to be written as a derivation in the `measure.py` docstring, checked by brute force):**

- *Edge adjacency*: adding `{u, v}` changes only `d_u -> d_u + 1` and `d_v -> d_v + 1`; the attribute rows are fixed. If a node's bin changes, its part of the change is `-e_(a,j) + e_(a,j+1)` (norm sqrt(2)); otherwise zero. Worst case, both endpoints have the same attribute value and cross the same boundary: `-2e + 2e'`, norm **2 sqrt(2)**. If the values differ the cells are disjoint (norm 2); if one's destination cell is the other's source cell the middle terms cancel (norm sqrt(2)). No cap is needed and no factor 3.
- *Attribute-row adjacency*: node `u` changes value `a -> a'`; edges and degrees are unchanged, so one unit moves from `(a, j)` to `(a', j)` for `j = bin(d_u)`: **sqrt(2)**.
- So `sensitivity = (2 sqrt(2), sqrt(2))`, the same as `degree_hist` on the edge side and `attr_1way` on the attribute side. Same ledgers, same Gaussian mechanism, same zCDP accounting.

**Public parameters.** `attr_degree_bin_edges` in the config (explicit list), default `degree_bins(max_degree, n_bins=6, last=max_degree + 1)`. Keep `B <= 8`. Edges are a public setting chosen by looking at the data, so they join the existing caveat about bin edges. Optional `attr_degree_attrs` (list of attribute names, default all) to restrict the set for a dataset where some attributes are irrelevant or the budget is tight.

**Budget.** The config gets a new share, `"attr_degree"` under `shares` (start at 0.20; tuned by sweep). Shares are normalised per adjacency over the kinds that have positive sensitivity, and a kind's part is split equally among its measurements (`privacy.calibrate`), so adding the kind takes budget from the others. Rough estimate for the shipped config (eps=4, delta=1e-6, rho = 0.254 per adjacency), before water-filling returns slack: 8 tables at share 0.20 gives about 0.0043 each, sigma about 30 per cell, against about 84 nodes per cell on average at `B = 6`. That is enough for the broad pattern (mean degree 99 vs 36), poor for small cells. The sweep must confirm; fewer bins or `attr_degree_attrs` are the levers.

**Post-processing** (inside, free): clamp at zero and rescale to `n`, like `attr_2way`. Consistency with the other releases (row sums with `attr_1way`/`attr_2way`, column sums with the coarsened `degree_hist`) is done **in the generator against the sampled margins** (package C), where the exact margins exist, so the stored table stays a plain noisy release that is easy to inspect.

**Contract changes** (transfer zone, additive, so old artifacts still load and `FORMAT_VERSION` stays):

- `types.py`: document the kind in the `Measurement` docstring (shape `(K_a, B)`, bin meaning).
- `artifact.py`: add `attr_degree` to `_KINDS`; shape check `values.shape == (len(domains[a]), len(bin_edges) - 1)`; require `bin_edges` increasing starting at 0.
- `inspect.py`: show the kind (it enumerates measurements generically; verify).
- Provenance allowlist: unchanged (no new provenance key).

**Inside changes**: `measure.py` (`KINDS`, `workload()`, `exact_values()`, `sensitivity()`, `postprocess()`), `fit.py` (`plan()` config keys), `inside/config/graph_aim_v1.json` (share, bins), `docs` (README "Add features inside" already describes this checklist; follow it).

## 5. Package C: generator adaptation (must)

File: `src/graph_aim/outside/generate.py` only (plus its tests). Imports stay within numpy/pandas/scipy and `graph_aim.transfer`/`outside`.

Today: attributes (Chow-Liu tree) -> degrees drawn from `degree_hist`, shuffled over nodes -> structure. New order is the same, with **degree assignment conditional on attributes**:

1. **Attributes**: unchanged.
2. **Degree pool**: unchanged (systematic rounding of `degree_hist` to exactly `n` degrees, hub tail preserved). Let `c_j` = number of pool degrees in `attr_degree` bin `j`.
3. **Target tables**: for each `attr_degree:<k>` table `T_k`, clean (clamp at zero, uniform rows if a row is empty), then rake (IPF) so its row sums are the **sampled** value counts `r_k[v]` and its column sums are `c_j`. Result `R_k[v, j]`: the target joint count with exactly consistent margins.
4. **Node-level fit** (this is what combines several attributes without double counting): model `P[i, j] proportional to exp(phi_j + sum_k theta_k[a_k(i), j])`, the maximum-entropy distribution consistent with the tables. Fit `theta`, `phi` by damped iterative scaling until, for every `k`, `sum_i P[i, j] 1[a_k(i) = v]` matches `R_k[v, j]`, and `sum_i P[i, j] = c_j`. Tables are tiny (at most 8 x 8), so this costs nothing. Noisy tables may disagree with each other; the fit is then a compromise and the residual is reported (see tests).
5. **Assignment with exact capacity**: draw each node's bin from `P` subject to column totals `c_j` (Gumbel top-`c_j` per bin, bins in random order, last bin takes the remainder; the exact scheme is chosen by test on achieved vs target tables). Then hand out the pool's degrees of that bin to its nodes by random permutation. The synthetic degree histogram is therefore **exactly** the released one.
6. Make the degree sum even (as now), then structure generation is **unchanged**. Phase two's class-pair step uses classes of (attribute group, degree bin), so it now sees the true group degree volumes. This also removes the "IPF absorbs the resulting mismatch" caveat in the docstring.

Fallbacks and edge cases:

- No `attr_degree` measurements in the artifact (old artifacts): current independent shuffle, byte-identical behaviour for the same seed.
- Only a subset of attributes released: use those; the others are unconstrained.
- `B = 1`, empty value, all-zero table, or one value of an attribute absent from the sample: handled without error.
- `n` different from `schema.n`: margins come from the sampled data, so nothing special is needed.
- Determinism: all randomness from the seeded generator, the order of operations fixed.
- Update the module docstring: "What each measurement controls" gets `attr_degree`; "Known simplifications" loses the degree-independence item.

## 6. Package D: edge x attribute for a few attribute groups (optional)

Do this **after** A to C have been measured. The goal is not "all attributes" or "all pairs": a dataset chooses a handful of **groups** (a group is one attribute, or a small product of attributes, at most 64 cells, the existing `MAX_GROUPS`).

**Reuse the existing kind.** `attr_edge_pair` already takes `attrs` (a list whose product is the group) and has proved sensitivities (edge 3, attribute `sqrt(2) * Dp`). The change is to allow **several instances** instead of one, with no new sensitivity analysis.

- Config: `homophily_groups: [["school"], ["hometown"]]` (list of attribute lists). `homophily_attrs` stays as an alias for a single group. A hard limit `max_homophily_groups` (default 3) is enforced in `fit.plan()` so the budget cannot be spread over everything by accident.
- Artifact: measurements named `attr_edge_pair:<a>` or `attr_edge_pair:<a>+<b>`. The generator selects by **kind**, never by name. `Schema.homophily_attrs` keeps the primary group for compatibility.
- Selection is **per dataset and data-dependent**. Provide a non-private ranking of candidate groups (lift, excluding missing) in `snap/inside/reports/attributes.md` (the current "same-group edge share" table counts none-none matches and understates the signal), and fix the choice in the config. The choice is outside the DP guarantee and is added to `caveats`. A DP selection (exponential mechanism over a lift score) is a later option, not part of this plan.
- Budget: the `attr_edge_pair` share is split equally among its instances, so each extra group makes all of them noisier. The cost is on the **attribute-row** ledger: each instance has attribute sensitivity `sqrt(2) * Dp` (212 at `Dp = 150`). Levers: fewer groups, a smaller `Dp` (the cap of 100 dropped 21% of edges, so not far), or a larger `epsilon_attr` than `epsilon_edge` (the two adjacencies are separate, so this is a policy decision, not a technical one). Decide before enabling.

**Generator, multi-channel rewiring** (`outside/generate.py`):

- The initial sampling stays steered by one primary channel (class = attribute group x degree bin, as now). The primary group is the instance with the highest lift **computed from the released table itself**, so it needs no outside knowledge. A product seed over all groups would explode the class count and is rejected.
- `_rewire` currently takes one `(cls, target)`. Change it to take a list of channels `(cls_k, target_k)`. The swap cost becomes the sum over channels of the existing linearised chi-square change, each channel normalised by its total mass so none dominates. Per channel the state is a `G_k x G_k` count matrix (at most 64 x 64, usually 6 x 6): cheap.
- Channel targets: the ordered-pair mass of the released table, scaled so that its total equals the synthetic total degree volume (same convention as the current class-pair target; the table counts only the capped graph, the same assumption school already makes).
- Limits: swaps move only phase-two edges and preserve degrees, so channels that fight each other may not all be reached. The generator should stop early when no swap lowers the cost, and the eval reports the achieved lift per channel.

## 7. Package E: attribute-aware triangle blocks (optional experiment)

Phase-one blocks (BTER-style, clustering) are formed by sorting on (degree, then attribute group as tie-break). Try sorting on (primary group, then degree) so dense communities are mostly within a group. Zero privacy cost, generator-only. **Keep only if** degree KS, triangles and clustering do not get worse beyond the seed-to-seed spread and within-group triangle share improves. Needs a within-group triangle metric in package A (share of triangles whose three corners share the primary group), added only if E is attempted.

## 8. Privacy and caveats

- B adds one kind; sensitivities (2 sqrt(2), sqrt(2)) are proved above and added to the brute-force test. Ledger entries are generated by the existing `build_ledger`; nothing in `privacy.py` changes except the share key in config.
- New or extended caveats for `CAVEATS` in `privacy.py` (and therefore every artifact's `privacy.json`): the degree bins of `attr_degree`; the `attr_degree_attrs` restriction if used; in D, the choice of `homophily_groups` and its `max_homophily_groups`.
- The guarantee statement is unchanged: per adjacency, edge-DP and attribute-row-DP separately; no node-DP.
- Boundary: all generator work is in `outside/`; `coupling.py` is in `eval/`. `tests/graph_aim/test_boundary.py` must still pass, with no new imports in `transfer`/`outside` beyond numpy/pandas/scipy.
- Note for the reader of results: a degree-by-attribute table can reveal that a small group has unusual degrees. That is exactly what DP bounds, but it is also where noise will be largest, so small cells will be unreliable by design.

## 9. Tests

| Test | Where | What it proves |
|---|---|---|
| Coupling metrics on tiny graphs with known answers | `tests/graph_aim/eval/test_coupling.py` | perfect homophily gives r = 1, random pairing about 0, permuted-attribute null gives lift about 1; missing-label exclusion; ordered-attribute distance; hub exclusion; invariance to attribute names |
| `attr_degree` sensitivity, brute force | `tests/graph_aim/inside/test_measure.py` | every single-edge toggle and every single-attribute change on small random graphs stays within (2 sqrt(2), sqrt(2)); the edge bound is attained; includes edges that cross bin boundaries and equal-value endpoints; varied `K` and bin edges |
| Artifact validation | `tests/graph_aim/transfer/test_artifact.py` | round trip; rejects wrong shape or bad `bin_edges`; old artifacts without the kind still load |
| Fit and ledger | `tests/graph_aim/inside/test_fit.py`, `test_privacy.py` | the kind appears in the ledger; calibration with the new share; missing share is an error |
| Degree assignment | `tests/graph_aim/outside/test_generate.py` | on a constructed artifact where degree depends on an attribute: conditional tables recovered within tolerance; degree histogram **exactly** preserved; attribute counts untouched; no-table fallback identical to today; subset of attributes; `B = 1`; empty value; deterministic per seed |
| Multi-channel rewiring (D) | same | each channel's mixing matrix moves toward its target; degrees preserved; no loops or duplicates |
| **Different-dataset toy test** | `tests/graph_aim/test_generality.py` | fit and generate on a small graph with different attribute names, 3 attributes, domain sizes 2 to 5, one ordered attribute; no code path mentions SNAP names |
| Boundary | `tests/graph_aim/test_boundary.py` | unchanged; must pass |

Run: `PYTHONPATH=src:snap/inside uv run pytest tests/graph_aim -q`. All tests offline.

## 10. Acceptance criteria

Initial numbers, to be revisited once package A has recorded the baseline:

- B and C: for the attributes with a real `eta2` above the null, **gap closed >= 0.5 at eps = inf and >= 0.3 at eps = 4** on `deg_by_value_tvd` and `eta2` (3 seeds, mean).
- No regression: degree KS, triangles, clustering, transitivity, assortativity and attribute 2-way TVD stay within the seed-to-seed spread of today's `sweep_summary.csv`.
- D (if done): gap closed >= 0.5 on `lift`/`assort` for each configured group at eps = inf; attributes **not** configured stay near chance (this is expected, and the report must say so).
- Full test suite green, including the boundary test and the generality test.

## 11. Order of work

1. **A**: metrics, baseline table into `PROCESS.md` ("before"). This also fixes the misleading homophily report.
2. **B**: measurement, sensitivity derivation and brute-force test, artifact validation, config. Review the derivation before running anything.
3. **C**: generator. Check at eps = inf first (isolates the model from the noise), then the eps sweep.
4. Re-run the three stages and the sweep; update `snap/reports` (a v1.1 section), notebook, `snap/README.md`, `src/graph_aim/README.md`, `docs/snap/PROCESS.md` (decisions), `CHECKPOINTS.md`.
5. **Decision point**: with the "after" numbers, decide on D (and its budget policy: number of groups, `Dp`, `epsilon_attr`). Then D, then optionally E.

## 12. Risks and open questions

- Noise in small cells of `attr_degree` may make the conditional degree distributions unreliable for rare values; coarser bins or fewer attributes are the levers. The sweep decides.
- The maximum-entropy combination across attributes assumes the noisy tables are roughly consistent; strongly conflicting tables give a compromise. The fit residual is surfaced in tests.
- Ego-network confound: part of the real degree-attribute dependence may come from how the data was collected (10 ego networks). It is still the property of the data we are asked to reproduce, but results on a second graph would be more convincing.
- Ordered attributes are supported in the metrics and the table shapes but not in the generator's mixing (a K x K table already encodes "near" if the domain is ordered). A dataset with real ages needs `prepare.py` to bin them in order; SNAP cannot test this.
- Package D's budget policy (attribute-row epsilon) is a decision for the project owner.

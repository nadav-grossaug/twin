# aim

## 1. TL;DR

AIM (Adaptive and Iterative Mechanism, McKenna et al., VLDB 2022) is a workload-aware, DP synthetic-data mechanism for discrete tabular data. It iteratively selects the worst-approximated marginals from a workload with the exponential mechanism, measures them with Gaussian noise under zero-concentrated DP (zCDP), fits a graphical model (Private-PGM) to all noisy measurements, and samples a synthetic dataset from it. The privacy budget is tracked in zCDP and converted to (ε, δ)-DP. Synthcity ports the original `private-pgm` reference implementation. **Notable Synthcity behaviour:** (1) the whole private mechanism runs inside `generate()`, not `fit()`, ignores `count`, and **spends a fresh (ε, δ) on every `generate()` call (and on every retry of `_safe_generate`)**; (2) the mbi layer expects every column to be integer codes `0..k-1`, but the plugin only label-encodes categoricals (< 15 unique values), so **numeric columns with ≥ 15 distinct values are passed as raw values and partly fall outside the histogram bins**, and come back as codes, not original values; (3) pairs with more than `max_cells=1000` cells are not in the workload, so high-cardinality columns are never paired. Pre-discretize numeric columns before using this plugin.

| | |
|---|---|
| Family | Marginal-based DP synthesis (select–measure–generate) with Private-PGM inference |
| DP | (ε, δ)-DP via zCDP, `epsilon=1.0`, `delta=1e-9` by default |
| Papers | R. McKenna, B. Mullins, D. Sheldon, G. Miklau. "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data." PVLDB 15(11), 2022 (arXiv:2201.12677). Related: McKenna, Miklau, Sheldon. "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data." J. Privacy & Confidentiality 11(3), 2021 (arXiv:2108.04978) (MST); McKenna, Sheldon, Miklau. "Graphical-model based estimation and inference for differential privacy." ICML 2019 (arXiv:1901.09136) (Private-PGM); Bun, Steinke. "Concentrated Differential Privacy." TCC 2016 (arXiv:1605.02065) |
| Library | Vendored copy of `private-pgm` ("mbi") in `synthcity/plugins/core/models/mbi/` (Python, NumPy, SciPy, NetworkX; CPU only) |
| Synthcity plugin | `aim` → `synthcity/plugins/privacy/plugin_aim.py` (`AIMPlugin`) → `core/models/tabular_aim.py` (`TabularAIM`) → `core/models/aim.py` (`AIM`) |

## 2. How it works (paper)

**Inputs**: a dataset $D$ over discrete attributes (public domain), a workload $W$ of marginal queries (sets of attributes, here all attribute pairs), and a budget $(\varepsilon,\delta)$ converted to a zCDP parameter $\rho$.

**Measure.** For a clique $r$ of attributes, the marginal $M_r(D)$ is a vector of cell counts (L2 sensitivity 1 under add/remove of a record). The Gaussian mechanism with scale $\sigma$ costs $\rho = 1/(2\sigma^2)$ (zCDP). The exponential mechanism with parameter $\epsilon$ costs $\epsilon^2/8$.

**Algorithm 1 (AIM)**:
1. Set rounds $T = 16d$; $\sigma_0 = \sqrt{T/(2\cdot 0.9\rho)}$, $\epsilon_0 = \sqrt{8\cdot 0.1\rho/T}$ (10% of the budget for selection, 90% for measurement in the nominal schedule).
2. **Initialize**: measure all one-way marginals of the workload attributes with $\sigma_0$ and estimate a graphical model $\hat p$ with Private-PGM (a mirror-descent fit of a junction-tree-factored distribution to the noisy marginals).
3. **Iterate**:
   - candidates = downward closure of $W$, filtered by the model-size constraint (the junction tree of the cliques chosen so far plus the candidate must stay below `max_model_size` MB, scaled by the fraction of budget used);
   - **select** a clique with the exponential mechanism using the *worst-approximated* quality score $q_r = w_r(\|M_r(D) - M_r(\hat p)\|_1 - \sqrt{2/\pi}\,\sigma\, n_r)$, where the subtracted term is the expected L1 error of the Gaussian noise (so already-well-fit marginals are not over-selected); sensitivity $w_r$;
   - **measure** it with Gaussian noise $\sigma_t$; add to the measurement set; refit $\hat p$ (warm start);
   - **adaptive step**: if the model barely changed on the measured clique (L1 change below the noise level), halve $\sigma$ (double the per-round measurement budget); if the remaining budget cannot cover two more rounds, spend all of it in a final round.
4. Final fit with more iterations and sample from $\hat p$.

The key idea is that rounds, noise scale and selected marginals are adapted to the data and the budget, with a guarantee of $\rho$-zCDP overall (→ (ε, δ)-DP, with $\delta>0$ required).

## 3. Synthcity implementation

Sources: `plugins/privacy/plugin_aim.py` (the plugin), `plugins/core/models/tabular_aim.py` (workload construction), `plugins/core/models/aim.py` (the AIM mechanism; `cdp_rho`, `compile_workload`, `filter_candidates`) and `plugins/core/models/mbi/*` (vendored Private-PGM). Line numbers refer to those files.

### 3.1 Budget accounting
- `Mechanism.__init__`: `rho = cdp_rho(epsilon, delta)` found by 1000 bisection steps on `cdp_delta` (the Bun–Steinke conversion with optimal order α) in `aim.py` 70–120. `delta` must be > 0 (default `1e-9`). Hence the privacy parameters are (ε = 1, δ = 1e-9) by default, and `delta` does matter.
- `AIM.run` (276–345) follows the reference schedule: `rounds = 16·d`, `sigma = sqrt(rounds/(2·0.9·rho))`, `epsilon = sqrt(8·0.1·rho/rounds)`, all one-way marginals first (`rho_used = n_oneway·0.5/σ²`), the final-round clause and the σ-halving rule (`‖w−z‖₁ ≤ σ√(2/π)·n` ⇒ `σ /= 2; epsilon *= 2`). `worst_approximated` and the model-size filter (`size_limit = max_model_size·rho_used/rho`) are unchanged from the reference. The exponential-mechanism helper uses `softmax(0.5·ε/Δ·q)`. Structural zeros are not used. The total number of records is not supplied to the inference (estimated from the noisy measurements), so the output size is approximately the training size, not `count` (below).

### 3.2 Where the mechanism runs: `generate`, not `fit` (tabular_aim.py 74–133)
- `TabularAIM.fit` only computes the **workload** and instantiates `AIM(epsilon, delta, max_model_size)`; it does **not** look at the data privately (it stores the raw training dataframe `self.dataset` in the model object, so the pickled plugin contains the private data).
- `TabularAIM.generate(count, ...)` calls `self.model.run(self.dataset, self.workload)` **on every call** and returns `synth_dataset.df`. The `count` argument is ignored: the synthetic dataset has approximately the training size (the noisy total estimated by Private-PGM).
- `AIMPlugin._generate` calls `_safe_generate`, which calls `generate` in a loop until at least `count` rows have passed the schema filter (up to `sampling_patience` = 500 times). Each call re-runs the full private mechanism with a fresh noise draw. **Consequence**: the end-to-end privacy cost is (number of `run` calls) × (ε, δ) under basic composition (or √k·ε under advanced composition). A single `plugin.generate(n)` that filters enough rows costs 1×; multiple `generate()` calls, evaluation code that samples twice, or filtering-induced retries multiply the spend. Cache one synthetic frame if the budget must hold.

### 3.3 Workload and domain (tabular_aim.py 87–111)
- Domain: `Domain(columns, X.nunique())`: the domain size of each column is the number of distinct values **in the data** (a non-private choice; the paper assumes a public domain).
- Workload = all `degree`-way attribute combinations (`degree=2`), **dropped if the product of domain sizes exceeds `max_cells`** (default 1000; the hyper-parameter space samples 5000–25 000). If `num_marginals` is given, a random subset of that many cliques is kept (`np.random.choice`, independent of the data). All weights are 1.
- Candidates (`compile_workload`) are the downward closure of the workload (pairs and their one-way marginals). Columns that appear in no retained pair are never candidates, so **they are neither measured nor selected**: a column with more than `max_cells` distinct values, or whose products with all other columns exceed the cap, is not modelled. (We could not observe the sampled values for such a column in our environment; expect an uninformative marginal.)
- The model-size filter uses `max_model_size` (default 80 MB), which bounds the junction-tree size and effectively the number of cliques that can be added.

### 3.4 Preprocessing mismatch: the mbi layer needs integer codes `0..k-1`
- `mbi.Dataset.datavector` computes `np.histogramdd(df.values, bins=[range(n+1) ...])`, i.e. it assumes each column holds integers in `[0, domain_size)`.
- The plugin gives it `X.dataframe()` as produced by the base `Plugin.fit`, in which only object/bool columns and columns with `< 15` unique values are label-encoded to `0..k-1`; other numeric columns remain raw values. The domain size is still `nunique`.
- Verified on a 2 000-row toy table: for an integer column `age` with 74 distinct values in 17..90, the domain size is 74, the minimum value in the AIM dataframe is 17 and the maximum 90, and `Dataset.project(("age",)).datavector().sum()` is **1 567 of 2 000 records**: the 433 rows with `age ≥ 74` fall outside the bins and are silently dropped from the measurements, and bins 0–16 are empty. Continuous (float) columns are worse: values are binned by `floor` into 0..nunique or discarded.
- On the output side, `synth.df` contains codes `0..k-1` and the plugin applies no inverse transform for non-encoded numerics, so **synthetic numeric values are codes, not values**, and the data-cleaning schema filter in `_safe_generate` (min/max of the training column) may reject many rows and trigger retries, each of which spends privacy budget (3.2).
- Practical remedy: discretize every numeric column yourself into at most ~15 to 30 equal-width or quantile bins *before* calling the plugin, encode bins as `0..k-1`, and map back (bin centre or uniform within the bin) afterwards. Public bin edges (or edges not derived from the data) keep the end-to-end guarantee.

### 3.5 Runtime and robustness
- `FactoredInference(iters=1000)` is run for every round (`rounds = 16·d`: 112 rounds for 7 columns) plus 2 500 iterations at the end, all on CPU with NumPy; runtime grows quickly with the number of columns and the junction-tree size. In our environment (2 CPU cores, 8 GB RAM in WSL) AIM runs on 4-column tables completed in seconds, whereas runs with 7 columns (including a 74-value and a 20-value column) terminated the WSL session during `generate`; we could not diagnose whether this was memory or CPU exhaustion. Plan for much higher resource use than the other generators, and test on your hardware.
- `mbi/mechanism.py` contains leftovers of the reference repo (a `run` helper with `bounded`), unused by the plugin.
- No `cond` support (raises `NotImplementedError`).
- The `device` and `random_state` arguments are accepted but not used: sampling and noise use NumPy's global RNG (not seeded by the plugin).

### 3.6 Hyperparameters

| param | default | meaning |
|---|---|---|
| `epsilon` | 1.0 | target ε |
| `delta` | 1e-9 | target δ (> 0 required) |
| `max_model_size` | 80 | MB bound on the junction-tree parameter vector |
| `degree` | 2 | clique size of the workload (marginal order) |
| `num_marginals` | None | optional random subset size of the workload |
| `max_cells` | 1000 | drop workload cliques with more cells than this |
| `random_state`, `device`, `sampling_patience`, `workspace`, `compress_dataset` | 0, auto, 500, `workspace`, False | base-class plumbing (`device`/`random_state` effectively unused) |

`hyperparameter_space()`: `epsilon ∈ [0.5, 3]`, `delta ∈ [1e-10, 1e-5]` (log), `max_model_size ∈ {50..200}`, `degree ∈ {2..5}`, `num_marginals ∈ {0..5}` (0 is an invalid choice because `np.random.choice(..., 0)` returns an empty workload), `max_cells ∈ {5000..25 000}`.

## 4. Privacy properties

- **Nominal and (modulo the points below) actual**: the AIM mechanism as coded is (ε, δ)-DP for **one run** with respect to the discrete input, given a public domain and fixed attribute set. Gaussian measurement and exponential-mechanism selection are correctly calibrated in the zCDP accounting.
- **Per-run, not per-model**: because the mechanism runs inside `generate`, privacy is spent at sampling time and **accumulates over calls and retries** (3.2). Report the number of calls.
- **Not covered**: (a) the domain (number of distinct values, which values occur) is read from the data; (b) any data-dependent preprocessing before the plugin (encoding choices, binning); (c) the raw training dataframe is retained inside the plugin object; (d) number of rows is ≈ n (noisy total), which is standard.
- **Memorization**: marginal-based synthesis from noisy 2-way marginals is structurally unable to reproduce individual records, so record-level privacy metrics (distance to closest record, membership inference) should look strong. The cost is utility: only pairwise structure is preserved, joint structure of order ≥ 3 is lost.
- If numerics are not pre-binned, the measurements are partially wrong (3.4), which is a utility, not a privacy, defect: dropped records only reduce counts.

## 5. Strengths / weaknesses / when to use

- **Strengths**: state-of-the-art accuracy among marginal-based DP synthesizers on discrete benchmarks; principled budget allocation; no gradient training; low memorization; deterministic accounting in zCDP with `delta` selectable.
- **Weaknesses**:
  - Needs discretized data; the plugin does not discretize and mis-handles raw numerics (3.4).
  - Privacy cost per `generate()` call; `count` ignored (3.2).
  - Heavy CPU/RAM; slow inference; resource spikes with larger domains.
  - Only pairwise (`degree=2`) marginals by default; a `max_cells` filter excludes pairs of larger columns.
  - Domain read from data; no support for conditional generation.
- **Mixed-type expectations (n ≈ 10 k, ε = 1)**: if numeric columns are first binned to ≤ ~20 bins and categoricals encoded, one-way and two-way marginals are reproduced closely; without pre-binning, columns with ≥ 15 distinct values are mis-measured (partly dropped or returned as codes) and high-cardinality columns are excluded.
- **When to use**: as the DP reference method for discrete tabular data when you control discretization and can afford the compute; pair it with an explicit single-generation protocol (generate once, reuse).

## 6. References

- R. McKenna, B. Mullins, D. Sheldon, G. Miklau. "AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data." PVLDB 15(11):2599–2612, 2022. https://www.vldb.org/pvldb/vol15/p2599-mckenna.pdf; arXiv:2201.12677. Reference code: https://github.com/ryan112358/private-pgm
- R. McKenna, D. Sheldon, G. Miklau. "Graphical-model based estimation and inference for differential privacy." ICML 2019. https://arxiv.org/abs/1901.09136
- R. McKenna, G. Miklau, D. Sheldon. "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data." J. Privacy and Confidentiality 11(3), 2021. https://arxiv.org/abs/2108.04978
- M. Bun, T. Steinke. "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds." TCC 2016. https://arxiv.org/abs/1605.02065
- F. McSherry, K. Talwar. "Mechanism Design via Differential Privacy." FOCS 2007.
- Synthcity: Qian, Cebere, van der Schaar. arXiv:2301.07573.

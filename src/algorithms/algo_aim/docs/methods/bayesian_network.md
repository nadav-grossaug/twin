# bayesian_network

## 1. TL;DR

The plugin learns a DAG over the columns with pgmpy (default: a Chow–Liu tree), fits a maximum-likelihood conditional probability table (CPT) per node, and forward-samples ancestrally. It is not DP. **Synthcity 0.2.12 quirk:** the network is fitted on the raw (label-encoded but not discretized) data. Every distinct numeric value becomes its own state, so a near-unique numeric parent makes the sampler **replay training rows**. The only obfuscation is a small post-hoc noise step through the mode-specific GMM encoder, and that step can also corrupt high-cardinality categorical codes.

| | |
|---|---|
| Family | Probabilistic graphical model (discrete Bayesian network) |
| DP | No (see `privbayes` for the DP variant) |
| Papers | No dedicated Synthcity paper; the plugin cites pgmpy. Building blocks: Chow & Liu, *IEEE Trans. Inf. Theory* 1968 (tree search); Cooper & Herskovits, *Machine Learning* 1992 (K2 score); Heckerman, Geiger & Chickering, *Machine Learning* 1995 (BDeu); Schwarz, *Ann. Stat.* 1978 (BIC); Scutari, PGM 2016 (BDs); Spirtes, Glymour & Scheines, *Causation, Prediction, and Search* 1993/2000 (PC); Tsamardinos, Brown & Aliferis, *Machine Learning* 2006 (MMHC); Henrion, UAI 1986/1988 (probabilistic logic / forward sampling) |
| Library | pgmpy 0.1.26: Ankan & Panda, "pgmpy: Probabilistic graphical models using Python", SciPy 2015. https://pgmpy.org |
| Synthcity plugin | `bayesian_network` → `synthcity/plugins/generic/plugin_bayesian_network.py` (`BayesianNetworkPlugin`) |

## 2. How it works

A Bayesian network factorizes the joint over a DAG $G$:

$$p(x_1,\dots,x_d) = \prod_{j=1}^d p\big(x_j \mid x_{\mathrm{pa}_G(j)}\big).$$

For discrete variables each factor is a CPT.

**Structure learning** (pick $G$):
- *Score-based* (hill climbing): greedily add, remove, or reverse edges to maximize a decomposable score.
  - **K2** and **BDeu** are Bayesian-Dirichlet marginal likelihoods. K2 uses a uniform Dirichlet(1) prior. BDeu uses an equivalent sample size spread uniformly over each CPT.
  - **BDs** is a BD variant that only spreads the prior over parent configurations actually observed.
  - **BIC** is $\log \hat L - \tfrac{\log N}{2}\,\#\text{params}$.
  - Search is limited by `max_indegree` and `max_iter`; pgmpy uses tabu search.
- *Constraint-based* (**PC**): start from the complete graph, delete edges using conditional-independence tests (χ² by default) with conditioning sets of growing size, then orient v-structures and propagate orientations.
- *Hybrid* (**MMHC**): use Max-Min Parents & Children to restrict candidate edges, then hill-climb inside that skeleton.
- **Tree search (Chow–Liu)**: the maximum-weight spanning tree under pairwise mutual information $I(X_i;X_j)$. This is the KL-optimal tree-structured approximation (Chow & Liu 1968). The tree is oriented away from a root; pgmpy picks as root the node with the largest total MI. Each node has at most one parent.
- **Exhaustive**: enumerates all DAGs. Only feasible for a handful of variables.

**Parameter learning**: maximum likelihood, so $\hat\theta_{j,k\mid u} = N(x_j{=}k, \mathrm{pa}{=}u)/N(\mathrm{pa}{=}u)$. There is no smoothing. Parent configurations never seen in training get no probability mass.

**Sampling**: forward (ancestral) sampling in topological order, with $x_j \sim \hat p(x_j\mid x_{\mathrm{pa}(j)})$.

## 3. Synthcity implementation

Source: `.venv/lib/python3.11/site-packages/synthcity/plugins/generic/plugin_bayesian_network.py` (line numbers below refer to that file unless stated otherwise).

### 3.1 Preprocessing
1. Base `Plugin.fit` label-encodes every object/bool column and every column with `< 15` unique values (`plugins/core/plugin.py:235`, `dataloader.py:212–244`). Continuous columns stay as raw floats/ints.
2. `_fit` (166–176) fits a `TabularEncoder(max_clusters=encoder_max_clusters)` (95, 168). This encoder **is not used for learning**. It is only used later to add noise.
3. The DAG and CPTs are learned directly on that label-encoded dataframe (170–173). There is **no discretization**: pgmpy treats every distinct value of a numeric column as a separate state.

### 3.2 Structure + parameters
- `_get_dag` (142–164) maps the search method:
  - `hillclimb` → `HillClimbSearch.estimate(scoring_method, max_indegree=struct_max_indegree, max_iter=struct_learning_n_iter)`
  - `pc` → `PC.estimate(...)` with pgmpy defaults: `variant="stable"`, `ci_test="chi_square"`, `significance_level=0.01`, `max_cond_vars=5`. The `scoring_method` argument is passed but **ignored** (swallowed by `**kwargs`).
  - `tree_search` → `TreeSearch.estimate()`: Chow–Liu, MI edge weights, root = node with the largest MI sum (`pgmpy/estimators/TreeSearch.py:64–140`).
  - `mmhc` → `MmhcEstimator.estimate(scoring_method)`
  - `exhaustive` → `ExhaustiveSearch.estimate()`
- `_get_structure_scorer` (134–140): `k2`, `bdeu`, `bic`, `bds` → pgmpy `K2Score`, `BDeuScore`, `BicScore`, `BDsScore`. This scorer **only matters for `hillclimb` and `mmhc`**. With the default `tree_search`, the score, `struct_max_indegree`, and `struct_learning_n_iter` are all unused.
- `BayesianNetwork(dag).fit(df)` uses pgmpy's default `MaximumLikelihoodEstimator` (`pgmpy/models/BayesianNetwork.py:566–567`).

### 3.3 Sampling (178–184, 118–132)
1. `BayesianModelSampling.forward_sample(size=count)` draws ancestral samples. These are label-encoded rows with values taken from the training support.
2. `_encode_decode`: transform with the fitted `TabularEncoder`, add $\epsilon\sim\mathcal N(0,\texttt{encoder\_noise\_scale}^2)$ to every `*.value` column, then inverse-transform.
   - For a column modelled by the BayesianGMM encoder (`feature_encoder.py:181–257`), `.value` is $(x-\mu_k)/(4\sigma_k)$ for the assigned mode $k$, clipped to $\pm0.99$. Noise with `scale=0.1` is therefore ≈ $\mathcal N(0,(0.4\,\sigma_k)^2)$ in data units, and the result is clipped to the training `[min, max]`.
   - **A single noise value is drawn per row** (`size=len(encoded)`, line 122–124) and added to *all* `.value` columns of that row. The noise is shared across columns. In my test the per-column perturbations had correlation 0.79.
   - The `TabularEncoder` treats a column as discrete only if it has ≤ `categorical_limit=10` unique values (`tabular_encoder.py:147–148`, `utils/dataframe.py:12–23`). **Label-encoded categoricals with > 10 levels are modelled as continuous**, so they receive noise and are rounded back to a *different, arbitrary* level code. On a toy 20-level categorical, 58% of sampled codes were changed by this step.
3. `_safe_generate` (`plugin.py:395–428`) filters rows to the schema bounds and drops duplicate rows within each batch. `DataLoader.decode` maps label codes back to strings.

### 3.4 Quirk: replay of training rows
Because numeric columns are not binned, a near-unique numeric column has $\approx N$ states, and its CPT conditioned on it is a point mass on the training row's value of the child. MI with a near-unique column is maximal, so Chow–Liu tends to hang other columns off such columns. Ancestral sampling then reconstructs whole training rows.

Toy check (3000 rows; columns `c20`, a copy `c20b`, `x = N(0,1)+c20`, `y ~ Exp(1)`). The learned tree was `y → x → {c20, c20b}`, and **100% of `forward_sample` rows were exact copies of training rows**. The noise in step 2 is the only thing that separates the output from the training data. The real-data size of this effect belongs in the dataset report.

Also: CPT size is (#child states) × (#parent states). With continuous parents, fitting is memory- and time-heavy. `struct_max_indegree` caps this only for `hillclimb`.

### 3.5 Hyperparameters

| param | default | meaning |
|---|---|---|
| `struct_learning_search_method` | `"tree_search"` | `hillclimb`, `pc`, `tree_search`, `mmhc`, `exhaustive` |
| `struct_learning_score` | `"k2"` | `k2`, `bdeu`, `bic`, `bds`; used only by `hillclimb`/`mmhc` |
| `struct_learning_n_iter` | 1000 | `max_iter` for hill climbing only |
| `struct_max_indegree` | 4 | max parents, hill climbing only |
| `encoder_max_clusters` | 10 | BayesianGMM components for the noise encoder |
| `encoder_noise_scale` | 0.1 | std of noise on the normalized `.value` (≈ 0.4 σ of the mode) |
| `sampling_patience` | 500 | max `_safe_generate` rounds |
| `random_state` | 0 | global seed (pgmpy sampling uses the global numpy RNG) |
| `workspace`, `compress_dataset` | `Path("workspace")`, False | base-class plumbing |

`hyperparameter_space` (107–116): `struct_learning_search_method ∈ {hillclimb, pc, tree_search}` × `struct_learning_score ∈ {k2, bdeu, bic, bds}`. `mmhc`/`exhaustive` are excluded.

Conditioning: no `cond` support. `constraints=` is applied by rejection.

## 4. Privacy properties
- Not DP. The MLE CPTs are exact counts. `privbayes` is the DP counterpart: noisy, low-degree CPTs on discretized data.
- **High memorization risk in this implementation** (Section 3.4). With near-unique numeric columns the model is close to a lookup table of training rows. Expect high `identifiability_score`, low authenticity, and elevated DomiasMIA AUC. The privacy-looking effect of `encoder_noise_scale` is a heuristic perturbation with no formal guarantee. Increasing it trades fidelity for distance and corrupts > 10-level categoricals.
- Even with discrete-only data, unsmoothed MLE CPTs reproduce rare parent/child combinations exactly.

## 5. Strengths / weaknesses / when to use
- **Strengths**: fast to fit for low-cardinality discrete data. The structure is interpretable. Logical dependencies between discrete columns are captured exactly when they are parent–child pairs (for example a deterministic mapping such as education ↔ education-num), because a CPT can be deterministic. It never invents parent/child combinations that were not observed.
- **Weaknesses**:
  - There is no proper model of numeric columns. They are either replayed (no binning) or, in a re-implementation with binning, quantized.
  - Chow–Liu gives each node a single parent, so higher-order interactions are lost.
  - Hill climbing with `max_indegree=4` on high-cardinality columns gives huge, sparse CPTs.
  - The noise step can scramble > 10-level categoricals.
  - The PC/hill-climb runtime grows quickly with the number of states.
- **Mixed-type expectations**: multimodal numerics look perfect (they are resampled real values plus noise). High-cardinality categoricals can be corrupted by the encoder noise. Logical constraints hold for parent–child pairs. Fidelity metrics may look deceptively good while privacy metrics look bad. Always read the two together.
- **When to use**: as a classical, fast, interpretable non-neural baseline, preferably on pre-discretized data.

## 6. References
- C. Chow, C. Liu. "Approximating discrete probability distributions with dependence trees." IEEE Trans. Information Theory 14(3), 1968.
- G. F. Cooper, E. Herskovits. "A Bayesian method for the induction of probabilistic networks from data." Machine Learning 9, 1992.
- D. Heckerman, D. Geiger, D. M. Chickering. "Learning Bayesian networks: the combination of knowledge and statistical data." Machine Learning 20, 1995.
- M. Scutari. "An empirical-Bayes score for discrete Bayesian networks." PGM 2016 (PMLR 52).
- P. Spirtes, C. Glymour, R. Scheines. *Causation, Prediction, and Search*. 2nd ed., MIT Press, 2000.
- I. Tsamardinos, L. E. Brown, C. F. Aliferis. "The max-min hill-climbing Bayesian network structure learning algorithm." Machine Learning 65, 2006.
- M. Henrion. "Propagating uncertainty in Bayesian networks by probabilistic logic sampling." UAI 1986 (proc. 1988).
- A. Ankan, A. Panda. "pgmpy: Probabilistic graphical models using Python." SciPy 2015. (The plugin docstring misspells the first author as "Jankan".)

# privbayes

## 1. TL;DR

PrivBayes learns a low-degree Bayesian network over the columns with a differentially private greedy structure search (exponential mechanism), then releases Laplace-noised low-order marginals as the conditional probability tables and ancestrally samples from them. The paper's guarantee is pure ε-DP. **The Synthcity 0.2.12 implementation does not deliver that guarantee.** Four independent deviations break it: (1) the Laplace noise is calibrated for probabilities but added to raw counts, so it is smaller than it should be by a factor of n; (2) the structure score is a non-private k-means + normalized-MI heuristic whose sensitivity does not match the formula used; (3) numeric columns are never binned, so every distinct value is its own state; (4) noise is added only to non-empty cells. Treat the plugin as "a Bayesian network with lightly smoothed MLE tables on label-encoded data", with an ε argument that has almost no effect.

| | |
|---|---|
| Family | Probabilistic graphical model (discrete Bayesian network), DP variant |
| DP | Claimed: ε-DP (pure), `epsilon` split 50/50 between structure and tables. **In practice: no meaningful guarantee** (Section 3.3 to 3.5) |
| Paper | J. Zhang, G. Cormode, C. M. Procopiuc, D. Srivastava, X. Xiao. "PrivBayes: Private Data Release via Bayesian Networks." ACM TODS 42(4), 2017 (SIGMOD 2014 version). https://dl.acm.org/doi/10.1145/3134428 |
| Library | pgmpy 0.1.26 (`TabularCPD`, `BayesianNetwork`, `BayesianModelSampling`), scikit-learn (`KMeans`, `normalized_mutual_info_score`) |
| Synthcity plugin | `privbayes` → `synthcity/plugins/privacy/plugin_privbayes.py` (`PrivBayesPlugin`, wrapping `PrivBayes`) |

## 2. How it works (paper)

**Goal**: release a synthetic copy of $D$ ($n$ rows, $d$ discrete attributes) under ε-DP, without noising the full $\prod_j |\mathrm{dom}(X_j)|$-cell joint. Idea: approximate the joint by a degree-$k$ Bayesian network $\mathcal N$ ($k$ = max number of parents),

$$\Pr_{\mathcal N}[X_1,\dots,X_d] = \prod_{i=1}^d \Pr[X_i \mid \Pi_i],\qquad |\Pi_i|\le k,$$

and release noisy low-dimensional marginals only.

**Step 1 – private network learning (budget ε₁).** Greedy construction (a private analogue of Chow–Liu): start with a random root; at each of the remaining $d-1$ steps, enumerate all candidate pairs $(X, \Pi)$ with $X$ not yet placed and $\Pi$ a size-$\min(k,\#\text{placed})$ subset of already placed attributes; score each by the mutual information $I(X;\Pi)$ on the data; sample one pair with the **exponential mechanism**

$$\Pr[(X,\Pi)] \propto \exp\!\Big(\frac{\varepsilon_1\, I(X;\Pi)}{2(d-1)\,\Delta}\Big).$$

$\Delta$ is the sensitivity of $I$ (Lemma 4.1: $\tfrac{1}{n}\log n + \tfrac{n-1}{n}\log\tfrac{n}{n-1}$ if $X$ or $\Pi$ is binary; $\tfrac{2}{n}\log\tfrac{n+1}{2}+\tfrac{n-1}{n}\log(1+\tfrac{2}{n-1})$ otherwise). The paper also proposes the $F$ function (a low-sensitivity surrogate for binary attributes) and the $R$ function.

**Step 2 – private conditional distributions (budget ε₂).** For each $X_i$ release the noisy joint marginal $\Pr[X_i,\Pi_i]$ with Laplace noise of scale $2(d-k)/(n\varepsilon_2)$ on the *probabilities* (sensitivity $2/n$, composed over $d-k$ released marginals). The first $k$ attributes need only one marginal: the joint of $X_1..X_k$. Negative cells are clamped to zero, then each marginal is normalized into a conditional.

**Step 3 – release.** Sample $n'$ tuples ancestrally from the noisy network. By sequential composition the whole mechanism is $(\varepsilon_1+\varepsilon_2)$-DP.

**Choosing $k$ (Section 4.5).** A noisy marginal is *θ-useful* if its average information-to-noise ratio is at least θ (θ = 5 recommended). Lemma 3 gives usefulness $= \frac{n\varepsilon_2}{(d-k)\,2^{k+3}}$ for binary data; $k$ is the largest value with usefulness ≥ θ.

**Non-binary attributes.** The paper discretizes continuous attributes to a fixed number of bins up front (and recommends a taxonomy/hierarchical encoding for categoricals) so that the domain is a public, data-independent quantity.

## 3. Synthcity implementation

Source: `.venv/lib/python3.11/site-packages/synthcity/plugins/privacy/plugin_privbayes.py` (line numbers refer to that file). `PrivBayesPlugin` (539–638) builds `PrivBayes` (66–536) on `X.dataframe()`. Training uses only NumPy global RNG and pgmpy; no GPU, no torch.

### 3.1 Preprocessing and "discretization"
1. The base `Plugin.fit` has already label-encoded every object/bool column and every column with `< 15` unique values; other numeric columns reach the model as raw floats/ints.
2. `_encode` (146–168): a column is treated as categorical (plain `LabelEncoder`, one state per distinct value) **if `nunique < n_bins` OR the dtype is not `object`/`category`**. Because of the `or`, every numeric column takes the categorical branch whatever its cardinality. The `pd.cut` branch is only reachable for object columns with ≥ `n_bins` (default 100) distinct strings, and since the base plugin label-encodes object columns first, it is effectively dead code. Even if reached, `pd.cut` on strings would raise.
   - **Consequence**: `n_bins` does nothing for numeric data. A continuous column with 10 000 distinct values in 10 000 rows becomes a 10 000-state variable.
3. `_decode` (170–186): categorical codes are mapped straight back with `LabelEncoder.inverse_transform`, so **synthetic numeric values are always values that exist in the training data** (no interpolation, no new values). In a test (10 000 rows, one near-unique float column) 100% of synthetic values in that column occurred in the real column.
4. The domain (set of states of each column, hence the CPT shapes) is read from the data (`LabelEncoder.fit`, `data.nunique()`), so which values exist is leaked even in the idealized setting; the paper treats the (binned) domain as public.

### 3.2 Budget split and K (90–95, 499–524)
- `self.epsilon = epsilon / 2`. The same halved value is used for **both** the structure phase (ε₁) and the noise phase (ε₂), which is consistent with total budget `epsilon` under sequential composition. (The comment in the source about "2eps" is misleading; the total is `epsilon`, not 2·`epsilon`.)
- `K=0` (default) → `_compute_K`: if the usefulness formula for `default_k=3` exceeds `target_usefulness` (default 5), K=3; otherwise solve for K numerically with `fsolve`; fall back to 3 if the solution is outside `[1, d]`. The formula $n\varepsilon_2/((d-k)2^{k+3})$ is the binary-attribute version and ignores domain sizes, so K is much too optimistic for high-cardinality columns. In our synthetic test (n = 10 000, d = 7, ε = 1) K = 3.
- `self.K` is set once on the first `fit` and then reused (the object is mutated).

### 3.3 Structure learning (189–258, 426–497)
- Candidates: for each remaining attribute and each `split` in `range(#placed - K + 1)`, all `K`-subsets of `placed[split:]`. `placed` is a Python `set` coerced to a list by pydantic, so its order depends on string hashing (`PYTHONHASHSEED`) and **the same parent set appears several times** (once per `split` that contains it), skewing the sampling distribution toward subsets with low-order members. Number of candidates grows as $\binom{|\text{placed}|}{K}$ per remaining attribute.
- **Score is not mutual information.** `mutual_info_score` (462–478) runs `KMeans(n_clusters=10)` on the label-encoded parent columns (unseeded, non-private, uses raw label codes as coordinates so the geometry is arbitrary), bins the target with `pd.cut(bins=n_bins)`, and returns `normalized_mutual_info_score(src_bins, target_bins)` ∈ [0, 1]. The sensitivity used for the exponential mechanism is still the Lemma 4.1 formula for plain MI (`_calculate_sensitivity`, 399–416). Neither the k-means partition (which depends on the full data) nor the NMI normalization is covered by that sensitivity bound, so the exponential mechanism's ε₁ guarantee does not hold.
- Because $\Delta \approx \log n / n \approx 10^{-3}$ for n = 10 000, the sampling weights are $\exp(\varepsilon_1 \cdot \mathrm{NMI}/(2(d-1)\Delta))$ with a large multiplier (≈ 20 to 40 on the score for d = 7, ε₁ = 0.5): selection is close to argmax of the heuristic NMI.
- `mi_thresh` (default 0.01): if the sampled candidate's score is ≤ threshold, its parents are replaced by `[]`. The comparison uses the exact, un-noised score of the sampled candidate, so the decision leaks information that is not accounted for.
- `_normalize_given_distribution` has a bug on the overflow path (`np.isinf(distribution)` is passed as frequencies, giving a boolean array); it only matters if `exp` overflows.
- The binary-attribute shortcut in the sensitivity (`nunique ≤ 2`) is used, but the F function from the paper is not.
- Cost: the number of k-means fits is the number of distinct (candidate, parent-set) pairs; for d ≈ 15, K = 3 it is thousands. In our synthetic test (n = 10 000, d = 7) fit took ≈ 1 minute on 2 CPU cores, dominated by pgmpy's CPT assembly and sampling.

### 3.4 Noisy conditionals (260–385)
- `_laplace_noise_parameter` (260–265) returns `2 * (n_features - K) / (n_items * epsilon)`. The docstring says the noise "are over counts, instead of the probability distributions in Algorithm 1", but the formula is the paper's **probability-scale** noise ($2(d-k)/(n\varepsilon_2)$). Applied to counts it should be $2(d-k)/\varepsilon_2$. The noise is therefore **n times too small**: for n = 10 000, d = 7, K = 3, ε₂ = 0.5 the Laplace scale is 0.0016 counts (measured), versus the correct 16 counts. This is the dominant defect.
- `_get_noisy_counts_for_attributes` (267–284): counts come from `groupby(attributes).size()`, which only yields **non-empty** cells; Laplace noise is added to those cells only, clamped at 0. Empty cells receive no noise. Even with a correct scale, this would reveal the exact support of every marginal (not DP).
- `_get_noisy_distribution_from_counts` (286–308): builds a cross-tab over the full product domain (`dropna=False`), fills missing cells with 0, **adds a pseudo-count of 1 to every cell**, and normalizes each parent column. With noise ≈ 0.002 counts, the table is essentially "MLE + Laplace(1) smoothing". The pseudo-count is data-independent and harmless, but it is also the only thing preventing zero probabilities.
- For the first K nodes in the order, the CPTs are derived from one noisy joint over `ordered_nodes[:K]` (as in the paper); the rest each use a fresh noisy marginal `P(X_i, Π_i)` (as in the paper). The same noise scale is used for both.
- Memory: the cross-tab has shape (#states of child) × $\prod_{p\in\Pi}$(#states of p). With raw numerics as parents this is huge (in the test: 10 000 × 2·20·72 ≈ 29 M cells for one node). pgmpy `TabularCPD` and `forward_sample` are then slow and memory-heavy.

### 3.5 Sampling (138–144)
`BayesianModelSampling.forward_sample(size=count)` (ancestral), then `_decode`. `_safe_generate` retries until `count` rows pass the schema filter, de-duplicating rows within each batch. pgmpy logs one WARNING per CPT column about probabilities not summing to exactly 1 (hundreds of thousands of lines for big tables; the log is the only visible symptom).

### 3.6 Hyperparameters

| param | default | meaning |
|---|---|---|
| `epsilon` | 1.0 | total budget; halved internally for structure and tables |
| `K` | 0 | max parents; 0 = auto from usefulness formula (fallback 3) |
| `n_bins` | 100 | bins for "continuous" columns; effectively **unused** for numeric data (Section 3.1) |
| `mi_thresh` | 0.01 | NMI below which a node gets no parents |
| `target_usefulness` | 5 | θ in the K computation |
| `random_state` | 0 | calls `enable_reproducible_results`; k-means, `np.random` and set ordering are not all controlled |
| `workspace`, `compress_dataset`, `sampling_patience` | `Path("workspace")`, False, 500 | base-class plumbing |

`hyperparameter_space()` returns `[]` (nothing to tune). No `delta` (pure ε-DP claimed). No `cond` support.

## 4. Privacy properties

- **Nominal**: ε-DP with ε = `epsilon`, provided the domain is public.
- **Actual** (this code): not ε-DP for any ε, because of (a) the n-fold under-scaled Laplace noise, (b) noise only on non-empty cells, (c) a data-dependent, non-private structure score with mismatched sensitivity, (d) `mi_thresh` post-selection on the exact score, and (e) the data-derived domain. With the measured noise scale of 0.002 counts, the released CPTs are exact empirical conditionals up to smoothing, so **memorization is governed by the model capacity rather than by ε**. In the replay-prone regime (near-unique numeric parents, CPT cells with 1–3 rows) synthetic rows reproduce real values of those columns exactly. Changing `epsilon` from 0.1 to 10 changes the output only through K and the structure sampling.
- **Consequence for benchmarking**: privacy metrics will look like a weakly regularized graphical model rather than a DP method. Do not report this plugin as "DP at ε=1". If a faithful PrivBayes is needed, pre-bin all numeric columns to a small number of bins before calling the plugin (so the CPTs are small), and fix the noise scale in a wrapper or patch (`n_items * epsilon` → `epsilon`, noise on all cells of the cross-tab).
- Preprocessing outside the plugin (any binning edges derived from the data) is also not accounted for.

## 5. Strengths / weaknesses / when to use

- **Strengths**: no neural training and no GPU; deterministic structure; sampling is cheap; interpretable DAG; handles categoricals natively and preserves exact logical dependencies between parent–child pairs; synthetic values are always valid training-set values.
- **Weaknesses**:
  - Broken privacy accounting (Section 4).
  - Raw numerics are never binned; high-cardinality numerics blow up CPT size and are replayed.
  - Structure scoring is a heuristic (k-means on label codes), not MI; the candidate list contains duplicates and depends on hash order.
  - Only low-order dependencies (K parents) survive; the K formula ignores domain sizes.
  - Mixed data: a 100-state numeric column and a binary column are treated the same by the sensitivity and noise calibrations.
  - No new numeric values are generated (support = training values).
- **Mixed-type expectations (n ≈ 10 k, ε = 1)**: the run completes in about a minute for d ≈ 7; low-cardinality categoricals and a few integer columns with ≤ ~100 values behave well; any near-unique float column makes the output a recombination of real values. Pre-bin first if that is not wanted.
- **When to use**: as a classical DP-labelled baseline **only after** discretizing numerics yourself and, ideally, patching the noise scale. Otherwise label it "non-DP tabular BN baseline".

## 6. References

- J. Zhang, G. Cormode, C. M. Procopiuc, D. Srivastava, X. Xiao. "PrivBayes: Private Data Release via Bayesian Networks." ACM Trans. Database Systems 42(4), Article 25, 2017. https://dl.acm.org/doi/10.1145/3134428 (conference version: SIGMOD 2014, https://doi.org/10.1145/2588555.2588573)
- C. Chow, C. Liu. "Approximating discrete probability distributions with dependence trees." IEEE Trans. Information Theory 14(3), 1968.
- F. McSherry, K. Talwar. "Mechanism Design via Differential Privacy." FOCS 2007 (exponential mechanism).
- C. Dwork, F. McSherry, K. Nissim, A. Smith. "Calibrating noise to sensitivity in private data analysis." TCC 2006 (Laplace mechanism).
- A. Ankan, A. Panda. "pgmpy: Probabilistic graphical models using Python." SciPy 2015. https://pgmpy.org
- Synthcity: Qian, Cebere, van der Schaar. "Synthcity: facilitating innovative use cases of synthetic data in different data modalities." arXiv:2301.07573.

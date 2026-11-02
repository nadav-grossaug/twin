# arf (Adversarial Random Forests)

## 1. TL;DR

A random forest is trained iteratively to tell real rows from synthetic ones. Each round, new synthetic rows are drawn by sampling each column independently *within the forest's leaves*. The loop stops when the forest can no longer separate real from synthetic (OOB accuracy ≤ 0.5 + δ). At that point features are approximately independent within each leaf, so the forest becomes a mixture of leaf-wise product distributions: truncated normals for continuous columns and multinomials for categorical ones. Training takes minutes on CPU, and there is no neural network.

| | |
|---|---|
| Family | Tree-based density estimator / generator (mixture of leaf-wise product distributions) |
| DP | No |
| Paper | D. S. Watson, K. Blesch, J. Kapar, M. N. Wright. "Adversarial random forests for density estimation and generative modeling." AISTATS 2023 (PMLR 206). https://arxiv.org/abs/2205.09435 |
| Implementation | `arfpy` 0.1.1 (Python port of the R package `arf`), `arfpy/arf.py` |
| Synthcity plugin | `arf` → `synthcity/plugins/generic/plugin_arf.py` (`ARFPlugin`), wrapper `synthcity/plugins/core/models/tabular_arf.py` (`TabularARF`) |

## 2. How it works (paper)

**Adversarial training loop (ARF).**
1. Build an initial synthetic dataset $\tilde X^{(0)}$ from the product of marginals by permuting each column independently.
2. Fit a random forest $f^{(0)}$ to classify real ($y{=}0$) vs synthetic ($y{=}1$), and record its out-of-bag accuracy.
3. While OOB accuracy $> 0.5+\delta$ and fewer than `max_iters` rounds have run: generate $\tilde X^{(t)}$ from the current forest and refit the classifier.
   - To generate: pick a tree and a leaf $\ell$ with probability proportional to the leaf's real-data coverage. Then draw **each feature independently** from the real data in that leaf.

The forest gradually partitions the space so that, within each leaf, features are (approximately) mutually independent. The paper calls this *local independence*. When the discriminator is at chance, independent sampling inside leaves is indistinguishable from real data.

**Density estimation (FORDE).** For each tree $b$ and leaf $\ell$, the paper fits:
- a coverage weight $\theta_\ell^b$ (fraction of real samples in the leaf),
- a univariate density $\hat p^b_{\ell,j}$ per feature: a truncated normal on the leaf's bounding box for continuous features, and a categorical distribution for discrete ones.

The density estimate is

$$\hat p(x) = \frac1B \sum_{b=1}^B \sum_{\ell \in L_b} \theta^b_\ell \,\mathbb 1[x\in \ell]\, \prod_{j=1}^d \hat p^b_{\ell,j}(x_j).$$

The paper proves consistency of this estimator under regularity assumptions. I did not re-derive the exact conditions; see the paper's theorem. It reports competitive likelihood and synthesis quality against probabilistic circuits and deep generative models at roughly two orders of magnitude lower runtime.

**Generation (FORGE).**
1. Draw a tree uniformly and a leaf with probability $\theta^b_\ell$, which is equivalent to drawing $(b,\ell)$ with probability $\theta^b_\ell / B$.
2. Sample each feature independently from $\hat p^b_{\ell,j}$.

## 3. Synthcity implementation

### 3.1 Plugin → wrapper → arfpy
- `ARFPlugin._fit` (`plugin_arf.py:136–164`) builds `TabularARF(X.dataframe(), num_trees, delta, max_iters, early_stop, verbose, min_node_size, **kwargs)` and calls `.fit`.
- `cond` raises `NotImplementedError` (159–162, 167–170), so there is **no conditional generation**.

### 3.2 Preprocessing
- Base `Plugin.fit` label-encodes object/bool columns and any column with `< 15` unique values (`plugin.py:235`, `dataloader.py:212–244`).
- `TabularARF.fit` (`tabular_arf.py:115–134`) casts every column with `nunique() ≤ var_threshold = 10` to `object`. arfpy turns those into `category` and therefore treats them as **categorical**. All other columns are treated as **continuous**. This includes label-encoded categoricals with **> 10 levels**, which then get a truncated normal over their integer codes.
- There is no TabularEncoder, no one-hot encoding, and no scaling. Trees do not need any of them. The `encoder_max_clusters`, `encoder_whitelist`, `learning_rate`, `weight_decay`, `batch_size`, `logging_epoch`, and `device` arguments of `TabularARF.__init__` (`tabular_arf.py:41–48`) are **unused**. So is the plugin docstring's `n_iter` / `learning_rate` text (`plugin_arf.py:75–99`), which is copy-paste from a neural plugin.

### 3.3 Training (arfpy `arf.__init__`, `arfpy/arf.py:31–177`)
- Initial synthetic: independent column permutation (line 69).
- The discriminator is `sklearn.ensemble.RandomForestClassifier(oob_score=True, n_estimators=num_trees, min_samples_leaf=min_node_size)` with sklearn defaults otherwise, including `max_features="sqrt"` (80, 131).
- Each round, leaves are sampled from the real-data leaf assignments and each column is resampled with replacement inside the leaf (95–121). In other words, the generator used *during* training uses the leaf's empirical marginals, not truncated normals.
- Stopping (140–146): stop if `acc ≤ 0.5 + delta`, or `iters ≥ max_iters`, or (`early_stop` and accuracy *increased* versus the previous round). The forest kept is `clf_0`, the one whose samples fooled the last discriminator. This matches the R package.
- Pruning (151–177): leaves with fewer than `min_node_size` real samples are merged into their sibling.

### 3.4 Density + sampling (`forde`, `forge`, `arfpy/arf.py:179–365`)
- `TabularARF.generate` calls `forde(dist="truncnorm", oob=False, alpha=0)` **on every call** (`tabular_arf.py:141`). That means on every `_safe_generate` round as well, which adds runtime but does not change the result.
- `forde`:
  - Coverage `cvg` is the fraction of real rows in each leaf. If any continuous column exists, leaves holding a single real row get coverage 0 (line 224).
  - Continuous columns: leaf mean, leaf sd, and the leaf bounds from the tree splits (`utils.bnd_fun`). Infinite bounds where no split occurred.
  - Categorical columns: within-leaf frequencies, with Laplace smoothing only if `alpha > 0` (default 0).
- `forge(n)`:
  - Draw $(b,\ell)$ with $p = \mathrm{cvg}/B$ (313).
  - Draw continuous features from `scipy.stats.truncnorm(a, b, loc=mean, scale=sd)` within the leaf bounds (342–346). If `sd = 0` (all leaf values equal), the code comments say this returns the location, i.e. an exact copy of that value.
  - Draw categorical features from the within-leaf multinomial (331).
- Afterwards Synthcity's `_safe_generate` casts to the training dtypes, filters to the schema, and drops duplicates within each batch. `DataLoader.decode` then maps label codes back. For > 10-level categoricals sampled as continuous, the float code is cast to `int` (truncation) before the inverse label mapping (`dataloader.py:252–258`).

### 3.5 Quirk: high-cardinality categoricals
Code order is alphabetical, i.e. arbitrary, so sampling a truncated normal over codes can produce neighbouring but semantically unrelated levels. I tested a 20-level categorical with an exact copy column. The copy relation held in only **61%** of ARF samples, against 100% in the real data. Typical errors were off-by-one codes such as `k17`/`m18`.

Other notes:
- `verbose=True` by default prints the OOB accuracy per round.
- `random_state` is not passed to arfpy or sklearn. Reproducibility relies on the global numpy seed that the base `Plugin` sets.

### 3.6 Hyperparameters

| param | default | meaning |
|---|---|---|
| `num_trees` | 30 | trees per forest |
| `delta` | 0 | convergence tolerance: stop when OOB acc ≤ 0.5 + δ. arfpy asserts `0 ≤ δ ≤ 0.5` |
| `max_iters` | 10 | max adversarial rounds |
| `early_stop` | True | stop when OOB accuracy rises versus the previous round |
| `verbose` | True | print accuracy per round |
| `min_node_size` | 5 | `min_samples_leaf` of the RF, and the pruning threshold. Increase it if `forge` hits domain errors (`tabular_arf.py:146–148`) |
| (wrapper) `dist` | `"truncnorm"` | only option |
| (wrapper) `oob` | False | use OOB samples for FORDE parameters |
| (wrapper) `alpha` | 0 | Laplace pseudocount for categoricals |
| (wrapper) `var_threshold` | 10 | nunique ≤ this → treated as categorical |
| `sampling_patience` | 500 | `_safe_generate` rounds |
| `random_state`, `device`, `workspace`, `compress_dataset` | 0, auto, `workspace`, False | base-class plumbing; `device` is unused (CPU sklearn) |

`hyperparameter_space` (`plugin_arf.py:125–134`):
- `num_trees` ∈ {10, 20, …, 100}
- `delta` ∈ {0, 2, …, 50}
- `max_iters` ∈ {1, …, 5}
- `early_stop` ∈ {True, False}
- `min_node_size` ∈ {2, 4, …, 20}

**Bug:** `delta` values ≥ 2 violate arfpy's `assert 0 <= delta <= 0.5` (`arfpy/arf.py:39`), so random HPO draws other than 0 crash. The source carries a `# TODO: check if this is the right range` comment.

## 4. Privacy properties
- Not DP.
- Leaves hold ≥ `min_node_size` real rows, and continuous values are drawn from fitted truncated normals rather than copied. Singleton leaves get zero coverage. This gives some protection against verbatim copying.
- Risks remain:
  - Categorical and low-cardinality columns are drawn from within-leaf frequencies of small groups (≥ 5 rows by default), so rare combinations can be reproduced.
  - Leaves with `sd = 0` re-emit the exact value.
  - Small `min_node_size` means more local, more memorizing leaves.
- My expectation (not verified here) is that ARF memorizes less than the replay-style `bayesian_network` plugin and more than smooth neural models. Measure it with authenticity, identifiability, and DomiasMIA.

## 5. Strengths / weaknesses / when to use
- **Strengths**:
  - Very fast on CPU: few forests, no gradient descent.
  - Few hyperparameters.
  - Handles mixed types natively without one-hot or GMM encoding.
  - Multimodal numerics are captured well because leaves partition the modes.
  - Bounded support: values stay within leaf bounds and therefore within the training range.
  - Strong results reported against deep models on tabular benchmarks.
- **Weaknesses**:
  - Within-leaf independence: dependence finer than the leaf partition is lost, and it is noticeable with few trees or large leaves.
  - Truncated normals per leaf are a coarse fit for skewed or heavy-tailed or point-mass-heavy columns (for example zero-inflated values).
  - The Synthcity wrapper treats > 10-level categoricals as continuous codes (Section 3.5).
  - No conditional generation.
  - The HPO space is broken for `delta`.
- **Mixed-type expectations**:
  - Good marginals and pairwise structure for low-cardinality categoricals and multimodal numerics.
  - Degraded fidelity on high-cardinality categoricals.
  - Logical constraints, both deterministic and functional dependencies, hold only approximately: they hold within a leaf only if the leaf separates them.
  - Zero-inflated numerics get smeared mass near zero unless a leaf isolates the zeros, in which case `sd=0` reproduces 0 exactly.
- **When to use**: as a strong, fast, non-neural default for tabular data. It is a good first model before trying GANs or diffusion.

## 6. References
- D. S. Watson, K. Blesch, J. Kapar, M. N. Wright. "Adversarial random forests for density estimation and generative modeling." AISTATS 2023, PMLR 206. https://arxiv.org/abs/2205.09435
- arfpy (Python port by K. Blesch et al.): https://github.com/bips-hb/arfpy ; R package `arf` on CRAN.
- L. Breiman. "Random forests." Machine Learning 45, 2001.

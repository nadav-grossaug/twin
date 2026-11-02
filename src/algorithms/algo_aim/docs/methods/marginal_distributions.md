# marginal_distributions

## 1. TL;DR

A baseline that samples every column independently and keeps no dependence between columns. It is the floor that any real generator has to beat on joint-distribution metrics. **Synthcity 0.2.12 quirk:** the plugin does not sample the empirical marginals. It samples **uniformly** within each column's training support: `U[min, max]` for numeric columns and uniform over the levels of categorical columns. See [Section 3.3](#33-quirk-it-samples-uniform-not-empirical-marginals).

| | |
|---|---|
| Family | Independent-marginals baseline (Synthcity type `debug`) |
| DP | No |
| Paper | None. This is a standard sanity baseline, for example the "independent marginals" reference in the synthetic-data literature. |
| Synthcity plugin | `marginal_distributions` → `synthcity/plugins/generic/plugin_marginal_distributions.py` (`MarginalDistributionPlugin`) |

## 2. How it works (intended method)

The intended model is the product of the empirical marginals:

$$\hat p(x_1,\dots,x_d) = \prod_{j=1}^d \hat p_j(x_j),$$

where $\hat p_j$ is the empirical distribution of column $j$. Sampling draws each column independently from its empirical distribution, which is a bootstrap of each column separately. By construction this:
- preserves each univariate distribution exactly (in expectation),
- destroys all dependence. Every correlation, mutual information, and conditional distribution is that of independent variables.

On fidelity metrics it is the reference point for "zero dependence modelled". Univariate metrics (`ks_test`, `jensenshannon_dist`) should look good. Joint metrics (`detection_xgb`, `wasserstein_dist`, `prdc`, TSTR) should look bad.

## 3. Synthcity implementation

Source: `.venv/lib/python3.11/site-packages/synthcity/plugins/generic/plugin_marginal_distributions.py`.

### 3.1 Training
- `__init__` passes `sampling_strategy="marginal"` to the base `Plugin` (line 37).
- `_fit` is a no-op that returns `self` (lines 51–54). All the "learning" happens in the base class `Plugin.fit` (`plugins/core/plugin.py:172–257`):
  1. `X.encode()` (`plugin.py:235` → `dataloader.py:212–244`) label-encodes every column that is object/bool **or** has `< 15` unique values. Integer columns that are already `0..K-1` are kept as they are.
  2. `self._training_schema = Schema(data=X_encoded, sampling_strategy="marginal")` (`plugin.py:248`). `Schema._infer_domain` (`schema.py:252–359`) creates one `Distribution` per column with `data=` attached, so the empirical `value_counts` are stored (`distribution.py:179–182, 317–325`).

### 3.2 Sampling
- `_generate` (lines 56–68) iterates over `syn_schema` and calls `syn_schema[feature].sample(count)` for each column. It then goes through `_safe_generate` (`plugin.py:395–428`), which filters rows against the schema constraints, **drops duplicate rows within each batch** (`plugin.py:419`), and repeats up to `sampling_patience` times.
- The output is decoded back to the original categories by `DataLoader.decode` with the stored `LabelEncoder`s (`plugin.py:354–355`).

### 3.3 Quirk: it samples uniform, not empirical, marginals
`syn_schema` is **not** the training schema. In `Plugin.generate` it is rebuilt from constraints:

```python
gen_constraints = self.training_schema().as_constraints()   # plugin.py:343
syn_schema = Schema.from_constraints(gen_constraints)        # plugin.py:347
```

`as_constraints()` keeps only the support: `ge`/`le`/`dtype` rules for numeric columns and `in` rules for categorical ones (`distribution.py:256–260, 406–416`). `Schema.from_constraints` (`schema.py:168–250`) then builds `FloatDistribution(low, high)` / `IntegerDistribution(low, high)` **without data**, so `marginal_distribution is None`. `FloatDistribution.sample` then falls back to `rng.uniform(low, high)` (`distribution.py:369–381`), and integers fall back to uniform integers in the same way. Because categoricals were label-encoded to integers before the schema was built, they become `IntegerDistribution(0, K-1)`, which is **uniform over levels**.

I checked this empirically (5k rows, synthcity 0.2.12):

| column | real | synthetic |
|---|---|---|
| `cat` with p = (0.80, 0.15, 0.05) | 0.80 / 0.15 / 0.05 | 0.34 / 0.35 / 0.32 |
| `skew` ~ Exp(1): mean / std | 1.01 / 1.03 | 4.37 / 2.48 (≈ U[0, 8.7]) |
| `age` ~ N(35, 5) clipped: std | 5.0 | 10.9 (≈ U[17, 54]) |

So in 0.2.12 this plugin is a **uniform-in-bounding-box** baseline. It is weaker than a true product-of-marginals: univariate metrics will also be poor, especially for skewed numerics and imbalanced categoricals. If you need a true marginal baseline, sample each column with `df[col].sample(n, replace=True)` outside Synthcity.

### 3.4 Hyperparameters

| param | default | meaning |
|---|---|---|
| (none specific) | | `hyperparameter_space()` returns `[]` (line 48–49) |
| `sampling_patience` | 500 | max `_safe_generate` rounds (base `Plugin`) |
| `strict` | True | filter rows that violate schema constraints |
| `random_state` | 0 | seeds the per-column RNGs (`schema.py:261`). Note that `from_constraints` sets `random_state=None` (`schema.py:178`), so sampling is seeded only by the global numpy seed. |
| `workspace`, `compress_dataset` | `Path("workspace")`, False | base-class plumbing |

Conditioning: none. `cond` is not supported. `constraints=` filters rows by rejection.

## 4. Privacy properties
- No DP guarantee.
- Under the 0.2.12 behaviour it releases only per-column `min`/`max` (numerics) and the set of observed levels (categoricals). Those can themselves be sensitive: the extreme values are real records' values, and rare levels reveal that some record had them.
- A true product-of-marginals would release the exact marginal histograms. That leaks no joint information, but it reproduces rare values verbatim.
- Row-level memorization risk is negligible. A synthetic row matches a real one only by chance.

## 5. Strengths / weaknesses / when to use
- **Use as**: a floor / sanity check. A model that does not beat it on `detection_xgb` or TSTR has not learned any dependence.
- **Strengths**: instant, no failure modes, well-defined privacy "budget" (support only).
- **Weaknesses**: no dependence at all. In 0.2.12 it does not even keep the marginals. Multimodal or skewed numerics become flat. High-cardinality categoricals become uniform, so rare categories are heavily over-represented. Logical constraints across columns (for example education ↔ education-num, relationship ↔ marital-status/sex) are violated at the rate implied by independence. TSTR AUC should be ≈ 0.5.

## 6. References
- Synthcity: Qian, Cebere, van der Schaar, "Synthcity: facilitating innovative use cases of synthetic data in different data modalities", arXiv:2301.07573, 2023 (reportedly NeurIPS 2023 Datasets & Benchmarks; venue not verified). https://arxiv.org/abs/2301.07573

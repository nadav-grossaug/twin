# nflow (Normalizing Flows)

## 1. TL;DR

The plugin fits an invertible neural map from a standard Gaussian to the data by exact maximum likelihood. Synthcity first encodes rows with the CTGAN-style tabular encoder (mode-specific BayesianGMM normalization for numerics, one-hot for discrete columns). By default the flow is a single masked autoregressive **rational-quadratic spline** layer (Neural Spline Flow) built on the `nflows` library. Sampling pushes Gaussian noise through the inverse and decodes with argmax and inverse-GMM.

| | |
|---|---|
| Family | Normalizing flow (explicit-likelihood deep generative model) |
| DP | No |
| Papers | C. Durkan, A. Bekasov, I. Murray, G. Papamakarios. "Neural Spline Flows." NeurIPS 2019. https://arxiv.org/abs/1906.04032. Also: Papamakarios, Pavlakou, Murray, "Masked Autoregressive Flow for Density Estimation", NeurIPS 2017 (https://arxiv.org/abs/1705.07057); Dinh, Sohl-Dickstein, Bengio, "Density estimation using Real NVP", ICLR 2017 (https://arxiv.org/abs/1605.08803); Müller et al., "Neural Importance Sampling", ACM TOG 2019 (quadratic splines); Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and Inference", JMLR 2021 (review) |
| Library | `nflows` 0.14 (Durkan, Bekasov, Murray, Papamakarios). https://github.com/bayesiains/nflows |
| Synthcity plugin | `nflow` → `synthcity/plugins/generic/plugin_nflow.py` (`NormalizingFlowsPlugin`); models `plugins/core/models/tabular_flows.py` (`TabularFlows`), `plugins/core/models/flows.py` (`NormalizingFlows`); encoder `plugins/core/models/tabular_encoder.py` |

## 2. How it works (papers)

A flow defines $x = T(z)$ with $z\sim p_Z = \mathcal N(0,I)$ and $T$ a composition of invertible, differentiable maps $T = T_K\circ\cdots\circ T_1$. By change of variables,

$$\log p_X(x) = \log p_Z\big(T^{-1}(x)\big) + \sum_{k=1}^K \log\left|\det \frac{\partial T_k^{-1}}{\partial u_{k}}\right|.$$

Training maximizes the exact log-likelihood $\frac1N\sum_i \log p_X(x_i)$. Sampling computes $T(z)$. Expressiveness depends on the elementwise bijection and on how its parameters are conditioned:

- **Coupling** (Real NVP): split $u=(u_A,u_B)$. Then $u'_B = \tau(u_B;\,\mathrm{NN}(u_A))$ and $u'_A = u_A$. Both directions take a single pass, and the Jacobian is triangular.
- **Autoregressive** (MAF): $u'_i = \tau(u_i;\,\mathrm{NN}(u_{<i}))$, parameterized by a MADE network with masked weights. Density evaluation takes one pass. **Sampling is sequential in the dimension** (D passes).
- **Elementwise transforms $\tau$**:
  - *affine*: $\tau(u)=a u + b$, $a>0$.
  - *piecewise quadratic* (Müller et al.).
  - *rational-quadratic spline* (NSF): a monotone spline with $K$ bins on $[-B,B]$. Bin widths, heights, and knot derivatives are produced by the network, and the inverse is analytic (it solves a quadratic). Outside $[-B,B]$ the map is the identity ("linear tails").
- **Linear layers** between transforms mix dimensions: a fixed random permutation, an LU-parameterized invertible matrix, or an SVD (Householder) parameterization.

NSF showed that RQ-spline coupling and autoregressive layers are markedly more flexible than affine ones at similar cost.

## 3. Synthcity implementation

### 3.1 Preprocessing (`TabularFlows`, `tabular_flows.py:81–160`)
- Base `Plugin.fit` label-encodes object/bool columns and any column with `< 15` uniques (`plugin.py:235`).
- `TabularEncoder(max_clusters=encoder_max_clusters)` (`tabular_flows.py:109–111`, `tabular_encoder.py:57–245`) then encodes each column:
  - **discrete** if `nunique ≤ 10`: sklearn one-hot with `handle_unknown="ignore"`.
  - **continuous** otherwise: `BayesianGMMEncoder` (`feature_encoder.py:181–257`). A `BayesianGaussianMixture(n_components=max_clusters, weight_concentration_prior=1e-3)` is fitted, and each value is encoded as `[ (x-μ_k)/(4σ_k) clipped to ±0.99 , one_hot(k) ]` with $k=\arg\max_k p(k\mid x)$. This is the mode-specific normalization of CTGAN (Xu et al. 2019), except that the component is chosen by argmax rather than sampled.
  - This means **label-encoded categoricals with > 10 levels are modelled as continuous** codes via the GMM.
- Encoded width: $\sum_{\text{cont}} (1+\texttt{max\_clusters}) + \sum_{\text{disc}} K_j$.
- The flow models this whole vector as a **continuous density**. One-hot and component-indicator dims are 0/1 values with **no dequantization**, so the likelihood on these degenerate dims can grow without bound. In practice this is controlled only by the small capacity and early stopping. `fit` raises `RuntimeError("The loss contains NaNs")` if training diverges (`flows.py:244–245`).

### 3.2 Model (`NormalizingFlows._create_transform`, `flows.py:398–411`)
- The transform is `[Linear, Base] × n_layers_hidden` plus a final `Linear`.
  - `Linear` is set by `linear_transform_type`: `permutation` → `RandomPermutation`; `lu` → permutation + `LULinear(identity_init)`; `svd` → permutation + `SVDLinear(num_householder=10)`.
  - `Base` is set by `base_transform_type` (`flows.py:302–396`). Coupling variants use an alternating binary mask and a `ResidualNet(hidden_features=n_units_hidden, num_blocks=num_transform_blocks, dropout, batch_norm)`. Autoregressive variants use MADE with residual blocks (`random_mask=False`). Spline variants use `num_bins`, `tails="linear"`, `tail_bound`.
- Base distribution: `StandardNormal`. It is the only option.
- **Plugin defaults**: 1 flow layer (`n_layers_hidden=1`) of `MaskedPiecewiseRationalQuadraticAutoregressiveTransform`, 100 hidden units, 1 residual block, 8 bins, tail bound 3, random permutations. This is a shallow flow.

### 3.3 Training (`NormalizingFlows.fit`, `flows.py:216–266`)
- A random **80/20 split** is made (`flows.py:164–172`). Only 80% of the rows are used for gradient steps. The other 20% is used only by the early-stopping metric.
- The objective is the mean negative log-likelihood, optimized with Adam(`lr`). The `DataLoader` has **no shuffling** (`flows.py:152–154`), so batch order is fixed after the initial split shuffle.
- `n_iter` counts **epochs**, not flow steps as the docstring says.
- Early stopping: every `n_iter_print` epochs, generate `len(X_val)` samples and compute `patience_metric`. The plugin default is `WeightedMetrics([("detection","detection_mlp")])` (`plugin_nflow.py:151–156`): an MLP real-vs-synthetic AUC in *encoded* space, against the validation split. Training stops when `patience` consecutive checks fail to improve and `it ≥ n_iter_min`.
- **Bug: best-checkpoint restore is a no-op.** `best_state_dict = self.state_dict()` (`flows.py:258`) stores references that share storage with the live parameters, and Adam updates them in place. `load_state_dict(best_state_dict)` therefore reloads the *last* weights. I verified this: a state_dict taken before an Adam step equals the parameters after it. The model returned is the final epoch's, not the best-scoring one.
- The detection-metric check every 50 epochs is the main runtime cost besides the flow itself.

### 3.4 Sampling
- `flow.sample(count)` draws $z\sim\mathcal N(0,I)$ and applies $T$. For autoregressive bases this is **D sequential MADE passes**, which is slow for wide encodings.
- `TabularEncoder.inverse_transform` then decodes:
  - one-hot blocks via sklearn `OneHotEncoder.inverse_transform` (argmax),
  - GMM columns via argmax over the component dims, then $x = v\cdot 4\sigma_k + \mu_k$ with $v$ clipped to $[-1,1]$, then a hard clip to the training `[min, max]`.
  - Clipping puts point masses at the exact training min/max. That explains a small fraction of "exact" values.
- The plugin's `_internal_generate` (`plugin_nflow.py:270–293`) generates 1 row and then batches of ≤ 5000. **It swallows any exception** (`except BaseException: pass`), so a failing batch silently returns fewer rows. `_safe_generate` then refills, up to `sampling_patience` rounds.
- `cond` is not supported. `_fit`/`_generate` silently ignore it: the flow is unconditional. `constraints=` is applied by rejection.

### 3.5 Hyperparameters

| param | default (plugin) | meaning |
|---|---|---|
| `n_iter` | 1000 | max training **epochs** |
| `n_layers_hidden` | 1 | number of [linear + base transform] flow layers |
| `n_units_hidden` | 100 | hidden width of the conditioner nets |
| `num_transform_blocks` | 1 | residual blocks per conditioner net |
| `batch_size` | 200 | minibatch size |
| `dropout` | 0.1 | conditioner dropout |
| `batch_norm` | False | conditioner batch norm |
| `num_bins` | 8 | spline bins (quadratic/RQ) |
| `tail_bound` | 3 | spline interval $[-B,B]$; identity outside |
| `lr` | 1e-3 | Adam learning rate |
| `apply_unconditional_transform` | True | coupling only: also spline-transform the identity half, unconditionally |
| `base_distribution` | `"standard_normal"` | only option |
| `linear_transform_type` | `"permutation"` | `permutation` / `lu` / `svd` |
| `base_transform_type` | `"rq-autoregressive"` | `affine-`/`quadratic-`/`rq-` × `coupling`/`autoregressive` |
| `encoder_max_clusters` | 10 | BayesianGMM components per numeric column |
| `tabular` | True | use `TabularFlows` (encoder + flow). False means a raw flow on the label-encoded matrix |
| `n_iter_min` | 100 | min epochs before early stopping |
| `n_iter_print` | 50 | epochs between early-stopping checks |
| `patience` | 5 | checks without improvement before stopping |
| `patience_metric` | `detection_mlp` | early-stopping criterion (minimize) |
| `sampling_patience`, `random_state`, `device`, `workspace`, `compress_dataset` | 500, 0, auto, `workspace`, False | base-class plumbing |

The `TabularFlows`/`NormalizingFlows` constructors have *different* defaults (5 layers, 10 units, 2 blocks, dropout 0.25, batch 100, patience 10/20, encoder 20 clusters). The plugin overrides all of them.

`hyperparameter_space` (`plugin_nflow.py:191–214`):
- `n_iter` ∈ [100, 5000] step 100
- `n_layers_hidden` ∈ [1, 10]
- `n_units_hidden` ∈ [10, 100]
- `batch_size` ∈ {32, 64, 128, 256, 512}
- `dropout` ∈ [0, 0.2]
- `batch_norm` ∈ {T, F}
- `lr` ∈ {1e-3, 1e-4, 2e-4}
- `linear_transform_type` ∈ {lu, permutation, svd}
- `base_transform_type` ∈ all 6 options

## 4. Privacy properties
- Not DP. There is no gradient clipping or noise.
- An MLE flow can overfit and place high density near training points. This is the overfitting that DomiasMIA targets. The risk grows with capacity (layers, units, bins) and epochs.
- The default shallow, early-stopped configuration is unlikely to memorize strongly, but there is no guarantee. The 20% validation split reduces the training data but gives no privacy.
- Clipping to `[min, max]` emits the real extreme values exactly.

## 5. Strengths / weaknesses / when to use
- **Strengths**:
  - Exact likelihood and stable training (no adversarial game, no mode collapse in the GAN sense).
  - Smooth, well-calibrated continuous marginals, helped by GMM mode normalization for multimodal numerics.
  - Spline transforms model non-Gaussian shapes well.
- **Weaknesses**:
  - Flows are a poor fit for discrete and one-hot dims: there is no dequantization, and they rely on an argmax decode.
  - Flows are bijective, so they cannot represent disconnected supports or hard constraints exactly. Logical constraints (deterministic mappings, impossible combinations) are only approximated, and leakage between modes produces implausible rows.
  - High-cardinality categoricals: ≤ 10 levels become wide one-hots, which inflates D and slows autoregressive sampling. > 10 levels are treated as GMM-encoded numeric codes, which gives arbitrary neighbouring-code errors. On a toy deterministic copy of a 20-level categorical, only 72.5% of nflow samples kept the relation.
  - Implementation bugs: no best-checkpoint restore, exceptions silently dropped during generation.
  - Training cost is dominated by the periodic `detection_mlp` checks.
- **When to use**: a likelihood-based neural baseline that sits between VAEs and diffusion models. Increase `n_layers_hidden` (for example 4–8) for real capacity. Prefer coupling (`rq-coupling`) when sampling speed matters.

## 6. References
- C. Durkan, A. Bekasov, I. Murray, G. Papamakarios. "Neural Spline Flows." NeurIPS 2019. https://arxiv.org/abs/1906.04032
- G. Papamakarios, T. Pavlakou, I. Murray. "Masked Autoregressive Flow for Density Estimation." NeurIPS 2017. https://arxiv.org/abs/1705.07057
- M. Germain, K. Gregor, I. Murray, H. Larochelle. "MADE: Masked Autoencoder for Distribution Estimation." ICML 2015. https://arxiv.org/abs/1502.03509
- L. Dinh, J. Sohl-Dickstein, S. Bengio. "Density estimation using Real NVP." ICLR 2017. https://arxiv.org/abs/1605.08803
- T. Müller, B. McWilliams, F. Rousselle, M. Gross, J. Novák. "Neural Importance Sampling." ACM TOG 2019. https://arxiv.org/abs/1808.03856
- G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, B. Lakshminarayanan. "Normalizing Flows for Probabilistic Modeling and Inference." JMLR 22, 2021. https://arxiv.org/abs/1912.02762
- L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni. "Modeling Tabular Data using Conditional GAN." NeurIPS 2019 (mode-specific normalization). https://arxiv.org/abs/1907.00503
- nflows library: https://github.com/bayesiains/nflows

# TabDDPM (Tabular Denoising Diffusion Probabilistic Model)

**TL;DR.** TabDDPM runs two diffusion processes side by side. Quantile-normalized numerical features get Gaussian diffusion, and each categorical feature gets multinomial (categorical) diffusion. A single MLP denoiser, conditioned on a timestep embedding and optionally on the class label, predicts the Gaussian noise and the clean categorical distribution. Synthcity's `ddpm` is a close port of the authors' code; most differences are in training-loop details and defaults. On mixed-type data it typically gives the best fidelity and utility among the non-DP deep models, at the cost of slow sampling and a measurable privacy (closeness-to-real) penalty.

| | |
|---|---|
| Family | Diffusion model (Gaussian + multinomial) |
| Differential privacy | No |
| Paper | A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko, "TabDDPM: Modelling Tabular Data with Diffusion Models", ICML 2023. [arXiv:2209.15421](https://arxiv.org/abs/2209.15421) |
| Synthcity plugin | `ddpm` (class `TabDDPMPlugin`, type `generic`) |
| Source | `synthcity/plugins/generic/plugin_ddpm.py` → `plugins/core/models/tabular_ddpm/__init__.py` (`TabDDPM`) → `tabular_ddpm/gaussian_multinomial_diffsuion.py` (`GaussianMultinomialDiffusion`, sic) + `tabular_ddpm/modules.py` (`DiffusionModel`) + `tabular_ddpm/utils.py` |

All paths below are relative to `.venv/lib/python3.11/site-packages/`. Line numbers refer to synthcity 0.2.12.

## 2. How it works (paper)

**Preprocessing.** Numerical features are passed through scikit-learn's Gaussian `QuantileTransformer`. Categorical features are one-hot encoded.

**Forward processes**, with a variance schedule $\beta_t$, $t=1..T$:
- **Numerical:** $q(x_t\mid x_{t-1})=\mathcal N\big(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\big)$, i.e. $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.
- **Categorical** (one-hot, $K_i$ classes; multinomial diffusion, Hoogeboom et al. 2021): $q(x_t\mid x_{t-1})=\mathrm{Cat}\big(x_t;(1-\beta_t)x_{t-1}+\beta_t/K_i\big)$. The process converges to the uniform distribution.

**Reverse model.**
- Numerical: a Gaussian whose mean is given by the $\epsilon$-prediction parameterization.
- Categorical: $p_\theta(x_{t-1}\mid x_t)=q\big(x_{t-1}\mid x_t,\hat x_0(x_t,t)\big)$, where the network predicts $\hat x_0$.
- One network handles all features: $\mathrm{MLP}(x)=\mathrm{Linear}(\mathrm{MLPBlock}(\dots))$ with $\mathrm{MLPBlock}(x)=\mathrm{Dropout}(\mathrm{ReLU}(\mathrm{Linear}(x)))$. Its input is $x=\mathrm{Linear}(x_{in})+t_{emb}+y_{emb}$, where $t_{emb}$ is a sinusoidal timestep embedding (dim 128) passed through an MLP.
- The paper uses only an MLP denoiser. I found no ResNet denoiser in the paper.

**Loss.** For each sampled $t$:
$$L^{\text{TabDDPM}}_t = L^{\text{simple}}_t + \frac{1}{C}\sum_{i\le C} L^i_t,$$
where $L^{\text{simple}}_t=\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$ and $L^i_t$ is the KL term of the multinomial variational bound for categorical feature $i$.

**Target handling.**
- Classification: class-conditional, i.e. $p_\theta(x_{t-1}\mid x_t,y)$ is learned and $y$ is sampled from its empirical distribution.
- Regression: the target is treated as an extra numerical feature.

**Training and tuning.**
- AdamW, cosine schedule, $T\in\{100,1000\}$.
- Search space: lr in $[10^{-5},3\cdot10^{-3}]$ (log-uniform), batch size in {256, 4096}, {5k, 10k, 20k} *iterations*, 2–8 MLP layers of width 128–1024, dropout 0.
- 50 Optuna trials, tuned on synthetic-data ML utility.

**Sampling.** Ancestral sampling over $T$ steps, starting from $\mathcal N(0,I)$ for numerical features and the uniform distribution for categorical ones. Then invert the quantile transform.

**Privacy (paper's own evaluation).** In distance-to-closest-record (DCR) comparisons, TabDDPM was more private than SMOTE but *less* private than the GAN/VAE baselines (TVAE, CTAB-GAN+). The authors attribute this to the baselines' much lower utility, and state they "do not give a definite answer" on whether TabDDPM is safe for privacy-critical use.

Verified against the arXiv HTML: the processes, the loss, preprocessing, the MLP formula, target handling, the search space, and the DCR conclusion.

## 3. Synthcity implementation

### Preprocessing

1. **`Plugin.fit`.** `DataLoader.encode()` label-encodes object, bool, and <15-unique columns (`plugins/core/dataloader.py:225-239`).
2. **`TabDDPMPlugin.__init__`** builds a `TabularEncoder(continuous_encoder="quantile", categorical_encoder="none")` (`plugin_ddpm.py:151-159`).
   - Columns with **≤10 unique values** are discrete and are passed through as integer codes.
   - All other columns get `GaussianQuantileTransformer` (`feature_encoder.py:260-283`, `n_quantiles = clip(n/30, 10, 1000)`, normal output).
   - **Consequence:** categoricals with more than 10 levels are diffused as *continuous* quantile-normalized integer codes. On decode, values are truncated with `astype(int)` (`dataloader.py:253-254`).
3. **`TabDDPM.fit`** (`tabular_ddpm/__init__.py:84-208`) re-detects discrete columns (≤10 unique values) and moves them to the end of the table (`__init__.py:100-110`).

### Model (`gaussian_multinomial_diffsuion.py`)

- **Schedules:** cosine (Nichol & Dhariwal) or linear (`:38-63`).
- **Gaussian part:**
  - $\epsilon$-parameterization (`gaussian_parametrization="eps"`).
  - Loss `mse` = $\lVert\epsilon-\hat\epsilon\rVert^2$ (`:384-408`), or `kl` (the variational bound).
  - The reverse variance is fixed: $\tilde\beta_1$, then $\beta_{t}$ for $t\ge1$ (`:291-298`).
- **Multinomial part:**
  - Log-space multinomial diffusion with `x0` parameterization.
  - Loss `vb_stochastic` = $\mathrm{KL}_t/p_t + \mathrm{KL}_{\text{prior}}$ (`:628-652`), with $t$ sampled uniformly. The loss is divided by the number of categorical features, matching the paper's $1/C$ (`:681-684`).
  - The denoiser receives $\log$ one-hot categorical inputs: $0$ and $\log(10^{-30})\approx-69$ (`utils.py:142-148`, `:668-671`). The paper describes one-hot inputs. This mirrors the authors' public code as far as I know; not verified.
- **Total loss:** `loss_multi + loss_gauss`, unweighted (`__init__.py:163-164`).
- **Denoiser** (`modules.py:62-117`): `proj = Linear(dim_in, dim_emb)` plus a sinusoidal time embedding through a 2-layer SiLU MLP, plus an optional label embedding. The label embedding is `nn.Embedding` for classification, or `Linear(1, ·)` when a regression-style `cond` is given.
  - Backbone: `model_type="mlp"`, defaulting to 3 hidden layers × 256, ReLU, dropout 0 (`modules.py:97-100`).
  - `model_type="tabnet"` is also accepted.
  - **`"resnet"` is mentioned in the plugin docstring but is not registered** in `factory.MODELS` (`factory.py:32-50`), so it would raise `ValueError: Unknown type`.

### Training loop (`tabular_ddpm/__init__.py:139-200`)

- AdamW with `lr` and `weight_decay`.
- **`n_iter` counts epochs**, not optimizer steps as in the paper's "iterations". The step count is `n_iter × ceil(N / batch_size)`.
- Linear lr decay to 0, recomputed from the epoch index on every step (`:62-66,168`). The final epoch runs with lr = 0.
- **The DataLoader is built without `shuffle=True`** (`:123`). Minibatches are the same contiguous slices in the same order every epoch, so if the input is sorted, batches are not i.i.d.
- An EMA copy of the denoiser (rate 0.999) is maintained (`:139-141,179-181`), but **it is never used for sampling**. `generate` uses `self.diffusion`.
- Optional validation and early stopping via `validation_size>0` plus `validation_metric` (`utils/callbacks.py:61-128`). The default is off.

### Conditioning (`plugin_ddpm.py:195-253`)

- With `is_classification=True`, the DataLoader's target column is split off via `X.unpack()` (`GenericDataLoader` defaults the target to the last column) and used as the class label. At generation, labels are sampled from the training class frequencies and re-attached as the target column.
- With `is_classification=False` (default), the target is just another feature. That is categorical if it has ≤10 unique values, otherwise numeric, which matches the paper's regression mode. A user `cond` (column name or array) may be given and is embedded with `Linear(1, ·)`.
- Quirk: `_generate` raises if `len(cond) > count` with the inverted message "less than" (`plugin_ddpm.py:243-244`).

### Sampling (`gaussian_multinomial_diffsuion.py:906-961`)

- Full ancestral sampling over all `num_timesteps` steps, in chunks of 2000 rows.
- A DDIM sampler exists (`sample_ddim`, `:873-904`), but the plugin does not expose it.
- Cost: `num_timesteps` denoiser forward passes per chunk, i.e. 1000 by default. Generation is much slower than for GANs and VAEs.
- `Plugin._safe_generate` (`strict=True`) then filters rows that violate the training schema and drops duplicates within each batch.

### Hyperparameters (`plugin_ddpm.py:93-120`)

| Param | Default | Meaning |
|---|---|---|
| `is_classification` | False | If True, class-conditional on the DataLoader target |
| `n_iter` | 1000 | Training **epochs** |
| `lr` | 0.002 | AdamW learning rate (linearly annealed to 0) |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `batch_size` | 1024 | Minibatch size |
| `num_timesteps` | 1000 | Diffusion steps $T$ (training and sampling) |
| `gaussian_loss_type` | `mse` | `mse` ($\epsilon$-MSE) or `kl` |
| `scheduler` | `cosine` | `cosine` or `linear` β schedule |
| `model_type` | `mlp` | Denoiser backbone (`mlp`, `tabnet`) |
| `model_params` | `{}` → `{n_units_hidden: 256, n_layers_hidden: 3, dropout: 0.0}` | Backbone kwargs (synthcity `MLP` args) |
| `dim_embed` | 128 | Projection, timestep, and label embedding dimension |
| `continuous_encoder` / `cont_encoder_params` | `quantile` / `{}` | Numeric preprocessing (any `FEATURE_ENCODERS` key) |
| `validation_size` / `validation_metric` | 0 / None | Optional held-out early stopping |
| `callbacks` | () | Training callbacks |
| `log_interval` | 100 | Steps between loss-history entries |
| `random_state`, `device`, `workspace`, `compress_dataset`, `sampling_patience` | 0, auto, `workspace`, False, 500 | Core plugin args |

**`hyperparameter_space`** (`plugin_ddpm.py:170-193`):

| Hyperparameter | Search space | Paper |
|---|---|---|
| `lr` | log-uniform [1e-5, **1e-1**] | upper bound 3e-3 |
| `batch_size` | int-log [256, 4096] | |
| `num_timesteps` | 10–1000 | |
| `n_iter` | int-log [1000, 10000] *epochs* | |

Layer count and width are commented out of the space.

## 4. Privacy properties

- **Not differentially private.**
- **Memorization risk is real.** Diffusion models are strong density estimators. With enough capacity and training they can generate near-copies of training rows, which is well documented for image diffusion. The TabDDPM paper itself finds its samples closer to real records (lower DCR) than TVAE/CTAB-GAN+ samples, though far from SMOTE-level copying.
- Low-cardinality categorical cells are reproduced exactly by construction. That is expected and not a leak in itself, but rare combinations of them can be.
- **Recommended checks:** DCR (`distance_closest_record`), `identifiability_score`, and DOMIAS membership inference. Watch these as `n_iter` grows.

## 5. Strengths, weaknesses, when to use

**Strengths**
- State-of-the-art fidelity and ML utility on many tabular benchmarks (per the paper).
- Stable, non-adversarial training with a simple MSE/KL loss.
- Native, principled handling of categoricals through multinomial diffusion.
- Class-conditional mode preserves label-conditional structure.

**Weaknesses**
- Slow sampling ($T$ network passes per batch).
- Many epochs are needed on small data: the defaults give about 10 steps per epoch at 10k rows.
- Sensitive to lr. The search space allows lr up to 0.1, which tends to diverge.
- Categoricals with more than 10 levels are treated as numeric.
- No shuffling, and the EMA weights go unused.
- Higher privacy risk than weaker models.

**Expectations on mixed-type data**
- Typically the strongest of the four methods in this group (ctgan, tvae, ddpm, adsgan) on marginal and pairwise fidelity and on train-on-synthetic, test-on-real utility, provided it gets enough training steps.
- Set `is_classification=True` with the target column declared for supervised datasets.

**When to use:** when fidelity and utility matter most and there is no formal privacy requirement.

## 6. References

- Kotelnikov, A., Baranchuk, D., Rubachev, I., Babenko, A. "TabDDPM: Modelling Tabular Data with Diffusion Models." ICML 2023. <https://arxiv.org/abs/2209.15421>. Code: <https://github.com/yandex-research/tab-ddpm>
- Ho, J., Jain, A., Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
- Hoogeboom, E. et al. "Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions." NeurIPS 2021.
- Nichol, A., Dhariwal, P. "Improved Denoising Diffusion Probabilistic Models." ICML 2021 (cosine schedule).
- Synthcity: <https://github.com/vanderschaarlab/synthcity>

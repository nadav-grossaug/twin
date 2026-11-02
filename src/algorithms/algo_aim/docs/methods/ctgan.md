# CTGAN (Conditional Tabular GAN)

**TL;DR.** CTGAN is a Wasserstein GAN with gradient penalty (WGAN-GP) for mixed-type tables. Continuous columns go through mode-specific normalization (a per-column variational Gaussian mixture). A conditional vector over discrete columns is fed to both networks, and training-by-sampling counters category imbalance. Synthcity's `ctgan` keeps the encoding and the WGAN-GP loss but drops PacGAN and log-frequency sampling. It also has a bug that misaligns the conditional vector with the real rows, so in practice it behaves close to an unconditional WGAN-GP over a CTGAN-style encoding. See [Synthcity implementation](#3-synthcity-implementation).

| | |
|---|---|
| Family | GAN (WGAN-GP, conditional) |
| Differential privacy | No |
| Paper | L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni, "Modeling Tabular Data using Conditional GAN", NeurIPS 2019. [arXiv:1907.00503](https://arxiv.org/abs/1907.00503) |
| Synthcity plugin | `ctgan` (type `generic`) |
| Source | `synthcity/plugins/generic/plugin_ctgan.py` → `plugins/core/models/tabular_gan.py` (`TabularGAN`) → `plugins/core/models/gan.py` (`GAN`); encoding in `plugins/core/models/tabular_encoder.py` and `feature_encoder.py`; sampler in `synthcity/utils/samplers.py` |

All paths below are relative to `.venv/lib/python3.11/site-packages/`. Line numbers refer to synthcity 0.2.12.

## 2. How it works (paper)

**The problem.** Tabular data mixes discrete and continuous columns. Continuous columns are often multimodal and non-Gaussian, and discrete columns are often highly imbalanced. A vanilla GAN that min-max-scales the columns suffers mode collapse and ignores minor categories.

**Mode-specific normalization.** For each continuous column $C_i$, fit a variational Gaussian mixture (VGM) that picks the number of modes $m_i$ itself. For a value $c_{ij}$, compute the mode responsibilities $\rho_k \propto \mu_k \mathcal{N}(c_{ij};\eta_k,\phi_k)$, then *sample* one mode $k$ from $\rho$. Encode the value as a one-hot vector $\beta_{ij}$ over modes plus the scalar $\alpha_{ij} = (c_{ij}-\eta_k)/(4\phi_k)$. A row becomes $r_j = \alpha_{1j}\oplus\beta_{1j}\oplus\dots\oplus d_{1j}\oplus\dots$, where each discrete column $d$ is one-hot encoded.

**Conditional generator and training-by-sampling.**
1. Pick a discrete column $D_{i^*}$ uniformly at random.
2. Build a PMF over its categories in which each category's mass is the *logarithm* of its frequency, then sample a category $k^*$.
3. Form the conditional vector $\mathrm{cond} = m_1\oplus\dots\oplus m_{N_d}$, where only $m_{i^*}^{(k^*)}=1$.
4. Sample a real row that has $D_{i^*}=k^*$.

The generator $G(z,\mathrm{cond})$ gets an extra cross-entropy loss between $m_{i^*}$ and its generated one-hot $\hat d_{i^*}$, so that it respects the condition. The critic sees $(r,\mathrm{cond})$ pairs. Log-frequency sampling over-represents minor categories during training. At generation time, conditions are drawn from the empirical frequencies, so the marginals are restored.

**Networks and loss.**
- Generator: two FC layers of 256 units with batch normalization and ReLU, and residual concatenation.
- Generator output: $\tanh$ for $\alpha$; Gumbel-softmax for $\beta$ and $d$.
- Critic: two FC layers with LeakyReLU and dropout.
- Loss: WGAN-GP, $L_D = \mathbb{E}[D(\tilde x)] - \mathbb{E}[D(x)] + \lambda\,\mathbb{E}[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-1)^2]$ and $L_G = -\mathbb{E}[D(\tilde x)] + \mathrm{CE}(m_{i^*},\hat d_{i^*})$.
- PacGAN: the critic sees "pacs" of 10 samples concatenated, which penalizes mode collapse.
- Training: Adam with lr $2\cdot10^{-4}$, batch size 500, 300 epochs.

These details were verified against the arXiv HTML, including GP weight 10 and critic LeakyReLU slope 0.2. The paper does not state the critic dropout rate numerically.

**Sampling.** Draw $z\sim\mathcal N(0,I)$ and a condition vector, run $G$, take the argmax of the mode and category one-hots, and invert the normalization: $c=4\phi_k\alpha+\eta_k$.

## 3. Synthcity implementation

### Pipeline

1. **`Plugin.fit`** (`plugins/core/plugin.py:218-254`). `DataLoader.encode()` label-encodes object, bool, and low-cardinality (<15 unique) columns to integers (`plugins/core/dataloader.py:225-239`). The generated data is decoded back the same way (`dataloader.py:246-260`, with `astype(int)`, i.e. truncation).
2. **`TabularEncoder`** (`tabular_encoder.py:57-291`).
   - A column is **discrete iff it has ≤ `categorical_limit`=10 unique values** (`tabular_encoder.py:75,147-148`, `utils/dataframe.py:12-23`). Discrete columns are one-hot encoded.
   - All other columns get a **`BayesianGMMEncoder`** (`feature_encoder.py:181-257`). This is sklearn `BayesianGaussianMixture(n_components=encoder_max_clusters, weight_concentration_prior=1e-3)`. The value is encoded as $(x-\mu_k)/(4\sigma_k)$, clipped to $[-0.99,0.99]$, plus a one-hot over **all** `n_components` modes.
   - The mode is the **argmax** of the responsibilities, not sampled as in the paper (`feature_encoder.py:223-227`).
   - The `weight_threshold=0.005` argument is stored but never used, so negligible modes are not pruned. Each continuous column always takes `1 + encoder_max_clusters` dimensions.
   - On decode, the scalar is clipped to $[-1,1]$, mapped back with the argmax mode, and clipped to the training min/max (`feature_encoder.py:244-257`).
   - **Consequence:** a categorical column with more than 10 levels is treated as a *continuous* integer code and GMM-encoded. Its arbitrary label order then acts as an ordinal scale.
3. **Conditional sampler.** With no user `cond`, `TabularGAN` builds a `ConditionalDatasetSampler` (`tabular_gan.py:201-208`, `utils/samplers.py:88-321`). The conditional dimension is the total one-hot width of the discrete columns.
4. **`GAN`** (`gan.py`):
   - Generator: `MLP(residual=True)`, input `n_units_latent + n_cond`. Residual layers concatenate their input to their output (`layers.py:56-64`, `mlp.py:151-186`), so widths grow layer by layer.
   - Output head (`tabular_encoder.py:270-291`, `layers.py:88-102`): Gumbel-softmax with $\tau=0.2$ on every one-hot block, including the GMM mode indicators. The continuous scalar gets **no activation** (`"none"`), where the paper uses $\tanh$.
   - Critic: plain MLP, no residual, no batch norm.

### Losses (`gan.py`)

- **Critic** (`gan.py:389-469`):
  - WGAN loss `errD = -mean(D(real)) + mean(D(fake))` (`gan.py:438`).
  - Gradient penalty on real/fake interpolates (with the conditional appended), $\lambda$=`lambda_gradient_penalty`=10 (`gan.py:672-700`).
  - The critic runs `discriminator_n_iter` (default 1) steps per generator step.
  - The real/fake masking at `gan.py:425-431` silently drops critic outputs that are exactly 0. This is harmless in practice.
- **Generator** (`gan.py:335-387`): `-mean(D(fake))` plus the conditional loss (`tabular_gan.py:212-264`). The conditional loss is an NLL between the generated softmax and the **real batch row's** category, computed for the column selected by `cond`. The paper uses the condition vector itself as the target.
- Gradient-norm clipping at `clipping_value`=1 on both networks (`gan.py:377-380,458-461`).
- Adam with betas (0.5, 0.999).

### Differences vs the paper

| Aspect | Paper | Synthcity 0.2.12 |
|---|---|---|
| PacGAN | pac=10 | **Absent.** The critic scores single rows (`gan.py:201-215`). |
| Training-by-sampling PMF | log-frequency | **Raw frequency** (`samplers.py:278-283`). The row-sampling marginal is therefore uniform, with no minority oversampling. |
| Resampling | every batch | Condition vectors and matched rows are drawn **once** at construction for $N$ rows with replacement (`samplers.py:300-318`), then restricted to the 80% train split. The same multiset is reused every epoch. In a 2,000-row test, only about 63% of train-split rows (about 50% of all rows) were ever used. |
| Cond/row pairing | matched | **Misaligned (bug).** `GAN._train_test_split` indexes the conditional array with *row ids* (`cond[train_idx]`, `gan.py:558-569`), but row $k$ of the conditional array was drawn for row `sampling_indices[k]`. In a synthetic test, the conditional category matched the paired real row only 72.5% of the time, close to chance (about 68%), versus 100% if aligned. **Analysis (not empirically confirmed):** the critic cannot learn $p(x\mid\mathrm{cond})$, and the conditional loss only pulls $G$ toward the marginal of the selected column. Expect `cond` to have weak control at generation time. |
| Mode assignment | sampled | argmax |
| Continuous output activation | tanh | none; values are clipped at decode |
| Generator batch norm | yes | no (`generator_batch_norm=False`) |
| Generator dropout | – | The plugin's `generator_dropout` is **ignored**: `TabularGAN(generator_dropout=0)` at `plugin_ctgan.py:253`. |
| Weight decay | – | The plugin's `weight_decay` is **ignored**: `GAN` builds both MLPs without `weight_decay` (`gan.py:184-215`), so the `MLP` default 1e-3 always applies (`mlp.py:123`). |
| Latent dim | 128 (my recollection of the reference code; not verified here) | `n_units_latent = generator_n_units_hidden` (500) |
| Epochs | 300 | `n_iter`=2000 *epochs*, with early stopping (below) |

**Early stopping.**
- Every `n_iter_print`=50 epochs, the model generates rows and computes the `detection_mlp` score against a 20% held-out split. Both sides are in *encoded* space (`gan.py:520-552,631-640`).
- Training stops after `patience`=5 checks without improvement, once `n_iter_min`=100 is reached.
- **Bug:** the "best" checkpoint is saved as `self.state_dict()` without a deep copy (`gan.py:636`). Those tensors alias the live parameters, so `load_state_dict(best)` at `gan.py:643` restores the *final* weights.
- 20% of the data is never used for training.

**Generation** (`tabular_gan.py:384-406`).
- Draws conditions from the sampler: column uniformly, category by frequency.
- `adjust_inference_sampling=True` reweights the per-category condition probabilities by fitting a softmax via `scipy.optimize.minimize`, so that the synthetic marginals match the real ones (`tabular_gan.py:370-382,437-460`).
- The user can pass `cond=` to `fit`/`generate`. That condition is one-hot encoded by `OneHotEncoder` and *replaces* the built-in sampler and conditional loss (`tabular_gan.py:188-199,217`). With a user `cond`, the pairing is correct, because no sampler is used and the `cond` rows are given.
- `Plugin._safe_generate` (`plugin.py:395-428`): with `strict=True` (default), rows violating the training schema's range/category constraints are dropped and within-batch duplicates are removed. It then resamples, up to `sampling_patience`=500 times.

### Hyperparameters (`plugin_ctgan.py:106-139`)

| Param | Default | Meaning |
|---|---|---|
| `n_iter` | 2000 | Training epochs (max) |
| `generator_n_layers_hidden` | 2 | Generator hidden layers |
| `generator_n_units_hidden` | 500 | Generator width; also the latent noise dimension |
| `generator_nonlin` | `relu` | Generator activation |
| `generator_dropout` | 0.1 | **Ignored** (forced to 0) |
| `generator_opt_betas` | (0.5, 0.999) | Adam betas, generator |
| `discriminator_n_layers_hidden` | 2 | Critic hidden layers |
| `discriminator_n_units_hidden` | 500 | Critic width |
| `discriminator_nonlin` | `leaky_relu` | Critic activation |
| `discriminator_n_iter` | 1 | Critic steps per generator step |
| `discriminator_dropout` | 0.1 | Critic dropout |
| `discriminator_opt_betas` | (0.5, 0.999) | Adam betas, critic |
| `lr` | 1e-3 | Adam lr (both networks) |
| `weight_decay` | 1e-3 | **Ignored** (MLP default 1e-3 always used) |
| `batch_size` | 200 | Minibatch size |
| `clipping_value` | 1 | Gradient-norm clip (0 disables) |
| `lambda_gradient_penalty` | 10 | WGAN-GP weight |
| `encoder_max_clusters` | 10 | BayesianGMM components per continuous column |
| `encoder` | None | Pre-fitted `TabularEncoder`. This is the only way to change `categorical_limit` |
| `dataloader_sampler` | None | Custom sampler (disables the built-in conditional) |
| `adjust_inference_sampling` | False | Re-fit condition probabilities after training |
| `patience` / `patience_metric` | 5 / `detection_mlp` | Early stopping |
| `n_iter_print` / `n_iter_min` | 50 / 100 | Eval interval / min epochs before stopping |
| `random_state`, `device`, `workspace`, `compress_dataset`, `sampling_patience` | 0, auto, `workspace`, False, 500 | Core plugin args |

**`hyperparameter_space`** (`plugin_ctgan.py:195-220`):

| Hyperparameter | Search space |
|---|---|
| generator / discriminator layers | 1–4 |
| generator / discriminator units | 50–150, step 50 (*far below the 500 default*) |
| nonlin | {relu, leaky_relu, tanh, elu} |
| `n_iter` | 100–1000, step 100 |
| generator / discriminator dropout | [0, 0.2] (generator value has no effect) |
| `discriminator_n_iter` | 1–5 |
| `lr` | {1e-3, 2e-4, 1e-4} |
| `weight_decay` | {1e-3, 1e-4} (no effect) |
| `batch_size` | {100, 200, 500} |
| `encoder_max_clusters` | 2–20 |

## 4. Privacy properties

- **Not differentially private.** `TabularGAN`/`GAN` support Opacus DP-SGD on the critic (`gan.py:589-608`), but `ctgan` never sets `dp_enabled`. Use `dpgan` for that.
- **Memorization risk.** The generator never sees real rows directly; only critic gradients flow back. The paper notes this makes CTGAN easier to make DP than TVAE. It gives some practical protection, but no guarantee. Long training and a large critic can still reproduce rare records, especially in small datasets or rare categorical combinations.
- **Recommended checks:** Synthcity `privacy` metrics (`identifiability_score`, `DomiasMIA*`, DCR-style `distance_closest_record`).

## 5. Strengths, weaknesses, when to use

**Strengths**
- Widely used baseline.
- The GMM encoding handles multimodal and skewed continuous columns.
- The Gumbel-softmax heads give valid one-hot outputs.
- Optional user conditioning (e.g. on the target) works correctly when passed via `cond=`.

**Weaknesses**
- GAN training is unstable and slow on CPU.
- Defaults (500-wide, 2000 epochs) are expensive.
- Early stopping relies on a noisy detection metric, and the best-checkpoint restore is broken.
- In this version, the built-in conditional mechanism is misaligned, PacGAN is missing, and minority oversampling is gone. Expect weaker rare-category fidelity than the original `sdv-dev/CTGAN`.
- Categorical columns with more than 10 levels are treated as continuous, which hurts high-cardinality categoricals.

**Expectations on mixed-type data**
- Reasonable continuous marginals.
- Low-cardinality categoricals are decent.
- Pairwise correlations and ML utility are typically below Bayesian networks and TVAE at small budgets.

**When to use:** as a "standard deep generative baseline", or to compare against published CTGAN numbers. Note that those numbers come from other implementations.

## 6. References

- Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K. "Modeling Tabular Data using Conditional GAN." NeurIPS 2019. <https://arxiv.org/abs/1907.00503>
- Gulrajani, I. et al. "Improved Training of Wasserstein GANs." NeurIPS 2017 (WGAN-GP).
- Lin, Z. et al. "PacGAN: The power of two samples in generative adversarial networks." NeurIPS 2018.
- Synthcity: Qian, Z., Cebere, B.-C., van der Schaar, M. "Synthcity: facilitating innovative use cases of synthetic data in different data modalities." NeurIPS 2023 Datasets & Benchmarks. <https://github.com/vanderschaarlab/synthcity>

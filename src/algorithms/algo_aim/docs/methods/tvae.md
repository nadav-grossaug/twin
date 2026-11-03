# TVAE (Tabular Variational Autoencoder)

**TL;DR.** TVAE is a VAE over the same mode-specific-normalized encoding as CTGAN. Its decoder outputs a Gaussian for each continuous scalar and a categorical distribution for each one-hot block, and it is trained by maximizing the ELBO. Synthcity's `tvae` differs from the paper in three main ways:
- It is a *conditional* VAE (it reuses CTGAN's conditional-vector sampler, including its misalignment bug).
- It uses a fixed-variance squared error instead of learned per-column variances.
- It runs larger networks by default.

It is stable and fast, and usually one of the stronger deep baselines on mixed-type data.

| | |
|---|---|
| Family | VAE (conditional, in Synthcity) |
| Differential privacy | No |
| Paper | L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni, "Modeling Tabular Data using Conditional GAN", NeurIPS 2019 (TVAE is introduced there as a baseline). [arXiv:1907.00503](https://arxiv.org/abs/1907.00503) |
| Synthcity plugin | `tvae` (type `generic`) |
| Source | `synthcity/plugins/generic/plugin_tvae.py` → `plugins/core/models/tabular_vae.py` (`TabularVAE`) → `plugins/core/models/vae.py` (`VAE`, `Encoder`, `Decoder`); encoding in `tabular_encoder.py`/`feature_encoder.py`; sampler in `utils/samplers.py` |

All paths below are relative to `.venv/lib/python3.11/site-packages/`. Line numbers refer to synthcity 0.2.12.

## 2. How it works (paper)

**Representation.** TVAE uses the same encoding as CTGAN. Each continuous value becomes a scalar $\alpha$ plus a VGM mode one-hot $\beta$, and each discrete column becomes a one-hot $d$. See [ctgan.md](ctgan.md).

**Model.**
- Prior: $z\sim\mathcal N(0,I)$.
- Encoder $q_\phi(z\mid r)$: two 128-unit FC layers outputting $\mu$ and $\sigma=\exp(\tfrac12\mathrm{FC}(h))$.
- Decoder $p_\theta(r\mid z)$: two 128-unit FC layers. It outputs $\hat\alpha_i=\tanh(\cdot)$ with $\alpha_i\sim\mathcal N(\hat\alpha_i,\delta_i)$, where $\delta_i$ is a learned per-column variance. Each $\beta_i$ and $d_i$ follows a softmax categorical distribution.

**Objective.** Maximize the ELBO:
$$\mathcal L = \mathbb E_{q_\phi(z\mid r)}\big[\log p_\theta(r\mid z)\big] - \mathrm{KL}\big(q_\phi(z\mid r)\,\Vert\,\mathcal N(0,I)\big),$$
where $\log p_\theta$ is a sum of Gaussian log-likelihoods for the $\alpha$'s and cross-entropies for the one-hots. Training uses Adam with lr 1e-3.

Verified against the arXiv HTML: the architecture, the $\sigma$ parameterization, the learned $\delta_i$, and lr 1e-3. In the paper TVAE is unconditional; it has no training-by-sampling.

**Sampling.** Draw $z\sim\mathcal N(0,I)$, decode, sample or argmax the one-hots, and invert the normalization.

## 3. Synthcity implementation

### Encoding

The encoding is identical to `ctgan`'s (`tabular_encoder.py`, `feature_encoder.py:181-257`):
- `DataLoader.encode()` label-encodes the columns.
- Columns with ≤10 unique values are one-hot encoded.
- All other columns, **including categoricals with more than 10 levels**, get a `BayesianGMMEncoder` with `n_components=data_encoder_max_clusters`. This uses argmax mode assignment, $(x-\mu_k)/(4\sigma_k)$ clipped to ±0.99, and no pruning of low-weight modes.

### Conditioning

`TabularVAE` (`tabular_vae.py:138-147`) builds a `ConditionalDatasetSampler` when no `cond` is given. It appends the CTGAN-style conditional vector to the inputs of both the encoder and the decoder (`vae.py:252-273`). It also adds the same conditional NLL penalty as `ctgan` (`tabular_vae.py:149-198`), on the reconstruction. The sampler issues described in [ctgan.md](ctgan.md#3-synthcity-implementation) apply unchanged:
- frequency (not log-frequency) category sampling;
- a fixed one-off resample of $N$ rows that covers only about 50% of distinct rows;
- conditional vectors **misaligned** with the real rows (`vae.py:360-374`, same `cond[train_idx]` indexing).

A user `cond=` replaces the sampler (one-hot via `OneHotEncoder`) and is paired correctly.

### Architecture (`vae.py:21-118`, `plugin_tvae.py:196-225`)

- **Encoder:** an `MLP` with `encoder_n_layers_hidden-1` hidden layers, then a linear layer of width `encoder_n_units_hidden`, then linear heads for $\mu$ and $\log\sigma^2$. There is no nonlinearity between the shared MLP's final linear layer and the heads. No residual connections, no batch norm, dropout 0.1.
- **Decoder:** a residual `MLP` (concatenating skips, `layers.py:56-64`).
  - Output heads (`tabular_encoder.py:270-291`): **Gumbel-softmax ($\tau=0.2$)** on every one-hot block, and `tanh` on each continuous scalar.

### Loss (`vae.py:499-540`)

$$\mathcal L = \texttt{loss\_factor}\cdot\frac1B\Big[\sum_{\text{one-hot}}\mathrm{NLL}(\log\tilde p,\ \arg\max r) + \sum_{\text{cont}} 50\,(\hat\alpha-\alpha)^2\Big] + \frac1B\mathrm{KL} + \text{cond-NLL}$$

- The factor $50 = 1/(2\cdot0.1^2)$ makes the continuous term a Gaussian NLL with a **fixed** $\sigma=0.1$. The paper learns $\delta_i$.
- The categorical NLL is computed on Gumbel-softmax *samples*, not on clean softmax probabilities.
- `loss_factor=1` in the plugin. The `VAE` class default is 2 (`vae.py:217`).
- A `robust_divergence` loss strategy exists (`vae.py:542-601`), but the plugin does not expose it.

### Training (`vae.py:376-458`)

- Adam with lr and `weight_decay`. These *are* honored, unlike in `ctgan`.
- Trains on the sampler's 80% split. Every `n_iter_print`=50 epochs it computes the ELBO on the 20% held-out split. It stops after `patience`=5 non-improving checks once past `n_iter_min`=100.

### Bugs and quirks

- **Gradient clipping is a no-op.** `clip_grad_norm_` runs *before* `loss.backward()`, right after `zero_grad()` (`vae.py:417-423`). It therefore clips zero gradients, and `clipping_value` has no effect.
- **Best-checkpoint restore is a no-op.** `best_state_dict = self.state_dict()` is not deep-copied (`vae.py:448,456`), so the final weights are kept.
- **Deterministic continuous outputs.** At generation, continuous scalars come straight from the decoder mean, with no observation noise added. The original `sdv-dev/CTGAN` TVAE adds $\mathcal N(0,\delta_i)$ noise at inverse transform; that comes from my reading of that library and was not verified here. One-hots are argmaxed after the Gumbel-softmax head, which is stochastic.
- **Partial final batch.** `generate` draws `count // batch_size + 1` latent batches and truncates (`vae.py:308-348`).
- **Strict sampling.** `Plugin._safe_generate` (`plugin.py:395-428`, `strict=True`) drops rows outside the training schema and duplicates within each batch, then resamples.

### Hyperparameters (`plugin_tvae.py:94-122`)

| Param | Default | Meaning |
|---|---|---|
| `n_iter` | 1000 | Training epochs (max) |
| `n_units_embedding` | 500 | Latent dimension (paper: 128) |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | Adam L2 penalty |
| `batch_size` | 200 | Minibatch size |
| `decoder_n_layers_hidden` / `decoder_n_units_hidden` | 3 / 500 | Decoder depth and width (paper: 2 × 128) |
| `decoder_nonlin` | `leaky_relu` | Decoder activation |
| `decoder_dropout` | 0 | Decoder dropout |
| `encoder_n_layers_hidden` / `encoder_n_units_hidden` | 3 / 500 | Encoder depth and width |
| `encoder_nonlin` | `leaky_relu` | Encoder activation |
| `encoder_dropout` | 0.1 | Encoder dropout |
| `loss_factor` | 1 | Weight on the reconstruction term vs KL |
| `data_encoder_max_clusters` | 10 | BayesianGMM components per continuous column |
| `dataloader_sampler` | None | Custom sampler (disables the built-in conditional) |
| `clipping_value` | 1 | Gradient clip. **No effect** (bug above) |
| `n_iter_print` / `n_iter_min` / `patience` | 50 / 100 / 5 | Early stopping on validation ELBO |
| `random_state`, `device`, `workspace`, `compress_dataset`, `sampling_patience` | 0, auto, `workspace`, False, 500 | Core plugin args |

**`hyperparameter_space`** (`plugin_tvae.py:165-189`):

| Hyperparameter | Search space |
|---|---|
| `n_iter` | 100–500, step 100 |
| `lr` | {1e-3, 2e-4, 1e-4} |
| `weight_decay` | {1e-3, 1e-4} |
| `batch_size` | {64, 128, 256, 512} |
| `n_units_embedding` | 50–500, step 50 |
| encoder / decoder layers | 1–5 |
| encoder / decoder units | 50–500, step 50 |
| encoder / decoder nonlin | {relu, leaky_relu, tanh, elu} |
| encoder / decoder dropout | [0, 0.2] |

## 4. Privacy properties

- **Not differentially private.** There is no DP option anywhere in the VAE path.
- **Memorization risk is higher in principle than for a GAN.** The encoder sees each real row, and the decoder is trained to *reconstruct* it. With a 500-dimensional latent space, large networks, and many epochs, the model can overfit to individual records. Generation decodes prior samples rather than encoded real rows, so exact copies are not reproduced directly. A high-capacity VAE can still place prior mass near training points.
- **Recommended checks:** `identifiability_score`, DCR (`distance_closest_record`), and DOMIAS membership inference.

## 5. Strengths, weaknesses, when to use

**Strengths**
- Stable likelihood-based training with no adversarial dynamics.
- Faster and easier to tune than CTGAN.
- Usually competitive or better than CTGAN on utility (train-on-synthetic, test-on-real) and on pairwise fidelity. The original paper reports TVAE outperforming CTGAN on several benchmarks.
- Smooth latent space.

**Weaknesses**
- The fixed-σ MSE and the Gaussian prior tend to over-smooth. Expect tails and rare modes to shrink and multimodal continuous columns to become less sharp.
- The fixed variance and noise-free decoding can reduce diversity (*precision > recall*).
- The same high-cardinality-categorical-as-continuous issue as `ctgan`.
- The built-in conditional vector is effectively noise, because of the misalignment.

**Expectations on mixed-type data:** good marginals for low-cardinality categoricals, solid ML utility, and somewhat under-dispersed continuous variables.

**When to use:** as a fast, reliable deep baseline, or when GAN instability is a concern.

## 6. References

- Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K. "Modeling Tabular Data using Conditional GAN." NeurIPS 2019. <https://arxiv.org/abs/1907.00503>
- Kingma, D. P., Welling, M. "Auto-Encoding Variational Bayes." ICLR 2014.
- Sohn, K., Lee, H., Yan, X. "Learning Structured Output Representation using Deep Conditional Generative Models." NeurIPS 2015 (CVAE).
- Synthcity: <https://github.com/vanderschaarlab/synthcity>

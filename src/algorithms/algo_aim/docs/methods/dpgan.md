# dpgan

## 1. TL;DR

DPGAN trains a GAN whose discriminator is updated with DP-SGD (per-sample gradient clipping plus Gaussian noise, tracked by a privacy accountant); the generator only sees the discriminator, so it inherits the guarantee by post-processing. In Synthcity this is a WGAN-GP-style `TabularGAN` (mode-specific `TabularEncoder`, residual MLP generator) wrapped with Opacus. **Two implementation caveats matter for a benchmark at ε = 1:** (1) at the default `n_iter=2000` the calibrated noise multiplier is about 22, i.e. the signal is buried; (2) in the DP branch of `GAN._train_epoch_discriminator` the fake-sample and gradient-penalty gradients are accumulated outside Opacus' per-sample machinery and are then **overwritten** by `DPOptimizer.add_noise`, and the real-term is back-propagated with the opposite sign to the non-DP loss. In a short experiment the DP critic drifted to large negative scores and the generator collapsed. The formal guarantee is also only approximate (no Poisson sampling, a non-DP validation split).

| | |
|---|---|
| Family | GAN (WGAN-GP critic) with DP-SGD on the critic |
| DP | (ε, δ)-DP via Opacus (PRV accountant); `epsilon=1`, `delta=1/len(train split)` by default |
| Papers | L. Xie, K. Lin, S. Wang, F. Wang, J. Zhou. "Differentially Private Generative Adversarial Network." arXiv:1802.06739, 2018. Building blocks: M. Abadi et al. "Deep Learning with Differential Privacy." CCS 2016 (arXiv:1607.00133); I. Gulrajani et al. "Improved Training of Wasserstein GANs." NeurIPS 2017 (arXiv:1704.00028); Synthcity encoder follows Xu et al. (CTGAN) NeurIPS 2019 |
| Library | Opacus 1.5.2 (`PrivacyEngine`, `GradSampleModule`, `DPOptimizer`), PyTorch 2.2.2 |
| Synthcity plugin | `dpgan` → `synthcity/plugins/privacy/plugin_dpgan.py` (`DPGANPlugin`), built on `plugins/core/models/tabular_gan.py` and `gan.py` |

## 2. How it works (paper)

**DP-GAN** (Xie et al. 2018) is a Wasserstein GAN in which only the critic touches real data. Each critic update uses DP-SGD:

1. Sample a (Poisson) batch $B$ of real rows.
2. Compute per-example gradients $g_i = \nabla_w \ell(w; x_i)$ of the critic loss on the real rows.
3. Clip: $\bar g_i = g_i / \max(1, \|g_i\|_2 / C)$.
4. Noise: $\tilde g = \frac{1}{|B|}\big(\sum_i \bar g_i + \mathcal N(0,\sigma^2 C^2 I)\big)$.
5. Update the critic; the generator is updated from the critic's output on fake data only (no real-data access), so it is DP by post-processing.

Because the Wasserstein critic is Lipschitz-constrained, its gradients are naturally bounded, which is why the paper argues WGAN suits DP. The privacy cost $(\varepsilon,\delta)$ is the DP-SGD composition over critic steps, computed with the moments accountant (Abadi et al.). In the paper the critic is trained several times per generator step and the budget governs the total number of critic updates; training stops when the budget is exhausted.

## 3. Synthcity implementation

Sources (under `.venv/lib/python3.11/site-packages/synthcity/plugins/`): `privacy/plugin_dpgan.py`, `core/models/tabular_gan.py`, `core/models/gan.py`. Line numbers below refer to `gan.py` unless stated.

### 3.1 Preprocessing (not DP-protected)
- `TabularGAN` fits a `TabularEncoder` on the **real training data** (`encoder_max_clusters=5` for DPGAN): continuous columns get a `BayesianGMM` mode-specific normalization (mode index one-hot + normalized value), categorical columns are one-hot. Number of modes, mode means and variances are learned non-privately from the data and are used by decoding. They are part of the released model. A discrete/continuous decision uses the number of unique values (the base `Plugin.fit` additionally label-encodes categories and columns with < 15 unique values).
- Hence DP applies to the critic training only, **not to the encoder, to the column typing, nor to the schema** (min/max bounds and category sets used for post-hoc filtering by `_safe_generate`). This is the usual caveat of DP-GAN libraries.

### 3.2 Architecture and training loop
- Generator: residual MLP (`generator_n_layers_hidden=2`, `generator_n_units_hidden=500`, `relu`), latent size = `generator_n_units_hidden`, softmax output for one-hot blocks and linear output for continuous. The plugin passes `generator_dropout=0` and `generator_batch_norm=False` regardless of the `generator_dropout` argument (the argument is ignored). Critic: MLP (2 × 500 units, `leaky_relu`, dropout 0.1, no batch-norm, since batch norm is incompatible with per-sample gradients). Both use Adam with `lr=1e-3`, `weight_decay=1e-3`, betas (0.5, 0.999).
- `n_iter=2000` = number of epochs (`generator_n_iter`). Each epoch iterates the dataloader in batches of `batch_size=200`; **per batch**: one critic step (`discriminator_n_iter=1` repetitions) then one generator step (lines 135–253).
- WGAN-GP: critic loss $-\bar D(\text{real}) + \bar D(\text{fake})$ plus `lambda_gradient_penalty=10` times the gradient penalty (418–446); generator loss $-\bar D(\text{fake})$. `clipping_value=1` additionally clips the *final* parameter gradient norm of both networks (after the DP noise; post-processing, harmless).

### 3.3 The DP mechanism (lines 187–208, 334–354)
- `_train` wraps the critic with Opacus: `PrivacyEngine(secure_mode=dp_secure_mode).make_private_with_epsilon(module=discriminator, optimizer, data_loader, epochs=generator_n_iter, target_epsilon=dp_epsilon, target_delta=dp_delta, max_grad_norm=dp_max_grad_norm, poisson_sampling=False)`. The noise multiplier σ is solved so that `epochs × len(loader)` DP-SGD steps cost (ε, δ). Defaults: `epsilon=1`, `delta=None → 1/len(X)` where `X` is the (80%) training split, `dp_max_grad_norm=2`, `secure_mode=False`. The accountant is Opacus' default, **PRV** (not RDP).
- The noise multiplier for the defaults (n_train = 8000 from a 10 000-row dataset, batch 200, sample rate 0.025, δ = 1/8000, ε = 1) is, computed with `opacus.accountants.utils.get_noise_multiplier`:

| epochs | σ (PRV) | σ (RDP) |
|---|---|---|
| 50 | 3.6 | 4.0 |
| 300 | 8.8 | 9.5 |
| 2000 (default) | **22.5** | 24.5 |

  Noise std on the *averaged* gradient is σ·C/|B| = 22.5·2/200 ≈ 0.22 per coordinate, compared with an average clipped gradient norm of at most 2 spread over ~10⁶ parameters. Fewer epochs reduce σ, so at ε = 1 `n_iter` of order 50 to 300 is a more sensible regime than 2000 (tuning `n_iter` changes the noise level, not only training time).
- **Inside the critic step in DP mode** (lines 187–198), Synthcity does `disable_hooks()`, back-propagates the gradient penalty and the fake loss, `enable_hooks()`, then back-propagates `errD_real` (note: **not** `-errD_real`). Opacus only records per-sample gradients while hooks are enabled, so only the `errD_real` backward produces `grad_sample`. `DPOptimizer.add_noise` then assigns `p.grad = p.summed_grad + noise` (Opacus 1.5.2, `optimizers/optimizer.py:461-477`), **replacing** the `p.grad` accumulated by the hook-less backward passes (the code comment claims it "accumulates"). Net effect, from reading the code:
  1. the fake-sample term and the gradient penalty never influence the critic update;
  2. the DP critic step descends `mean(D(real))` (decreasing scores on real rows), whereas the non-DP critic descends `-mean(D(real)) + mean(D(fake))`. The generator maximizes `D(fake)`, so the signs are inconsistent with a working WGAN.
  An isolated experiment (3 000-row numeric table, 15 epochs, ε = 50 so noise is small): non-DP critic gave mean D(real) = +0.49, D(fake) = +0.37; DP critic gave D(real) = -30.5, D(fake) = -31.7. A default-config run (ε = 1, 10 epochs, 10 000-row mixed table) produced a collapsed sample (94% of one category, age mean 83 vs 53). These are short runs, not proof about convergence at 2000 epochs, but they indicate the plugin may not train a useful generator under DP. Verify on your data before trusting results.
- `poisson_sampling=False`, and the `DataLoader` is built with `shuffle=False` (default) and `sampler=dataloader_sampler` (None): batches are the same fixed consecutive slices each epoch. Opacus' RDP/PRV accountant assumes Poisson-sampled batches with rate $q = B/N$, so the reported ε is not a rigorous bound for this sampling scheme (typically optimistic).
- `discriminator_n_iter > 1` would take several optimizer steps per batch, while σ is calibrated for one step per batch, so the actual spend exceeds the target. Keep it at 1.
- A budget cap exists only through calibration: training for fewer epochs (early stopping) spends less than ε, never more.

### 3.4 Early stopping and the validation split (not DP)
`DPGANPlugin` builds a `WeightedMetrics([("detection","detection_mlp")])` as `patience_metric` by default, hence `_train_test_split` (300–317) **splits the data 80/20**. The critic (and δ) see only the 80% part, but every `n_iter_print=50` epochs a detection-MLP score between the held-out real 20% and fresh synthetic data picks the best checkpoint (`best_state_dict`) and may stop training (`patience=5`, `n_iter_min=100`). That selection uses real data outside the DP accounting; the 20% hold-out affects the released generator but is not covered by the guarantee. It also costs one detection-MLP fit each time (see timing below).

### 3.5 Conditioning, sampling, runtime
- `cond` is passed through but the docstring warns of a known issue when training conditionals with DP.
- Sampling is a plain forward pass; `_safe_generate` filters the decoded rows by schema bounds and retries (up to `sampling_patience` = 500 batches).
- Runtime on 2 CPU cores, 10 000 rows, 7 columns (about 80 encoded columns after mode-specific encoding): about 30 s per epoch with `n_iter_print=10` (includes the detection check). The default 2000 epochs would take many hours; early stopping triggers no earlier than ~300 epochs. For a benchmark set `n_iter` explicitly and `patience_metric` knowingly.

### 3.6 Hyperparameters

| param | default | notes |
|---|---|---|
| `n_iter` | 2000 | epochs; also the horizon used to calibrate σ |
| `generator_n_layers_hidden`, `generator_n_units_hidden` | 2, 500 | |
| `generator_nonlin` | `relu` | |
| `generator_dropout` | 0.1 | **ignored** (0 passed to the model) |
| `discriminator_n_layers_hidden`, `discriminator_n_units_hidden` | 2, 500 | |
| `discriminator_nonlin`, `discriminator_dropout` | `leaky_relu`, 0.1 | |
| `discriminator_n_iter` | 1 | critic steps per batch; keep 1 for correct accounting |
| `lr`, `weight_decay` | 1e-3, 1e-3 | shared by G and D |
| `batch_size` | 200 | also sets the DP sampling rate |
| `clipping_value` | 1 | extra norm clip on final grads (post-noise) |
| `lambda_gradient_penalty` | 10 | has no effect in DP mode (see 3.3) |
| `encoder_max_clusters` | 5 | BayesianGMM modes per continuous column |
| `epsilon`, `delta` | 1, None→1/len(train) | passed as `dp_epsilon`, `dp_delta` |
| `dp_max_grad_norm` | 2 | per-sample clip C |
| `dp_secure_mode` | False | CSPRNG noise (floating-point attack hardening), slower |
| `patience`, `n_iter_min`, `n_iter_print`, `patience_metric` | 5, 100, 50, detection_mlp | early stopping |
| `random_state` | 0 | |

`hyperparameter_space()` (201–235) tunes layers, units, nonlinearity, `n_iter ∈ [100,1000]`, dropouts, lr, weight decay, batch size and `encoder_max_clusters`; it does not tune ε.

## 4. Privacy properties

- **Nominal**: (ε, δ)-DP with respect to the 80% training split for the critic weights; generator by post-processing. Opacus is a mature library and the PRV accounting is tight *for Poisson sampling*.
- **Caveats**: (a) no Poisson sampling (fixed unshuffled batches), so the stated ε is approximate; (b) the encoder (GMM modes, category sets), schema bounds and column selection are fit non-privately; (c) the 20% validation split and detection-MLP-based checkpoint selection use real data outside the accountant; (d) δ = 1/n_train is larger than the common n⁻¹·⁰ to n⁻¹·¹ guideline for large data but is conventional in Synthcity; (e) secure mode is off, so floating-point noise attacks are theoretically possible; (f) the training-step issue in 3.3 means the guarantee is intact (nothing but clipped, noised real-term gradients reaches the update, because the other gradients are overwritten) but model utility is affected.
- **Empirically** a strongly noised GAN gives high distance-to-closest-record and low membership-inference AUC; at σ ≈ 22 almost no information about specific rows survives, but neither does much structure.

## 5. Strengths / weaknesses / when to use

- **Strengths**: genuine (ε, δ) accounting through a standard library; model-agnostic post-processing; handles mixed types through the shared `TabularEncoder`; supports GPU; the generator is a small residual MLP that samples instantly.
- **Weaknesses**:
  - Huge noise at ε = 1 with the default number of epochs (σ ≈ 22); quality depends strongly on `n_iter` and `batch_size` (the DP-SGD trade-off).
  - The DP branch of the critic step looks incorrect (3.3); training can collapse.
  - Slow on CPU because of per-sample gradients and the detection-MLP early-stopping check.
  - Only the critic is protected; encoder and checkpoint selection leak a little.
  - 20% of the rows are not used for training.
- **Mixed-type expectations (n ≈ 10 k, ε = 1)**: expect mode collapse or near-marginal-only output, categoricals dominated by a few levels, numeric columns approximating the encoder's GMM modes rather than the real joint distribution.
- **When to use**: as the canonical DP deep-learning baseline, **after** setting `n_iter` to a few hundred at most, checking the critic scores during training (log level DEBUG prints "Privacy budget"), and ideally comparing against a non-DP `TabularGAN` to see what the DP path loses.

## 6. References

- L. Xie, K. Lin, S. Wang, F. Wang, J. Zhou. "Differentially Private Generative Adversarial Network." arXiv:1802.06739, 2018. https://arxiv.org/abs/1802.06739
- M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. "Deep Learning with Differential Privacy." CCS 2016. https://arxiv.org/abs/1607.00133
- I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, A. Courville. "Improved Training of Wasserstein GANs." NeurIPS 2017. https://arxiv.org/abs/1704.00028
- A. Yousefpour et al. "Opacus: User-Friendly Differential Privacy Library in PyTorch." arXiv:2109.12298, 2021. https://arxiv.org/abs/2109.12298
- S. Gopi, Y. T. Lee, L. Wutschitz. "Numerical Composition of Differential Privacy." NeurIPS 2021 (PRV accountant). https://arxiv.org/abs/2106.02848
- L. Xu, M. Skoularidou, A. Cuesta-Infante, K. Veeramachaneni. "Modeling Tabular Data using Conditional GAN." NeurIPS 2019. https://arxiv.org/abs/1907.00503
- Synthcity: Qian, Cebere, van der Schaar. arXiv:2301.07573.

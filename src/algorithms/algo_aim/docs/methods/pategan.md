# pategan

## 1. TL;DR

PATE-GAN trains a GAN whose discriminator ("student") never sees real data directly. Instead, an ensemble of teacher classifiers trained on disjoint real-data partitions votes on whether each generated sample looks real, the votes are noised with Laplace noise, and the noisy label is what the student is trained on. The privacy cost is tracked by a data-independent/-dependent moments accountant over the number of teacher queries. **In Synthcity 0.2.12 the student discriminator is also fed the real rows directly with label 1, so the PATE mechanism does not protect the released generator.** In addition, the default noise parameter `lamda=1e-3` (Laplace scale 1000 on vote counts ≤ 10) makes the teacher labels essentially random coin flips, the plugin's `delta` argument is ignored, and the privacy loop checks the budget only after a full outer iteration, so the default run at ε = 1 stops with a self-reported ε of about 2.1. The reported ε is not a meaningful guarantee for the output.

| | |
|---|---|
| Family | GAN with a PATE-trained discriminator (teacher ensemble + noisy aggregation) |
| DP | Claimed (ε, δ)-DP via a moments accountant (`epsilon=1`, `delta=1/(n√n)`). **Not valid for this implementation** (Section 3.3) |
| Paper | J. Jordon, J. Yoon, M. van der Schaar. "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees." ICLR 2019. https://openreview.net/forum?id=S1zk9iRqF7 |
| Building blocks | N. Papernot et al. "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data" (PATE), ICLR 2017, arXiv:1610.05755; Papernot et al. "Scalable Private Learning with PATE", ICLR 2018, arXiv:1802.08908 |
| Libraries | scikit-learn `LogisticRegression` or XGBoost `XGBClassifier` as teachers; PyTorch GAN from Synthcity `TabularGAN` |
| Synthcity plugin | `pategan` → `synthcity/plugins/privacy/plugin_pategan.py` (`PATEGANPlugin`, `PATEGAN`, `Teachers`) |

## 2. How it works (paper)

PATE-GAN modifies the standard GAN so that the discriminator's training signal about real data passes through the PATE aggregation mechanism.

**Players.** Generator $G$, $k$ **teacher** discriminators $T_1..T_k$, one **student** discriminator $S$.

**Algorithm** (repeat until the privacy budget is spent):
1. Partition the real data into $k$ disjoint subsets $D_1..D_k$ (each record is in exactly one).
2. **Teacher step**: train each $T_i$ to separate real samples from $D_i$ (label 1) from generated samples (label 0).
3. **Student step**: draw a minibatch of generated samples $\tilde x = G(z)$. For each, obtain the noisy teacher label

$$\mathrm{PATE}_\lambda(\tilde x) = \operatorname{argmax}_{j\in\{0,1\}}\big(n_j(\tilde x) + \mathrm{Lap}(1/\lambda)\big),$$

   where $n_j$ is the number of teachers voting $j$. Train $S$ on the pairs $(\tilde x, \mathrm{PATE}_\lambda(\tilde x))$ **only**: $S$ sees no real data.
4. **Generator step**: update $G$ to fool $S$.

Each call to $\mathrm{PATE}_\lambda$ is a $\lambda$-DP Laplace mechanism (changing one real record changes one teacher, hence one vote count by at most 1). A moments accountant tracks the cumulative cost over all queries, with a data-dependent bound using the vote gap; the result is (ε, δ)-DP for $S$ and, by post-processing, for $G$. Training stops when $\hat\varepsilon$ exceeds the target, so the number of queries is bounded by the budget. In the paper the noise scale 1/λ is chosen relative to the number of teachers so that the vote gap dominates the noise.

## 3. Synthcity implementation

Source: `.venv/lib/python3.11/site-packages/synthcity/plugins/privacy/plugin_pategan.py` (line numbers refer to that file unless stated), with the GAN in `plugins/core/models/gan.py` and `tabular_gan.py`.

### 3.1 Preprocessing (not DP-protected)
`TabularGAN` fits a `TabularEncoder` (`encoder_max_clusters=5`) on the real data; mode-specific normalization for numerics (BayesianGMM) and one-hot for categoricals (the base plugin first label-encodes objects and columns with < 15 unique values). The encoded real data `X_train_enc` is what teachers see. The encoder is fit non-privately and is part of the released model. Not accounted by the PATE budget.

### 3.2 Teachers (38–139)
- `n_teachers=10` teachers; `samples_per_teacher = int(n / n_teachers)` (1000 for n = 10 000). **A new random permutation** of the data is drawn at every outer iteration, so partitions are disjoint within an iteration but differ across iterations (acceptable for composition).
- Teacher template: `teacher_template="xgboost"` in the plugin (the inner `PATEGAN` class defaults to `"linear"`, a mismatch that only matters if you instantiate it directly). XGBoost teachers are built with `{"verbosity":0, "depth":3, "nthread":2}`; `depth` is not an XGBoost parameter (`max_depth` is), so the depth setting is silently ignored. Linear teachers use `LogisticRegression(solver="sag", max_iter=10000)` without feature scaling.
- Each teacher trains on 1000 real encoded rows (label 1) plus 1000 generator samples drawn from the **current generator** (label 0). Teachers are retrained from scratch at every outer iteration (every call to `Teachers.fit`).
- `pate_lamda` (111–139): collects the 0/1 predictions of all teachers, counts `n0`, `n1`, adds one Laplace draw of **scale `1/lamda`** per sample to `n1`, and outputs the label `((n1 + noise)/(n0+n1) > 0.5)`. Note the noise is added to the count *before* dividing by the number of teachers, so the decision is `n1 + Lap(1/λ) > (n0+n1)/2 ≈ 5`. With the default `lamda=1e-3` the noise scale is **1000** versus counts in [0, 10], so each label is a fair coin flip that depends on the votes only weakly (the probability of label 1 moves by about 0.5 percentage points between a 0-10 and a 10-0 vote): the teachers carry practically no information, but each query also costs almost no privacy.

### 3.3 Student/discriminator training and the real-data path (gan.py 135–253, plugin 275–296)
- The student is the WGAN-GP discriminator of `TabularGAN` (2 × 500 units, `leaky_relu`, dropout 0.1, `discriminator_n_iter=1`, `lr=1e-3`, `batch_size=200`) trained through `self.model.fit(X_train_enc, fake_labels_generator=..., encoded=True)`; `generator_n_iter=10` epochs per outer iteration; `n_iter` (the plugin argument; `max_iter` of the model) is the maximum number of **outer** iterations (default 200).
- `fake_labels_generator` returns the PATE labels for the generated batch, with an extra rule: if the number of samples labelled 1 is ≥ half of the batch, **all labels are set to 0** (and the accountant is not updated for that batch). This data-dependent branch is a function of the noisy labels, so it is post-processing in principle, but queries in skipped batches were answered and are not charged. At `lamda=1e-3` roughly half of the batches are skipped (191 of 500 batches were counted in our run).
- **The critic loss is** `-errD_real + errD_fake` with
  `errD_real = mean(D(real) ∪ D(fake labelled 1))`, `errD_fake = mean(D(real labelled 0) ∪ D(fake labelled 0))`.
  `true_labels_generator` is the GAN default (all ones), so **every real row of the minibatch is passed through the student with label 1**. In the paper the student is trained on generated samples only. Here the discriminator (and thus the generator) receives gradients computed directly on the real batch, with no clipping and no noise. This is the standard WGAN-GP real term, plus the PATE labels on the fake side.
  **Consequence**: the released generator is a function of the real data through a non-private channel. The PATE accountant, which counts only teacher queries, therefore does not bound the privacy loss of the output. The model is effectively a regular GAN with some extra noisy fake labels.
- Early stopping is disabled (no `patience_metric`), so all outer iterations run to completion.

### 3.4 Accountant (314–330, 297–309)
- `_update_moments_accountant`: $q = (2+\lambda|n_0-n_1|)/(4\exp(\lambda|n_0-n_1|))$, the data-dependent bound from the PATE paper.
- `_update_alpha`: for each 0-based moment index `l` (`lidx` in the code, `alpha` = 100 orders), computes
  `t = log((1-q)·((1-q)/(1-e^{2λ} q))^{l+1} + q·exp(2λ l + 1))`, clips `t` to `[0, 2λ²(l+1)(l+2)]`, sums over the batch, and adds to `alpha_dict[l]`. The `+ 1` in the exponent (`2λl + 1` rather than `2λ(l+1)`) looks like a precedence slip; in the small-λ regime this makes `t` large so the clip always binds. The accountant then reduces to the **data-independent** bound $2\lambda^2 (l+1)(l+2)$ per query, which is valid, so the slip is conservative.
- ε is recovered as $\hat\varepsilon = \min_l \frac{\alpha_l - \ln\delta}{l+1}$ and the loop `while epsilon_hat < epsilon and it < max_iter` (255–309) checks it **only after a complete outer iteration** (10 epochs × all batches ≈ 500 batches × 200 queries). With the defaults the first iteration alone accumulates 37 595 counted queries and gives $\hat\varepsilon$ = **2.11** for a target of 1 on our 10 000-row test; the loop then stops. The target cannot be respected at this granularity (smaller `generator_n_iter`, a smaller batch size, or larger `lamda` would be needed).
- **`delta` is ignored**: `PATEGAN.__init__` sets `self.delta = None` unconditionally (line 199), so `fit` always uses $\delta = 1/(n\sqrt n)$ (1e-6 for n = 10 000), whatever the plugin argument says.

### 3.5 Sampling
`sample` runs the generator and `TabularEncoder.inverse_transform`; `_safe_generate` then filters by schema bounds and retries. No conditional generation. Runtime for the defaults on 2 CPU cores, 10 000 rows, 7 columns: ≈ 40 s for the single outer iteration.

### 3.6 Hyperparameters

| param | default | meaning |
|---|---|---|
| `n_iter` | 200 | max outer iterations (teacher retraining rounds) |
| `generator_n_iter` | 10 | GAN epochs per outer iteration |
| `generator_n_layers_hidden`, `generator_n_units_hidden` | 2, 500 | |
| `generator_nonlin`, `generator_dropout` | `relu`, 0 | |
| `discriminator_n_layers_hidden`, `discriminator_n_units_hidden` | 2, 500 | |
| `discriminator_nonlin`, `discriminator_dropout`, `discriminator_n_iter` | `leaky_relu`, 0.1, 1 | |
| `lr`, `weight_decay`, `batch_size` | 1e-3, 1e-3, 200 | |
| `clipping_value` | 1 | gradient-norm clip |
| `encoder_max_clusters` | 5 | |
| `n_teachers` | 10 | number of disjoint partitions |
| `teacher_template` | `"xgboost"` | or `"linear"` |
| `epsilon` | 1.0 | target for $\hat\varepsilon$ |
| `delta` | None | **ignored**; fixed at 1/(n√n) |
| `lamda` | 1e-3 | Laplace scale = 1/lamda |
| `alpha` | 100 | number of moment orders |

`hyperparameter_space()` includes `lamda ∈ [0,1]` (0 gives a division by zero) and `delta` choices that have no effect.

## 4. Privacy properties

- **Nominal**: $(\hat\varepsilon,\delta)$-DP on the discriminator via PATE, post-processed into the generator.
- **Actual**: the student receives real rows directly (Section 3.3), so no DP statement holds for the released generator, regardless of the reported $\hat\varepsilon$. The remaining accountant is also internally loose: the target is overshot (about 2.1 vs 1 in our run), `delta` is not user-configurable, queries in skipped batches are not charged, and the encoder is non-private.
- **If the real-data path were removed** (student trained on teacher-labelled fakes only, as in the paper): at `lamda=1e-3` the labels are nearly random, so the generator would receive almost no signal; practical PATE-GAN needs λ in the range where vote counts dominate the noise (λ of order 1 or larger with ≥ 10 teachers), at which point ε grows quickly with the number of queries.
- **Empirical expectation**: sample quality similar to a plain Synthcity GAN with the same architecture (it does produce plausible mixed-type marginals), and privacy metrics comparable to a non-DP GAN with a similar number of training steps (low epochs, so moderate), not to a DP method.

## 5. Strengths / weaknesses / when to use

- **Strengths**: conceptually appealing teacher-ensemble design (no per-sample gradient clipping, so no Opacus constraints on the architecture); inexpensive in this configuration (≈ 40 s for the whole fit at ε = 1); works with the Synthcity mixed-type encoder.
- **Weaknesses**:
  - Real data bypasses the PATE mechanism (Section 3.3).
  - Default λ makes teachers uninformative; with a meaningful λ the budget is exhausted within a single outer iteration.
  - Budget overshoot: ε is evaluated once per outer iteration.
  - `delta` and teacher `depth` arguments silently ignored; `lamda=0` in the hyperparameter space divides by zero.
  - Few training steps at default settings (10 epochs per outer iteration × 1 iteration); GAN quality limited.
  - Teachers are retrained from scratch each outer iteration (cost grows with `n_iter` if the budget is not exhausted).
- **Mixed-type expectations (n ≈ 10 k, ε = 1)**: numeric and categorical marginals are roughly learned (as in a lightly trained GAN); correlations moderately. No DP-style distortion appears, because the real-data path dominates.
- **When to use**: only as a "labelled-DP" comparison point with the caveat above, or after patching the implementation (remove real rows from the student's loss, set λ appropriately, check ε before each batch). Do not report its self-reported ε as a guarantee.

## 6. References

- J. Jordon, J. Yoon, M. van der Schaar. "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees." ICLR 2019. https://openreview.net/forum?id=S1zk9iRqF7
- N. Papernot, M. Abadi, Ú. Erlingsson, I. Goodfellow, K. Talwar. "Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data." ICLR 2017. https://arxiv.org/abs/1610.05755
- N. Papernot, S. Song, I. Mironov, A. Raghunathan, K. Talwar, Ú. Erlingsson. "Scalable Private Learning with PATE." ICLR 2018. https://arxiv.org/abs/1802.08908
- M. Abadi et al. "Deep Learning with Differential Privacy." CCS 2016 (moments accountant). https://arxiv.org/abs/1607.00133
- I. Gulrajani et al. "Improved Training of Wasserstein GANs." NeurIPS 2017. https://arxiv.org/abs/1704.00028
- Synthcity: Qian, Cebere, van der Schaar. arXiv:2301.07573.

# ADS-GAN (Anonymization through Data Synthesis GAN)

**TL;DR.** ADS-GAN is a WGAN-GP whose generator is pushed to produce records that are *far* from real records. An identifiability penalty with weight $\lambda$ trades fidelity for a lower re-identification risk, which the paper formalizes as "$\epsilon$-identifiability". It is a heuristic privacy mechanism, **not differential privacy**. Synthcity's `adsgan` is its CTGAN-style `TabularGAN` plus a $-\lambda\,\lVert x-\tilde x\rVert_2$ term. Unlike the paper, the generator is not conditioned on the real record. The distance is therefore taken between randomly paired real and fake rows, which weakens the paper's per-record anonymization mechanism into a generic "spread away from the data" push.

| | |
|---|---|
| Family | GAN (WGAN-GP + identifiability penalty) |
| Differential privacy | **No.** It sits under `plugins/privacy/`, but there is no formal DP guarantee. |
| Paper | J. Yoon, L. N. Drumright, M. van der Schaar, "Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN)", IEEE J. Biomed. Health Inform. 24(8):2378–2388, 2020. [doi:10.1109/JBHI.2020.2980262](https://doi.org/10.1109/JBHI.2020.2980262) |
| Synthcity plugin | `adsgan` (type `privacy`) |
| Source | `synthcity/plugins/privacy/plugin_adsgan.py` → `plugins/core/models/tabular_gan.py` (`TabularGAN`) → `plugins/core/models/gan.py` (`GAN._loss_identifiability_penalty`) |

All paths below are relative to `.venv/lib/python3.11/site-packages/`. Line numbers refer to synthcity 0.2.12.

**Verification note.** I could not retrieve the paper's full text; IEEE needs JavaScript and it is not on arXiv. What I did verify:
- the abstract (PubMed 32167919);
- the authors' reference code, `vanderschaarlab/mlforhealthlabpub/alg/adsgan/adsgan.py` and `main_adsgan.py`, including the loss;
- Synthcity's implementation of the paper's identifiability metric (`metrics/eval_privacy.py:267-377`).

The equations in §2 are reconstructed from these sources. The theoretical claims are summarized from the abstract only.

## 2. How it works (paper)

**Identifiability.** Let the real data be $\{x_i\}_{i=1}^N$ and the synthetic data be $\{\hat x_j\}$. Use a weighted Euclidean distance with feature weights $w_k$; the paper weights features by inverse entropy, a detail I take from Synthcity's metric code (see below). Define:
- $r_i=\min_{j\ne i}\lVert w\odot(x_i-x_j)\rVert$, the distance from $x_i$ to its nearest *other* real record;
- $\hat r_i=\min_j\lVert w\odot(x_i-\hat x_j)\rVert$, the distance from $x_i$ to its nearest synthetic record.

A real record counts as "identifiable" if some synthetic record is closer to it than any other real record ($\hat r_i<r_i$). The **identifiability** of the synthetic set is
$$\mathcal I(\mathcal D,\hat{\mathcal D})=\frac1N\sum_i \mathbb 1[\hat r_i<r_i],$$
and $\hat{\mathcal D}$ is **$\epsilon$-identifiable** if $\mathcal I<\epsilon$. Intuitively, the synthetic data should not sit inside the "privacy ball" around any real person.

**Model.** A *conditional* GAN in which the generator takes the real record as input, $\hat x=G(z,x)$. The critic is a WGAN-GP critic on $\hat x$.

**Objective** (from the reference code):
$$L_D=\mathbb E[D(\hat x)]-\mathbb E[D(x)]+\eta\,\mathbb E\big[(\lVert\nabla D(\bar x)\rVert_2-1)^2\big],\qquad \eta=10$$
$$L_G=-\mathbb E[D(G(z,x))]\;-\;\lambda\,\mathbb E\big[\lVert x-G(z,x)\rVert_2\big].$$
The second generator term rewards each synthetic record for being far from the specific real record it was generated from. This is a Lagrangian relaxation of the identifiability constraint. Larger $\lambda$ gives more distance, so lower identifiability and lower fidelity. $\lambda=0$ reduces the model to a (conditional) WGAN-GP.

**Reference implementation defaults.** `lamda`=0.1, 10,000 iterations, batch 128, hidden dim 30, $z$-dim 10, lr 1e-4. Generator: tanh hidden layers with a sigmoid output (min-max-scaled data). Critic: ReLU layers.

**Paper's claims (abstract).** ADS-GAN "consistently outperformed state-of-the-art methods" on four datasets, in both utility (models trained on synthetic vs real data) and identifiability.

## 3. Synthcity implementation

### Core

`AdsGANPlugin` (`plugin_adsgan.py:112-277`) instantiates the same `TabularGAN` as `ctgan`, with `generator_extra_penalties=["identifiability_penalty"]` (`plugin_adsgan.py:170,252`). Everything in [ctgan.md §3](ctgan.md#3-synthcity-implementation) therefore applies:
- BayesianGMM mode-specific normalization plus one-hot encoding, with columns of ≤10 unique values treated as discrete;
- the built-in `ConditionalDatasetSampler` conditional vector, with its frequency-based sampling, one-off row resampling, and **cond/row misalignment bug**;
- WGAN-GP with $\lambda_{GP}=10$, no PacGAN, Gumbel-softmax heads, gradient clipping;
- `generator_dropout` and `weight_decay` ignored (`plugin_adsgan.py:249`, `gan.py:184-215`);
- detection-metric early stopping with the non-copying "best checkpoint" bug (`gan.py:636,643`).

### Identifiability term (`gan.py:702-711`, added to the generator loss at `gan.py:366-371`)

```python
-self.lambda_identifiability_penalty * (real_samples - fake_samples).square().sum(dim=-1).sqrt().mean()
```

This matches the reference code's `G_loss1` (an L2 norm, then the batch mean, with weight $\lambda$). Default `lambda_identifiability_penalty`=0.1, as in the reference.

### Differences vs the paper and reference code

| Aspect | Paper / reference | Synthcity |
|---|---|---|
| Generator input | $G(z,x)$, conditioned on the real record | $G(z,\mathrm{cond})$: noise plus the CTGAN conditional vector only. **The real record is not an input.** |
| Pairing in the penalty | $x_i$ vs $G(z,x_i)$, the same individual | Row $i$ of the real minibatch vs row $i$ of an independently generated fake minibatch, an **arbitrary pairing**. The conditional-vector columns are appended to both and cancel. |
| Effect of the penalty | Push each synthetic twin away from *its* source record | Increase the mean distance between fake rows and random real rows. In effect it pushes generated mass outward and inflates variance. It does not specifically target nearest-neighbor proximity, which is what $\epsilon$-identifiability measures. |
| Penalty space | min-max-scaled features | Encoded space: GMM scalar $\alpha\in[-1,1]$, mode one-hots, category one-hots. One-hot blocks contribute up to $\sqrt2$ each, so columns with many GMM modes or categories dominate. |
| Boundedness | – | The penalty is unbounded. Large $\lambda$ can overpower the adversarial term, and the generator can push values to the clip limits, which are clipped at decode. |
| Iterations | 10,000 minibatch iterations (reference code; I believe each iteration is one minibatch but did not verify the loop) | `n_iter`=10,000 **epochs** with early stopping. Early stopping usually triggers, but the nominal budget is huge. |
| Architecture | small tanh/sigmoid MLPs | 2×500 residual ReLU generator, 2×500 LeakyReLU critic |

### Hyperparameters (`plugin_adsgan.py:112-148`)

These are the same as `ctgan` except for the values marked †.

| Param | Default | Meaning |
|---|---|---|
| `n_iter` | **10000**† | Training epochs (max) |
| `lambda_identifiability_penalty` | **0.1**† | Weight $\lambda$ of the identifiability (distance) term |
| `encoder_max_clusters` | **5**† | BayesianGMM components per continuous column (ctgan: 10) |
| `generator_n_layers_hidden` / `generator_n_units_hidden` | 2 / 500 | Generator depth and width (width is also the latent dim) |
| `generator_nonlin` | `relu` | Generator activation |
| `generator_dropout` | 0.1 | **Ignored** (forced to 0) |
| `generator_opt_betas` | (0.5, 0.999) | Adam betas |
| `discriminator_n_layers_hidden` / `discriminator_n_units_hidden` | 2 / 500 | Critic depth and width |
| `discriminator_nonlin` | `leaky_relu` | Critic activation |
| `discriminator_n_iter` | 1 | Critic steps per generator step |
| `discriminator_dropout` | 0.1 | Critic dropout |
| `discriminator_opt_betas` | (0.5, 0.999) | Adam betas |
| `lr` | 1e-3 | Adam lr (both networks) |
| `weight_decay` | 1e-3 | **Ignored** (MLP default 1e-3) |
| `batch_size` | 200 | Minibatch size |
| `clipping_value` | 1 | Gradient-norm clip |
| `lambda_gradient_penalty` | 10 | WGAN-GP weight |
| `encoder`, `dataloader_sampler` | None | Pre-fitted encoder / custom sampler |
| `adjust_inference_sampling` | False | Re-fit condition probabilities after training |
| `patience` / `patience_metric` | 5 / `detection_mlp` | Early stopping |
| `n_iter_print` / `n_iter_min` | 50 / 100 | Eval interval / min epochs |
| `random_state`, `device`, `workspace`, `compress_dataset`, `sampling_patience` | 0, auto, `workspace`, False, 500 | Core plugin args |

**`hyperparameter_space`** (`plugin_adsgan.py:206-228`) covers:
- generator and discriminator layers (1–4), units (50–150), nonlin, and dropout (0–0.2);
- `lr` in {1e-3, 2e-4, 1e-4} and `weight_decay` in {1e-3, 1e-4};
- `encoder_max_clusters` in 2–20.

It does **not** include `lambda_identifiability_penalty`, `n_iter`, `batch_size`, or `discriminator_n_iter`. The privacy/fidelity knob must be swept manually.

**Conditional generation** works as in `ctgan`: the built-in sampler, or a user `cond=` (one-hot encoded, correctly paired). Note that `_generate` forwards `cond` without the 1-D reshape helper that `ctgan` uses (`plugin_adsgan.py:279-284`); `TabularGAN.forward` reshapes it anyway (`tabular_gan.py:396-401`).

## 4. Privacy properties

**Not differentially private.** ADS-GAN gives no $(\epsilon,\delta)$ guarantee. The paper's "$\epsilon$" in $\epsilon$-identifiability is an empirical fraction of real records, not a DP privacy-loss parameter. Do not compare it with DP epsilons.

**What the penalty buys.**
- **In the paper's design:** each synthetic record is generated *from* a real record but pushed away from it, with the strength of the push set by $\lambda$. This aims directly at keeping $\hat r_i>r_i$, i.e. low identifiability, while preserving distributional structure through the conditional GAN.
- **In Synthcity:** the record-level coupling is absent. The penalty spreads the generated distribution away from the data on average. It does lower the chance that synthetic points sit exactly on real points, but more crudely, and it can hurt fidelity more than the paper's version for the same reduction in identifiability.

**The λ trade-off.**
- $\lambda=0$ gives a plain WGAN-GP, the same as `ctgan` with these defaults.
- Increasing $\lambda$ increases the distance from synthetic to real rows, lowers identifiability, DCR-closeness and membership-inference AUC, and lowers fidelity and utility (worse marginals, inflated variance, worse detection scores).
- No closed form links $\lambda$ to a privacy level. Measure the trade-off by sweeping $\lambda$ (e.g. 0, 0.1, 1, 10) and evaluating identifiability, DCR, and DOMIAS against fidelity and utility.

**Measuring identifiability in Synthcity.** `metrics/eval_privacy.py:267-377` (`IdentifiabilityScore`, name `identifiability_score`) implements $\mathcal I$ above:
- `score` is computed on raw features; `score_OC` is computed on a one-class embedding and is the default.
- Quirk: the code computes per-feature entropies and then overwrites the weights with ones (`W = np.ones_like(W)`, around `eval_privacy.py:359`). So it uses *unweighted* Euclidean distance, not the paper's entropy weighting.

**Residual risks.**
- Attribute inference: the model can still reveal sensitive attributes through learned correlations.
- Rare-category membership: the distance penalty is dominated by one-hot blocks and does not protect specifically against singling out rare categorical combinations.
- No guarantee against a stronger adversary than nearest-neighbor re-identification.

## 5. Strengths, weaknesses, when to use

**Strengths**
- A simple, tunable privacy/fidelity knob on top of a standard tabular GAN.
- Directly targets the intuitive "no synthetic record too close to a real one" criterion.
- Cheap compared with DP training, with no budget accounting.

**Weaknesses**
- No formal guarantee.
- In Synthcity, the per-record mechanism of the paper is lost.
- It inherits all of Synthcity's `TabularGAN` issues: conditional misalignment, no PacGAN, ignored generator dropout and weight decay, broken best-checkpoint restore, and high-cardinality categoricals treated as continuous.
- The default `n_iter`=10000 epochs is very expensive if early stopping does not fire.
- Fewer GMM modes (5) than `ctgan` slightly reduces continuous fidelity.

**Expectations on mixed-type data:** fidelity and utility similar to or slightly below `ctgan` at $\lambda=0.1$, with slightly higher distance to real records. Larger $\lambda$ degrades fidelity quickly.

**When to use:** as a non-DP "privacy-aware" baseline showing the fidelity/identifiability trade-off. Prefer `dpgan`, `pategan`, `privbayes`, or `aim` when a formal DP guarantee is required.

## 6. References

- Yoon, J., Drumright, L. N., van der Schaar, M. "Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN)." IEEE Journal of Biomedical and Health Informatics 24(8):2378–2388, 2020. doi:10.1109/JBHI.2020.2980262. PMID 32167919.
- Reference code: <https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/adsgan>
- Gulrajani, I. et al. "Improved Training of Wasserstein GANs." NeurIPS 2017.
- Synthcity: <https://github.com/vanderschaarlab/synthcity>

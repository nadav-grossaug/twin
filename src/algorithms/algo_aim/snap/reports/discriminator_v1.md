# Real-vs-fake discriminator v1 on ego-Facebook

A classifier two-sample test of the synthetic graphs: train a model to tell random-walk subgraphs of the real graph from walks of a synthetic graph. AUC 0.5 means indistinguishable at walk level (the real-vs-real control, 0.51-0.53, is the achievable floor); 1.0 means trivially detectable. Design: `docs/realfake/PLAN.md` and `docs/realfake/snap.md`, log in `docs/realfake/PROCESS.md`. Figures: `snap/eval/notebooks/viz_discriminator.ipynb`. Raw numbers: `snap/eval/results/discriminator*.csv`, written by stage-3 `snap/eval/discriminate.py`.

## Method

Per method, 2,000 walk subgraphs per label (walk size log-uniform on 10..200 distinct nodes, same sampler for both labels), featurized into 104 features in four groups (size, degree, structure, attributes) after dropping representation giveaways (node ids, visit order, column order). Models: numpy logistic regression and sklearn gradient boosting; the **strongest** model is reported. Scoring: 3 folds rotating the held-out fake generation seed; real walks group-split by start node. Controls per fold: label permutation; real-vs-real (two halves of the real graph) once per model class. Baselines (Erdos-Renyi, Chung-Lu, DC-SBM, BTER, from `graph_aim.eval.baselines`) mark the easy-to-detect end.

**What this does and does not claim.** Detectability is a *fidelity* statement at walk level, not a privacy failure; the DP guarantee is unchanged. The converse also holds: a failed discriminator would have been evidence of fidelity, not of privacy.

## Results (strongest model, mean +- std over 3 folds)

| Method | eps | AUC | permuted | real vs real |
|---|---|---|---|---|
| real vs real (control) | - | 0.528 (gbm; logreg 0.515) | - | - |
| graph_aim | 0.5 | 0.9999 +- 0.0002 (gbm) | 0.55 | 0.528 |
| graph_aim | 1 | 0.9999 +- 0.0001 (gbm) | 0.52 | 0.528 |
| graph_aim | 2 | 0.9994 +- 0.0009 (gbm) | 0.50 | 0.528 |
| graph_aim | 4 | 0.9998 +- 0.0001 (gbm) | 0.48 | 0.528 |
| graph_aim | 8 | 0.9999 +- 0.0001 (gbm) | 0.52 | 0.528 |
| graph_aim | inf | 0.9999 +- 0.0001 (gbm) | 0.47 | 0.528 |
| graph_aim no-cap non-private | inf | 0.9999 +- 0.0000 (gbm) | 0.50 | 0.528 |
| erdos_renyi | - | 1.0000 +- 0.0000 (logreg) | 0.43 | 0.515 |
| chung_lu | - | 1.0000 +- 0.0000 (logreg) | 0.42 | 0.515 |
| dc_sbm | - | 0.9999 +- 0.0001 (gbm) | 0.48 | 0.528 |
| bter | - | 0.9993 +- 0.0010 (gbm) | 0.50 | 0.528 |

- Controls are clean: real-vs-real 0.515/0.528 (logreg/gbm), permuted labels ~0.5 within fold noise, baselines at the easy end. The numbers can be trusted.
- **Every graph_aim variant is near-perfectly detectable**, eps = 0.5 through inf and the no-cap non-private variant alike (gbm 0.9994-0.9999; logistic regression alone 0.993-0.997). At walk level the gap to the real graph comes from the generator, not from the privacy noise.
- AUC >= 0.995 in every walk-size bucket, slightly lowest at k 10-20: no sample size at which synthetic walks pass.
- This is a fidelity finding the scalar metrics hid: the sweep (`graph_aim_v1.md`) shows eps = 4 matching non-private BTER on degree/triangles/attributes, and link-prediction utility is 0.99 everywhere (`docs/snap/PROCESS.md` D12). The discriminator separates what they could not.

## What gives it away (gbm permutation importance, AUC drop per feature group)

| Group | eps 0.5 | 1 | 2 | 4 | 8 | inf | no-cap |
|---|---|---|---|---|---|---|---|
| attributes (86) | 0.43 | 0.38 | 0.35 | 0.31 | 0.28 | 0.24 | 0.26 |
| degree (11) | 0.001 | 0.008 | 0.043 | 0.013 | 0.010 | 0.033 | 0.018 |
| structure (4) | 0.003 | 0.004 | 0.014 | 0.006 | 0.010 | 0.008 | 0.009 |
| size (3) | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 |

The epsilon gradient the AUC hides lives here: the attributes group dominates everywhere and falls monotonically with budget (0.43 at eps 0.5 to 0.24 at inf) - more budget buys more faithful attribute statistics, never faithful enough to lower the AUC. Walk-level attribute statistics (marginal shares, `none`/`other` shares, same-value edge rates, attribute-pair co-occurrence on edges) are the giveaway at every epsilon; degree is a distant second, structure and size negligible.

**Caveat.** The attributes group mixes genuine statistics with a candidate generation artifact: exact `none`/`other` code shares (the open point in `docs/realfake/snap.md` section 2, issue 1 in `docs/realfake/PROCESS.md`). Until the group is split per feature, "attributes dominate" must not be read as "homophily is the sole cause".

## Limitations and next steps

- One real graph; all folds share it. The group split and the real-vs-real control bound what that sharing can leak, but samples are not independent.
- Split the attributes group per feature and re-run the importance, to isolate the `none`/`other` artifact from homophily and marginals (issue 1).
- Try walk sizes and features targeted at the known D14-era gaps from `viz_comparison.ipynb`: path lengths, off-diagonal degree mixing, same-group edge rate (42% synthetic vs 59% real).
- Feed what the discriminator finds back into the generator (homophily is already its weakest metric), then re-run; the discriminator is the acceptance test.

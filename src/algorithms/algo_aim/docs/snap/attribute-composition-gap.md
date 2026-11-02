# The attribute-composition gap

Status: open problem statement (2026-11-02), for the generator discussion. This file describes **what property of the real
data the synthetic graphs lack**, independent of the evaluation that found it. Evidence and numbers:
`docs/realfake/snap.md` section 5 and `snap/eval/notebooks/viz_discriminator_v1_1.ipynb`; log `docs/realfake/PROCESS.md` D6-D7.

## 1. The property, in one paragraph

In the real graph, attribute values are **locally concentrated**: any small connected region (a neighbourhood, a community,
the nodes reachable in a few steps) is dominated by a few values — one hometown, one locale, one school — so the value
histogram *inside* a region is skewed and low-entropy, and *which* value dominates differs from region to region. The
synthetic graphs reproduce the global value histograms and the global same-value edge rates, but their local regions look
like independent draws from the global mixture: every patch has roughly the same, globally-averaged composition. Real
patches are each "about" somewhere; synthetic patches are all "about" the average.

## 2. What this is NOT

- **Not the marginals.** Global 1-way and 2-way value histograms match well (v1 scoring, `sweep.csv` attr metrics).
- **Not the edge-level same-value rate alone.** v1.1 measured three homophily tables (school, hometown, location) and
  rewired edges against all of them. The targeted rates improved, yet local composition did not: a same-group edge *rate*
  can be met by thinly spread matching pairs, while the real graph meets it by **concentration** — whole regions sharing
  one value. Pairwise edge counts constrain the first moment of co-occurrence; concentration is a property of the joint
  distribution over a *region*, visible only in higher-order statistics (variance across regions, multi-hop agreement).
- **Not a bookkeeping artifact.** The suspicion that exact `none`/`other` shares were a generation artifact was tested and
  refuted (they carry ~1e-4 of the separability).

## 3. Evidence (summary; details in docs/realfake/)

The real-vs-fake track trains a classifier on small connected patches cut identically from real and synthetic graphs.
Framed as data properties, it established:

1. Patches of every variant, eps 0.5 through non-private, are separable from real patches almost perfectly — the gap is a
   generator property, not a privacy-noise property.
2. The separating information is **distributed over the per-patch value shares of many attributes at once** (no single
   value or attribute suffices; removing any one changes little). Per-patch value shares carry what is attributable;
   same-value edge rates and pairwise co-occurrence carry almost nothing *on top of them*.
3. hometown is the single strongest attribute, and the only one v1.1 visibly improved (its share-signal halved) — the
   multi-channel rewiring did what it aimed at, but edge-rate matching was not the binding constraint.
4. Secondary, much smaller gaps sit in degree (parent-degree profile of patches) and structure; they will become the
   binding constraint once composition is fixed. Known from `viz_comparison` (D14): paths too short, degree mixing too
   diagonal.

## 4. Measurable restatements (candidate targets, all checkable on both graphs)

The gap should close on statistics of this family — listed roughly from cheapest to hardest to release under DP:

| Statistic | What it captures |
|---|---|
| per-community attribute histograms (block x value counts, blocks from structure) | which value dominates where; region-to-region variation |
| two-hop same-value rate per attribute (agreement at distance 2, not only across an edge) | concentration beyond the edge level |
| distribution over regions of the dominant-value share (or entropy) of a region's value histogram | low-entropy local composition directly |
| within-group triangle share per attribute (already in `graph_aim.eval.coupling`, kept from package E) | whether closed structure lives inside value groups |
| joint (attribute value, community) assignment consistency across attributes | regions coherent in several attributes at once (hometown+locale+school together) |

The real graph will show: skewed per-region histograms with different modes per region, two-hop agreement well above the
edge-level rate times itself, dominant-share distribution concentrated near 1 for hometown/locale in many regions.
Current synthetic graphs will show: per-region histograms close to the global mixture, two-hop agreement near the
independence prediction, dominant shares near the global maximum share.

## 5. Requirements on a fix (not a design)

1. **Attribute assignment must be correlated across structurally close nodes**, calibrated so region-level composition
   statistics (section 4) match their released values — not only edge-count tables. Equivalently: structure and
   attributes must share a latent region, whether that is a block, a community, or the sampler's affinity groups.
2. **Several attributes must cohere on the same regions** (real communities share hometown AND locale AND school);
   independent per-attribute coherence is not enough if regions disagree across attributes.
3. Everything stays inside the existing pipeline rules: any new statistic is measured inside, released with a proved
   sensitivity and a budget share, listed in the artifact, and consumed outside from the artifact alone. More
   measurements means more noise per measurement; the composition statistics above compete with the existing workload.
4. Structure fidelity must not regress: package E showed region-aware changes to the sampler can wreck degree/clustering
   (degree KS 0.006 -> 0.080). Any fix is gated on the D14 structure metrics staying within the seed spread.

## 6. Acceptance check

`snap/eval/discriminate.py` (~15 min) is the regression test for the gap. Progress order, since the headline AUC is
saturated: (a) the attributes-group importance falls toward the degree group's level, (b) logistic-regression AUC falls
below ~0.9, (c) gradient-boosting AUC falls below the structure baselines (dc_sbm 0.9999, bter 0.9993), (d) long-term,
AUC approaches the real-vs-real floor (0.51-0.53). Alongside it, the section-4 statistics compared real vs synthetic in
the standard scoring. A fix that moves (a)-(c) without regressing D14 structure metrics is a win even if (d) stays far.

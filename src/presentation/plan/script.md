# Presentation script — hackathon demo (5 min)

## Slide 1: The problem (~40 sec)

"Every organisation has data it can't share. Not won't — can't. Legal, regulation, air-gap policy. Researchers and partner teams on the outside need to work with it, but the raw records never leave the building. Synthetic data sounds like the answer, but most approaches either leak too much about real people, or produce data so degraded it's useless. We need both: provably safe and actually useful."

## Slide 2: Solution — the air-gap architecture (~50 sec)

"Here's the idea. Real data stays inside. The only thing that ever crosses the boundary is a small, auditable bundle we call an artifact — a set of summary statistics, never records, never GPS points, never edges. A generator on the outside reads the artifact alone and produces synthetic data. Open the artifact and you can read exactly what it contains.

To make the guarantee rigorous, every statistic in the artifact is computed with differential privacy: calibrated noise is added so that any one person's data can shift the released numbers by at most a small, mathematically bounded amount. The privacy-fidelity trade-off is a dial you can tune."

**[Switch to demo]**

## Demo: graphs (~1 min 20 sec)

"Let me show you what comes out. This is a social graph — nodes are people, edges are friendships, colours are attributes like school or hometown.

*[Load the real graph, briefly pan/zoom — point out the clusters by colour.]*

Now the synthetic version — generated from the artifact alone, never saw the original.

*[Load the synthetic graph side by side.]*

The structure is there: degree distribution roughly matches, the community clusters exist. Let me show you two actual metrics: the degree distribution and the attribute mixing — how often people of the same group connect. The synthetic closely tracks the real."

## Demo: mobility (~1 min 20 sec)

"Second dataset: GPS mobility traces from Beijing. 182 users, 19,000 outings.

*[Show real traces on map tile — a few coloured paths.]*

The synthetic version is generated from the artifact — noisy summaries of outing lengths, stop patterns, co-visit frequencies. No raw coordinates in the artifact.

*[Show synthetic traces on the same map tile.]*

And the method is city-agnostic: we can anchor the same statistical skeleton to a different map. Amsterdam, Paris, anywhere.

*[Switch to Amsterdam or another city if available.]*

Two key metrics: outing-length distribution and revisit rate — how often a user returns to the same place. Both track the real data reasonably well."

## Slide 5: Is it useful? (~30 sec)

"Three checks. EDA distributions match — we just saw that. Downstream tasks: place-of-interest recovery on the mobility data, precision/recall 0.74/0.72 on held-out data. And we trained a classifier to tell real from synthetic — it can, at AUC 0.9999 — but the classifier tells us exactly what's off, which is our roadmap for v2."

## Slide 6: Real-vs-fake (~20 sec)

"The discriminator says: synthetic graphs are detectable mainly through attribute composition — a generator artifact we already know how to fix. Synthetic mobility is detectable through step-timing texture. Both are engineering problems, not fundamental limits. The pipeline works; these are the next pull requests."

# Composition round (v2): neighbourhood statistics and a refinement chain

Spec for the next round of Graph-AIM on the SNAP track (2026-11-03). Fixes the gap stated in
`docs/snap/attribute-composition-gap.md`. Builds on `docs/snap/graph-aim-plan.md` (v1 method, privacy model) and
`docs/snap/attribute-coupling-plan.md` (v1.1: `attr_degree`, several `attr_edge_pair` instances, multi-channel
rewiring). Code layout and boundary rules: `CLAUDE.md`, `snap/README.md`. Log: `docs/snap/PROCESS.md`.

**v2 is additive. Nothing in the v1 / v1.1 pipeline changes behaviour.** The v1 config, the v1 artifacts, the v1
generator path (`graph_aim.outside.generate.generate`) and every existing output directory stay byte-identical; this is
pinned by the existing fallback digests (`tests/graph_aim/outside/test_generate_attr_degree.py`) and by a new test that
fits the v1 config before and after this round. v2 lives in a new config (`snap/inside/config/graph_aim_v2.json`), new
measurement kinds that are emitted only when that config gives them a budget share, a new generator module that
*wraps* the v1 generator, a `--method v2` flag on stage 2, and its own output directories (`snap/transfer/artifacts_v2`,
`snap/outside/synthetic_v2`, `snap/eval/results_v2`).

## 1. Why: what v1.1 cannot express

The real-vs-fake classifier (D19, `docs/realfake/snap.md` section 5) separates real from synthetic walks at AUC
0.9999 for every variant including the non-private one, and the signal is the *per-patch value shares of many
attributes at once*: real neighbourhoods are each "about" one hometown, one locale, one school; synthetic ones are all
about the global mixture. v1.1 matched the edge-level same-value rates (three `attr_edge_pair` channels) and the
discriminator did not move.

The reason is the model class, not the calibration. In `generate.py`, attribute rows are i.i.d. across nodes (Chow-Liu
tree), and everything downstream only reshuffles which rows get which edges. With i.i.d. rows the composition of any
patch of size k concentrates on the global mixture with variance about 1/k: no edge-level rewiring can create
region-to-region variation that the attribute sampler never produced. Two numbers make the point: with the same
same-value edge rate of 0.50, a neighbourhood whose other half is spread over the remaining values has Simpson index
about 0.30, while a neighbourhood that is 90% one value and 10% one other has 0.82. First moments equal, second
moments far apart. The fix must introduce dependence between the attributes of structurally close nodes and calibrate
it against a released statistic of *local concentration*.

## 2. Design: tables of local statistics, matched by a refinement chain

The artifact stays what it is: noisy tables with proved L2 sensitivities. The generator keeps its one-sweep initializer
(the v1.1 `generate`, good at degrees and triangles) and adds a **refinement chain**: a Markov chain of
measure-preserving moves whose energy is the noise-normalised distance between the current synthetic graph and every
released table. Each released kind is one energy term with a cheap change statistic; the chain stops when every term is
within its own noise, so it never fits the noise.

Two chains, by what they move:

| Chain | Move | Preserved exactly | Energy terms (released kinds) | Package |
|---|---|---|---|---|
| label chain | swap the value of one attribute between two nodes of the same `attr_degree` stratum | `attr_1way`, `attr_degree`, the edge set | `attr_edge_pair` instances, `attr_neigh_simpson` instances, `attr_2way` tables | **C** (this round) |
| structure chain | degree-preserving double-edge swap (what `_rewire` does today) | `degree_hist`, attributes | `joint_degree`, `attr_edge_pair`, `clustering_by_degree`, later `ball2_size` | **D** (conditional) |

In exponential-family terms this is a degree-conditioned ERGM whose sufficient statistics are the released tables; the
artifact *is* the model. Nothing names a community: whatever neighbour dependence a dataset has (homophily, heterophily,
"similar age connects" on an ordered domain, degree mixing) is expressed as a table over a local neighbourhood and the
same engine consumes it. The sampling is sequential after initialization — every accepted move changes the state the
next proposals are scored against — which is exactly "an edge affects the likelihood of other edges" (a Markov random
graph), batched into disjoint proposals per round for vectorization, as `_rewire` already does.

A scale caveat the design depends on: a 1-hop concentration target makes neighbourhoods pure, but how *large* the
pure regions grow is decided by the structure — on a clustered graph the edge-disagreement energy puts region
boundaries on sparse cuts, so regions grow to community size on their own. The label chain can therefore only
produce regions as coherent as the synthetic structure's own communities, which D14 says are too weak (paths too
short, mixing too diagonal). Package A measures this before anything else is built; packages D and E are the
backstops if it binds.

## 3. Package A: diagnostics and eval metrics (must, first)

Where: `src/graph_aim/eval/composition.py` (library), `snap/eval/score.py` (new `cmp_*` columns), `tools/v2_diag/`
(throwaway diagnostic scripts, eval zone: they read the real graph). Nothing in `transfer` or `outside` may import any
of it.

**Composition metrics** (`composition_metrics(real, syn, *, groups, bin_edges, min_degree) -> dict`), generic over
the schema, for every configured attribute group (a group is a list of attribute names, product code as in
`attr_edge_pair`):

- `simpson_by_degree`: mean neighbourhood Simpson index per public degree bin, real vs synthetic vs absolute
  difference, plus the node-weighted mean over bins. Definition: for a node `u` with degree `d_u >= min_degree`,
  `s_u = sum_g (n_{u,g} / d_u)^2`, `n_{u,g}` = neighbours of `u` with group code `g`. This is exactly the statistic
  package B releases, so eval and artifact agree.
- `two_hop_same_rate`: share of 2-paths `(u, w, v)` whose endpoints have the same group code, real vs synthetic, and
  the independence prediction `sum_g a_g^2` (edge-endpoint shares) for reference.
- `dominant_share_hist`: histogram (10 bins on [0, 1]) over nodes with `d_u >= min_degree` of `max_g n_{u,g} / d_u`,
  real vs synthetic, and their total variation distance.
- `gap closed` against the permuted-attribute null (`coupling.permuted_null`) for the scalar metrics, same
  convention as `coupling.gap_closed_table`: 0 = as bad as the null, 1 = as good as the real graph.

`score.py` gets the `cmp_*` columns (flattened like `cpl_*`), groups taken from the config's `neigh_simpson_groups`
(fallback: `homophily_groups`), bins and `min_degree` from the config's `neigh_simpson_*` keys (fallback: the
`cluster_bin_edges` / `cluster_min_degree`). Existing columns are unchanged.

**Structure-community diagnostic** (`tools/v2_diag/community_strength.py`, writes `tools/v2_diag/community_strength.csv`
and a short note `docs/snap/v2-diagnostic.md`): for the real graph and the v1.1 synthetic graphs
(`snap/outside/synthetic_v1_1/`, eps 4 and inf, seed 0):

1. modularity and the size distribution of a Louvain / label-propagation partition (networkx, allowed in eval);
   mean conductance of the parts; top-32 adjacency eigenvalues (the spectrum package B releases).
2. the **painted upper bound**: keep the synthetic structure, detect its communities, and re-assign the sampled
   attribute rows so that each community is as pure as the real graph's Simpson level allows (greedy: sort rows by
   the primary group, deal them community by community within each `attr_degree` stratum). Measure
   `simpson_by_degree` and the per-walk value-share variance (walks sampled exactly as the discriminator does,
   `graph_aim.eval.discriminator.WalkSampler`) for real, synthetic-as-is, and synthetic-painted.

Decision rule recorded in the note: if the painted synthetic graph reaches at least 70% of the real per-walk share
variance for the walk sizes 10-200, package C alone is expected to move the discriminator and D/E wait; if not, package
E (latent tilt) is scheduled right after C.

**Outcome (2026-11-03, `docs/snap/v2-diagnostic.md`, `tools/v2_diag/community_strength.csv`): rule NOT met.** The
synthetic structure has communities that are far too small and too leaky: Louvain modularity 0.71 vs 0.835 real, 96-152
parts vs 16, median part size 7-42 nodes vs 231, node-weighted conductance 0.20 vs 0.04; the spectrum reproduces the top
three eigenvalues but is *larger* from rank 4 on (many medium blobs instead of a few strong regions). Painting
attributes optimally on that structure reaches 85% of the real per-walk share variance at walk size 10-39 but only
37-41% at 40-119 and 19-21% at 120-200 — the discriminator's walks cross several synthetic regions and re-mix. The
same painting on the real structure gives 1.33x the real variance, so the procedure is not the limit; the structure is.
Consequence: package C alone will match the 1-hop Simpson tables but not the patch-level statistics; **package E is
required and is scheduled right after C** (section 7, now specified in full). Two more findings: (i) the column-permuted
null of `coupling` is not a floor for the Simpson metric on groups dominated by the missing label (it breaks the
degree x attribute coupling and raises the i.i.d. level above the real value for hometown); the degree-stratified null
`composition.permuted_null_by_degree` (`cmp_*_gapdeg_*` columns) is the floor the acceptance criteria use. (ii) real
regions are coherent in school AND hometown at once (painting with hometown as the key restores only half of school's
concentration): a `school+hometown` joint instance is a sweep option for the budget step.

## 4. Package B: measurement kinds `attr_neigh_simpson` and `spectrum` (must)

Both are new kinds in the existing fixed-workload design (`measure.py` `KINDS`, `workload()`, `exact_values()`,
`sensitivity()`, `postprocess()`; `transfer/types.py` docstring; `transfer/artifact.py` `_KINDS` and shape checks;
`fit.plan()` config keys; `privacy.CAVEATS`). They are emitted **only** when the config gives the kind a positive
share, and they are appended to the workload **after** every existing kind, so the noise draws of the v1 kinds are
unchanged and the v1 config still produces byte-identical artifacts (tested).

### 4.1 `attr_neigh_simpson` — neighbourhood concentration per degree bin

One instance per configured group: name `attr_neigh_simpson:<a>` or `attr_neigh_simpson:<a>+<b>`,
`params = {"attrs": [...], "cap": Dp, "bin_edges": [...], "min_degree": dmin, "counts_from_hist": true}`.
`Dp` is required (an int; the attribute sensitivity is proportional to it). `dmin >= 1` must be one of the bin edges.
The group's domain product is capped at `MAX_GROUPS` (64), like `attr_edge_pair`.

**Statistic.** On the graph capped at `Dp` (`capping.cap_edges`, same stable cap as `attr_edge_pair`): for node `u`
with capped degree `d_u`, group code `g(u)` (row-major product of the group's attribute codes),
`n_{u,g}` = number of capped neighbours of `u` with code `g`, and

    s_u = [d_u >= dmin] * sum_g (n_{u,g} / d_u)^2        (in [0, 1]; 0 when d_u < dmin)

The `(B, 2)` array has column 0 = `sum_{u in bin b} s_u` (bin by capped degree, last bin open-ended, as
`clustering_by_degree`) and column 1 = node count per bin. **Only column 0 is released** (`counts_from_hist`);
`postprocess` fills column 1 from the processed `degree_hist` (free post-processing of an already paid release; the
histogram counts uncapped degrees, so bins above the cap are approximate — a documented simplification, negligible at
`Dp = 150` where a handful of hubs sit above the cap). Post-processing of column 0: clip to `[0, count]`; bins
entirely below `dmin` carry no column-0 information and are left at 0 (the generator ignores them).

**Sensitivity** (to be written as a derivation in the `measure.py` docstring and checked by brute force):

- *Edge adjacency, capped at `Dp`*: the capped graphs differ in at most 3 edges (the `capping` argument used by every
  capped kind); each step adds one edge `{u, v}` to a graph whose max degree stays `<= Dp`. Only `s_u` and `s_v`
  change: `s_w` for `w != u, v` depends on `N(w)` and on the attributes of `N(w)`, both unchanged. Endpoint `u`: if
  its bin is unchanged, one coordinate moves by `s'_u - s_u` in `[-1, 1]`, norm `<= 1`; if its bin changes (or it
  crosses `dmin`), the change is `-s_u e_b + s'_u e_b'`, norm `<= sqrt(2)`. Two endpoints, triangle inequality:
  `s1 = 2 sqrt(2)`. **edge = 3 s1 = 6 sqrt(2)**. No common-neighbour term and no `Dp` factor (contrast
  `clustering_by_degree`).
- *Attribute-row adjacency* (capped graph unchanged): node `u` changes code `g -> g'`. `s_u` is unchanged (it
  depends on the neighbours' codes, not on `u`'s). For every neighbour `w` of `u` with `d_w >= dmin`:
  `delta s_w = [(n_{w,g'} + 1)^2 - n_{w,g'}^2 + (n_{w,g} - 1)^2 - n_{w,g}^2] / d_w^2 = 2 (n_{w,g'} - n_{w,g} + 1) / d_w^2`.
  Since `1 <= n_{w,g}` (u itself), `0 <= n_{w,g'}`, and `n_{w,g} + n_{w,g'} <= d_w`: `|delta s_w| <= 2 / d_w <= 2 / dmin`.
  The change vector is `sum_{w in N(u)} delta s_w e_{b(w)}`; its L2 norm is at most its L1 norm, at most
  `|N(u)| * 2 / dmin <= 2 Dp / dmin`. **attr = 2 Dp / dmin**. (With `Dp = 150`, `dmin = 16`: 18.75 — against
  `sqrt(2) Dp = 212` for `attr_edge_pair`.)
- A group of several attributes: one row change changes the node's code once; the same bounds hold.

Noise: Gaussian on column 0, the existing mechanism; zCDP on both ledgers via the existing `calibrate`.

### 4.2 `spectrum` — top-k adjacency eigenvalues

One instance, name `spectrum`, `params = {"k": K, "cap": None, "matrix": "adjacency"}`, values `(K,)` = the `K`
algebraically largest eigenvalues of the (uncapped) adjacency matrix, descending (`scipy.sparse.linalg.eigsh`;
`K` public, default 32, at most `n - 1`).

**Sensitivity.** For real symmetric `A` and `A' = A + E`, Mirsky's inequality (the Hoffman-Wielandt theorem for
Hermitian matrices with eigenvalues in sorted order) gives `sum_i (lambda_i(A) - lambda_i(A'))^2 <= ||E||_F^2`. One
edge toggle has `||E||_F^2 = 2`, and the top-`K` sub-vector is a sub-sum. **edge = sqrt(2)**, uncapped,
independent of `K`, `n` and the degrees; **attr = 0**. Post-processing: sort descending (free).

Consumers in this round: package A compares released and synthetic spectra (the eval's existing `eigenvalues` /
`eigenvalue_rel` metrics work on the graphs; the released table lets the generator know the number and strength of
mesoscale communities without seeing the graph); package E, if triggered, calibrates its tilt from it. Until E, the
generator does not consume it — it is released and ignored (allowed by `snap/README.md`, "Add features inside"),
and its share is kept small (0.03) so it costs the other measurements almost nothing.

### 4.3 Config `snap/inside/config/graph_aim_v2.json`

A copy of the shipped v1 config plus (initial values, tuned by the sweep in section 8):

```
"shares": { ...v1 shares unchanged..., "attr_neigh_simpson": 0.40, "spectrum": 0.03 },
"caps": { ...v1 caps..., "attr_neigh_simpson": 150 },
"neigh_simpson_groups": [["school"], ["hometown"], ["location"], ["hometown", "location"]],
"neigh_simpson_bin_edges": [0, 16, 35, 60, 110, 1201],
"neigh_simpson_min_degree": 16,
"spectrum_k": 32
```

`neigh_simpson_groups` null = the `homophily_groups`. The fourth group is the joint of the two strongest homophily
attributes: it releases coherence of a region in *both* at once (gap doc requirement 2). Each group's product is
bounded by `MAX_GROUPS`. All keys are public settings chosen by looking at the real graph: added to the caveats.

**Shipped (2026-11-03, after the stage-1 budget sweep `tools/v2_runs/sweep_budget.py` / `.csv`): cap 60, bins
`[0, 16, 60, 1201]`, shares as above.** At the initial values each instance got sigma 109 at eps = 4, i.e. a per-bin
mean error of 0.11-0.26 (RMSE of released vs exact means over the informative bins, 3 noise seeds: 0.20 mean / 0.28
max), which is the size of the real-vs-i.i.d. signal itself. The cap is the lever (attribute sensitivity
`2 Dp / dmin`): cap 100 gives 0.12, cap 60 gives 0.09, cap 40 gives 0.07; three bins instead of five halve it again
(cap 60 + 3 bins: 0.037 mean / 0.058 max, sigma 54). Share 0.55 taken from `attr_edge_pair` improves it by only 0.005
while pushing that table's sigma from 994 to 1227, so the shares stay. Dropping the joint group saves little (0.096 vs
0.089 at cap 60). The price of the cap is that a hub's neighbourhood is measured on a 60-neighbour subsample: the
Simpson index of a random subsample is biased upward by at most `1 / cap` = 0.017, below the noise.

### 4.4 Contract and inside changes (checklist of `snap/README.md`, "A new statistic")

- `transfer/types.py`: document both kinds in the `Measurement` docstring (shapes, bin meaning, column-1 convention,
  descending eigenvalues). `transfer/artifact.py`: `_KINDS` += both; shape checks: `(B, 2)` with
  `bin_edges` ascending from 0, `min_degree` an edge, `cap` an int, group product `<= MAX_GROUPS`; `(K,)` with
  `k == len(values)` and `cap is None`. `FORMAT_VERSION` stays 1 (old artifacts load unchanged). `inspect.py`:
  verify it shows both (it enumerates generically).
- `inside/measure.py`: `KINDS`, `workload()` (new keyword arguments, instances appended last), `exact_values()`,
  `sensitivity()`, `released_vector()` / `add_noise()` (column 0 only, as the `counts_from_hist` clustering case),
  `postprocess()`.
- `inside/fit.py`: `plan()` reads the new keys; kinds emitted iff their share is positive.
- `inside/privacy.py`: caveat sentence for the new public settings (groups, bins, `dmin`, `Dp`, `K`).
- No new provenance key. `METHOD` stays `graph_aim_v1` (it names the artifact format, which is unchanged).

## 5. Package C: generator v2, the label chain (must)

Files: new `src/graph_aim/outside/refine.py` (the chain engine) and `src/graph_aim/outside/generate_v2.py`
(`generate_v2(artifact, n=None, seed=0, **chain_options) -> AttributedGraph`, plus `generate_v2_with_info` returning
the residual report). `generate.py` is **not modified**. Imports: numpy/pandas/scipy, `graph_aim.transfer`,
`graph_aim.outside` only (boundary test). `snap/outside/generate_synthetic.py` gets `--method {v1,v2}` (default
`v1`, behaviour unchanged); the index entries record the method.

`generate_v2` = `generate(artifact, n, seed)` (v1.1, untouched: attributes, degrees, structure, rewiring) followed by
`refine_labels(graph, artifact, rng)`. If the artifact has no `attr_neigh_simpson` measurement, `refine_labels` is a
no-op and the output is byte-identical to `generate` (tested with the fallback digests).

### 5.1 State

The edge set is fixed (CSR adjacency), degrees `d`, the attribute DataFrame (int codes). For every released
instance `k` of `attr_edge_pair` and `attr_neigh_simpson`: the group code per node `g_k(u)` and the per-node
neighbour-code counts `N_k[u, g]` (dense `(n, G_k)` int array, `G_k <= 64`), maintained incrementally. Strata for
the moves: the `attr_degree` degree bin of each node (from the released tables' `bin_edges`); one stratum when the
artifact has no `attr_degree`. Degrees used for the Simpson bins are the synthetic (uncapped) degrees; the released
statistic was capped at `Dp` — a documented simplification (few nodes above the cap).

### 5.2 Energy

`E = sum_k term_k`, each term already in noise units (so no weights to tune; a `weights` option exists for
diagnostics):

- `attr_edge_pair` instance `k`: `C_k` = ordered class-pair counts on the synthetic edges (as `_class_counts`),
  `T_k` = released ordered mass scaled to the synthetic ordered edge count. `term = sum_ij (C_ij - T_ij)^2 / (sigma_k^2 + T_ij + 1)`.
- `attr_neigh_simpson` instance `k`: for every bin `b` with lower edge `>= dmin`: `S_b` = `sum s_u` over synthetic
  nodes in bin `b`; target `S*_b = t_b * n_b` where `t_b` = released column 0 / column 1 (the released per-bin mean)
  and `n_b` = synthetic node count of the bin. `term = sum_b (S_b - S*_b)^2 / (sigma_k^2 + 1)`.
- `attr_2way` table `(a, b)`: `C` = synthetic row counts, `T` = released table scaled to `n`.
  `term = sum (C - T)^2 / (sigma^2 + T + 1)`. Only tables touching an attribute that moves need a term.

Stopping: a term is *satisfied* when it is `<=` its number of cells (every cell within about one sigma); the chain
stops when all terms are satisfied, or when a round accepts no move, or at `max_rounds` (default 200). Non-private
artifacts (`sigma = 0`) are handled by the `+ 1`.

### 5.3 Moves and one round

Attributes that move: those appearing in any `attr_neigh_simpson` or `attr_edge_pair` instance; every other
attribute keeps the values the Chow-Liu sampler gave it. A round:

1. Propose `B` (default `n // 2`) pairs `(u, v)` with the same stratum and `g(u) != g(v)` for a uniformly chosen
   moving attribute `a`; drop pairs that are adjacent (`u ~ v`) or share a node within the batch.
2. Change statistics per proposal, vectorized over the batch from the CSR structure and the `N_k` arrays:
   - edge-pair: for every neighbour `w` of `u`, cell `(g_u, g_w) -> (g'_u, g_w)` (and mirrored), likewise for `v`;
   - Simpson: for every neighbour `w` of `u` with `d_w >= dmin`: `delta s_w = 2 (N_k[w, g'] - N_k[w, g] + 1) / d_w^2`
     (closed form, section 4.1), summed per bin; likewise for `v` with `g, g'` exchanged;
   - 2-way: rows `u` and `v` move between four cells.
   `Delta E` is the linearised change of every term (the same linearisation `_rewire` uses).
3. Accept proposals with `Delta E < 0`; damp a batch to at most 25% of the proposals (random subset), apply them,
   update `N_k`, recompute every term exactly. If the exact energy rose, halve the damping for the next round
   (the linearisation broke) and restore it slowly afterwards.
4. Optional temperature `T > 0` (Metropolis acceptance) for a later entropy study; default 0.

All randomness from `np.random.default_rng([seed, 2])`, order fixed, so the output is deterministic per seed.
Cost per round is `O(B * mean degree * instances)` vectorized; 200 rounds on the SNAP graph should take seconds to
a minute. The info dict reports per-term energy before/after, satisfied flags, rounds, acceptance per round.

### 5.4 Tests (`tests/graph_aim/outside/test_refine.py`)

On a planted graph (communities with concentrated attributes, built in the test; an exact artifact from
`graph_aim.inside.fit` with the v2 keys, tests may import inside): the Simpson per-bin sums move from the i.i.d.
level to within tolerance of the targets; `attr_1way` and `attr_degree` tables of the output are exactly those of
the input; the 2-way tables stay within noise; the edge array is identical to `generate`'s; deterministic per seed;
artifact without the kinds gives byte-identical output; a hand-built artifact with inconsistent targets terminates
and reports the residual; `B = 1`, a single stratum, an attribute absent from every instance, `n != schema.n`.

## 6. Package D: structure chain (conditional)

Fold `_rewire` and the `_calibrated_structure` rho loop into one structure chain with the same engine: moves are
degree-preserving double-edge swaps restricted to phase-two edges (as now), terms are `joint_degree`,
`attr_edge_pair` and `clustering_by_degree` (change statistic: common-neighbour counts of the four touched
pairs, a sparse row intersection per proposal). Trigger: package C's discriminator run shows the structure gaps
(degree-group importance, paths, mixing) have become binding, or package A's decision rule fires and E needs the
within-region density. Gate: every D14 structure metric within the v1.1 seed spread. Not built before C is measured.

## 7. Package E: latent regions in the structure initializer (must, after C)

Triggered by package A's outcome (section 3). Structure only: attributes stay with the label chain (package C), which
will paint the regions E creates. Everything is a **generator-side option with a default that leaves the v1 path
byte-identical** (the fallback digests of `test_generate_attr_degree.py` and the v1 pin test stay the guard).

### 7.1 Regions

After degrees are assigned and before structure: a latent region label `Z(u)` per node, a random partition into `R`
regions of near-equal size (a random permutation cut into `R` parts; heterogeneous sizes are a later option).
`Z` is independent of attributes and degrees. `R = 1` is the identity (no regions).

### 7.2 Phase one: blocks that never cross a region

Today `_phase_one` sorts nodes by (degree, group tie-break) and cuts consecutive blocks of `d + 1` nodes, `d` = degree
of the block's first node; since the sort is ascending every member has degree `>= d`, which is what keeps the degree
sequence right. D18 broke this by sorting on (group, degree): a block that starts near the top of one group spilled
into the low-degree start of the next group, giving nodes of degree 2-5 about `0.84 * d` edges of a block sized for
`d = 100`. The fix is to keep the invariant, not the global order: sort by (`Z`, degree, group tie-break) and **close
a block at the region boundary** (`size = min(d + 1, nodes left in the region)`), so every member still has degree
`>= d`; the last block of a region is simply smaller and its members get the remainder in phase two. Members may now
have degrees further above `d` than in the global sort (a region holds fewer nodes per degree level); the
`_calibrated_structure` rho loop per bin absorbs the clustering effect, and the gate below checks it. A hub whose
`d + 1` exceeds its region's size gets the whole region as its block — the ego-network shape the real data has.
Phase one carries most edges on this dataset (density `rho = c^(1/3)` about 0.84 at clustering 0.6), which is why the
region structure has to enter here and not only in phase two.

### 7.3 Phase two: a within-region mixture

The phase-two excess degree is wired as a mixture: a fraction `q` of the edges is drawn **within regions** (region
chosen proportional to its excess-degree volume, both endpoints inside it proportional to excess degree), the rest by
the existing class-pair draw (`joint_degree` x `attr_edge_pair` seed, IPF margins) unchanged. Both parts draw
endpoints proportional to excess degree, so expected degrees are preserved (the degree-corrected SBM argument); the
duplicate / top-up / repair steps and the final `_rewire` (which pulls the class-pair counts back toward their
targets over all phase-two edges) are unchanged. `q = 0` is the identity.

### 7.4 Calibration of `(R, q)` from the released `spectrum` (outside, free)

`generate_v2` chooses `(R, q)` by simulation against the released top-`K` eigenvalues: for each candidate on a small
grid (`R` in {1, 8, 16, 32, 64} x `q` in {0, 0.5, 0.75, 0.9}, coarse-to-fine in two steps), generate the structure
once (seed fixed), compute its top-`K` adjacency eigenvalues (`scipy.sparse.linalg.eigsh`) and score
`sum_i (lambda_i^syn - lambda_i^released)^2 / (sigma^2 + 1)`; take the best. A dataset without mesoscale structure
has no outlier eigenvalues, the fit lands on `R = 1` or `q = 0`, and the generator is the v1 one — the generality
requirement. Cost: a structure generation per candidate (seconds each on SNAP), once per artifact; the chosen pair and
the score go into the info dict. If the artifact has no `spectrum`, `(R, q)` fall back to `(1, 0)` unless passed
explicitly.

**Noise margin (added 2026-11-03 after the first full run, D24):** a calibrated `(R, q)` is used only if it beats the
identity by at least `MIN_MARGIN = 3` noise standard deviations of the released spectrum
(`info["margin_vs_identity"]`); otherwise the identity is used and the rejected candidate is reported. Without it,
at eps <= 4 (spectrum sigma about 12 against candidate differences of a few units per eigenvalue) the calibration
picked regions by noise and the structure got worse than v1.1 (degeneracy 49 vs 26 at eps = 1, triangles 0.46 vs
0.31). The rule is the same generality argument one level up: regions need evidence in the artifact.

### 7.5 Tests and gate

`tests/graph_aim/outside/test_regions.py`: with `R = 1, q = 0` the output is byte-identical to `generate` (digests);
with regions, the degree sequence is exactly the pool (phase one never gives a node more than its degree, blocks never
cross regions — assert on the block membership), modularity of the planted `Z` on the output rises with `q`, the
cross-region edge fraction falls with `q`; the calibration picks `R = 1` / `q = 0` on an artifact whose spectrum came
from an Erdos-Renyi-like graph and picks `R > 1` on one from a planted-partition graph; determinism per seed.
Gate on the real data (section 10 criterion 3): degree KS, triangles, clustering, transitivity, assortativity within the
v1.1 seed spread; path length, degeneracy and the Louvain part-size / conductance diagnostic expected to move toward
the real values. Then package C's chain runs on top and the discriminator decides.

## 8. Privacy and caveats

- B adds two kinds with sensitivities `(6 sqrt(2), 2 Dp / dmin)` and `(sqrt(2), 0)`, both proved above and added to
  the brute-force tests. Ledger entries come from the existing `build_ledger`; `calibrate` is unchanged.
- New caveat text: the Simpson groups, bins, `dmin` and `Dp`, and the spectrum `K`, were chosen by looking at the
  real graph. The *choice to measure composition at all* was made after an eval-zone classifier compared synthetic
  graphs with the real one (D19): a data-dependent selection step that is not paid for in the ledger, listed as such.
- Budget policy: the new attribute-side budget comes out of `attr_edge_pair`'s share by a sweep (section 9 step 5):
  v1.1 showed that channel's precision is not what binds. `epsilon_attr` stays equal to `epsilon_edge`.
- Boundary: the chain is `outside/`; `composition.py` is `eval/`; `tests/graph_aim/test_boundary.py` must pass with
  no new imports in `transfer` / `outside` beyond numpy/pandas/scipy.

## 9. Tests

| Test | Where | What it proves |
|---|---|---|
| composition metrics on tiny graphs | `tests/graph_aim/eval/test_composition.py` | hand-computed Simpson / two-hop / dominant share; a perfectly concentrated graph gives 1; permuted null near the independence level; invariance to attribute names; `min_degree` exclusion |
| `attr_neigh_simpson` sensitivity, brute force | `tests/graph_aim/inside/test_measure.py` | every single-edge change on graphs within the cap stays within `2 sqrt(2)` per step; every single-row change within `2 Dp / dmin`; the edge bound is approached; varied `K`, groups of two attributes, bin crossings, `dmin` |
| `spectrum` sensitivity, brute force | same | every single-edge toggle moves the top-`K` vector by `<= sqrt(2)` in L2 |
| v1 config unchanged | `tests/graph_aim/inside/test_fit.py` | fitting the v1 config (noise on, fixed seed) gives values byte-identical to a pinned digest taken before this round |
| artifact validation | `tests/graph_aim/transfer/test_artifact.py` | round trip of both kinds; wrong shape / `dmin` not an edge / missing cap / `k` mismatch rejected; old artifacts still load |
| label chain | `tests/graph_aim/outside/test_refine.py` | section 5.4 |
| v2 no-op fallback | same | artifacts without the new kinds: `generate_v2 == generate` byte for byte, including the existing fallback digests |
| boundary, generality | existing | unchanged; must pass |

Run: `PYTHONPATH=src:snap/inside:snap/eval python3 -m pytest tests/graph_aim -q` (this machine: `/opt/venv/bin/python3`, no `uv`).

## 10. Acceptance criteria

The regression test for the gap is `snap/eval/discriminate.py` on the v2 graphs (`--synthetic snap/outside/synthetic_v2
--out snap/eval/results_v2`), read in the order of `attribute-composition-gap.md` section 6:

1. **Composition matched**: at eps = inf, every released Simpson per-bin mean is matched within 0.02 absolute;
   at eps = 4, gap closed `>= 0.5` on `cmp_*_gapdeg_simpson_mean` (the degree-stratified null is the floor, see the
   section 3 outcome; v1.1 sits at 0.11-0.28 there), 3 seeds.
2. **(a)** the attributes feature-group importance falls toward the degree group's level; **(b)** logistic-regression
   AUC below 0.9; **(c)** gradient-boosting AUC below the structure baselines (dc_sbm 0.9999, bter 0.9993).
   A move on (a)-(b) is the win condition for this round; (c) is the stretch.
3. **No structure regression**: the D14 metrics (degree KS, triangles, clustering, transitivity, assortativity,
   path length, degeneracy) within the v1.1 seed spread — trivially true for C (edges untouched), the gate for D/E.
4. `attr_1way` / `attr_degree` exactly preserved; 2-way TVD within the v1.1 spread.
5. Full suite green, boundary and generality tests included; v1 digests unchanged.

## 11. Order of work and orchestration

Branch `aim-v2` (worktree `.claude/worktrees/aim-v2`, cut from `aim`; `aim` is the track branch, nothing goes to
`main`). Packages run as subagents in their own worktrees **branched from `aim-v2`** (never from `main`: see the
2026-11-03 note in the orchestrator's memory — a worktree based on `main` carries trees `aim` lacks), one package per
agent, merged into `aim-v2` by the orchestrator after a re-test, then `aim-v2` into `aim`.

1. **A** (eval + tools) and **B** (inside + transfer) and **C** (outside) in parallel; C codes against the contract
   in sections 4.1 and 5 with hand-built `Measurement` objects until B lands.
2. Orchestrator: merge, run the full suite, fit the v2 config (3 seeds, 7 variants) into `artifacts_v2`, generate
   with `--method v2` into `synthetic_v2`, score into `results_v2`, run the discriminator.
3. Read A's decision note and C's residuals; decide D / E. (Taken 2026-11-03: E is required, section 3 outcome.)
3b. **E** (outside, after C merges: it shares `generate_v2.py`): regions + calibration; re-run stages 2-3 and the
   discriminator; the structure gate of section 7.5.
4. Sweep the `attr_neigh_simpson` share (0.25 / 0.40 / 0.55, taken from `attr_edge_pair`) at eps = 4, 3 seeds;
   ship the best by criterion 1 without breaking criterion 4.
5. Docs: `PROCESS.md` (decisions D20+ and a "Composition round (v2)" section with before / after tables),
   `CHECKPOINTS.md`, `snap/README.md` (the `--method v2` flag and the new kinds), `src/graph_aim/README.md`,
   `snap/reports/graph_aim_v1.md` (a v2 section), `docs/realfake/snap.md` (v2 discriminator results).

## 12. Risks and open questions

- **Regions capped by structure** (section 2). Package A's decision rule decides whether E is needed now.
- **Entropy loss**: a zero-temperature chain matched exactly would be a low-entropy graph; the within-noise stopping
  rule and the i.i.d. initialization bound how far it moves. A later entropy study can use the temperature option.
- **Conflicting tables**: noisy tables may be mutually inconsistent; the chain finds a compromise and reports which
  term lost. Non-convergence within `max_rounds` is informative (the structure cannot express the target), not a bug.
- **Cap mismatch**: the released Simpson is on the capped graph, the synthetic one uncapped; negligible at `Dp = 150`
  on this dataset, to be re-checked on a dataset with many hubs above the cap (move `cap_edges` to `transfer` then).
- **Single-attribute swaps perturb the 2-way joints**; the 2-way energy term keeps them within noise. If that term
  dominates the residuals, the fallback is whole-row swaps (every node-level joint exact, less freedom per attribute).
- **Budget**: four Simpson instances share one kind's budget; if eps = 4 targets are noise-dominated, drop the joint
  group first, then the weakest single attribute.

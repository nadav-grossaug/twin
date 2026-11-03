# v2 diagnostic (package A): are the synthetic communities strong enough to paint?

Date 2026-11-03. Spec: `docs/snap/composition-plan.md` section 3. Script: `tools/v2_diag/community_strength.py`
(eval zone; `PYTHONPATH=src:snap/inside:snap/eval python3 tools/v2_diag/community_strength.py`, 20 s), numbers in
`tools/v2_diag/community_strength.csv` (long form: graph, section, metric, value). Graphs: the real graph, the v1.1
synthetic graphs `snap/outside/synthetic_v1_1/graph_aim_{eps4,epsinf}_seed0`, and derived ones: `real_permuted`
(real structure, whole attribute rows permuted: the no-composition floor), `<syn>_painted` (synthetic structure,
the synthetic rows re-dealt over the structure's own Louvain communities: the painted upper bound of plan section
3) and `real_painted` (the same re-dealing on the real structure, as a check of the painting itself).

**Verdict: the decision rule fails.** The painted synthetic graph reaches 85% of the real per-walk share variance
for walks of 10-39 nodes, but only 37-41% at 40-119 nodes and 19-21% at 120-200 nodes (threshold 70% in every
bucket). The synthetic structure has no regions of the size the real graph has, so no label assignment on it can
reproduce the real graph's composition beyond the 1-hop scale. **Package E (latent tilt in the initializer) is
needed, scheduled right after C**; package C alone will fix the neighbourhood-level statistics (and it should
still be built: the painted graph shows the 1-hop targets are reachable on the current structure) but not the
patch-level ones the discriminator reads.

## 1. Structure: community strength

Louvain (`networkx.community.louvain_communities(seed=0)`) and semi-synchronous label propagation; conductance of
a part = cut / min(vol, total - vol); top-32 adjacency eigenvalues (algebraically largest, `eigsh`).

| | real | syn eps 4 | syn eps inf |
|---|---|---|---|
| Louvain modularity | **0.835** | 0.710 | 0.711 |
| Louvain parts | 16 | 96 | 152 |
| part size: median / q75 / max | **231 / 425 / 548** | 42 / 59 / 268 | 7 / 36 / 277 |
| share of nodes in the 5 largest parts | 0.59 | 0.21 | 0.22 |
| conductance of the parts, node-weighted mean | **0.038** | 0.202 | 0.196 |
| label propagation: modularity / parts / median size | 0.737 / 44 / 9 | 0.708 / 232 / 9 | 0.686 / 240 / 9 |
| label propagation: share of nodes in parts >= 20 | 0.945 | 0.647 | 0.637 |
| eigenvalues 1-3 | 162, 125, 106 | 149, 126, 112 | 153, 120, 92 |
| eigenvalues 4-6 | 73, 65, 65 | 84, 77, 69 | 81, 72, 66 |
| eigenvalue 10 / 20 / 32 | 43 / 30 / 23 | 53 / 35 / 25 | 51 / 33 / 25 |
| bulk scale `sqrt(<d^2>/<d>)` | 10.3 | 10.5 | 10.0 |

Reading. The real graph is 10 ego networks and Louvain finds them: 16 parts of median size 231, with a
node-weighted conductance of 0.04 (one edge in 25 leaves the part). The synthetic graphs have a similar
modularity *score* (0.71) but reach it with 6-10 times more parts of a fifth the size and 5 times the conductance:
their best partition is a mosaic of small dense patches, not a handful of large regions. The spectrum says the
same thing from the other side: the three leading eigenvalues (the three dominant ego networks) are reproduced,
but from the 4th on the synthetic eigenvalues are *larger* than the real ones (84 vs 73 at rank 4, 53 vs 43 at
rank 10, 25 vs 23 at rank 32) — many medium-strength blobs instead of a few strong regions plus a bulk. The
synthetic bulk edge is the same (`sqrt(<d^2>/<d>)` about 10, so about 20 for the edge), so everything above 20 is
mesoscale structure in both; the synthetic has more of it, spread thinner. This is the D14 picture (paths too
short, mixing too diagonal) seen through the partition.

## 2. The painted upper bound

Painting keeps the synthetic edge set and re-deals the synthetic attribute rows (whole rows, so every 1-way and
2-way table and the `attr_degree` tables are untouched): within each `attr_degree` stratum (degree bins
`[0, 1, 11, 113, 1201]`) the rows are sorted by the group code of hometown (then location, then school) and dealt
to the nodes community by community, largest community first. It is greedy and more concentrated than the real
graph where the structure allows it: on the *real* structure it overshoots the real Simpson means (hometown 0.70
vs 0.59) and the real per-walk variance (1.33x), so painting itself is not the limiting factor below.

### 2.1 Neighbourhood Simpson (`simpson_by_degree`, bins `[0, 16, 35, 60, 110, 1201]`, `min_degree` 16)

Node-weighted mean over the 2,644 real nodes of degree >= 16 (per graph, its own nodes); per-bin values in the CSV.

| group | real | real permuted | real painted | syn eps 4 | syn eps 4 painted | syn eps inf | syn eps inf painted |
|---|---|---|---|---|---|---|---|
| school | **0.551** | 0.283 | 0.502 | 0.349 | 0.512 | 0.370 | 0.513 |
| hometown | **0.585** | 0.582 | 0.701 | 0.482 | 0.683 | 0.517 | 0.722 |
| location | **0.429** | 0.416 | 0.542 | 0.380 | 0.593 | 0.391 | 0.589 |
| hometown+location | **0.294** | 0.299 | 0.471 | 0.237 | 0.537 | 0.255 | 0.529 |

Two-hop same-code rate (share of 2-paths with equal endpoint codes), real / syn eps inf / syn eps inf painted:
school 0.60 / 0.36 / 0.36, hometown 0.50 / 0.45 / 0.56, location 0.38 / 0.36 / 0.40, hometown+location 0.23 /
0.21 / 0.32. Independence prediction (`sum_g a_g^2`) for the real graph: 0.29 / 0.43 / 0.32 / 0.19.

Reading. (a) At the 1-hop scale the painted synthetic graph reaches or exceeds the real means for hometown,
location and the joint group, and gets school to 0.51 of 0.55: the 1-hop targets that package B releases are
reachable on the current structure, so package C has something to converge to. (b) Per bin it overshoots in the
low-degree bins (hometown bin 16-35: 0.72-0.77 vs 0.66 real) and stays short in the hub bin (degree >= 110:
0.45-0.55 vs 0.47 real for hometown, 0.31-0.35 vs 0.67 for school): hubs in the synthetic graph span communities,
in the real graph the hubs *are* the ego centres and sit inside one. (c) School is special: its real Simpson is
twice the permuted null, the largest excess of the four, and painting with hometown as the primary key only half
restores it (0.51); the real regions are coherent in school *and* hometown at once (gap doc requirement 2), the
greedy order can serve only one. (d) The column-permuted null (`coupling.permuted_null`, the gap-closed floor of
the spec) is not below the real value for hometown (0.582 vs 0.585) and location (0.416 vs 0.429): `none` is 74%
of hometown rows but a smaller share of edge ends (degree x attribute coupling), and permuting columns moves the
i.i.d. level up. The `cmp_*_gap_*` columns are therefore meaningless for those groups; `score.py` also reports
`cmp_*_gapdeg_*` against a degree-stratified permutation (`composition.permuted_null_by_degree`, rows permuted
within the `attr_degree` bins), which keeps the degree coupling and is the right floor for an i.i.d.-per-stratum
label assignment. That floor is school 0.30, hometown 0.51, location 0.38, hometown+location 0.20 (seed 0; seed 1
within 0.01), below the real values for every group; against it the v1.1 eps-inf graph closes 0.28 of the school
gap, 0.11 of hometown, 0.28 of location — it sits essentially at the i.i.d.-per-stratum level, which is what
section 1 of the plan says. Acceptance criterion 1 of the plan should read the `gapdeg` columns.

### 2.2 Per-walk value-share variance (the discriminator's statistic)

2,000 walks per graph from `graph_aim.eval.discriminator.WalkSampler()` (k log-uniform in [10, 200], seed 0; seed
1 in the CSV agrees within 0.03 on every ratio). Per walk: the share of each value of hometown, location and
school among the k visited nodes. Statistic: variance of the share across walks, mean over the values of an
attribute, then mean over the three attributes. Values x 1000; the per-attribute rows are in the CSV.

| walk size k | real | real permuted (null) | real painted | syn eps 4 | syn eps 4 painted | syn eps inf | syn eps inf painted |
|---|---|---|---|---|---|---|---|
| 10-39 | **24.8** | 4.9 | 33.0 | 8.6 | 21.0 | 9.8 | 21.2 |
| 40-119 | **17.5** | 1.5 | 23.4 | 2.3 | 7.1 | 2.5 | 6.5 |
| 120-200 | **13.7** | 0.6 | 18.1 | 0.9 | 2.9 | 1.1 | 2.6 |
| all | 20.1 | 2.8 | 26.7 | 5.0 | 12.7 | 5.6 | 12.7 |

Ratio to the real value (the decision rule; threshold 0.70 in every bucket) and, in brackets, the excess over the
null, `(x - null) / (real - null)`:

| walk size k | syn eps 4 | syn eps 4 painted | syn eps inf | syn eps inf painted | real painted |
|---|---|---|---|---|---|
| 10-39 | 0.35 [0.19] | **0.85** [0.81] | 0.39 [0.25] | **0.86** [0.82] | 1.33 |
| 40-119 | 0.13 [0.05] | **0.41** [0.35] | 0.14 [0.07] | **0.37** [0.31] | 1.33 |
| 120-200 | 0.07 [0.03] | **0.21** [0.17] | 0.08 [0.04] | **0.19** [0.16] | 1.32 |

Reading. In the real graph the share variance falls slowly with the walk size (24.8 -> 13.7 over 10 -> 160
nodes): a 150-node patch is still "about" one hometown. With i.i.d. rows it falls like 1/k (the null: 4.9 ->
0.6), and the synthetic graphs as they are sit at the null. Painting lifts the synthetic graph to the real level
for small walks (which stay inside one of its small dense parts) and then loses it exactly where the parts end:
at 40-119 nodes a walk has crossed several synthetic parts of median size 7-42 and the shares average out. The
painted *real* structure keeps 1.3x the real variance at every size because its parts are 231 nodes in the median.
The scale caveat of plan section 2 binds: the label chain can only produce regions as coherent as the synthetic
structure's communities, and these are a fifth of the size they need to be.

## 3. Decision

- **Rule**: painted synthetic >= 70% of the real per-walk share variance for 10-200 node walks. **Not met**: 0.85 /
  0.41 / 0.21 (eps 4) and 0.86 / 0.37 / 0.19 (eps inf) for 10-39 / 40-119 / 120-200 nodes.
- **Implication**: package C (label chain) alone will not move the discriminator past the small-walk regime; it
  matches the 1-hop Simpson tables, which the painted graph shows are reachable, and it is the mechanism that will
  calibrate *any* structure, so it is still built first. **Package E** (latent region label `Z` before structure,
  tilted phase-two draw, calibrated against the released `spectrum`) is scheduled right after C, with the target
  of a Louvain part-size median in the hundreds and a node-weighted conductance near 0.05 rather than 0.2; the
  released spectrum shows what to aim for (three strong outliers, then a faster decay than the synthetic graphs
  have today). Package D (structure chain on `joint_degree` / clustering) does not address the part size and
  stays conditional.
- Two smaller findings for the orchestrator: (i) read acceptance criterion 1 on the `cmp_*_gapdeg_*` columns, not
  `cmp_*_gap_*` (section 2.1 d); (ii) school needs its own key in whatever assigns labels to regions: a region
  order by hometown alone restores only half of school's concentration (section 2.1 c), which argues for the joint
  group `hometown+location` in the released Simpson tables and for adding `school` to it or releasing the
  `school+hometown` joint if the budget allows.

"""Generator v2: latent regions in the structure initializer, then the label refinement chain.

``generate_v2`` = (package E) a latent region label ``Z`` per node and a calibrated ``(R, q)``, passed to the v1.1
sampler ``graph_aim.outside.generate.generate(..., regions=Z, within_region_frac=q)``, followed by (package C)
``graph_aim.outside.refine.refine_labels``, a Markov chain that swaps attribute values between nodes of the same
degree stratum until the synthetic neighbourhood statistics are within noise of the released tables. Without a
``spectrum`` measurement the regions are off (``R = 1, q = 0``) and without an ``attr_neigh_simpson`` measurement
the chain is skipped; with neither the output is exactly that of ``generate``.

What the chain controls
-----------------------
- ``attr_neigh_simpson`` (one instance per released attribute group): the concentration of every node's
  neighbourhood on the group's values, ``s_u = sum_g (n_{u,g} / d_u)^2``, summed per degree bin. The v1.1
  attribute rows are i.i.d. across nodes, so neighbourhoods all look like the global mixture; the chain
  introduces dependence between the labels of structurally close nodes and calibrates it per bin against the
  released per-bin sum (``values[b, 0]``) scaled by ``n_syn / schema.n``, on the synthetic graph capped as the
  released statistic was.
- ``attr_edge_pair`` (every instance): the ordered class-pair edge counts stay on the released mixing while
  the labels move (the rewiring of v1.1 matched them once; the chain keeps matching them).
- ``attr_2way`` tables touching a moving attribute: the node-level joints the swaps could otherwise drift from.
- Preserved exactly, by construction of the move: the edge set, ``attr_1way`` counts, and the ``attr_degree``
  tables (bin x value counts, with the bins of the released ``attr_degree`` tables as strata; one stratum when
  the artifact has none). Attributes that appear in no ``attr_neigh_simpson`` / ``attr_edge_pair`` instance
  keep the values the v1.1 sampler gave them.

What the regions control (plan section 7)
------------------------------------------
The v1.1 structure is a mosaic of small dense phase-one blocks (``d + 1`` nodes of nearly equal degree ``d``)
glued by a global class-pair Chung-Lu draw: its communities are a fifth of the size the real graph has and five
times as leaky (``docs/snap/v2-diagnostic.md``). Regions give the glue a scale:

- ``Z``: a partition of the nodes into ``R`` regions of near-equal size. It is drawn **per phase-one block**
  (the blocks are a deterministic function of the degrees, the group codes and the seed, so they are known before
  the structure is sampled): the blocks, and every node that is in no block, are dealt in random order from
  ``np.random.default_rng([seed, 3])`` to the currently smallest region. ``Z`` is independent of the attributes;
  it is coupled to the degrees only through block co-membership (a block's members share a region), and every
  region holds blocks of all degree levels.
- Phase one (``generate._phase_one``): blocks never cross a region. With a block-aligned ``Z`` the blocks are
  exactly the v1.1 ones, so the degree invariant of the sampler (every member has degree >= the block's ``d``)
  and the per-bin clustering calibration are untouched.
- Phase two (``generate._generate_structure``): a fraction ``q`` of the excess-degree edges is drawn inside
  regions, the rest by the class-pair draw. A within-region edge draws its class-pair cell and its first
  endpoint exactly as the class-pair draw does (the ``joint_degree`` x ``attr_edge_pair`` steering is kept) and
  its partner from the partner class's degree bin INSIDE the first endpoint's region, proportional to excess
  degree, so expected degrees are preserved; a draw whose expected demand on that (bin, region) pool exceeds
  half the pool's size stays global (the largest hubs, and mid-degree nodes' edges into the sparse hub bin:
  otherwise they would be lost to duplicates and skew the degree mixing). The degree-preserving rewiring that
  pulls the class-pair counts to their targets proposes swaps between edges of the same region (or between
  cross-region edges) and applies only those that keep the number of within-region edges, so the glue survives
  it; the attribute mixing of the glue is left to the rewiring and to the label chain.
- ``(R, q)`` are calibrated against the released ``spectrum`` (the top-``K`` adjacency eigenvalues): every
  candidate on a coarse grid (``R`` in {1, 8, 16, 32, 64} x ``q`` in {0, 0.5, 0.75, 0.9}; ``R = 1`` is the
  identity and is scored once) generates the structure once, its top-``K`` eigenvalues are computed
  (``scipy.sparse.linalg.eigsh``, fixed start vector) and scored ``sum_i (lambda_i^syn - lambda_i^released)^2 /
  (sigma^2 + 1)``; a refinement step tries ``R x {1/2, 1, 2}`` and ``q +- 0.1`` (clipped to ``[0, 0.9]``: at
  ``q = 1`` no class-pair draw is left and on SNAP the degree assortativity drifted beyond the v1.1 seed spread
  although the spectrum score preferred it) around the best, and the best overall is used. Ties go to the smaller ``R``, then the smaller ``q``. The ``q = 0`` candidates are the
  identity up to the random stream (the same blocks in another order), so their spread measures the score's
  simulation noise (about 4% on SNAP): a ``q > 0`` candidate is used only if it beats the best ``q = 0`` score
  by more than that spread, otherwise the identity is used (a dataset without mesoscale structure lands on
  ``R = 1``). Explicit ``regions_R`` / ``within_region_frac`` keyword arguments skip the calibration of that
  parameter.

Known simplifications
---------------------
- Every neighbourhood statistic is computed on the synthetic graph capped exactly as the released one was
  (each instance's ``params["cap"]``, the artifact's ``cap_salt``, ``graph_aim.transfer.capping``): the
  capped degrees give the Simpson bins and ``d_u``, the capped neighbour sets the counts. The Simpson target
  is the released per-bin SUM scaled by ``n_syn / schema.n``; the released node counts (column 1) are not
  used, because they come from the uncapped degree histogram while the sums are binned by capped degree, so
  a per-bin mean built from them is not the statistic's mean under an active cap (with cap 60 on a graph whose
  hubs reach 1000, every hub sits in the last bin with capped degree exactly 60).
- A move swaps a single attribute, so the node-level joints with the other attributes (the ``attr_2way``
  tables) are perturbed; the 2-way energy term holds them within noise rather than exactly. Whole-row swaps
  would keep every joint exact at the price of far less freedom per attribute.
- Per-proposal changes are exact for the edge-pair and 2-way terms and linearised (closed form per neighbour,
  shared neighbours of the two endpoints counted independently) for the Simpson term. Interactions between the
  moves accepted in one batch are handled by a prefix search over the damped batch (at most 25% of the
  proposals, in random order: the prefix with the lowest combined change is applied; exact for the edge-pair
  and 2-way terms, linearised for Simpson) and by halving the damping when the exact energy still rises.
- Only bins whose lower edge is at or above ``min_degree`` carry a target; bins above the cap are empty by
  construction (a bin edge above the cap is pointless).
- Regions have near-equal sizes; the real mesoscale structure may be heterogeneous. ``Z`` is drawn per block,
  not per node (plan section 7.1): on the SNAP artifacts a per-node ``Z`` with blocks re-cut inside regions
  scatters the high-degree nodes over the regions, so the dense hub blocks that carry the leading eigenvalues
  disappear and most of a node's degree moves to phase two; the spectrum score was 8-15 times worse than the
  v1.1 structure for every ``(R, q)``, which is why the block-aligned variant is used.
- The calibration is by simulation on a reduced sampler (``calibration_rounds`` rho rounds, no rewiring by
  default: the rewiring preserves the within-region edge count, so it moves the eigenvalues little and the
  ranking of the candidates is the same as with the full sampler on the SNAP artifacts); the final graph uses the
  full settings. The score is dominated by the ``K`` released values' noise when ``sigma`` is of the order of the
  differences between candidates (on SNAP at ``epsilon = 4``, ``sigma`` about 12 against candidate differences of
  a few units per eigenvalue): ``info["calibration"]["margin_vs_identity"]`` reports how many noise standard
  deviations the chosen candidate beats the identity by, so a noise-driven choice can be recognised.
- The chain's simplifications above are described in more detail in ``graph_aim.outside.refine``.

Randomness: ``default_rng(seed)`` for attributes and degrees, ``[seed, 1]`` for the structure, ``[seed, 2]`` for
the chain, ``[seed, 3]`` for the regions; the output is deterministic per seed.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from graph_aim.outside.generate import (
    _by_kind,
    _calibrated_structure,
    _group_codes,
    _phase_one_blocks,
    _sample_attributes,
    _sample_degrees,
    generate,
)
from graph_aim.outside.refine import refine_labels
from graph_aim.transfer.types import Artifact, AttributedGraph

COARSE_R = (1, 8, 16, 32, 64)
COARSE_Q = (0.0, 0.5, 0.75, 0.9)
Q_MAX = 0.9  # the refinement never removes the class-pair draw entirely (q = 1 costs degree mixing, see docstring)
REWIRE_ROUNDS_WITH_REGIONS = 120
# regions are switched on only if the calibrated candidate beats the identity by this many noise standard
# deviations of the released spectrum (on SNAP: eps = inf gives margins > 10, eps = 4 gives 1.6-3.1, and the
# noise-chosen regions at eps <= 4 damaged triangles, degeneracy and assortativity; see PROCESS.md D24)
MIN_MARGIN = 3.0
_DENSE_N = 400  # below this many nodes the eigenvalues are computed densely (eigsh needs k < n and is slow there)


# ------------------------------------------------------------------------------------------- regions
def region_labels(blocks: list[np.ndarray], n: int, R: int, rng: np.random.Generator) -> np.ndarray:
    """Deal the phase-one blocks (and every node in none of them) in random order to the currently smallest of
    ``R`` regions; returns an ``(n,)`` int label array. ``R = 1`` gives all zeros."""
    R = max(int(R), 1)
    z = np.zeros(n, dtype=np.int64)
    if R == 1:
        return z
    units = [np.asarray(b, dtype=np.int64) for b in blocks if len(b)]
    covered = np.concatenate(units) if units else np.zeros(0, dtype=np.int64)
    rest = np.setdiff1d(np.arange(n), covered)
    units += [np.array([u]) for u in rest]
    sizes = np.array([len(u) for u in units], dtype=np.int64)
    load = np.zeros(R, dtype=np.int64)
    for i in rng.permutation(len(units)):
        r = int(np.argmin(load))  # the smallest region (lowest index on ties)
        z[units[i]] = r
        load[r] += sizes[i]
    return z


def top_eigenvalues(edges: np.ndarray, n: int, k: int) -> np.ndarray:
    """The ``k`` algebraically largest adjacency eigenvalues, descending (deterministic: fixed start vector)."""
    k = int(min(k, max(n - 1, 0)))
    if k <= 0 or n == 0:
        return np.zeros(0)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if n <= _DENSE_N:
        a = np.zeros((n, n))
        if len(edges):
            a[edges[:, 0], edges[:, 1]] = 1.0
            a[edges[:, 1], edges[:, 0]] = 1.0
        vals = np.linalg.eigvalsh(a)
        return np.sort(vals)[::-1][:k]
    if len(edges) == 0:
        return np.zeros(k)
    a = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    vals = eigsh(a, k=k, which="LA", return_eigenvectors=False, v0=np.ones(n) / np.sqrt(n))
    return np.sort(np.asarray(vals, dtype=float))[::-1]


def _released_spectrum(artifact: Artifact) -> tuple[np.ndarray, float] | None:
    ms = _by_kind(artifact, "spectrum")
    if not ms:
        return None
    m = ms[0]
    v = np.nan_to_num(np.asarray(m.values, dtype=float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0:
        return None
    return np.sort(v)[::-1], float(m.sigma or 0.0)


def spectrum_score(lam_syn: np.ndarray, lam_rel: np.ndarray, sigma: float) -> float:
    """``sum_i (lambda_i^syn - lambda_i^released)^2 / (sigma^2 + 1)`` over the common top ranks."""
    k = min(len(lam_syn), len(lam_rel))
    if k == 0:
        return 0.0
    return float(((lam_syn[:k] - lam_rel[:k]) ** 2).sum() / (sigma**2 + 1.0))


def _candidate_grid(r_values, q_values, done: set) -> list[tuple[int, float]]:
    out = []
    for R in r_values:
        for q in q_values:
            key = (int(R), 0.0 if int(R) == 1 else round(float(q), 4))
            if key not in done:
                done.add(key)
                out.append(key)
    return out


def calibrate_regions(
    artifact: Artifact,
    n: int,
    seed: int,
    *,
    regions_R: int | None = None,
    within_region_frac: float | None = None,
    rounds: int = 4,
    rewire_rounds: int = 0,
    min_margin: float = MIN_MARGIN,
) -> tuple[int, float, np.ndarray | None, dict]:
    """Choose ``(R, q)`` against the released spectrum (plan section 7.4); returns ``(R, q, Z, info)``.

    ``Z`` is ``None`` when ``R = 1``. ``regions_R`` / ``within_region_frac`` pin one or both parameters (both
    pinned: no structure is generated here). Without a ``spectrum`` measurement the result is ``(1, 0)`` unless
    pinned. A calibrated ``(R, q)`` is used only if it beats the identity by at least ``min_margin`` noise
    standard deviations of the released spectrum (``info["margin_vs_identity"]``); otherwise the identity is used
    and the rejected candidate is reported under ``info["rejected_below_margin"]``.
    """
    t0 = time.time()
    info: dict = {"grid": [], "rounds": rounds, "rewire_rounds": rewire_rounds, "skipped": None}
    n = int(n)
    rng = np.random.default_rng(seed)
    attrs = _sample_attributes(artifact, n, rng)
    deg = _sample_degrees(artifact, n, rng, attrs)
    grp0, _ = _group_codes(artifact, attrs, n)
    blocks = _phase_one_blocks(deg, grp0, np.random.default_rng([int(seed), 1]))

    def labels(R: int) -> np.ndarray | None:
        R = int(min(max(R, 1), n))
        return region_labels(blocks, n, R, np.random.default_rng([int(seed), 3])) if R > 1 else None

    pinned_R = regions_R is not None
    pinned_q = within_region_frac is not None
    if pinned_R and pinned_q:
        R = int(min(max(int(regions_R), 1), n))
        q = float(within_region_frac) if R > 1 else 0.0
        info.update(skipped="pinned", chosen={"R": R, "q": q, "score": None}, seconds=time.time() - t0)
        return R, q, labels(R), info
    rel = _released_spectrum(artifact)
    if rel is None:
        R = int(min(max(int(regions_R), 1), n)) if pinned_R else 1
        q = (float(within_region_frac) if pinned_q else 0.0) if R > 1 else 0.0
        info.update(skipped="no spectrum measurement", chosen={"R": R, "q": q, "score": None},
                    seconds=time.time() - t0)
        return R, q, labels(R), info
    lam_rel, sigma = rel
    K = len(lam_rel)

    cache: dict[int, np.ndarray | None] = {}
    spectra: dict[tuple[int, float], np.ndarray] = {}

    def score(R: int, q: float) -> float:
        R = int(min(max(R, 1), n))
        if R not in cache:
            cache[R] = labels(R)
        z = cache[R]
        edges = _calibrated_structure(artifact, deg, attrs, seed, rounds=rounds, rewire_rounds=rewire_rounds,
                                      regions=z, within_region_frac=q if z is not None else 0.0)
        lam = top_eigenvalues(edges, n, K)
        spectra[(R, q)] = lam
        s = spectrum_score(lam, lam_rel, sigma)
        info["grid"].append({"R": R, "q": q, "score": s})
        return s

    r_values = (int(regions_R),) if pinned_R else COARSE_R
    q_values = (float(within_region_frac),) if pinned_q else COARSE_Q
    done: set = set()
    scores: dict[tuple[int, float], float] = {}
    scores[(1, 0.0)] = score(1, 0.0)  # the identity, always scored (the reference of the margin)
    done.add((1, 0.0))
    for R, q in _candidate_grid(r_values, q_values, done):
        scores[(R, q)] = score(R, q)
    best = min(scores, key=lambda k: (scores[k], k[0], k[1]))
    # refinement around the best
    fine_r = (int(regions_R),) if pinned_R else tuple(sorted({max(best[0] // 2, 1), best[0], min(best[0] * 2, n)}))
    fine_q = (float(within_region_frac),) if pinned_q else tuple(
        sorted({round(min(max(best[1] + dq, 0.0), Q_MAX), 4) for dq in (-0.1, 0.0, 0.1)}))
    for R, q in _candidate_grid(fine_r, fine_q, done):
        scores[(R, q)] = score(R, q)
    raw_best = min(scores, key=lambda k: (scores[k], k[0], k[1]))
    # the q = 0 candidates are the identity up to the random stream (same blocks in another order): their
    # spread estimates the simulation noise of the score. A q > 0 candidate is used only if it beats the best
    # q = 0 score by more than that spread; a q = 0 best IS the identity.
    q0 = [s for (_, q_), s in scores.items() if q_ == 0.0]
    noise_q0 = float(max(q0) - min(q0)) if len(q0) > 1 else 0.0
    if raw_best[1] == 0.0 or not scores[raw_best] < min(q0) - noise_q0:
        best = (1, 0.0)
    else:
        best = raw_best
    base = scores.get((1, 0.0))
    margin = None
    rejected = None
    if base is not None and best != (1, 0.0):
        diff = spectra[best][: min(K, len(spectra[best]))] - spectra[(1, 0.0)][: min(K, len(spectra[(1, 0.0)]))]
        noise = 2.0 * sigma * float(np.sqrt((diff**2).sum())) / (sigma**2 + 1.0)
        margin = float((base - scores[best]) / noise) if noise > 0 else float("inf")
        # the released noise can fake a preference for regions (on SNAP at eps <= 4: sigma about 12 against
        # candidate differences of a few units per eigenvalue, and the noise-chosen regions damaged the
        # structure). Regions are used only when the released spectrum supports them beyond its noise.
        if margin < min_margin:
            rejected = {"R": best[0], "q": best[1], "score": scores[best], "margin": margin}
            best = (1, 0.0)
    R, q = best
    info.update(chosen={"R": R, "q": q, "score": scores[best]}, identity_score=base, margin_vs_identity=margin,
                min_margin=min_margin, rejected_below_margin=rejected,
                raw_best={"R": raw_best[0], "q": raw_best[1], "score": scores[raw_best]}, noise_q0=noise_q0,
                released=lam_rel.tolist(), chosen_spectrum=spectra[best].tolist(), sigma=sigma,
                seconds=time.time() - t0)
    return R, q, cache.get(R, labels(R)), info


# ------------------------------------------------------------------------------------------------ API
def generate_v2_with_info(
    artifact: Artifact,
    n: int | None = None,
    seed: int = 0,
    *,
    regions_R: int | None = None,
    within_region_frac: float | None = None,
    calibration_rounds: int = 4,
    calibration_rewire_rounds: int = 0,
    min_margin: float = MIN_MARGIN,
    rewire_rounds: int | None = None,
    **chain_options,
) -> tuple[AttributedGraph, dict]:
    """Regions + ``generate`` + ``refine_labels``; returns the graph and a report.

    The report is the chain's (energy per term before and after, satisfied flags, rounds, acceptance) plus
    ``regions_R``, ``within_region_frac``, ``regions`` (the ``(n,)`` label array, ``None`` when ``R = 1``),
    ``rewire_rounds`` and ``calibration`` (every scored candidate, the chosen one, the identity's score and the
    noise margin). ``regions_R`` / ``within_region_frac`` pin the parameters (no calibration of a pinned one);
    ``calibration_rounds`` / ``calibration_rewire_rounds`` are the reduced sampler settings used while scoring.
    ``rewire_rounds`` is the final sampler's class-pair rewiring (default: ``generate``'s 40 without regions,
    120 with them: the rho calibration densifies the blocks a little more with regions, which the rewiring
    takes a few more rounds to pull back; 40 rounds stop well before the swaps run out in both cases).
    ``chain_options`` go to ``refine_labels`` (``max_rounds``, ``batch``, ``damping``, ``temperature``, ``weights``).
    """
    n = int(artifact.schema.n if n is None else n)
    if n < 1:
        raise ValueError("n must be positive")
    R, q, z, cal = calibrate_regions(artifact, n, seed, regions_R=regions_R, within_region_frac=within_region_frac,
                                     rounds=calibration_rounds, rewire_rounds=calibration_rewire_rounds,
                                     min_margin=min_margin)
    if rewire_rounds is None:
        rewire_rounds = REWIRE_ROUNDS_WITH_REGIONS if z is not None else 40
    graph = generate(artifact, n, seed, rewire_rounds=rewire_rounds, regions=z,
                     within_region_frac=q if z is not None else 0.0)
    graph, info = refine_labels(graph, artifact, seed, **chain_options)
    info.update(regions_R=R, within_region_frac=q, regions=z, rewire_rounds=rewire_rounds, calibration=cal)
    return graph, info


def generate_v2(artifact: Artifact, n: int | None = None, seed: int = 0, **options) -> AttributedGraph:
    """Sample a synthetic ``AttributedGraph`` with regions, the v1.1 sampler and the label chain (module docstring)."""
    return generate_v2_with_info(artifact, n, seed, **options)[0]

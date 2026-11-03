"""Generator v2: the v1.1 one-sweep sampler followed by the label refinement chain.

``generate_v2`` runs ``graph_aim.outside.generate.generate`` unchanged (attributes, degrees, structure,
rewiring: everything v1.1 does, byte for byte) and then ``graph_aim.outside.refine.refine_labels``, a Markov
chain that swaps attribute values between nodes of the same degree stratum until the synthetic neighbourhood
statistics are within noise of the released tables. Without an ``attr_neigh_simpson`` measurement the chain
is skipped and the output is exactly that of ``generate``.

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

Every term is measured in its own noise units (``(C - T)^2 / (sigma^2 + T + 1)`` per cell, or
``(S - S*)^2 / (sigma^2 + 1)`` per bin), so the chain stops when each table is within about one sigma per
cell and never fits the noise. ``generate_v2_with_info`` also returns the chain's report: energy per term
before and after, satisfied flags, rounds, acceptance per round, per-bin Simpson residuals, and warnings for
measurements it could not use (unexpected shapes are skipped, never fatal).

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
"""

from __future__ import annotations

from graph_aim.outside.generate import generate
from graph_aim.outside.refine import refine_labels
from graph_aim.transfer.types import Artifact, AttributedGraph


def generate_v2_with_info(
    artifact: Artifact, n: int | None = None, seed: int = 0, **chain_options
) -> tuple[AttributedGraph, dict]:
    """``generate`` followed by ``refine_labels``; returns the graph and the chain's report.

    ``chain_options`` go to ``refine_labels`` (``max_rounds``, ``batch``, ``damping``, ``temperature``,
    ``weights``). The chain draws from ``np.random.default_rng([seed, 2])``, independent of the sampler's streams.
    """
    graph = generate(artifact, n, seed)
    return refine_labels(graph, artifact, seed, **chain_options)


def generate_v2(artifact: Artifact, n: int | None = None, seed: int = 0, **chain_options) -> AttributedGraph:
    """Sample a synthetic ``AttributedGraph`` with the v1.1 sampler and refine its labels (see the module docstring)."""
    return generate_v2_with_info(artifact, n, seed, **chain_options)[0]

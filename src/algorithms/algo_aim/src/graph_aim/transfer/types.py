"""Shared data contract between fitting (inside) and generation (outside).

Everything the generator may rely on is defined here: the synthetic graph type, one released statistic
(``Measurement``), the privacy bookkeeping, and the ``Artifact`` that bundles them. Change a field here only together
with the code on both sides and the shape checks in ``graph_aim.transfer.artifact``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AttributedGraph:
    """Undirected simple graph with categorical node attributes.

    ``edges`` is an ``(m, 2)`` int64 array with ``u < v``, unique rows, no self loops, and
    nodes labelled ``0..n-1``. ``attrs`` has one row per node (index ``0..n-1``) and one int
    code column per attribute; code ``i`` of column ``c`` means ``domains[c][i]``. Every domain
    contains explicit ``"other"`` and ``"none"`` labels, so codes are never negative.
    """

    n: int
    edges: np.ndarray
    attrs: pd.DataFrame
    domains: dict[str, list[str]]


@dataclass
class Measurement:
    """One noisy, post-processed statistic released by the mechanism.

    ``kind`` and ``values`` shapes (the contract between ``graph_aim.inside.measure`` and ``graph_aim.outside.generate``):

    - ``degree_hist``: ``(Dmax + 1,)`` node counts per degree 0..Dmax, measured on the UNCAPPED graph with
      ``params["max_degree"] = Dmax`` (degrees above Dmax are clamped into the last bin). Consumers take Dmax
      from ``len(values) - 1``, not from ``params["D"]``.
    - ``clustering_by_degree``: ``(B, 2)`` per degree bin: [sum of local clustering, node count];
      ``params["bin_edges"]`` has ``B + 1`` ascending degree edges, bin ``b`` is ``[e_b, e_{b+1})``
    - ``joint_degree``: ``(B, B)`` symmetric edge counts between degree bins (same ``bin_edges``)
    - ``attr_1way``: ``(K,)`` counts, ``params["attrs"] = [col]``
    - ``attr_2way``: ``(Ka, Kb)`` counts, ``params["attrs"] = [col_a, col_b]``
    - ``attr_edge_pair``: ``(G, G)`` symmetric edge counts between attribute groups,
      ``params["attrs"] = [col_a, ...]`` (group = product of those attributes' codes, row-major)
    - ``attr_degree``: ``(K_a, B)`` counts of nodes with attribute code ``v`` (row) and degree in bin ``j`` (column),
      ``params["attrs"] = [col]``, ``params["cap"] = None`` (UNCAPPED degree), ``params["bin_edges"]`` has ``B + 1``
      ascending edges starting at 0, bin ``j`` is ``[e_j, e_{j+1})`` and the last bin is open-ended (see
      ``graph_aim.transfer.binning.degree_bin``). Optional kind: older artifacts do not have it.

    Conventions:

    - Symmetric matrices (``joint_degree``, ``attr_edge_pair``): a cross-cell edge adds 1 to both
      ``M[a, b]`` and ``M[b, a]``; an edge inside one cell adds 1 to the diagonal once.
    - Names: ``attr_1way:<col>``, ``attr_2way:<a>|<b>``, ``attr_degree:<col>``, and the kind itself for the other three.
    - ``degree_hist`` has ``params["D"]``; ``clustering_by_degree`` has ``params["clip"]``.
    - Non-private artifacts have ``sigma == 0`` and infinite ledger totals.
    - Every measurement that uses the degree cap records its own cap in ``params["cap"]``
      (an int, or ``None`` if uncapped). Different measurements may use different caps.
    """

    name: str
    kind: str
    values: np.ndarray
    mechanism: str  # "gaussian"
    sigma: float  # noise standard deviation actually added
    sensitivity_edge: float  # L2 sensitivity under edge adjacency (proved upper bound)
    sensitivity_attr: float  # L2 sensitivity under attribute-row adjacency (proved upper bound)
    params: dict = field(default_factory=dict)


@dataclass
class PrivacyLedger:
    """zCDP bookkeeping, one ledger per adjacency relation (edge, attribute row); each guarantee holds separately.

    A query with noise ``sigma`` and L2 sensitivity ``s`` under an adjacency costs
    ``s**2 / (2 * sigma**2)`` zCDP on that adjacency's ledger.
    """

    delta: float
    rho_edge: float
    rho_attr: float
    entries: list[dict] = field(default_factory=list)  # one per measurement: name, rho_edge, rho_attr
    caveats: list[str] = field(default_factory=list)  # parts of the pipeline NOT covered by the guarantee


@dataclass
class Schema:
    """Public, data-independent description needed to interpret the measurements."""

    n: int  # node count; public under edge / attribute-row adjacency
    domains: dict[str, list[str]]
    degree_cap: int  # D
    cap_salt: int  # public salt of the edge hash used by the stable degree cap
    homophily_attrs: list[str]


@dataclass
class Artifact:
    """Everything the generator is allowed to see. No per-node data."""

    schema: Schema
    measurements: dict[str, Measurement]
    ledger: PrivacyLedger
    provenance: dict = field(default_factory=dict)  # software versions, seed policy, config hash

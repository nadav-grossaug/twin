"""Fit: cap, measure, add Gaussian noise, post-process, account (plan section 5)."""

from __future__ import annotations

import hashlib
import json
import math
import platform

import numpy as np
import pandas as pd
import scipy

from graph_aim.inside import measure as M
from graph_aim.inside import privacy as P
from graph_aim.inside.capping import cap_edges
from graph_aim.transfer.types import Artifact, AttributedGraph, Measurement, Schema

MAX_GROUPS = 64  # guards the (G, G) edge-pair table


def _totals(config: dict) -> dict:
    delta = float(config["delta"])
    out = {}
    for adj in P.ADJACENCIES:
        rho = config.get(f"rho_{adj}")
        out[adj] = float(rho) if rho is not None else P.eps_to_rho(float(config[f"epsilon_{adj}"]), delta)
    return out


def plan(graph: AttributedGraph, config: dict) -> tuple[list[M.Spec], list[str]]:
    """The public workload for ``graph``'s schema under ``config`` plus the homophily attributes.

    Config keys (all public): ``D`` (default cap), ``max_degree`` (Dmax of ``degree_hist``), ``caps``
    (per-kind cap, ``null`` = uncapped; ignored when ``D >= max_degree``, which is the no-cap reference
    mode: every capped measurement then uses ``D`` and is a no-op for graphs with max degree <= D),
    ``hist`` (``exact_upto``, ``growth``, ``threshold``, ``spread_power``), ``cluster_bin_edges``, ``cluster_min_degree``,
    ``joint_bin_edges``, ``cluster_counts_from_hist`` (release only column 0 of the clustering statistic and take
    the node counts from the processed ``degree_hist``), ``n_bins``, ``cluster_clip``. Noise-free runs use the full per-degree
    histogram and no minimum degree (nothing to save by coarsening an exact release).

    ``attr_degree`` tables (one per attribute, degree x attribute counts) are generated if and only if
    ``config["shares"].get("attr_degree", 0) > 0``, so configs without that share keep their old workload.
    Keys ``attr_degree_bin_edges`` (explicit edges; null or absent = default coarse log grid, at most 8 bins)
    and ``attr_degree_attrs`` (attribute names; null or absent = all) are public.
    """
    D = int(config["D"])
    noise = bool(config.get("noise", True))
    cols = list(graph.domains)
    homophily = config.get("homophily_attrs")
    homophily = list(cols[:2]) if homophily is None else list(homophily)
    for c in homophily:
        if c not in graph.domains:
            raise ValueError(f"homophily attribute {c!r} not in domains")
    if int(np.prod([len(graph.domains[c]) for c in homophily] or [1])) > MAX_GROUPS:
        raise ValueError("too many attribute groups for attr_edge_pair")
    max_degree = int(config["max_degree"]) if config.get("max_degree") is not None else D
    caps = config.get("caps") or {}
    if config.get("max_degree") is not None and D >= max_degree:
        caps = {}
    hist = config.get("hist") or {}
    specs = M.workload(
        graph.domains,
        D,
        homophily,
        int(config.get("n_bins", 12)),
        float(config.get("cluster_clip", 1.0)),
        caps=caps,
        max_degree=max_degree,
        hist_exact_upto=hist.get("exact_upto") if noise else None,
        hist_growth=float(hist.get("growth", 1.5)),
        hist_threshold=float(hist.get("threshold", 1.0)),
        hist_spread_power=float(hist.get("spread_power", 0.0)),
        cluster_bin_edges=config.get("cluster_bin_edges"),
        min_degree=config.get("cluster_min_degree") if noise else None,
        joint_bin_edges=config.get("joint_bin_edges"),
        n_nodes=graph.n,
        counts_from_hist=bool(config.get("cluster_counts_from_hist", False)) and noise,
        attr_degree=float((config.get("shares") or {}).get("attr_degree", 0) or 0) > 0,
        attr_degree_bin_edges=config.get("attr_degree_bin_edges"),
        attr_degree_attrs=config.get("attr_degree_attrs"),
    )
    return specs, homophily


def fit(graph: AttributedGraph, config: dict, seed: int) -> Artifact:
    """Release an :class:`Artifact` for ``graph``.

    ``config`` keys: see :func:`plan`, plus ``salt``, ``delta``, ``epsilon_edge`` / ``epsilon_attr``
    (or ``rho_edge`` / ``rho_attr`` which take precedence), ``shares`` (weight per kind), ``noise``
    (False gives the exact, non-private artifact), ``homophily_attrs`` (null means the first two
    attributes). Each measurement is computed on the graph capped at its OWN cap (``params["cap"]``).
    """
    D, salt = int(config["D"]), int(config["salt"])
    noise = bool(config.get("noise", True))
    delta = float(config["delta"])
    cols = list(graph.domains)
    if list(graph.attrs.columns) != cols:
        graph = AttributedGraph(graph.n, graph.edges, graph.attrs[cols], graph.domains)
    specs, homophily = plan(graph, config)
    sens = {s.name: M.sensitivity(s) for s in specs}
    sigmas = P.calibrate(
        [(s.name, s.kind, *sens[s.name]) for s in specs], _totals(config), config["shares"], noise
    )

    cache: dict = {}
    rng = np.random.default_rng(seed)
    raw = {}
    for s in specs:
        exact = M.exact_values(s, M.capped_for(s, graph, salt, cache), graph.n, graph.attrs, graph.domains)
        raw[s.name] = M.add_noise(s, exact, sigmas[s.name], rng)
    final = M.postprocess(raw, specs, sigmas, graph.n)

    ms = {
        s.name: Measurement(
            name=s.name,
            kind=s.kind,
            values=final[s.name],
            mechanism="gaussian",
            sigma=float(sigmas[s.name]),
            sensitivity_edge=sens[s.name][0],
            sensitivity_attr=sens[s.name][1],
            params=dict(s.params),
        )
        for s in specs
    }
    ledger = P.build_ledger(ms.values(), delta, noise)
    eps = P.ledger_epsilons(ledger)
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    provenance = {
        "method": "graph_aim_v1",
        "config_sha256": cfg_hash,
        "seed": int(seed),
        "seed_policy": "numpy.random.default_rng(seed); noise drawn per measurement in workload order",
        "noise": noise,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "epsilon_edge": None if math.isinf(eps["epsilon_edge"]) else eps["epsilon_edge"],
        "epsilon_attr": None if math.isinf(eps["epsilon_attr"]) else eps["epsilon_attr"],
    }  # nothing computed from the real graph without noise may be added here (see PROVENANCE_ALLOWED)
    schema = Schema(n=graph.n, domains={c: list(v) for c, v in graph.domains.items()}, degree_cap=D, cap_salt=salt, homophily_attrs=homophily)
    return Artifact(schema=schema, measurements=ms, ledger=ledger, provenance=provenance)

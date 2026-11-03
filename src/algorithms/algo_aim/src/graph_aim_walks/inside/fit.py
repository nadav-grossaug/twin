"""Fit the graph-aim-walks artifact on the real graph (``docs/snap/walks-plan.md`` section 3).

NOT PRIVATE: the base graph-AIM artifact is fitted with the noise switched off and the walk tables are exact
statistics of the real graph (up to walk-sampling error).
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from graph_aim.inside.fit import fit
from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.transfer import walkstats as W
from graph_aim_walks.transfer.types import METHOD, SPECTRUM, WalksArtifact

DEFAULT_SPECTRUM_K = 32


def load_config(path) -> dict:
    """Read a walks config and inline its ``base_config`` (a path relative to the config file) as a dict."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    base = cfg.get("base_config")
    if isinstance(base, str):
        bp = Path(base)
        cfg["base_config"] = json.loads((bp if bp.is_absolute() else path.parent / bp).read_text())
    return cfg


def spec_from_config(spec_cfg: dict, domains: dict) -> W.WalkSpec:
    """``WalkSpec`` from the config's ``spec`` block; ``share_attrs: "all"`` (or null) = every schema attribute."""
    d = dict(spec_cfg or {})
    share = d.get("share_attrs", "all")
    kw: dict = {}
    for key in ("length", "share_bins", "dominance_bins"):
        if key in d:
            kw[key] = int(d[key])
    for key in ("lags", "windows", "degree_bin_edges"):
        if key in d:
            kw[key] = tuple(int(x) for x in d[key])
    kw["groups"] = tuple(tuple(g) for g in d.get("groups", ()))
    kw["share_attrs"] = tuple(domains) if share in (None, "all") else tuple(share)
    kw["dominance_pairs"] = tuple(tuple(p) for p in d.get("dominance_pairs", ()))
    spec = W.WalkSpec(**kw)
    used = {c for g in spec.groups for c in g} | set(spec.share_attrs) | {c for p in spec.dominance_pairs for c in p}
    missing = sorted(used - set(domains))
    if missing:
        raise ValueError(f"walk spec names attributes not in the graph: {missing}")
    if any(len(p) != 2 for p in spec.dominance_pairs):
        raise ValueError("every dominance pair must name two attributes")
    return spec


def fit_walks(graph: AttributedGraph, config: dict, seed: int) -> WalksArtifact:
    """Base artifact (graph-AIM, noise off) + walk tables of ``config["n_walks"]`` walks + transition spectrum.

    ``config``: ``base_config`` (a graph-AIM config dict, or a path to one; see :func:`load_config`), ``n_walks``,
    ``spec`` (see :func:`spec_from_config`), ``spectrum_k`` (default 32). ``seed`` drives the walk pool.
    """
    base_cfg = config["base_config"]
    if not isinstance(base_cfg, dict):
        base_cfg = json.loads(Path(base_cfg).read_text())
    cols = list(graph.domains)
    if list(graph.attrs.columns) != cols:
        graph = AttributedGraph(graph.n, graph.edges, graph.attrs[cols], graph.domains)
    base = fit(graph, dict(base_cfg, noise=False), seed=seed)

    spec = spec_from_config(config.get("spec") or {}, graph.domains)
    n_walks = int(config["n_walks"])
    if n_walks < 1:
        raise ValueError("n_walks must be >= 1")
    k = int(config.get("spectrum_k") or DEFAULT_SPECTRUM_K)
    tables = W.walk_tables(graph.edges, graph.n, graph.attrs, graph.domains, spec, n_walks=n_walks, seed=seed)
    tables[SPECTRUM] = W.transition_spectrum(graph.edges, graph.n, k=k)

    resolved = dict(config, base_config=base_cfg)
    provenance = {
        "method": METHOD,
        "private": False,
        "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest(),
        "seed": int(seed),
        "seed_policy": "walk pool: walkstats.draw_pool with numpy.random.default_rng(seed); base fitted with noise off",
        "n_walks": n_walks,
        "spectrum_k": k,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__},
    }  # public settings only (see graph_aim_walks.transfer.artifact.PROVENANCE_ALLOWED)
    return WalksArtifact(base=base, spec=spec, tables=tables, n_walks=n_walks, provenance=provenance)

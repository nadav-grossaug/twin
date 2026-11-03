"""Scalar metrics for score rows: community structure (``com_*``) and walk statistics (``wlk_*``). Eval zone.

``community_metrics(g)``: the partition scalars of ``graph_aim.eval.community.partition_metrics`` (Louvain /
label-propagation modularity, number of parts, share of nodes in parts of >= 100 nodes, node-weighted median part
size and conductance; the v2-diagnostic definitions) plus ``spectral_gap`` = ``1 - lambda_2`` of the walk matrix
``D^-1 A`` (``walkstats.transition_spectrum``; 0 when the graph has two or more components with edges).

``walk_profile(g)``: the walk statistics of ``g`` measured with the contract's walk process
(``graph_aim_walks.transfer.walkstats``) on ONE pool of ``n_walks`` walks of ``length = 128`` steps. The first
``spec.length`` (32) steps of a 128-step walk are exactly a 32-step walk of the contract, so the matched quantities
(return / coverage for t <= 32, the ``attr_lag`` tables for the spec's lags, share variance in the spec's windows)
are read from the prefix with ``walkstats.walk_counts``; the held-out quantities (t > 32, lags 24-64, windows 64 and
128) extend the same walks. Nothing held out is ever matched by the generator (``docs/snap/walks-plan.md`` sec. 2).

- ``return[t]``: P(X_t == X_0), t = 1..128; ``coverage[t]``: mean distinct nodes among X_0..X_t.
- ``same[g][k]``: P(group of X_t == group of X_{t+k}). Matched lags: the trace of the normalised ``attr_lag:<g>``
  table (t + k <= 32); held-out lags: the same rate over t + k <= 128.
- ``attr_lag[g]``: the normalised ``attr_lag:<g>`` tables of the matched lags (for the TVD).
- ``share_var[a][w]``: across-walk variance of the share of each value of ``a`` among the distinct nodes of the
  window X_0..X_w, averaged over the values (the ``tools/v2_diag/community_strength.py`` walk-share statistic, here
  on the contract's walks and windows).
- ``spectrum``: top-16 eigenvalues of the walk matrix.

``walk_metrics(real, syn)`` turns two profiles into ``wlk_*`` gaps (see its docstring).
"""

from __future__ import annotations

import numpy as np

from graph_aim.eval.community import partition_metrics
from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.transfer import walkstats
from graph_aim_walks.transfer.walkstats import WalkSpec

# the plan's groups and dominance pairs (docs/snap/walks-plan.md section 2); filtered to the columns present
GROUPS = (("school",), ("hometown",), ("location",), ("hometown", "location"))
DOMINANCE_PAIRS = (("hometown", "location"), ("hometown", "school"), ("school", "locale"))
LENGTH = 128
HELD_OUT_WINDOWS = (64, 128)
HELD_OUT_LAGS = (24, 32, 48, 64)
N_SPECTRUM = 16


def default_spec(g: AttributedGraph) -> WalkSpec:
    """The plan's spec on the columns of ``g`` (groups, share for every attribute, dominance pairs)."""
    cols = set(g.attrs.columns)
    groups = tuple(gr for gr in GROUPS if set(gr) <= cols)
    pairs = tuple(p for p in DOMINANCE_PAIRS if set(p) <= cols)
    return WalkSpec(groups=groups, share_attrs=tuple(g.attrs.columns), dominance_pairs=pairs)


def group_label(group) -> str:
    return "+".join(group)


def variance_attrs(spec: WalkSpec) -> list[str]:
    """Attributes whose share variance enters the headline ratio: those of the attr-lag groups, in order."""
    out = []
    for gr in spec.groups:
        for a in gr:
            if a not in out:
                out.append(a)
    return out or list(spec.share_attrs)


# --------------------------------------------------------------------------------------------- community
def community_metrics(g: AttributedGraph, seed: int = 0) -> dict[str, float]:
    """``partition_metrics`` plus ``spectral_gap`` (1 - second eigenvalue of the walk matrix)."""
    out = partition_metrics(g, seed)
    spec = walkstats.transition_spectrum(g.edges, g.n, k=2)
    out["spectral_gap"] = float(1.0 - spec[1]) if len(spec) > 1 else float("nan")
    return out


# ------------------------------------------------------------------------------------------------- walks
def window_shares(paths: np.ndarray, first: np.ndarray, distinct: np.ndarray, codes: np.ndarray, k: int, w: int):
    """``(P, K)`` share of each value among the distinct nodes of X_0..X_w (as ``walkstats.walk_counts``)."""
    P = paths.shape[0]
    m = first[:, : w + 1]
    rows = np.nonzero(m)[0]
    vals = codes[paths[:, : w + 1]][m]
    cnt = np.bincount(rows * k + vals, minlength=P * k).reshape(P, k).astype(float)
    return cnt / distinct[:, w][:, None]


def walk_profile(
    g: AttributedGraph,
    spec: WalkSpec | None = None,
    n_walks: int = 50000,
    seed: int = 0,
    length: int = LENGTH,
    held_out_windows=HELD_OUT_WINDOWS,
    held_out_lags=HELD_OUT_LAGS,
) -> dict:
    """Walk statistics of ``g`` (module docstring). Keys: ``spec``, ``length``, ``return``, ``coverage``,
    ``lags``, ``held_out_lags``, ``same``, ``attr_lag``, ``windows``, ``held_out_windows``, ``share_var``,
    ``spectrum``."""
    spec = spec or default_spec(g)
    if length < spec.length or any(w > length for w in held_out_windows) or any(k > length for k in held_out_lags):
        raise ValueError("length must cover the spec's length and every held-out window and lag")
    attrs, domains = g.attrs, g.domains
    indptr, indices = walkstats.csr(g.edges, g.n)
    deg = np.diff(indptr)
    starts, U = walkstats.draw_pool(deg, n_walks, length, np.random.default_rng(seed))
    paths = walkstats.simulate(indptr, indices, starts, U)

    # matched: the contract's own counting on the first spec.length steps
    lag_spec = WalkSpec(length=spec.length, lags=spec.lags, windows=spec.windows,
                        degree_bin_edges=spec.degree_bin_edges, groups=spec.groups)
    tabs = walkstats.normalise(walkstats.walk_counts(paths[:, : spec.length + 1], deg, attrs, domains, lag_spec), n_walks)

    first = walkstats._first_occurrence(paths)
    distinct = np.cumsum(first, axis=1)
    out = {
        "spec": spec, "length": length, "n_walks": int(n_walks),
        "return": (paths[:, 1:] == paths[:, :1]).mean(axis=0),
        "coverage": distinct[:, 1:].mean(axis=0),
        "lags": list(spec.lags), "held_out_lags": list(held_out_lags),
        "windows": list(spec.windows), "held_out_windows": list(held_out_windows),
        "same": {}, "attr_lag": {}, "share_var": {},
        "spectrum": walkstats.transition_spectrum(g.edges, g.n, k=N_SPECTRUM),
    }
    for gr in spec.groups:
        lab = group_label(gr)
        t = tabs["attr_lag:" + lab]
        out["attr_lag"][lab] = t
        same = {k: float(np.trace(t[i])) for i, k in enumerate(spec.lags)}
        codes, _ = walkstats.group_codes(attrs, domains, list(gr))
        cp = codes[paths]
        for k in held_out_lags:
            same[k] = float((cp[:, :-k] == cp[:, k:]).mean())
        out["same"][lab] = same
    for a in spec.share_attrs:
        codes = np.asarray(attrs[a], dtype=np.int64)
        k = len(domains[a])
        out["share_var"][a] = {
            w: float(window_shares(paths, first, distinct, codes, k, w).var(axis=0, ddof=1).mean())
            for w in [*spec.windows, *held_out_windows]
        }
    return out


def walk_metrics(
    real: AttributedGraph,
    syn: AttributedGraph,
    spec: WalkSpec | None = None,
    n_walks: int = 50000,
    seed: int = 0,
    real_profile: dict | None = None,
    syn_profile: dict | None = None,
) -> dict[str, float]:
    """``wlk_*`` gaps between the walk profiles of ``syn`` and ``real`` (pass ``real_profile`` to reuse it):

    - ``wlk_return_abs`` / ``wlk_coverage_abs``: mean absolute error over t = 1..32 (matched); ``*_heldout``:
      over t = 33..128.
    - ``wlk_attr_lag_tvd_<group>``: total variation distance of the ``attr_lag`` tables, mean over the matched
      lags; ``wlk_attr_lag_tvd_mean`` over groups. ``wlk_attr_same_abs_heldout``: mean absolute error of the
      same-group rate over groups and held-out lags (24-64).
    - ``wlk_share_var_ratio_w<w>``: syn / real share variance for window ``w`` (8, 16, 32 matched; 64, 128 held
      out), the variance averaged over the values and over the group attributes (school, hometown, location);
      ``wlk_share_var_ratio_<attr>_w<w>`` per attribute. 1 = real; below 1 = walks see attribute mixes closer to
      the global one (weaker regions).
    - ``wlk_spectrum_abs``: mean absolute error of the top-16 walk-matrix eigenvalues.

    With ``real_profile`` the synthetic graph is measured with its spec, pool size, length and held-out settings
    (``spec`` and ``n_walks`` are then ignored).
    """
    if real_profile is not None:  # measure the synthetic graph exactly as the real one was measured
        r = real_profile
        spec, n_walks = r["spec"], r["n_walks"]
        held = dict(length=r["length"], held_out_windows=r["held_out_windows"], held_out_lags=r["held_out_lags"])
    else:
        spec = spec or default_spec(real)
        held = {}
        r = walk_profile(real, spec, n_walks, seed)
    s = syn_profile or walk_profile(syn, spec, n_walks, seed, **held)
    T = spec.length
    out = {}
    for key in ("return", "coverage"):
        d = np.abs(np.asarray(s[key]) - np.asarray(r[key]))
        out[f"wlk_{key}_abs"] = float(d[:T].mean())
        out[f"wlk_{key}_abs_heldout"] = float(d[T:].mean()) if len(d) > T else float("nan")
    tvds, held = [], []
    for lab in r["attr_lag"]:
        tv = 0.5 * np.abs(s["attr_lag"][lab] - r["attr_lag"][lab]).sum(axis=(1, 2))
        out[f"wlk_attr_lag_tvd_{lab}"] = float(tv.mean())
        tvds.append(tv.mean())
        held += [abs(s["same"][lab][k] - r["same"][lab][k]) for k in r["held_out_lags"]]
    out["wlk_attr_lag_tvd_mean"] = float(np.mean(tvds)) if tvds else float("nan")
    out["wlk_attr_same_abs_heldout"] = float(np.mean(held)) if held else float("nan")
    va = [a for a in variance_attrs(spec) if a in r["share_var"]]
    for w in [*r["windows"], *r["held_out_windows"]]:
        rv = np.mean([r["share_var"][a][w] for a in va]) if va else float("nan")
        sv = np.mean([s["share_var"][a][w] for a in va]) if va else float("nan")
        out[f"wlk_share_var_ratio_w{w}"] = float(sv / rv) if rv > 0 else float("nan")
        for a in va:
            ra = r["share_var"][a][w]
            out[f"wlk_share_var_ratio_{a}_w{w}"] = float(s["share_var"][a][w] / ra) if ra > 0 else float("nan")
    k = min(len(r["spectrum"]), len(s["spectrum"]))
    out["wlk_spectrum_abs"] = float(np.abs(np.asarray(s["spectrum"][:k]) - np.asarray(r["spectrum"][:k])).mean())
    return out

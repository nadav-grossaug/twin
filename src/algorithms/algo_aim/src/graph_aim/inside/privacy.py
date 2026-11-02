"""Gaussian mechanism, zCDP accounting and ledger construction (plan section 3).

zCDP: a Gaussian release with noise ``sigma`` and L2 sensitivity ``s`` is ``rho``-zCDP with
``rho = s**2 / (2 * sigma**2)``; zCDP composes additively. Conversion (Bun and Steinke 2016, Prop.
1.3): ``rho``-zCDP implies ``(rho + 2 sqrt(rho ln(1/delta)), delta)``-DP for every ``delta > 0``.
Two ledgers are kept (edge adjacency, attribute-row adjacency); each guarantee holds on its own.
"""

from __future__ import annotations

import math

import numpy as np

from graph_aim.transfer.types import PrivacyLedger

ADJACENCIES = ("edge", "attr")

CAVEATS = (
    "Selection of attributes and of their top-K values is data-dependent when done on the real data "
    "and is NOT covered by the guarantee.",
    "The per-measurement degree caps, the public maximum degree Dmax, the histogram grid, the degree "
    "bin edges (including the degree bins of the attribute-degree tables and, when set, the choice of "
    "attributes they are measured for), the minimum degree of the clustering statistic and the budget "
    "shares were chosen by looking at (non-private statistics of) the real graph; this choice is NOT covered by the "
    "guarantee. The shipped values and the cap salt are fixed public constants.",
    "The node count n is public: it is the same under both adjacency relations.",
    "The guarantee is stated per adjacency (edge-DP and attribute-row-DP separately); nothing is "
    "claimed for a simultaneous change of an edge and an attribute row, or for node-DP.",
    "Sensitivities are proved worst-case bounds (not empirical maxima) and are not tight for every "
    "measurement; the realised privacy loss can only be smaller than the ledger states.",
    "Post-processing (clamping, rescaling to the public n, symmetrisation, consistency) is free.",
)


def gaussian_noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    """Gaussian noise from a seeded generator; ``sigma == 0`` returns zeros without consuming RNG."""
    if sigma == 0:
        return np.zeros(shape)
    return rng.normal(0.0, sigma, size=shape)


def zcdp_cost(sensitivity: float, sigma: float) -> float:
    """rho of one Gaussian release. Zero sensitivity costs nothing; sigma 0 costs infinity."""
    if sensitivity == 0:
        return 0.0
    if sigma == 0:
        return math.inf
    return sensitivity**2 / (2.0 * sigma**2)


def rho_to_eps(rho: float, delta: float) -> float:
    if math.isinf(rho):
        return math.inf
    if rho == 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def eps_to_rho(eps: float, delta: float) -> float:
    """Largest rho whose conversion gives at most ``eps`` (inverse of :func:`rho_to_eps`)."""
    if math.isinf(eps):
        return math.inf
    root = math.log(1.0 / delta)
    return (math.sqrt(root + eps) - math.sqrt(root)) ** 2


def sigma_for(sensitivity: float, rho: float) -> float:
    """Smallest sigma giving cost <= rho; 0 if sensitivity is 0 or rho is infinite."""
    if sensitivity == 0 or math.isinf(rho):
        return 0.0
    return sensitivity / math.sqrt(2.0 * rho)


def calibrate(specs, rho_total: dict, shares: dict, noise: bool = True) -> dict:
    """Per-measurement sigma from total rho per adjacency and the budget shares per kind.

    ``specs``: iterable of ``(name, kind, s_edge, s_attr)``. ``rho_total``: ``{"edge": r, "attr": r}``.
    ``shares``: ``{kind: weight}``. On each adjacency the weights of the kinds with positive
    sensitivity there are normalised to sum to 1 and a kind's part is split equally among its
    measurements. A measurement's sigma is the larger of the values needed on the two adjacencies, so
    neither ledger exceeds its total. A measurement whose sigma is set by the OTHER adjacency (e.g.
    ``attr_edge_pair``: tiny edge sensitivity, large attribute sensitivity) would leave part of its
    allocation unspent; that slack is handed back to the remaining measurements of the adjacency in
    proportion to their weights (water-filling, iterated to a fixed point). Returns ``{name: sigma}``
    (all 0.0 when ``noise`` is False).
    """
    specs = list(specs)
    if not noise:
        return {name: 0.0 for name, *_ in specs}
    sens_idx = {"edge": 2, "attr": 3}
    names = [sp[0] for sp in specs]
    sens = {adj: {sp[0]: float(sp[idx]) for sp in specs if sp[idx] > 0} for adj, idx in sens_idx.items()}
    omega: dict[str, dict[str, float]] = {}
    for adj, idx in sens_idx.items():
        kinds: dict[str, list[str]] = {}
        for sp in specs:
            if sp[idx] > 0:
                kinds.setdefault(sp[1], []).append(sp[0])
        weight = {k: float(shares.get(k, 0.0)) for k in kinds}
        if kinds and sum(weight.values()) <= 0:
            raise ValueError(f"no budget share for kinds {sorted(kinds)} on adjacency {adj!r}")
        omega[adj] = {name: weight[k] / len(ns) for k, ns in kinds.items() for name in ns}

    slack: dict[str, dict[str, float]] = {adj: {} for adj in sens_idx}  # name -> spent rho on adj
    sigma = {name: 0.0 for name in names}
    for _ in range(100):
        alloc: dict[str, dict[str, float]] = {}
        for adj in sens_idx:
            free = [nm for nm in sens[adj] if nm not in slack[adj]]
            budget = rho_total[adj] - sum(slack[adj].values())
            tw = sum(omega[adj][nm] for nm in free)
            alloc[adj] = {nm: (budget * omega[adj][nm] / tw if tw > 0 else 0.0) for nm in free}
            alloc[adj].update(slack[adj])
        new_sigma = {}
        for nm in names:
            new_sigma[nm] = max(
                (sigma_for(sens[adj][nm], alloc[adj][nm]) for adj in sens_idx if nm in sens[adj]), default=0.0
            )
        changed = False
        for adj in sens_idx:
            for nm, s_ in sens[adj].items():
                if new_sigma[nm] > 0 and not math.isinf(rho_total[adj]):
                    spent = s_**2 / (2.0 * new_sigma[nm] ** 2)
                    if spent < alloc[adj][nm] * (1 - 1e-9) and nm not in slack[adj]:
                        slack[adj][nm] = spent
                        changed = True
                    elif nm in slack[adj] and abs(slack[adj][nm] - spent) > 1e-15 * max(1.0, spent):
                        slack[adj][nm] = spent
                        changed = True
        sigma = new_sigma
        if not changed:
            break
    # final guard: never exceed a total (covers non-convergence and rounding)
    for adj in sens_idx:
        if math.isinf(rho_total[adj]):
            continue
        spend = sum(sens[adj][nm] ** 2 / (2.0 * sigma[nm] ** 2) for nm in sens[adj] if sigma[nm] > 0)
        if spend > rho_total[adj] * (1 + 1e-12):
            f = math.sqrt(spend / rho_total[adj])
            for nm in sens[adj]:
                sigma[nm] *= f
    return sigma


def build_ledger(measurements, delta: float, noise: bool = True, extra_caveats=()) -> PrivacyLedger:
    """Fill a ledger from released measurements: totals are exactly the sums of the entries."""
    entries = []
    for m in measurements:
        entries.append(
            {
                "name": m.name,
                "kind": m.kind,
                "sigma": m.sigma,
                "rho_edge": zcdp_cost(m.sensitivity_edge, m.sigma),
                "rho_attr": zcdp_cost(m.sensitivity_attr, m.sigma),
            }
        )
    rho_edge = float(sum(e["rho_edge"] for e in entries))
    rho_attr = float(sum(e["rho_attr"] for e in entries))
    caveats = list(CAVEATS) + list(extra_caveats)
    if not noise:
        caveats.append("Non-private run: no noise added; ledger rho is infinite. Upper bound only.")
    return PrivacyLedger(delta=delta, rho_edge=rho_edge, rho_attr=rho_attr, entries=entries, caveats=caveats)


def ledger_epsilons(ledger: PrivacyLedger) -> dict:
    return {
        "epsilon_edge": rho_to_eps(ledger.rho_edge, ledger.delta),
        "epsilon_attr": rho_to_eps(ledger.rho_attr, ledger.delta),
    }

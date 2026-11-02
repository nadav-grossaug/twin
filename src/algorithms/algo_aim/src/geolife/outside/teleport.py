"""L6 teleport: ground latent places onto the target city's places (Hungarian per category).

Cost = w_pop*(popularity percentile gap)^2 + w_dens*(local density percentile gap)^2
     + w_pos*(normalised latent-vs-map position gap)^2 (keeps the latent geometry, hence the
       fitted distance decay, roughly intact). One mapping for the whole population, so shared
       latent places stay shared. Deterministic given the seed (seed only breaks ties).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from geolife.transfer.maps import MapProvider
from geolife.outside.population import Population


@dataclass
class TeleportResult:
    mapping: dict[str, str]
    report: dict = field(default_factory=dict)


def _pct(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.full(len(x), 0.5)
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def ground_places(pop: Population, provider: MapProvider, grounding: dict | None = None,
                  seed: int = 0, only_used: bool = True) -> TeleportResult:
    g = {"w_pop": 1.0, "w_dens": 0.5, "w_pos": 2.0, "category_map": {}, "density_radius_frac": 1 / 15}
    g.update(grounding or {})
    rng = np.random.default_rng(seed)
    mp = list(provider.places())
    if only_used:
        used = {v.place for u in pop.users for v in u.visits}
        for u in pop.users:
            used.update([u.home, *( [u.work] if u.work else []), *u.others])
        lat = [pop.places[k] for k in sorted(used)]
    else:
        lat = [pop.places[k] for k in sorted(pop.places)]
    lxy = np.array([[p.x, p.y] for p in lat])
    mxy = np.array([[p.x, p.y] for p in mp])

    # fit latent square into the map bounding box (uniform scale, centred)
    lmin, lmax = lxy.min(0), lxy.max(0)
    mmin, mmax = mxy.min(0), mxy.max(0)
    s = float(min((mmax - mmin)[0] / max(1e-9, (lmax - lmin)[0]),
                  (mmax - mmin)[1] / max(1e-9, (lmax - lmin)[1])))
    lmap = (lxy - (lmin + lmax) / 2) * s + (mmin + mmax) / 2
    diag = float(np.linalg.norm(mmax - mmin)) or 1.0

    r_l = pop.extent_m * g["density_radius_frac"]
    ldens = _pct(np.array([len(x) for x in cKDTree(lxy).query_ball_point(lxy, r_l)], float))
    mdens = _pct(np.array([len(x) for x in cKDTree(mxy).query_ball_point(mxy, r_l * s)], float))

    mcat = np.array([p.category for p in mp])
    mw = np.array([p.weight for p in mp])
    mpop = np.zeros(len(mp))
    for c in set(mcat):
        idx = np.flatnonzero(mcat == c)
        mpop[idx] = _pct(mw[idx])

    taken = np.zeros(len(mp), bool)
    mapping: dict[str, str] = {}
    reused = missing_cat = 0
    costs: list[float] = []
    lcat = np.array([p.category for p in lat])
    for c in sorted(set(lcat)):
        rows = np.flatnonzero(lcat == c)
        rw = np.array([lat[i].weight for i in rows])
        rpop = _pct(rw)
        provs = g["category_map"].get(c, [c])
        cand = np.flatnonzero(np.isin(mcat, provs))
        if len(cand) == 0:
            cand = np.arange(len(mp))
            missing_cat += len(rows)
        free = cand[~taken[cand]]
        use = free if len(free) else cand

        def cost(ri, ci):
            dp = (rpop[ri][:, None] - mpop[ci][None, :]) ** 2
            dd = (ldens[rows[ri]][:, None] - mdens[ci][None, :]) ** 2
            dx = (np.linalg.norm(lmap[rows[ri]][:, None, :] - mxy[ci][None, :, :], axis=2) / diag) ** 2
            return (g["w_pop"] * dp + g["w_dens"] * dd + g["w_pos"] * dx
                    + 1e-9 * rng.random((len(ri), len(ci))))

        # solve in chunks of rows that fit the free supply, then reuse for any overflow
        order = rng.permutation(len(rows))
        n_fit = min(len(rows), len(use))
        fit_rows = np.sort(order[:n_fit])
        cm = cost(fit_rows, use)
        ri, ci = linear_sum_assignment(cm)
        for a, b in zip(ri, ci):
            j = use[b]
            mapping[lat[rows[fit_rows[a]]].id] = mp[j].id
            taken[j] = True
            costs.append(float(cm[a, b]))
        over = np.sort(order[n_fit:])
        if len(over):
            cm = cost(over, cand)
            for a, i in enumerate(over):
                b = int(np.argmin(cm[a]))
                mapping[lat[rows[i]].id] = mp[cand[b]].id
                costs.append(float(cm[a, b]))
            reused += len(over)
    report = {"n_latent": len(lat), "n_reused": reused, "n_category_missing": missing_cat,
              "mean_cost": float(np.mean(costs)) if costs else 0.0, "scale": s}
    pop.truth.latent_to_map = dict(mapping)
    return TeleportResult(mapping, report)

"""Behaviour fit: DP measurements -> small parametric model (pure post-processing).

Only `Measurement.values` (already noisy) and public keys are read; nothing here touches
private data, so it costs no privacy. Noisy values are denoised by clamping (non-negativity),
sigma-thresholding of small cells, and normalisation.

Returned params dict is JSON-serialisable; see `fit_behaviour` for the fields.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from geolife.transfer.types import Measurement, PrivacyLedger
from geolife.transfer import schema as S

DEFAULT_TAU = 3.0


def _pmf(vals, sigma: float, floor_sigmas: float = 1.0) -> np.ndarray:
    """Denoise to a probability vector: zero cells below floor_sigmas*sigma, then normalise."""
    v = np.asarray(vals, dtype=float)
    keep = np.where(v >= floor_sigmas * sigma, v, 0.0)
    if keep.sum() <= 0:
        keep = np.maximum(v, 0.0)
    if keep.sum() <= 0:
        return np.full(len(v), 1.0 / len(v))
    return keep / keep.sum()


def _binned(m: Measurement | None, prefix_len: int, prefix: tuple, default_edges: list[float]):
    """Select rows whose key starts with `prefix`; return (edges, pmf, raw_total)."""
    if m is None:
        e = default_edges
        return list(e), np.full(len(e) - 1, 1.0 / (len(e) - 1)), 0.0
    rows = [(float(k[prefix_len]), float(k[prefix_len + 1]), v) for k, v in zip(m.keys, m.values)
            if tuple(k[:prefix_len]) == prefix]
    rows.sort()
    edges = [r[0] for r in rows] + [rows[-1][1]]
    vals = np.array([r[2] for r in rows])
    return edges, _pmf(vals, m.sigma), float(np.maximum(vals, 0).sum())


def _mean_log_dist(edges: list[float], p: np.ndarray) -> float:
    mids = np.array([S.mid(edges[i], edges[i + 1]) for i in range(len(p))])
    return float((p * np.log(np.maximum(mids, 1.0))).sum())


def fit_decay(mean_log_d: float, extent_m: float, zipf_s: float, n_candidates: int = 250) -> float:
    """Distance-decay exponent beta so that P(j|home) ~ w_j * max(d,200)^-beta reproduces the
    observed mean log trip length on a unit-density random layout of the given extent."""
    rng = np.random.default_rng(12345)
    pts = rng.uniform(0, extent_m, size=(n_candidates, 2))
    homes = rng.uniform(0, extent_m, size=(60, 2))
    w = np.arange(1, n_candidates + 1, dtype=float) ** -zipf_s
    rng.shuffle(w)
    d = np.maximum(np.linalg.norm(homes[:, None, :] - pts[None, :, :], axis=2), 200.0)
    best, best_err = 1.0, 1e9
    for beta in np.linspace(0.0, 3.0, 31):
        p = w[None, :] * d ** -beta
        p /= p.sum(axis=1, keepdims=True)
        err = abs(float((p * np.log(d)).sum(axis=1).mean()) - mean_log_d)
        if err < best_err:
            best, best_err = float(beta), err
    return best


def _fit_zipf(counts_desc: np.ndarray, n_users: float, a_mean: float) -> tuple[float, int]:
    """Fit (s, M): users at rank r ~ N*(1-exp(-lam*w_r)), w_r ~ r^-s, lam s.t. sum = a_mean per user.
    Only the observed top-n ranks enter the loss; M (pool size) is chosen from a small grid."""
    n = len(counts_desc)
    if n < 3 or a_mean <= 0:
        return 1.0, max(int(n * 2), 10)
    obs = np.log(np.maximum(counts_desc, 0.5))
    best = (1e18, 1.0, n * 2)
    for mult in (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0):
        M = max(n + 1, int(math.ceil(n * mult)), int(a_mean) + 2)
        r = np.arange(1, M + 1, dtype=float)

        def model(s: float) -> np.ndarray:
            w = r ** -s
            w = w / w.sum()
            f = lambda lam: float((1 - np.exp(-lam * w)).sum()) - a_mean
            lam = brentq(f, 1e-9, 1e12)
            return n_users * (1 - np.exp(-lam * w))

        def loss(s: float) -> float:
            try:
                return float(np.mean((np.log(np.maximum(model(s)[:n], 1e-9)) - obs) ** 2))
            except ValueError:
                return 1e18

        res = minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded")
        if res.fun < best[0]:
            best = (float(res.fun), float(res.x), M)
    return best[1], best[2]


def _hist(counts: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in counts:
        out[str(int(k))] = out.get(str(int(k)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _tier_stats(pr: Measurement, tau: float) -> dict:
    """Significant (>= max(tau*sigma, multiple-testing cut)) counts, histograms and category mix
    per role for one place-tier measurement."""
    cut = tau * pr.sigma
    vals = np.array(pr.values)
    roles = np.array([k[2] for k in pr.keys])
    cats = np.array([k[1] for k in pr.keys])
    out = {"hists": {}, "cuts": {}, "cat_mix": {}, "sig_counts": {}, "n_keys": len(pr.keys)}
    for ro in S.ROLES:
        sel = (roles == ro)
        # multiple-testing guard: with V noisy empty cells, expect ~1 false positive at sqrt(2 ln V)
        out["cuts"][ro] = max(cut, math.sqrt(2 * math.log(max(2, int(sel.sum())))) * pr.sigma)
        sig = sel & (vals >= out["cuts"][ro])
        c = np.rint(vals[sig]).astype(int)
        out["sig_counts"][ro] = np.sort(c)[::-1]
        out["hists"][ro] = _hist(c)
        cm: dict[str, float] = {}
        for cat_, v in zip(cats[sel], vals[sel]):
            if v >= pr.sigma:
                cm[str(cat_)] = cm.get(str(cat_), 0.0) + float(v)
        if not cm and sel.any():
            cm = {str(c_): 1.0 for c_ in set(cats[sel])}
        tot = sum(cm.values())
        out["cat_mix"][ro] = {k: v / tot for k, v in sorted(cm.items())} if tot > 0 else {}
    return out


def fit_behaviour(measurements: Sequence[Measurement], ledger: PrivacyLedger | None = None,
                  tau: float = DEFAULT_TAU, extent_m: float = 10000.0) -> dict:
    """DP measurements -> params. Fields:

    n_users_source, work_frac, other_anchor_pmf[n], zipf_s, n_other_places_source,
    hist_users_per_place {role: {k: n_places}} (k >= tau*sigma only), min_observed_k,
    group_frac {home|work: share of role-holders in observed shared groups},
    category_mix {role: {cat: p}}, hour_profile {role: [24]}, dwell {role: {edges,p}},
    role_trans {from: {to: p}}, trip_dist {kind: {edges,p}}, beta {commute|other},
    explore_prob, epr_gamma, detour {edges,p}, extent_m, privacy {rho_total, epsilon}.
    """
    ms = {m.name: m for m in measurements}
    if ledger is not None:
        have = {e.get("name") for e in ledger.entries}
        missing = [n for n in ms if n not in have]
        if missing:
            raise ValueError(f"measurements without ledger entry: {missing}")
    if S.N_USERS not in ms or not any(nm in ms for nm in S.PLACE_TIERS):
        raise ValueError("artifact lacks required measurements")

    n_users = max(1.0, ms[S.N_USERS].values[0])
    params: dict = {"extent_m": float(extent_m), "n_users_source": float(n_users), "tau": tau}

    # ---- roles / anchors
    ru = {k[0]: max(0.0, v) for k, v in zip(ms[S.ROLE_USERS].keys, ms[S.ROLE_USERS].values)} \
        if S.ROLE_USERS in ms else {}
    params["work_frac"] = float(min(1.0, ru.get("work", 0.0) / n_users))
    if S.ANCHORS_OTHER in ms:
        ao = ms[S.ANCHORS_OTHER]
        order = np.argsort([int(k[0]) for k in ao.keys])
        pmf = _pmf(np.array(ao.values)[order], ao.sigma)
    else:
        pmf = np.array([0.2, 0.3, 0.3, 0.2])
    params["other_anchor_pmf"] = [float(x) for x in pmf]
    a_mean = float((pmf * np.arange(len(pmf))).sum())

    # ---- place popularity, users-per-place, categories (hierarchical tiers, coarse -> fine)
    # Tiers present: place_role_users_r6 (full), _r7 / place_role_users (only under parents whose
    # noisy count passed the release threshold). Legacy artifacts carry only place_role_users.
    tiers = {nm: _tier_stats(ms[nm], tau) for nm in S.PLACE_TIERS if nm in ms}
    empty = {"hists": {ro: {} for ro in S.ROLES}, "cuts": {ro: 0.0 for ro in S.ROLES},
             "cat_mix": {ro: {} for ro in S.ROLES},
             "sig_counts": {ro: np.zeros(0) for ro in S.ROLES}, "n_keys": 0}
    fine = tiers.get(S.PLACE_ROLE_USERS, empty)       # fallback: no refined tier released
    params["hist_users_per_place"] = fine["hists"]
    params["min_observed_k"] = {ro: int(math.ceil(c)) for ro, c in fine["cuts"].items()}
    cat_mix = {ro: dict(v) for ro, v in fine["cat_mix"].items()}
    for ro in S.ROLES:          # categories of coarse tiers are all "unknown": use them only as a fallback
        if not cat_mix.get(ro):
            cat_mix[ro] = {"unknown": 1.0}
    params["category_mix"] = cat_mix
    sig_counts = fine["sig_counts"]
    coarse_name = next((nm for nm in S.PLACE_TIERS if nm in tiers), None)
    params["place_tiers"] = {nm: {"n_keys": t["n_keys"], "hist_users_per_cell": t["hists"],
                                  "min_observed_k": {ro: int(math.ceil(c)) for ro, c in t["cuts"].items()}}
                             for nm, t in tiers.items()}
    if coarse_name is not None and coarse_name != S.PLACE_ROLE_USERS:
        ct = tiers[coarse_name]
        params["coarse"] = {
            "tier": coarse_name,
            "popularity": {ro: [int(x) for x in ct["sig_counts"][ro]] for ro in S.ROLES},
            "group_frac": {ro: float(min(1.0, ct["sig_counts"][ro].sum() / denom)) if denom > 0 else 0.0
                           for ro, denom in (("work", params["work_frac"] * n_users), ("home", n_users))},
        }
    if S.WORK_HIST in ms:   # shared-anchor distribution: #coarse cells by #users with work there
        wm = ms[S.WORK_HIST]
        e, _, _ = _binned(wm, 0, (), S.WORK_HIST_EDGES)
        v = np.maximum(np.array([x for _, x in sorted(
            ((float(k[0]), x) for k, x in zip(wm.keys, wm.values)))]), 0.0)
        params["work_users_per_cell"] = {"edges": [x if math.isfinite(x) else 1e18 for x in e],
                                         "p": [float(x) for x in (v / v.sum() if v.sum() > 0 else v)],
                                         "sigma": float(wm.sigma)}
    s, M = _fit_zipf(sig_counts["other"].astype(float), n_users, a_mean)
    params["zipf_s"], params["n_other_places_source"] = float(s), int(M)
    gf = {}
    for ro, denom in (("work", params["work_frac"] * n_users), ("home", n_users)):
        shared = float(sig_counts[ro].sum())
        gf[ro] = float(min(1.0, shared / denom)) if denom > 0 else 0.0
    params["group_frac"] = gf

    # ---- time structure
    hp = {}
    for ro in S.ROLES:
        if S.ARRIVAL_HOUR in ms:
            m = ms[S.ARRIVAL_HOUR]
            v = np.zeros(24)
            for k, x in zip(m.keys, m.values):
                if k[0] == ro:
                    v[int(k[1])] = x
            hp[ro] = [float(x) for x in _pmf(v, m.sigma)]
        else:
            hp[ro] = [1 / 24] * 24
    params["hour_profile"] = hp
    params["dwell"] = {}
    for ro in S.ROLES:
        e, p, _ = _binned(ms.get(S.DWELL_HIST), 1, (ro,), S.DWELL_EDGES)
        params["dwell"][ro] = {"edges": [x if math.isfinite(x) else 1e18 for x in e],
                               "p": [float(x) for x in p]}
    # ---- schedule transitions
    rt = {a: {b: 0.0 for b in S.ROLES} for a in S.ROLES}
    if S.ROLE_TRIPS in ms:
        m = ms[S.ROLE_TRIPS]
        for k, x in zip(m.keys, m.values):
            rt[k[0]][k[1]] = float(x) if x >= m.sigma else 0.0
    for a in S.ROLES:
        tot = sum(rt[a].values())
        rt[a] = ({b: v / tot for b, v in rt[a].items()} if tot > 0
                 else {b: (1.0 if b == "home" else 0.0) for b in S.ROLES})
    params["role_trans"] = rt

    # ---- flows: distance decay
    params["trip_dist"], params["beta"] = {}, {}
    for kind in S.KINDS:
        e, p, tot = _binned(ms.get(S.TRIP_DIST), 1, (kind,), S.DIST_EDGES)
        params["trip_dist"][kind] = {"edges": [x if math.isfinite(x) else 1e18 for x in e],
                                     "p": [float(x) for x in p]}
        params["beta"][kind] = (fit_decay(_mean_log_dist(e, p), extent_m,
                                          params["zipf_s"] if kind == "other" else 0.0)
                                if tot > 0 else (1.0 if kind == "other" else 0.7))

    # ---- EPR
    explore = 0.1
    if S.RETURN_RATE in ms:
        rr = {k[0]: max(0.0, v) for k, v in zip(ms[S.RETURN_RATE].keys, ms[S.RETURN_RATE].values)}
        new_per_user = rr.get("new", 0.0) / n_users
        anchors = 1.0 + params["work_frac"] + a_mean
        other_visits = 0.0
        if S.ARRIVAL_HOUR in ms:
            other_visits = sum(max(0.0, v) for k, v in zip(ms[S.ARRIVAL_HOUR].keys,
                                                           ms[S.ARRIVAL_HOUR].values)
                               if k[0] == "other") / n_users
        explore = float(np.clip(max(0.0, new_per_user - anchors) / max(1.0, other_visits), 0.0, 0.5))
    params["explore_prob"] = explore
    params["epr_gamma"] = 0.21

    # ---- detour
    if S.DETOUR in ms:
        e, p, _ = _binned(ms[S.DETOUR], 0, (), S.DETOUR_EDGES)
    else:
        e, p = S.DETOUR_EDGES, np.array([0.45, 0.25, 0.15, 0.1, 0.04, 0.01])
    params["detour"] = {"edges": [x if math.isfinite(x) else 1e18 for x in e],
                        "p": [float(x) for x in p]}
    if ledger is not None:
        params["privacy"] = {"rho_total": ledger.rho_total, "epsilon": ledger.epsilon}
    return params

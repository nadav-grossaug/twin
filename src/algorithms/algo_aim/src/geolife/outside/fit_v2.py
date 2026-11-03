"""Behaviour fit v2 (pure post-processing of noisy v2 measurements -> `params` of OUTINGS_PARAMS.md).

Only Measurement.values (noisy) and public keys are read, so this costs no privacy. Each table is
denoised (clamp negatives, zero cells below 1 sigma, normalise). A table counts as "mostly noise"
when its released total mass is below SNR_MIN standard deviations of the noise on that total
(sigma * sqrt(K)); every parameter that depends on such a table falls back to DEFAULTS and is
listed in params["diagnostics"]["fallbacks"].

Conventions (documented because the spec leaves them open):
  * start_pmf[daytype][tod] and cat_profile[cat][daytype][tod] are CONDITIONAL on the daytype
    (each innermost dict sums to 1); the weekday/weekend mix is params["time"]["daytype_share"].
  * mode_pmf excludes "unknown" (labels exist for a subset of users only); the excluded share is in
    params["outing"]["mode_unknown_share"].
  * stops_pmf keys: "0".."9","10+".
  * lognormal (mu, sigma) are of the natural log of the quantity in the stated unit (minutes, km,
    counts, days, ratio), fitted by binned quantiles (median and IQR/decile spread), which a single
    noisy far-tail cell moves far less than the old moment matching did (PLAN step 4: the three
    noise replicates gave wildly different moment-matched fits).
  * detour_ratio is the lognormal of the EXCESS (ratio - 1), the convention of fakedata_v2 and
    render_v2 (`excess = lognormal(mu, sigma)`, route target = 1 + excess).
  * users_per_place_hist comes from the finest tier (res 8) only; coarser tiers are exposed in
    params["places"]["tier_hists"]. Zipf exponent uses the finest tier with >= MIN_SIG_KEYS
    significant keys (diagnostics say which); pool_size is only trusted from the res-8 tier.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from geolife.transfer.types import DAYTYPES, TOD_BINS, Measurement, PrivacyLedger
from geolife.transfer import schema_v2 as S2

SNR_MIN = 3.0
MIN_SIG_KEYS = 8
ANCHOR_MIN_CELLS = 5   # released place cells (above 1 sigma) needed before positions follow them
EXTENT_M = 40000.0
MAX_CUT_FOR_FULL_HIST = 5.0   # users: a histogram whose floor is above this misses the bulk of places
SIGMA_CAP = 2.0     # public plausibility cap on any fitted log-sd (a noisy pmf can put mass anywhere)


# ---- shared denoising / fitting helpers (moved here from the retired v1 outside/fit.py)

def _pmf(vals, sigma: float, floor_sigmas: float = 1.0) -> np.ndarray:
    """Denoise to a probability vector: zero cells below floor_sigmas*sigma, then normalise."""
    v = np.asarray(vals, dtype=float)
    keep = np.where(v >= floor_sigmas * sigma, v, 0.0)
    if keep.sum() <= 0:
        keep = np.maximum(v, 0.0)
    if keep.sum() <= 0:
        return np.full(len(v), 1.0 / len(v))
    return keep / keep.sum()


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
    from scipy.optimize import brentq, minimize_scalar
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


def _pav_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-increasing fit (post-processing of a noisy pmf)."""
    vals, wts = [], []
    for x in y:
        vals.append(float(x))
        wts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
            wts[-2:] = [w]
    return np.repeat(vals, wts)

# Generic priors (round numbers, set by the implementer; weakly informed by having read the EDA
# report, which is itself non-DP: stated in the ledger caveats of the learner).
DEFAULTS = {
    "version": 2,
    # users_per_place prior: mean ~2.8 users per visited place, half the places single-user, a thin
    # hub tail (EDA: strong point-level overlap, 6 hub places with 51-67 users). The old prior
    # (mean 1.78) was below anything the outing process can produce and made the planted
    # users-per-place check fail whenever the res-8 tier is under its noise floor.
    "places": {"zipf_s": 1.0, "pool_size": 500, "cat_mix": {"unknown": 1.0},
               "users_per_place_hist": {"1": 450, "2": 180, "3": 90, "4": 60, "6": 60, "10": 40,
                                        "20": 15, "40": 4}},
    # user priors are for the population the artifact measures (139 users PAST the minimum-activity
    # rule), hence higher than the all-182-user EDA medians
    "users": {"outings_per_user": {"mu": math.log(45), "sigma": 1.3},
              "active_days": {"mu": math.log(90), "sigma": 1.3},
              "distinct_places": {"mu": math.log(25), "sigma": 0.9},
              "revisit_rate": 0.5},
    "groups": {"n_groups": 5, "concentration": 0.3},
    "outing": {"stops_pmf": {"0": 0.5, "1": 0.2, "2": 0.1, "3": 0.07, "4": 0.04, "5": 0.03,
                             "6": 0.02, "7": 0.01, "8": 0.01, "9": 0.01, "10+": 0.01},
               "duration_min": {"mu": math.log(45), "sigma": 1.1},
               "length_km": {"mu": math.log(9), "sigma": 1.1},
               "mode_pmf": {"walk": 0.25, "bike": 0.2, "bus": 0.2, "car": 0.25, "train": 0.1},
               "stop_dwell_min": {"mu": math.log(25), "sigma": 1.0},
               "distance_decay": 1.0,
               "start_pmf": {d: {t: 0.25 for t in TOD_BINS} for d in DAYTYPES}},
    "time": {"cat_profile": {}, "daytype_share": {"weekday": 5 / 7, "weekend": 2 / 7}},
    # excess convention: ratio = 1 + lognormal(mu, sigma); median excess 0.2 (ratio 1.2)
    "route": {"detour_ratio": {"mu": math.log(0.2), "sigma": 0.8}},
}


def _copy(x):
    import copy
    return copy.deepcopy(x)


def _signal(m: Measurement | None) -> float:
    """Total released mass in units of the noise sd of that total."""
    if m is None or not m.keys or m.sigma <= 0:
        return 0.0
    return max(0.0, float(sum(m.values))) / (m.sigma * math.sqrt(len(m.keys)))


TAU_MU = 0.3        # prior sd of mu around the DEFAULTS value (empirical-Bayes shrinkage)
TAU_SIGMA = 0.15    # prior sd of sigma around the DEFAULTS value (the mean goes with exp(sigma^2/2),
                    # so a noise-inflated tail bin otherwise multiplies the generated volume).
                    # The prior weight scales with the release noise: at high budget the data wins.


def _lognormal(edges: list[float], v: np.ndarray, noise_sigma: float, prior: dict | None = None,
               floor_sigma: float = 0.05, shift: float = 0.0) -> dict:
    """MAP fit of a lognormal DIRECTLY to the raw noisy bin counts on a (mu, sigma) grid: model
    A * [Phi((log hi - mu)/s) - Phi((log lo - mu)/s)] per bin, amplitude A solved in closed form,
    Gaussian likelihood with the release noise sd, and a Gaussian prior around the public DEFAULTS
    value (TAU_MU / TAU_SIGMA). Least squares on the unclamped values pools every bin, so no cell
    thresholding is needed - the old pipeline (zero cells below 1 sigma, moment-match the
    survivors) flipped whole bins in and out between noise replicates and gave wildly different
    fits; the prior settles the near-flat directions the noise leaves behind (PLAN step 4).
    `shift` subtracts a constant from the edges first (detour excess = ratio - 1). sigma is
    searched inside [floor_sigma, SIGMA_CAP] (public plausibility window)."""
    from scipy.special import ndtr
    lo = np.asarray(edges[:-1], float) - shift
    hi = np.asarray(edges[1:], float) - shift
    llo = np.where(lo > 0, np.log(np.maximum(lo, 1e-12)), -np.inf)
    lhi = np.where(np.isfinite(hi), np.log(np.maximum(hi, 1e-12)), np.inf)
    v = np.asarray(v, float)
    mus = np.linspace(llo[np.isfinite(llo)].min() - 0.5, lhi[np.isfinite(lhi)].max() + 0.5, 96)
    sig = np.linspace(max(floor_sigma, 0.05), SIGMA_CAP, 40)
    M, S = np.meshgrid(mus, sig, indexing="ij")
    z_hi = (lhi[None, None, :] - M[:, :, None]) / S[:, :, None]
    z_lo = (llo[None, None, :] - M[:, :, None]) / S[:, :, None]
    P = ndtr(z_hi) - ndtr(z_lo)                              # (mu, sigma, bin)
    A = np.maximum((P * v).sum(-1), 0.0) / np.maximum((P * P).sum(-1), 1e-12)
    nll = ((v[None, None, :] - A[:, :, None] * P) ** 2).sum(-1) / (2 * max(noise_sigma, 1e-9) ** 2)
    if prior is not None:
        nll = nll + ((M - prior["mu"]) / TAU_MU) ** 2 / 2 + ((S - prior["sigma"]) / TAU_SIGMA) ** 2 / 2
    i, j = np.unravel_index(int(np.argmin(nll)), nll.shape)
    return {"mu": float(mus[i]), "sigma": float(sig[j])}


def _stops_fit(v: np.ndarray, noise_sigma: float, r0: float = 0.6, tau_r: float = 0.15) -> dict:
    """Stops-per-outing pmf: MAP fit of [p0, A r^(k-1) for k >= 1] on the raw noisy counts
    (geometric tail, closed-form amplitude, Gaussian prior on the tail ratio around the public
    EDA-informed r0). The old pipeline (threshold + monotone PAV) pooled the zeroed tail bins into
    a flat plateau and nearly doubled the planted stop count."""
    v = np.asarray(v, float)
    n = len(v)                                   # keys "0".."9", "10+"
    k = np.arange(1, n, dtype=float)
    best, best_nll = None, np.inf
    for r in np.linspace(0.1, 0.9, 33):
        w = r ** (k - 1)
        w[-1] = w[-1] / max(1.0 - r, 1e-9)       # "10+" bin carries the whole remaining tail
        A = max(0.0, float((v[1:] * w).sum() / (w * w).sum()))
        nll = (float(((v[1:] - A * w) ** 2).sum()) / (2 * max(noise_sigma, 1e-9) ** 2)
               + ((r - r0) / tau_r) ** 2 / 2)
        if nll < best_nll:
            best_nll, best = nll, A * w
    q = np.concatenate([[max(v[0], 0.0)], best])
    q = q / q.sum() if q.sum() > 0 else np.full(n, 1.0 / n)
    return {kk: float(x) for kk, x in zip(S2.STOPS_KEYS, q)}


def _hist_table(m: Measurement, edges: list[float]):
    """(edges, denoised pmf) with rows ordered by lower edge."""
    rows = sorted(((float(k[0]), v) for k, v in zip(m.keys, m.values)))
    return list(edges), _pmf(np.array([v for _, v in rows]), m.sigma)


def _raw_table(m: Measurement) -> np.ndarray:
    """Raw noisy values ordered by the bin's lower edge (no clamping, no thresholding)."""
    return np.array([v for _, v in sorted(((float(k[0]), v) for k, v in zip(m.keys, m.values)))])


def fit_behaviour_v2(measurements: Sequence[Measurement], ledger: PrivacyLedger | None = None,
                     extent_m: float = EXTENT_M) -> dict:
    ms = {m.name: m for m in measurements}
    if ledger is not None:
        have = {e.get("name") for e in ledger.entries}
        missing = [n for n in ms if n not in have]
        if missing:
            raise ValueError(f"measurements without ledger entry: {missing}")
    params = _copy(DEFAULTS)
    fb: dict[str, str] = {}
    snr = {n: round(_signal(m), 2) for n, m in ms.items()}

    def usable(name: str) -> Measurement | None:
        m = ms.get(name)
        if m is None:
            fb[name] = "table missing"
            return None
        if snr[name] < SNR_MIN:
            fb[name] = f"mostly noise (signal {snr[name]:.1f} < {SNR_MIN})"
            return None
        return m

    n_users = max(1.0, ms[S2.N_USERS].values[0]) if S2.N_USERS in ms else 1.0
    params["n_users_source"] = float(n_users)

    # ---- users
    for tname, edges, key in ((S2.USER_OUTINGS, S2.OUTINGS_EDGES, "outings_per_user"),
                              (S2.USER_SPAN, S2.SPAN_DAYS_EDGES, "active_days"),
                              (S2.USER_PLACES, S2.PLACES_EDGES, "distinct_places")):
        m = usable(tname)
        if m is not None:
            params["users"][key] = _lognormal(list(edges), _raw_table(m), m.sigma,
                                              prior=DEFAULTS["users"][key])
    m = usable(S2.REVISIT)
    if m is not None:
        e, p = _hist_table(m, S2.REVISIT_EDGES)
        params["users"]["revisit_rate"] = float(sum(p[i] * (e[i] + min(e[i + 1], 1.0)) / 2
                                                    for i in range(len(p))))
        # the released table is a histogram OVER USERS: keep it, so each synthetic user gets its own
        # rate (the scalar mean alone gave every user ~the same rate - no low or high revisiters)
        params["users"]["revisit_pmf"] = {"edges": [float(min(x, 1.0)) for x in e],
                                          "pmf": [float(x) for x in p]}

    # ---- outing shape
    o = params["outing"]
    m = usable(S2.OUTING_STOPS)
    if m is not None:
        v = np.array([dict(zip([k[0] for k in m.keys], m.values))[k] for k in S2.STOPS_KEYS])
        o["stops_pmf"] = _stops_fit(v, m.sigma)
    for tname, edges, key in ((S2.OUTING_DURATION, S2.DURATION_MIN_EDGES, "duration_min"),
                              (S2.OUTING_LENGTH, S2.LENGTH_KM_EDGES, "length_km"),
                              (S2.STOP_DWELL, S2.DWELL_MIN_EDGES, "stop_dwell_min")):
        m = usable(tname)
        if m is not None:
            o[key] = _lognormal(list(edges), _raw_table(m), m.sigma,
                                prior=DEFAULTS["outing"][key])
    m = usable(S2.OUTING_MODE)
    if m is not None:
        d = {k[0]: max(0.0, v) for k, v in zip(m.keys, m.values)}
        # clamp at zero only (no sigma cut): the cut zeroed small-but-real shares (train) on
        # some noise replicates and kept them on others
        known = np.array([d[x] for x in S2.MODES if x != "unknown"])
        tot_all = sum(d.values())
        if known.sum() > 0:
            o["mode_pmf"] = {x: float(v / known.sum()) for x, v in
                             zip([x for x in S2.MODES if x != "unknown"], known)}
            o["mode_unknown_share"] = float(d["unknown"] / tot_all) if tot_all > 0 else 1.0
        else:
            fb[S2.OUTING_MODE] = "no known-mode mass above 1 sigma"
    m = usable(S2.OUTING_START)
    if m is not None:
        grid = {d: np.zeros(len(S2.TOD_3)) for d in DAYTYPES}
        for (c, dt), v in zip(m.keys, m.values):
            # clamp at zero only: the old sigma cut zeroed whole cells (e.g. weekend morning)
            # on some noise replicates, which planted a world without weekend-morning outings
            grid[dt][S2.TOD_3.index(c)] += max(0.0, v)
        share = {d: float(g.sum()) for d, g in grid.items()}
        if sum(share.values()) > 0:
            tot = sum(share.values())
            params["time"]["daytype_share"] = {d: s / tot for d, s in share.items()}
            o["start_pmf"] = {d: _expand_coarse(g) for d, g in grid.items()}
        else:
            fb[S2.OUTING_START] = "no positive mass"

    # ---- places: categories and time profile. cat_time_visits is no longer released (D27: at or
    # below the usability edge on every dev seed; the per-category content never cleared the
    # internal floors even when the table passed the overall floor). An OLD artifact may still
    # carry the table; it is ignored either way and the public defaults are used.
    cat_profile_global = None
    fb["cat_mix"] = fb["cat_profile"] = (
        "cat_time_visits not released (D27); category mix and visit-time profiles are the public "
        "defaults")

    m = usable(S2.OUTING_DETOUR)
    if m is not None:
        # released bins are of the ratio (>= 1); the param is the lognormal of the EXCESS, the
        # convention render_v2 draws with (the old fit returned the ratio itself, which render
        # then read as an excess: every route was detoured to ~2.3x the direct path)
        params["route"]["detour_ratio"] = _lognormal(list(S2.DETOUR_EDGES), _raw_table(m), m.sigma,
                                                     prior=DEFAULTS["route"]["detour_ratio"], shift=1.0)

    # ---- place popularity tiers
    tier_info = {}
    for tn in S2.PLACE_TIERS:
        m = ms.get(tn)
        if m is None or not m.keys:
            tier_info[tn] = {"n_keys": 0, "n_sig": 0}
            continue
        cut = max(3.0 * m.sigma, math.sqrt(2 * math.log(max(2, len(m.keys)))) * m.sigma)
        vals = np.array(m.values)
        sig = np.sort(np.rint(vals[vals >= cut]).astype(int))[::-1]
        tier_info[tn] = {"n_keys": len(m.keys), "n_sig": int(len(sig)), "cut": float(cut),
                         "sig": sig}
    params["places"]["tier_hists"] = {tn: _hist(t["sig"]) for tn, t in tier_info.items()
                                      if t.get("n_sig", 0) > 0}
    # where the users are: the released tier KEYS are H3 cells of the source city. Keep the finest
    # coarse tier with enough cells above 1 sigma as spatial anchors (cell -> denoised user count);
    # the population layer uses them only when the target map covers the source city.
    for tn, res in ((S2.PLACE_TIERS[1], S2.PLACE_RES[1]), (S2.PLACE_TIERS[0], S2.PLACE_RES[0])):
        m = ms.get(tn)
        if m is None or not m.keys:
            continue
        cells = [(str(k[0]), float(v)) for k, v in zip(m.keys, m.values) if v >= m.sigma]
        if len(cells) >= ANCHOR_MIN_CELLS:
            params["places"]["anchor_cells"] = {"res": int(res), "tier": tn, "cells": cells}
            break
    else:
        fb["anchor_cells"] = f"no coarse place tier with >= {ANCHOR_MIN_CELLS} cells above 1 sigma"
    fine = tier_info[S2.PLACE_TIERS[-1]]
    params["places"]["users_per_place_tail"] = {
        "min_k": int(math.ceil(fine.get("cut", 0.0))),
        "hist": _hist(fine["sig"]) if fine["n_sig"] else {}}
    if fine["n_sig"] >= 3 and fine["cut"] <= MAX_CUT_FOR_FULL_HIST:
        params["places"]["users_per_place_hist"] = _hist(fine["sig"])
    else:
        fb["users_per_place_hist"] = (
            f"res-8 tier: {fine['n_sig']} keys above the noise floor, floor at {fine.get('cut', 0):.0f} "
            f"users (> {MAX_CUT_FOR_FULL_HIST}); only the tail is observed (places.users_per_place_tail)")
    zipf_tier = next((tn for tn in reversed(S2.PLACE_TIERS) if tier_info[tn]["n_sig"] >= MIN_SIG_KEYS), None)
    if zipf_tier is None:
        fb["zipf_s"] = fb["pool_size"] = f"no place tier has >= {MIN_SIG_KEYS} significant keys"
    else:
        tn = zipf_tier
        if tn == S2.PLACE_TIERS[-1]:
            a_mean = math.exp(params["users"]["distinct_places"]["mu"]
                              + params["users"]["distinct_places"]["sigma"] ** 2 / 2)
            a_mean = min(a_mean, float(S2.MAX_STOP_CELLS))
        else:
            a_mean = max(1.0, float(sum(max(0.0, v) for v in ms[tn].values)) / n_users)
        s, M = _fit_zipf(tier_info[tn]["sig"].astype(float), n_users, a_mean)
        params["places"]["zipf_s"] = float(s)
        if tn == S2.PLACE_TIERS[-1]:
            params["places"]["pool_size"] = int(M)
        else:
            fb["pool_size"] = f"Zipf fitted on tier {tn}; pool size not identifiable at that resolution"
        if tn != S2.PLACE_TIERS[-1]:
            fb["zipf_s_tier"] = f"zipf_s from coarse tier {tn} (cells, not places)"

    # ---- distance decay from hop distances
    m = usable(S2.STOP_HOP)
    if m is not None:
        e, p = _hist_table(m, S2.HOP_KM_EDGES)
        mids_m = np.array([S2.mid(e[i], e[i + 1]) for i in range(len(p))]) * 1000.0
        mld = float((p * np.log(np.maximum(mids_m, 1.0))).sum())
        o["distance_decay"] = float(fit_decay(mld, extent_m, params["places"]["zipf_s"]))
        if o["distance_decay"] >= 2.99:
            fb["distance_decay_boundary"] = "fit_decay hit its grid bound 3.0: treat as a lower bound"
    else:
        fb["distance_decay"] = "stop_hop mostly noise"

    # ---- parameters whose tables were dropped in v2b (noise in the first real-data run)
    fb["start_time_fine"] = ("time of day is released at 3 bins (morning/midday/evening x daytype) "
                             "only; the 4-bin values are an assumed HOUR_SHAPE-weighted split of "
                             "each released bin (a fixed public prior, not measured)")
    fb["dropped_tables"] = ("place_time_users, stop_transitions_r6, trav_users_r8, trav_users_r9 "
                            "(v2b) and cat_time_visits, cat_transitions, sensor_step (v2d, D27) "
                            "are no longer released (noise on real data); their parameters use "
                            "the documented defaults")

    # ---- groups: no co-visit table released
    fb["groups"] = "no co-visit table released; defaults used"

    # ---- sensor tables (render draws; post-processing in outside/sensors.py)
    if any(n in ms for n in S2.SENSOR_NAMES):
        from geolife.outside.sensors import fit_sensors
        sens = fit_sensors(ms)
        params["sensor"] = sens
        snr.update({f"{k}": v for k, v in sens["signal"].items()})
        for k, v in sens["fallbacks"].items():
            fb[f"sensor:{k}"] = v
    else:
        fb["sensor"] = "no sensor tables in the artifact; render_v2 uses its hand-set defaults"

    params["diagnostics"] = {"fallbacks": fb, "signal_ratio": snr,
                             "cat_profile_global": cat_profile_global,
                             "zipf_tier": zipf_tier,
                             "tiers": {tn: {k: v for k, v in t.items() if k != "sig"}
                                       for tn, t in tier_info.items()},
                             "snr_min": SNR_MIN}
    if ledger is not None:
        params["privacy"] = {"rho_total": ledger.rho_total, "epsilon": ledger.epsilon}
    return params


def _tod3_fine_shape() -> dict:
    """TOD_3 bin -> {fine TOD bin: weight}: each released bin's mass split over the fine bins it
    overlaps, proportional to the public HOUR_SHAPE prior (commute peaks at 8h and 17-18h, almost
    nothing at 0-5h) summed over the shared local hours. Replaces the uniform-over-hours split of
    schema_v2.TOD3_TO_FINE, which put ~38% of the evening/night [16,05) mass into the 0-5h fine bin
    and made synthetic outings start at night (PLAN step 5: the hour-profile TV). Still a fixed
    public assumption, not a measurement (ledger caveat unchanged)."""
    from geolife.outside.population_v2 import HOUR_SHAPE
    from geolife.transfer.types import local_tod3_bin
    w = {c: {t: 0.0 for t in TOD_BINS} for c in S2.TOD_3}
    for h in range(24):
        t_utc = h * 3600 - 8 * 3600 + 86400 * 700          # any day; only the local hour matters
        w[local_tod3_bin(t_utc)][TOD_BINS[h // 6]] += float(HOUR_SHAPE[h])
    return {c: {t: v / sum(d.values()) for t, v in d.items()} for c, d in w.items()}


_TOD3_FINE_SHAPE: dict | None = None


def _expand_coarse(a) -> dict:
    """Released TOD_3 (morning, midday, evening) masses -> normalised pmf over the 4 TOD_BINS, each
    released bin split with the fixed HOUR_SHAPE weights of _tod3_fine_shape (an assumption, not
    measured). Zero mass -> uniform over the 4 fine bins (documented default)."""
    global _TOD3_FINE_SHAPE
    if _TOD3_FINE_SHAPE is None:
        _TOD3_FINE_SHAPE = _tod3_fine_shape()
    a = np.where(np.asarray(a) > 0, a, 0.0)
    if a.sum() <= 0:
        return {t: 0.25 for t in TOD_BINS}
    a = a / a.sum()
    return {t: float(sum(a[i] * _TOD3_FINE_SHAPE[c][t] for i, c in enumerate(S2.TOD_3)))
            for t in TOD_BINS}

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
    counts, days, ratio), moment-matched on bin midpoints.
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
from geolife.outside.fit import _fit_zipf, _hist, _pmf, fit_decay

SNR_MIN = 3.0
MIN_SIG_KEYS = 8
EXTENT_M = 40000.0
CAT_K_SIGMA = 4.0   # per-category mass must exceed this many sd of its noise (multiple testing over ~10 cats)
MAX_CUT_FOR_FULL_HIST = 5.0   # users: a histogram whose floor is above this misses the bulk of places


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
    "places": {"zipf_s": 1.0, "pool_size": 500, "cat_mix": {"unknown": 1.0},
               "users_per_place_hist": {"1": 600, "2": 150, "3": 70, "4": 40, "6": 25, "10": 10, "20": 4}},
    "users": {"outings_per_user": {"mu": math.log(30), "sigma": 1.2},
              "active_days": {"mu": math.log(60), "sigma": 1.5},
              "distinct_places": {"mu": math.log(20), "sigma": 1.0},
              "revisit_rate": 0.5},
    "groups": {"n_groups": 5, "concentration": 0.3},
    "outing": {"stops_pmf": {"0": 0.5, "1": 0.2, "2": 0.1, "3": 0.07, "4": 0.04, "5": 0.03,
                             "6": 0.02, "7": 0.01, "8": 0.01, "9": 0.01, "10+": 0.01},
               "duration_min": {"mu": math.log(40), "sigma": 1.0},
               "length_km": {"mu": math.log(7), "sigma": 1.2},
               "mode_pmf": {"walk": 0.25, "bike": 0.2, "bus": 0.2, "car": 0.25, "train": 0.1},
               "stop_dwell_min": {"mu": math.log(25), "sigma": 0.7},
               "distance_decay": 1.0,
               "start_pmf": {d: {t: 0.25 for t in TOD_BINS} for d in DAYTYPES}},
    "time": {"cat_profile": {}, "daytype_share": {"weekday": 5 / 7, "weekend": 2 / 7}},
    "route": {"detour_ratio": {"mu": math.log(1.3), "sigma": 0.25}},
}


def _copy(x):
    import copy
    return copy.deepcopy(x)


def _signal(m: Measurement | None) -> float:
    """Total released mass in units of the noise sd of that total."""
    if m is None or not m.keys or m.sigma <= 0:
        return 0.0
    return max(0.0, float(sum(m.values))) / (m.sigma * math.sqrt(len(m.keys)))


def _lognormal(edges: list[float], p: np.ndarray, floor_sigma: float = 0.05) -> dict:
    mids = np.array([S2.mid(edges[i], edges[i + 1]) for i in range(len(p))])
    lm = np.log(np.maximum(mids, 1e-9))
    mu = float((p * lm).sum())
    var = float((p * (lm - mu) ** 2).sum())
    return {"mu": mu, "sigma": float(max(floor_sigma, math.sqrt(var)))}


def _hist_table(m: Measurement, edges: list[float]):
    """(edges, denoised pmf) with rows ordered by lower edge."""
    rows = sorted(((float(k[0]), v) for k, v in zip(m.keys, m.values)))
    return list(edges), _pmf(np.array([v for _, v in rows]), m.sigma)


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
            e, p = _hist_table(m, edges)
            params["users"][key] = _lognormal(e, p)
    m = usable(S2.REVISIT)
    if m is not None:
        e, p = _hist_table(m, S2.REVISIT_EDGES)
        params["users"]["revisit_rate"] = float(sum(p[i] * (e[i] + min(e[i + 1], 1.0)) / 2
                                                    for i in range(len(p))))

    # ---- outing shape
    o = params["outing"]
    m = usable(S2.OUTING_STOPS)
    if m is not None:
        v = np.array([dict(zip([k[0] for k in m.keys], m.values))[k] for k in S2.STOPS_KEYS])
        q = _pav_decreasing(np.maximum(v, 0.0))      # monotone smoothing: no holes in the tail
        q = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))
        o["stops_pmf"] = {k: float(x) for k, x in zip(S2.STOPS_KEYS, q)}
    for tname, edges, key in ((S2.OUTING_DURATION, S2.DURATION_MIN_EDGES, "duration_min"),
                              (S2.OUTING_LENGTH, S2.LENGTH_KM_EDGES, "length_km"),
                              (S2.STOP_DWELL, S2.DWELL_MIN_EDGES, "stop_dwell_min")):
        m = usable(tname)
        if m is not None:
            e, p = _hist_table(m, edges)
            o[key] = _lognormal(e, p)
    m = usable(S2.OUTING_MODE)
    if m is not None:
        d = {k[0]: max(0.0, v) for k, v in zip(m.keys, m.values)}
        known = np.array([d[x] if d[x] >= m.sigma else 0.0 for x in S2.MODES if x != "unknown"])
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
            grid[dt][S2.TOD_3.index(c)] += max(0.0, v) if v >= m.sigma else 0.0
        share = {d: float(g.sum()) for d, g in grid.items()}
        if sum(share.values()) > 0:
            tot = sum(share.values())
            params["time"]["daytype_share"] = {d: s / tot for d, s in share.items()}
            o["start_pmf"] = {d: _expand_coarse(g) for d, g in grid.items()}
        else:
            fb[S2.OUTING_START] = "no mass above 1 sigma"

    # ---- places: categories and time profile
    ct = usable(S2.CAT_TIME)
    if ct is not None:
        cats = sorted({k[0] for k in ct.keys})
        arr = {c: np.zeros((2, len(S2.TOD_3))) for c in cats}
        for (c, t, d), v in zip(ct.keys, ct.values):
            arr[c][DAYTYPES.index(d), S2.TOD_3.index(t)] = max(0.0, v)
        raw = {c: 0.0 for c in cats}
        for (c, _, _), v in zip(ct.keys, ct.values):
            raw[c] += v                  # unclamped sum: zero-mean noise, no positive bias
        mass = {c: float(a.sum()) for c, a in arr.items()}
        tot = sum(mass.values())
        glob = sum(arr.values())
        gprof = {d: _expand_coarse(glob[i]) for i, d in enumerate(DAYTYPES)}
        keep = {c: mass[c] for c in cats if raw[c] >= CAT_K_SIGMA * ct.sigma * math.sqrt(2 * len(S2.TOD_3))}
        if tot > 0 and keep:
            ktot = sum(keep.values())
            params["places"]["cat_mix"] = {c: float(m_ / ktot) for c, m_ in sorted(keep.items())}
        else:
            fb["cat_mix"] = "category mass below 3 sigma"
        prof, global_fb = {}, []
        for c in cats:
            if c in keep:
                prof[c] = {d: _expand_coarse(arr[c][i]) for i, d in enumerate(DAYTYPES)}
            else:
                prof[c] = gprof
                global_fb.append(c)
        params["time"]["cat_profile"] = prof
        if global_fb:
            fb["cat_profile"] = f"global profile used for {len(global_fb)} of {len(cats)} categories"
    else:
        fb["cat_mix"] = fb["cat_profile"] = "cat_time_visits mostly noise"

    m = usable(S2.OUTING_DETOUR)
    if m is not None:
        e, p = _hist_table(m, S2.DETOUR_EDGES)
        params["route"]["detour_ratio"] = _lognormal(e, p)

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
                             "only; the 4-bin values are an assumed uniform-over-hours split of "
                             "each released bin (not measured)")
    fb["dropped_tables"] = ("place_time_users, stop_transitions_r6, trav_users_r8, trav_users_r9 are no "
                            "longer released (pure noise on real data); no parameter reads them, "
                            "so nothing further falls back")

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
                             "zipf_tier": zipf_tier,
                             "tiers": {tn: {k: v for k, v in t.items() if k != "sig"}
                                       for tn, t in tier_info.items()},
                             "snr_min": SNR_MIN}
    if ledger is not None:
        params["privacy"] = {"rho_total": ledger.rho_total, "epsilon": ledger.epsilon}
    return params


def _expand_coarse(a) -> dict:
    """Released TOD_3 (morning, midday, evening) masses -> normalised pmf over the 4 TOD_BINS. Each
    released bin's mass is split over its fine bins with the FIXED shape schema_v2.TOD3_TO_FINE
    (uniform over local hours: an assumption, not measured). Zero mass -> uniform over the 4 fine
    bins (documented default)."""
    a = np.where(np.asarray(a) > 0, a, 0.0)
    if a.sum() <= 0:
        return {t: 0.25 for t in TOD_BINS}
    a = a / a.sum()
    return {t: float(sum(a[i] * S2.TOD3_TO_FINE[c][t] for i, c in enumerate(S2.TOD_3)))
            for t in TOD_BINS}

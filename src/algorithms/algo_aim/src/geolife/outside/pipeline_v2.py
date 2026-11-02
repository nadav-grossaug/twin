"""params (or artifact dir) + MapProvider -> synthetic outings. End to end: population -> teleport -> render."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from geolife.transfer.maps import MapProvider
from geolife.outside.fit_v2 import fit_behaviour_v2
from geolife.outside.population_v2 import PopulationV2, latent_view, sample_population_v2
from geolife.outside.render_v2 import RenderResultV2, render_v2
from geolife.outside.teleport import TeleportResult, ground_places


@dataclass
class GeneratedV2:
    params: dict
    population: PopulationV2
    teleport: TeleportResult
    result: RenderResultV2
    timings: dict = field(default_factory=dict)


def _params_from_artifact(art) -> dict:
    """Artifact tables (already noisy) -> generator params; post-processing only."""
    return fit_behaviour_v2(art.measurements, art.ledger)


def generate_v2(params_or_artifact: dict | str | Path, provider: MapProvider, n: int, seed: int,
                gen: dict | None = None, grounding: dict | None = None) -> GeneratedV2:
    """`gen` overrides sensor / process knobs (sensitivity studies); it is layered on the artifact
    `generation` dict when an artifact directory is given."""
    if isinstance(params_or_artifact, dict):
        params, agen, agr = params_or_artifact, {}, {}
    else:
        from geolife.transfer.artifact import load_artifact
        art = load_artifact(params_or_artifact)
        params, agen, agr = _params_from_artifact(art), dict(art.generation or {}), dict(art.grounding or {})
    g = {**agen, **(gen or {})}
    gr = grounding if grounding is not None else agr
    tm: dict = {}
    t = time.perf_counter()
    pop = sample_population_v2(params, n, seed, provider, g)
    tm["population_s"] = time.perf_counter() - t
    t = time.perf_counter()
    tp = ground_places(latent_view(pop), provider, gr, seed)
    tm["teleport_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res = render_v2(pop, tp.mapping, provider, g, seed)
    tm["render_s"] = time.perf_counter() - t
    return GeneratedV2(params, pop, tp, res, tm)


# ------------------------------------------------------------------ planted-statistics check
def planted_stats(gen: GeneratedV2, n_mc: int = 40000) -> dict:
    """Compare statistics of the generated population (rendered truth) with the `params` they came from.
    Returns {name: {"target", "got", "tol", "kind", "ok"}}; tolerances are meant for n >= ~50 users."""
    import numpy as np
    from geolife.transfer.types import LOCAL_UTC_OFFSET_H
    from geolife.outside.population_v2 import T0_LOCAL, joint_pmf, stops_sampler

    params, pop, res = gen.params, gen.population, gen.result
    g = pop.gen
    we = g["weekend_share_default"]
    out: dict = {}

    def put(name, target, got, tol, kind="abs"):
        err = abs(got - target) / (abs(target) if kind == "rel" else 1.0)
        out[name] = {"target": float(target), "got": float(got), "tol": tol, "kind": kind, "ok": bool(err <= tol)}

    def put_min(name, got, floor):
        out[name] = {"target": floor, "got": float(got), "tol": floor, "kind": "min", "ok": bool(got >= floor)}

    def tod_dt(t):
        loc = int(t) + LOCAL_UTC_OFFSET_H * 3600          # local clock; day 0 of the synthetic calendar is a Monday
        return (1 if ((int(t) - T0_LOCAL) // 86400) % 7 >= 5 else 0), (loc % 86400) // 3600 // 6

    n_out = np.array([len(u.outings) for u in pop.users], float)
    put("outings_per_user_median", float(np.exp(params["users"]["outings_per_user"]["mu"])),
        float(np.median(n_out)), 0.35, "rel")
    # stops per outing and revisit rate of stops (earlier visit of any kind by the same user)
    n_stops, rev, tot = [], 0, 0
    for u in pop.users:
        seen: set[str] = set()
        for o in res.outings[u.uid]:
            n_stops.append(sum(v.kind == "stop" for v in o.visits))
            for v in o.visits:
                if v.kind == "stop":
                    tot += 1
                    rev += v.place_id in seen
                seen.add(v.place_id)
    draw = stops_sampler(params["outing"]["stops_pmf"], g["stops_tail_p"], g["max_stops"])
    r = np.random.default_rng(0)
    put("stops_per_outing_mean", float(np.mean([draw(r) for _ in range(n_mc)])), float(np.mean(n_stops)), 0.15, "rel")
    put("stop_revisit_rate", params["users"]["revisit_rate"], rev / max(1, tot), 0.12)
    # outing start time: joint (daytype, tod) pmf in local time
    tgt = joint_pmf(params["outing"]["start_pmf"], we)
    obs = np.zeros((2, 4))
    for u in pop.users:
        for o in res.outings[u.uid]:
            d, b = tod_dt(o.t_start)
            obs[d, b] += 1
    obs /= obs.sum()
    put("start_tod_daytype_tv", 0.0, 0.5 * float(np.abs(obs - tgt).sum()), 0.10)
    put("weekend_share_of_starts", float(tgt[1].sum()), float(obs[1].sum()), 0.08)
    # category time profile: weekend share per category, and log-lift of each cell over all categories
    cat_prof = params["time"]["cat_profile"]
    cnt: dict = {}
    for vs in res.truth.visits.values():
        for v in vs:
            if v.kind == "stop":
                d, b = tod_dt(v.t_arrive)
                cnt.setdefault(v.category, np.zeros((2, 4)))[d, b] += 1
    cats = [c for c in cat_prof if c in cnt and cnt[c].sum() >= 100]
    if len(cats) >= 3:
        prof = {c: joint_pmf(cat_prof[c], we) for c in cats}
        a = [prof[c][1].sum() for c in cats]
        b_ = [cnt[c][1].sum() / cnt[c].sum() for c in cats]
        put_min("cat_weekend_share_corr", np.corrcoef(a, b_)[0, 1], 0.5)
        allc = sum(cnt[c] for c in cats)
        allc = allc / allc.sum()
        pall = sum(prof.values())
        pall = pall / pall.sum()
        lo = np.concatenate([np.log((cnt[c] / cnt[c].sum() + 1e-3) / (allc + 1e-3)).ravel() for c in cats])
        lm = np.concatenate([np.log(prof[c] / pall).ravel() for c in cats])
        put_min("cat_tod_lift_corr", np.corrcoef(lo, lm)[0, 1], 0.5)
    # users per place (distinct users per visited place)
    upp: dict[str, set] = {}
    for uid, vs in res.truth.visits.items():
        for v in vs:
            upp.setdefault(v.place_id, set()).add(uid)
    ks = np.array([len(s) for s in upp.values()], float)
    hist = params["places"].get("users_per_place_hist") or {}
    if hist:
        hk = np.array([int(str(k).rstrip("+")) for k in hist], float)
        hn = np.array(list(hist.values()), float)
        put("users_per_place_mean", float((hk * hn).sum() / hn.sum()), float(ks.mean()), 0.30, "rel")
        put("share_places_ge2_users", float(hn[hk >= 2].sum() / hn.sum()), float((ks >= 2).mean()), 0.22)
    put("distinct_places_per_user_median", float(np.exp(params["users"]["distinct_places"]["mu"])),
        float(np.median([len({v.place_id for v in vs}) for vs in res.truth.visits.values()])), 1.0, "rel")
    # shared places and group concentration
    put_min("n_shared_places", len(res.truth.shared_places), 1)
    conc = params.get("groups", {}).get("concentration", 0.0)
    shares = [np.mean([v.place_id in set(res.truth.group_places.get(gid, ())) for v in res.truth.visits[uid]])
              for gid, members in res.truth.groups.items() for uid in members if res.truth.visits[uid]]
    if shares and conc > 0:
        put("group_visit_share", conc, float(np.mean(shares)), 0.2)
    # informative only (depends on the target city extent)
    ln = [o.length_m / 1000 for os_ in res.outings.values() for o in os_]
    du = [(o.t_end - o.t_start) / 60 for os_ in res.outings.values() for o in os_]
    out["info"] = {"length_km_median": float(np.median(ln)), "duration_min_median": float(np.median(du)),
                   "target_length_km_median": float(np.exp(params["outing"]["length_km"]["mu"]) * g["length_scale"]),
                   "target_duration_min_median": float(np.exp(params["outing"]["duration_min"]["mu"]))}
    return out

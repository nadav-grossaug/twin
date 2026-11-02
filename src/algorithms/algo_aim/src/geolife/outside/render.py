"""L7 render: route the population's visits on the target map and emit raw-style GPS.

Sensor behaviour (noise sd, outliers, sampling interval + jitter, signal gaps, dwell jitter,
per-role stay logging) comes from the artifact `generation` dict (see population.DEFAULT_GENERATION)
so it can be calibrated from the real data. Co-travel is planted: the follower is picked up at the
leader's home and rides the leader's exact route at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geolife.transfer.types import GpsPoint
from geolife.transfer.maps import MapProvider, Route
from geolife.outside.population import GroundTruth, Population, T0, merged_generation, mode_speed, sample_bin


@dataclass
class RenderResult:
    traces: dict[str, list[GpsPoint]]
    truth: GroundTruth
    stays: dict[str, list[tuple[str, int, int, str]]]            # (map place, t0, t1, role)
    trips: dict[str, list[dict]] = field(default_factory=dict)   # origin,dest,t0,t1,segments,length_m,mode


def _times(t0: float, t1: float, interval: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    if t1 <= t0:
        return np.array([t0])
    out, t = [t0], t0
    while t < t1:
        n = int((t1 - t) / interval * 1.3) + 16
        steps = interval * np.clip(1 + jitter * rng.standard_normal(n), 0.2, None)
        ts = t + np.cumsum(steps)
        out.extend(ts[ts <= t1].tolist())
        t = float(ts[-1])
    return np.array(out)


def _detoured(provider: MapProvider, a: str, b: str, ids: list[str], det: dict, gen: dict,
              rng: np.random.Generator) -> Route:
    base = provider.route(a, b)
    k = int(gen["detour_candidates"])
    if k <= 0 or base.length_m < 600:
        return base
    r = sample_bin(det["edges"], det["p"], rng)
    if r <= 1.05:
        return base
    best, best_err = base, abs(1.0 - r)
    for v in rng.choice(len(ids), size=min(k, len(ids)), replace=False):
        r1, r2 = provider.route(a, ids[v]), provider.route(ids[v], b)
        L = r1.length_m + r2.length_m
        err = abs(L / base.length_m - r)
        if err < best_err:
            best_err = err
            best = Route(r1.segments + r2.segments, L, np.vstack([r1.polyline, r2.polyline[1:]]))
    return best


def render(pop: Population, mapping: dict[str, str], provider: MapProvider, gen: dict | None = None,
           seed: int = 0) -> RenderResult:
    g = merged_generation({**pop.gen, **(gen or {})})
    places = {p.id: p for p in provider.places()}
    ids = sorted(places)
    t_end = T0 + g["n_days"] * 86400
    truth = pop.truth
    for ct in truth.cotravel:
        ct.days, ct.segments = [], []
    by_uid = {u.uid: i for i, u in enumerate(pop.users)}
    cache: dict[tuple, tuple] = {}
    stays_out: dict[str, list] = {}
    trips_out: dict[str, list] = {}
    traces: dict[str, list[GpsPoint]] = {}
    ct_of = {c.follower: c for c in truth.cotravel}

    def run_user(u) -> None:
        rng = np.random.default_rng([seed, by_uid[u.uid], 7])
        cur, cur_arr, cur_role = mapping[u.home], float(T0), "home"
        stays, trips = [], []
        for v in u.visits:
            dest = mapping[v.place]
            if dest == cur:
                continue
            joint = None
            if v.joint_leader:
                ci = cache.get((v.joint_leader, round(v.t_depart)))
                if ci and ci["origin"] == cur and ci["dest"] == dest and cur_arr <= ci["depart"] - 30:
                    joint = ci
            if joint:
                route, depart, speed = joint["route"], joint["depart"], joint["speed"]
            else:
                route = _detoured(provider, cur, dest, ids, pop.params["detour"], g, rng)
                depart = max(v.t_depart, cur_arr + 60)
                nominal = "walk" if v.speed < 3.0 else "car"
                actual = "walk" if route.length_m < g["walk_max_m"] else "car"
                speed = v.speed * g["speed_mps"][actual] / g["speed_mps"][nominal]
            arrival = depart + route.length_m / speed
            stays.append((cur, cur_arr, depart, cur_role))
            trips.append({"origin": cur, "dest": dest, "t0": depart, "t1": arrival,
                          "segments": route.segments, "length_m": route.length_m, "speed": speed,
                          "route": route,
                          "mode": "walk" if route.length_m < g["walk_max_m"] else "car"})
            if not joint:
                cache[(u.uid, round(v.t_depart))] = {"origin": cur, "dest": dest, "depart": depart,
                                                     "route": route, "speed": speed}
            elif u.uid in ct_of:
                ct = ct_of[u.uid]
                ct.days.append(int((depart - T0) // 86400))
                ct.segments.append((ct.days[-1], tuple(route.segments)))
            cur, cur_arr, cur_role = dest, arrival, v.role
        stays.append((cur, cur_arr, max(t_end, cur_arr), cur_role))
        stays_out[u.uid] = [(p, int(a), int(b), r) for p, a, b, r in stays]
        trips_out[u.uid] = [{k: v for k, v in t.items() if k not in ("route",)} for t in trips]
        traces[u.uid] = _emit(provider, g, rng, stays, trips, places, t_end)

    for u in pop.users:
        if u.carpool_leader is None:
            run_user(u)
    for u in pop.users:
        if u.carpool_leader is not None:
            run_user(u)
    traces = {u.uid: traces[u.uid] for u in pop.users}
    stays_out = {u.uid: stays_out[u.uid] for u in pop.users}
    trips_out = {u.uid: trips_out[u.uid] for u in pop.users}

    _finish_truth(truth, mapping, stays_out, trips_out, g)
    return RenderResult(traces, truth, stays_out, trips_out)


def _emit(provider, g, rng, stays, trips, places, t_end) -> list[GpsPoint]:
    T, X, Y = [], [], []
    for p, a, b, role in stays:
        if rng.random() > g["stay_logged_prob"].get(role, 1.0):
            continue
        t = _times(a, b, g["stay_interval_s"], g["interval_jitter"], rng)
        T.append(t)
        X.append(places[p].x + g["dwell_jitter_m"] * rng.standard_normal(len(t)))
        Y.append(places[p].y + g["dwell_jitter_m"] * rng.standard_normal(len(t)))
        T[-1] = t
    for tr in trips:
        r: Route = tr["route"]
        t = _times(tr["t0"], tr["t1"], g["trip_interval_s"], g["interval_jitter"], rng)
        seg = np.linalg.norm(np.diff(r.polyline, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.clip((t - tr["t0"]) / max(1e-9, tr["t1"] - tr["t0"]), 0, 1) * cum[-1]
        T.append(t)
        X.append(np.interp(s, cum, r.polyline[:, 0]))
        Y.append(np.interp(s, cum, r.polyline[:, 1]))
    if not T:
        return []
    t, x, y = np.concatenate(T), np.concatenate(X), np.concatenate(Y)
    n = len(t)
    sd = g["sensor_noise_m"]
    out = rng.random(n) < g["outlier_prob"]
    x = x + sd * rng.standard_normal(n) * np.where(out, g["outlier_scale"], 1.0)
    y = y + sd * rng.standard_normal(n) * np.where(out, g["outlier_scale"], 1.0)
    # signal gaps over the whole timeline
    keep = np.ones(n, bool)
    n_gaps = rng.poisson(g["gap_rate_per_hr"] * (t_end - T0) / 3600)
    for st, du in zip(rng.uniform(T0, t_end, n_gaps), rng.exponential(g["gap_mean_s"], n_gaps)):
        keep &= ~((t >= st) & (t < st + du))
    t, x, y = t[keep], x[keep], y[keep]
    ti = np.rint(t).astype(np.int64)
    ti, idx = np.unique(ti, return_index=True)
    lat, lon = provider.to_latlon(x[idx], y[idx])
    return [GpsPoint(int(a), float(b), float(c)) for a, b, c in zip(ti, lat, lon)]


def _finish_truth(truth: GroundTruth, mapping, stays, trips, g) -> None:
    seg_users: dict[str, set] = {}
    for uid, ts in trips.items():
        for t in ts:
            for s in t["segments"]:
                seg_users.setdefault(s, set()).add(uid)
    truth.crossing_segments = {s: sorted(u) for s, u in sorted(seg_users.items()) if len(u) >= 2}
    by_place: dict[str, list] = {}
    for uid, ss in stays.items():
        for p, a, b, _ in ss:
            by_place.setdefault(p, []).append((a, b, uid))
    cop = []
    for p, iv in sorted(by_place.items()):
        iv.sort()
        for i, (a, b, u) in enumerate(iv):
            for a2, b2, u2 in iv[i + 1:]:
                if a2 >= b:
                    break
                lo, hi = max(a, a2), min(b, b2)
                if u != u2 and hi - lo >= g["min_copresence_s"]:
                    cop.append((p, min(u, u2), max(u, u2), lo, hi))
    truth.copresence = cop
    truth.latent_to_map = dict(mapping)

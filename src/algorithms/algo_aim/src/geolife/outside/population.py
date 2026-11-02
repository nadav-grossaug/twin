"""Synthetic population over LATENT places: anchors (EPR), shared workplaces/homes, schedules.

Everything is driven by the params dict from fit.fit_behaviour and a seeded numpy Generator.
Planted relations are recorded in GroundTruth (render.py completes it with routed facts).
Latent geometry lives in an abstract square of `extent_m` metres; teleport.py maps it onto a city.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T0 = 1704067200  # Monday 2024-01-01 00:00:00 UTC (synthetic clock)

DEFAULT_GENERATION: dict = {
    "n_days": 7, "weekday_work_prob": 0.9, "carpool_frac": 0.3, "walk_max_m": 900.0,
    "speed_mps": {"walk": 1.3, "car": 8.0}, "speed_sd": 0.15, "detour_factor": 1.3,
    "max_visits_per_day": 6, "evening_outing_prob": 0.25,
    # sensor / render knobs (calibrated from real data when available)
    "sensor_noise_m": 8.0, "outlier_prob": 0.005, "outlier_scale": 6.0,
    "trip_interval_s": 5.0, "stay_interval_s": 60.0, "interval_jitter": 0.3,
    "gap_rate_per_hr": 0.3, "gap_mean_s": 120.0, "dwell_jitter_m": 6.0,
    "stay_logged_prob": {"home": 0.6, "work": 0.9, "other": 1.0},
    "min_copresence_s": 600, "detour_candidates": 8,
}


@dataclass
class LatentPlace:
    id: str
    kind: str            # home | work | other
    category: str
    weight: float
    x: float
    y: float


@dataclass
class Visit:
    place: str                   # latent place id
    role: str                    # home | work | other | pickup
    t_depart: float              # departure time from the previous place (epoch s)
    speed: float                 # m/s, sampled here so co-travellers can share it
    day: int
    joint_leader: str | None = None


@dataclass
class SimUser:
    uid: str
    home: str
    work: str | None = None
    others: list[str] = field(default_factory=list)   # visit-frequency order
    carpool_leader: str | None = None                 # set on followers
    visits: list[Visit] = field(default_factory=list)


@dataclass
class PlantedGroup:
    kind: str            # "workplace" | "household"
    place: str           # latent place id
    users: list[str]


@dataclass
class CoTravel:
    leader: str
    follower: str
    workplace: str       # latent id
    pickup_place: str    # latent id (leader's home)
    days: list[int] = field(default_factory=list)          # filled by render (days really shared)
    segments: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)  # (day, segments)


@dataclass
class GroundTruth:
    users: list[str] = field(default_factory=list)
    anchors: dict[str, dict] = field(default_factory=dict)       # uid -> {home, work, others} (latent)
    groups: list[PlantedGroup] = field(default_factory=list)
    shared_places: dict[str, list[str]] = field(default_factory=dict)  # latent id -> users (>=2 anchors)
    cotravel: list[CoTravel] = field(default_factory=list)
    copresence: list[tuple[str, str, str, int, int]] = field(default_factory=list)  # place,u,v,t0,t1
    crossing_segments: dict[str, list[str]] = field(default_factory=dict)  # segment -> users (>=2)
    latent_to_map: dict[str, str] = field(default_factory=dict)


@dataclass
class Population:
    users: list[SimUser]
    places: dict[str, LatentPlace]
    truth: GroundTruth
    params: dict
    gen: dict
    extent_m: float
    seed: int


# ------------------------------------------------------------------ helpers

def merged_generation(gen: dict | None) -> dict:
    out = {**DEFAULT_GENERATION}
    for k, v in (gen or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def sample_bin(edges, p, rng: np.random.Generator) -> float:
    i = int(rng.choice(len(p), p=np.asarray(p) / np.sum(p)))
    lo, hi = float(edges[i]), float(edges[i + 1])
    if hi > 1e17:
        return lo * (1 + rng.exponential(1.0)) if lo > 0 else rng.exponential(3600.0)
    if lo > 0:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def mode_speed(dist_m: float, gen: dict) -> float:
    return gen["speed_mps"]["walk" if dist_m < gen["walk_max_m"] else "car"]


def _scatter(n: int, extent: float, rng: np.random.Generator, core_bias: float) -> np.ndarray:
    uni = rng.uniform(0, extent, (n, 2))
    core = np.clip(rng.normal(extent / 2, extent / 6, (n, 2)), 0, extent)
    return np.where((rng.random(n) < core_bias)[:, None], core, uni)


def _group_sizes(hist: dict, n_in_groups: int, rng: np.random.Generator) -> list[int]:
    if not hist or n_in_groups <= 0:
        return []
    ks = np.array([int(k) for k in hist])
    p = np.array(list(hist.values()), float)
    p /= p.sum()
    sizes, tot = [], 0
    while tot < n_in_groups:
        k = min(int(rng.choice(ks, p=p)), n_in_groups - tot)
        sizes.append(k)
        tot += k
    return sizes


def _choice_cat(mix: dict, n: int, rng: np.random.Generator, default: str) -> list[str]:
    if not mix:
        return [default] * n
    cats = list(mix)
    p = np.array([mix[c] for c in cats], float)
    return [cats[i] for i in rng.choice(len(cats), size=n, p=p / p.sum())]


# ------------------------------------------------------------------ population

def sample_population(params: dict, n_users: int, seed: int, gen: dict | None = None) -> Population:
    gen = merged_generation(gen)
    rng = np.random.default_rng(seed)
    extent = float(params["extent_m"])
    scale = n_users / max(1.0, params["n_users_source"])
    used_ids: set[str] = set()

    def rid() -> str:
        while True:
            s = f"L{int(rng.integers(0, 2 ** 40)):010x}"
            if s not in used_ids:
                used_ids.add(s)
                return s

    places: dict[str, LatentPlace] = {}

    def make(kind: str, cats: list[str], xy: np.ndarray, weights=None) -> list[str]:
        ids = []
        for i in range(len(cats)):
            pid = rid()
            places[pid] = LatentPlace(pid, kind, cats[i], 1.0 if weights is None else float(weights[i]),
                                      float(xy[i, 0]), float(xy[i, 1]))
            ids.append(pid)
        return ids

    uids = [f"u{i:04d}" for i in range(n_users)]
    users = [SimUser(u, "") for u in uids]
    cm = params["category_mix"]

    # homes: shared households + singletons
    n_shared = int(round(n_users * params["group_frac"].get("home", 0.0)))
    sizes = _group_sizes(params["hist_users_per_place"].get("home", {}), n_shared, rng)
    sizes += [1] * (n_users - sum(sizes))
    hxy = _scatter(len(sizes), extent, rng, 0.3)
    home_ids = make("home", _choice_cat(cm.get("home", {}), len(sizes), rng, "residential"), hxy)
    order = rng.permutation(n_users)
    pos = 0
    groups: list[PlantedGroup] = []
    for pid, k in zip(home_ids, sizes):
        members = [users[i] for i in order[pos:pos + k]]
        pos += k
        for u in members:
            u.home = pid
        if k >= 2:
            groups.append(PlantedGroup("household", pid, [u.uid for u in members]))

    # workplaces: capacity groups, assigned by gravity on commute distance
    has_work = rng.random(n_users) < params["work_frac"]
    workers = [users[i] for i in np.flatnonzero(has_work)]
    nw = len(workers)
    wsizes = _group_sizes(params["hist_users_per_place"].get("work", {}),
                          int(round(nw * params["group_frac"].get("work", 0.0))), rng)
    wsizes = [k for k in wsizes if k > 0]
    wsizes += [1] * (nw - sum(wsizes))
    wxy = _scatter(len(wsizes), extent, rng, 0.7)
    work_ids = make("work", _choice_cat(cm.get("work", {}), len(wsizes), rng, "office"), wxy)
    cap = np.array(wsizes, float)
    wpos = np.array([[places[w].x, places[w].y] for w in work_ids]).reshape(-1, 2)
    members: dict[str, list[SimUser]] = {w: [] for w in work_ids}
    beta_c = params["beta"]["commute"]
    for i in rng.permutation(nw):
        u = workers[i]
        h = places[u.home]
        ok = np.flatnonzero(cap > 0)
        d = np.maximum(np.hypot(wpos[ok, 0] - h.x, wpos[ok, 1] - h.y), 200.0)
        p = d ** -beta_c * cap[ok]
        j = int(ok[rng.choice(len(ok), p=p / p.sum())])
        cap[j] -= 1
        u.work = work_ids[j]
        members[work_ids[j]].append(u)
    for w, ms in members.items():
        if len(ms) >= 2:
            groups.append(PlantedGroup("workplace", w, [u.uid for u in ms]))

    # other places: Zipf pool, users draw anchors by popularity x distance decay
    M = max(5, int(round(params["n_other_places_source"] * scale)))
    w_other = np.arange(1, M + 1, dtype=float) ** -params["zipf_s"]
    w_other = w_other[rng.permutation(M)]
    oxy = _scatter(M, extent, rng, 0.5)
    other_ids = make("other", _choice_cat(cm.get("other", {}), M, rng, "shop"), oxy, w_other)
    beta_o = params["beta"]["other"]
    pmf = np.array(params["other_anchor_pmf"], float)
    A = rng.choice(len(pmf), size=n_users, p=pmf / pmf.sum())
    for u, a in zip(users, A):
        h = places[u.home]
        d = np.maximum(np.hypot(oxy[:, 0] - h.x, oxy[:, 1] - h.y), 200.0)
        p = w_other * d ** -beta_o
        k = int(min(a, M))
        if k:
            u.others = [other_ids[j] for j in rng.choice(M, size=k, replace=False, p=p / p.sum())]

    # carpool pairs inside workplace groups (planted co-travel)
    cotravel: list[CoTravel] = []
    taken: set[str] = set()
    for w, ms in members.items():
        ms = [u for u in ms if u.uid not in taken]
        if len(ms) >= 2 and rng.random() < gen["carpool_frac"]:
            lead, fol = ms[0], ms[1]
            if lead.home != fol.home:
                fol.carpool_leader = lead.uid
                taken.update((lead.uid, fol.uid))
                cotravel.append(CoTravel(lead.uid, fol.uid, w, lead.home))

    ctx = _Ctx(params, gen, places, other_ids, oxy, w_other, rng)
    by_uid = {u.uid: u for u in users}
    days: dict[str, dict[int, list[Visit]]] = {u.uid: {} for u in users}
    for u in users:
        if u.carpool_leader is None:
            for d in range(gen["n_days"]):
                days[u.uid][d] = _day(u, d, ctx)
    for u in users:
        if u.carpool_leader is not None:
            lead = by_uid[u.carpool_leader]
            for d in range(gen["n_days"]):
                vs = _day(u, d, ctx)
                lv = days[lead.uid][d]
                if vs and lv and vs[0].role == "work" and lv[0].role == "work":
                    t_day = T0 + d * 86400
                    sp = ctx.speed(places[u.home], places[lead.home])
                    t_pick = lv[0].t_depart - 300 - ctx.est_travel(places[u.home], places[lead.home], sp)
                    if t_pick > t_day + 600:
                        vs = [Visit(lead.home, "pickup", t_pick, sp, d),
                              Visit(u.work, "work", lv[0].t_depart, lv[0].speed, d, lead.uid)] + vs[1:]
                days[u.uid][d] = vs
    for u in users:
        u.visits = [v for d in sorted(days[u.uid]) for v in days[u.uid][d]]

    # ground truth (latent level)
    truth = GroundTruth(users=list(uids), groups=groups, cotravel=cotravel)
    occupancy: dict[str, list[str]] = {}
    for u in users:
        truth.anchors[u.uid] = {"home": u.home, "work": u.work, "others": list(u.others)}
        for pid in {u.home, u.work, *u.others} - {None}:
            occupancy.setdefault(pid, []).append(u.uid)
    truth.shared_places = {p: sorted(v) for p, v in occupancy.items() if len(v) >= 2}
    return Population(users, places, truth, params, gen, extent, seed)


class _Ctx:
    def __init__(self, params, gen, places, other_ids, oxy, w_other, rng):
        self.params, self.gen, self.places = params, gen, places
        self.other_ids, self.oxy, self.w_other, self.rng = other_ids, oxy, w_other, rng
        self.dwell = {r: v for r, v in params["dwell"].items()}

    def dist(self, a: LatentPlace, b: LatentPlace) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def speed(self, a: LatentPlace, b: LatentPlace) -> float:
        base = mode_speed(self.dist(a, b) * self.gen["detour_factor"], self.gen)
        return float(base * np.exp(self.rng.normal(0, self.gen["speed_sd"])))

    def est_travel(self, a: LatentPlace, b: LatentPlace, speed: float) -> float:
        return self.dist(a, b) * self.gen["detour_factor"] / speed


def _arrival(profile, earliest: float, rng: np.random.Generator) -> float | None:
    h0 = int(earliest // 3600)
    if h0 >= 24:
        return None
    p = np.array(profile, float)
    p[:h0] = 0
    if p.sum() <= 0:
        return None
    h = int(rng.choice(24, p=p / p.sum()))
    return max(earliest, h * 3600 + rng.random() * 3600)


def _day(u: SimUser, d: int, ctx: _Ctx) -> list[Visit]:
    P, G, rng, places = ctx.params, ctx.gen, ctx.rng, ctx.places
    t_day = T0 + d * 86400
    workday = (d % 7) < 5 and rng.random() < G["weekday_work_prob"]
    cur = places[u.home]
    cur_role = "home"
    free = 0.0
    visited_work = False
    out: list[Visit] = []
    freq = np.array([1.0 / (k + 1) ** (1 + P.get("epr_gamma", 0.21)) for k in range(len(u.others))])
    for _ in range(G["max_visits_per_day"]):
        row = dict(P["role_trans"][cur_role])
        if not (u.work and workday and not visited_work):
            row["work"] = 0.0
        if cur_role == "home":
            row["home"] = 0.0
        tot = sum(row.values())
        if tot <= 0:
            break
        roles = list(row)
        dest_role = roles[int(rng.choice(len(roles), p=np.array([row[r] for r in roles]) / tot))]
        if dest_role == "home":
            dest = places[u.home]
        elif dest_role == "work":
            dest = places[u.work]
        else:
            dest = _pick_other(u, cur, ctx, freq)
            if dest is None or dest.id == cur.id:
                break
        sp = ctx.speed(cur, dest)
        est = ctx.est_travel(cur, dest, sp)
        arr = _arrival(P["hour_profile"][dest_role], free + est, rng)
        if arr is None:
            if cur_role == "home" or dest_role == "home" and cur_role == "home":
                break
            dest_role, dest = "home", places[u.home]
            sp = ctx.speed(cur, dest)
            est = ctx.est_travel(cur, dest, sp)
            arr = free + est
        out.append(Visit(dest.id, dest_role, t_day + arr - est, sp, d))
        visited_work |= dest_role == "work"
        dw = ctx.dwell[dest_role]
        free = arr + sample_bin(dw["edges"], dw["p"], rng)
        cur, cur_role = dest, dest_role
        if dest_role == "home" and (arr > 18 * 3600 or rng.random() > G["evening_outing_prob"]):
            break
    if cur_role != "home" and out:
        home = places[u.home]
        sp = ctx.speed(cur, home)
        est = ctx.est_travel(cur, home, sp)
        out.append(Visit(home.id, "home", t_day + free, sp, d))
    return out


def _pick_other(u: SimUser, cur: LatentPlace, ctx: _Ctx, freq) -> LatentPlace | None:
    rng, places = ctx.rng, ctx.places
    explore = rng.random() < ctx.params["explore_prob"] or not u.others
    if not explore:
        k = int(rng.choice(len(u.others), p=freq / freq.sum()))
        return places[u.others[k]]
    d = np.maximum(np.hypot(ctx.oxy[:, 0] - cur.x, ctx.oxy[:, 1] - cur.y), 200.0)
    p = ctx.w_other * d ** -ctx.params["beta"]["other"]
    return places[ctx.other_ids[int(rng.choice(len(p), p=p / p.sum()))]]

"""Generator v2, population layer: an OUTING PROCESS over LATENT places (no home/work roles).

Each user has an activity region (centre + radius), a taste vector over categories and an optional latent
group (shared favoured places, e.g. a campus). A user is a list of outings placed in time inside a bursty
active-days window. An outing = start place, n stops, end place. Every place draw is
    popularity x taste x activity-region x time-of-day category profile x distance decay (x length target)
and, with probability `revisit_rate`, is instead a (preferential) revisit of an already visited place, which
creates favourites without any role. Time convention: all hour/day-of-week logic is local time UTC+8;
epoch timestamps are UTC. Latent geometry is metric (extent_m square); teleport maps it onto a city.

Planted truth is recorded in GroundTruthV2 (latent level here, completed by render_v2 after grounding).
`params` is the dict of docs/geolife/OUTINGS_PARAMS.md; every extra knob lives in the `gen` dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from geolife.transfer.types import DAYTYPES, LOCAL_UTC_OFFSET_H, TOD_BINS
from geolife.outside.population import LatentPlace

# Monday 2024-01-01 00:00 local (UTC+8) expressed as a UTC epoch. Day d: weekday iff d % 7 < 5.
T0_LOCAL = 1704067200 - LOCAL_UTC_OFFSET_H * 3600
MODES = ("walk", "bike", "bus", "car", "train")

# relative hour-of-day shape inside each tod bin: commute peaks at 8h and 17-18h (EDA section 3).
HOUR_SHAPE = np.array([1.0, 0.6, 0.4, 0.3, 0.3, 0.5,            # night 0-5
                       1.5, 3.0, 5.0, 3.5, 2.5, 2.5,            # morning 6-11 (peak 8h)
                       2.5, 2.2, 2.0, 2.2, 3.0, 4.5,            # afternoon 12-17 (peak 17h)
                       4.0, 3.0, 2.5, 2.0, 1.5, 1.0])           # evening 18-23 (peak 18h)

DEFAULT_GEN_V2: dict = {
    # world
    "extent_m": 10000.0, "horizon_days": 365, "core_bias": 0.5, "pool_from_hist": True, "pool_cap_frac": 0.7,
    "pool_size_override": None, "calibrate_pool": True, "calibrate_iters": 4,
    # users
    "outings_per_active_day": 1.5, "min_outings": 2, "max_outings": 2500, "min_gap_s": 3 * 3600,
    "region_radius_frac": 0.15, "taste_alpha": 20.0, "group_fav_size": 14, "ungrouped_frac": 0.2,
    "group_size_decay": 0.7,
    # outing process. Stop draws target the released revisit_rate exactly (a proportional
    # controller on the user's realised stop-revisit share); start/end draws lean to familiar
    # places (people leave from and return to home-like places), which keeps the distinct-place
    # count near the released user_places histogram instead of growing with every outing.
    "distinct_activity_corr": 0.8, "distinct_cap_mult": 6.0,
    "continue_prob": 0.5, "roundtrip_prob": 0.15, "revisit_pref": 0.8, "d0_m": 200.0,
    "start_revisit_prob": 0.85, "end_revisit_prob": 0.9,
    "length_targeting": True, "length_bump_sigma": 0.9, "length_scale": 1.0, "detour_nominal": 1.6,
    # rendered tokenizer length per unit of latent straight-line x detour_nominal length (route
    # geometry beyond the nominal detour, drawn extra detours, in-stop drift path), observed on
    # the comparison runs; the latent length calibration divides its target by this. The same
    # pattern for users per place: grounding merge, start/end snap flicker and h3 fallbacks lift
    # the tokenized users-per-place mean above the latent one.
    "calibrate_length": True, "length_render_bias": 1.3, "users_per_place_render_bias": 1.2,
    # walk outings are much shorter than motorised ones (a 0.4 factor still made the median walk
    # outing ~8 km, i.e. a two-hour walk, which inflated durations and points per user), and the
    # lognormal length tail is capped per mode (nobody walks 30 km): public physical priors
    "mode_length_factor": {"walk": 0.25, "bike": 0.5, "bus": 1.4, "car": 2.0, "train": 3.0},
    "mode_length_cap_km": {"walk": 3.5, "bike": 10.0, "bus": 35.0, "car": 60.0, "train": 120.0},
    "weekend_share_default": 0.26, "max_stops": 15, "stops_tail_p": 0.3,
    "dwell_min_s": 600.0, "dwell_max_s": 8 * 3600.0,
    # co-travel (carpool / same route)
    "cotravel_user_frac": 0.2, "cotravel_outing_frac": 0.15,
    # speeds used for planning arrival times (render uses the same table). Cruising values: the
    # renderer's stop-and-go dips the realised median ~25% below these, so the motorised defaults
    # sit above the EDA labelled medians (tokenized real medians: bus 5.7, car 10.9, train 17)
    "speed_ms_by_mode": {"walk": 1.3, "bike": 3.5, "bus": 6.5, "car": 12.0, "train": 22.0},
}


def merged_gen_v2(gen: dict | None) -> dict:
    out = {**DEFAULT_GEN_V2}
    for k, v in (gen or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ------------------------------------------------------------------ data
@dataclass
class SimOuting:
    t_start: int                       # UTC epoch, departure from the start place
    mode: str
    speed_f: float                     # per-outing speed multiplier (shared with co-travellers)
    places: list[str]                  # latent ids: [start, stop_1..stop_k, end]
    dwell_s: list[int]                 # per stop (len k)
    copy_of: tuple[str, int] | None = None   # (leader uid, leader outing index) for co-travel followers


@dataclass
class SimUserV2:
    uid: str
    group: int | None
    centre: tuple[float, float]
    radius: float
    sampler_id: int                    # index used to seed render
    outings: list[SimOuting] = field(default_factory=list)
    window: tuple[int, int] = (0, 0)   # first/last local day index of the active window
    leader: str | None = None          # set on co-travel followers


@dataclass
class GroundTruthV2:
    groups: dict = field(default_factory=dict)             # group id -> [user]
    favourite_places: dict = field(default_factory=dict)   # user -> [place id] (latent, then map ids)
    shared_places: dict = field(default_factory=dict)      # place id -> [user] (>= 2 users)
    shared_segments: dict = field(default_factory=dict)    # segment id -> [user] (>= 2 users)
    co_traveled: list = field(default_factory=list)        # (leader, follower, leader outing index)
    visits: dict = field(default_factory=dict)             # user -> [Visit] as rendered
    # extras (not in the T3/T4 contract)
    co_traveled_detail: list = field(default_factory=list)  # (leader, i_leader, follower, i_follower)
    group_places: dict = field(default_factory=dict)       # group id -> [place id] favoured by the group
    user_group: dict = field(default_factory=dict)         # user -> group id | None
    latent_to_map: dict = field(default_factory=dict)


@dataclass
class PopulationV2:
    users: list[SimUserV2]
    places: dict[str, LatentPlace]
    truth: GroundTruthV2
    params: dict
    gen: dict
    extent_m: float
    seed: int
    report: dict = field(default_factory=dict)


# ------------------------------------------------------------------ param parsing helpers
def joint_pmf(dd: dict, we_share: float) -> np.ndarray:
    """{daytype: {tod: p}} -> (2, 4) joint pmf. Accepts a joint pmf (total ~1) or per-daytype conditional
    pmfs (total ~2), in which case the weekend share is `we_share`."""
    a = np.array([[float(dd.get(dt, {}).get(b, 0.0)) for b in TOD_BINS] for dt in DAYTYPES])
    tot = a.sum()
    if tot <= 0:
        a = np.ones((2, 4))
        tot = 8.0
    rs = a.sum(1)
    if abs(tot - 2.0) < 0.15 and np.all(np.abs(rs - 1.0) < 0.15):
        a = a / rs[:, None] * np.array([[1 - we_share], [we_share]])
    return a / a.sum()


def stops_sampler(pmf: dict, tail_p: float, cap: int):
    ks = sorted(pmf, key=lambda k: int(str(k).rstrip("+")))
    p = np.array([pmf[k] for k in ks], float)
    p /= p.sum()
    base = [int(str(k).rstrip("+")) for k in ks]
    plus = [str(k).endswith("+") for k in ks]

    def draw(rng: np.random.Generator) -> int:
        i = int(rng.choice(len(p), p=p))
        n = base[i]
        if plus[i]:
            n += int(rng.geometric(tail_p)) - 1
        return min(n, cap)
    return draw


def _lognorm(rng, spec: dict, size=None):
    return rng.lognormal(float(spec["mu"]), float(spec["sigma"]), size)


def _hist_mean_k(hist: dict) -> float:
    ks = np.array([int(str(k).rstrip("+")) for k in hist], float)
    n = np.array(list(hist.values()), float)
    return float((ks * n).sum() / n.sum()) if n.sum() > 0 else 2.0


def users_per_place_targets(hist: dict, n: float, n_src: float | None) -> tuple[float, float]:
    """(expected mean users per visited place, expected share of places with >= 2 users) for a
    population of `n` users, given the released histogram measured on `n_src` users. Plug-in
    binomial model: a place with k of n_src users is visited independently with q = k / n_src, so
    at n users its count is Binomial(n, q) conditioned on >= 1. Without n_src (fake params) the
    histogram is read as-is. Overlap scales with the population size, so the calibration target
    and the planted check must scale with it too."""
    ks = np.array([int(str(k).rstrip("+")) for k in hist], float)
    w = np.array(list(hist.values()), float)
    if w.sum() <= 0:
        return 2.0, 0.3
    h_mean = float((ks * w).sum() / w.sum())
    h_ge2 = float(w[ks >= 2].sum() / w.sum())
    if not n_src or n_src <= 0:
        return h_mean, h_ge2

    def model(m: float) -> tuple[float, float]:
        q = np.clip(ks / float(n_src), 1e-9, 1.0)
        pv = 1.0 - (1.0 - q) ** m                      # P(visited by >= 1 of m users)
        p1 = m * q * (1.0 - q) ** (max(m - 1, 0))      # P(exactly 1)
        denom = max(float((w * pv).sum()), 1e-9)
        return float((w * m * q).sum() / denom), float((w * np.maximum(pv - p1, 0.0)).sum() / denom)

    # the model is only used as a RATIO against its own value at n_src, so the histogram values
    # are reproduced exactly when n == n_src
    m_n, g_n = model(float(n))
    m_s, g_s = model(float(n_src))
    mean = h_mean * m_n / max(m_s, 1e-9)
    ge2 = h_ge2 * g_n / max(g_s, 1e-9)
    return max(mean, 1.0), min(max(ge2, 0.0), 1.0)


# ------------------------------------------------------------------ main
def sample_population_v2(params: dict, n_users: int, seed: int, provider=None,
                         gen: dict | None = None) -> PopulationV2:
    """Sample a population. `provider` (optional) fixes the spatial extent and the usable categories."""
    g = merged_gen_v2(gen)
    hist = params.get("places", {}).get("users_per_place_hist") or {}
    pool = g["pool_size_override"] or int(params["places"].get("pool_size", 600))
    cap_note = None
    if provider is not None:
        pl = list(provider.places())
        xy = np.array([[p.x, p.y] for p in pl])
        if gen is None or "extent_m" not in gen:
            g["extent_m"] = float(min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
        pool_cap = max(20, int(g["pool_cap_frac"] * len(pl)))
        prov_cats = {p.category for p in pl}
    else:
        pool_cap, prov_cats = 10 ** 9, None
    calibrate = g["calibrate_pool"] and bool(hist) and g["pool_size_override"] is None and g["pool_from_hist"]
    target_k = (users_per_place_targets(hist, n_users, params.get("n_users_source"))[0]
                / g["users_per_place_render_bias"]) if hist else None
    # calibrate length_scale so the latent (straight-line x nominal detour) outing length median
    # hits the fitted length_km median - the distance-decay/bump process alone lands wherever the
    # target city's extent puts it. Skipped when the caller set length_scale explicitly.
    cal_len = (g["calibrate_length"] and g["length_targeting"]
               and "length_scale" not in (gen or {}))
    target_len = (float(np.exp(float(params["outing"]["length_km"]["mu"]))) * g["length_scale"]
                  / float(g["length_render_bias"]))
    iters = g["calibrate_iters"] if (calibrate or cal_len) else 0
    if calibrate:   # start from the source pool, scaled down to this population if it is smaller
        pool = int(params["places"].get("pool_size", pool))
    pop = None
    prev_len = None
    for it in range(iters + 1):
        pool_eff = int(min(max(pool, 20), pool_cap))
        pop = _simulate(params, n_users, seed, pool_eff, g, prov_cats)
        k_gen = pop.report["mean_users_per_place"]
        med_len = pop.report["latent_length_km_median"]
        pop.report.update(pool_size=pool_eff, calibration_iter=it, target_mean_users_per_place=target_k,
                          target_length_km_median=target_len if cal_len else None,
                          length_scale=g["length_scale"], pool_capped=bool(pool > pool_cap))
        if it == iters:
            break
        done = True
        if calibrate and k_gen > 0 and abs(k_gen / target_k - 1) >= 0.08:
            pool = int(np.clip(pool * k_gen / target_k, pool / 6, pool * 6))
            done = False
        if cal_len and med_len > 0 and abs(med_len / target_len - 1) >= 0.08:
            if prev_len is not None and abs(med_len / prev_len - 1) < 0.03:
                cal_len = False       # saturated: the map geometry / distance decay cannot stretch more
            else:
                # damped multiplicative step, bounded: the bump term saturates, so a full step
                # can run the scale away without moving the realised median
                g = {**g, "length_scale": float(np.clip(g["length_scale"] * (target_len / med_len) ** 0.7,
                                                        0.1, 8.0))}
                done = False
            prev_len = med_len
        if done:
            break
    return pop


def _simulate(params: dict, n_users: int, seed: int, pool: int, g: dict, prov_cats) -> PopulationV2:
    rng = np.random.default_rng([seed, 0])
    extent = float(g["extent_m"])
    P = params["places"]
    mix = dict(P["cat_mix"])
    if prov_cats is not None:
        mix = {c: v for c, v in mix.items() if c in prov_cats} or {c: 1.0 for c in prov_cats}
    cats = sorted(mix)
    cat_p = np.array([mix[c] for c in cats], float)
    cat_p /= cat_p.sum()
    nc = len(cats)
    we_share = g["weekend_share_default"]

    # ---- latent place pool: Zipf popularity, core-biased positions, categories by cat_mix
    ids = []
    seen: set[str] = set()
    while len(ids) < pool:
        s = f"L{int(rng.integers(0, 2 ** 40)):010x}"
        if s not in seen:
            seen.add(s)
            ids.append(s)
    pcat = rng.choice(nc, size=pool, p=cat_p)
    w = np.arange(1, pool + 1, dtype=float) ** -float(P["zipf_s"])
    w = w[rng.permutation(pool)]
    uni = rng.uniform(0, extent, (pool, 2))
    core = np.clip(rng.normal(extent / 2, extent / 6, (pool, 2)), 0, extent)
    pxy = np.where((rng.random(pool) < g["core_bias"])[:, None], core, uni)
    places = {ids[i]: LatentPlace(ids[i], "other", cats[pcat[i]], float(w[i]), float(pxy[i, 0]), float(pxy[i, 1]))
              for i in range(pool)}

    # ---- time tables
    cat_prof = params.get("time", {}).get("cat_profile", {})
    flat = joint_pmf({}, we_share)
    prof = np.zeros((nc, 2, 4))
    for ci, c in enumerate(cats):
        prof[ci] = joint_pmf(cat_prof[c], we_share) if c in cat_prof else flat
    # P(daytype, tod | category) is a likelihood: keep each category's mass (no per-category rescaling)
    prof = np.maximum(prof, 0.02 * prof.max(axis=(1, 2), keepdims=True))
    TF = np.transpose(prof[pcat], (1, 2, 0))             # (daytype, tod, pool)
    start_j = joint_pmf(params["outing"]["start_pmf"], we_share)
    we_p = float(start_j[1].sum())
    tod_cond = start_j / start_j.sum(1, keepdims=True)

    # ---- groups
    n_groups = int(min(max(1, params.get("groups", {}).get("n_groups", 0)), max(1, n_users // 5))) \
        if params.get("groups", {}).get("n_groups", 0) else 0
    conc = float(params.get("groups", {}).get("concentration", 0.0))
    gcentre, gfav = [], []
    for _ in range(n_groups):
        c = rng.integers(pool)
        d = np.hypot(pxy[:, 0] - pxy[c, 0], pxy[:, 1] - pxy[c, 1])
        sc = w * np.exp(-0.5 * (d / (0.07 * extent)) ** 2) + 1e-12
        m = min(g["group_fav_size"], pool)
        gfav.append(rng.choice(pool, size=m, replace=False, p=sc / sc.sum()))
        gcentre.append(np.array(pxy[c]))
    gp = np.arange(1, n_groups + 1, dtype=float) ** -g["group_size_decay"] if n_groups else np.array([])

    # ---- shared samplers
    stops_draw = stops_sampler(params["outing"]["stops_pmf"], g["stops_tail_p"], g["max_stops"])
    mode_p = params["outing"].get("mode_pmf", {"car": 1.0})
    modes = [m for m in MODES if m in mode_p] or ["car"]
    mp = np.array([mode_p[m] for m in modes], float)
    mp /= mp.sum()
    mlf = np.array([g["mode_length_factor"].get(m, 1.0) for m in modes])
    len_shift = float((mp * np.log(mlf)).sum())
    speeds = g["speed_ms_by_mode"]
    decay = float(params["outing"].get("distance_decay", 1.0))
    revisit = float(params["users"]["revisit_rate"])
    mix_vec = cat_p

    users: list[SimUserV2] = []
    all_visits: dict[str, set[int]] = {}
    latent_len_km: list[float] = []
    for ui in range(n_users):
        ur = np.random.default_rng([seed, ui + 1, 11])
        uid = f"u{ui:04d}"
        # group + activity region
        grp = None
        if n_groups and ur.random() > g["ungrouped_frac"]:
            grp = int(ur.choice(n_groups, p=gp / gp.sum()))
        Ru = extent * float(np.clip(ur.lognormal(np.log(g["region_radius_frac"]), 0.35), 0.04, 0.5))
        if grp is not None:
            centre = np.clip(gcentre[grp] + ur.normal(0, 0.5 * Ru, 2), 0, extent)
        else:
            centre = pxy[ur.integers(pool)] + ur.normal(0, 0.2 * Ru, 2)
        taste = ur.dirichlet(g["taste_alpha"] * mix_vec + 1e-3) / np.maximum(mix_vec, 1e-3) + 0.05
        dc = np.hypot(pxy[:, 0] - centre[0], pxy[:, 1] - centre[1])
        base = w * taste[pcat] * (np.exp(-0.5 * (dc / Ru) ** 2) + 1e-3)
        gbase = (w * taste[pcat])[gfav[grp]] if grp is not None else None

        # ---- how many outings, window and time slots
        spec_n = params["users"]["outings_per_user"]
        z_n = float(ur.standard_normal())
        N = int(np.clip(round(float(np.exp(spec_n["mu"] + spec_n["sigma"] * z_n))),
                        g["min_outings"], g["max_outings"]))
        Wd = int(np.clip(round(_lognorm(ur, params["users"]["active_days"])), 1, 3 * 365))
        Wd = int(min(max(Wd, np.ceil(N / 2.5)), 3 * 365))
        horizon = max(g["horizon_days"], Wd + 30)
        d0 = int(ur.integers(0, horizon - Wd + 1))
        days = np.arange(d0, d0 + Wd)
        is_we = (days % 7) >= 5
        A = int(min(Wd, max(1, np.ceil(N / g["outings_per_active_day"]))))
        dw = np.where(is_we, we_p / max(1, is_we.sum()), (1 - we_p) / max(1, (~is_we).sum()))
        if is_we.sum() == 0 or (~is_we).sum() == 0:
            dw = np.ones(Wd)
        act = days[np.sort(ur.choice(Wd, size=A, replace=False, p=dw / dw.sum()))]
        act_we = (act % 7) >= 5
        starts_by_day: dict[int, list[float]] = {}
        slots: list[float] = []
        flat_p = start_j.ravel()
        for _ in range(N):
            for _try in range(8):
                k = int(ur.choice(8, p=flat_p))
                dti, tod = k // 4, k % 4
                pool_days = act[act_we == bool(dti)]
                if len(pool_days) == 0:
                    pool_days = act
                day = int(ur.choice(pool_days))
                hs = HOUR_SHAPE[_tod_hours(tod)]
                hour = _tod_hours(tod)[int(ur.choice(6, p=hs / hs.sum()))]
                t_loc = day * 86400 + hour * 3600 + ur.random() * 3600
                if all(abs(t_loc - o) >= g["min_gap_s"] for o in starts_by_day.get(day, ())):
                    starts_by_day.setdefault(day, []).append(t_loc)
                    slots.append(t_loc)
                    break
        slots.sort()

        # ---- spatial process, chronological
        visited: dict[int, int] = {}
        # distinct-places cap, correlated with activity (heavy users know more places); beyond it, draws
        # are forced revisits. cap_mult relaxes the cap so that revisit_rate stays the primary knob.
        spec_d = params["users"]["distinct_places"]
        rho = g["distinct_activity_corr"]
        z_d = rho * z_n + np.sqrt(1 - rho ** 2) * float(ur.standard_normal())
        dcap = max(2, int(round(g["distinct_cap_mult"] * float(np.exp(spec_d["mu"] + spec_d["sigma"] * z_d)))))
        rev = {"n": 0, "r": 0}            # stop-draw revisit controller state (one per user)
        outs: list[SimOuting] = []
        prev_end: int | None = None
        prev_t_end = -1e18
        for t_loc in slots:
            mi = int(ur.choice(len(modes), p=mp))
            mode = modes[mi]
            sp = speeds[mode] * float(np.exp(ur.normal(0, 0.15)))
            k = stops_draw(ur)
            leg_target = None
            if g["length_targeting"]:
                L = float(np.exp(ur.normal(params["outing"]["length_km"]["mu"],
                                           params["outing"]["length_km"]["sigma"]) + np.log(mlf[mi]) - len_shift))
                L = max(L * 1000.0 * g["length_scale"], 300.0)
                cap = g["mode_length_cap_km"].get(mode)
                if cap:
                    L = min(L, float(cap) * 1000.0)
                leg_target = L / (k + 1) / g["detour_nominal"]
            t_loc = max(t_loc, prev_t_end + 7200 + 600)
            t_now = t_loc
            ctx = _Ctx(ur, pxy, pcat, base, TF, gfav[grp] if grp is not None else None, gbase, visited,
                       revisit, conc if grp is not None else 0.0, dcap, decay, g, leg_target, rev)
            seq: list[int] = []
            dwell: list[int] = []
            # start place
            if prev_end is not None and ur.random() < g["continue_prob"]:
                s0 = prev_end
            else:
                s0 = ctx.pick(None, _dt_tod(t_now), forbid=(), role="start")
            seq.append(s0)
            for _ in range(k):
                nxt = ctx.pick(seq[-1], _dt_tod(t_now), forbid=(seq[-1],), role="stop")
                rev["n"] += 1
                rev["r"] += int(nxt in visited or nxt in seq)
                # estimate arrival using the straight-line distance to the picked place
                dist = float(np.hypot(*(pxy[nxt] - pxy[seq[-1]]))) * g["detour_nominal"]
                t_now += dist / sp
                dw_s = int(np.clip(np.exp(ur.normal(params["outing"]["stop_dwell_min"]["mu"],
                                                    params["outing"]["stop_dwell_min"]["sigma"])) * 60.0,
                                   g["dwell_min_s"], g["dwell_max_s"]))
                seq.append(nxt)
                dwell.append(dw_s)
                t_now += dw_s
            if k >= 1 and ur.random() < g["roundtrip_prob"]:
                end = seq[0]
            else:
                end = ctx.pick(seq[-1], _dt_tod(t_now), forbid=(seq[-1],), role="end")
            seq.append(end)
            t_now += float(np.hypot(*(pxy[end] - pxy[seq[-2]]))) * g["detour_nominal"] / sp
            latent_len_km.append(sum(float(np.hypot(*(pxy[b] - pxy[a])))
                                     for a, b in zip(seq, seq[1:])) * g["detour_nominal"] / 1000.0)
            for pi in seq:
                visited[pi] = visited.get(pi, 0) + 1
            prev_end = end
            prev_t_end = t_now
            outs.append(SimOuting(int(round(T0_LOCAL + t_loc)), mode, sp / speeds[mode],
                                  [ids[i] for i in seq], dwell))
        users.append(SimUserV2(uid, grp, (float(centre[0]), float(centre[1])), float(Ru), ui, outs,
                               (int(days[0]), int(days[-1]))))
        all_visits[uid] = set(visited)

    # ---- co-travel: followers copy a group-mate's outings (same places, times, mode)
    truth = GroundTruthV2()
    by_uid = {u.uid: u for u in users}
    detail: list[tuple[str, int, str, int]] = []
    crng = np.random.default_rng([seed, 99])
    by_group: dict[int, list[SimUserV2]] = {}
    for u in users:
        if u.group is not None:
            by_group.setdefault(u.group, []).append(u)
    taken: set[str] = set()
    for gid, ms in sorted(by_group.items()):
        ms = [u for u in ms if len(u.outings) >= 3]
        crng.shuffle(ms)
        for lead, fol in zip(ms[0::2], ms[1::2]):
            if crng.random() >= 2 * g["cotravel_user_frac"] or lead.uid in taken or fol.uid in taken:
                continue
            n_c = max(1, int(round(g["cotravel_outing_frac"] * len(lead.outings))))
            pick = np.sort(crng.choice(len(lead.outings), size=min(n_c, len(lead.outings)), replace=False))
            keep = [o for o in fol.outings]
            new_idx = []
            for li in pick:
                lo = lead.outings[int(li)]
                keep = [o for o in keep if abs(o.t_start - lo.t_start) > g["min_gap_s"]]
                keep.append(SimOuting(lo.t_start, lo.mode, lo.speed_f, list(lo.places), list(lo.dwell_s),
                                      (lead.uid, int(li))))
                new_idx.append(int(li))
            keep.sort(key=lambda o: o.t_start)
            fol.outings = keep
            fol.leader = lead.uid
            taken.update((lead.uid, fol.uid))
            for o_i, o in enumerate(fol.outings):
                if o.copy_of and o.copy_of[0] == lead.uid:
                    detail.append((lead.uid, o.copy_of[1], fol.uid, o_i))
    truth.co_traveled_detail = detail
    truth.co_traveled = [(a, b, i) for a, i, b, _ in detail]
    truth.groups = {int(gid): [u.uid for u in users if u.group == gid] for gid in range(n_groups)
                    if any(u.group == gid for u in users)}
    truth.user_group = {u.uid: u.group for u in users}
    truth.group_places = {int(gid): [ids[i] for i in gfav[gid]] for gid in truth.groups}

    # latent-level summary
    cnt: dict[str, int] = {}
    for vs in all_visits.values():
        for i in vs:
            cnt[ids[i]] = cnt.get(ids[i], 0) + 1
    report = {"n_visited_places": len(cnt),
              "mean_users_per_place": float(np.mean(list(cnt.values()))) if cnt else 0.0,
              "latent_length_km_median": float(np.median(latent_len_km)) if latent_len_km else 0.0,
              "group_places_latent": dict(truth.group_places), "n_groups": len(truth.groups), "n_cotravel_pairs": len({(a, b) for a, _, b, _ in detail})}
    return PopulationV2(users, places, truth, params, g, extent, seed, report)


def _tod_hours(tod: int) -> np.ndarray:
    return np.arange(tod * 6, tod * 6 + 6)


def _dt_tod(t_loc: float) -> tuple[int, int]:
    day = int(t_loc // 86400)
    hour = int((t_loc % 86400) // 3600)
    return (1 if day % 7 >= 5 else 0), hour // 6


class _Ctx:
    """Per-outing place chooser bound to one user's state."""

    def __init__(self, rng, pxy, pcat, base, TF, gidx, gbase, visited, revisit, conc, dcap, decay, g,
                 leg_target, rev=None):
        self.rng, self.pxy, self.pcat, self.base, self.TF = rng, pxy, pcat, base, TF
        self.gidx, self.gbase, self.visited = gidx, gbase, visited
        self.revisit, self.conc, self.dcap, self.decay, self.g = revisit, conc, dcap, decay, g
        self.leg_target = leg_target
        self.rev = rev                    # {"n", "r"}: realised stop-revisit counters of this user

    def _dist_factor(self, idx, prev) -> np.ndarray | float:
        if prev is None:
            return 1.0
        g = self.g
        d = np.hypot(self.pxy[idx, 0] - self.pxy[prev, 0], self.pxy[idx, 1] - self.pxy[prev, 1]) + g["d0_m"]
        f = d ** -self.decay
        if self.leg_target is not None:
            f = f * np.exp(-0.5 * (np.log(d / (self.leg_target + g["d0_m"])) / g["length_bump_sigma"]) ** 2)
        return f

    def pick(self, prev: int | None, dt_tod: tuple[int, int], forbid=(), role: str = "mid") -> int:
        rng, vis = self.rng, self.visited
        tf = self.TF[dt_tod[0], dt_tod[1]]
        forced = len(vis) >= self.dcap
        # stop draws target the released revisit rate with a proportional controller on the
        # realised share; starts and ends lean to familiar places (public modelling choice)
        if role == "stop" and self.rev is not None and self.rev["n"] >= 8:
            share = self.rev["r"] / self.rev["n"]
            p_rev = float(min(max(self.revisit + (self.revisit - share), 0.0), 1.0))
        elif role == "end":
            p_rev = float(self.g["end_revisit_prob"])
        elif role == "start":
            p_rev = float(self.g["start_revisit_prob"])
        else:
            p_rev = self.revisit
        if vis and (forced or rng.random() < p_rev):
            vi = np.fromiter(vis.keys(), int, len(vis))
            vc = np.fromiter(vis.values(), float, len(vis))
            s = vc ** self.g["revisit_pref"] * tf[vi] * self._dist_factor(vi, prev)
            s = np.where(np.isin(vi, forbid), 0.0, s)
            if s.sum() > 0:
                return int(vi[rng.choice(len(vi), p=s / s.sum())])
        for _try in range(2):
            if self.gidx is not None and rng.random() < self.conc:
                gi = self.gidx
                s = self.gbase * tf[gi] * self._dist_factor(gi, prev)
                s = np.where(np.isin(gi, list(vis) + list(forbid)), 0.0, s)
                if s.sum() > 0:
                    return int(gi[rng.choice(len(gi), p=s / s.sum())])
            s = self.base * tf * self._dist_factor(np.arange(len(self.base)), prev)
            if vis:
                s[np.fromiter(vis.keys(), int, len(vis))] = 0.0
            s[list(forbid)] = 0.0
            if s.sum() > 0:
                return int(rng.choice(len(s), p=s / s.sum()))
            break
        # degenerate fall-back: any non-forbidden place
        s = np.ones(len(self.base))
        s[list(forbid)] = 0.0
        return int(rng.choice(len(s), p=s / s.sum()))


def latent_view(pop: PopulationV2) -> SimpleNamespace:
    """Adapter so generate.teleport.ground_places (written for v1) can ground v2 latent places.
    It only needs the set of used latent ids, the latent places, the extent and a truth object."""
    used = sorted({p for u in pop.users for o in u.outings for p in o.places})
    shim_user = SimpleNamespace(visits=[], home=used[0], work=None, others=used)
    return SimpleNamespace(users=[shim_user], places=pop.places, extent_m=pop.extent_m, truth=pop.truth)

"""Generator v2, render layer: route planted outings on the MapProvider and emit raw-style bursty GPS.

Logging happens only while an outing is under way (plus a short pre/post sit): no overnight stays. Sensor
regime: when the artifact released usable sensor tables (params["sensor"], fitted by outside/sensors.py),
the per-user sampling period, the per-user gap rate (converted from per-logged-hour to wall-clock), every
gap length and the in-stop positional spread are drawn from those DP histograms by inverse-CDF sampling;
per-mode speeds and the sensor-noise scale come from the same fit (injected into `gen` by pipeline_v2).
Receiver noise is an AR(1) process per axis (marginal sd `sensor_noise_m`, time constant `noise_tau_s`),
not white noise, and in-stop points drift slowly around the place (`stop_drift_tau_s`) with a per-stop
spread drawn from the stop-jitter histogram: real GPS error is strongly autocorrelated, and white noise
at a 1-5 s sampling period fabricated metres of travel per step (speed, heading-change and outing-length
artefacts; PLAN step 5). Any table that was mostly noise falls back to the hand-set defaults below (EDA
section 5): per-user sampling period from {1,2,3,4,5,~13} s (median 2 s), signal-loss gaps (~5/h,
lognormal median 210 s, heavy tail), indoor signal loss at some stops, stop jitter, mode-dependent speeds
with stop-and-go. All knobs are overridable via `gen` (sensitivity studies; `use_sensor_hists: false`
disables every histogram draw). Timestamps are UTC; hour/day logic upstream is local time UTC+8.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from geolife.transfer.types import GpsPoint, Outing, Visit
from geolife.transfer.maps import MapProvider, Route
from geolife.outside.population_v2 import GroundTruthV2, PopulationV2, merged_gen_v2
from geolife.outside.sensors import HistSampler

SENSOR_DEFAULTS: dict = {
    "outing_gap_s": 7200.0, "sample_periods_s": [1, 2, 3, 4, 5, 13], "sample_period_w": [32, 50, 12, 5, 44, 5],
    "sample_interval_s": None,            # fixed period for everybody when set
    "interval_jitter": 0.15,
    "gap_rate_per_hr": 5.0, "gap_median_s": 210.0, "gap_sigma": 1.7, "gap_max_s": 3600.0,
    "gap_max_hist_s": 7100.0,             # cap of gap draws from the DP histogram (outing gap is 7200)
    "gap_mean_s": None,                   # v1 name; if set it is converted to the lognormal median
    "stop_signal_loss_prob": 0.25,        # whole dwell unlogged (indoors); fallback path only
    "stop_signal_loss_prob_hist": 0.0,    # with the DP gap tables, indoor loss is already in gap_len/gap_rate
    "sensor_noise_m": 6.0, "outlier_prob": 0.003, "outlier_scale": 8.0, "dwell_jitter_m": 25.0,
    # GPS error is strongly autocorrelated (the loggers emit Kalman-smoothed tracks: the real
    # heading change is 3.5 deg median per step): AR(1) per axis with this time constant and the
    # sensor_noise_m marginal sd. White 6 m noise at a 2 s sampling period added ~10 m of fake
    # travel per step (~5 m/s of fake speed, 121 deg median heading change, tens of km of fake
    # outing length); correlated noise keeps the marginal error without the fake movement.
    "noise_tau_s": 600.0,
    "stop_drift_tau_s": 900.0,            # slow in-stop drift (indoor multipath), same AR(1) form
    "pre_sit_s": 45.0, "post_sit_s": 60.0,
    "speed_sd_leg": 0.1, "stopgo_sd": 0.35, "stopgo_block_s": 30.0,
    "detour_candidates": 3, "detour_min_leg_m": 600.0, "compute_cells": True, "cell_res": 8, "trav_res": 9,
    "use_sensor_hists": True,             # draw from params["sensor"] histograms when available
}


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """(2, n) stationary unit-variance AR(1) noise per axis."""
    from scipy.signal import lfilter
    w = rng.standard_normal((2, n + 1))
    e, _ = lfilter([float(np.sqrt(max(1.0 - rho * rho, 1e-12)))], [1.0, -rho],
                   w[:, 1:], axis=1, zi=rho * w[:, :1])
    return e


def sensor_samplers(params: dict, g: dict) -> dict[str, HistSampler]:
    """Inverse-CDF samplers of the usable released sensor histograms ({} when disabled/absent)."""
    if not g.get("use_sensor_hists", True):
        return {}
    hists = (params.get("sensor") or {}).get("hists") or {}
    return {k: HistSampler(v["edges"], v["pmf"]) for k, v in hists.items()
            if isinstance(v, dict) and "edges" in v}


def sensor_gen(gen: dict | None) -> dict:
    g = merged_gen_v2(None)
    g.update({k: v for k, v in SENSOR_DEFAULTS.items()})
    for k, v in (gen or {}).items():
        if v is None:
            continue
        g[k] = {**g[k], **v} if isinstance(v, dict) and isinstance(g.get(k), dict) else v
    if (gen or {}).get("gap_mean_s") and not (gen or {}).get("gap_median_s"):
        g["gap_median_s"] = float(gen["gap_mean_s"]) / np.exp(g["gap_sigma"] ** 2 / 2)
    return g


@dataclass
class RenderResultV2:
    traces: dict[str, list[GpsPoint]]
    truth: GroundTruthV2
    outings: dict[str, list[Outing]]       # planted outings as the tokenizer should recover them
    meta: dict = field(default_factory=dict)


def _times(t0: float, t1: float, period: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    n = int((t1 - t0) / period * 1.2) + 16
    steps = period * np.clip(1 + jitter * rng.standard_normal(n), 0.3, None)
    t = t0 + np.cumsum(steps)
    while t[-1] < t1:
        t = np.concatenate([t, t[-1] + np.cumsum(period * np.clip(1 + jitter * rng.standard_normal(n), 0.3, None))])
    return np.concatenate([[t0], t[t <= t1]])


def _detoured(provider, a: str, b: str, ratio_excess: float, k: int, min_leg, tree, ids,
              rng: np.random.Generator) -> Route:
    base = provider.route(a, b)
    if k <= 0 or ratio_excess < 0.05 or base.length_m < min_leg:
        return base
    target = 1.0 + ratio_excess
    mid = (base.polyline[0] + base.polyline[-1]) / 2
    near = tree.query_ball_point(mid, 0.45 * base.length_m)
    if len(near) < 2:
        return base
    best, best_err = base, abs(1.0 - target)
    for v in rng.choice(near, size=min(k, len(near)), replace=False):
        r1, r2 = provider.route(a, ids[v]), provider.route(ids[v], b)
        L = r1.length_m + r2.length_m
        err = abs(L / base.length_m - target)
        if err < best_err:
            best_err, best = err, Route(r1.segments + r2.segments, L, np.vstack([r1.polyline, r2.polyline[1:]]))
    return best


def _cells_along(poly: np.ndarray, provider, res: int, step_m: float = 60.0) -> list[str]:
    import h3
    d = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    s = np.arange(0.0, cum[-1] + step_m, step_m)
    s = np.minimum(s, cum[-1])
    x, y = np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])
    lat, lon = provider.to_latlon(x, y)
    out, seen = [], set()
    for a, b in zip(lat.tolist(), lon.tolist()):
        c = h3.latlng_to_cell(a, b, res)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def render_v2(pop: PopulationV2, mapping: dict[str, str], provider: MapProvider, gen: dict | None = None,
              seed: int = 0) -> RenderResultV2:
    import h3
    g = sensor_gen({**pop.gen, **(gen or {})})
    smp = sensor_samplers(pop.params, g)
    plist = list(provider.places())
    places = {p.id: p for p in plist}
    ids = [p.id for p in plist]
    tree = cKDTree(np.array([[p.x, p.y] for p in plist]))
    det = pop.params.get("route", {}).get("detour_ratio", {"mu": np.log(0.12), "sigma": 0.8})
    cell_cache: dict[str, str] = {}

    def cell_of(pid: str) -> str:
        c = cell_cache.get(pid)
        if c is None:
            lat, lon = provider.to_latlon(np.array([places[pid].x]), np.array([places[pid].y]))
            c = cell_cache[pid] = h3.latlng_to_cell(float(lat[0]), float(lon[0]), g["cell_res"])
        return c

    plans: dict[tuple[str, int], dict] = {}
    traces: dict[str, list[GpsPoint]] = {}
    outings_out: dict[str, list[Outing]] = {}
    visits_out: dict[str, list[Visit]] = {}
    seg_users: dict[str, set] = {}
    final_idx: dict[tuple[str, int], tuple[str, int]] = {}
    idx = {u.uid: i for i, u in enumerate(pop.users)}
    stats = {"n_outings": 0, "n_points": 0, "n_routes": 0}

    def plan_outing(u, oi, o, rng) -> dict:
        """Route every leg, fix timestamps of the visits."""
        pls = [mapping[p] for p in o.places]
        legs, t = [], float(o.t_start)
        speed_mode = g["speed_ms_by_mode"][o.mode] * o.speed_f
        vis = [Visit(pls[0], places[pls[0]].category, cell_of(pls[0]), int(t), 0, "start")]
        segs: list[tuple] = []
        for j in range(len(pls) - 1):
            a, b = pls[j], pls[j + 1]
            if a == b:      # grounding merged two latent places (or an immediate loop): no leg
                segs.append((t, t, None, 0.0))
            else:
                excess = max(0.0, float(rng.lognormal(det["mu"], det["sigma"])))
                r = _detoured(provider, a, b, excess, int(g["detour_candidates"]), g["detour_min_leg_m"],
                              tree, ids, rng)
                stats["n_routes"] += 1
                v = speed_mode * float(np.exp(rng.normal(0, g["speed_sd_leg"])))
                dur = r.length_m / max(v, 0.3)
                segs.append((t, t + dur, r, dur))
                t += dur
            if j < len(o.dwell_s):
                dw = float(o.dwell_s[j])
                vis.append(Visit(b, places[b].category, cell_of(b), int(t), int(dw), "stop"))
                segs.append((t, t + dw, b, -1.0))
                t += dw
        vis.append(Visit(pls[-1], places[pls[-1]].category, cell_of(pls[-1]), int(t), 0, "end"))
        return {"places": pls, "segs": segs, "t0": float(o.t_start), "t1": t, "visits": vis, "mode": o.mode}

    def emit(u, plan, period, gap_rate, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0, t1 = plan["t0"] - g["pre_sit_s"] * rng.random(), plan["t1"] + g["post_sit_s"] * rng.random()
        t = _times(t0, t1, period, g["interval_jitter"], rng)
        x, y = np.empty(len(t)), np.empty(len(t))
        keep = np.ones(len(t), bool)
        first, last = places[plan["places"][0]], places[plan["places"][-1]]
        for lo, hi, what, dur in [(t0, plan["t0"], "sit0", 0), *[(a, b, w, d) for a, b, w, d in plan["segs"]],
                                  (plan["t1"], t1, "sit1", 0)]:
            m = (t >= lo) & (t < hi if hi < t1 else t <= hi)
            if not m.any():
                continue
            if what == "sit0" or what == "sit1" or (dur == -1.0) or what is None:
                p = first if what == "sit0" else last if what == "sit1" else (places[what] if what else first)
                mi = int(m.sum())
                if "stop_jitter" in smp:
                    # measured in-stop displacement: one spread scale per stop from the DP
                    # histogram (mean Rayleigh radius -> per-axis sd), walked as a slow AR(1)
                    # drift - per-point independent draws made in-stop points jump tens of
                    # metres between consecutive samples (fake speed at stops)
                    s_ax = max(float(smp["stop_jitter"].draw(rng)), 1.0) / 1.2533
                    rho = float(np.exp(-period / max(float(g["stop_drift_tau_s"]), period)))
                    e = _ar1(rng, mi, rho)
                    x[m] = p.x + s_ax * e[0]
                    y[m] = p.y + s_ax * e[1]
                else:
                    x[m] = p.x + g["dwell_jitter_m"] * rng.standard_normal(mi)
                    y[m] = p.y + g["dwell_jitter_m"] * rng.standard_normal(mi)
                loss_p = (g["stop_signal_loss_prob_hist"] if "gap_len" in smp
                          else g["stop_signal_loss_prob"])
                if dur == -1.0 and rng.random() < loss_p:
                    keep[m] = False
            else:
                r: Route = what
                tau = t[m] - lo
                nb = int(dur / g["stopgo_block_s"]) + 1
                bounds = np.arange(nb + 1) * g["stopgo_block_s"]
                sp = np.clip(1 + g["stopgo_sd"] * rng.standard_normal(nb), 0.1, 3.0)
                prog = np.concatenate([[0.0], np.cumsum(sp * g["stopgo_block_s"])])
                frac = np.interp(tau, bounds, prog) / np.interp(dur, bounds, prog)
                seg = np.hypot(*np.diff(r.polyline, axis=0).T)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                s = np.clip(frac, 0, 1) * cum[-1]
                x[m] = np.interp(s, cum, r.polyline[:, 0])
                y[m] = np.interp(s, cum, r.polyline[:, 1])
        n = len(t)
        tau = float(g["noise_tau_s"] or 0.0)
        if tau > 0:                              # autocorrelated receiver error (see SENSOR_DEFAULTS)
            e = _ar1(rng, n, float(np.exp(-period / tau)))
            x = x + g["sensor_noise_m"] * e[0]
            y = y + g["sensor_noise_m"] * e[1]
        else:
            x = x + g["sensor_noise_m"] * rng.standard_normal(n)
            y = y + g["sensor_noise_m"] * rng.standard_normal(n)
        out = rng.random(n) < g["outlier_prob"]
        if out.any():
            sc = g["sensor_noise_m"] * g["outlier_scale"]
            x[out] += sc * rng.standard_normal(int(out.sum()))
            y[out] += sc * rng.standard_normal(int(out.sum()))
        # signal-loss gaps over the whole outing (also hide parts of stops)
        n_gaps = rng.poisson(gap_rate * (t1 - t0) / 3600.0)
        if n_gaps:
            st = rng.uniform(t0, t1, n_gaps)
            if "gap_len" in smp:                # measured gap-length distribution
                du = np.minimum(smp["gap_len"].draw(rng, n_gaps), g["gap_max_hist_s"])
            else:
                du = np.minimum(rng.lognormal(np.log(g["gap_median_s"]), g["gap_sigma"], n_gaps), g["gap_max_s"])
            for a, d in zip(st, du):
                keep &= ~((t >= a) & (t < a + d))
        return t[keep], x[keep], y[keep]

    def run_user(u) -> None:
        ui = idx[u.uid]
        rng = np.random.default_rng([seed, ui, 7])
        if g["sample_interval_s"]:                       # explicit override wins
            period = float(g["sample_interval_s"])
        elif "interval_user" in smp:                     # measured per-user heterogeneity (1 s vs 5 s users)
            period = float(np.clip(smp["interval_user"].draw(rng), 1.0, 60.0))
        else:
            period = float(rng.choice(g["sample_periods_s"],
                                      p=np.array(g["sample_period_w"], float) / sum(g["sample_period_w"])))
        gap_rate = (float(np.clip(smp["gap_rate"].draw(rng), 0.0, 30.0)) if "gap_rate" in smp
                    else float(g["gap_rate_per_hr"]))
        if "gap_len" in smp and gap_rate > 0:
            # the DP rate is gaps per LOGGED hour; the Poisson below runs on wall-clock outing
            # time, so convert with the expected duty cycle
            gap_rate /= 1.0 + gap_rate * smp["gap_len"].mean() / 3600.0
        T, X, Y = [], [], []
        outs, visits = [], []
        # routed plans in time order; planning used straight-line estimates, so enforce the outing gap here:
        # shift an own outing later when it would start too soon, drop own outings that a fixed co-travel
        # copy would overlap (copies keep the leader's exact timing).
        gap = float(g["outing_gap_s"]) + 900.0                    # outing definition (2 h) plus margin
        kept: list[tuple[int, dict]] = []
        for oi, o in enumerate(u.outings):
            if o.copy_of:
                plan = plans[o.copy_of]
                while kept and plan["t0"] < kept[-1][1]["t1"] + gap and u.outings[kept[-1][0]].copy_of is None:
                    kept.pop()
                if kept and plan["t0"] < kept[-1][1]["t1"] + gap:
                    continue
            else:
                prev_end = kept[-1][1]["t1"] if kept else -1e18
                o2 = o if o.t_start >= prev_end + gap else replace(o, t_start=int(prev_end + gap) + 1)
                plan = plan_outing(u, oi, o2, np.random.default_rng([seed, ui, oi, 13]))
                plans[(u.uid, oi)] = plan
            kept.append((oi, plan))
        for oi, plan in kept:
            if u.outings[oi].copy_of:
                lead_uid, li = u.outings[oi].copy_of
                final_idx[(lead_uid, li)] = (u.uid, len(outs))
            t, x, y = emit(u, plan, period, gap_rate, rng)
            T.append(t); X.append(x); Y.append(y)
            length = float(sum(r.length_m for _, _, r, _ in plan["segs"] if isinstance(r, Route)))
            cells: list[str] = []
            segids: list[str] = []
            for _, _, r, _ in plan["segs"]:
                if isinstance(r, Route):
                    segids.extend(s for s in r.segments if not segids or segids[-1] != s)
                    if g["compute_cells"]:
                        for c in _cells_along(r.polyline, provider, g["trav_res"]):
                            if c not in cells:
                                cells.append(c)
            for s in set(segids):
                seg_users.setdefault(s, set()).add(u.uid)
            outs.append(Outing(int(plan["t0"]), int(plan["t1"]), plan["mode"], length, tuple(plan["visits"]),
                               tuple(cells), tuple(segids)))
            visits.extend(plan["visits"])
        outings_out[u.uid] = outs
        visits_out[u.uid] = visits
        stats["n_outings"] += len(outs)
        if T:
            t, x, y = np.concatenate(T), np.concatenate(X), np.concatenate(Y)
            ti, ix = np.unique(np.rint(t).astype(np.int64), return_index=True)
            lat, lon = provider.to_latlon(x[ix], y[ix])
            traces[u.uid] = [GpsPoint(int(a), float(b), float(c)) for a, b, c in zip(ti, lat, lon)]
        else:
            traces[u.uid] = []
        stats["n_points"] += len(traces[u.uid])

    for u in pop.users:                      # leaders (and independents) first
        if not any(o.copy_of for o in u.outings):
            run_user(u)
    for u in pop.users:
        if any(o.copy_of for o in u.outings):
            run_user(u)
    traces = {u.uid: traces[u.uid] for u in pop.users}

    truth = pop.truth
    truth.visits = {u.uid: visits_out[u.uid] for u in pop.users}
    truth.latent_to_map = dict(mapping)
    truth.co_traveled_detail = sorted((lead, li, fol, fi) for (lead, li), (fol, fi) in final_idx.items())
    truth.co_traveled = [(lead, fol, li) for lead, li, fol, _ in truth.co_traveled_detail]
    users_at: dict[str, set] = {}
    cnt: dict[str, dict[str, int]] = {}
    for uid, vs in truth.visits.items():
        c = cnt[uid] = {}
        for v in vs:
            c[v.place_id] = c.get(v.place_id, 0) + 1
            users_at.setdefault(v.place_id, set()).add(uid)
    truth.shared_places = {p: sorted(s) for p, s in sorted(users_at.items()) if len(s) >= 2}
    truth.favourite_places = {uid: [p for p, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 3]
                              for uid, c in cnt.items()}
    truth.shared_segments = {s: sorted(v) for s, v in sorted(seg_users.items()) if len(v) >= 2}
    truth.group_places = {gid: sorted({mapping[p] for p in ps if p in mapping})
                          for gid, ps in pop.report["group_places_latent"].items()}
    return RenderResultV2(traces, truth, outings_out, stats)

"""Evaluation harness for M v2 (outings): planted-truth recovery on generate_v2 data, sensitivity,
split-half reliability, and the label-free real-data stage.

Run: python -m geolife.eval.m.outings_transfer [--quick] [--out geolife/eval/reports/m-v2.json]
(the report geolife/eval/reports/m-v2.md is written by hand from this output).
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

import networkx as nx
from shapely.geometry import LineString

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline_v2 import generate_v2
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.transfer.types import LOCAL_UTC_OFFSET_H, UserOutings
from geolife.eval.m.outings import (NPROF, M2Config, M2Result, MapView, _summ, analyze, distribution_stats,
                                    place_merge_map, run_m2, tv_hist)


# ---- toy-map helpers and ARI (moved here from the retired v1 eval/m/transfer.py)

def places_table(provider) -> PlaceTable:
    ps = provider.places()
    lat, lon = provider.to_latlon(np.array([p.x for p in ps]), np.array([p.y for p in ps]))
    return PlaceTable([p.id for p in ps], [p.category for p in ps], lat, lon,
                      [p.weight for p in ps])


def grid_graph(provider: GridCityProvider) -> nx.Graph:
    """Public road graph of the toy city in build_graph format (node ids, segment_id, length, geom)."""
    lat, lon = provider.to_latlon(provider.node_xy[:, 0], provider.node_xy[:, 1])
    G = nx.Graph()
    for i, (la, lo) in enumerate(zip(lat, lon)):
        G.add_node(i, lat=float(la), lon=float(lo))
    for (a, b), sid in provider._seg.items():
        if a < b:
            ln = float(np.hypot(*(provider.node_xy[a] - provider.node_xy[b])))
            G.add_edge(a, b, segment_id=sid, length=ln, class_="residential",
                       geom=LineString([(G.nodes[a]["lon"], G.nodes[a]["lat"]),
                                        (G.nodes[b]["lon"], G.nodes[b]["lat"])]))
    return G


def _ari(labels_a: dict, labels_b: dict) -> float:
    """Adjusted Rand index of two labelings over the same keys."""
    from collections import Counter
    keys = sorted(labels_a)
    a = [labels_a[k] for k in keys]
    b = [labels_b[k] for k in keys]
    n = len(keys)
    comb2 = lambda x: x * (x - 1) / 2
    cont = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    idx = sum(comb2(v) for v in cont.values())
    ia, ib = sum(comb2(v) for v in sa.values()), sum(comb2(v) for v in sb.values())
    exp = ia * ib / comb2(n) if n > 1 else 0
    mx = 0.5 * (ia + ib)
    return (idx - exp) / (mx - exp) if mx != exp else 1.0

_LEGACY_AMS = Path("C:/Users/NaomiD/Documents/Projects/hackaton-2026/.claude/worktrees/agent-a63109956b0e67a7c/geolife/cache")
AMS_CACHE = next((c for c in (Path("geolife/outside/cache"), _LEGACY_AMS) if (c / "places_amsterdam.parquet").exists()),
                 Path("geolife/outside/cache"))
AMS_BBOX = (4.85, 52.33, 4.95, 52.41)

# dev draw: sensor/noise as in the generator tests (fast); held-out: new seed, different map, noisier / gappier sensor,
# shorter active windows, more unlogged stops, fewer outings per user.
DEV_GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 120, "sample_interval_s": 10}
HELD_GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 120, "sample_interval_s": 20,
            "sensor_noise_m": 15.0, "gap_rate_per_hr": 8.0, "stop_signal_loss_prob": 0.4, "dwell_jitter_m": 35.0}


def toy_provider(seed: int = 1):
    return GridCityProvider(n=40, spacing=250, n_places=6000, seed=seed)


# dev-sparse draw for the relation-rule study ONLY (step 8 task 2): sparser than dev but deliberately
# different from the held-out draw (map seed 2 vs 7, gen seed 13 vs 4242, 15 s / 10 m / 6 gaps/h
# vs 20 s / 15 m / 8 gaps/h, milder activity cut). The held-out draw is never used for rule selection.
SPARSE_GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 120, "sample_interval_s": 15,
              "sensor_noise_m": 10.0, "gap_rate_per_hr": 6.0, "stop_signal_loss_prob": 0.35}

# the pre-step-8 relation rule (m-v2.md as of D17): BH over ALL pairs at FDR 0.2, >=1 shared item
OLD_RULE = {"rel_universe": "all", "rel_alpha": 0.2, "rel_min_shared": 1}


def sparse_params() -> dict:
    p = copy.deepcopy(fake_params_v2())
    p["users"]["active_days"]["mu"] = float(np.log(45.0))
    p["users"]["outings_per_user"]["mu"] = float(np.log(28.0))
    p["groups"]["concentration"] = 0.5
    return p


def held_params() -> dict:
    p = copy.deepcopy(fake_params_v2())
    p["users"]["active_days"]["mu"] = float(np.log(40.0))
    p["users"]["outings_per_user"]["mu"] = float(np.log(22.0))
    p["groups"]["concentration"] = 0.3
    return p


def make_mv(provider, with_graph: bool) -> MapView:
    return MapView(places_table(provider), grid_graph(provider) if with_graph else None)


def amsterdam_provider():
    if not (AMS_CACHE / "places_amsterdam.parquet").exists():
        return None
    from geolife.outside.map.provider import OvertureProvider
    return OvertureProvider(AMS_BBOX, cache_dir=AMS_CACHE, max_poi=15000)


# ------------------------------------------------------------------------------------ truth

def _observed_planted(g, tol_s: int = 300):
    """Per user: list of (outing_index, planted Outing, observed visits) for planted outings that have a
    tokenized counterpart is decided later; here only 'observed' visits: start/end always, a stop only if the
    trace has >= 2 points inside its dwell (otherwise the stop was unlogged and cannot be detected)."""
    res = g.result
    out = {}
    for uid, outs in res.outings.items():
        ts = [p.t for p in res.traces[uid]]
        rows = []
        for o in outs:
            vis = []
            for v in o.visits:
                if v.kind == "stop":
                    a = bisect.bisect_left(ts, v.t_arrive)
                    b = bisect.bisect_right(ts, v.t_arrive + v.dwell_s)
                    if b - a < 2:
                        continue
                vis.append(v)
            rows.append((o, vis))
        out[uid] = rows
    return out


def matched_truth(g, res: M2Result, tol_s: int = 300):
    """Planted outings that M also found (start time within tol_s), restricted to the users M kept.
    -> {user: [(planted Outing, observed visits)]}"""
    obs = _observed_planted(g)
    out = {}
    for uo in res.outings:
        starts = np.array(sorted(o.t_start for o in uo.outings))
        rows = []
        for o, vis in obs.get(uo.user, []):
            i = np.searchsorted(starts, o.t_start)
            near = min((abs(starts[j] - o.t_start) for j in (i - 1, i) if 0 <= j < len(starts)), default=1e18)
            if near <= tol_s:
                rows.append((o, vis))
        out[uo.user] = rows
    return out


def remap_truth(mt, merge: dict):
    """Express truth place ids at M's map resolution (neighbouring map places merged by place_merge_map)."""
    if not merge:
        return mt
    return {u: [(o, [replace(v, place_id=merge.get(v.place_id, v.place_id)) for v in vis]) for o, vis in rows]
            for u, rows in mt.items()}


def truth_pois(mt, min_outings: int, min_days: int) -> dict[str, set]:
    out = {}
    for u, rows in mt.items():
        outs, days = defaultdict(set), defaultdict(set)
        for oi, (o, vis) in enumerate(rows):
            for v in vis:
                outs[v.place_id].add(oi)
                days[v.place_id].add(int((v.t_arrive + LOCAL_UTC_OFFSET_H * 3600) // 86400))
        out[u] = {p for p in outs if len(outs[p]) >= min_outings and len(days[p]) >= min_days}
    return out


def truth_place_users(mt) -> dict[str, set]:
    d = defaultdict(set)
    for u, rows in mt.items():
        for o, vis in rows:
            for v in vis:
                d[v.place_id].add(u)
    return d


def truth_trav_users(mt, kind: str) -> dict[str, set]:
    d = defaultdict(set)
    for u, rows in mt.items():
        for o, vis in rows:
            for s in (o.segments if kind == "segment" else o.cells):
                d[s].add(u)
    return d


def pair_auc(pos: set, pvals: dict, users: list) -> float:
    """AUC of -log p (0 for pairs that share nothing) for separating `pos` pairs from the other pairs."""
    from scipy.stats import rankdata
    prs = [tuple(sorted(q)) for q in combinations(users, 2)]
    sc = np.array([-np.log10(max(pvals.get(q, 1.0), 1e-300)) for q in prs])
    y = np.array([q in pos for q in prs])
    if y.sum() == 0 or y.all():
        return float("nan")
    r = rankdata(sc)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def prf(pred: set, true: set) -> dict:
    tp = len(pred & true)
    return {"p": tp / len(pred) if pred else float("nan"), "r": tp / len(true) if true else float("nan"),
            "n_pred": len(pred), "n_true": len(true)}


def ari(a: dict, b: dict) -> float:
    ks = sorted(set(a) & set(b))
    return _ari({k: a[k] for k in ks}, {k: b[k] for k in ks}) if len(ks) > 1 else float("nan")


# ------------------------------------------------------------------------------------ evaluation

def evaluate(g, res: M2Result, ks=(2, 3, 5)) -> dict:
    mt = remap_truth(matched_truth(g, res), res.merge)
    mg = res.merge.get
    cfg = res.config
    out: dict = {}
    # (a) places of interest, pooled over users (user, place) pairs
    tp = truth_pois(mt, cfg.poi_min_outings, cfg.poi_min_days)
    pred = {(u, p.place_id) for u in tp for p in res.pois(u)}
    true = {(u, p) for u, s in tp.items() for p in s}
    out["poi"] = prf(pred, true)
    fav = {(u, mg(p, p)) for u in tp for p in g.result.truth.favourite_places.get(u, [])}
    out["poi_vs_favourites"] = {"r": len(pred & fav) / max(1, len(fav)), "n_true": len(fav)}
    # rank quality: precision of the top-3 predicted places vs the user's truth top-3 by outings
    # (b) common places
    tpu = truth_place_users(mt)
    for k in ks:
        out[f"common_places_k{k}"] = prf({p for p, u in res.place_users.items() if len(u) >= k},
                                         {p for p, u in tpu.items() if len(u) >= k})
    ttu = truth_trav_users(mt, res.trav_kind)
    for k in ks:
        out[f"common_{res.trav_kind}s_k{k}"] = prf({s for s, u in res.trav_users.items() if len(u) >= k},
                                                    {s for s, u in ttu.items() if len(u) >= k})
    # (c) relations
    truth = g.result.truth
    gid = {u: gr for gr, us in truth.groups.items() for u in us}
    users = [u.user for u in res.outings]
    same_group = {tuple(sorted(p)) for p in combinations(users, 2)
                  if p[0] in gid and p[1] in gid and gid[p[0]] == gid[p[1]]}
    cot = {tuple(sorted((a, b))) for a, b, _ in truth.co_traveled if a in set(users) and b in set(users)}
    gp = {gr: {mg(p, p) for p in ps} for gr, ps in truth.group_places.items()}
    obs_vis = {u: {v.place_id for o, vis in rows for v in vis} for u, rows in mt.items()}
    obs_pair = {pr for pr in same_group
                if len((obs_vis[pr[0]] & obs_vis[pr[1]]) & gp.get(gid[pr[0]], set())) >= 2}
    edges = {tuple(sorted(e)) for e in res.graph.edges}
    pvals = res.graph.graph.get("pvals", {})
    out["relations"] = {"auc_group": pair_auc(same_group, pvals, sorted(users)),
                        "auc_cotravel": pair_auc(cot, pvals, sorted(users)),
                        "n_edges": len(edges), "precision_group_or_cotravel": len(edges & (same_group | cot)) / max(1, len(edges)),
                        "precision_group": len(edges & same_group) / max(1, len(edges)),
                        "recall_group": len(edges & same_group) / max(1, len(same_group)),
                        "recall_group_observable": len(edges & obs_pair) / max(1, len(obs_pair)),
                        "recall_cotravel": len(edges & cot) / max(1, len(cot)),
                        "n_group_pairs": len(same_group), "n_obs_pairs": len(obs_pair), "n_cotravel_pairs": len(cot)}
    lab = {}
    for i, c in enumerate(res.communities):
        for u in c:
            lab[u] = i
    tl = {u: gid.get(u, f"s{u}") for u in users}
    out["community_ari"] = ari(lab, tl) if edges else float("nan")
    grouped = [u for u in users if u in gid]
    out["community_ari_grouped"] = ari({u: lab[u] for u in grouped}, {u: gid[u] for u in grouped}) if edges else float("nan")
    out["n_users_kept"] = len(users)
    return out


# ------------------------------------------------------------------------------------ split-half

def split_user(uo: UserOutings, how: str, rng: np.random.Generator) -> tuple[UserOutings, UserOutings]:
    outs = sorted(uo.outings, key=lambda o: o.t_start)
    n = len(outs)
    if how == "time":
        a, b = outs[:n // 2], outs[n // 2:]
    else:
        idx = rng.permutation(n)
        a = sorted([outs[i] for i in idx[:n // 2]], key=lambda o: o.t_start)
        b = sorted([outs[i] for i in idx[n // 2:]], key=lambda o: o.t_start)
    return UserOutings(uo.user, a), UserOutings(uo.user, b)


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def split_half(uos, cfg: M2Config, has_graph: bool, how: str, seed: int = 0, merge: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    A, B = zip(*[split_user(u, how, rng) for u in uos])
    ra, rb = analyze(A, cfg, has_graph, merge), analyze(B, cfg, has_graph, merge)
    users = [u.user for u in uos]
    # places of interest: Jaccard per user (users with a POI in both halves) vs other-user baseline
    jac, base, top1 = [], [], []
    sa = {u: {p.place_id for p in ra.pois(u)} for u in users}
    sb = {u: {p.place_id for p in rb.pois(u)} for u in users}
    both = [u for u in users if sa[u] and sb[u]]
    for i, u in enumerate(both):
        jac.append(_jacc(sa[u], sb[u]))
        v = both[(i + 1) % len(both)]
        base.append(_jacc(sa[u], sb[v]))
        ta, tb = ra.pois(u)[0].place_id, [p.place_id for p in rb.pois(u)[:3]]
        top1.append(float(ta in tb))
    # share of half-B visited places inside half-A's POI set (cf. EDA section 6)
    cover, cover_o = [], []
    for k, u in enumerate(both):
        vb = {p.place_id for p in rb.users[u]}
        v2 = both[(k + 1) % len(both)]
        cover.append(len(vb & sa[u]) / max(1, len(vb)))
        cover_o.append(len(vb & sa[v2]) / max(1, len(vb)))
    out = {"how": how, "poi_users_both": len(both), "poi_jaccard_median": float(np.median(jac)) if jac else float("nan"),
           "poi_jaccard_other_user": float(np.median(base)) if base else float("nan"),
           "poi_top1_in_top3": float(np.mean(top1)) if top1 else float("nan"),
           "B_visits_in_A_poi": float(np.mean(cover)) if cover else float("nan"),
           "B_visits_in_A_poi_other_user": float(np.mean(cover_o)) if cover_o else float("nan")}
    for k in (2, 3):
        out[f"common_places_k{k}_jaccard"] = _jacc(ra.common_at(k), rb.common_at(k))
        # precision-like: share of A's common places that are also *visited* by >= 1 user in B at all
        out[f"common_places_k{k}_nA_nB"] = (len(ra.common_at(k)), len(rb.common_at(k)))
    out["common_trav_k2_jaccard"] = _jacc(ra.common_trav_at(2), rb.common_trav_at(2))
    ea = {tuple(sorted(e)) for e in ra.graph.edges}
    eb = {tuple(sorted(e)) for e in rb.graph.edges}
    n = len(users)
    out["edges_nA_nB"] = (len(ea), len(eb))
    out["edge_jaccard"] = _jacc(ea, eb)
    out["edge_B_given_A"] = len(ea & eb) / len(ea) if ea else float("nan")
    out["edge_density_B"] = len(eb) / max(1, n * (n - 1) // 2)
    out["edge_lift"] = out["edge_B_given_A"] / out["edge_density_B"] if eb and ea else float("nan")
    # held-out visits predicted by the co-visit graph: new places in B (not visited by the user in A) that a graph
    # neighbour visited in A, versus the same number of random other users
    vis_a = {u: {p.place_id for p in ra.users[u]} for u in users}
    vis_b = {u: {p.place_id for p in rb.users[u]} for u in users}
    hit, hit_rand = [], []
    rr = np.random.default_rng(seed + 1)
    for u in users:
        nb = list(ra.graph.neighbors(u))
        new = vis_b[u] - vis_a[u]
        if not nb or not new:
            continue
        cov = set().union(*[vis_a[v] for v in nb])
        others = [v for v in users if v != u]
        rnd = set().union(*[vis_a[v] for v in rr.choice(others, size=len(nb), replace=False)])
        hit.append(len(new & cov) / len(new))
        hit_rand.append(len(new & rnd) / len(new))
    out["new_places_from_neighbors"] = float(np.mean(hit)) if hit else float("nan")
    out["new_places_from_random_users"] = float(np.mean(hit_rand)) if hit_rand else float("nan")
    out["n_users_with_neighbor"] = len(hit)
    lab_a = {u: i for i, c in enumerate(ra.communities) for u in c}
    lab_b = {u: i for i, c in enumerate(rb.communities) for u in c}
    out["community_ari"] = ari(lab_a, lab_b) if ea and eb else float("nan")
    out["edge_auc_A_to_B"] = pair_auc(eb, ra.graph.graph.get("pvals", {}), sorted(users)) if eb else float("nan")
    out["edge_auc_B_to_A"] = pair_auc(ea, rb.graph.graph.get("pvals", {}), sorted(users)) if ea else float("nan")
    return out


# ------------------------------------------------------------------------------------ synthetic runs

def synth_run(provider, n_users: int, seed: int, gen: dict, params: dict | None = None,
              cfg: M2Config | None = None, with_graph: bool = True, mv: MapView | None = None) -> dict:
    t = time.time()
    g = generate_v2(params or fake_params_v2(), provider, n_users, seed, gen=gen)
    t1 = time.time()
    mv = mv or make_mv(provider, with_graph)
    res = run_m2(g.result.traces, mv, cfg)
    ev = evaluate(g, res)
    return {"g": g, "res": res, "eval": ev, "secs_gen": t1 - t, "secs_m": time.time() - t1,
            "n_points": sum(len(v) for v in g.result.traces.values())}


def row(ev: dict) -> dict:
    r = ev["relations"]
    seg = next(k for k in ev if k.startswith("common_") and k.endswith("_k2") and "places" not in k)
    return {"poi_p": ev["poi"]["p"], "poi_r": ev["poi"]["r"],
            "cp2_p": ev["common_places_k2"]["p"], "cp2_r": ev["common_places_k2"]["r"],
            "cp3_r": ev["common_places_k3"]["r"], "cp5_r": ev["common_places_k5"]["r"],
            "seg2_p": ev[seg]["p"], "seg2_r": ev[seg]["r"],
            "rel_p": r["precision_group_or_cotravel"], "rel_auc": r["auc_group"], "rel_r": r["recall_group"], "rel_r_obs": r["recall_group_observable"],
            "ari": ev["community_ari"], "ari_grouped": ev["community_ari_grouped"], "n_edges": r["n_edges"], "users": ev["n_users_kept"]}


def sensitivity(provider, base_gen: dict, n_users: int, seed: int, cfg: M2Config, mv: MapView, params=None) -> list[dict]:
    cases = [("baseline", {}),
             ("noise=15m", {"sensor_noise_m": 15.0}), ("noise=30m", {"sensor_noise_m": 30.0}),
             ("noise=60m", {"sensor_noise_m": 60.0}),
             ("gaps=10/h", {"gap_rate_per_hr": 10.0}), ("gaps=20/h", {"gap_rate_per_hr": 20.0}),
             ("sampling=30s", {"sample_interval_s": 30}), ("sampling=60s", {"sample_interval_s": 60}),
             ("stops unlogged=0.5", {"stop_signal_loss_prob": 0.5}),
             ("stops unlogged=0.75", {"stop_signal_loss_prob": 0.75})]
    rows = []
    for name, gen in cases:
        r = synth_run(provider, n_users, seed, {**base_gen, **gen}, params, cfg, mv=mv)
        rows.append({"case": name, **row(r["eval"])})
    return rows


def window_study(provider, base_gen: dict, n_users: int, seed: int, cfg: M2Config, mv: MapView) -> list[dict]:
    rows = []
    for days in (10, 20, 40, 80, 160):
        p = copy.deepcopy(fake_params_v2())
        p["users"]["active_days"]["mu"] = float(np.log(days))
        p["users"]["active_days"]["sigma"] = 0.3
        r = synth_run(provider, n_users, seed, base_gen, p, cfg, mv=mv)
        rows.append({"case": f"window~{days}d", "outings_median": float(np.median([len(u.outings) for u in r["res"].outings])),
                     **row(r["eval"])})
    return rows


def user_count_study(provider, base_gen, counts, seeds, cfg, mv) -> list[dict]:
    rows = []
    for n in counts:
        rs = [row(synth_run(provider, n, s, base_gen, None, cfg, mv=mv)["eval"]) for s in seeds]
        keys = rs[0].keys()
        rows.append({"case": f"users={n}", **{k: float(np.nanmean([r[k] for r in rs])) for k in keys}})
    return rows


def few_outings_study(provider, base_gen, n_users, seed, cfg, mv) -> list[dict]:
    """Share of users with few outings: shift the outings-per-user lognormal down."""
    rows = []
    for med in (60, 30, 15, 8):
        p = copy.deepcopy(fake_params_v2())
        p["users"]["outings_per_user"]["mu"] = float(np.log(med))
        r = synth_run(provider, n_users, seed, base_gen, p, cfg, mv=mv)
        n_out = [len(u.outings) for u in r["res"].outings]
        rows.append({"case": f"median outings {med}", "kept": f"{len(n_out)}/{n_users}",
                     "share_le15": float(np.mean(np.array(n_out) <= 15)) if n_out else float("nan"), **row(r["eval"])})
    return rows


# ------------------------------------------------------------------------------------ relation rule study

def reanalyze(res: M2Result, cfg: M2Config) -> M2Result:
    """Re-run everything after tokenization with a different config (cheap: no generation, no tokenize)."""
    res2 = analyze(res.outings, cfg, res.trav_kind == "segment", merge=res.merge)
    res2.merge = res.merge
    return res2


def rel_row(ev: dict) -> dict:
    r = ev["relations"]
    return {"n_edges": r["n_edges"], "prec": r["precision_group_or_cotravel"],
            "recall_group": r["recall_group"], "recall_obs": r["recall_group_observable"],
            "auc": r["auc_group"], "ari_grouped": ev["community_ari_grouped"]}


RULES = [(u, al, s) for u in ("all", "sharing") for al in (0.05, 0.1, 0.2, 0.3, 0.5) for s in (1, 2, 3)]


def rule_study(n: int, cfg: M2Config) -> dict:
    """DEV-ONLY sweep of the relation significance rule over dev and a sparser dev draw.
    The held-out draw (map seed 7, gen seed 4242) is never touched here."""
    prov, prov2 = toy_provider(1), toy_provider(2)
    draws = {"dev": synth_run(prov, n, 11, DEV_GEN, None, cfg, mv=make_mv(prov, True)),
             "dev_sparse": synth_run(prov2, n, 13, SPARSE_GEN, sparse_params(), cfg, mv=make_mv(prov2, True))}
    out = {}
    for name, r in draws.items():
        rows = []
        for u, al, s in RULES:
            cfg2 = replace(cfg, rel_universe=u, rel_alpha=al, rel_min_shared=s)
            ev = evaluate(r["g"], reanalyze(r["res"], cfg2))
            rows.append({"rule": f"{u},alpha={al},shared>={s}", **rel_row(ev)})
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
        out[name] = rows
    return out


# ------------------------------------------------------------------------------------ M output distributions

def snap_counts(uos) -> dict:
    """Tokenization counts (cross-checked against geolife/inside/reports/outings-eda.md on the real side)."""
    n_out = sum(len(u.outings) for u in uos)
    stops = [v for u in uos for o in u.outings for v in o.visits if v.kind == "stop"]
    vis = [v for u in uos for o in u.outings for v in o.visits]
    snapped = lambda vv: float(np.mean([not v.place_id.startswith("h3:") for v in vv])) if vv else float("nan")
    return {"users": len(uos), "outings": n_out, "stops": len(stops),
            "stops_per_outing": len(stops) / max(1, n_out),
            "snap_share_stops": snapped(stops), "snap_share_visits": snapped(vis)}


def _poi_profile(res: M2Result) -> np.ndarray:
    """Pooled visit-time profile over every (user, place-of-interest), each POI L1-normalised first."""
    p = np.zeros(NPROF)
    for u in res.outings:
        for up in res.pois(u.user):
            tot = up.profile.sum()
            if tot > 0:
                p += up.profile / tot
    return p / max(p.sum(), 1e-12)


def m_output_compare(res_real: M2Result, res_syn: M2Result) -> dict:
    """Distributions of M's OUTPUTS, real vs synthetic (label-free; the transfer test of PLAN section 7)."""
    from collections import Counter

    def side(res: M2Result) -> dict:
        n = len(res.outings)
        npoi = [len(res.pois(u.user)) for u in res.outings]
        deg = [d for _, d in res.graph.degree()]
        comm = sorted((len(c) for c in res.communities if len(c) >= 2), reverse=True)
        return {"n_users": n,
                "pois_per_user": _summ(npoi), "pois_hist": dict(Counter(npoi)),
                "poi_profile": _poi_profile(res).tolist(),
                "common_places": {k: len(res.common_at(k)) for k in (2, 3, 5)},
                "common_place_sizes": dict(Counter(len(u) for u in res.place_users.values() if len(u) >= 2)),
                "common_trav": {k: len(res.common_trav_at(k)) for k in (2, 3, 5)},
                "common_trav_sizes": dict(Counter(len(u) for u in res.trav_users.values() if len(u) >= 2)),
                "degree": _summ(deg), "degree_hist": dict(Counter(deg)),
                "edge_density": float(np.mean(deg) / max(1, n - 1)) if deg else float("nan"),
                "share_users_with_relation": float(np.mean([d > 0 for d in deg])) if deg else float("nan"),
                "n_edges": res.graph.number_of_edges(),
                "community_sizes": comm, "n_communities_ge2": len(comm)}

    r, s = side(res_real), side(res_syn)
    pr, ps = np.array(r["poi_profile"]), np.array(s["poi_profile"])
    return {"real": r, "synthetic": s,
            "tv_pois_per_user": tv_hist(r["pois_hist"], s["pois_hist"], cap=30),
            "tv_poi_profile": float(0.5 * np.abs(pr - ps).sum()),
            "tv_common_place_sizes": tv_hist(r["common_place_sizes"], s["common_place_sizes"], cap=10),
            "tv_common_trav_sizes": tv_hist(r["common_trav_sizes"], s["common_trav_sizes"], cap=10),
            "tv_degree": tv_hist(r["degree_hist"], s["degree_hist"], cap=30),
            "tv_community_sizes": tv_hist(dict(Counter(r["community_sizes"])),
                                          dict(Counter(s["community_sizes"])), cap=20)}


# ------------------------------------------------------------------------------------ real data

def load_real_points(max_users: int | None = None, path: str | None = None):
    """-> {uid: (t, lat, lon)} from the points parquet (`path`, else env GEOLIFE_POINTS) or None."""
    import pandas as pd
    pq = path or os.environ.get("GEOLIFE_POINTS")
    if not pq:
        return None
    df = pd.read_parquet(pq, columns=["uid", "t", "lat", "lon"])
    out = {}
    for uid, g in df.groupby("uid", sort=True):
        out[str(uid)] = (g.t.to_numpy(np.int64), g.lat.to_numpy(float), g.lon.to_numpy(float))
        if max_users and len(out) >= max_users:
            break
    return out


def real_outings(cfg: M2Config, places_path: str, graph=None, max_users=None, points_path: str | None = None):
    import pandas as pd
    from geolife.transfer.tokenize.outings import PlaceIndex, user_outings_from_arrays
    pts = load_real_points(max_users, points_path)
    if pts is None:
        return None
    pidx = PlaceIndex(pd.read_parquet(places_path))
    out = []
    for uid in sorted(pts):
        uo = user_outings_from_arrays(uid, *pts[uid], pidx, cfg.tok, ())
        if uo is not None:
            out.append(uo)
    return out


def compare_stats(a: dict, b: dict) -> dict:
    """Synthetic vs real distribution gaps: relative gap of medians/means and TV of histograms."""
    out = {}
    for k in ("outings_per_user", "stops_per_outing", "length_km", "duration_min", "distinct_places_per_user",
              "pois_per_user", "stop_revisit_rate", "active_days", "window_days", "degree"):
        out[k] = {"synthetic": {m: a[k][m] for m in ("mean", "median", "p90")},
                  "real": {m: b[k][m] for m in ("mean", "median", "p90")}}
    out["tv_users_per_place"] = tv_hist(a["users_per_place_hist"], b["users_per_place_hist"])
    out["tv_users_per_trav"] = tv_hist(a["users_per_trav_hist"], b["users_per_trav_hist"])
    for k in ("share_places_ge2_users", "share_users_with_relation", "n_communities_ge2"):
        out[k] = {"synthetic": a[k], "real": b[k]}
    da, db = np.array(a["degree_list"], float), np.array(b["degree_list"], float)
    out["degree_density"] = {"synthetic": float(da.mean() / max(1, a["n_users"] - 1)), "real": float(db.mean() / max(1, b["n_users"] - 1))}
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k not in ("g", "res")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", default="dev,held,ams,sens,split,real")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--syn-dir", default="geolife/outside/synthetic/outings_v2_seed0",
                    help="stage-2 output dir for the 'beijing' stage (real-vs-synthetic transfer test)")
    ap.add_argument("--out", default="geolife/eval/reports/m-v2.json")
    a = ap.parse_args(argv)
    stages = set(a.stages.split(","))
    cfg = M2Config()
    out: dict = {}
    prov = toy_provider(1)
    mv = make_mv(prov, True)
    mv_nograph = make_mv(prov, False)
    n = 40 if a.quick else a.n
    store = {}
    if "rules" in stages:
        out["rule_study"] = rule_study(n, cfg)
    if "dev" in stages:
        r = synth_run(prov, n, 11, DEV_GEN, None, cfg, mv=mv)
        out["dev"] = {"eval": r["eval"], "stats": distribution_stats(r["res"]), "secs": (r["secs_gen"], r["secs_m"]),
                      "eval_oldrule": evaluate(r["g"], reanalyze(r["res"], replace(cfg, **OLD_RULE)))}
        store["dev"] = r
        print("dev", row(r["eval"]))
        r2 = synth_run(prov, n, 11, DEV_GEN, None, cfg, mv=mv_nograph)
        out["dev_cells"] = {"eval": r2["eval"], "stats": distribution_stats(r2["res"])}
        store["dev_cells"] = r2
        print("dev cells", row(r2["eval"]))
    if "held" in stages:
        prov2 = toy_provider(7)
        mv2 = make_mv(prov2, True)
        r = synth_run(prov2, n, 4242, HELD_GEN, held_params(), cfg, mv=mv2)
        out["held"] = {"eval": r["eval"], "stats": distribution_stats(r["res"]),
                       "eval_oldrule": evaluate(r["g"], reanalyze(r["res"], replace(cfg, **OLD_RULE)))}
        store["held"] = r
        r3 = synth_run(prov2, n, 4242, HELD_GEN, held_params(), cfg, mv=make_mv(prov2, False))
        out["held_cells"] = {"eval": r3["eval"], "stats": distribution_stats(r3["res"])}
        store["held_cells"] = r3
        print("held cells", row(r3["eval"]))
        print("held", row(r["eval"]))
    if "ams" in stages:
        ap_ = amsterdam_provider()
        if ap_ is not None:
            mva = MapView(places_table(ap_), None)
            r = synth_run(ap_, 40 if a.quick else 60, 5, {**DEV_GEN, "max_outings": 60, "sample_interval_s": 10}, None, cfg, mv=mva)
            out["amsterdam"] = {"eval": r["eval"]}
            print("ams", row(r["eval"]))
    if "sens" in stages:
        out["sens"] = sensitivity(prov, DEV_GEN, 60 if a.quick else 80, 21, cfg, mv)
        out["window"] = window_study(prov, DEV_GEN, 60 if a.quick else 80, 21, cfg, mv)
        out["users"] = user_count_study(prov, DEV_GEN, (10, 20, 40, 80), (31, 32), cfg, mv)
        out["few"] = few_outings_study(prov, DEV_GEN, 80, 21, cfg, mv)
        for k in ("sens", "window", "users", "few"):
            for r in out[k]:
                print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items()})
    if "split" in stages:
        sp = {}
        for tag in ("dev_cells", "held_cells"):
            if tag in store:
                res = store[tag]["res"]
                sp[tag] = [split_half(res.outings, cfg, False, how, merge=res.merge) for how in ("random", "time")]
        out["split_synth"] = sp
        print(json.dumps(_clean(sp), indent=1)[:2500])
    if "real" in stages and os.environ.get("GEOLIFE_POINTS"):
        pl = os.environ["GEOLIFE_PLACES"]
        uos = real_outings(cfg, pl)
        import pandas as pd
        mvr = MapView(PlaceTable.from_parquet(pl), None)
        merge = place_merge_map(mvr, cfg.merge_m)
        res = analyze(uos, cfg, has_graph=False, merge=merge)
        out["real"] = {"stats": distribution_stats(res), "n_users": len(uos)}
        out["split_real"] = [split_half(uos, cfg, False, how, merge=merge) for how in ("random", "time")]
        for tag in ("dev_cells", "held_cells"):
            if tag in store:
                out["compare_" + tag] = compare_stats(distribution_stats(store[tag]["res"]), out["real"]["stats"])
        out["real_time_bonus"] = [split_half(uos, replace(cfg, time_bonus=1.0), False, "random", merge=merge)]
        print(json.dumps(_clean(out["split_real"]), indent=1)[:2500])
    if "beijing" in stages and os.environ.get("GEOLIFE_POINTS"):
        # the transfer test proper: M unchanged on real Beijing vs on synthetic generated INTO Beijing
        # from the committed artifact (same map, same tokenizer, cell proxy on both sides).
        from geolife.outside.synthetic_io import read_synthetic
        pl = os.environ["GEOLIFE_PLACES"]
        g = read_synthetic(a.syn_dir)
        mvb = MapView(PlaceTable.from_parquet(pl), None)
        res_s = run_m2(g.result.traces, mvb, cfg)
        out["beijing"] = {"eval": evaluate(g, res_s),
                          "eval_oldrule": evaluate(g, reanalyze(res_s, replace(cfg, **OLD_RULE))),
                          "stats": distribution_stats(res_s), "snap": snap_counts(res_s.outings),
                          "meta": {k: g.meta.get(k) for k in ("variant", "artifact", "n_users", "epsilon")}}
        print("beijing", row(out["beijing"]["eval"]))
        uos = real_outings(cfg, pl)
        merge = place_merge_map(mvb, cfg.merge_m)
        res_r = analyze(uos, cfg, has_graph=False, merge=merge)
        res_r.merge = merge
        out["real_beijing"] = {"stats": distribution_stats(res_r), "snap": snap_counts(uos),
                               "n_users": len(uos)}
        out["compare_beijing"] = compare_stats(out["beijing"]["stats"], out["real_beijing"]["stats"])
        out["m_output_compare"] = m_output_compare(res_r, res_s)
        out["split_beijing_synth"] = [split_half(res_s.outings, cfg, False, how, merge=res_s.merge)
                                      for how in ("random", "time")]
        out["split_beijing_real"] = [split_half(uos, cfg, False, how, merge=merge) for how in ("random", "time")]
        print(json.dumps(_clean({k: out[k] for k in ("m_output_compare",)}), indent=1)[:2500])
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(out), indent=1))


if __name__ == "__main__":
    main()

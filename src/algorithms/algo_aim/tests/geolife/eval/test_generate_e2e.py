"""records -> DP measurements -> artifact on disk -> fresh process -> synthetic users on a toy city."""
import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage
from geolife.transfer import schema as S
from geolife.eval.fakedata import cached_fake_artifact
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline import generate

SRC = str(Path(__file__).resolve().parents[3] / "src")
GEN = {"n_days": 4, "stay_interval_s": 120.0, "carpool_frac": 0.5}
N, SEED = 300, 11


def _provider():
    return GridCityProvider(n=36, spacing=225.0, n_places=2500, seed=3)


@lru_cache(maxsize=None)
def _run():
    d, world, ms, led = cached_fake_artifact()
    art = load_artifact(d)
    art.generation = {**art.generation, **GEN}
    return art, generate(art, _provider(), N, SEED)


def _digest(traces):
    h = hashlib.sha256()
    for u in sorted(traces):
        for p in traces[u]:
            h.update(f"{u},{p.t},{p.lat:.7f},{p.lon:.7f};".encode())
    return h.hexdigest()


def _tv(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return 0.5 * np.abs(p / p.sum() - q / q.sum()).sum()


def test_artifact_is_clean_and_loadable():
    d, *_ = cached_fake_artifact()
    art = load_artifact(d)
    validate_no_raw_leakage(art)


def test_users_per_place_histogram_matches_artifact():
    art, g = _run()
    cut = g.params["min_observed_k"]["other"]
    target = np.array(sorted((int(k) for k, n in g.params["hist_users_per_place"]["other"].items()
                              for _ in range(n)), reverse=True), float)
    truth = g.result.truth
    occ = {}
    for uid, a in truth.anchors.items():
        for p in a["others"]:
            occ[p] = occ.get(p, 0) + 1
    gen_counts = np.array(sorted(occ.values(), reverse=True), float)
    n = min(len(target), len(gen_counts), 12)
    assert n >= 8
    rel = np.abs(gen_counts[:n] - target[:n]) / target[:n]
    assert np.median(rel) < 0.35, (gen_counts[:n], target[:n])
    # number of places above the observable threshold is of the same order
    ng = int((gen_counts >= cut).sum())
    assert 0.5 * len(target) <= ng <= 2.0 * len(target)
    # total anchor memberships ~ N * mean anchors
    pmf = np.array(g.params["other_anchor_pmf"])
    assert abs(gen_counts.sum() - N * (pmf * np.arange(len(pmf))).sum()) < 0.15 * gen_counts.sum()
    # shared-workplace group structure
    wt = g.params["hist_users_per_place"]["work"]
    tgt_mean = sum(int(k) * n_ for k, n_ in wt.items()) / max(1, sum(wt.values()))
    sizes = [len(gr.users) for gr in truth.groups if gr.kind == "workplace"]
    assert sizes and 0.5 * tgt_mean < np.mean(sizes) * N / g.params["n_users_source"] < 2.0 * tgt_mean


def test_od_flows_match_artifact_targets():
    art, g = _run()
    res, truth = g.result, g.result.truth
    m2l = truth.latent_to_map
    lens = {"commute": [], "other": []}
    pairs = {}
    for uid, trips in res.trips.items():
        a = truth.anchors[uid]
        hw = {m2l[a["home"]], m2l.get(a["work"])}
        for t in trips:
            kind = "commute" if {t["origin"], t["dest"]} == hw else "other"
            lens[kind].append(t["length_m"])
            pairs[(t["origin"], t["dest"])] = pairs.get((t["origin"], t["dest"]), 0) + 1
    edges = [0, 500, 1000, 2000, 4000, 8000, 16000, np.inf]
    for kind in S.KINDS:
        h, _ = np.histogram(lens[kind], bins=edges)
        target = g.params["trip_dist"][kind]["p"]
        assert _tv(h, target) < 0.4, (kind, h / h.sum(), target)
    # role-to-role flow shares vs artifact role_trips
    m = next(x for x in art.measurements if x.name == S.ROLE_TRIPS)
    tgt = np.array([max(0, v) if v >= m.sigma else 0 for v in m.values])
    idx = {k: i for i, k in enumerate(m.keys)}
    gen = np.zeros(len(tgt))
    for uid, ss in res.stays.items():
        roles = ["other" if r == "pickup" else r for *_, r in ss]
        for a, b in zip(roles, roles[1:]):
            gen[idx[(a, b)]] += 1
    assert _tv(gen, tgt) < 0.3, (gen / gen.sum(), tgt / tgt.sum())
    # OD pairs repeat (preferential return) rather than being all distinct
    assert sum(pairs.values()) / len(pairs) > 1.5


def test_planted_truth_is_consistent_with_traces():
    art, g = _run()
    res, truth = g.result, g.result.truth
    m2l = truth.latent_to_map
    by_stays = {u: {p for p, *_ in ss} for u, ss in res.stays.items()}
    wg = [gr for gr in truth.groups if gr.kind == "workplace"]
    assert wg
    seen = total = 0
    for gr in wg:
        for u in gr.users:
            total += 1
            seen += m2l[gr.place] in by_stays[u]
    assert seen / total > 0.9                      # planted colleagues really go to the shared place
    # every co-presence is a real overlap at a place both stayed at
    place_users = {}
    for p, u, v, t0, t1 in truth.copresence:
        assert p in by_stays[u] and p in by_stays[v] and t1 > t0
        place_users.setdefault(p, set()).update((u, v))
    shared_map = {m2l[p] for p in truth.shared_places if p in m2l}
    frac = np.mean([p in shared_map for p, *_ in truth.copresence])
    assert frac > 0.7
    # co-travel: leader and follower share segments, listed as crossings
    done = [c for c in truth.cotravel if c.days]
    assert done
    for c in done:
        for _, segs in c.segments:
            assert segs and all({c.leader, c.follower} <= set(truth.crossing_segments[s]) for s in segs)
    assert truth.crossing_segments


def test_determinism_across_processes():
    d, *_ = cached_fake_artifact()
    code = ("import sys, hashlib, json\n"
            "from geolife.outside.grid_city import GridCityProvider\n"
            "from geolife.outside.pipeline import generate\n"
            "from geolife.transfer.artifact import load_artifact\n"
            f"art = load_artifact({d!r}); art.generation = {{**art.generation, **{GEN!r}}}\n"
            f"g = generate(art, GridCityProvider(n=36, spacing=225.0, n_places=2500, seed=3), {N}, {SEED})\n"
            "h = hashlib.sha256()\n"
            "for u in sorted(g.result.traces):\n"
            "    for p in g.result.traces[u]:\n"
            "        h.update(f'{u},{p.t},{p.lat:.7f},{p.lon:.7f};'.encode())\n"
            "print(h.hexdigest())\n")
    env = {**os.environ, "PYTHONPATH": SRC}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout.strip()
    art, g = _run()
    assert out == _digest(g.result.traces)
    # different seed -> different world
    g2 = generate(art, _provider(), 40, SEED + 1)
    g3 = generate(art, _provider(), 40, SEED)
    g4 = generate(art, _provider(), 40, SEED)
    assert _digest(g3.result.traces) == _digest(g4.result.traces)
    assert _digest(g2.result.traces) != _digest(g3.result.traces)

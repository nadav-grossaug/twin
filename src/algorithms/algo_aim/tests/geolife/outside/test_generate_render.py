import numpy as np

from geolife.eval.fakedata import cached_fake_artifact
from geolife.outside.fit import fit_behaviour
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.population import T0, sample_population
from geolife.outside.render import render
from geolife.outside.teleport import ground_places
from geolife.transfer.types import GpsPoint


def _build(n=60, gen=None, seed=2):
    _, _, ms, led = cached_fake_artifact()
    params = fit_behaviour(ms, led, extent_m=8000.0)
    g = {"n_days": 3, "stay_interval_s": 120.0, "carpool_frac": 1.0, **(gen or {})}
    prov = GridCityProvider(n=36, spacing=225.0, n_places=2500, seed=3)
    pop = sample_population(params, n, seed, g)
    tp = ground_places(pop, prov, {}, seed)
    return pop, tp, prov, g


def _xy(prov, pts):
    lat0, lon0 = prov.origin
    return np.array([[(p.lon - lon0) * 111320 * np.cos(np.radians(lat0)), (p.lat - lat0) * 111320]
                     for p in pts])


def test_render_emits_sorted_gps_points():
    pop, tp, prov, g = _build()
    res = render(pop, tp.mapping, prov, g, seed=2)
    assert set(res.traces) == {u.uid for u in pop.users}
    n = 0
    for uid, pts in res.traces.items():
        n += len(pts)
        assert all(isinstance(p, GpsPoint) for p in pts)
        ts = [p.t for p in pts]
        assert ts == sorted(ts) and len(set(ts)) == len(ts)
        assert not ts or (ts[0] >= T0 - 1 and ts[-1] <= T0 + 4 * 86400)  # late returns may spill past the last midnight
    assert n > 10_000
    # points near the place they stay at (noise ~ sensor sd)
    places = {p.id: p for p in prov.places()}
    u = pop.users[0].uid
    p0, a, b, _ = max(res.stays[u], key=lambda s: s[2] - s[1])
    pts = [p for p in res.traces[u] if a <= p.t <= b]
    if len(pts) > 5:
        xy = _xy(prov, pts)
        err = np.linalg.norm(xy - [places[p0].x, places[p0].y], axis=1)
        assert np.median(err) < 40


def test_noise_parameters_take_effect():
    pop, tp, prov, g = _build(n=30)
    quiet = render(pop, tp.mapping, prov, {**g, "sensor_noise_m": 1.0, "dwell_jitter_m": 0.5,
                                           "gap_rate_per_hr": 0.0, "outlier_prob": 0.0}, seed=2)
    noisy = render(pop, tp.mapping, prov, {**g, "sensor_noise_m": 30.0, "dwell_jitter_m": 10.0,
                                           "gap_rate_per_hr": 3.0, "gap_mean_s": 600.0}, seed=2)
    nq = sum(len(v) for v in quiet.traces.values())
    nn = sum(len(v) for v in noisy.traces.values())
    assert nn < nq                                   # gaps drop points
    places = {p.id: p for p in prov.places()}

    def spread(res):
        out = []
        for u, ss in res.stays.items():
            p0, a, b, _ = max(ss, key=lambda s: s[2] - s[1])
            pts = [p for p in res.traces[u] if a <= p.t <= b]
            if len(pts) > 5:
                out.append(np.median(np.linalg.norm(_xy(prov, pts) - [places[p0].x, places[p0].y], axis=1)))
        return np.median(out)
    assert spread(noisy) > 3 * spread(quiet)


def test_truth_cotravel_shares_exact_route_and_crossings():
    pop, tp, prov, g = _build(n=150, gen={"n_days": 5})
    res = render(pop, tp.mapping, prov, g, seed=2)
    truth = res.truth
    done = [c for c in truth.cotravel if c.days]
    assert done, "carpool_frac=1 should realise some co-travel"
    for c in done:
        lead_trips = res.trips[c.leader]
        for day, segs in c.segments:
            assert any(tuple(t["segments"]) == segs and int((t["t0"] - T0) // 86400) == day
                       for t in lead_trips)
            # same time as leader's trip, same speed -> identical timing
            ft = [t for t in res.trips[c.follower] if tuple(t["segments"]) == segs
                  and int((t["t0"] - T0) // 86400) == day][0]
            lt = [t for t in lead_trips if tuple(t["segments"]) == segs
                  and int((t["t0"] - T0) // 86400) == day][0]
            assert abs(ft["t0"] - lt["t0"]) < 1e-6 and abs(ft["t1"] - lt["t1"]) < 1e-6
            for s in segs:
                assert {c.leader, c.follower} <= set(truth.crossing_segments[s])
    assert all(len(u) >= 2 for u in truth.crossing_segments.values())
    for p, u, v, t0, t1 in truth.copresence:
        assert u < v and t1 - t0 >= g.get("min_copresence_s", 600)


def test_render_deterministic():
    pop, tp, prov, g = _build(n=25)
    a = render(pop, tp.mapping, prov, g, seed=4)
    b = render(pop, tp.mapping, prov, g, seed=4)
    c = render(pop, tp.mapping, prov, g, seed=5)
    assert a.traces == b.traces
    assert a.traces != c.traces

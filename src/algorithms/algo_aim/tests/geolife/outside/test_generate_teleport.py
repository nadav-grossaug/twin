import numpy as np

from geolife.eval.fakedata import cached_fake_artifact
from geolife.outside.fit import fit_behaviour
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.population import sample_population
from geolife.outside.teleport import ground_places

GEN = {"n_days": 3}


def _setup(seed=3, n=60):
    _, _, ms, led = cached_fake_artifact()
    params = fit_behaviour(ms, led, extent_m=8000.0)
    prov = GridCityProvider(n=36, spacing=225.0, n_places=2500, seed=seed)
    return params, prov, sample_population(params, n, 5, GEN)


def test_grid_provider_routes_are_consistent():
    prov = GridCityProvider(n=10, spacing=200.0, n_places=100, seed=1)
    p = prov.places()
    r = prov.route(p[0].id, p[7].id)
    assert r.polyline.shape[1] == 2 and r.length_m > 0
    assert np.isclose(r.length_m, np.linalg.norm(np.diff(r.polyline, axis=0), axis=1).sum())
    assert prov.route(p[0].id, p[7].id).segments == r.segments   # deterministic
    assert prov.route(p[7].id, p[0].id).length_m > 0
    lat, lon = prov.to_latlon(np.array([0.0, 1000.0]), np.array([0.0, 1000.0]))
    assert lat[1] > lat[0] and lon[1] > lon[0]


def test_teleport_category_injective_and_consistent():
    params, prov, pop = _setup()
    tp = ground_places(pop, prov, {"extent_m": 8000.0}, seed=1)
    bycat = {p.id: p.category for p in prov.places()}
    assert tp.report["n_reused"] == 0 and tp.report["n_category_missing"] == 0
    assert len(set(tp.mapping.values())) == len(tp.mapping)            # injective
    for lid, mid in tp.mapping.items():
        assert pop.places[lid].category == bycat[mid]
    # every anchor of every user is grounded (shared latent places stay shared)
    for u in pop.users:
        for pid in [u.home, u.work, *u.others]:
            if pid:
                assert pid in tp.mapping
    for lid, users in pop.truth.shared_places.items():
        if lid in tp.mapping:
            assert len(users) >= 2


def test_teleport_deterministic_and_seeded():
    params, prov, pop = _setup()
    a = ground_places(pop, prov, {}, seed=1).mapping
    b = ground_places(pop, prov, {}, seed=1).mapping
    c = ground_places(pop, prov, {}, seed=2).mapping
    assert a == b
    assert set(c) == set(a)   # seed only breaks ties; the grounded set is unchanged


def test_teleport_preserves_popularity_order_roughly():
    params, prov, pop = _setup(n=120)
    tp = ground_places(pop, prov, {}, seed=1)
    w = {p.id: p.weight for p in prov.places()}
    lat = [p for p in pop.places.values() if p.kind == "other" and p.id in tp.mapping]
    if len(lat) > 5:
        from scipy.stats import spearmanr
        rho = spearmanr([p.weight for p in lat], [w[tp.mapping[p.id]] for p in lat])[0]
        assert rho > 0.3

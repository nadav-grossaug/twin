"""Teleport (ground_places) on the v2 population via latent_view, and the toy grid provider."""
import numpy as np

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.population_v2 import latent_view, sample_population_v2
from geolife.outside.teleport import ground_places

GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 25}


def _setup(seed=3, n=40):
    prov = GridCityProvider(n=36, spacing=225.0, n_places=2500, seed=seed)
    pop = sample_population_v2(fake_params_v2(), n, 5, prov, GEN)
    return prov, pop, latent_view(pop)


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


def test_teleport_grounds_every_used_latent_place():
    prov, pop, lv = _setup()
    tp = ground_places(lv, prov, {}, seed=1)
    used = {p for u in pop.users for o in u.outings for p in o.places}
    assert used <= set(tp.mapping)
    assert len(set(tp.mapping.values())) == len(tp.mapping) or tp.report["n_reused"] > 0
    prov_ids = {p.id for p in prov.places()}
    assert set(tp.mapping.values()) <= prov_ids
    # shared latent places stay shared: one global mapping
    for lid, users in pop.truth.shared_places.items():
        if lid in tp.mapping:
            assert len(users) >= 2


def test_teleport_deterministic_and_seeded():
    prov, pop, lv = _setup()
    a = ground_places(lv, prov, {}, seed=1).mapping
    b = ground_places(lv, prov, {}, seed=1).mapping
    c = ground_places(lv, prov, {}, seed=2).mapping
    assert a == b
    assert set(c) == set(a)   # seed only breaks ties; the grounded set is unchanged


def test_teleport_preserves_popularity_order_roughly():
    prov, pop, lv = _setup(n=80)
    tp = ground_places(lv, prov, {}, seed=1)
    w = {p.id: p.weight for p in prov.places()}
    lat = [p for p in pop.places.values() if p.id in tp.mapping]
    if len(lat) > 5:
        from scipy.stats import spearmanr
        rho = spearmanr([p.weight for p in lat], [w[tp.mapping[p.id]] for p in lat])[0]
        assert rho > 0.3

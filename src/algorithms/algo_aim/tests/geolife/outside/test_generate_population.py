import re

import numpy as np

from geolife.eval.fakedata import cached_fake_artifact
from geolife.outside.fit import fit_behaviour
from geolife.outside.population import T0, sample_population


def _params():
    _, _, ms, led = cached_fake_artifact()
    return fit_behaviour(ms, led, extent_m=8000.0)


def test_population_structure_and_planted_truth():
    pop = sample_population(_params(), 200, 7, {"n_days": 5, "carpool_frac": 1.0})
    t = pop.truth
    assert len(pop.users) == 200 and t.users == [u.uid for u in pop.users]
    assert all(re.fullmatch(r"L[0-9a-f]{10}", k) for k in pop.places)       # random latent ids
    by = {u.uid: u for u in pop.users}
    wg = [g for g in t.groups if g.kind == "workplace"]
    assert wg, "shared workplaces expected"
    for g in wg:
        assert len(g.users) >= 2
        assert all(by[u].work == g.place for u in g.users)
    # shared places == places that >= 2 users hold as an anchor
    for pid, us in t.shared_places.items():
        assert len(us) >= 2
        assert all(pid in (by[u].home, by[u].work, *by[u].others) for u in us)
    assert t.cotravel
    for ct in t.cotravel:
        lead, fol = by[ct.leader], by[ct.follower]
        assert lead.work == fol.work == ct.workplace and ct.pickup_place == lead.home
        assert fol.carpool_leader == ct.leader and lead.carpool_leader is None


def test_schedules_are_chronological_and_start_at_home():
    pop = sample_population(_params(), 80, 8, {"n_days": 5})
    for u in pop.users:
        ts = [v.t_depart for v in u.visits]
        assert all(t >= T0 for t in ts)
        by_day = {}
        for v in u.visits:
            by_day.setdefault(v.day, []).append(v)
        for d, vs in by_day.items():
            assert vs[-1].role == "home"                         # every day ends at home
            assert all(vs[i].t_depart <= vs[i + 1].t_depart + 1e-6 for i in range(len(vs) - 1)
                       if vs[i + 1].role != "pickup" and vs[i].role != "pickup")
    # no work on weekends (days 5,6 of an n_days=7 week)
    pop = sample_population(_params(), 60, 9, {"n_days": 7})
    assert not any(v.role == "work" and v.day >= 5 for u in pop.users for v in u.visits)
    assert any(v.role == "work" for u in pop.users for v in u.visits)


def test_population_deterministic_and_seed_sensitive():
    p = _params()
    a = sample_population(p, 50, 3, {"n_days": 3})
    b = sample_population(p, 50, 3, {"n_days": 3})
    c = sample_population(p, 50, 4, {"n_days": 3})
    sig = lambda pop: [(u.home, u.work, tuple(u.others), tuple((v.place, round(v.t_depart)) for v in u.visits))
                       for u in pop.users]
    assert sig(a) == sig(b)
    assert sig(a) != sig(c)

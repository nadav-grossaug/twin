import math

import numpy as np
import pytest

from geolife.inside import dp
from geolife.transfer.types import Stay, Trip, UserRecord

PLACES = [f"p{i}" for i in range(12)]


def rand_user(rng, i):
    stays, trips = [], []
    t = 0
    for _ in range(int(rng.integers(0, 40))):
        p = PLACES[int(rng.integers(len(PLACES)))]
        d = int(rng.integers(60, 20000))
        stays.append(Stay(p, "c", "h", t, t + d))
        t += d + 10
    for _ in range(int(rng.integers(0, 30))):
        a, b = rng.choice(PLACES, 2)
        trips.append(Trip(str(a), str(b), t, t + int(rng.integers(10, 900)), "walk"))
        t += 1000
    return UserRecord(f"u{i}", stays, trips)


def users(seed, n=30):
    rng = np.random.default_rng(seed)
    return [rand_user(rng, i) for i in range(n)]


def test_cap_deterministic_and_bounded():
    recs = users(1)
    a, rep = dp.cap_user_records(recs, 5, 4, 3, np.random.default_rng(0))
    b, _ = dp.cap_user_records(recs, 5, 4, 3, np.random.default_rng(0))
    assert a == b
    for r in a:
        assert len(r.stays) <= 5 and len(r.trips) <= 4
        assert len({s.place_id for s in r.stays}) <= 3
    assert rep.stays_after <= rep.stays_before and 0 <= rep.stays_dropped_frac <= 1
    assert rep.trips_dropped_frac >= 0


def test_cap_keeps_longest_places():
    r = UserRecord("u", [Stay("a", "c", "h", 0, 100), Stay("b", "c", "h", 200, 210),
                         Stay("c", "c", "h", 300, 500)])
    out, _ = dp.cap_user_records([r], 10, 10, 2, np.random.default_rng(0))
    assert {s.place_id for s in out[0].stays} == {"a", "c"}


def test_cap_user_independent_of_others():
    recs = users(2)
    a, _ = dp.cap_user_records(recs, 5, 4, 3, np.random.default_rng(0))
    for i in range(0, len(recs), 7):
        # distinct-place + longest-dwell selection ignores other users (ties aside)
        one, _ = dp.cap_user_records([recs[i]], 5, 4, 3, np.random.default_rng(0))
        assert {s.place_id for s in one[0].stays} == {s.place_id for s in a[i].stays}


KINDS = [
    ("presence", dp.presence_by_place, lambda: [(p,) for p in PLACES], dict(max_keys=4)),
    ("dwell", dp.dwell_by_place, lambda: [(p,) for p in PLACES],
     dict(max_keys=3, per_key_clip=5000.0)),
    ("transition", dp.transitions, lambda: [(a, b) for a in PLACES for b in PLACES],
     dict(max_keys=5, per_key_clip=2.0)),
]


@pytest.mark.parametrize("kind,spec,vocab_fn,kw", KINDS)
def test_sensitivity_add_remove_one_user(kind, spec, vocab_fn, kw):
    """Removing any single user changes the true table by at most l2_sensitivity."""
    for seed in range(15):
        recs = users(seed)
        full = dp.build_count_table(recs, spec, vocab_fn(), kind, **kw)
        bound = full.l2_sensitivity
        for i in range(len(recs)):
            rest = dp.build_count_table(recs[:i] + recs[i + 1:], spec, vocab_fn(), kind, **kw)
            diff = np.linalg.norm(full.true_values - rest.true_values)
            assert diff <= bound + 1e-9
        assert full.observed_max_norm <= bound + 1e-9


def test_sensitivity_formulas():
    t = dp.build_count_table([], dp.presence_by_place, [("a",)], "presence", max_keys=9)
    assert t.l2_sensitivity == 3.0
    t = dp.build_count_table([], dp.dwell_by_place, [("a",)], "dwell", max_keys=4,
                             per_key_clip=10)
    assert t.l2_sensitivity == 20.0
    with pytest.raises(ValueError):
        dp.build_count_table([], dp.dwell_by_place, [("a",)], "dwell")


def test_oov_dropped():
    recs = users(3)
    t = dp.build_count_table(recs, dp.presence_by_place, [("p0",)], "presence")
    assert all(set(v) <= {0} for v in t.user_vectors) and t.dropped_out_of_vocab > 0


def test_noise_std_matches_sigma_and_rho():
    t = dp.build_count_table([], dp.presence_by_place, [(str(i),) for i in range(20000)],
                             "presence", max_keys=4)
    m = dp.gaussian_measure(t, sigma=3.0, rng=np.random.default_rng(0), name="x")
    v = np.array(m.values)
    assert abs(v.std() - 3.0) < 0.06 and abs(v.mean()) < 0.1
    assert m.rho == pytest.approx(4.0 / 18.0)
    m2 = dp.gaussian_measure(t, rho=0.5, rng=np.random.default_rng(0), name="y")
    assert m2.sigma == pytest.approx(2.0)
    assert m2.rho == pytest.approx(0.5)


def test_measure_validation():
    t = dp.build_count_table([], dp.presence_by_place, [("a",)], "presence", max_keys=4)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        dp.gaussian_measure(t, sigma=1, rho=1, rng=rng)
    with pytest.raises(ValueError):
        dp.gaussian_measure(t, l2_sensitivity=1.0, sigma=1, rng=rng)
    with pytest.raises(ValueError):
        dp.gaussian_measure(t, sigma=1, rng=rng, name="Bad Name")


def test_postprocessing():
    t = dp.build_count_table([], dp.presence_by_place, [(str(i),) for i in range(50)],
                             "presence")
    m = dp.gaussian_measure(t, sigma=2.0, rng=np.random.default_rng(1), name="z")
    assert min(dp.project_nonneg(m).values) >= 0
    th = dp.threshold_release(m, 1.0)
    assert all(v >= 2.0 for v in th.values) and th.rho == m.rho


def test_accountant_and_ledger():
    t = dp.build_count_table([], dp.presence_by_place, [("a",)], "presence", max_keys=1)
    acc = dp.ZcdpAccountant(rho_budget=0.3)
    rng = np.random.default_rng(0)
    for r, n in zip(dp.split_budget(0.3, [1, 2]), ["a1", "a2"]):
        acc.spend(dp.gaussian_measure(t, rho=r, rng=rng, name=n))
    assert acc.rho_total == pytest.approx(0.3)
    assert acc.to_epsilon(1e-6) == pytest.approx(0.3 + 2 * math.sqrt(0.3 * math.log(1e6)))
    led = acc.ledger(1e-6, caps={"max_stays_per_user": 50}, caveats=["extra"])
    assert led.adjacency == "user" and len(led.entries) == 2 and "extra" in led.caveats
    with pytest.raises(ValueError):
        acc.spend(dp.gaussian_measure(t, rho=0.1, rng=rng, name="a3"))
    assert sum(dp.split_budget(1.0, [1, 1, 2])) == pytest.approx(1.0)


def test_hierarchical():
    recs = users(5, n=200)
    children = {("A",): [("p0",), ("p1",), ("p2",)], ("B",): [("p3",), ("p4",)],
                ("C",): [("p5",)]}
    cell = {"p0": "A", "p1": "A", "p2": "A", "p3": "B", "p4": "B", "p5": "C"}

    def pspec(r):
        return {(cell[s.place_id],): 1.0 for s in r.stays if s.place_id in cell}

    pm, kept, cm = dp.hierarchical_release(
        recs, pspec, dp.presence_by_place, children, "presence",
        parent_params=dict(max_keys=2), child_params=dict(max_keys=3), tau=1.0,
        rho_parent=1.0, rho_child=1.0, rng=np.random.default_rng(0))
    assert len(pm.keys) == 3
    allowed = {c for p in kept.keys for c in children[p]}
    assert set(cm.keys) == allowed
    assert pm.rho == pytest.approx(1.0) and cm.rho == pytest.approx(1.0)

"""Exact add/remove-one-user sensitivity of every v2 table, plus per-user determinism."""
import math

import numpy as np
import pytest

from geolife.transfer import schema_v2 as S2
from geolife.inside.measure_v2 import (VocabV2, children_of, make_view, pmf_table, pmf_vocab,
                                         presence_table)
from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users


def _counter(name):
    if name in S2.PLACE_TIERS:
        k = S2.PLACE_TIERS.index(name)
        return lambda v: v.place[k]
    k = S2.TRAV_TIERS.index(name)
    return lambda v: v.trav[k]


def _tables(views, vocab):
    """All tables, with vocabularies covering every cell a user touches (worst case)."""
    out = {}
    for name in S2.ALL_NAMES:
        if S2.KIND[name] == "presence":
            f = _counter(name)
            voc = [(c,) for c in sorted({c for v in views for c in f(v)})]
            out[name] = presence_table(views, f, voc, S2.max_keys_of(name))
        else:
            out[name] = pmf_table(views, name, pmf_vocab(name, vocab))
    return out


@pytest.fixture(scope="module")
def setup():
    users = make_users(30, seed=1)
    vocab = VocabV2.from_bbox(BBOX, CATS)
    return users, vocab, [make_view(u, vocab, 0) for u in users]


def test_stated_bound_holds_for_every_table(setup):
    _, vocab, views = setup
    for name, t in _tables(views, vocab).items():
        assert t.l2_sensitivity == pytest.approx(S2.l2_bound(name))
        assert t.observed_max_norm <= S2.l2_bound(name) + 1e-9, name


@pytest.mark.parametrize("seed", range(5))
def test_property_remove_one_user(seed):
    """Removing a random user changes every true table by <= its stated bound (and exactly by
    that user's own vector, so the vocabulary is held fixed)."""
    rng = np.random.default_rng(seed)
    users = make_users(12, seed=100 + seed, pool=60)
    vocab = VocabV2.from_bbox(BBOX, CATS)
    views = [make_view(u, vocab, seed) for u in users]
    j = int(rng.integers(len(views)))
    for name, full in _tables(views, vocab).items():
        if S2.KIND[name] == "presence":
            f = _counter(name)
            voc = full.vocab
            sub = presence_table(views[:j] + views[j + 1:], f, voc, S2.max_keys_of(name))
        else:
            sub = pmf_table(views[:j] + views[j + 1:], name, full.vocab)
        d = np.linalg.norm(full.true_values - sub.true_values)
        assert d <= S2.l2_bound(name) + 1e-9, name


def test_presence_bound_is_tight_for_heavy_user(setup):
    _, vocab, views = setup
    f = _counter("trav_users_r7")
    voc = [(c,) for c in sorted({c for v in views for c in f(v)})]
    t = presence_table(views, f, voc, 5)
    assert max(len(v) for v in t.user_vectors) <= 5
    assert t.observed_max_norm == pytest.approx(math.sqrt(5))


def test_view_is_deterministic_and_local(setup):
    users, vocab, _ = setup
    a, b = make_view(users[3], vocab, 0), make_view(users[3], vocab, 0)
    assert a.place == b.place and a.trav == b.trav and a.pmf == b.pmf


def test_caps_enforced():
    users = make_users(5, seed=3, max_outings=400)
    vocab = VocabV2.from_bbox(BBOX, CATS)
    big = max(users, key=lambda u: len(u.outings))
    v = make_view(big, vocab, 0, caps={"outings": 20, "stop_cells": 4, "trav_cells": 30})
    assert len(v.place[2]) <= 4
    assert len(v.trav[1]) <= 30
    assert sum(v.pmf[S2.OUTING_STOPS].values()) <= 20
    assert sum(v.pmf[S2.USER_OUTINGS].values()) == 1       # uncapped statistic, one count


def test_hierarchy_children_cover_parent():
    import h3
    c = h3.latlng_to_cell(39.9, 116.4, 6)
    ch = children_of(c, 6)
    assert len(ch) == 7 and all(h3.cell_to_parent(x, 6) == c for x in ch)

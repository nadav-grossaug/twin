"""Offline tests of the empirical attack machinery (small fixtures, no real data)."""
import math

import numpy as np
import pytest

from geolife.eval import attacks as A
from geolife.inside.learn_v2 import run
from geolife.inside.measure_v2 import VocabV2
from geolife.transfer.artifact import load_artifact
from tests.geolife.inside.test_learn_v2_common import BBOX, CATS, make_users


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    users = make_users(30, seed=3)
    root = tmp_path_factory.mktemp("arts")
    arts = []
    for k in range(2):
        run(users, root / f"toy_seed{k}", epsilon=6.0, bbox=BBOX, categories=CATS, seed=k,
            verbose=False, with_snr=False)
        arts.append(load_artifact(root / f"toy_seed{k}"))
    vocab = VocabV2.from_bbox(BBOX, CATS)
    uv = A.build_user_vectors(users, arts[0], vocab, seed=0)
    return users, arts, uv


def test_user_vectors_shapes_and_bounds(world):
    users, arts, uv = world
    assert uv.users == [u.user for u in users]
    # sensor tables were not measured in this fixture, so none should be skipped or present
    assert all(not n.startswith("sensor_") for n in uv.tables)
    for m in arts[0].measurements:
        M = uv.tables[m.name]
        assert M.shape == (len(users), len(m.keys))
        # per-user L2 never exceeds the ledger sensitivity (the clipping bound)
        assert np.sqrt((M ** 2).sum(axis=1)).max() <= m.l2_sensitivity + 1e-9
    # the sum over users of the vectors is the TRUE table; the released one is that plus noise
    name = "outing_mode"
    m = next(x for x in arts[0].measurements if x.name == name)
    resid = np.asarray(m.values) - uv.tables[name].sum(axis=0)
    assert np.abs(resid).max() < 6 * m.sigma           # noise-sized, not signal-sized


def test_informed_mi_exact_and_bounded(world):
    users, arts, uv = world
    r = A.informed_mi(uv, rho_total=arts[0].ledger.rho_total)
    assert r["q"]["max"] <= r["q"]["bound_2rho"] + 1e-9
    assert 0.5 < r["auc"]["p50"] <= r["auc"]["max"] <= r["auc"]["bound"] <= 1.0
    # AUC formula: Phi(sqrt(Q/2))
    assert r["auc"]["max"] == pytest.approx(
        0.5 * (1 + math.erf(math.sqrt(r["q"]["max"] / 2) / math.sqrt(2))))
    assert 0.0 <= r["tpr_at_fpr_worst_user"]["0.01"] <= 1.0


def test_montecarlo_mi_below_informed(world):
    users, arts, uv = world
    inf = A.informed_mi(uv, rho_total=arts[0].ledger.rho_total)
    mc = A.montecarlo_mi(uv, n_worlds=200, seed=1)
    assert 0.4 < mc["auc"]["p50"] < 1.0
    # the uninformed attacker cannot beat the informed one by more than MC noise
    assert mc["auc"]["p50"] <= inf["auc"]["max"] + 0.05


def test_seed_averaging_replicates(world):
    users, arts, uv = world
    r = A.seed_averaging(arts)
    assert r["n_seeds"] == 2 and r["noise_reduction_factor"] == pytest.approx(math.sqrt(2))
    assert r["rho_k_seeds"] == pytest.approx(2 * r["rho_single"])
    assert r["epsilon_k_seeds"] > r["epsilon_single"]
    tabs = {t["table"]: t for t in r["tables"]}
    assert "outing_mode" in tabs
    # cross-seed std estimates the ledger sigma (loose: 2 seeds only)
    ratios = [t["std_ratio"] for t in r["tables"] if t["n_keys"] >= 6]
    assert 0.5 < float(np.median(ratios)) < 1.6


def test_linkage_same_users_look_like_copies():
    """Attacking the real data with ITSELF must light every alarm; disjoint users must not."""
    users = make_users(26, seed=5)
    copy = A.top_cell_linkage(users, users, seed=0)
    assert copy["best_jaccard_syn"]["mean"] == pytest.approx(1.0)
    assert copy["reid_rate_syn"] == pytest.approx(1.0)
    nu = A.nearest_user_distance(users, users, seed=0)
    assert nu["synthetic"]["median"] == pytest.approx(0.0, abs=1e-12) and nu["too_close"]
    other = make_users(26, seed=11)
    disj = A.top_cell_linkage(users, other, seed=0)
    assert disj["best_jaccard_syn"]["mean"] < copy["best_jaccard_syn"]["mean"]
    nu2 = A.nearest_user_distance(users, other, seed=0)
    assert nu2["ratio_median"] > nu["ratio_median"]


def test_signature_linkage_runs_and_is_bounded():
    users = make_users(30, seed=7)
    other = make_users(30, seed=8)
    r = A.signature_linkage(users, other, seed=0)
    assert 0.0 <= r["rate_syn"] <= 1.0 and -1.0 <= r["excess"] <= 1.0
    full = A.signature_linkage(users, users, seed=0)
    assert full["rate_syn"] == pytest.approx(1.0)

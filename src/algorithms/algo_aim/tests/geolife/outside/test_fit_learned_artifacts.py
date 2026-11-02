"""PLAN step 4: the fits must be stable over the committed noise replicates (outings_v2_seed0-2),
and the planted-statistics check must pass on LEARNED artifacts, not only on the fake params.
Skipped when the committed artifacts are not present (they are in git, so normally they are)."""
import math
from pathlib import Path

import numpy as np
import pytest

from geolife.outside.fit_v2 import fit_behaviour_v2
from geolife.outside.grid_city import GridCityProvider
from geolife.outside.pipeline_v2 import generate_v2, planted_stats
from geolife.transfer.artifact import load_artifact

ART = Path("geolife/transfer/artifacts")
SEEDS = [ART / f"outings_v2_seed{k}" for k in range(3)]
pytestmark = pytest.mark.skipif(not (SEEDS[0] / "manifest.json").exists(),
                                reason="committed artifacts not present")


@pytest.fixture(scope="module")
def fits():
    return [fit_behaviour_v2(a.measurements, a.ledger) for a in map(load_artifact, SEEDS)]


def _ln_mean(s: dict) -> float:
    return math.exp(s["mu"] + s["sigma"] ** 2 / 2)


def _spread(vals) -> float:
    return (max(vals) - min(vals)) / float(np.mean(vals))


def test_fit_spread_over_noise_replicates(fits):
    """The three seeds are noise replicates of the SAME data: key fitted parameters must agree to
    well under a factor of two (the old moment-matched fits gave 3x spreads in the means)."""
    assert _spread([_ln_mean(p["users"]["outings_per_user"]) for p in fits]) < 0.8
    assert _spread([_ln_mean(p["outing"]["length_km"]) for p in fits]) < 0.5
    assert _spread([_ln_mean(p["outing"]["duration_min"]) for p in fits]) < 0.9
    assert _spread([p["users"]["revisit_rate"] for p in fits]) < 0.3
    for p in fits:
        assert 0.8 <= p["users"]["outings_per_user"]["sigma"] <= 1.6
        assert 0.3 <= p["users"]["revisit_rate"] <= 0.7
        assert p["route"]["detour_ratio"]["mu"] < 0.0     # excess convention: median excess < 1


def test_planted_stats_pass_and_volume_stable_on_learned_artifacts():
    prov = GridCityProvider(n=40, spacing=250, n_places=6000, seed=1)
    sizes = []
    for k, d in enumerate(SEEDS):
        g = generate_v2(d, prov, 30, k, gen={"compute_cells": False})
        sizes.append(sum(len(v) for v in g.result.traces.values()))
        st = planted_stats(g)
        bad = {kk: v for kk, v in st.items() if kk != "info" and not v["ok"]}
        assert not bad, (d.name, bad)
    # the replicates must give comparable output volumes (the old fits gave 2.1M-3.8M points for
    # 15 users / 6M-29M for 50 across seeds)
    assert _spread(sizes) < 1.0, sizes

"""Stage 3 on a tiny stage-2 output: planted-truth scoring runs from the files alone."""
import json

import pytest

from geolife.eval.score import main, score_variant
from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.pipeline_v2 import generate_v2
from geolife.outside.providers import make_provider
from geolife.outside.synthetic_io import read_synthetic, write_synthetic

SPEC = {"kind": "grid", "n": 20, "spacing": 250.0, "n_places": 1200, "seed": 1}
GEN = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 40, "sample_interval_s": 20, "compute_cells": False}


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    g = generate_v2(fake_params_v2(), make_provider(SPEC), 20, 3, gen=GEN)
    meta = {"variant": "toy", "artifact": "toy_seed0", "seed": 3, "n_users": 20, "map": SPEC, "epsilon": 6.0}
    write_synthetic(root / "toy_seed0", g.result, meta)
    (root / "index.json").write_text(json.dumps([{"variant": "toy", "seed": 3, "dir": "toy_seed0"}]))
    return root, g


def test_roundtrip_preserves_traces_and_truth(syn):
    root, g = syn
    r = read_synthetic(root / "toy_seed0").result
    assert r.traces == g.result.traces
    assert r.outings == g.result.outings
    assert r.truth.visits == g.result.truth.visits
    assert r.truth.groups == g.result.truth.groups
    assert r.truth.co_traveled == g.result.truth.co_traveled


def test_score_matches_in_memory_evaluation(syn, tmp_path):
    root, g = syn
    r = score_variant(root / "toy_seed0")
    from geolife.eval.m.outings import M2Config, run_m2
    from geolife.eval.m.outings_transfer import evaluate, make_mv
    live = evaluate(g, run_m2(g.result.traces, make_mv(make_provider(SPEC), True), M2Config()))
    assert json.dumps(live, sort_keys=True, default=float) == json.dumps(r["eval"], sort_keys=True, default=float)
    assert r["variant"] == "toy" and "poi" in r["eval"] and r["eval"]["n_users_kept"] > 0
    assert main(["--synthetic", str(root), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "toy_seed0.json").is_file() and (tmp_path / "summary.csv").is_file()

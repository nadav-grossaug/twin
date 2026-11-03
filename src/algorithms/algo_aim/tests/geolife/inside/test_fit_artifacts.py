"""Stage 1 end to end on synthetic stand-ins for the real inputs: points.parquet + public places -> artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geolife.outside.fakedata_v2 import fake_generation_v2, fake_params_v2
from geolife.outside.pipeline_v2 import generate_v2
from geolife.outside.providers import make_provider
from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage

SCRIPT = Path(__file__).resolve().parents[3] / "geolife" / "inside" / "fit_artifacts.py"
SPEC = {"kind": "grid", "n": 20, "spacing": 250.0, "n_places": 1200, "seed": 1}


def _script():
    spec = importlib.util.spec_from_file_location("fit_artifacts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage1_writes_validated_artifacts(tmp_path):
    prov = make_provider(SPEC)
    gen = {**fake_generation_v2(), "length_scale": 0.3, "max_outings": 40, "sample_interval_s": 20,
           "compute_cells": False}
    g = generate_v2(fake_params_v2(), prov, 24, 3, gen=gen)
    rows = [(uid, p.t, p.lat, p.lon) for uid, pts in g.result.traces.items() for p in pts]
    pd.DataFrame(rows, columns=["uid", "t", "lat", "lon"]).to_parquet(tmp_path / "points.parquet")
    pl = prov.places()
    lat, lon = prov.to_latlon(np.array([p.x for p in pl]), np.array([p.y for p in pl]))
    pd.DataFrame({"id": [p.id for p in pl], "category": [p.category for p in pl], "lat": lat, "lon": lon,
                  "popularity": [p.weight for p in pl]}).to_parquet(tmp_path / "places.parquet")
    out = tmp_path / "artifacts"
    assert _script().main(["--points", str(tmp_path / "points.parquet"), "--places", str(tmp_path / "places.parquet"),
                           "--data", str(tmp_path), "--out", str(out), "--variant", "toy", "--seeds", "2"]) == 0
    index = json.loads((out / "index.json").read_text())
    assert [e["dir"] for e in index] == ["toy_seed0", "toy_seed1"]
    for e in index:
        art = load_artifact(out / e["dir"])
        validate_no_raw_leakage(art)
        assert abs(art.ledger.epsilon - 6.0) < 1e-3 and art.ledger.adjacency == "user"
        assert any("outing gap" in c for c in art.ledger.caveats)           # public settings are listed
        assert any("DATA-DEPENDENT" in c for c in art.ledger.caveats)       # dropped tables, coarse time

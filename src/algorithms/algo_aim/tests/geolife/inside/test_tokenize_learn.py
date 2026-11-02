"""learn.py end-to-end on a tiny synthetic points.parquet (no real data, no network)."""
import numpy as np
import pandas as pd
import pytest

from geolife.inside import dp
from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage
from geolife.inside.data import default_data_dir
from geolife.inside.learn import h3_vocab, main, rho_for_epsilon, run
from geolife.inside.loaders import load_points_parquet

LAT, LON = 39.98, 116.30
BOX = (39.95, 40.02, 116.25, 116.35)


def _points(n_users=12, days=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    kx, ky = 111195.0 * np.cos(np.radians(LAT)), 111195.0
    for u in range(n_users):
        hx, hy = rng.normal(0, 800), rng.normal(0, 800)
        for d in range(days):
            t = 1_200_000_000 + d * 86400 + u * 7
            ax, ay = hx, hy
            bx, by = 4000.0, 1500.0           # shared "work" place
            legs = [(ax, ay, ax, ay, 1500), (ax, ay, bx, by, 600), (bx, by, bx, by, 1800),
                    (bx, by, ax, ay, 600), (ax, ay, ax, ay, 1500)]
            for x0, y0, x1, y1, dur in legs:
                for k in range(dur // 5):
                    f = k / max(1, dur // 5 - 1)
                    x = x0 + (x1 - x0) * f + rng.normal(0, 4)
                    y = y0 + (y1 - y0) * f + rng.normal(0, 4)
                    rows.append((f"{u:03d}", d, t, LAT + y / ky, LON + x / kx))
                    t += 5
    return pd.DataFrame(rows, columns=["uid", "traj", "t", "lat", "lon"])


@pytest.fixture(scope="module")
def pq(tmp_path_factory):
    d = tmp_path_factory.mktemp("geo")
    _points().to_parquet(d / "points.parquet")
    return d


def test_rho_for_epsilon_inverts_accountant():
    for eps in (1.0, 6.0, 8.0):
        assert dp.zcdp_to_epsilon(rho_for_epsilon(eps, 1e-6), 1e-6) == pytest.approx(eps, rel=1e-6)


def test_h3_vocab_public_and_sorted():
    v = h3_vocab(BOX, 8)
    assert len(v) > 100 and v == sorted(v) and all(c == "unknown" for _, c in v)


def test_load_points_parquet_thins_and_filters(pq):
    t = load_points_parquet(pq / "points.parquet", BOX, thin_s=10)
    assert len(t) == 12 and all(len(tr) > 10 for tr in t["000"])
    assert load_points_parquet(pq / "points.parquet", (0, 1, 0, 1)) == {}


def test_run_writes_valid_artifact(pq, tmp_path):
    out = tmp_path / "art"
    res = run(pq, out, epsilon=6.0, bbox=BOX, seed=1, caps=(50, 50, 10), verbose=False)
    art = load_artifact(out)
    validate_no_raw_leakage(art)
    assert art.ledger.epsilon == pytest.approx(6.0, rel=1e-3)
    assert art.ledger.adjacency == "user"
    names = {m.name for m in art.measurements}
    assert {"n_users", "role_users", "place_role_users", "dwell_hist", "detour"} <= names
    assert not any(r.user.isdigit() for r in res["records"])   # pseudonyms
    n = next(m for m in art.measurements if m.name == "n_users")
    assert abs(n.values[0] - 12) < 8 * n.sigma
    assert len(res["snr"]) == len(art.measurements)
    rows = {r["name"]: r for r in res["snr"]}
    assert rows["n_users"]["true_total"] == pytest.approx(12, abs=1e-3)


def test_cli(pq, tmp_path):
    out = tmp_path / "a2"
    assert main(["--data", str(pq), "--out", str(out), "--rho", "0.5", "--bbox", *map(str, BOX),
                 "--report", str(tmp_path / "r.json")]) == 0
    assert (out / "manifest.json").exists() and (tmp_path / "r.json").exists()


def test_default_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOLIFE_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path
    (tmp_path / "Geolife Trajectories 1.3").mkdir()
    assert default_data_dir() == tmp_path / "Geolife Trajectories 1.3"

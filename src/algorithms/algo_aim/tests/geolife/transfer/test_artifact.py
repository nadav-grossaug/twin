import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geolife.transfer.artifact import (ArtifactError, LeakageError, load_artifact, save_artifact,
                              validate_no_raw_leakage)
from geolife.transfer.inspect import render_report
from geolife.transfer.types import ArtifactManifest, Measurement, PrivacyLedger

SRC = Path(__file__).resolve().parents[3] / "src"


def _meas(name, sigma=2.0, sens=1.0, keys=None):
    keys = keys or [("a", "b"), ("b", "a"), ("a", "a")]
    return Measurement(name, keys, [10.123456789, 1.0, -0.5][:len(keys)], sigma, sens,
                       sens ** 2 / (2 * sigma ** 2), "n")


def _ledger(ms, delta=1e-6):
    rho = sum(m.rho for m in ms)
    return PrivacyLedger(rho_total=rho, epsilon=rho + 2 * math.sqrt(rho * math.log(1 / delta)),
                         delta=delta, caps={"max_stays": 50},
                         entries=[{"name": m.name, "rho": m.rho, "sigma": m.sigma,
                                   "l2_sensitivity": m.l2_sensitivity} for m in ms],
                         caveats=["h3 snapping not accounted"])


def build(tmp, ms=None, schema=None, ledger=None):
    ms = ms or [_meas("transitions"), _meas("users_per_place")]
    save_artifact(tmp, ArtifactManifest(created="x"), schema or {"levels": ["cat", "cell"]},
                  {"h3_resolution": 8}, ms, ledger or _ledger(ms), {"dwell": {"mean": 3.5}},
                  {"emb": np.arange(6, dtype=np.float32).reshape(3, 2)},
                  {"extent_m": 9000.0}, {"snr_min": 0.1}, {"n_days": 1})
    return ms


def test_roundtrip_exact(tmp_path):
    ms = build(tmp_path)
    a = load_artifact(tmp_path)
    assert a.measurements == sorted(ms, key=lambda m: m.name)
    assert a.behaviour == {"dwell": {"mean": 3.5}}
    assert np.array_equal(a.embeddings["emb"], np.arange(6, dtype=np.float32).reshape(3, 2))
    assert a.ledger == _ledger(ms)
    validate_no_raw_leakage(a)
    assert not list(tmp_path.rglob("*.pkl"))


def test_tamper_missing_version_extra(tmp_path):
    build(tmp_path)
    f = tmp_path / "schema.json"
    orig = f.read_bytes()
    f.write_bytes(orig + b" ")
    with pytest.raises(ArtifactError, match="hash"):
        load_artifact(tmp_path)
    f.write_bytes(orig)
    load_artifact(tmp_path)
    (tmp_path / "extra.txt").write_text("x")
    with pytest.raises(ArtifactError, match="unlisted"):
        load_artifact(tmp_path)
    (tmp_path / "extra.txt").unlink()
    f.unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(tmp_path)
    f.write_bytes(orig)
    mp = tmp_path / "manifest.json"
    d = json.loads(mp.read_text())
    d["format_version"] = "9.9"
    mp.write_text(json.dumps(d))
    with pytest.raises(ArtifactError, match="format_version"):
        load_artifact(tmp_path)


@pytest.mark.parametrize("bad", ["../x", "A", "a-b", "a.b", "", "a b"])
def test_bad_names(tmp_path, bad):
    with pytest.raises(ValueError):
        build(tmp_path, ms=[_meas(bad)])


def test_leak_coordinates_and_ids(tmp_path):
    build(tmp_path, schema={"origin": {"lat": 39.9042, "lon": 116.4074}})
    with pytest.raises(LeakageError, match="coordinate"):
        validate_no_raw_leakage(load_artifact(tmp_path))
    shutil.rmtree(tmp_path)
    build(tmp_path, schema={"pts": [[39.90421, 116.40741], [39.9, 116.4]]})
    with pytest.raises(LeakageError, match="raw"):
        validate_no_raw_leakage(load_artifact(tmp_path))
    shutil.rmtree(tmp_path)
    ms = [_meas("transitions", keys=[("017", "x")])]
    build(tmp_path, ms=ms)
    with pytest.raises(LeakageError, match="user id"):
        validate_no_raw_leakage(load_artifact(tmp_path))
    validate_no_raw_leakage(load_artifact(tmp_path), user_id_regex=r"^u-\d+$")


def test_privacy_inconsistency(tmp_path):
    ms = [_meas("transitions")]
    led = _ledger(ms)
    led.epsilon += 1
    led.adjacency = "record"
    build(tmp_path, ms=ms, ledger=led)
    with pytest.raises(LeakageError) as e:
        validate_no_raw_leakage(load_artifact(tmp_path))
    assert "adjacency" in str(e.value) and "epsilon" in str(e.value)
    shutil.rmtree(tmp_path)
    led = _ledger(ms)
    led.rho_total *= 2
    build(tmp_path, ms=ms, ledger=led)
    with pytest.raises(LeakageError, match="rho_total"):
        validate_no_raw_leakage(load_artifact(tmp_path))


def test_provenance_allowlist_rejects_unlisted_keys(tmp_path):
    """A free-form json file must not carry an un-noised statistic such as a user count."""
    ms = [_meas("transitions")]
    save_artifact(tmp_path, ArtifactManifest(created="x"), {}, {"n_users_hint": 182}, ms, _ledger(ms), {}, None,
                  {}, {}, {})
    with pytest.raises(LeakageError, match="n_users_hint"):
        validate_no_raw_leakage(load_artifact(tmp_path))


def test_inspect(tmp_path):
    build(tmp_path / "art")
    a = load_artifact(tmp_path / "art")
    txt = render_report(a)
    assert "LOW-SNR" in txt and "emb: shape (3, 2)" in txt
    assert "caveats" in txt.lower() and "| name" in render_report(a, markdown=True)


def test_cli(tmp_path):
    build(tmp_path / "art")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    for cmd in ("verify", "inspect"):
        r = subprocess.run([sys.executable, "-m", "geolife.transfer.artifact_cli", cmd, str(tmp_path / "art")],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
    (tmp_path / "art" / "schema.json").write_text("{}")
    r = subprocess.run([sys.executable, "-m", "geolife.transfer.artifact_cli", "verify", str(tmp_path / "art")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1


CLEAN = r"""
import sys, os
reads = []
def hook(ev, args):
    if ev == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        reads.append(os.path.abspath(os.fsdecode(args[0])))
sys.addaudithook(hook)
from geolife.transfer.artifact import load_artifact, validate_no_raw_leakage
start = len(reads)
art = load_artifact(sys.argv[1]); validate_no_raw_leakage(art)
root = os.path.abspath(sys.argv[1])
CODE = (".py", ".pyc", ".pyd", ".dll", ".so")
# the tz database is benign (datetime/pandas resolve UTC through it on some machines); everything
# else outside the artifact dir and the interpreter's own code still fails the test
bad = [p for p in reads[start:] if not p.startswith(root) and not p.lower().endswith(CODE) and "site-packages" not in p and not p.endswith("nul") and "/zoneinfo/" not in p.replace(os.sep, "/")]
assert not bad, bad
assert any(p.startswith(root) for p in reads[start:])
print("CLEAN", len(art.measurements))
"""


def test_clean_room(tmp_path):
    build(tmp_path / "orig")
    iso = tmp_path / "iso"
    iso.mkdir()
    shutil.copytree(tmp_path / "orig", iso / "artifact")
    shutil.rmtree(tmp_path / "orig")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    r = subprocess.run([sys.executable, "-c", CLEAN, str(iso / "artifact")], cwd=iso,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "CLEAN 2" in r.stdout

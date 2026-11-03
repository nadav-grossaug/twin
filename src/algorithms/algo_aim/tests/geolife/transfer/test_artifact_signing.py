"""HMAC signing of artifacts (PLAN step 7): sign/verify round trip, tamper and wrong-key detection,
backward compatibility (unsigned artifacts stay valid), and the CLI commands."""
import json

import pytest

from geolife.transfer.artifact import (SIGNATURE_FILE, ArtifactError, load_artifact, sign_artifact,
                                       verify_signature)
from geolife.transfer.artifact_cli import main as cli
from tests.geolife.transfer.test_artifact import build

KEY = b"0123456789abcdef0123456789abcdef"


@pytest.fixture
def signed(tmp_path):
    art = tmp_path / "art"
    build(art)
    key = tmp_path / "artifact.key"
    key.write_bytes(KEY + b"\n")           # trailing newline must not matter
    sign_artifact(art, key)
    return art, key


def test_sign_verify_roundtrip(signed):
    art, key = signed
    verify_signature(art, key)
    a = load_artifact(art, key_path=key)   # signature file is not an "unlisted file"
    assert a.ledger.rho_total > 0
    sig = json.loads((art / SIGNATURE_FILE).read_text())
    assert sig["algorithm"] == "HMAC-SHA256" and sig["signs"] == "manifest.json"
    load_artifact(art)                     # loading without a key still works


def test_wrong_key_and_tampered_manifest_fail(signed, tmp_path):
    art, key = signed
    bad = tmp_path / "bad.key"
    bad.write_bytes(b"f" * 32)
    with pytest.raises(ArtifactError, match="signature mismatch"):
        verify_signature(art, bad)
    mp = art / "manifest.json"
    mp.write_bytes(mp.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="signature mismatch"):
        load_artifact(art, key_path=key)


def test_tampered_table_is_caught_via_manifest_hash(signed):
    """The MAC covers manifest.json; manifest.files covers everything else, so editing a table and
    re-hashing it in the manifest breaks the MAC, and editing it without re-hashing breaks the hash."""
    art, key = signed
    f = art / "schema.json"
    f.write_bytes(f.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_artifact(art, key_path=key)


def test_unsigned_artifact_with_key_fails_and_without_key_loads(tmp_path):
    art = tmp_path / "art"
    build(art)
    key = tmp_path / "k"
    key.write_bytes(KEY)
    load_artifact(art)
    with pytest.raises(ArtifactError, match="not signed"):
        load_artifact(art, key_path=key)


def test_short_key_rejected(tmp_path):
    art = tmp_path / "art"
    build(art)
    key = tmp_path / "k"
    key.write_bytes(b"short")
    with pytest.raises(ArtifactError, match="too short"):
        sign_artifact(art, key)


def test_garbage_signature_file(signed):
    art, key = signed
    (art / SIGNATURE_FILE).write_text("{\"algorithm\": \"MD5\", \"signs\": \"manifest.json\", \"mac\": \"00\"}")
    with pytest.raises(ArtifactError, match="unsupported signature format"):
        verify_signature(art, key)
    (art / SIGNATURE_FILE).write_text("not json")
    with pytest.raises(ArtifactError, match="unreadable"):
        verify_signature(art, key)


def test_cli_sign_and_verify(tmp_path, capsys):
    art = tmp_path / "art"
    build(art)
    key = tmp_path / "k"
    key.write_bytes(KEY)
    assert cli(["sign", str(art), "--key", str(key)]) == 0
    assert cli(["verify", str(art), "--key", str(key)]) == 0
    assert "signature valid" in capsys.readouterr().out
    bad = tmp_path / "bad"
    bad.write_bytes(b"f" * 32)
    assert cli(["verify", str(art), "--key", str(bad)]) == 1
    assert cli(["verify", str(art)]) == 0     # unsigned-style verify still passes

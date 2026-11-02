"""Enclave platform abstraction (platform not yet selected).

The release layer depends only on these interfaces:
  * AttestationProvider.evidence(nonce) -> Evidence
  * Verifier.verify(evidence, nonce)    -> VerifiedClaims (raises on failure)
  * EgressGuard.check(path)             (restricted egress: only release/ is writable)

`DevNoEnclave` is NOT an enclave: it produces unattested evidence so the
pipeline can be exercised on a laptop. Release refuses it unless explicitly
allowed (GEO_ALLOW_DEV_RELEASE=1), and marks the export `attested: false`.
`SimulatedTEE` is a test double with HMAC-signed evidence used to test the
attestation / egress logic; it provides no hardware protection. A real
deployment implements these classes for the selected platform (e.g. SEV-SNP /
TDX confidential VMs, Nitro Enclaves, SGX via Gramine).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from provenance import code_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_ROOT = PROJECT_ROOT / "release"


class AttestationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Evidence:
    platform: str
    measurement: str     # hash of the code/config loaded in the protected boundary
    nonce: str
    signature: str


@dataclass(frozen=True)
class VerifiedClaims:
    platform: str
    measurement: str
    attested: bool


class AttestationProvider:
    def evidence(self, nonce: str) -> Evidence:  # pragma: no cover - interface
        raise NotImplementedError


class DevNoEnclave(AttestationProvider):
    def evidence(self, nonce: str) -> Evidence:
        return Evidence("none-dev", code_fingerprint(), nonce, signature="")


class SimulatedTEE(AttestationProvider):
    """Test double: signs (measurement, nonce) with a 'hardware' key."""

    def __init__(self, hw_key: bytes, measurement: str | None = None):
        self._key, self._measurement = hw_key, measurement or code_fingerprint()

    def evidence(self, nonce: str) -> Evidence:
        sig = hmac.new(self._key, f"{self._measurement}:{nonce}".encode(), hashlib.sha256).hexdigest()
        return Evidence("simulated-tee", self._measurement, nonce, sig)


class Verifier:
    def __init__(self, trusted_keys: dict[str, bytes], expected_measurements: set[str]):
        self.trusted_keys, self.expected = trusted_keys, expected_measurements

    def verify(self, ev: Evidence, nonce: str) -> VerifiedClaims:
        if ev.nonce != nonce:
            raise AttestationError("stale or replayed evidence (nonce mismatch)")
        key = self.trusted_keys.get(ev.platform)
        if key is None:
            raise AttestationError(f"untrusted platform {ev.platform!r}")
        want = hmac.new(key, f"{ev.measurement}:{nonce}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, ev.signature):
            raise AttestationError("bad evidence signature")
        if ev.measurement not in self.expected:
            raise AttestationError("measurement not in the approved set")
        return VerifiedClaims(ev.platform, ev.measurement, attested=True)


class EgressGuard:
    """Restricted egress: exported files may only be written under release/."""

    def __init__(self, root: Path = RELEASE_ROOT):
        self.root = Path(root).resolve()

    def check(self, path: Path) -> Path:
        p = Path(path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"egress outside release root refused: {p.name}")
        return p


def new_nonce() -> str:
    return os.urandom(16).hex()

"""Content-addressed, immutable model artifact store with an active registry.

Version id = sha256(name, data fingerprint, hyperparameters, code
fingerprint)[:12] — training with unchanged inputs is a no-op (the artifact
already exists); any change to data, hparams or code yields a new version.
Artifact files are chmod'd read-only after writing; overwrites are refused.
`registry.json` maps name -> {active, history}; a newly written version
becomes active only when nothing is active yet or BOOKING_RETRAIN=1
(goal.md guideline 3: explicit retraining rules).
"""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from provenance import code_fingerprint, dependency_versions, sha256_json

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "models_out"


def artifact_root() -> Path:
    return Path(os.environ.get("BOOKING_ARTIFACT_ROOT", str(DEFAULT_ROOT)))


def candidate_version(name: str, data_fingerprint: str, hparams: dict,
                      code_paths: list[Path]) -> str:
    return sha256_json({
        "name": name,
        "data": data_fingerprint,
        "hparams": hparams,
        "code": code_fingerprint(code_paths),
    })[:12]


def _registry_path() -> Path:
    return artifact_root() / "registry.json"


def load_registry() -> dict:
    p = _registry_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _save_registry(reg: dict) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2))


def artifact_dir(name: str, version: str) -> Path:
    return artifact_root() / name / version


def write_artifact(name: str, version: str, files: dict[str, bytes],
                   metadata: dict) -> Path:
    """Write an immutable artifact; refuses to overwrite an existing version."""
    vdir = artifact_dir(name, version)
    if vdir.exists():
        raise FileExistsError(f"{vdir} is immutable and already exists")
    vdir.mkdir(parents=True)
    meta = dict(metadata)
    meta.setdefault("name", name)
    meta.setdefault("version", version)
    meta.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta.setdefault("dependencies", dependency_versions())
    meta.setdefault("release_status", "internal_unreviewed")
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    for fname, blob in files.items():
        (vdir / fname).write_bytes(blob)
    ro = stat.S_IRUSR | stat.S_IRGRP
    for f in vdir.iterdir():
        f.chmod(ro)
    return vdir


def maybe_activate(name: str, version: str) -> bool:
    """Activate `version` only if nothing is active yet or BOOKING_RETRAIN=1."""
    reg = load_registry()
    entry = reg.setdefault(name, {"active": None, "history": []})
    if version not in entry["history"]:
        entry["history"].append(version)
    activate = entry["active"] is None or os.environ.get("BOOKING_RETRAIN") == "1"
    if activate:
        entry["active"] = version
    _save_registry(reg)
    return activate


def active_version(name: str) -> str:
    reg = load_registry()
    v = reg.get(name, {}).get("active")
    if not v:
        raise LookupError(f"no active artifact for {name!r} — run `make train`")
    return v


def load_active(name: str) -> tuple[str, Path, dict]:
    v = active_version(name)
    vdir = artifact_dir(name, v)
    meta = json.loads((vdir / "meta.json").read_text())
    return v, vdir, meta


def train_or_reuse(name: str, data_fingerprint: str, hparams: dict,
                   code_paths: list[Path], fit_fn) -> tuple[str, bool]:
    """Idempotent training: skip fit_fn entirely if the candidate exists.

    fit_fn() -> (files: dict[str, bytes], metadata: dict).
    Returns (version, trained_now).
    """
    version = candidate_version(name, data_fingerprint, hparams, code_paths)
    if artifact_dir(name, version).exists():
        maybe_activate(name, version)
        return version, False
    files, metadata = fit_fn()
    metadata = dict(metadata)
    metadata["data_fingerprint"] = data_fingerprint
    metadata["hparams"] = hparams
    metadata["code_fingerprint"] = code_fingerprint(code_paths)
    write_artifact(name, version, files, metadata)
    maybe_activate(name, version)
    return version, True

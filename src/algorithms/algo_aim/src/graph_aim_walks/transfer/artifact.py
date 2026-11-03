"""Artifact I/O and validation for graph-aim-walks (shipped to both sides of the boundary).

On-disk layout (JSON plus ``.npz``; no pickle, arrays loaded with ``allow_pickle=False``)::

    artifact/
      manifest.json                 format version, method = graph_aim_walks, created, sha256 of every other file
      base/                         the exact graph-AIM base artifact (graph_aim.transfer.artifact.save, own manifest)
      spec.json                     the WalkSpec (public parameters of the walk statistics)
      tables/<index>_<name>.npz     array ``values`` (normalised walk table, or the transition spectrum)
      tables/<index>_<name>.json    metadata: name, kind, shape, n_walks (walk tables) or k (spectrum)
      privacy.json                  "private": false and a plain-language warning
      provenance.json               allowlisted keys only

THIS ARTIFACT IS NOT PRIVATE. The tables are exact statistics of the real graph (up to walk-sampling error) and the
base artifact is fitted without noise. It must never be presented as a differentially private release.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graph_aim.transfer import artifact as base_io
from graph_aim.transfer.artifact import _read_json, _sha256, _write_json
from graph_aim_walks.transfer.types import METHOD, SPECTRUM, WalksArtifact, expected_shapes, table_names
from graph_aim_walks.transfer.walkstats import WalkSpec

FORMAT_VERSION = 1
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
_TOL = 1e-6

# Public settings and software facts only, never a statistic of the real graph.
PROVENANCE_ALLOWED = frozenset(
    {"method", "private", "config_sha256", "seed", "seed_policy", "n_walks", "spectrum_k",
     "versions"}
)

PRIVACY_WARNING = (
    "NOT PRIVATE. This artifact is not a differentially private release and gives no privacy guarantee of any kind. "
    "Every walk table is an exact statistic of the real graph (estimated from random walks on it, so exact up to "
    "sampling error), the transition spectrum is computed exactly from the real graph, and the base graph-AIM "
    "artifact under base/ is fitted with the noise switched off. Treat it like the real graph: it must not leave "
    "the inside and must never be presented as a private or anonymised release. It exists to answer a research "
    "question (can a generator that only sees readable walk tables produce graphs whose walks look real?)."
)


def privacy_record() -> dict:
    return {
        "private": False,
        "method": METHOD,
        "warning": PRIVACY_WARNING,
        "epsilon": None,
        "base": "base/privacy.json: noise off, infinite rho (not differentially private)",
    }


# ----------------------------------------------------------------------------- save / load


def _stem(i: int, name: str) -> str:
    return f"{i:03d}_{_UNSAFE_RE.sub('_', name)}"  # the real name lives in the JSON; the stem is path-safe


def save(art: WalksArtifact, path) -> Path:
    """Write ``art`` to the directory ``path`` (created; must not contain files already)."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    base_io.save(art.base, path / "base")
    (path / "tables").mkdir()
    _write_json(path / "spec.json", art.spec.to_json())

    for i, (name, values) in enumerate(art.tables.items()):
        v = np.asarray(values)
        if v.dtype.kind not in "biuf":
            raise TypeError(f"table {name!r}: dtype {v.dtype} not storable without pickle")
        stem = _stem(i, name)
        np.savez(path / "tables" / f"{stem}.npz", values=v)
        meta = {"name": name, "kind": _kind(name), "shape": list(v.shape)}
        if name == SPECTRUM:
            meta["k"] = art.provenance.get("spectrum_k", len(v))
            meta["matrix"] = "D^-1 A (eigenvalues of D^-1/2 A D^-1/2), isolated nodes dropped"
        else:
            meta["n_walks"] = int(art.n_walks)
        _write_json(path / "tables" / f"{stem}.json", meta)

    _write_json(path / "privacy.json", privacy_record())
    _write_json(path / "provenance.json", art.provenance)

    files = {p.relative_to(path).as_posix(): _sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    _write_json(
        path / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "method": METHOD,
            "private": False,
            "created": datetime.now(timezone.utc).isoformat(),
            "sha256": files,
        },
    )
    return path


def _kind(name: str) -> str:
    return name.split(":", 1)[0]


def load(path) -> WalksArtifact:
    """Load an artifact directory, verifying every hash (its own manifest and the base's). Raises ``ValueError``."""
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.is_file():
        raise ValueError(f"{path}: manifest.json missing")
    manifest = _read_json(mpath)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unknown artifact format version {manifest.get('format_version')!r}")
    if manifest.get("method") != METHOD:
        raise ValueError(f"unknown artifact method {manifest.get('method')!r} (expected {METHOD!r})")

    expected = manifest["sha256"]
    present = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} - {"manifest.json"}
    if present != set(expected):
        raise ValueError(
            f"file set differs from manifest: missing={sorted(set(expected) - present)}, "
            f"unexpected={sorted(present - set(expected))}"
        )
    for rel, digest in expected.items():
        if _sha256(path / rel) != digest:
            raise ValueError(f"hash mismatch for {rel}: artifact was modified or corrupted")

    priv = _read_json(path / "privacy.json")
    if priv.get("private") is not False:
        raise ValueError("privacy.json must say \"private\": false (graph_aim_walks artifacts are never private)")

    base = base_io.load(path / "base")
    spec = WalkSpec.from_json(_read_json(path / "spec.json"))

    tables: dict[str, np.ndarray] = {}
    n_walks: set[int] = set()
    for rel in sorted(expected):
        if not (rel.startswith("tables/") and rel.endswith(".json")):
            continue
        meta = _read_json(path / rel)
        with np.load(path / (rel[: -len(".json")] + ".npz"), allow_pickle=False) as z:
            values = z["values"]
        if meta["name"] in tables:
            raise ValueError(f"duplicate table {meta['name']!r}")
        tables[meta["name"]] = values
        if "n_walks" in meta:
            n_walks.add(int(meta["n_walks"]))
    if len(n_walks) > 1:
        raise ValueError(f"tables disagree on n_walks: {sorted(n_walks)}")
    return WalksArtifact(
        base=base,
        spec=spec,
        tables=tables,
        n_walks=n_walks.pop() if n_walks else 0,
        provenance=_read_json(path / "provenance.json"),
    )


# ----------------------------------------------------------------------------- validate


def _validate_table(name: str, v: np.ndarray, shape, art: WalksArtifact) -> list[str]:
    tag = f"table {name!r}"
    v = np.asarray(v)
    if v.dtype.kind not in "iuf":
        return [f"{tag}: dtype {v.dtype} is not numeric"]
    if not np.all(np.isfinite(v)):
        return [f"{tag}: values contain non-finite entries"]
    errs: list[str] = []
    if name == SPECTRUM:
        k = art.provenance.get("spectrum_k")
        n = art.base.schema.n
        if v.ndim != 1 or len(v) < 1:
            return [f"{tag}: must be a non-empty 1-D array, got shape {v.shape}"]
        if k is not None and len(v) > int(k):
            errs.append(f"{tag}: {len(v)} eigenvalues, more than spectrum_k = {k}")
        if isinstance(n, (int, np.integer)) and len(v) > max(n - 1, 1):
            errs.append(f"{tag}: {len(v)} eigenvalues for a graph of {n} nodes")
        if np.any(np.diff(v) > 1e-9):
            errs.append(f"{tag}: eigenvalues must be in descending order")
        if np.any(np.abs(v) > 1 + _TOL):
            errs.append(f"{tag}: eigenvalues of D^-1 A must lie in [-1, 1]")
        return errs
    if v.shape != shape:
        return [f"{tag}: shape {v.shape}, expected {shape}"]
    if np.any(v < 0):
        errs.append(f"{tag}: negative entries (min {v.min()})")
    T = art.spec.length
    if name == "return":
        if np.any(v > 1 + _TOL):
            errs.append(f"{tag}: a return probability exceeds 1")
    elif name == "coverage":
        t = np.arange(1, T + 1)
        if np.any(v < 1 - _TOL) or np.any(v > t + 1 + _TOL):
            errs.append(f"{tag}: coverage after t steps must lie in [1, t + 1]")
        if np.any(np.diff(v) < -_TOL):
            errs.append(f"{tag}: coverage must be non-decreasing in t")
    else:
        axes = (2,) if name.startswith("share:") else (1, 2)
        s = v.sum(axis=axes)
        if not np.allclose(s, 1.0, rtol=0, atol=_TOL):
            errs.append(f"{tag}: distributions do not sum to 1 (sums from {s.min():.6g} to {s.max():.6g})")
    return errs


def validate(art: WalksArtifact) -> list[str]:
    """Return a list of human-readable problems; empty means the artifact is consistent."""
    errs = [f"base: {e}" for e in base_io.validate(art.base)]
    extra = sorted(set(art.provenance) - PROVENANCE_ALLOWED)
    if extra:
        errs.append(f"provenance: keys {extra} are not on the allowlist (nothing computed from the real graph may go there)")
    if art.provenance.get("method", METHOD) != METHOD:
        errs.append(f"provenance: method {art.provenance.get('method')!r}, expected {METHOD!r}")
    if art.provenance.get("private", False) is not False:
        errs.append("provenance: 'private' must be false")
    if not (isinstance(art.n_walks, (int, np.integer)) and not isinstance(art.n_walks, bool) and art.n_walks >= 1):
        errs.append(f"n_walks must be a positive integer, got {art.n_walks!r}")

    domains = art.base.schema.domains
    spec = art.spec
    used = {c for g in spec.groups for c in g} | set(spec.share_attrs) | {c for p in spec.dominance_pairs for c in p}
    unknown = sorted(used - set(domains))
    if unknown:
        return errs + [f"spec: attributes {unknown} are not in the base schema domains"]
    if any(len(g) == 0 for g in spec.groups):
        errs.append("spec: empty attr_lag group")
    if spec.share_bins < 1 or spec.dominance_bins < 1:
        errs.append("spec: share_bins and dominance_bins must be >= 1")

    shapes = expected_shapes(spec, domains)
    want, got = table_names(spec), list(art.tables)
    if set(got) != set(want):
        errs.append(f"tables: missing {sorted(set(want) - set(got))}, unexpected {sorted(set(got) - set(want))}")
    for name, v in art.tables.items():
        if name in shapes:
            errs.extend(_validate_table(name, v, shapes[name], art))
    k = art.provenance.get("spectrum_k")
    if k is not None and not (isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= 1):
        errs.append(f"provenance: spectrum_k must be a positive integer, got {k!r}")
    nw = art.provenance.get("n_walks")
    if nw is not None and nw != art.n_walks:
        errs.append(f"provenance: n_walks {nw} differs from the artifact's n_walks {art.n_walks}")
    return errs

"""Artifact I/O and validation for the graph-AIM pipeline (shipped to both sides of the boundary).

On-disk layout (JSON plus ``.npz``; no pickle anywhere, arrays are loaded with
``allow_pickle=False``)::

    artifact/
      manifest.json      format version, method, created, sha256 of every other file
      schema.json
      privacy.json
      measurements/<index>_<name>.json   metadata (kind, sigma, sensitivities, params)
      measurements/<index>_<name>.npz    array ``values``
      provenance.json

Non-finite floats (``inf`` rho of non-private artifacts) are stored as the strings
``"inf"`` / ``"-inf"`` so the JSON stays standard.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graph_aim.transfer.binning import check_bin_edges
from graph_aim.transfer.types import Artifact, Measurement, PrivacyLedger, Schema

FORMAT_VERSION = 1
METHOD = "graph_aim_v1"
MAX_GROUPS = 64  # guards the (G, G) edge-pair table; G = product of the group's domain sizes
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")

# The only provenance keys an artifact may carry: public settings and software facts, never a statistic of the
# real data. (An exact count of capped edges used to be released here; it was not covered by the noise.)
PROVENANCE_ALLOWED = frozenset(
    {"method", "config_sha256", "seed", "seed_policy", "noise", "versions", "epsilon_edge", "epsilon_attr"}
)

ADJACENCY_DEFINITIONS = {
    "edge": (
        "Two graphs are edge-adjacent if they differ by one edge (added or removed). "
        "The node set and all attributes are identical."
    ),
    "attr": (
        "Two graphs are attribute-row-adjacent if they differ in the attribute vector of one "
        "node. The node set and all edges are identical."
    ),
}

_KINDS = ("degree_hist", "clustering_by_degree", "joint_degree", "attr_1way", "attr_2way", "attr_edge_pair", "attr_degree")


# ----------------------------------------------------------------------------- helpers


def _enc_float(x: float):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _jsonable(obj):
    """Convert numpy scalars/arrays and tuples to plain JSON types (non-finite floats to strings)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        return _enc_float(obj)
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    # bytes, not write_text: text mode would write CRLF on Windows and the artifact bytes (and their SHA-256)
    # would depend on the platform that fitted them
    text = json.dumps(_jsonable(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def zcdp_to_eps(rho: float, delta: float) -> float:
    """Standard zCDP -> (eps, delta)-DP: eps = rho + 2 sqrt(rho ln(1/delta)). inf for rho = inf."""
    rho = float(rho)
    if math.isinf(rho):
        return math.inf
    if rho <= 0:
        return 0.0
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def entry_rho(sigma: float, sensitivity: float) -> float:
    """zCDP cost s^2 / (2 sigma^2); 0 if the sensitivity is 0, inf if sigma is 0 and s > 0."""
    if sensitivity == 0:
        return 0.0
    if sigma == 0:
        return math.inf
    return sensitivity**2 / (2.0 * sigma**2)


# ----------------------------------------------------------------------------- save / load


def save(artifact: Artifact, path) -> Path:
    """Write ``artifact`` to the directory ``path`` (created; must not contain files already)."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory {path}")
    (path / "measurements").mkdir(parents=True, exist_ok=True)

    s = artifact.schema
    _write_json(
        path / "schema.json",
        {
            "n": int(s.n),
            "domains": s.domains,
            "degree_cap": int(s.degree_cap),
            "cap_salt": int(s.cap_salt),
            "homophily_attrs": list(s.homophily_attrs),
        },
    )

    led = artifact.ledger
    _write_json(
        path / "privacy.json",
        {
            "delta": led.delta,
            "rho_edge": led.rho_edge,
            "rho_attr": led.rho_attr,
            "derived": {
                "edge": {"epsilon": zcdp_to_eps(led.rho_edge, led.delta), "delta": led.delta},
                "attr": {"epsilon": zcdp_to_eps(led.rho_attr, led.delta), "delta": led.delta},
                "conversion": "eps = rho + 2*sqrt(rho*ln(1/delta)) (zCDP to (eps, delta)-DP)",
            },
            "adjacency": ADJACENCY_DEFINITIONS,
            "guarantee_scope": (
                "(eps, delta)-DP holds separately for each adjacency. No claim is made for a "
                "simultaneous change of edges and attributes."
            ),
            "entries": led.entries,
            "caveats": list(led.caveats),
        },
    )

    for i, (name, m) in enumerate(artifact.measurements.items()):
        stem = f"{i:03d}_{_UNSAFE_RE.sub('_', name)}"  # the real name lives in the JSON; the stem is path-safe
        values = np.asarray(m.values)
        if values.dtype.kind not in "biuf":
            raise TypeError(f"measurement {name!r}: dtype {values.dtype} not storable without pickle")
        np.savez(path / "measurements" / f"{stem}.npz", values=values)
        _write_json(
            path / "measurements" / f"{stem}.json",
            {
                "name": m.name,
                "kind": m.kind,
                "mechanism": m.mechanism,
                "sigma": m.sigma,
                "sensitivity_edge": m.sensitivity_edge,
                "sensitivity_attr": m.sensitivity_attr,
                "params": m.params,
            },
        )

    _write_json(path / "provenance.json", artifact.provenance)

    files = {p.relative_to(path).as_posix(): _sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    _write_json(
        path / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "method": METHOD,
            "created": datetime.now(timezone.utc).isoformat(),
            "sha256": files,
        },
    )
    return path


def load(path) -> Artifact:
    """Load an artifact directory, verifying every hash. Raises ``ValueError`` on any problem."""
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.is_file():
        raise ValueError(f"{path}: manifest.json missing")
    manifest = _read_json(mpath)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unknown artifact format version {manifest.get('format_version')!r}")
    if manifest.get("method") != METHOD:
        raise ValueError(f"unknown artifact method {manifest.get('method')!r}")

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

    sj = _read_json(path / "schema.json")
    schema = Schema(
        n=int(sj["n"]),
        domains={k: list(v) for k, v in sj["domains"].items()},
        degree_cap=int(sj["degree_cap"]),
        cap_salt=int(sj["cap_salt"]),
        homophily_attrs=list(sj["homophily_attrs"]),
    )

    pj = _read_json(path / "privacy.json")
    entries = [{k: (float(v) if k.startswith("rho") else v) for k, v in e.items()} for e in pj["entries"]]
    ledger = PrivacyLedger(
        delta=float(pj["delta"]),
        rho_edge=float(pj["rho_edge"]),
        rho_attr=float(pj["rho_attr"]),
        entries=entries,
        caveats=list(pj["caveats"]),
    )

    measurements: dict[str, Measurement] = {}
    for rel in sorted(expected):
        if not (rel.startswith("measurements/") and rel.endswith(".json")):
            continue
        meta = _read_json(path / rel)
        with np.load(path / rel.replace(".json", ".npz"), allow_pickle=False) as z:
            values = z["values"]
        name = meta["name"]
        measurements[name] = Measurement(
            name=name,
            kind=meta["kind"],
            values=values,
            mechanism=meta["mechanism"],
            sigma=float(meta["sigma"]),
            sensitivity_edge=float(meta["sensitivity_edge"]),
            sensitivity_attr=float(meta["sensitivity_attr"]),
            params=meta["params"],
        )
    return Artifact(
        schema=schema, measurements=measurements, ledger=ledger, provenance=_read_json(path / "provenance.json")
    )


# ----------------------------------------------------------------------------- validate


def _close(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def _validate_measurement(key: str, m: Measurement, schema: Schema) -> list[str]:
    errs: list[str] = []
    tag = f"measurement {key!r}"
    if m.name != key:
        errs.append(f"{tag}: name field {m.name!r} differs from its key")
    if m.kind not in _KINDS:
        return errs + [f"{tag}: unknown kind {m.kind!r}"]
    if m.mechanism != "gaussian":
        errs.append(f"{tag}: unsupported mechanism {m.mechanism!r}")
    for fld in ("sigma", "sensitivity_edge", "sensitivity_attr"):
        x = getattr(m, fld)
        if not (isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(x) and x >= 0):
            errs.append(f"{tag}: {fld} must be a finite non-negative number, got {x!r}")
    v = np.asarray(m.values)
    if v.dtype.kind not in "iuf":
        return errs + [f"{tag}: values dtype {v.dtype} is not numeric"]
    if not np.all(np.isfinite(v)):
        errs.append(f"{tag}: values contain non-finite entries")
    elif np.any(v < 0):
        errs.append(f"{tag}: values contain negative entries (min {v.min()})")

    dom = schema.domains
    p = m.params or {}

    def shape_is(expected: tuple) -> None:
        if v.shape != expected:
            errs.append(f"{tag}: shape {v.shape}, expected {expected}")

    def attrs_param(count: int | None) -> list[str] | None:
        a = p.get("attrs")
        if not isinstance(a, list) or not a or any(x not in dom for x in a):
            errs.append(f"{tag}: params['attrs'] must be a list of attributes in the schema domains, got {a!r}")
            return None
        if count is not None and len(a) != count:
            errs.append(f"{tag}: params['attrs'] must have {count} entries, got {len(a)}")
            return None
        return a

    def bins() -> int | None:
        be = p.get("bin_edges")
        if not isinstance(be, list) or len(be) < 2:
            errs.append(f"{tag}: params['bin_edges'] must be a list of at least 2 edges")
            return None
        if any(b <= a for a, b in zip(be, be[1:])):
            errs.append(f"{tag}: params['bin_edges'] must be strictly ascending")
        return len(be) - 1

    if m.kind == "degree_hist":
        # Dmax is the measurement's own public max_degree (uncapped histogram); fall back to the schema's D.
        md = p.get("max_degree", schema.degree_cap)
        if not (isinstance(md, (int, np.integer)) and md >= 1):
            errs.append(f"{tag}: params['max_degree'] must be a positive integer, got {md!r}")
        else:
            shape_is((int(md) + 1,))
    elif m.kind == "clustering_by_degree":
        b = bins()
        if b is not None:
            shape_is((b, 2))
    elif m.kind == "joint_degree":
        b = bins()
        if b is not None:
            shape_is((b, b))
    elif m.kind == "attr_1way":
        a = attrs_param(1)
        if a:
            shape_is((len(dom[a[0]]),))
    elif m.kind == "attr_2way":
        a = attrs_param(2)
        if a:
            shape_is((len(dom[a[0]]), len(dom[a[1]])))
    elif m.kind == "attr_degree":
        a = attrs_param(1)
        b = bins()
        if p.get("cap") is not None:
            errs.append(f"{tag}: attr_degree is measured on the uncapped graph (params['cap'] must be None)")
        if b is not None:
            try:
                check_bin_edges(p["bin_edges"])
            except (ValueError, TypeError) as exc:
                errs.append(f"{tag}: params['bin_edges'] invalid: {exc}")
        if a and b is not None:
            shape_is((len(dom[a[0]]), b))
    elif m.kind == "attr_edge_pair":
        # one instance per configured homophily group: the legacy name is exactly "attr_edge_pair"
        # (old artifacts), new instances are "attr_edge_pair:<a>" / "attr_edge_pair:<a>+<b>"
        a = attrs_param(None)
        if a:
            g = int(np.prod([len(dom[x]) for x in a]))
            if g > MAX_GROUPS:
                errs.append(f"{tag}: group of {g} cells exceeds the {MAX_GROUPS}-cell limit")
            else:
                shape_is((g, g))

    if m.kind in ("joint_degree", "attr_edge_pair") and v.ndim == 2 and v.shape[0] == v.shape[1]:
        if not np.allclose(v, v.T, rtol=0, atol=1e-9):
            errs.append(f"{tag}: matrix is not symmetric")
    return errs


def validate(artifact: Artifact) -> list[str]:
    """Return a list of human-readable problems; empty means the artifact is consistent."""
    errs: list[str] = []
    s = artifact.schema
    extra = sorted(set(artifact.provenance) - PROVENANCE_ALLOWED)
    if extra:
        errs.append(f"provenance: keys {extra} are not on the allowlist (nothing un-noised may leave the inside)")
    if not (isinstance(s.n, (int, np.integer)) and s.n > 0):
        errs.append(f"schema: n must be a positive integer, got {s.n!r}")
    if not (isinstance(s.degree_cap, (int, np.integer)) and s.degree_cap >= 1):
        errs.append(f"schema: degree_cap must be a positive integer, got {s.degree_cap!r}")
    for col, labels in s.domains.items():
        if len(labels) == 0 or len(set(labels)) != len(labels):
            errs.append(f"schema: domain {col!r} must be non-empty with unique labels")
        for req in ("other", "none"):
            if req not in labels:
                errs.append(f"schema: domain {col!r} lacks the explicit {req!r} label")
    for a in s.homophily_attrs:
        if a not in s.domains:
            errs.append(f"schema: homophily attribute {a!r} is not in domains")

    for key, m in artifact.measurements.items():
        errs.extend(_validate_measurement(key, m, s))

    led = artifact.ledger
    if not (0 < led.delta < 1):
        errs.append(f"ledger: delta must be in (0, 1), got {led.delta!r}")
    names = [e.get("name") for e in led.entries]
    if len(set(names)) != len(names):
        errs.append("ledger: duplicate entry names")
    if set(names) != set(artifact.measurements):
        errs.append(
            f"ledger: entries {sorted(map(str, names))} do not match measurements {sorted(artifact.measurements)}"
        )
    for side, total in (("edge", led.rho_edge), ("attr", led.rho_attr)):
        field_ = f"rho_{side}"
        if not (total >= 0):
            errs.append(f"ledger: {field_} must be non-negative, got {total!r}")
            continue
        try:
            summed = float(sum(float(e[field_]) for e in led.entries))
        except (KeyError, TypeError, ValueError):
            errs.append(f"ledger: an entry lacks a numeric {field_}")
            continue
        if not _close(float(total), summed):
            errs.append(f"ledger: {field_} {total} differs from sum of entries {summed}")
    for e in led.entries:
        m = artifact.measurements.get(e.get("name"))
        if m is None:
            continue
        for side, sens in (("edge", m.sensitivity_edge), ("attr", m.sensitivity_attr)):
            try:
                got = float(e[f"rho_{side}"])
            except (KeyError, TypeError, ValueError):
                continue
            want = entry_rho(m.sigma, sens)
            if not _close(got, want):
                errs.append(f"ledger: entry {e['name']!r} rho_{side} {got} != s^2/(2 sigma^2) = {want}")
    return errs

"""Artifact writer/reader/validator. JSON + Parquet (CSV fallback) + npz; never pickle."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from geolife.transfer.types import ArtifactManifest, Measurement, PrivacyLedger

try:  # pragma: no cover - environment dependent
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = pq = None

FORMAT_VERSION = "0.1"
NAME_RE = re.compile(r"^[a-z0-9_]+$")
JSON_FILES = ("schema", "preprocessing", "grounding", "diagnostics", "generation")

# The only keys the free-form json files of an artifact may carry. They hold PUBLIC settings (processing rules, caps,
# bins, public vocabulary, generator knobs) and diagnostics computed from the already-noisy tables, never a statistic
# of the real data: an un-noised count (e.g. a user total) cannot be released by accident through them.
# validate_no_raw_leakage() rejects any other top-level key; extend deliberately, with a reason, in review.
PROVENANCE_ALLOWED: dict[str, frozenset[str]] = {
    "preprocessing": frozenset({
        "h3_resolution", "stay_dist_m", "stay_min_s", "thin_s", "snap_radius_m", "merge_gap_s", "tz_offset_s",
        "trip_min_len_m", "dwell_edges_s", "dist_edges_m", "map_matching", "detour_definition", "region",
        "outing_gap_s", "real_outing", "stop_rule", "tz_offset_h", "tod_bins", "tod_coarse",
        "tod_coarse_local_hours", "tod3_bins", "tod3_edges_h", "daytypes", "caps", "hierarchy"}),
    # cat_profile_global: public category names whose released time profile fell back to the
    # global one (post-processing of noisy masses, same provenance as "fallbacks")
    "diagnostics": frozenset({"fallbacks", "signal_ratio", "zipf_tier", "tiers", "snr_min",
                              "cat_profile_global"}),
    "grounding": frozenset({"extent_m"}),
    "generation": frozenset({
        "calibrate_iters", "calibrate_pool", "carpool_frac", "category_map", "cell_res",
        "compute_cells", "continue_prob", "core_bias", "cotravel_outing_frac", "cotravel_user_frac",
        "d0_m", "density_radius_frac", "detour_candidates", "detour_factor", "detour_min_leg_m",
        "detour_nominal", "distinct_activity_corr", "distinct_cap_mult", "dwell_jitter_m",
        "dwell_max_s", "dwell_min_s", "evening_outing_prob", "extent_m", "gap_max_s", "gap_mean_s", "gap_median_s",
        "gap_rate_per_hr", "gap_sigma", "group_fav_size", "group_size_decay", "horizon_days",
        "interval_jitter", "learner", "length_bump_sigma", "length_scale", "length_targeting",
        "max_outings", "max_stops", "max_visits_per_day", "min_copresence_s", "min_gap_s",
        "min_outings", "mode_length_factor", "n_days", "note", "outing_gap_s", "outings_per_active_day",
        "outlier_prob", "outlier_scale", "pool_cap_frac", "pool_from_hist", "pool_size_override",
        "post_sit_s", "pre_sit_s", "region_radius_frac", "revisit_pref", "roundtrip_prob",
        "sample_interval_s", "sample_period_w", "sample_periods_s", "sensor_noise_m", "sensor_source",
        "speed_mps", "speed_ms_by_mode", "speed_sd", "speed_sd_leg", "stay_interval_s",
        "stay_logged_prob", "stop_signal_loss_prob", "stopgo_block_s", "stopgo_sd", "stops_tail_p",
        "taste_alpha", "trav_res", "trip_interval_s", "tz_offset_h", "ungrouped_frac",
        "use_sensor_hists", "w_dens",
        "w_pop", "w_pos", "walk_max_m", "weekday_work_prob", "weekend_share_default"
    }),
}


class ArtifactError(Exception):
    """Tampered, incomplete, or incompatible artifact."""


class LeakageError(ArtifactError):
    """Artifact content looks like raw data or inconsistent privacy accounting."""


@dataclass
class Artifact:
    manifest: ArtifactManifest
    schema: dict
    preprocessing: dict
    measurements: list[Measurement]
    ledger: PrivacyLedger
    behaviour: dict[str, dict] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jbytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _check_name(name: str, what: str = "measurement") -> None:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid {what} name {name!r}: must match [a-z0-9_]+")


# --- table (de)serialisation -------------------------------------------------------------


def _table_bytes(m: Measurement) -> tuple[bytes, str]:
    keys = [json.dumps(list(k)) for k in m.keys]
    vals = [float(v) for v in m.values]
    if len(keys) != len(vals):
        raise ValueError(f"measurement {m.name}: keys/values length mismatch")
    if pq is not None:
        buf = io.BytesIO()
        pq.write_table(pa.table({"key": pa.array(keys, pa.string()),
                                 "value": pa.array(vals, pa.float64())}), buf)
        return buf.getvalue(), "parquet"
    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")
    w.writerow(["key", "value"])
    for k, v in zip(keys, vals):
        w.writerow([k, repr(v)])
    return sio.getvalue().encode("utf-8"), "csv"


def _read_table(data: bytes, ext: str) -> tuple[list[tuple[str, ...]], list[float]]:
    if ext == "parquet":
        if pq is None:
            raise ArtifactError("pyarrow required to read parquet artifact")
        t = pq.read_table(io.BytesIO(data)).to_pydict()
        keys, vals = t["key"], t["value"]
    elif ext == "csv":
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))[1:]
        keys, vals = [r[0] for r in rows], [float(r[1]) for r in rows]
    else:
        raise ArtifactError(f"unknown table format {ext!r}")
    return [tuple(json.loads(k)) for k in keys], [float(v) for v in vals]


def _npz_bytes(arrs: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrs.items()})
    return buf.getvalue()


# --- save --------------------------------------------------------------------------------


def save_artifact(dir, manifest: ArtifactManifest, schema: dict, preprocessing: dict,
                  measurements: list[Measurement], ledger: PrivacyLedger,
                  behaviour: dict[str, dict], embeddings: dict[str, np.ndarray] | None,
                  grounding: dict, diagnostics: dict, generation: dict) -> ArtifactManifest:
    """Write the artifact; fills and returns manifest.files (sha256 per file)."""
    root = Path(dir)
    names = [m.name for m in measurements]
    for n in names:
        _check_name(n)
    if len(set(names)) != len(names):
        raise ValueError("duplicate measurement names")
    for n in (behaviour or {}):
        _check_name(n, "behaviour")
    for n in (embeddings or {}):
        _check_name(n, "embedding")
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {manifest.format_version}")

    blobs: dict[str, bytes] = {}
    for key, obj in zip(JSON_FILES, (schema, preprocessing, grounding, diagnostics, generation)):
        blobs[f"{key}.json"] = _jbytes(obj)
    blobs["privacy.json"] = _jbytes(asdict(ledger))
    for m in measurements:
        data, ext = _table_bytes(m)
        blobs[f"model/behaviour/{m.name}.{ext}"] = data
        meta = {"name": m.name, "sigma": m.sigma, "l2_sensitivity": m.l2_sensitivity,
                "rho": m.rho, "notes": m.notes, "format": ext}
        blobs[f"model/behaviour/{m.name}.meta.json"] = _jbytes(meta)
    blobs["model/behaviour_summaries.json"] = _jbytes(behaviour or {})
    emb = embeddings or {}
    if emb:
        blobs["model/embeddings/embeddings.npz"] = _npz_bytes(emb)
    blobs["model/embeddings/metadata.json"] = _jbytes(
        {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
         for k, v in emb.items()})

    root.mkdir(parents=True, exist_ok=True)
    for rel, data in blobs.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    manifest.files = {rel: _sha(d) for rel, d in sorted(blobs.items())}
    (root / "manifest.json").write_bytes(_jbytes(asdict(manifest)))
    return manifest


# --- load --------------------------------------------------------------------------------


def load_artifact(dir) -> Artifact:
    """Load and verify; raises ArtifactError on tamper, missing/unlisted files, version mismatch."""
    root = Path(dir)
    mp = root / "manifest.json"
    if not mp.is_file():
        raise ArtifactError("missing manifest.json")
    md = json.loads(mp.read_text("utf-8"))
    if md.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"format_version mismatch: {md.get('format_version')!r}")
    manifest = ArtifactManifest(**md)

    data: dict[str, bytes] = {}
    for rel, digest in manifest.files.items():
        if ".." in Path(rel).parts or Path(rel).is_absolute():
            raise ArtifactError(f"unsafe path in manifest: {rel}")
        p = root / rel
        if not p.is_file():
            raise ArtifactError(f"missing file: {rel}")
        b = p.read_bytes()
        if _sha(b) != digest:
            raise ArtifactError(f"hash mismatch (tampered?): {rel}")
        data[rel] = b
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel != "manifest.json" and rel not in data:
                raise ArtifactError(f"unlisted file in artifact: {rel}")

    def J(rel):
        if rel not in data:
            raise ArtifactError(f"missing file: {rel}")
        return json.loads(data[rel].decode("utf-8"))

    ledger = PrivacyLedger(**J("privacy.json"))
    measurements = []
    for rel in sorted(data):
        if rel.endswith(".meta.json") and rel.startswith("model/behaviour/"):
            meta = J(rel)
            _check_name(meta["name"])
            trel = f"model/behaviour/{meta['name']}.{meta['format']}"
            if trel not in data:
                raise ArtifactError(f"missing file: {trel}")
            keys, vals = _read_table(data[trel], meta["format"])
            measurements.append(Measurement(
                name=meta["name"], keys=keys, values=vals, sigma=meta["sigma"],
                l2_sensitivity=meta["l2_sensitivity"], rho=meta["rho"], notes=meta["notes"]))
    emb: dict[str, np.ndarray] = {}
    ek = "model/embeddings/embeddings.npz"
    if ek in data:
        with np.load(io.BytesIO(data[ek]), allow_pickle=False) as z:
            emb = {k: z[k] for k in z.files}
    return Artifact(
        manifest=manifest, schema=J("schema.json"), preprocessing=J("preprocessing.json"),
        measurements=measurements, ledger=ledger,
        behaviour=J("model/behaviour_summaries.json"), embeddings=emb,
        grounding=J("grounding.json"), diagnostics=J("diagnostics.json"),
        generation=J("generation.json"))


# --- leak guard --------------------------------------------------------------------------

_COORD_KEYS = re.compile(r"(^|_)(lat|lon|lng|latitude|longitude)(_|$)", re.I)
_COORD_PAIR = re.compile(r"-?\d{1,3}\.\d{4,}\s*[,;]\s*-?\d{1,3}\.\d{4,}")
DEFAULT_USER_ID_RE = r"^\d{3}$"  # GeoLife user dirs are 000..181


def _coordlike(x) -> bool:
    """float with >=4 decimals in coordinate range."""
    if not isinstance(x, float) or abs(x) > 180:
        return False
    s = repr(x)
    return "e" not in s and "." in s and len(s.split(".")[1]) >= 4


def _walk(obj, path, issues, uid_re):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _COORD_KEYS.search(k) and isinstance(v, (int, float, list)):
                issues.append(f"{path}/{k}: coordinate-like key")
            _walk(v, f"{path}/{k}", issues, uid_re)
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and _coordlike(obj[0]) and _coordlike(obj[1]):
            issues.append(f"{path}: looks like a raw (lat, lon) point")
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", issues, uid_re)
    elif isinstance(obj, str):
        if _COORD_PAIR.search(obj):
            issues.append(f"{path}: string contains lat,lon pair")
        if uid_re.search(obj):
            issues.append(f"{path}: matches real user id pattern")


def validate_no_raw_leakage(artifact: Artifact, user_id_regex: str = DEFAULT_USER_ID_RE,
                            rho_tol: float = 1e-9, eps_tol: float = 1e-6) -> None:
    """Raise LeakageError listing every problem found."""
    issues: list[str] = []
    uid_re = re.compile(user_id_regex)
    _walk(artifact.schema, "schema", issues, uid_re)
    for fname, allowed in PROVENANCE_ALLOWED.items():
        extra = sorted(set(getattr(artifact, fname) or {}) - allowed)
        if extra:
            issues.append(f"{fname}.json: keys not on the provenance allowlist: {extra}")
        _walk(getattr(artifact, fname) or {}, fname, issues, uid_re)
    for m in artifact.measurements:
        for i, k in enumerate(m.keys):
            _walk(list(k), f"measurement:{m.name}:key{i}", issues, uid_re)
    led = artifact.ledger
    if led.adjacency != "user":
        issues.append(f"privacy: adjacency is {led.adjacency!r}, expected 'user'")
    s = sum(float(e.get("rho", 0.0)) for e in led.entries)
    if abs(s - led.rho_total) > max(rho_tol, 1e-9 * abs(led.rho_total)):
        issues.append(f"privacy: sum of entry rho {s} != rho_total {led.rho_total}")
    if led.rho_total > 0 and 0 < led.delta < 1:
        eps = led.rho_total + 2 * math.sqrt(led.rho_total * math.log(1 / led.delta))
        if abs(eps - led.epsilon) > eps_tol * max(1.0, eps):
            issues.append(f"privacy: epsilon {led.epsilon} != zCDP conversion {eps}")
    names = {m.name for m in artifact.measurements}
    en = {e.get("name") for e in led.entries}
    if names != en:
        issues.append(f"privacy: ledger entries {sorted(map(str, en))} != measurements {sorted(names)}")
    for m in artifact.measurements:
        if m.sigma > 0:
            r = m.l2_sensitivity ** 2 / (2 * m.sigma ** 2)
            if abs(r - m.rho) > 1e-9 * max(1.0, r):
                issues.append(f"measurement {m.name}: rho {m.rho} != Delta^2/(2 sigma^2) {r}")
    if issues:
        raise LeakageError("; ".join(issues))

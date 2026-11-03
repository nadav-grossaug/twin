"""Stage-2 output format: synthetic GPS traces + the planted truth, as parquet / json files.

    <variant>_seed<k>/
      traces.parquet   uid, t (UTC epoch s), lat, lon      the synthetic data
      outings.json     {uid: [outing]}                      planted outings (what a tokenizer should recover)
      truth.json       planted structure: groups, favourite/shared places, co-travel, rendered visits
      meta.json        variant, seed, n users, map spec, generation knobs, artifact epsilon

Nothing here is derived from real data: it is the generator's own ground truth, which is what lets the eval zone
score method M against it. `read_synthetic` returns the same attribute layout as `GeneratedV2` (`.result.traces`,
`.result.outings`, `.result.truth`) so the eval harness works on files and on in-memory runs alike.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from geolife.outside.population_v2 import GroundTruthV2
from geolife.transfer.types import GpsPoint, Outing, Visit

FILES = ("traces.parquet", "outings.json", "truth.json", "meta.json")


def _jsonable(o):
    if isinstance(o, (Visit, Outing)):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_synthetic(out_dir, result, meta: dict) -> Path:
    """`result` is a RenderResultV2 (traces, outings, truth). Returns the directory."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    uids, ts, la, lo = [], [], [], []
    for uid in sorted(result.traces):
        pts = result.traces[uid]
        uids += [uid] * len(pts)
        ts += [p.t for p in pts]
        la += [p.lat for p in pts]
        lo += [p.lon for p in pts]
    pd.DataFrame({"uid": uids, "t": np.asarray(ts, np.int64), "lat": np.asarray(la, float),
                  "lon": np.asarray(lo, float)}).to_parquet(d / "traces.parquet", index=False)
    _dump(d / "outings.json", result.outings)
    _dump(d / "truth.json", {f.name: getattr(result.truth, f.name) for f in fields(result.truth)})
    _dump(d / "meta.json", meta)
    return d


def _visit(v) -> Visit:
    return Visit(place_id=v["place_id"], category=v["category"], cell=v["cell"], t_arrive=int(v["t_arrive"]),
                 dwell_s=int(v["dwell_s"]), kind=v["kind"])


def _outing(o) -> Outing:
    return Outing(int(o["t_start"]), int(o["t_end"]), o["mode"], float(o["length_m"]),
                  tuple(_visit(v) for v in o["visits"]), tuple(o["cells"]), tuple(o["segments"]))


def read_synthetic(path) -> SimpleNamespace:
    """Directory written by `write_synthetic` -> object with .result.{traces,outings,truth,meta} and .meta."""
    d = Path(path)
    df = pd.read_parquet(d / "traces.parquet")
    traces: dict[str, list[GpsPoint]] = {}
    for uid, g in df.groupby("uid", sort=True):
        traces[str(uid)] = [GpsPoint(int(t), float(a), float(b)) for t, a, b in zip(g.t, g.lat, g.lon)]
    raw = json.loads((d / "outings.json").read_text(encoding="utf-8"))
    outings = {u: [_outing(o) for o in os_] for u, os_ in raw.items()}
    tj = json.loads((d / "truth.json").read_text(encoding="utf-8"))
    tj["visits"] = {u: [_visit(v) for v in vs] for u, vs in tj.get("visits", {}).items()}
    tj["co_traveled"] = [tuple(x) for x in tj.get("co_traveled", [])]
    tj["co_traveled_detail"] = [tuple(x) for x in tj.get("co_traveled_detail", [])]
    for k in ("groups", "group_places"):          # group ids are ints in memory, strings in json
        tj[k] = {(int(g) if g.lstrip("-").isdigit() else g): v for g, v in tj.get(k, {}).items()}
    truth = GroundTruthV2(**tj)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    result = SimpleNamespace(traces=traces, outings=outings, truth=truth, meta=meta)
    return SimpleNamespace(result=result, meta=meta)

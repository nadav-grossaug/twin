"""Air-gap side learner v2 (outings): list[UserOutings] -> capped DP measurements -> artifact.

CLI:
  python -m geolife.inside.learn_v2 --outings outings.parquet --out geolife/transfer/artifacts/outings_v2_eps6_seed0 \
        [--epsilon 6 --delta 1e-6 | --rho 0.5] [--places places.parquet | --categories a,b,c]
  python -m geolife.inside.learn_v2 --points points.parquet --places places.parquet --out ...   (builds
        outings with geolife.transfer.tokenize.outings first; labels read from --data)

Only the artifact (noisy tables, ledger, fitted params) is written to --out. `--report` writes a LOCAL
SNR file comparing noisy to noiseless tables; it must never leave the air-gap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from geolife.inside import dp
from geolife.transfer.artifact import load_artifact, save_artifact, validate_no_raw_leakage
from geolife.transfer import schema_v2 as S2
from geolife.outside.fit_v2 import fit_behaviour_v2
from geolife.inside.measure_v2 import VocabV2, build_measurements_v2
from geolife.transfer.geo import pseudonyms
from geolife.transfer.types import ArtifactManifest, Outing, UserOutings, Visit

BEIJING_CORE = (39.75, 40.10, 116.15, 116.60)

# Release policy (PLAN step 7): the k seeds of a dev run are NOISE REPLICATES of the same
# measurements; averaging k released seeds divides the effective noise std by sqrt(k) and the
# honest accounting of releasing them all is k * rho (eps grows accordingly). A real release
# therefore publishes EXACTLY ONE seed; this caveat is stamped into privacy.json by
# fit_artifacts.py --release.
ONE_SEED_CAVEAT = (
    "Release policy: exactly one noise seed is released. Multiple seeds of the same fit are noise "
    "replicates; averaging k of them reduces the effective noise std by sqrt(k), and releasing k "
    "seeds costs k * rho_total under zCDP composition. This artifact is the single released seed.")


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Largest rho with zcdp_to_epsilon(rho, delta) <= epsilon (bisection).
    (Moved here from the retired v1 inside/learn.py.)"""
    lo, hi = 0.0, epsilon
    for _ in range(100):
        mid = (lo + hi) / 2
        if dp.zcdp_to_epsilon(mid, delta) > epsilon:
            hi = mid
        else:
            lo = mid
    return lo


def snr_report(true_ms, noisy_ms, k_sigma: float = 3.0) -> list[dict]:
    """Per-table signal-to-noise of the noisy release vs the noiseless value (local aid; must never
    leave the air gap). (Moved here from the retired v1 inside/learn.py.)"""
    rows = []
    for t, n in zip(true_ms, noisy_ms):
        tv, nv = np.asarray(t.values), np.asarray(n.values)
        sig = n.sigma
        if len(tv) == 0:   # tier skipped (no parent passed the threshold); rho was reserved
            rows.append({"name": n.name, "keys": 0, "keys_nonzero": 0, "sigma": float(sig),
                         "true_total": 0.0, "max_true": 0.0, "keys_above_3sigma": 0,
                         "mass_above_3sigma": 0.0, "corr_top10": float("nan"),
                         "l1_tv_distance": float("nan"), "rmse_over_mean_nz": float("nan")})
            continue
        nz = tv > 0
        above = tv > k_sigma * sig
        mass = tv.sum()
        # normalised distribution fidelity after non-negative projection
        pn = np.clip(nv, 0, None)
        l1 = float(np.abs(tv / mass - pn / pn.sum()).sum() / 2) if mass > 0 and pn.sum() > 0 else float("nan")
        top = np.argsort(-tv)[:10]
        rows.append({
            "name": n.name, "keys": len(tv), "keys_nonzero": int(nz.sum()), "sigma": float(sig),
            "true_total": float(mass), "max_true": float(tv.max()),
            "keys_above_3sigma": int(above.sum()),
            "mass_above_3sigma": float(tv[above].sum() / mass) if mass > 0 else 0.0,
            "corr_top10": float(np.corrcoef(tv[top], nv[top])[0, 1]) if len(top) > 2 and tv[top].std() > 0 else float("nan"),
            "l1_tv_distance": l1,
            "rmse_over_mean_nz": float(np.sqrt(np.mean((nv - tv)[nz] ** 2)) / tv[nz].mean()) if nz.any() else float("nan"),
        })
    return rows


def load_outings_parquet(path: str | Path) -> list[UserOutings]:
    """Rebuild UserOutings from the T1 export (`visits` JSON column; optional `cells`/`segments`
    JSON columns). The T1 frame currently has no traversed cells, so traversal tables are empty
    unless the column exists."""
    import pandas as pd
    df = pd.read_parquet(path).sort_values(["user", "t_start"])
    users: dict[str, list[Outing]] = {}
    for r in df.itertuples(index=False):
        vis = tuple(Visit(place_id=v[1], category=v[2], cell=v[3], t_arrive=int(v[4]),
                          dwell_s=int(v[5]), kind=v[0]) for v in json.loads(r.visits))
        cells = tuple(json.loads(r.cells)) if hasattr(r, "cells") and r.cells else ()
        segs = tuple(json.loads(r.segments)) if hasattr(r, "segments") and r.segments else ()
        users.setdefault(r.user, []).append(Outing(int(r.t_start), int(r.t_end), r.mode,
                                                   float(r.length_m), vis, cells, segs))
    return [UserOutings(u, o) for u, o in sorted(users.items())]


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def learn_from_outings(users: Sequence[UserOutings], vocab: VocabV2, rho: float, seed: int = 0,
                       delta: float = 1e-6, sensor_views=None):
    rng = np.random.default_rng(seed)
    ms, led = build_measurements_v2(users, vocab, rho, rng, delta=delta, seed=seed,
                                    sensor_views=sensor_views)
    return ms, led


def run(users: Sequence[UserOutings], out: str | Path, *, rho: float | None = None,
        epsilon: float = 6.0, delta: float = 1e-6, bbox=BEIJING_CORE,
        categories: Sequence[str] = ("unknown",), seed: int = 0, verbose: bool = True,
        with_snr: bool = True, sensor_views=None, extra_caveats: Sequence[str] = ()):
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    rho = rho if rho is not None else rho_for_epsilon(epsilon, delta)
    vocab = VocabV2.from_bbox(bbox, categories)
    log(f"{len(users)} users, {len(vocab.r6_cells)} r6 cells, rho={rho:.4f}"
        + (f", sensor views={len(sensor_views)}" if sensor_views is not None else ""))
    ms, led = learn_from_outings(users, vocab, rho, seed, delta, sensor_views)
    params = fit_behaviour_v2(ms, led)
    if sensor_views is not None:
        sfb = (params.get("sensor") or {}).get("fallbacks") or {}
        if sfb:
            led.caveats.append(
                "Sensor fallbacks in THIS release (the generator uses the hand-set defaults of "
                "outside/render_v2.SENSOR_DEFAULTS for these draws): "
                + "; ".join(f"{k}: {v}" for k, v in sorted(sfb.items())))
    led.caveats.extend(extra_caveats)
    pre = {"outing_gap_s": 7200, "real_outing": {"min_s": 300, "min_m": 500},
           "stop_rule": {"radius_m": 200, "dwell_s": 600}, "tz_offset_h": 8,
           "tod_bins": list(S2.TOD_BINS), "tod_coarse": list(S2.TOD_COARSE),
           "tod_coarse_local_hours": {"am": [5, 14], "pm": [14, 5]},
           "tod3_bins": list(S2.TOD_3), "tod3_edges_h": list(S2.TOD3_EDGES_H),
           "daytypes": list(S2.DAYTYPES),
           "caps": {"outings": S2.MAX_OUTINGS, "stop_cells_r8": S2.MAX_STOP_CELLS,
                    "trav_cells_r8": S2.MAX_TRAV_CELLS,
                    **({"sensor_steps": S2.MAX_SENSOR_STEPS} if sensor_views is not None else {})},
           "hierarchy": {"place": list(S2.PLACE_RES), "trav": list(S2.TRAV_RES)},
           "region": list(bbox)}
    schema = {"version": 2, "measurements": [m.name for m in ms], "kinds": S2.KIND,
              "categories": vocab.categories, "key_levels": ["h3_cell", "category", "bin"]}
    man = ArtifactManifest(learner="geolife-outings-v2", source_map_snapshot="h3+places",
                           h3_resolution=8, created=_now())
    gen = {"tz_offset_h": 8, "use_sensor_hists": True,
           "sensor_source": ("dp sensor histograms (sensor_* tables, decision D20)"
                             if sensor_views is not None else
                             "generator defaults (not measured by learner v2)")}
    save_artifact(out, man, schema, pre, ms, led, {"params_v2": _clean(params)}, None, {},
                  _clean(params["diagnostics"]), gen)
    art = load_artifact(out)
    validate_no_raw_leakage(art)
    log(f"artifact written to {out} (eps={led.epsilon:.3f}, rho={led.rho_total:.4f})")
    res = {"ledger": led, "measurements": ms, "params": params, "rho": rho, "vocab": vocab}
    if with_snr:
        from geolife.inside.measure_v2 import build_measurements_v2 as b
        true_ms, _ = b(users, vocab, 1e12, np.random.default_rng(seed + 1), delta=delta, seed=seed,
                       vocab_override={m.name: m.keys for m in ms}, sensor_views=sensor_views)
        res["snr"] = snr_report(true_ms, ms)
        res["true"] = true_ms
    return res


def users_from_points(points: str | Path, places: str | Path, data_root: str | Path | None = None,
                      seed: int = 0) -> list[UserOutings]:
    """points.parquet (uid, t, lat, lon) + public places parquet + GeoLife labels -> pseudonymised UserOutings.
    Inside only: this is where the real traces are read."""
    import pandas as pd
    from geolife.inside.data import default_data_dir, read_labels
    from geolife.inside.loaders import load_points_arrays
    from geolife.transfer.tokenize import outings as T
    root = Path(data_root) if data_root else default_data_dir()
    pts = load_points_arrays(Path(points))
    pidx = T.PlaceIndex(pd.read_parquet(places))
    names = pseudonyms(list(pts), seed)
    users = []
    for uid in sorted(pts):
        lp = root / "Data" / uid / "labels.txt"
        uo = T.user_outings_from_arrays(names[uid], *pts[uid], pidx, T.OutingsConfig(),
                                        read_labels(lp) if lp.exists() else [], {})
        if uo is not None:
            users.append(uo)
    return users


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outings", default=None, help="outings parquet (T1 export)")
    ap.add_argument("--points", default=None, help="points.parquet cache; builds outings first")
    ap.add_argument("--data", default=None, help="GeoLife root (labels), with --points")
    ap.add_argument("--places", default=None, help="public places parquet (id, category, lat, lon, popularity)")
    ap.add_argument("--categories", default=None, help="comma-separated public categories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=6.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--bbox", type=float, nargs=4, default=list(BEIJING_CORE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="local SNR json (never leaves the air-gap)")
    ap.add_argument("--no-sensors", action="store_true", help="skip the sensor tables (with --points)")
    a = ap.parse_args(argv)
    cats: list[str] = ["unknown"]
    if a.categories:
        cats = a.categories.split(",")
    elif a.places:
        import pandas as pd
        cats = sorted(set(pd.read_parquet(a.places, columns=["category"]).category.astype(str)))
    sensor_views = None
    if a.outings:
        users = load_outings_parquet(a.outings)
    elif a.points and a.places:
        users = users_from_points(a.points, a.places, a.data, a.seed)
        if not a.no_sensors:
            from geolife.inside.measure_sensors import sensor_views_from_points
            sensor_views = sensor_views_from_points(a.points, a.data, seed=a.seed, bbox=a.bbox)
    else:
        ap.error("give --outings, or --points and --places")
    res = run(users, a.out, rho=a.rho, epsilon=a.epsilon, delta=a.delta, bbox=a.bbox,
              categories=cats, seed=a.seed, with_snr=bool(a.report), sensor_views=sensor_views)
    if a.report:
        Path(a.report).write_text(json.dumps(_clean({"snr": res["snr"], "epsilon": res["ledger"].epsilon,
                                                     "rho": res["rho"], "users": len(users)}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

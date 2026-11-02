"""Air-gap side learner: GPS trajectories -> tokens -> capped records -> DP measurements -> artifact.

CLI:
  python -m geolife.inside.learn --data <dir with points.parquet | GeoLife root> --out <artifact dir>
         [--epsilon 6 --delta 1e-6 | --rho 0.5] [--places places.parquet] [--report report.md]

Only the artifact (noisy, clipped, user-level tables over a public vocabulary) is written to --out.
The optional --report is a local evaluation aid: it compares noisy tables to their noiseless
values, which must never leave the air-gap.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from geolife.inside import dp
from geolife.transfer.artifact import load_artifact, save_artifact, validate_no_raw_leakage
from geolife.inside.data import default_data_dir
from geolife.transfer import schema as S
from geolife.inside.measure import HierarchySpec, build_behaviour_measurements
from geolife.transfer.tokenize.stays import PlaceTable
from geolife.inside.loaders import load_labels_for, load_points_parquet
from geolife.transfer.tokenize.tokens import (BEIJING_CORE, TokenizeConfig,
                              tokenize_all)
from geolife.transfer.types import ArtifactManifest, UserRecord

TZ_OFFSET_S = 8 * 3600  # Beijing local time (GeoLife timestamps are GMT)


def rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Largest rho with zcdp_to_epsilon(rho, delta) <= epsilon (bisection)."""
    lo, hi = 0.0, epsilon
    for _ in range(100):
        mid = (lo + hi) / 2
        if dp.zcdp_to_epsilon(mid, delta) > epsilon:
            hi = mid
        else:
            lo = mid
    return lo


def h3_vocab(bbox: Sequence[float], res: int) -> list[tuple[str, str]]:
    """Public vocabulary of H3 cells covering the (public) analysis region."""
    import h3
    lat0, lat1, lon0, lon1 = bbox
    poly = h3.LatLngPoly([(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)])
    return [(f"h3:{c}", "unknown") for c in sorted(h3.polygon_to_cells(poly, res))]


def build_places(path: str | None, bbox, max_places: int) -> PlaceTable:
    """Public places inside the region (top `max_places` by popularity; a public-data rule)."""
    if not path:
        return PlaceTable.empty()
    import pandas as pd
    df = pd.read_parquet(path)
    lat0, lat1, lon0, lon1 = bbox
    df = df[(df.lat >= lat0) & (df.lat <= lat1) & (df.lon >= lon0) & (df.lon <= lon1)]
    if "popularity" in df:
        df = df.sort_values(["popularity", "id"], ascending=[False, True])
    df = df.head(max_places)
    return PlaceTable(df["id"], df["category"], df["lat"], df["lon"],
                      df["popularity"] if "popularity" in df else None)


def learn_from_records(records: list[UserRecord], place_vocab, rho: float, seed: int, circ: dict,
                       caps: tuple[int, int, int] = (100, 100, 20), delta: float = 1e-6,
                       hierarchy: HierarchySpec | None = None, split: str = "weighted"):
    """cap -> measure. Returns (measurements, ledger, cap_report, capped)."""
    rng = np.random.default_rng(seed)
    capped, rep = dp.cap_user_records(records, caps[0], caps[1], caps[2], rng)

    def detour(t):
        return circ.get((t.origin, t.dest, t.t_start))

    ms, led = build_behaviour_measurements(
        capped, place_vocab, rho, rng, tz_offset_s=TZ_OFFSET_S, detour_fn=detour, delta=delta,
        hierarchy=hierarchy, split=split,
        caps={"max_stays_per_user": caps[0], "max_trips_per_user": caps[1],
              "max_distinct_places_per_user": caps[2]})
    led.caveats += [
        "Stay detection, snapping and trip segmentation are per-user deterministic functions "
        "(user-level sensitivity is bounded by the caps and per-table clipping).",
        "Place vocabulary (H3 cells / public places) is chosen from the region box and a public "
        "map only, not from the data.",
        "Role assignment (home/work) is computed per user from that user's own data only.",
        "Contribution caps are applied before measurement; dropped data is not accounted.",
    ]
    if hierarchy is not None:
        led.caveats += [
            f"Place tier is hierarchical, fixed public resolutions H3 {list(hierarchy.res)}, refine "
            f"threshold tau={hierarchy.tau}, max keys per user {list(hierarchy.max_keys)}. The depth and "
            "resolutions were fixed from the brief BEFORE the v1 SNR was looked at, but an earlier "
            "equal-split v0 run on the same real data (place-level overlap pure noise at res 8, res 6 "
            "usable) informed the choice of res 6 as the coarse tier and of the weighted split; that "
            "data-dependent design choice is NOT accounted in epsilon.",
            f"rho split is weighted by public quantities only (l2 sensitivity/clip, vocabulary size) "
            f""
            f"with place-tier priorities {list(hierarchy.priorities)} (coarse to fine), histogram "
            f"priority {hierarchy.hist_priority}, and 40% of rho split equally as a floor. Priorities and the "
            "floor are declared design choices, set after seeing one weighted run on the real data "
            "(first attempt without the floor starved n_users/return_rate); also NOT accounted.",
            "Child tiers are released only under parents whose NOISY count passed tau*sigma "
            "(post-processing of the parent release); unspent budget if a tier is empty.",
            "work_users_hist: histogram over coarse cells of #users with a work anchor there; "
            "L2 sensitivity sqrt(2) (one user changes one cell count by 1, i.e. two bins).",
        ]
    return ms, led, rep, capped


def snr_report(true_ms, noisy_ms, k_sigma: float = 3.0) -> list[dict]:
    """Per-table signal-to-noise of the noisy release vs the noiseless value (local aid)."""
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


def run(data: str | Path, out: str | Path, *, rho: float | None = None, epsilon: float = 6.0,
        delta: float = 1e-6, bbox=BEIJING_CORE, h3_res: int = 8, places: str | None = None,
        max_places: int = 20000, seed: int = 0, caps=(100, 100, 20), max_users: int | None = None,
        report: str | None = None, cfg: TokenizeConfig | None = None, verbose: bool = True,
        hier: bool = True, split: str = "weighted"):
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    data = Path(data)
    if data.is_dir():
        root = data
        pq = next((c / "points.parquet" for c in (data, data.parent)
                   if (c / "points.parquet").exists()), data / "points.parquet")
    else:
        pq, root = data, data.parent
    rho = rho if rho is not None else rho_for_epsilon(epsilon, delta)
    cfg = cfg or TokenizeConfig(bbox=tuple(bbox), h3_res=h3_res)
    log(f"loading {pq}")
    trajs = load_points_parquet(pq, cfg.bbox, cfg.thin_s)
    if max_users:
        trajs = {u: trajs[u] for u in sorted(trajs)[:max_users]}
    raw_root = next((p for p in (root / "Geolife Trajectories 1.3", root, root.parent) if (p / "Data").is_dir()), None)
    labels = load_labels_for(raw_root, list(trajs)) if raw_root else {}
    table = build_places(places, cfg.bbox, max_places)
    log(f"{len(trajs)} users, {len(table)} public places, labels for {len(labels)} users")
    records, circ = tokenize_all(trajs, table, cfg, labels, seed)
    del trajs
    vocab = h3_vocab(cfg.bbox, cfg.h3_res) + [(i, c) for i, c in zip(table.ids, table.categories)]
    log(f"{len(records)} users with stays; vocab {len(vocab)} places; rho={rho:.4f}")
    hspec = None
    if hier:
        import h3
        pcell = {i: h3.latlng_to_cell(float(la), float(lo), cfg.h3_res)
                 for i, la, lo in zip(table.ids, table.lat, table.lon)}
        hspec = HierarchySpec(place_cell=pcell, res=(6, 7, cfg.h3_res))
    ms, led, rep, capped = learn_from_records(records, vocab, rho, seed, circ, caps, delta,
                                              hierarchy=hspec, split=split)
    lat0, lat1, lon0, lon1 = cfg.bbox
    extent = max((lat1 - lat0) * 111195.0, (lon1 - lon0) * 111195.0 * math.cos(math.radians((lat0 + lat1) / 2)))
    pre = {"h3_resolution": cfg.h3_res, "stay_dist_m": cfg.stay_dist_m, "stay_min_s": cfg.stay_min_s,
           "thin_s": cfg.thin_s, "snap_radius_m": cfg.snap_radius_m, "merge_gap_s": cfg.merge_gap_s,
           "tz_offset_s": TZ_OFFSET_S, "trip_min_len_m": cfg.trip_min_len_m,
           "dwell_edges_s": [e if e != S.INF else "inf" for e in S.DWELL_EDGES],
           "dist_edges_m": [e if e != S.INF else "inf" for e in S.DIST_EDGES],
           "map_matching": "off (trip length = GPS polyline length)",
           "detour_definition": "path length / straight-line distance (circuity)",
           "region": "beijing-core"}
    schema = {"roles": list(S.ROLES), "kinds": list(S.KINDS), "measurements": [m.name for m in ms],
              "categories": sorted({c for _, c in vocab}), "key_levels": ["place_id", "category", "role"]}
    man = ArtifactManifest(source_map_snapshot="h3-only" if not places else "places-parquet",
                           h3_resolution=cfg.h3_res, created=_now())
    save_artifact(out, man, schema, pre, ms, led, {}, None,
                  {"extent_m": float(extent)}, {},
                  {"learner": "geolife.inside.learn", "note": "values are noisy; true statistics are not stored"})
    art = load_artifact(out)
    validate_no_raw_leakage(art)
    log(f"artifact written to {out} (eps={led.epsilon:.3f}, rho={led.rho_total:.4f})")
    result = {"ledger": led, "caps": rep, "records": records, "capped": capped, "measurements": ms,
              "vocab": vocab, "rho": rho}
    if True:
        true_ms, _ = build_behaviour_measurements(
            capped, vocab, 1e12, np.random.default_rng(seed + 1), tz_offset_s=TZ_OFFSET_S,
            detour_fn=lambda t: circ.get((t.origin, t.dest, t.t_start)),
            hierarchy=hspec, split=split, vocab_override={m.name: m.keys for m in ms})
        result["snr"] = snr_report(true_ms, ms)
        result["true"] = true_ms
    return result


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="dir containing points.parquet (default: GEOLIFE_DATA_DIR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=6.0)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--places", default=None, help="places parquet (id, category, lat, lon, popularity)")
    ap.add_argument("--max-places", type=int, default=20000)
    ap.add_argument("--h3-res", type=int, default=8)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    default=list(BEIJING_CORE))
    ap.add_argument("--flat", action="store_true", help="v0 behaviour: flat place table, equal rho split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--max-stays", type=int, default=100)
    ap.add_argument("--max-trips", type=int, default=100)
    ap.add_argument("--max-distinct-places", type=int, default=20)
    ap.add_argument("--report", default=None, help="write the local SNR report (markdown data) here")
    a = ap.parse_args(argv)
    res = run(a.data or default_data_dir(), a.out, rho=a.rho, epsilon=a.epsilon, delta=a.delta,
              bbox=a.bbox, h3_res=a.h3_res, places=a.places, max_places=a.max_places, seed=a.seed,
              caps=(a.max_stays, a.max_trips, a.max_distinct_places), max_users=a.max_users,
              hier=not a.flat, split="equal" if a.flat else "weighted")
    if a.report:
        import json
        Path(a.report).write_text(json.dumps(
            {"snr": res["snr"], "caps": asdict(res["caps"]), "epsilon": res["ledger"].epsilon,
             "rho": res["rho"], "users": len(res["records"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

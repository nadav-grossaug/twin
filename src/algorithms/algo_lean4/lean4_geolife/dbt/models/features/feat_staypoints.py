# Stay extraction, two sources merged:
#   'sp'      in-trajectory staypoints (Li et al.): >= sp_min_dwell_s within
#             sp_radius_m of an anchor point, on a stream resampled to
#             >= sp_resample_s and broken at gaps > sp_stream_break_s.
#             Also measures GPS noise (residuals vs the centroid: sd, MAD,
#             p95, lag-1 autocorrelation) — the enclave-side source of the
#             released noise moments.
#   'implied' inter-trajectory stays: this labelled GeoLife subset records
#             MOVEMENT, so dwells at home/work mostly happen while the
#             device is off. The gap between one trajectory's end and the
#             next one's start (0 < gap <= implied_stay_max_gap_s) is a stay
#             at the end location — dwell observed exactly when the next
#             start is within sp_radius_m, departure-CENSORED otherwise
#             (the user left at an unknown time before the next start).
# Spec parameters come from dbt_project.yml vars (the single versioned
# concept spec); this node sits on trust boundary 2 (its OUTPUT is checked
# by tests and Lean deciders, its code is not proved).
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLUMNS = [
    "user_id", "origin", "lat_c", "lon_c", "ts_arrive_s", "ts_leave_s",
    "dwell_s", "n_points", "resid_sd_m", "resid_mad_m", "resid_p95_m",
    "resid_lag1", "arr_censored", "dep_censored", "day_idx",
    "arrive_hour", "leave_hour", "mid_hour", "mid_dow", "contains_night_hour",
]


def _spec_vars():
    for cand in ("dbt_project.yml", "dbt/dbt_project.yml", "../dbt/dbt_project.yml"):
        p = Path(cand)
        if p.exists():
            return yaml.safe_load(p.read_text())["vars"]
    raise FileNotFoundError("dbt_project.yml not found from " + str(Path.cwd()))


def _hour(ts):
    return (ts // 3600) % 24


def _dow(ts):
    return (ts // 86400 + 3) % 7  # 1970-01-01 is a Thursday; 0 = Monday


def _mk_stay(uid, origin, lat_c, lon_c, arrive, leave, n_points,
             resid=None, arr_censored=False, dep_censored=False,
             night_hour=3):
    mid = (arrive + leave) // 2
    r = resid or {}
    return {
        "user_id": uid,
        "origin": origin,
        "lat_c": lat_c,
        "lon_c": lon_c,
        "ts_arrive_s": arrive,
        "ts_leave_s": leave,
        "dwell_s": leave - arrive,
        "n_points": n_points,
        "resid_sd_m": r.get("sd", np.nan),
        "resid_mad_m": r.get("mad", np.nan),
        "resid_p95_m": r.get("p95", np.nan),
        "resid_lag1": r.get("lag1", np.nan),
        "arr_censored": arr_censored,
        "dep_censored": dep_censored,
        "day_idx": arrive // 86400,
        "arrive_hour": int(_hour(arrive)),
        "leave_hour": int(_hour(leave)),
        "mid_hour": int(_hour(mid)),
        "mid_dow": int(_dow(mid)),
        # sleep test: the stay spans night_anchor_hour local
        "contains_night_hour": bool(
            (leave - night_hour * 3600) // 86400
            >= -((-(arrive - night_hour * 3600)) // 86400)
        ),
    }


def _staypoints_for_user(uid, ts, lat, lon, v, out):
    radius = float(v["sp_radius_m"])
    min_dwell = int(v["sp_min_dwell_s"])
    resample = int(v["sp_resample_s"])
    stream_break = int(v["sp_stream_break_s"])
    night_hour = int(v["night_anchor_hour"])

    # resample to >= resample seconds between kept points
    keep = np.zeros(len(ts), dtype=bool)
    last = -10**18
    for i in range(len(ts)):
        if ts[i] - last >= resample:
            keep[i] = True
            last = ts[i]
    ts, lat, lon = ts[keep], lat[keep], lon[keep]
    if len(ts) == 0:
        return

    # local equirectangular meters at the user's median latitude
    lat0 = float(np.median(lat))
    lon0 = float(np.median(lon))
    kx = 111320.0 * np.cos(np.radians(lat0))
    ky = 111320.0
    x = (lon - lon0) * kx
    y = (lat - lat0) * ky

    # stream chunks at recording gaps
    breaks = np.flatnonzero(np.diff(ts) > stream_break)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(ts)]))

    for c0, c1 in zip(starts, ends):
        i = c0
        while i < c1 - 1:
            dx = x[i + 1:c1] - x[i]
            dy = y[i + 1:c1] - y[i]
            outside = np.flatnonzero(dx * dx + dy * dy > radius * radius)
            j = (i + 1 + outside[0]) if len(outside) else c1
            if ts[j - 1] - ts[i] >= min_dwell:
                sl = slice(i, j)
                cx, cy = float(np.mean(x[sl])), float(np.mean(y[sl]))
                ex, ey = x[sl] - cx, y[sl] - cy
                r = np.sqrt(ex * ex + ey * ey)
                denom = float(np.sum(ex * ex) + np.sum(ey * ey))
                lag1 = (
                    float(np.sum(ex[:-1] * ex[1:]) + np.sum(ey[:-1] * ey[1:])) / denom
                    if denom > 0 and j - i > 2 else np.nan
                )
                out.append(_mk_stay(
                    uid, "sp",
                    lat0 + cy / ky, lon0 + cx / kx,
                    int(ts[i]), int(ts[j - 1]), int(j - i),
                    resid={"sd": float(np.sqrt(np.mean(r * r))),
                           "mad": float(np.median(r)),
                           "p95": float(np.quantile(r, 0.95)),
                           "lag1": lag1},
                    arr_censored=i == c0,
                    dep_censored=j == c1,
                    night_hour=night_hour,
                ))
                i = j
            else:
                i += 1


def _implied_for_user(uid, trajs, v, out):
    """trajs: DataFrame with one row per trajectory (ts_start, ts_end,
    lat/lon of both ends), sorted by ts_start."""
    radius = float(v["sp_radius_m"])
    max_gap = int(v["implied_stay_max_gap_s"])
    night_hour = int(v["night_anchor_hour"])
    kx = 111320.0 * np.cos(np.radians(float(trajs.lat_end.median())))
    prev = None
    for row in trajs.itertuples():
        if prev is not None:
            gap = int(row.ts_start - prev.ts_end)
            if 0 < gap <= max_gap:
                dx = (row.lat_start - prev.lat_end) * 111320.0
                dy = (row.lon_start - prev.lon_end) * kx
                same_place = dx * dx + dy * dy <= radius * radius
                out.append(_mk_stay(
                    uid, "implied",
                    (prev.lat_end + row.lat_start) / 2 if same_place else prev.lat_end,
                    (prev.lon_end + row.lon_start) / 2 if same_place else prev.lon_end,
                    int(prev.ts_end), int(row.ts_start), 2,
                    dep_censored=not same_place,
                    night_hour=night_hour,
                ))
        prev = row


def model(dbt, session):
    dbt.config(materialized="table", packages=["pyyaml"])
    v = _spec_vars()
    pts = dbt.ref("stg_points").df()[
        ["user_id", "trajectory_id", "ts_s", "latitude", "longitude"]]
    pts = pts.sort_values(["user_id", "ts_s"], kind="stable")
    out = []
    for uid, g in pts.groupby("user_id", sort=True):
        _staypoints_for_user(
            uid,
            g.ts_s.to_numpy(np.int64),
            g.latitude.to_numpy(np.float64),
            g.longitude.to_numpy(np.float64),
            v, out,
        )
        t = g.groupby("trajectory_id").agg(
            ts_start=("ts_s", "min"), ts_end=("ts_s", "max"),
            lat_start=("latitude", "first"), lon_start=("longitude", "first"),
            lat_end=("latitude", "last"), lon_end=("longitude", "last"),
        ).sort_values("ts_start")
        _implied_for_user(uid, t, v, out)
    df = pd.DataFrame(out, columns=COLUMNS)
    df = df.sort_values(["user_id", "ts_arrive_s"]).reset_index(drop=True)
    df.insert(1, "sp_id", df.groupby("user_id").cumcount() + 1)
    return df

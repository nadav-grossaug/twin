"""Vectorised helpers for the outing EDA (T0). Pure numpy/pandas; no per-point Python loops.

Conventions: times are UTC epoch seconds (GeoLife is GMT); local Beijing time is UTC+8, so
hour-of-day / weekday analyses add LOCAL_OFFSET_S first. Points are sorted by (user, t) and
unique per (user, t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from geolife.transfer.geo import M_PER_DEG, detect_stays, from_xy, step_dist_dt, to_xy  # noqa: F401

LOCAL_OFFSET_S = 8 * 3600
CORE_BOX = (39.8, 40.1, 116.2, 116.6)  # lat_min, lat_max, lon_min, lon_max


def clean_points(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid coordinates and duplicate (uid, t); sort by (uid, t); add integer user code
    `u` and drop isolated speed spikes (both adjacent steps above 70 m/s)."""
    df = df[(df.lat.abs() <= 90) & (df.lon.abs() <= 180)]
    df = df.sort_values(["uid", "t", "traj"], kind="stable").drop_duplicates(["uid", "t"])
    df = df.reset_index(drop=True)
    df["u"] = pd.factorize(df.uid)[0].astype(np.int32)
    d, dt = step_dist_dt(df.u.values, df.t.values, df.lat.values, df.lon.values)
    with np.errstate(divide="ignore", invalid="ignore"):
        fast = (d / dt) > 70
    # step k joins point k and k+1; a spike point has fast step before and after
    spike = np.zeros(len(df), bool)
    spike[1:-1] = fast[:-1] & fast[1:]
    return df[~spike].reset_index(drop=True)


def outing_ids(u: np.ndarray, t: np.ndarray, gap_s: float) -> np.ndarray:
    """Outing id per point: a new outing starts at a user change or a time gap > gap_s
    (across file boundaries too, so consecutive files closer than gap_s merge)."""
    new = np.ones(len(u), bool)
    new[1:] = (u[1:] != u[:-1]) | (np.diff(t) > gap_s)
    return np.cumsum(new) - 1


def outing_table(df: pd.DataFrame, oid: np.ndarray, dist_m: np.ndarray) -> pd.DataFrame:
    """Per-outing summary: user, t0, t1, n points, length (m, steps inside the outing), n files."""
    starts = np.flatnonzero(np.r_[True, oid[1:] != oid[:-1]])
    ends = np.r_[starts[1:], len(oid)] - 1
    step = np.r_[np.nan_to_num(dist_m, nan=0.0), 0.0]  # step k belongs to point k
    cs = np.r_[0.0, np.cumsum(step)]
    length = cs[ends] - cs[starts]
    traj = df.traj.values
    nfiles = np.add.reduceat(np.r_[True, (traj[1:] != traj[:-1]) | (oid[1:] != oid[:-1])].astype(int), starts)
    return pd.DataFrame({"u": df.u.values[starts], "t0": df.t.values[starts], "t1": df.t.values[ends],
                         "n": ends - starts + 1, "len_m": length, "nfiles": nfiles})


def file_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per trajectory file: user, t0, t1, n; plus gap (s) to the previous file of the same user."""
    g = df.groupby(["u", "traj"], sort=False).agg(t0=("t", "min"), t1=("t", "max"), n=("t", "size")).reset_index()
    g = g.sort_values(["u", "t0"]).reset_index(drop=True)
    gap = g.t0.values[1:] - g.t1.values[:-1]
    g["gap_prev"] = np.r_[np.nan, np.where(g.u.values[1:] == g.u.values[:-1], gap, np.nan)]
    return g


def local_hour_dow(t: np.ndarray):
    lt = np.asarray(t) + LOCAL_OFFSET_S
    return (lt % 86400) // 3600, ((lt // 86400) + 3) % 7  # 1970-01-01 was a Thursday -> Mon=0


def local_day(t: np.ndarray):
    return (np.asarray(t) + LOCAL_OFFSET_S) // 86400


def tod_bin4(hour):
    """0 night 0-6, 1 morning 6-12, 2 afternoon 12-18, 3 evening 18-24 (local)."""
    return np.asarray(hour) // 6


def thin_30s(df: pd.DataFrame, step: int = 30) -> pd.DataFrame:
    key = df.u.values.astype(np.int64) * (1 << 36) + df.t.values // step
    keep = np.r_[True, key[1:] != key[:-1]]
    return df[keep].reset_index(drop=True)


def h3_cells(lat, lon, resolutions=(6, 7, 8, 9, 10), decimals: int = 4) -> dict[int, np.ndarray]:
    """H3 cell ids (as string-free int codes via pd.factorize across the unique rounded positions)
    for each resolution. Returns {res: int64 array of h3 int ids} aligned with the input."""
    import h3
    la = np.round(np.asarray(lat), decimals)
    lo = np.round(np.asarray(lon), decimals)
    key = (np.round(la * 10 ** decimals).astype(np.int64) + 90 * 10 ** decimals) * 10 ** 7 \
        + (np.round(lo * 10 ** decimals).astype(np.int64) + 180 * 10 ** decimals)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    ulat, ulon = la[first], lo[first]
    out = {}
    base = max(resolutions)
    fine = [h3.latlng_to_cell(a, b, base) for a, b in zip(ulat.tolist(), ulon.tolist())]
    for r in resolutions:
        cells = fine if r == base else [h3.cell_to_parent(c, r) for c in fine]
        out[r] = np.array([int(c, 16) for c in cells], np.int64)[inv]
    return out


def covisit_graph(users: np.ndarray, cells: np.ndarray, max_pop: int = 10, min_shared: int = 2):
    """Rarity-restricted co-visit graph from (user, cell) incidences.

    Cells visited by 2..max_pop distinct users count as 'rare shared'. Two users are linked if they
    share >= min_shared such cells. Edge weight = sum over shared rare cells of 1/log(1+n_c)...
    returned as (edges set of (a,b) a<b, weights dict, degree per user dict, n_users).
    Returns a DataFrame with columns a, b, shared, w."""
    import scipy.sparse as sp
    inc = pd.DataFrame({"u": users, "c": cells}).drop_duplicates()
    cc, ci = np.unique(inc.c.values, return_inverse=True)
    uu, ui = np.unique(inc.u.values, return_inverse=True)
    pop = np.bincount(ci)
    keep = (pop[ci] >= 2) & (pop[ci] <= max_pop)
    if not keep.any():
        return pd.DataFrame({"a": [], "b": [], "shared": [], "w": []})
    wcell = 1.0 / np.log1p(pop.astype(float))
    A = sp.csr_matrix((np.ones(keep.sum()), (ui[keep], ci[keep])), shape=(len(uu), len(cc)))
    S = (A @ A.T).tocoo()
    W = (A.multiply(wcell[None, :]) @ A.T).tocoo()
    m = S.row < S.col
    shared = pd.DataFrame({"a": uu[S.row[m]], "b": uu[S.col[m]], "shared": S.data[m]})
    wd = pd.DataFrame({"a": uu[W.row], "b": uu[W.col], "w": W.data})
    g = shared.merge(wd, on=["a", "b"])
    return g[g.shared >= min_shared].reset_index(drop=True)


def degrees(edges: pd.DataFrame, n_users: int, user_ids=None) -> np.ndarray:
    """Degree array over `user_ids` (all users if None => array length n_users, ids 0..n-1)."""
    ids = np.arange(n_users) if user_ids is None else np.asarray(user_ids)
    deg = pd.Series(np.r_[edges.a.values, edges.b.values]).value_counts()
    return deg.reindex(ids).fillna(0).astype(int).values


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def incidence(cells: np.ndarray, oid: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Unique (outing, cell) pairs from point-aligned cell ids (memory-light: packs dense codes)."""
    if mask is not None:
        cells, oid = cells[mask], oid[mask]
    codes, uniq = pd.factorize(cells)
    packed = np.unique(oid.astype(np.int64) * (1 << 24) + codes)
    return pd.DataFrame({"oid": packed >> 24, "cell": np.asarray(uniq)[packed & ((1 << 24) - 1)]})


def mode_per_outing(oid: np.ndarray, code: np.ndarray, n_out: int, n_modes: int, min_lab: float = 0.5):
    """Dominant labelled mode per outing (-1 if < min_lab of its points are labelled) and the
    dominant mode's share among labelled points."""
    m = code >= 0
    cnt = np.bincount(oid[m] * n_modes + code[m], minlength=n_out * n_modes).reshape(n_out, n_modes)
    tot = np.bincount(oid, minlength=n_out)
    lab = cnt.sum(1)
    mode = np.where(lab >= min_lab * np.maximum(tot, 1), cnt.argmax(1), -1)
    share = np.where(lab > 0, cnt.max(1) / np.maximum(lab, 1), np.nan)
    return mode, share


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

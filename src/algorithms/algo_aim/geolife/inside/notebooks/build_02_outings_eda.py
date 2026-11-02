"""Builds geolife/inside/notebooks/02_outings_eda.ipynb (python build_02_outings_eda.py; execute with run_nb.py).

Run from the repo root.  Heavy lifting is in src/geolife/inside/eda_outings.py (vectorised numpy).
"""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# GeoLife outings EDA (T0)

Fixes the parameters of the outings pipeline (docs/geolife/PLAN_OUTINGS.md) from all 182 real users.
User ids are seeded-pseudonym free here: only integer codes appear, never real ids.

**Time convention.** GeoLife timestamps are GMT. Beijing local time is UTC+8 (no DST), so every
hour-of-day / weekday statistic below uses `t + 8h`. Whatever hour convention the DP artifact and
`generate/` use must be declared in the artifact; the synthetic render should draw local hours and
subtract 8 h only when it writes GMT-like timestamps.

**Ledger caveat.** Every choice informed by this notebook is *tuning on real data* and has to be listed in the
privacy ledger as unaccounted public-parameter selection (the final table says which).""")

code("""import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "geolife").exists())
sys.path.insert(0, str(ROOT / "src"))
from geolife.inside import eda_outings as E, data as gd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

# dataviz palette (light): categorical slots 1-4, sequential blue ramp, neutrals
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CAT = [BLUE, ORANGE, AQUA, VIOLET]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.dpi": 110, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": .6, "lines.linewidth": 1.6, "legend.frameon": False,
    "legend.fontsize": 8})

# ---- PUBLIC PARAMETERS (fixed here by reasoning, then checked against the numbers below) ----
GAP = 7200          # outing break (s)
DIST, DWELL = 200.0, 600.0     # stop rule: radius (m), dwell (s)
QUAL_DUR, QUAL_LEN = 300, 500  # a 'real' outing: >= 5 min and >= 500 m
MIN_OUT, MIN_DAYS = 10, 5      # min-activity rule: >= MIN_OUT real outings on >= MIN_DAYS local days
GAPS = [1800, 3600, 7200, 14400]
SEED = 20261101
print("local = UTC+8; GAP", GAP, "stop", DIST, "m /", DWELL, "s")""")

code("""t0 = time.time()
cands = [os.environ.get("GEOLIFE_POINTS", ""), str(ROOT / "geolife" / "inside" / "data" / "points.parquet"),
         "C:/Users/NaomiD/Documents/Projects/hackaton-2026/.claude/worktrees/agent-a04900ec477b059b7/geolife/data/points.parquet"]
cache = next((c for c in cands if c and Path(c).exists()), None)
assert cache, "points.parquet cache not found (set GEOLIFE_POINTS)"
raw = pd.read_parquet(cache)
print("raw points", f"{len(raw):,}", "users", raw.uid.nunique(), "files", raw.groupby(["uid", "traj"]).ngroups)
df = E.clean_points(raw); del raw
N_USERS = df.u.nunique()
print("clean points", f"{len(df):,}", "(dropped invalid coords, duplicate timestamps, isolated >70 m/s spikes)", f"{time.time()-t0:.0f}s")
U, T, LAT, LON = df.u.values, df.t.values, df.lat.values, df.lon.values
D, DT = E.step_dist_dt(U, T, LAT, LON)                    # step k joins point k and k+1
SAMEFILE = (df.traj.values[1:] == df.traj.values[:-1]) & (U[1:] == U[:-1])
HOUR, DOW = E.local_hour_dow(T)
print("local hour/dow check (first pt): hour", HOUR[0], "dow", DOW[0], "UTC", pd.to_datetime(T[0], unit="s"))
df["day"] = E.local_day(T)""")

md("""## 1. Outing definition

An outing is a maximal run of one user's points with no time gap above the threshold, regardless of
file boundaries (consecutive files closer than the threshold merge; a long gap inside a file splits it).""")
code("""files = E.file_table(df)
g_prev = files.gap_prev.dropna().values
print("files:", len(files), "| consecutive-file gaps (s) quantiles 10/25/50/75/90/99:", np.percentile(g_prev, [10, 25, 50, 75, 90, 99]).round(0))
print("  overlapping/abutting files (gap<=0): %.3f | gap<=30s: %.3f | <=30min: %.3f | <=1h: %.3f | <=2h: %.3f | <=4h: %.3f | >1 day: %.3f" % (
      (g_prev <= 0).mean(), (g_prev <= 30).mean(), (g_prev <= 1800).mean(), (g_prev <= 3600).mean(), (g_prev <= 7200).mean(), (g_prev <= 14400).mean(), (g_prev > 86400).mean()))
wd = DT[SAMEFILE]
print("within-file dt (s) quantiles 50/90/99/99.9:", np.percentile(wd, [50, 90, 99, 99.9]))
nf_user = files.groupby("u").size()
for thr in (60, 300, 1800, 3600, 7200, 14400):
    print(f"  within-file gaps > {thr:>5}s: {(wd > thr).sum():>7,}  ({(wd>thr).sum()/files.shape[0]:.2f} per file)")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
lb = np.logspace(0, 7, 70)
ax[0].hist(g_prev[g_prev > 0], bins=lb, color=BLUE); ax[0].set_xscale("log")
ax[0].set(title="Gap between consecutive files of a user", xlabel="seconds (log)", ylabel="file pairs")
ax[1].hist(wd[wd > 10], bins=lb, color=ORANGE); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set(title="Gaps inside files (dt > 10 s)", xlabel="seconds (log)", ylabel="steps")
for a in ax:
    for v, l in ((1800, "30m"), (3600, "1h"), (7200, "2h"), (14400, "4h")): a.axvline(v, color="#8a8f98", lw=.7, ls=":")
plt.tight_layout(); plt.show()""")

code("""# labels -> per-point mode code (labelled subset: 69 users)
MODEMAP = {"walk": "walk", "run": "walk", "bike": "bike", "bus": "bus", "car": "car/taxi", "taxi": "car/taxi",
           "train": "rail", "subway": "rail", "railway": "rail", "airplane": "other", "boat": "other", "motorcycle": "other"}
MODES = ["walk", "bike", "bus", "car/taxi", "rail", "other"]
data_root = gd.default_data_dir()
umap = df.groupby("uid").u.first()
rows = []
for ud in sorted((data_root / "Data").iterdir()):
    lp = ud / "labels.txt"
    if lp.exists() and ud.name in umap.index:
        for s in gd.read_labels(lp): rows.append((umap[ud.name], s.t_start, s.t_end, MODEMAP.get(s.mode, "other")))
lab = pd.DataFrame(rows, columns=["u", "t0", "t1", "mode"]).sort_values(["u", "t0"]).reset_index(drop=True)
K0 = lab.u.values.astype(np.int64) * 10**10 + lab.t0.values
idx = np.searchsorted(K0, U.astype(np.int64) * 10**10 + T, side="right") - 1
idc = np.clip(idx, 0, None)
okl = (idx >= 0) & (lab.u.values[idc] == U) & (T <= lab.t1.values[idc])
MODECODE = np.where(okl, pd.Categorical(lab["mode"], categories=MODES).codes[idc], -1).astype(np.int8)
print("labelled users", lab.u.nunique(), "segments", len(lab), "| labelled points %.3f" % okl.mean())""")

code("""O = {}
rows = []
for g in GAPS:
    oid = E.outing_ids(U, T, g)
    o = E.outing_table(df, oid, D)
    o["dur_min"] = (o.t1 - o.t0) / 60; o["len_km"] = o.len_m / 1000
    o["qual"] = (o.dur_min * 60 >= QUAL_DUR) & (o.len_m >= QUAL_LEN)
    o["day"] = E.local_day(o.t0.values)
    O[g] = (oid, o)
    per_user = o[o.qual].groupby("u").size().reindex(range(N_USERS)).fillna(0)
    rows.append({"gap": g, "outings": len(o), "real(>=5min,>=500m)": int(o.qual.sum()), "share_real": round(o.qual.mean(), 3),
                 "med_dur_min": o.dur_min.median().round(1), "p90_dur_min": o.dur_min.quantile(.9).round(0),
                 "med_len_km": o.len_km.median().round(2), "p90_len_km": o.len_km.quantile(.9).round(1),
                 "med_pts": int(o.n.median()), "multi_file_share": round((o.nfiles > 1).mean(), 3),
                 "users>=10_real": int((per_user >= 10).sum()), "med_real/user": per_user.median(), "p90/user": per_user.quantile(.9), "max/user": int(per_user.max()),
                 "top10%users_share": round(per_user.sort_values(ascending=False).head(N_USERS // 10).sum() / per_user.sum(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
for g, c in zip(GAPS, CAT):
    o = O[g][1]; lbl = {1800: "30 min", 3600: "1 h", 7200: "2 h", 14400: "4 h"}[g]
    for a, v, rng in ((ax[0], o.dur_min.clip(lower=.1), (-1, 3.5)), (ax[1], o.len_km.clip(lower=.001), (-3, 3.5)), (ax[2], o.n.clip(lower=1), (0, 6.5))):
        x = np.sort(np.log10(v.values)); a.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=lbl)
ax[0].set(title="Outing duration (CDF)", xlabel="log10 minutes", ylabel="share of outings")
ax[1].set(title="Outing length (CDF)", xlabel="log10 km"); ax[2].set(title="Points per outing (CDF)", xlabel="log10 points")
ax[0].legend(title="gap", loc="lower right"); plt.tight_layout(); plt.show()""")

code("""# mode on the labelled subset, chosen gap
oid_g, og = O[GAP]
mode, share = E.mode_per_outing(oid_g, MODECODE, len(og), len(MODES))
og["mode"] = mode; og["mode_share"] = share
lo = og[og["mode"] >= 0]
print("outings with >=50%% labelled points: %d of %d (%.3f); dominant-mode share median %.2f, share pure(>=0.8): %.3f" % (
      len(lo), len(og), len(lo) / len(og), lo.mode_share.median(), (lo.mode_share >= .8).mean()))
tab = lo.assign(m=[MODES[i] for i in lo["mode"]]).groupby("m").agg(outings=("n", "size"), users=("u", "nunique"),
        med_dur_min=("dur_min", "median"), med_len_km=("len_km", "median"), p90_len_km=("len_km", lambda s: s.quantile(.9))).round(2)
print(tab.sort_values("outings", ascending=False))""")

code("""# per-user: outings, active days, span (chosen gap)
df["oid"] = oid_g
th = E.thin_30s(df)                                    # 30 s thinning (first point per user per 30 s)
days = th.groupby("u").day.agg(active_days="nunique", first="min", last="max")
days["span_days"] = days["last"] - days["first"] + 1
q = og[og.qual]
uq = pd.DataFrame({"real_outings": q.groupby("u").size(), "real_days": q.groupby("u").day.nunique()}).reindex(range(N_USERS)).fillna(0)
uq = uq.join(days[["active_days", "span_days"]]).fillna(0)
print(uq.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(1).T[["min", "10%", "25%", "50%", "75%", "90%", "99%", "max"]])
print("active-day density (active_days/span) median: %.2f" % (uq.active_days / uq.span_days.clip(lower=1)).median())
rule = []
for mo in (3, 5, 10, 20, 30):
    for md_ in (2, 3, 5, 10):
        ok = (uq.real_outings >= mo) & (uq.real_days >= md_)
        rule.append({"min_real_outings": mo, "min_days": md_, "users": int(ok.sum()), "share_real_outings_kept": round(q.u.isin(uq.index[ok]).mean(), 3)})
print(pd.DataFrame(rule).pivot(index="min_real_outings", columns="min_days", values="users"))
QU = uq.index[(uq.real_outings >= MIN_OUT) & (uq.real_days >= MIN_DAYS)].values
print(f"CHOSEN rule: >= {MIN_OUT} real outings on >= {MIN_DAYS} local days -> {len(QU)} users; keeps {q.u.isin(QU).mean():.3f} of real outings")
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
s = uq.real_outings.sort_values(ascending=False).values
ax[0].plot(np.arange(1, len(s) + 1), s + 1, color=BLUE); ax[0].set_yscale("log"); ax[0].set_xscale("log")
ax[0].set(title="Real outings per user, ranked", xlabel="user rank", ylabel="outings + 1"); ax[0].axhline(MIN_OUT, color=ORANGE, ls=":"); ax[0].text(1.2, MIN_OUT * 1.15, "min-activity", color=INK2, fontsize=8)
ax[1].hist(uq.active_days.clip(upper=400), bins=40, color=BLUE); ax[1].set(title="Active local days per user", xlabel="days (clipped 400)", ylabel="users")
ax[2].scatter(uq.span_days.clip(lower=1), uq.active_days.clip(lower=1), s=8, color=BLUE); ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].plot([1, 2000], [1, 2000], color="#8a8f98", lw=.7, ls=":"); ax[2].set(title="Active days vs span", xlabel="span (days)", ylabel="active days")
plt.tight_layout(); plt.show()""")

md("""## 2. Stops

Stop = Li et al. stay-point (distance radius around an anchor, minimum dwell), run on points thinned to one
per 30 s and segmented by outing (a stop never spans an outing break). Dwell ends only where the log
ends: an outing that ends *at* a destination yields no stop for it (start/end places are separate visits
in the pipeline, T1).""")
code("""X, Y = E.to_xy(th.lat.values, th.lon.values)
TT = th.t.values.astype(float); OID_T = th.oid.values
n_out = len(og)
def stays_for(dist, dwell):
    i0, i1, mx, my, rms = E.detect_stays(X, Y, TT, OID_T, dist, dwell)
    la, lo = E.from_xy(mx, my)
    return pd.DataFrame({"u": th.u.values[i0], "oid": OID_T[i0], "t0": th.t.values[i0], "t1": th.t.values[i1],
                         "lat": la, "lon": lo, "rms": rms, "npts": i1 - i0 + 1})
def revisit_rate(st, cells):
    k = pd.DataFrame({"u": st.u.values, "c": cells, "o": st.oid.values})
    first = k.groupby(["u", "c"]).o.transform("min")
    return float((k.o > first).mean())
rows, ST = [], {}
qual_oid = og.index[og.qual & og.u.isin(QU)].values
for dist in (100, 200, 300):
    for dwell in (300, 600, 1200):
        st = stays_for(dist, dwell); ST[(dist, dwell)] = st
        cs = E.h3_cells(st.lat.values, st.lon.values, (8, 9, 10))
        cnt = np.bincount(st.oid.values, minlength=n_out)[qual_oid]
        r = {"dist_m": dist, "dwell_min": dwell // 60, "stops": len(st), "users_with_stop": st.u.nunique(),
             "stops/real_outing": round(cnt.mean(), 2), "outings>=1": round((cnt >= 1).mean(), 3), "outings>=2": round((cnt >= 2).mean(), 3),
             "med_dwell_min": round(((st.t1 - st.t0) / 60).median(), 1)}
        for res in (8, 9, 10): r[f"revisit_r{res}"] = round(revisit_rate(st, cs[res]), 3)
        sq = st[st.u.isin(QU)]; csq = E.h3_cells(sq.lat.values, sq.lon.values, (8, 9))
        for res in (8, 9):
            r[f"places/user_r{res}"] = pd.Series(csq[res]).groupby(sq.u.values).nunique().median()
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))""")
code("""st = ST[(int(DIST), int(DWELL))]
cnt = np.bincount(st.oid.values, minlength=n_out)[qual_oid]
print("CHOSEN stop rule", DIST, "m /", DWELL // 60, "min:", len(st), "stops;", "stops per real outing (QU users) p50/p90/p99/max:", np.percentile(cnt, [50, 90, 99, 100]))
print("stop dwell (min) quantiles 10/50/90/99:", (((st.t1 - st.t0) / 60).quantile([.1, .5, .9, .99])).round(1).tolist())
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
ax[0].bar(*np.unique(np.minimum(cnt, 8), return_counts=True), color=BLUE); ax[0].set(title="Stops per real outing (8 = 8+)", xlabel="stops", ylabel="outings")
ax[1].hist(np.log10(((st.t1 - st.t0) / 60).clip(lower=1)), bins=40, color=BLUE); ax[1].set(title="Stop dwell", xlabel="log10 minutes")
ax[2].hist(st.rms.clip(upper=150), bins=40, color=BLUE); ax[2].set(title="Point spread inside a stop (rms radius)", xlabel="metres (clipped 150)")
plt.tight_layout(); plt.show()
print("jitter proxy: rms radius of 30s-thinned points around stop centre (m) quantiles 10/50/90:", st.rms.quantile([.1, .5, .9]).round(1).tolist())""")

md("""## 3. Time structure (local = UTC+8)""")
code("""stl = st.copy()
stl["hour"], stl["dow"] = E.local_hour_dow(stl.t0.values)
og["hour"], og["dow"] = E.local_hour_dow(og.t0.values)
def user_weighted_hist(user, val, bins):
    h = pd.crosstab(user, pd.Categorical(val, categories=range(bins))).values.astype(float)
    h = h[h.sum(1) > 0]; return (h / h.sum(1, keepdims=True)).mean(0)
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
qo = og[og.qual & og.u.isin(QU)]; qs = stl[stl.u.isin(QU)]
for a, d_, ttl in ((ax[0], qo, "Outing starts"), (ax[1], qs, "Stop arrivals")):
    ev = np.bincount(d_.hour.astype(int), minlength=24) / len(d_)
    uw = user_weighted_hist(d_.u.values, d_.hour.astype(int).values, 24)
    a.plot(range(24), ev, color=BLUE); a.plot(range(24), uw, color=ORANGE)
    a.text(23, ev[-1], " event-weighted", color=BLUE, fontsize=8, va="center", ha="right"); a.text(23, uw[-1] + .004, "user-weighted", color=ORANGE, fontsize=8, ha="right")
    a.set(title=ttl + " by local hour", xlabel="hour (UTC+8)", ylabel="share"); a.set_xticks(range(0, 24, 3))
wk = lambda dw: (dw >= 5).astype(int)
dd = pd.DataFrame({"Outing starts": np.bincount(qo.dow.astype(int), minlength=7) / len(qo), "Stop arrivals": np.bincount(qs.dow.astype(int), minlength=7) / len(qs)})
w = .38
ax[2].bar(np.arange(7) - w / 2, dd["Outing starts"], w, color=BLUE, label="outing starts"); ax[2].bar(np.arange(7) + w / 2, dd["Stop arrivals"], w, color=ORANGE, label="stop arrivals")
ax[2].set(title="Day of week (0=Mon)", xlabel="day", ylabel="share"); ax[2].legend()
plt.tight_layout(); plt.show()
print("weekend share (Sat+Sun, 2/7 = .286 if flat): outing starts %.3f  stops %.3f" % (wk(qo.dow.values).mean(), wk(qs.dow.values).mean()))
for nm, d_ in (("outing starts", qo), ("stop arrivals", qs)):
    print(nm, "4 time-of-day bins (night 0-6, morning 6-12, afternoon 12-18, evening 18-24):", (np.bincount(E.tod_bin4(d_.hour.values.astype(int)), minlength=4) / len(d_)).round(3))""")
code("""# distinctness of profiles: top coarse cells and map categories vs the overall 8-bin (4 tod x wd/we) profile
def bin8(hour, dow): return E.tod_bin4(np.asarray(hour).astype(int)) * 2 + (np.asarray(dow) >= 5)
qs = qs.copy(); qs["b8"] = bin8(qs.hour, qs.dow)
qs["c7"] = E.h3_cells(qs.lat.values, qs.lon.values, (7,))[7]
overall = np.bincount(qs.b8, minlength=8).astype(float)
rng = np.random.default_rng(SEED)
def noise_jsd(n, reps=200):
    return np.mean([E.js_divergence(np.bincount(rng.choice(qs.b8.values, n), minlength=8) + 1e-9, overall) for _ in range(reps)])
nu = qs.groupby("c7").u.nunique().sort_values(ascending=False)
rows = []
for c in nu.index[:8]:
    s_ = qs[qs.c7 == c]; p = np.bincount(s_.b8, minlength=8)
    j = E.js_divergence(p + 1e-9, overall); nj = noise_jsd(len(s_))
    rows.append({"cell_r7_rank": len(rows) + 1, "users": nu[c], "stops": len(s_), "JSD_vs_overall": round(j, 4), "JSD_if_random_same_size": round(nj, 4), "ratio": round(j / nj, 1),
                 "night%": round(100 * p[0:2].sum() / p.sum()), "morning%": round(100 * p[2:4].sum() / p.sum()), "afternoon%": round(100 * p[4:6].sum() / p.sum()), "evening%": round(100 * p[6:8].sum() / p.sum()), "weekend%": round(100 * p[1::2].sum() / p.sum())})
print(pd.DataFrame(rows).to_string(index=False))
# map categories (public places table, optional) - snap stop centres within 150 m
pp = next((p for p in [str(ROOT / "geolife" / "outside" / "cache" / "places_beijing_big.parquet"), "C:/Users/NaomiD/Documents/Projects/hackaton-2026/.claude/worktrees/agent-a63109956b0e67a7c/geolife/cache/places_beijing_big.parquet"] if Path(p).exists()), None)
cat_prof = None
if pp:
    from scipy.spatial import cKDTree
    pl = pd.read_parquet(pp); px, py = E.to_xy(pl.lat.values, pl.lon.values); tree = cKDTree(np.c_[px, py])
    sx, sy = E.to_xy(qs.lat.values, qs.lon.values); dd_, ii = tree.query(np.c_[sx, sy], distance_upper_bound=150)
    hit = np.isfinite(dd_); qs["cat"] = np.where(hit, pl.category.values[np.minimum(ii, len(pl) - 1)], None)
    print("places table:", len(pl), "places;", "stops snapped within 150 m: %.3f" % hit.mean(), "(coverage of the public table, not of reality)")
    rows = []
    for cname, s_ in qs[hit].groupby("cat"):
        if len(s_) < 150: continue
        p = np.bincount(s_.b8, minlength=8); j = E.js_divergence(p + 1e-9, overall); nj = noise_jsd(len(s_))
        rows.append({"category": cname, "stops": len(s_), "users": s_.u.nunique(), "JSD": round(j, 4), "noise_JSD": round(nj, 4), "ratio": round(j / nj, 1)})
    cat_prof = pd.DataFrame(rows).sort_values("stops", ascending=False); print(cat_prof.to_string(index=False))
    heat = qs[hit].groupby(["cat", "b8"]).size().unstack(fill_value=0)
    heat = heat[heat.sum(1) >= 150]; heat = heat.div(heat.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(heat) + 1.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(8)); ax.set_xticklabels([f"{n}\\n{w_}" for n in ("night", "morn", "aftn", "eve") for w_ in ("wd", "we")], fontsize=7)
    ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index, fontsize=8); ax.grid(False)
    for i in range(heat.shape[0]):
        for j in range(8): ax.text(j, i, f"{100*heat.values[i,j]:.0f}", ha="center", va="center", fontsize=7, color=INK if heat.values[i, j] < .3 else "white")
    ax.set_title("Stop arrival profile by map category (% of the category's stops)"); plt.tight_layout(); plt.show()
else:
    print("places table not available; category profiles skipped")""")
code("""# do finer time bins survive user-level counts?  distinct users per (res-7 cell, time bin)
schemes = {"1 (none)": lambda h, d: np.zeros(len(h), int), "2 (wd/we)": lambda h, d: (d >= 5).astype(int), "4 tod": lambda h, d: h // 6,
           "8 (4 tod x wd/we)": lambda h, d: (h // 6) * 2 + (d >= 5), "12 (4h x wd/we)": lambda h, d: (h // 4) * 2 + (d >= 5),
           "24 hours": lambda h, d: h, "48 (24h x wd/we)": lambda h, d: h * 2 + (d >= 5)}
h_, d_ = qs.hour.values.astype(int), qs.dow.values
rows = []
for nm, f in schemes.items():
    b = f(h_, d_); k = pd.DataFrame({"u": qs.u.values, "c": qs.c7.values, "b": b}).drop_duplicates()
    nuc = k.groupby(["c", "b"]).u.transform("nunique")
    ncells = k.groupby(["c", "b"]).ngroups
    rows.append({"bins": nm, "(cell,bin) keys": ncells, "keys>=10 users": int((k.groupby(["c", "b"]).u.nunique() >= 10).sum()),
                 "share user-keys in >=10-user keys": round(float((nuc >= 10).mean()), 3), "share in >=5-user keys": round(float((nuc >= 5).mean()), 3),
                 "stops/bin (median over bins)": int(pd.Series(b).value_counts().median())})
print(pd.DataFrame(rows).to_string(index=False))
print("stops (QU users):", len(qs), " users:", qs.u.nunique())""")

md("""## 4. Overlap (co-visit)

Cells are H3 at res 6-9. *Stops* = stop centres; *trajectory cells* = every cell traversed (a road proxy;
positions rounded to 1e-4 deg, about 11 m, before cell lookup). Only users passing the min-activity rule.
`core` = lat 39.8-40.1, lon 116.2-116.6 (Haidian / 3rd ring), everything else is `all`.""")
code("""t1 = time.time()
RES = (6, 7, 8, 9)
CELLS = E.h3_cells(LAT, LON, RES)
print("trajectory cells computed", f"{time.time()-t1:.0f}s")
core_pt = (LAT >= E.CORE_BOX[0]) & (LAT <= E.CORE_BOX[1]) & (LON >= E.CORE_BOX[2]) & (LON <= E.CORE_BOX[3])
u_of_oid = og.u.values
INC = {}   # (kind,res,scope) -> DataFrame(oid, cell)
for r in RES:
    INC[("traj", r, "all")] = E.incidence(CELLS[r], oid_g)
    INC[("traj", r, "core")] = E.incidence(CELLS[r], oid_g, core_pt)
CELLS = None
stq = st.copy(); stq["hour"], stq["dow"] = E.local_hour_dow(stq.t0.values)
stc = E.h3_cells(stq.lat.values, stq.lon.values, RES + (10,))
core_st = (stq.lat.values >= E.CORE_BOX[0]) & (stq.lat.values <= E.CORE_BOX[1]) & (stq.lon.values >= E.CORE_BOX[2]) & (stq.lon.values <= E.CORE_BOX[3])
for r in RES + (10,):
    INC[("stop", r, "all")] = pd.DataFrame({"oid": stq.oid.values, "cell": stc[r]}).drop_duplicates()
    INC[("stop", r, "core")] = pd.DataFrame({"oid": stq.oid.values[core_st], "cell": stc[r][core_st]}).drop_duplicates()
# restrict to users passing the activity rule; incidence by USER
QSET = set(QU.tolist())
def by_user(inc):
    x = inc.assign(u=u_of_oid[inc.oid.values]); x = x[x.u.isin(QSET)]
    return x[["u", "cell"]].drop_duplicates().reset_index(drop=True)
UC = {k: by_user(v) for k, v in INC.items()}
print(f"{time.time()-t1:.0f}s; users in play: {len(QU)}")
rows = []
for (kind, r, scope), uc in UC.items():
    if r == 10 and kind == "traj": continue
    nu_ = uc.groupby("cell").u.nunique(); ncell_u = uc.groupby("u").cell.nunique()
    inc_n = uc.cell.map(nu_)
    rows.append({"kind": kind, "res": r, "scope": scope, "cells": len(nu_), "med_users/cell": nu_.median(), "max_users/cell": nu_.max(),
                 "med_cells/user": ncell_u.median(), "incid>=2u": round((inc_n >= 2).mean(), 3), "incid>=5u": round((inc_n >= 5).mean(), 3), "incid>=10u": round((inc_n >= 10).mean(), 3),
                 "cells>=5u": int((nu_ >= 5).sum()), "cells>=10u": int((nu_ >= 10).sum())})
ovt = pd.DataFrame(rows).sort_values(["kind", "scope", "res"]); print(ovt.to_string(index=False))""")
code("""def graph_for(uc, max_pop=10, min_shared=2):
    return E.covisit_graph(uc.u.values, uc.cell.values, max_pop=max_pop, min_shared=min_shared)
rows, GR = [], {}
for kind, res_l in (("stop", (7, 8, 9)), ("traj", (8, 9))):
    for scope in ("all", "core"):
        for r in res_l:
            uc = UC[(kind, r, scope)]; users = np.sort(uc.u.unique())
            ed = graph_for(uc); deg = E.degrees(ed, N_USERS, users); GR[(kind, r, scope)] = (ed, users, deg)
            pairs = len(users) * (len(users) - 1) / 2
            rows.append({"kind": kind, "res": r, "scope": scope, "users": len(users), "edges": len(ed), "density": round(len(ed) / max(pairs, 1), 3),
                         "users_with_partner": int((deg > 0).sum()), "share": round((deg > 0).mean(), 3), "deg_med": float(np.median(deg)), "deg_p90": float(np.percentile(deg, 90)), "deg_max": int(deg.max())})
print("edge = share >= 2 'rare' cells (visited by 2..10 users in scope); weight = sum 1/log(1+n_cell)")
print(pd.DataFrame(rows).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for a, scope in zip(ax, ("all", "core")):
    for (kind, r), c, lbl in (((("stop", 7)), BLUE, "stops r7"), ((("stop", 8)), ORANGE, "stops r8"), ((("traj", 8)), AQUA, "traj r8"), ((("traj", 9)), VIOLET, "traj r9")):
        deg = GR[(kind, r, scope)][2]; x = np.arange(0, deg.max() + 1)
        a.step(x, [(deg >= k).mean() for k in x], where="post", color=c, label=lbl)
    a.set(title=f"Co-visit degree CCDF, scope={scope}", xlabel="degree (partners)"); a.legend(loc="upper right")
ax[0].set_ylabel("share of users with degree >= k")
plt.tight_layout(); plt.show()
# sensitivity of the edge definition (stops r8 / traj r9, all scope)
rows = []
for key in (("stop", 8, "all"), ("traj", 9, "all")):
    uc = UC[key]; users = np.sort(uc.u.unique())
    for mp in (5, 10, 20):
        for ms in (1, 2, 3):
            ed = graph_for(uc, mp, ms); deg = E.degrees(ed, N_USERS, users)
            rows.append({"kind": key[0], "res": key[1], "max_pop": mp, "min_shared": ms, "edges": len(ed), "users_with_partner": int((deg > 0).sum()), "deg_med": float(np.median(deg))})
print(pd.DataFrame(rows).to_string(index=False))""")

md("""## 5. Sampling regime (feeds the synthetic render)""")
code("""dtw = DT[SAMEFILE]; dd_ = D[SAMEFILE]
print("within-file dt (s): quantiles 5/25/50/75/90/95/99:", np.percentile(dtw, [5, 25, 50, 75, 90, 95, 99]))
print("dt==1: %.3f  ==2: %.3f  ==3: %.3f  ==5: %.3f  <=5: %.3f | >10 s: %.4f  >60 s: %.5f  >300 s: %.6f" % tuple(
      [(dtw == k).mean() for k in (1, 2, 3, 5)] + [(dtw <= 5).mean(), (dtw > 10).mean(), (dtw > 60).mean(), (dtw > 300).mean()]))
ud = pd.DataFrame({"u": U[1:][SAMEFILE], "dt": dtw}).groupby("u").dt.median()
print("per-user median dt (s) distribution: ", ud.value_counts().sort_index().head(8).to_dict(), "... users:", len(ud))
logged_h = dtw[dtw <= 300].sum() / 3600
print("gaps (dt>60 s) per logged hour (hours = sum of dt<=300 s): %.2f | share of elapsed in-file time spent inside dt>60 s gaps: %.3f" % ((dtw > 60).sum() / logged_h, dtw[dtw > 60].sum() / dtw.sum()))
print("gap-length quantiles for dt>60 s (s) 25/50/75/90/99:", np.percentile(dtw[dtw > 60], [25, 50, 75, 90, 99]).round(0))
spd = np.where(dtw > 0, dd_ / np.maximum(dtw, 1e-9), np.nan)
mc = MODECODE[:-1][SAMEFILE]; okm = (mc >= 0) & (dtw >= 1) & (dtw <= 60) & np.isfinite(spd) & (spd < 70)
rows = []
for k, m in enumerate(MODES):
    s_ = spd[okm & (mc == k)]
    if len(s_) > 100: rows.append({"mode": m, "steps": len(s_), "p10": round(np.percentile(s_, 10), 2), "p50": round(np.percentile(s_, 50), 2), "p90": round(np.percentile(s_, 90), 2), "p99": round(np.percentile(s_, 99), 2), "stationary<0.3m/s": round((s_ < .3).mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
allm = okm | ((mc < 0) & (dtw >= 1) & (dtw <= 60) & np.isfinite(spd) & (spd < 70))
print("unlabelled+labelled step speed quantiles 50/90/99 (m/s):", np.percentile(spd[allm], [50, 90, 99]).round(2))
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
ax[0].hist(np.clip(dtw, 0, 30), bins=np.arange(0, 31, 1), color=BLUE); ax[0].set(title="Sampling interval within files", xlabel="seconds (clipped 30)", ylabel="steps")
bins = np.linspace(0, 40, 80)
for k, c in zip((0, 1, 2, 3), CAT):
    s_ = spd[okm & (mc == k)]
    if len(s_) > 100: ax[1].hist(s_, bins=bins, histtype="step", color=c, density=True, label=MODES[k])
ax[1].set(title="Step speed by labelled mode", xlabel="m/s", ylabel="density"); ax[1].legend()
ax[2].hist(st.rms.clip(upper=100), bins=40, color=BLUE); ax[2].set(title="Jitter proxy: rms spread inside stops", xlabel="metres"); plt.tight_layout(); plt.show()""")

md("""## 6. Split-half reliability (label-free baseline)

Each qualifying user's real outings are split into two halves: by time (first half / second half of their
outings) and at random (seeded). Places = stop cells; segments proxy = traversed cells. Eligible users need
at least 2 outings with a stop in each half (place metrics) or any incidence in both halves (graph metrics).""")
code("""rngs = np.random.default_rng(SEED)
qo = og[og.qual & og.u.isin(QU)].copy()
qo["rank"] = qo.groupby("u").cumcount(); qo["n_u"] = qo.groupby("u").u.transform("size")
qo["half_time"] = (qo["rank"] >= qo.n_u / 2).astype(int)
qo["rk"] = qo.assign(r=rngs.random(len(qo))).groupby("u").r.rank(method="first")
qo["half_rand"] = (qo.rk > qo.n_u / 2).astype(int)
HALF = {"time": qo.half_time, "random": qo.half_rand}
def place_reliability(res, split):
    inc = INC[("stop", res, "all")]; inc = inc[inc.oid.isin(qo.index)].copy()
    inc["u"] = u_of_oid[inc.oid.values]; inc["h"] = HALF[split].reindex(inc.oid.values).values
    a, b = {}, {}
    for (u, h), s_ in inc.groupby(["u", "h"]):
        (a if h == 0 else b)[u] = s_
    users = [u for u in a if u in b and a[u].oid.nunique() >= 2 and b[u].oid.nunique() >= 2]
    jac, jt3, cov, base = [], [], [], []
    setA = {u: set(a[u].cell) for u in users}
    for k, u in enumerate(users):
        A, B = setA[u], set(b[u].cell)
        jac.append(E.jaccard(A, B))
        ta = set(a[u].cell.value_counts().index[:3]); tb = set(b[u].cell.value_counts().index[:3]); jt3.append(E.jaccard(ta, tb))
        cov.append(b[u].cell.isin(A).mean())
        other = setA[users[(k + 1) % len(users)]]; base.append(b[u].cell.isin(other).mean())
    return {"res": res, "split": split, "eligible_users": len(users), "jaccard_place_sets": round(np.median(jac), 3), "jaccard_top3": round(np.median(jt3), 3),
            "B_stops_in_A_set": round(np.mean(cov), 3), "same_for_other_user": round(np.mean(base), 3)}
print(pd.DataFrame([place_reliability(r, s) for r in (7, 8, 9) for s in ("time", "random")]).to_string(index=False))""")
code("""def graph_reliability(kind, res, split, scope="all"):
    inc = INC[(kind, res, scope)]; inc = inc[inc.oid.isin(qo.index)].copy()
    inc["u"] = u_of_oid[inc.oid.values]; inc["h"] = HALF[split].reindex(inc.oid.values).values
    both = set(inc[inc.h == 0].u) & set(inc[inc.h == 1].u)
    inc = inc[inc.u.isin(both)]
    gs = [E.covisit_graph(inc[inc.h == h].u.values, inc[inc.h == h].cell.values, max_pop=10, min_shared=2) for h in (0, 1)]
    ea, eb = (set(zip(g.a, g.b)) for g in gs)
    n = len(both); pairs = n * (n - 1) / 2
    inter = len(ea & eb)
    return {"kind": kind, "res": res, "split": split, "users": n, "edges_A": len(ea), "edges_B": len(eb), "edge_jaccard": round(inter / max(len(ea | eb), 1), 3),
            "P(edge in B | edge in A)": round(inter / max(len(ea), 1), 3), "density_B": round(len(eb) / pairs, 3), "lift": round((inter / max(len(ea), 1)) / max(len(eb) / pairs, 1e-9), 1)}
print(pd.DataFrame([graph_reliability(k, r, s) for k, r in (("stop", 7), ("stop", 8), ("traj", 8), ("traj", 9)) for s in ("time", "random")]).to_string(index=False))""")

md("""## 7. Caps: how many outings / places / segments per user""")
code("""q2 = og[og.qual & og.u.isin(QU)]
pu = pd.DataFrame({"real_outings": q2.groupby("u").size(), "active_days": uq.loc[QU, "active_days"]})
for r in (7, 8, 9): pu[f"stop_cells_r{r}"] = UC[("stop", r, "all")].groupby("u").cell.nunique()
for r in (7, 8, 9): pu[f"traj_cells_r{r}"] = UC[("traj", r, "all")].groupby("u").cell.nunique()
pu = pu.fillna(0)
print(pu.describe(percentiles=[.5, .75, .9, .95, .99]).round(0).T[["50%", "75%", "90%", "95%", "99%", "max"]])
def kept(vals, cap): return float(np.minimum(vals, cap).sum() / vals.sum())
rows = []
for col, caps in (("real_outings", (20, 50, 100, 200)), ("stop_cells_r8", (10, 20, 50, 100)), ("stop_cells_r9", (10, 20, 50, 100)), ("traj_cells_r7", (100, 200, 500, 1000)), ("traj_cells_r8", (200, 500, 1000, 2000)), ("traj_cells_r9", (200, 500, 1000, 2000))):
    for c in caps: rows.append({"quantity": col, "cap": c, "share_users_capped": round(float((pu[col] > c).mean()), 3), "share_incidences_kept": round(kept(pu[col].values, c), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("stops per outing: p50/p90/p99 =", np.percentile(cnt, [50, 90, 99]))""")

md(r"""## 8. Recommended public parameters

"Data-informed" = the value was chosen after looking at real GeoLife numbers in this notebook, so it counts as tuning for the ledger caveat (public-parameter selection on the sensitive data, not accounted in epsilon).

| Parameter | Recommendation | One-line justification | Data-informed? |
|---|---|---|---|
| Local time | UTC+8; artifact must state its hour convention | GeoLife is GMT; commute-like peaks only appear at 08h and 17-18h local | No (fact) |
| Outing gap | **2 h** (7200 s), across file boundaries | 84% of consecutive files are >2 h apart, but 0.22 in-file gaps/file exceed 2 h (signal loss indoors); 2 h keeps 19.7k outings, med 45 min / 9 km | Yes |
| Real outing | duration >= 5 min and length >= 500 m | drops 5% of fragments; no effect on the heavy tail | Yes (mild) |
| Min activity | **>= 10 real outings on >= 5 local days** | keeps 139 of 182 users and 99.0% of real outings; users below that cannot be split into halves | Yes |
| Stop rule | **200 m radius, 10 min dwell** | 1.38 stops per real outing, 49% of outings have a stop, median dwell 22 min; 100 m/5 min is dominated by jitter and short pauses | Yes |
| Start/end visits | add outing start and end as visits | dwell rule misses destinations where logging ends | No |
| Time bins | place/category x time: **4 time-of-day x weekday/weekend (8 bins)**; global outing-start and stop tables may use 24 h x 2 | 8 bins keep about 3.1k stops/bin overall and 28% of user-keys in cell-bins with >=10 users (24 h x 2: 17%) | Yes |
| Place hierarchy | H3 **6 -> 7 -> 8** for stop places; stop at 8 | cells with >=10 users saturate at ~100 (res 7: 81, 8: 100, 9: 102) while the incidence share in them falls 41% -> 26% -> 15% | Yes |
| Segment hierarchy | H3 **6 -> 7 -> 8**, optional refine to 9 / matched road segments under busy parents | traversed cells: share of incidences in >=10-user cells is flat across res 6-9 (32% -> 30%), so resolution is cheap; res 8 has 3.4k such cells | Yes |
| Cap: outings per user | **100** | p50 45, p90 280; cap touches 32% of users, keeps 43% of outings | Yes |
| Cap: distinct places per user | **50 stop cells at res 8** | p50 25, p90 127; caps 26% of users, keeps 53% of place incidences | Yes |
| Cap: distinct segments per user | **1000 traversed res-8 cells** (re-derive on matched segments in T2) | p50 362, p90 3981; caps 30% of users, keeps 33% of incidences; heavy-user tail is extreme | Yes |
| Co-visit edge | share >= 2 rare cells (2..10 users in scope), IDF weights; stops at res 8 | 84% of users get a partner, split-half edge lift ~7x; the traversal graph at res 9 links 35% of all pairs and is too dense to discriminate | Yes |
| Scope | keep all of Beijing; treat the core box (116.2-116.6E, 39.8-40.1N) as dense regime | outside the core overlap is far sparser | Yes |
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "geolife/inside/notebooks/02_outings_eda.ipynb")

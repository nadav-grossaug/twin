"""Real-vs-synthetic panels for GeoLife traces (eval zone; needs real and synthetic side by side).

Modelled on src/graph_aim/eval/viz.py: every function takes ``real`` (a `compare.TraceBundle`) and
``syns`` (``{label: TraceBundle}``) and returns a matplotlib Figure. The real data is always ink;
synthetic datasets take the categorical colour slots in order.

Frames: when both datasets sit in the same lon/lat frame (generation into the source city, Beijing)
the spatial panels overlay lon/lat directly; otherwise each dataset is drawn in a centred metric
frame (km from its own centroid) at the same scale, so shapes stay comparable across teleported
cities (`PLAN.md` section 7 item 4).

Panels: `density_maps`, `marginal_histograms`, `example_trajectories` (space); `ecdf_panels`,
`qq_panels` (moments/shape, over `compare.QUANTITIES`); `hour_profiles`, `timebin_shares` (time);
`structure_panels`, `split_half_structure` (social/place structure). `save_all` writes every panel
as PNG plus the summary distance table.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from geolife.eval import compare as C
from geolife.eval.compare import QUANTITIES, Quantity, TraceBundle, category_shares, distances

REAL_COLOUR = "#0b0b0b"
METHOD_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
SECONDARY_INK = "#52514e"
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK, "text.color": REAL_COLOUR,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
}
SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#eef4fb", "#2a78d6", "#123a6e"])
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list("blue_orange", ["#2a78d6", "#f1f0ec", "#eb6834"])

Bundles = Mapping[str, TraceBundle]


def _series(real: TraceBundle, syns: Bundles):
    yield "real", real, REAL_COLOUR, 2.2
    for i, (name, b) in enumerate(syns.items()):
        yield name, b, METHOD_COLOURS[i % len(METHOD_COLOURS)], 1.7


def _legend(fig, real: TraceBundle, syns: Bundles, y: float = -0.01) -> None:
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=lw, label=n) for n, _, c, lw in _series(real, syns)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y))


def _first(syns: Bundles) -> TraceBundle:
    return next(iter(syns.values()))


# --------------------------------------------------------------------------------------- frames

def shared_frame(real: TraceBundle, syn: TraceBundle, tol_deg: float = 0.5) -> bool:
    """True when both datasets sit in the same city (centroids closer than `tol_deg`)."""
    (la, lo), (lb, lob) = real.centroid(), syn.centroid()
    return bool(np.isfinite([la, lb]).all() and abs(la - lb) < tol_deg and abs(lo - lob) < tol_deg)


def points_xy(b: TraceBundle, lonlat: bool, max_points: int = 400_000, seed: int = 0):
    """All points of a bundle: (lon, lat) when `lonlat`, else km from the bundle's own centroid."""
    los = [a[2] for a in b.arrays.values() if len(a[0])]
    las = [a[1] for a in b.arrays.values() if len(a[0])]
    lon, lat = np.concatenate(los) if los else np.array([]), np.concatenate(las) if las else np.array([])
    if len(lon) > max_points:
        idx = np.random.default_rng(seed).choice(len(lon), size=max_points, replace=False)
        lon, lat = lon[idx], lat[idx]
    if lonlat:
        return lon, lat
    la0, lo0 = b.centroid()
    kx = 111.32 * math.cos(math.radians(la0))
    return (lon - lo0) * kx, (lat - la0) * 111.32


# --------------------------------------------------------------------------------------- space

def density_maps(real: TraceBundle, syns: Bundles, bins: int = 160, q: float = 0.005):
    """Point density of real and the first synthetic dataset side by side, plus the difference of the
    two normalised densities. Same frame when both are in the same city, else centred metric frames."""
    syn = _first(syns)
    lonlat = shared_frame(real, syn)
    xr, yr = points_xy(real, lonlat)
    xs, ys = points_xy(syn, lonlat)
    if lonlat:
        ext = [np.quantile(xr, [q, 1 - q]), np.quantile(yr, [q, 1 - q])]
    else:
        half = float(max(np.quantile(np.abs(np.r_[xr, yr]), 1 - q), np.quantile(np.abs(np.r_[xs, ys]), 1 - q)))
        ext = [(-half, half), (-half, half)]
    hr, ex, ey = np.histogram2d(xr, yr, bins=bins, range=ext)
    hs, _, _ = np.histogram2d(xs, ys, bins=bins, range=ext)
    pr, ps = hr / max(hr.sum(), 1), hs / max(hs.sum(), 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        extent = (ex[0], ex[-1], ey[0], ey[-1])
        vmax = np.log1p(max(hr.max(), hs.max()))
        for ax, h, title in ((axes[0], hr, f"real ({len(xr):,} pts)"), (axes[1], hs, f"{syn.label} ({len(xs):,} pts)")):
            ax.imshow(np.log1p(h).T, origin="lower", extent=extent, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.grid(False)
        d = (ps - pr).T
        lim = float(np.abs(d).max()) or 1e-9
        im = axes[2].imshow(d, origin="lower", extent=extent, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        axes[2].set_title("synthetic minus real (share per cell)")
        axes[2].grid(False)
        fig.colorbar(im, ax=axes[2], shrink=0.85)
        unit = ("lon", "lat") if lonlat else ("km east of centroid", "km north of centroid")
        for ax in axes:
            ax.set(xlabel=unit[0])
        axes[0].set_ylabel(unit[1])
        fig.tight_layout()
    return fig


def marginal_histograms(real: TraceBundle, syns: Bundles, bins: int = 80):
    """Marginal lon and lat (or centred-km) histograms, KS per synthetic dataset in the legend."""
    lonlat = all(shared_frame(real, b) for b in syns.values())
    data = [(n, points_xy(b, lonlat), c, lw) for n, b, c, lw in _series(real, syns)]
    xr, yr = data[0][1]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for k, (ax, ref, lab) in enumerate(((axes[0], xr, "lon" if lonlat else "km east"),
                                            (axes[1], yr, "lat" if lonlat else "km north"))):
            lo, hi = np.quantile(ref, [0.002, 0.998])
            for n, (x, y), c, lw in data:
                v = (x, y)[k]
                ks = distances(ref, v)["ks"] if n != "real" else None
                ax.hist(v, bins=bins, range=(lo, hi), density=True, histtype="step", color=c, lw=lw,
                        label=n if ks is None else f"{n} (KS {ks:.3f})")
            ax.set(xlabel=lab, ylabel="density", title=f"Marginal {lab}")
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


def example_trajectories(real: TraceBundle, syns: Bundles, n: int = 4, window_km: float = 8.0, seed: int = 0):
    """A few users' raw traces, real on the top row, synthetic below; each centred on the user's
    median position, all frames `window_km` half-width so shapes are comparable."""
    syn = _first(syns)
    rows = [("real", real, REAL_COLOUR), (syn.label, syn, METHOD_COLOURS[0])]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.0), squeeze=False)
        for (name, b, colour), axrow in zip(rows, axes):
            uids = [u for u in sorted(b.arrays) if len(b.arrays[u][0]) > 50]
            sizes = np.array([len(b.arrays[u][0]) for u in uids])
            order = np.argsort(np.abs(sizes - np.median(sizes)))
            pick = [uids[i] for i in order[:n]]
            for ax, uid in zip(axrow, pick):
                t, lat, lon, seg = b.arrays[uid]
                la0, lo0 = float(np.median(lat)), float(np.median(lon))
                kx = 111.32 * math.cos(math.radians(la0))
                x, y = (lon - lo0) * kx, (lat - la0) * 111.32
                for s in np.unique(seg):
                    m = seg == s
                    ax.plot(x[m], y[m], color=colour, lw=0.6, alpha=0.7)
                ax.set(xlim=(-window_km, window_km), ylim=(-window_km, window_km))
                ax.set_title(f"{name}: {len(t):,} pts", fontsize=8)
                ax.set_aspect("equal")
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axrow[len(pick):]:
                ax.axis("off")
        fig.suptitle(f"Example user trajectories ({window_km:g} km half-width)", x=0.01, ha="left", fontsize=10,
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------------------- shape

def _numeric(quantities: Sequence[Quantity]) -> list[Quantity]:
    return [q for q in quantities if q.kind == "numeric"]


def ecdf_panels(real: TraceBundle, syns: Bundles, quantities: Sequence[Quantity] | None = None, ncols: int = 4):
    """ECDF overlays for every numeric quantity (log x where the quantity is heavy-tailed)."""
    qs = _numeric(quantities or QUANTITIES)
    nrows = int(np.ceil(len(qs) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows + 0.5), squeeze=False)
        for ax, q in zip(axes.ravel(), qs):
            for n, b, c, lw in _series(real, syns):
                x = np.sort(np.asarray(q.extract(b), float))
                if q.log:
                    x = x[x > 0]
                if not len(x):
                    continue
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=lw)
            if q.log:
                ax.set_xscale("log")
            ax.set_title(f"{q.name}" + (f" ({q.unit})" if q.unit else ""), fontsize=8)
        for ax in axes.ravel()[len(qs):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("ECDF")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def qq_panels(real: TraceBundle, syns: Bundles, quantities: Sequence[Quantity] | None = None, ncols: int = 4):
    """QQ plots (synthetic quantiles against real quantiles, p1..p99) for every numeric quantity."""
    qs = _numeric(quantities or QUANTITIES)
    syn_series = [s for s in _series(real, syns)][1:]
    probs = np.linspace(1, 99, 50)
    nrows = int(np.ceil(len(qs) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.8 * nrows + 0.5), squeeze=False)
        for ax, q in zip(axes.ravel(), qs):
            xr = np.asarray(q.extract(real), float)
            if not len(xr):
                ax.set_title(f"{q.name} (no real data)", fontsize=8)
                continue
            qr = np.percentile(xr, probs)
            for n, b, c, lw in syn_series:
                xs = np.asarray(q.extract(b), float)
                if not len(xs):
                    continue
                ax.plot(qr, np.percentile(xs, probs), color=c, lw=0, marker="o", ms=2.4)
            lim = (min(qr.min(), 0), qr.max() * 1.05 or 1)
            ax.plot(lim, lim, color=GRID, lw=1)
            if q.log and qr.min() > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(f"{q.name}" + (f" ({q.unit})" if q.unit else ""), fontsize=8)
        for ax in axes.ravel()[len(qs):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("synthetic quantile")
        axes[-1, 0].set_xlabel("real quantile")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------------------- time

def _hour_share(b: TraceBundle, q: Quantity) -> np.ndarray:
    h = np.asarray(q.extract(b))
    s = np.zeros(24)
    if len(h):
        np.add.at(s, h.astype(int), 1.0)
        s /= s.sum()
    return s


def hour_profiles(real: TraceBundle, syns: Bundles):
    """Hour-of-day profiles of outing starts and stops, weekday and weekend (local time, UTC+8)."""
    panels = [("outing starts, weekday", C.by_name("outing_start_hour_weekday")),
              ("outing starts, weekend", C.by_name("outing_start_hour_weekend")),
              ("stops, weekday", C.by_name("stop_hour_weekday")),
              ("stops, weekend", C.by_name("stop_hour_weekend"))]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, (title, q) in zip(axes.ravel(), panels):
            for n, b, c, lw in _series(real, syns):
                ax.plot(np.arange(24), _hour_share(b, q), color=c, lw=lw, marker="o", ms=3, mec=SURFACE, mew=0.8)
            ax.set(title=title, xlabel="local hour", ylabel="share", xticks=range(0, 24, 4))
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def timebin_shares(real: TraceBundle, syns: Bundles):
    """Share of outing starts per (daytype x time-of-day) bin, real bar next to each synthetic bar."""
    series = list(_series(real, syns))
    shares = [category_shares(np.asarray(C.by_name("timebin_share").extract(b))) for _, b, _, _ in series]
    keys = sorted(set().union(*[set(s) for s in shares]))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.1 * max(len(keys), 4) + 2, 3.6))
        w = 0.8 / len(series)
        for j, ((n, _, c, _), s) in enumerate(zip(series, shares)):
            ax.bar(np.arange(len(keys)) + (j - (len(series) - 1) / 2) * w,
                   [s.get(k, 0.0) for k in keys], width=w * 0.92, color=c, label=n)
        ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="share of outing starts", title="Time-bin shares")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------- structure

def _ccdf(ax, x: np.ndarray, colour: str, lw: float, ls: str = "-") -> None:
    x = np.sort(np.asarray(x, float))
    if not len(x):
        return
    ax.plot(x, 1.0 - np.arange(len(x)) / len(x), color=colour, lw=lw, ls=ls, drawstyle="steps-post")


STRUCTURE = ("users_per_place", "users_per_cell", "revisit_rate", "covisit_degree")


def structure_panels(real: TraceBundle, syns: Bundles):
    """Users per place / per traversed cell (CCDF, log-log), revisit rate (ECDF), co-visit degree (CCDF)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
        for ax, name in zip(axes.ravel(), STRUCTURE):
            q = C.by_name(name)
            for n, b, c, lw in _series(real, syns):
                x = np.asarray(q.extract(b), float)
                if name == "revisit_rate":
                    x = np.sort(x)
                    if len(x):
                        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=lw)
                else:
                    _ccdf(ax, x[x > 0], c, lw)
            if name in ("users_per_place", "users_per_cell", "covisit_degree"):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set(xlabel=name.replace("_", " "), ylabel="share >= x (CCDF)")
            else:
                ax.set(xlabel="revisit rate per user", ylabel="ECDF")
            ax.set_title(name.replace("_", " "))
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def split_half_structure(real: TraceBundle, syns: Bundles, seed: int = 0):
    """The structure panels on split halves of each dataset (half A solid, half B dashed): a dataset
    whose halves disagree with each other is unstable before it is wrong."""
    halves = [(n, C.split_bundle(b, "random", seed), c) for n, b, c, _ in _series(real, syns)]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
        for ax, name in zip(axes.ravel(), STRUCTURE):
            q = C.by_name(name)
            for n, (ha, hb), c in halves:
                for h, ls in ((ha, "-"), (hb, "--")):
                    x = np.asarray(q.extract(h), float)
                    if name == "revisit_rate":
                        x = np.sort(x)
                        if len(x):
                            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.5, ls=ls)
                    else:
                        _ccdf(ax, x[x > 0], c, 1.5, ls)
            if name != "revisit_rate":
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(f"{name.replace('_', ' ')} (A solid, B dashed)")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------------------- driver

PANELS = ("density_maps", "marginal_histograms", "example_trajectories", "ecdf_panels", "qq_panels",
          "hour_profiles", "timebin_shares", "structure_panels", "split_half_structure")


def save_all(real: TraceBundle, syns: Bundles | TraceBundle, out_dir,
             quantities: Sequence[Quantity] | None = None, dpi: int = 130) -> dict[str, Path]:
    """Write every panel as `<out_dir>/<panel>.png` and the summary distance table as
    `<out_dir>/summary.{csv,md}` (real vs the FIRST synthetic bundle). Returns {name: path}."""
    if isinstance(syns, TraceBundle):
        syns = {syns.label: syns}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qs = list(quantities or QUANTITIES)
    written: dict[str, Path] = {}
    for name in PANELS:
        fn = globals()[name]
        fig = fn(real, syns, quantities=qs) if name in ("ecdf_panels", "qq_panels") else fn(real, syns)
        p = out / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written[name] = p
    df = C.summary_table(real, _first(syns), qs)
    pcsv, pmd = C.write_summary(df, out, "summary")
    written["summary_csv"], written["summary_md"] = pcsv, pmd
    return written

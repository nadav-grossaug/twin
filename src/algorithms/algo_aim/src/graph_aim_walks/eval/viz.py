"""Walk statistics of the real graph against synthetic ones, matched and held out (eval zone).

Same conventions as ``graph_aim.eval.viz``: ``(real, syns) -> Figure``, the real graph in ink, synthetic graphs in
the categorical slots, shared legend. The statistics are ``graph_aim_walks.eval.metrics.walk_profile`` (the
contract's walk process, one pool of 128-step walks per graph). The grey band marks what the generator never sees:
t > 32, lags > 16 and windows 64 / 128; hollow markers are held-out points, filled markers matched ones.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from graph_aim.eval.viz import SECONDARY_INK, STYLE, SURFACE, Graphs, _legend, _series
from graph_aim.transfer.types import AttributedGraph
from graph_aim_walks.eval.metrics import default_spec, variance_attrs, walk_profile
from graph_aim_walks.transfer.walkstats import WalkSpec

HELD_OUT_BAND = "#ecebe6"


def _held_out(ax, x0: float, x1: float) -> None:
    ax.axvspan(x0, x1, color=HELD_OUT_BAND, zorder=0, lw=0)
    ax.text(0.99, 0.97, "held out", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=SECONDARY_INK)


def _points(ax, xs, ys, held, col, lw) -> None:
    """A line through every point; matched points filled, held-out points hollow."""
    xs, ys, held = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(held, dtype=bool)
    ax.plot(xs, ys, color=col, lw=lw)
    ax.plot(xs[~held], ys[~held], "o", ms=4, color=col, mec=SURFACE, mew=0.8)
    ax.plot(xs[held], ys[held], "o", ms=5, mfc=SURFACE, mec=col, mew=1.4)


def walk_profiles(
    real: AttributedGraph, syns: Graphs, spec: WalkSpec | None = None, n_walks: int = 50000, seed: int = 0,
    profiles: dict | None = None,
):
    """Row 1: return probability and coverage against t, top-16 walk-matrix spectrum, share variance against the
    window (mean over the group attributes). Row 2: P(same group at t and t + k) against k, one panel per group.
    Row 3: share variance against the window, one panel per group attribute. ``profiles`` (``{label: profile}``,
    label ``"real"`` for the real graph) reuses profiles already computed."""
    spec = spec or default_spec(real)
    series = list(_series(real, syns))
    profiles = dict(profiles or {})
    for name, g, _, _ in series:
        if name not in profiles:
            profiles[name] = walk_profile(g, spec, n_walks, seed)
    p0 = profiles["real"]
    T, L = spec.length, p0["length"]
    groups = list(p0["same"])[:4]
    attrs = variance_attrs(spec)[:4]
    windows = [*p0["windows"], *p0["held_out_windows"]]
    held_w = [w > T for w in windows]
    lags = [*p0["lags"], *p0["held_out_lags"]]
    held_l = [k > T or k in p0["held_out_lags"] for k in lags]
    ncols = max(4, len(groups), len(attrs))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 9.6), squeeze=False)
        a_ret, a_cov, a_spec, a_var = axes[0][:4]
        t = np.arange(1, L + 1)
        for name, _, col, lw in series:
            p = profiles[name]
            a_ret.plot(t, np.where(p["return"] > 0, p["return"], np.nan), color=col, lw=lw)  # 0 at t = 1: off a log axis
            a_cov.plot(t, p["coverage"], color=col, lw=lw)
            a_spec.plot(np.arange(1, len(p["spectrum"]) + 1), p["spectrum"], color=col, lw=lw, marker="o", ms=3.5,
                        mec=SURFACE, mew=0.8)
            mean_var = [np.mean([p["share_var"][a][w] for a in attrs]) for w in windows]
            _points(a_var, windows, mean_var, held_w, col, lw)
            for ax, lab in zip(axes[1], groups):
                _points(ax, lags, [p["same"][lab][k] for k in lags], held_l, col, lw)
            for ax, a in zip(axes[2], attrs):
                _points(ax, windows, [p["share_var"][a][w] for w in windows], held_w, col, lw)
        for ax in (a_ret, a_cov):
            if L > T:
                _held_out(ax, T + 0.5, L + 0.5)
            ax.set_xlim(0.5, L + 0.5)
        a_ret.set(yscale="log", xlabel="steps t", ylabel="P(back at the start)", title="Return probability")
        a_cov.set(xlabel="steps t", ylabel="distinct nodes visited", title="Coverage")
        a_spec.set(xlabel="rank", ylabel="eigenvalue of D$^{-1}$A", title="Walk-matrix spectrum (top 16)")
        a_spec.set_ylim(top=1.01)
        any_held_w = any(held_w) and not all(held_w)
        if any_held_w:
            wlo = (max(w for w, h in zip(windows, held_w) if not h) * min(w for w, h in zip(windows, held_w) if h)) ** 0.5
        for ax in [a_var, *axes[2][: len(attrs)]]:
            if any_held_w:
                _held_out(ax, wlo, windows[-1] * 1.3)
            ax.set(xscale="log", xlabel="window (steps)")
            ax.set_xticks(windows, [str(w) for w in windows])
            ax.minorticks_off()
        a_var.set(yscale="log", ylabel="across-walk variance of a value's share",
                  title="Share variance (" + ", ".join(attrs) + ")")
        for ax, a in zip(axes[2], attrs):
            ax.set(yscale="log", ylabel="share variance", title=f"Share variance: {a}")
        llo = (max(p0["lags"]) * min(p0["held_out_lags"])) ** 0.5 if p0["held_out_lags"] else None
        for ax, lab in zip(axes[1], groups):
            if llo is not None:
                _held_out(ax, llo, lags[-1] * 1.3)
            ax.set(xscale="log", xlabel="lag k (steps)", ylabel="P(same group at t and t + k)",
                   title=f"Attribute agreement: {lab}")
            ax.set_xticks(lags, [str(k) for k in lags], fontsize=7)
            ax.minorticks_off()
        for row, used in ((axes[0], 4), (axes[1], len(groups)), (axes[2], len(attrs))):
            for ax in row[used:]:
                ax.axis("off")
        _legend(fig, real, syns)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig

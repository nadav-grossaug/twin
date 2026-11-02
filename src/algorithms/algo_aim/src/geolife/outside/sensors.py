"""Sensor fit (outside): noisy sensor_* tables -> render parameters + inverse-CDF samplers.

Pure post-processing of Measurement.values (noisy) over public keys; costs no privacy. A table
counts as "mostly noise" when its released total mass is below SNR_MIN standard deviations of the
noise on that total (same rule as fit_v2); every draw that depends on such a table falls back to
the hand-set defaults of render_v2.SENSOR_DEFAULTS and is listed in the returned "fallbacks".

The fitted dict (stored under params["sensor"]) is JSON-safe: open-ended top bins are truncated at
TOP_MULT x the last finite edge.
"""
from __future__ import annotations

import math

import numpy as np

from geolife.transfer import schema_v2 as S2
from geolife.transfer.types import Measurement

SNR_MIN = 3.0
MODE_K_SIGMA = 3.0     # per-mode mass must exceed this many sd of its own noise to be fitted
TOP_MULT = 2.0         # open top bin [lo, inf) is sampled as [lo, TOP_MULT * lo)


def _finite_edges(edges: list[float]) -> list[float]:
    e = [float(x) for x in edges]
    if math.isinf(e[-1]):
        e[-1] = TOP_MULT * e[-2] if e[-2] > 0 else 1.0
    return e


def _signal(m: Measurement | None) -> float:
    if m is None or not m.keys or m.sigma <= 0:
        return 0.0
    return max(0.0, float(sum(m.values))) / (m.sigma * math.sqrt(len(m.keys)))


def _denoise(values: np.ndarray, sigma: float) -> np.ndarray:
    """Clamp negatives, zero cells below 1 sigma, normalise (same rule as fit_v2's _pmf)."""
    v = np.clip(np.asarray(values, float), 0.0, None)
    v[v < sigma] = 0.0
    s = v.sum()
    return v / s if s > 0 else np.full(len(v), 1.0 / len(v))


class HistSampler:
    """Inverse-CDF sampler of a binned distribution (uniform within each bin)."""

    def __init__(self, edges: list[float], pmf: list[float]):
        self.edges = np.asarray(_finite_edges(list(edges)), float)
        p = np.clip(np.asarray(pmf, float), 0.0, None)
        if len(p) != len(self.edges) - 1:
            raise ValueError("pmf length must be len(edges) - 1")
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        self.pmf = p
        self.cdf = np.concatenate([[0.0], np.cumsum(p)])
        self.cdf[-1] = 1.0

    def draw(self, rng: np.random.Generator, n: int | None = None):
        u = rng.random(1 if n is None else n)
        x = np.interp(u, self.cdf, self.edges)
        return float(x[0]) if n is None else x

    def median(self) -> float:
        return float(np.interp(0.5, self.cdf, self.edges))

    def quantile(self, q: float) -> float:
        return float(np.interp(q, self.cdf, self.edges))

    def mean(self) -> float:
        mids = 0.5 * (self.edges[:-1] + self.edges[1:])
        return float((self.pmf * mids).sum())


def _hist_dict(m: Measurement, edges: list[float]) -> dict:
    rows = sorted(((float(k[0]), v) for k, v in zip(m.keys, m.values)))
    p = _denoise(np.array([v for _, v in rows]), m.sigma)
    return {"edges": _finite_edges(list(edges)), "pmf": [float(x) for x in p]}


def fit_sensors(measurements: dict[str, Measurement], snr_min: float = SNR_MIN) -> dict:
    """{name: Measurement} -> params['sensor'] dict (see render_v2 for how each entry is used)."""
    ms = {n: measurements.get(n) for n in S2.SENSOR_NAMES}
    snr = {n: round(_signal(m), 2) for n, m in ms.items() if m is not None}
    fb: dict[str, str] = {}
    hists: dict[str, dict] = {}

    def usable(name: str) -> Measurement | None:
        m = ms.get(name)
        if m is None:
            fb[name] = "table missing"
            return None
        if snr[name] < snr_min:
            fb[name] = f"mostly noise (signal {snr[name]:.1f} < {snr_min})"
            return None
        return m

    simple = {S2.SENSOR_INTERVAL: "interval", S2.SENSOR_INTERVAL_USER: "interval_user",
              S2.SENSOR_GAP_LEN: "gap_len", S2.SENSOR_GAP_RATE: "gap_rate",
              S2.SENSOR_HEADING: "heading", S2.SENSOR_STOP_JITTER: "stop_jitter",
              S2.SENSOR_PTS_OUTING: "points_per_outing"}
    for name, key in simple.items():
        m = usable(name)
        if m is not None:
            hists[key] = _hist_dict(m, S2.SENSOR_HIST_EDGES[name])

    # ---- speed and step length by mode (conditional pmf per mode above the noise floor)
    speed_by_mode: dict[str, dict] = {}
    step_by_mode: dict[str, dict] = {}
    speed_ms: dict[str, float] = {}
    for name, store in ((S2.SENSOR_SPEED, speed_by_mode), (S2.SENSOR_STEP, step_by_mode)):
        m = usable(name)
        if m is None:
            continue
        edges = _finite_edges(list(S2.SENSOR_HIST_EDGES[name]))
        nb = len(edges) - 1
        for mode in S2.MODES:
            rows = sorted(((float(k[1]), v) for k, v in zip(m.keys, m.values) if k[0] == mode))
            if len(rows) != nb:
                continue
            vals = np.array([v for _, v in rows])
            if max(0.0, float(vals.sum())) < MODE_K_SIGMA * m.sigma * math.sqrt(nb):
                fb[f"{name}:{mode}"] = "per-mode mass below the noise floor"
                continue
            p = _denoise(vals, m.sigma)
            store[mode] = {"edges": edges, "pmf": [float(x) for x in p]}
            if name == S2.SENSOR_SPEED and mode != "unknown":
                s = HistSampler(edges, p)
                # typical cruising speed: the upper-middle of the moving distribution (steps
                # include stop-and-go near 0, so the median underestimates the leg speed)
                speed_ms[mode] = round(max(s.quantile(0.75), 0.2), 2)
    if speed_by_mode:
        hists["speed_by_mode"] = speed_by_mode
    if step_by_mode:
        hists["step_by_mode"] = step_by_mode

    # sensor noise estimate from the lower tail of the stop-jitter distribution: assume the
    # quiet quarter of in-stop points is pure GPS noise (2-d Gaussian => radius ~ Rayleigh(sigma),
    # p25 = sigma * 0.7585); the walking-around tail inflates only the upper quantiles.
    noise_m = None
    if "stop_jitter" in hists:
        s = HistSampler(hists["stop_jitter"]["edges"], hists["stop_jitter"]["pmf"])
        noise_m = round(min(20.0, max(1.0, s.quantile(0.25) / 0.7585)), 2)

    out = {"available": bool(hists), "hists": hists, "signal": snr, "fallbacks": fb}
    if speed_ms:
        out["speed_ms_by_mode"] = speed_ms
    if noise_m is not None:
        out["sensor_noise_m"] = noise_m
    if "gap_rate" in hists:
        out["gap_rate_per_hr_mean"] = round(HistSampler(
            hists["gap_rate"]["edges"], hists["gap_rate"]["pmf"]).mean(), 3)
    return out

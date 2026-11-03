"""Fake `params` (docs/geolife/OUTINGS_PARAMS.md) with realistic values from geolife/inside/reports/outings-eda.md.

No real data: the numbers are the public EDA summaries (outings per user lognormal median 30, p90 ~250;
1.38 stops per outing; stop revisit ~0.55; outing starts peak 8h and 17-18h local; stops skew weekend).
Interpretation used by the generator: lognormal {mu, sigma} are the mean and sd of ln(value) with value in
the stated unit (days, minutes, km); start_pmf / cat_profile are JOINT pmfs over (daytype, tod bin).
"""
from __future__ import annotations

import numpy as np

from geolife.transfer.types import DAYTYPES, TOD_BINS


def _joint(wd, we, we_share):
    wd, we = np.array(wd, float), np.array(we, float)
    a = np.concatenate([(1 - we_share) * wd / wd.sum(), we_share * we / we.sum()])
    return {dt: {b: float(a[i * 4 + j]) for j, b in enumerate(TOD_BINS)} for i, dt in enumerate(DAYTYPES)}


def fake_params_v2() -> dict:
    # outing starts: tod 9/36/28/28 %, 26% weekend (EDA section 3)
    start = _joint([.08, .38, .27, .27], [.10, .30, .34, .26], 0.26)
    # category visit-time profiles (night, morning, afternoon, evening) for weekday / weekend, joint
    cp = {
        "office":      ([.03, .42, .45, .10], [.05, .25, .45, .25], .12),
        "school":      ([.01, .50, .42, .07], [.05, .30, .45, .20], .08),
        "food":        ([.04, .22, .36, .38], [.05, .25, .30, .40], .38),
        "shop":        ([.02, .22, .50, .26], [.03, .20, .52, .25], .40),
        "leisure":     ([.04, .18, .36, .42], [.05, .25, .40, .30], .50),
        "health":      ([.01, .55, .38, .06], [.02, .45, .43, .10], .10),
        "residential": ([.10, .15, .25, .50], [.12, .20, .28, .40], .35),
    }
    return {
        "version": 2,
        "places": {
            "zipf_s": 1.0, "pool_size": 900,
            "cat_mix": {"office": .24, "shop": .22, "food": .20, "leisure": .12, "school": .10,
                        "health": .05, "residential": .07},
            # distinct users per place (stops): 41% of incidences in >=10-user res-6/7 cells, median 1 user
            "users_per_place_hist": {"1": 620, "2": 150, "3": 70, "4": 40, "5": 25, "7": 25, "10": 18,
                                     "20": 8, "40": 3},
        },
        "users": {
            "outings_per_user": {"mu": float(np.log(30.0)), "sigma": 1.55},   # median 30, p90 ~220
            "active_days": {"mu": float(np.log(72.0)), "sigma": 1.2},         # window length (days)
            "distinct_places": {"mu": float(np.log(25.0)), "sigma": 0.9},
            "revisit_rate": 0.55,
        },
        "groups": {"n_groups": 12, "concentration": 0.35},
        "outing": {
            "stops_pmf": {"0": .50, "1": .22, "2": .10, "3": .06, "4": .04, "5+": .08},   # mean ~1.35
            "duration_min": {"mu": float(np.log(45.0)), "sigma": 1.2},
            "length_km": {"mu": float(np.log(9.0)), "sigma": 1.2},
            "mode_pmf": {"walk": .27, "bike": .22, "bus": .20, "car": .21, "train": .10},
            "stop_dwell_min": {"mu": float(np.log(23.0)), "sigma": 0.9},
            "distance_decay": 1.0,
            "start_pmf": start,
        },
        "time": {"cat_profile": {c: _joint(wd, we, s) for c, (wd, we, s) in cp.items()}},
        "route": {"detour_ratio": {"mu": float(np.log(0.12)), "sigma": 0.8}},   # ratio = 1 + LN(mu, sigma)
    }


def fake_generation_v2() -> dict:
    """Sensor regime from EDA section 5 (what the artifact `generation` dict would carry)."""
    return {"sample_interval_s": None, "gap_rate_per_hr": 5.0, "gap_mean_s": None,
            "sensor_noise_m": 6.0, "dwell_jitter_m": 25.0,
            "speed_ms_by_mode": {"walk": 1.3, "bike": 3.5, "bus": 5.0, "car": 9.0, "train": 19.0}}

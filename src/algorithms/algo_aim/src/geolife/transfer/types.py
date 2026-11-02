"""Shared contract for the geolife package (orchestrator-owned; agents must not change it).

Everything here is plain dataclasses + stdlib so modules can be built independently.
Time is seconds since epoch (UTC int); coordinates are WGS-84 degrees (lat, lon).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- L1/L2: per-user, map-relative tokens (never leave the air-gap un-aggregated) ---------


@dataclass(frozen=True)
class GpsPoint:
    t: int
    lat: float
    lon: float


@dataclass(frozen=True)
class Stay:
    """A stay point snapped to the public map."""
    place_id: str          # public-map id (e.g. Overture GERS id) or "h3:<cell>" fallback
    category: str          # public place category / land-use class ("unknown" allowed)
    cell: str              # H3 cell id at the working resolution
    t_start: int
    t_end: int


@dataclass(frozen=True)
class Trip:
    origin: str            # place_id of the stay before the trip
    dest: str              # place_id of the stay after
    t_start: int
    t_end: int
    mode: str              # "walk" | "bike" | "bus" | "car" | "train" | "unknown"
    segments: tuple[str, ...] = ()   # matched road segment ids, in order
    length_m: float = 0.0


@dataclass
class UserRecord:
    """One user's tokens. `user` is a random pseudonym, never the dataset's user id."""
    user: str
    stays: list[Stay] = field(default_factory=list)
    trips: list[Trip] = field(default_factory=list)


# --- L3: DP measurements (the only per-population quantities released) ---------------------


@dataclass
class Measurement:
    """A noisy, clipped, user-level count table over public-vocabulary keys."""
    name: str                          # [a-z0-9_]+ ; becomes a file name
    keys: list[tuple[str, ...]]        # vocabulary rows, public
    values: list[float]                # noisy counts, same length as keys
    sigma: float                       # Gaussian noise std actually added
    l2_sensitivity: float              # per-user clip bound used
    rho: float                         # zCDP spent = l2_sensitivity**2 / (2 sigma**2)
    notes: str = ""


@dataclass
class PrivacyLedger:
    adjacency: Literal["user"] = "user"
    rho_total: float = 0.0
    epsilon: float = 0.0
    delta: float = 1e-6
    caps: dict[str, float] = field(default_factory=dict)       # e.g. {"max_stays_per_user": 50}
    entries: list[dict] = field(default_factory=list)          # one per Measurement
    caveats: list[str] = field(default_factory=list)           # un-accounted steps, stated honestly


# --- Artifact (carried across the air-gap) -------------------------------------------------


@dataclass
class ArtifactManifest:
    format_version: str = "0.1"
    learner: str = "geolife-epr-v0"
    source_map_snapshot: str = ""      # e.g. "overture-2026-10-xx"
    h3_resolution: int = 8
    created: str = ""
    files: dict[str, str] = field(default_factory=dict)        # relative path -> sha256


# --- v2 (outings; see docs/geolife/PLAN_OUTINGS.md). Additive: v1 types above are unchanged. ---


@dataclass(frozen=True)
class Visit:
    """A stop (or outing start/end) at a place, in local time bins computed from UTC+8."""
    place_id: str
    category: str
    cell: str              # H3 cell at working resolution (res 8)
    t_arrive: int
    dwell_s: int
    kind: Literal["stop", "start", "end"] = "stop"


@dataclass(frozen=True)
class Outing:
    """One outing = one trajectory between gaps > gap threshold (default 2 h)."""
    t_start: int
    t_end: int
    mode: str                              # "walk" | "bike" | "bus" | "car" | "train" | "unknown"
    length_m: float
    visits: tuple[Visit, ...] = ()
    cells: tuple[str, ...] = ()            # distinct res-9 cells traversed (road proxy), in first-seen order
    segments: tuple[str, ...] = ()         # matched road segment ids, if map matching ran


@dataclass
class UserOutings:
    """One user's outings. `user` is a random pseudonym."""
    user: str
    outings: list[Outing] = field(default_factory=list)


# Time bins (local time = UTC+8): 4 time-of-day bins x weekday/weekend.
TOD_BINS = ("night", "morning", "afternoon", "evening")   # [0,6), [6,12), [12,18), [18,24)
DAYTYPES = ("weekday", "weekend")
LOCAL_UTC_OFFSET_H = 8


# Coarse time design (v2b): 2 bins x weekday/weekend. am = local [05:00,14:00), pm = local [14:00,05:00).
# Chosen so the 08h and 17-18h outing peaks fall in different bins. The 4 TOD_BINS cannot express the
# 05:00 / 14:00 edges, so the exact coarse bin comes from the local hour (`local_coarse_bin`);
# `tod_coarse` is only the majority-hour approximation for an already-binned value.
TOD_COARSE = ("am", "pm")
COARSE_AM_START_H, COARSE_PM_START_H = 5, 14


def local_coarse_bin(t_utc: int) -> str:
    """Exact coarse bin of a UTC timestamp (local = UTC+8)."""
    h = ((int(t_utc) + LOCAL_UTC_OFFSET_H * 3600) // 3600) % 24
    return TOD_COARSE[0] if COARSE_AM_START_H <= h < COARSE_PM_START_H else TOD_COARSE[1]


def tod_coarse(tod_bin: str) -> str:
    """Approximate coarse bin of a 4-bin value (majority of its hours): night->pm (5 of 6 h),
    morning->am, afternoon->pm (4 of 6 h), evening->pm. Use local_coarse_bin when the time is known."""
    return TOD_COARSE[0] if tod_bin == "morning" else TOD_COARSE[1]

"""GPS trajectories -> list[UserRecord] (stays + trips) with random pseudonyms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from geolife.transfer.geo import LabelSpan, clean_trajectory, pseudonyms
from geolife.transfer.types import GpsPoint, UserRecord
from geolife.transfer.tokenize.stays import PlaceTable, detect_stays, merge_adjacent, snap_stay, thin
from geolife.transfer.tokenize.trips import segment_trips

# Core Beijing box: lat_min, lat_max, lon_min, lon_max (88.7% of in-Beijing points, see EDA)
BEIJING_CORE = (39.75, 40.10, 116.15, 116.60)


@dataclass
class TokenizeConfig:
    stay_dist_m: float = 200.0
    stay_min_s: int = 1200
    thin_s: int = 10
    h3_res: int = 8
    snap_radius_m: float = 100.0
    merge_gap_s: int = 1800
    max_speed_ms: float = 70.0
    bbox: tuple[float, float, float, float] = BEIJING_CORE
    trip_min_len_m: float = 200.0
    trip_max_gap_s: int = 600
    max_gap_s: int = 1800     # a sampling gap this long ends a stay candidate


def tokenize_user(user: str, trajectories: Sequence[Sequence[GpsPoint]], places: PlaceTable,
                  cfg: TokenizeConfig = TokenizeConfig(), labels: Sequence[LabelSpan] = (),
                  matcher=None) -> tuple[UserRecord, dict]:
    """Returns (record, circuity) with circuity[(origin, dest, t_start)] = path/straight."""
    stays_all, trips_all, circ = [], [], {}
    for traj in trajectories:
        pts = thin(clean_trajectory(list(traj), cfg.bbox, cfg.max_speed_ms), cfg.thin_s)
        if len(pts) < 2:
            continue
        raws = detect_stays(pts, cfg.stay_dist_m, cfg.stay_min_s, cfg.max_gap_s)
        stays = [snap_stay(r, places, cfg.h3_res, cfg.snap_radius_m) for r in raws]
        stays_all.extend(stays)
        tr, c = segment_trips(pts, raws, stays, labels, cfg.trip_min_len_m, cfg.trip_max_gap_s,
                              matcher)
        trips_all.extend(tr)
        circ.update(c)
    trips_all.sort(key=lambda t: t.t_start)
    return UserRecord(user, merge_adjacent(stays_all, cfg.merge_gap_s), trips_all), circ


def tokenize_all(trajs: Mapping[str, Sequence[Sequence[GpsPoint]]], places: PlaceTable,
                 cfg: TokenizeConfig = TokenizeConfig(),
                 labels: Mapping[str, Sequence[LabelSpan]] | None = None, seed: int = 0,
                 matcher=None) -> tuple[list[UserRecord], dict]:
    """Tokenize every user; user ids are replaced by seeded random pseudonyms (the mapping is
    not retained). Users without any stay are dropped."""
    names = pseudonyms(list(trajs), seed)
    records, circ = [], {}
    for uid in sorted(trajs):
        rec, c = tokenize_user(names[uid], trajs[uid], places, cfg,
                               (labels or {}).get(uid, ()), matcher)
        circ.update(c)
        if rec.stays:
            records.append(rec)
    return records, circ

"""Measurement-name conventions shared by measure.py (writer side) and fit.py (reader side).

Every table is a user-level clipped count table over a PUBLIC vocabulary (see dp.py):
  n_users           ("users",)                          presence, 1 key per user
  role_users        (role,)                             users having that role
  anchors_other     (n,)                                users with n distinct 'other' anchors (n clipped)
  place_role_users  (place_id, category, role)          distinct users per place and role
  arrival_hour      (role, hour)                        stay starts per role and hour of day
  dwell_hist        (role, lo_s, hi_s)                  stays per role and dwell bin
  role_trips        (from_role, to_role)                trips between roles
  trip_dist         (kind, lo_m, hi_m)                  trips per kind (commute|other) and length bin
  detour            (lo, hi)                            route length / shortest-path length bins
  return_rate       ("new"|"return",)                   first visits vs repeat visits (stays)
  place_role_users_r6 / _r7   (node_id, "unknown", role)  coarse tiers (H3 res 6 / 7), released fully (r6)
                                                        or only under parents whose noisy count passed
  place_role_users  (place_id, category, role)          finest tier (res 8 cell / public place); when the
                                                        hierarchy is used, only under kept res-7 parents
  work_users_hist   (lo, hi)                            histogram over coarse cells of #users with work there
                                                        (shared-anchor distribution, D2); L2 sens sqrt(2)
"""
from __future__ import annotations

import math

ROLES = ("home", "work", "other")
N_USERS = "n_users"
ROLE_USERS = "role_users"
ANCHORS_OTHER = "anchors_other"
PLACE_ROLE_USERS = "place_role_users"
ARRIVAL_HOUR = "arrival_hour"
DWELL_HIST = "dwell_hist"
ROLE_TRIPS = "role_trips"
TRIP_DIST = "trip_dist"
DETOUR = "detour"
RETURN_RATE = "return_rate"
# hierarchical place tier (coarse -> fine); the finest tier keeps the legacy name PLACE_ROLE_USERS
PLACE_R6 = "place_role_users_r6"
PLACE_R7 = "place_role_users_r7"
PLACE_TIERS = (PLACE_R6, PLACE_R7, PLACE_ROLE_USERS)       # coarse -> fine
WORK_HIST = "work_users_hist"       # (lo, hi): number of coarse cells with lo <= #work users < hi
WORK_HIST_EDGES = [0, 1, 2, 3, 5, 9, 17, 33, float("inf")]

INF = float("inf")
DWELL_EDGES = [0, 600, 1800, 3600, 7200, 14400, 28800, 43200, INF]
DIST_EDGES = [0, 500, 1000, 2000, 4000, 8000, 16000, INF]
DETOUR_EDGES = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, INF]
MAX_OTHER_ANCHORS = 10
KINDS = ("commute", "other")


def fmt(x: float) -> str:
    return "inf" if x == INF else repr(float(x))


def bin_of(x: float, edges: list[float]) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return None


def bin_keys(edges: list[float]) -> list[tuple[str, str]]:
    return [(fmt(edges[i]), fmt(edges[i + 1])) for i in range(len(edges) - 1)]


def mid(lo: float, hi: float) -> float:
    """Representative value of a bin (geometric for finite lo>0, 1.5*lo for open top bin)."""
    if math.isinf(hi):
        return 1.5 * lo if lo > 0 else 1.0
    if lo <= 0:
        return hi / 2
    return math.sqrt(lo * hi)

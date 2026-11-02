"""Measurement schema for the outings learner (v2). Writer: measure_v2.py. Reader: fit_v2.py.

Adjacency: add/remove ALL outings of one user. Each table is a sum over users of a per-user vector
v_u (a deterministic function of that user's data only; per-user randomness is seeded from the
pseudonym, never from other users). Removing user u subtracts v_u, so the L2 sensitivity is
exactly max ||v_u||_2 over the allowed per-user vectors, stated per table below and unit-tested
(tests/test_learn_v2_sensitivity.py).

Two vector families:
  PRESENCE (distinct users per key): v_u in {0,1}^V with at most m ones            L2 <= sqrt(m)
  PMF      (user-level mixture):     v_u = counts / sum(counts) (L1-normalised)     L2 <= 1
           (||v||_2 <= ||v||_1 = 1; equality for a one-hot user). The released table is the sum of
           user pmfs, so every user has equal weight no matter how many outings it has (this is
           also the right weighting for "typical user" distributions and costs nothing in L2).
  ONEHOT   (user-level histograms): v_u = e_bucket(user statistic)                  L2 = 1
           The statistic may be uncapped (outings per user, ...): the sensitivity is still 1.

Per-user caps (applied first, deterministically; public values from docs/geolife/OUTINGS_PARAMS.md):
  MAX_OUTINGS = 100        uniform random subsample of the user's outings (seeded by pseudonym)
  MAX_STOP_CELLS = 50      distinct res-8 cells with visits: keep the 50 most visited (ties: cell id)
  MAX_TRAV_CELLS = 1000    distinct res-8 traversed cells: keep the 1000 most traversed (ties: cell id)
Per-table key caps m (the "max keys per user" of each presence table) are below. They only shrink
the sensitivity sqrt(m) (a user keeps its m most frequent keys at that level, ranked by number of
outings, ties by cell id). Quantile justification (EDA, qualifying users): res-8 stop cells per
user p50 25 / p90 127, traversed res-8 cells p50 362 / p90 3981. Coarse levels hold far fewer
distinct cells per user (res 6 is 49x larger in area than res 8; a user's stops sit in a handful
of res-6 cells), so m is small there: that is where the signal is (see learn-v2 report).

Table names and keys (all keys are tuples of strings; numbers are formatted by schema.fmt):

  name                    key                          vector  m / sens        meaning
  n_users                 ("users",)                   ONEHOT  1               number of users
  place_users_r6          (cell,)                      PRES    m=12            distinct users with a visit (stop, start or end) in the res-6 cell
  place_users_r7          (cell,)                      PRES    m=25            same at res 7; vocabulary = children of kept r6 cells
  place_users_r8          (cell,)                      PRES    m=50            same at res 8; vocabulary = children of kept r7 cells
  trav_users_r6           (cell,)                      PRES    m=20            distinct users whose outings traverse the res-6 cell
  trav_users_r7           (cell,)                      PRES    m=100           res 7, children of kept r6
  cat_time_visits         (category, coarse, daytype)  PMF     1               when each place category is visited (user pmf over category x am/pm x daytype)
  outing_stops            (n,) n in "0".."9","10+"     PMF     1               stops (kind=stop) per outing
  outing_duration         (lo_min, hi_min)             PMF     1               outing duration histogram
  outing_length           (lo_km, hi_km)               PMF     1               outing length histogram
  outing_mode             (mode,)                      PMF     1               walk|bike|bus|car|train|unknown
  outing_start            (coarse, daytype)            PMF     1               outing start, am/pm x weekday/weekend (2 x 2)
  stop_dwell              (lo_min, hi_min)             PMF     1               dwell of kind=stop visits
  stop_hop                (lo_km, hi_km)               PMF     1               distance between centres of consecutive visits in an outing
  outing_detour           (lo, hi)                     PMF     1               length / distance between first and last visit cell centres (only if that distance >= 2 km; proxy)
  user_outings_hist       (lo, hi)                     ONEHOT  1               number of outings per user (uncapped count)
  user_places_hist        (lo, hi)                     ONEHOT  1               number of distinct res-8 visit cells per user (uncapped)
  user_span_hist          (lo_days, hi_days)           ONEHOT  1               days between first and last outing of the user
  revisit_hist            (lo, hi)                     ONEHOT  1               share of the user's stops at a res-8 cell visited in an earlier outing
  cat_transitions         (from_cat, to_cat)           PMF     1               consecutive visits, categories

v2b (data-dependent redesign, see learn-v2.md section v2b). Dropped after a first run showed they were
pure noise: place_time_users, stop_transitions_r6, trav_users_r8, trav_users_r9 (their parameters fall
back to documented defaults); outing_start went from 24h x 2 to coarse 2 x 2 and cat_time_visits from
4 to 2 time bins. Their rho is re-spent on the remaining tables by the same weighted split.
Coarse time bins: types.TOD_COARSE = (am [05,14), pm [14,05) local). fit_behaviour_v2 expands them
to the 4 TOD_BINS with COARSE_TO_FINE below.

Not released (decision): co-visit degree histogram under a neighbour cap. Adding a user changes
the degree of every neighbour (up to the cap), and a symmetric cap depends on other users' data,
so the per-user vector is not a function of that user alone; the exact sensitivity would be
sqrt(1 + cap) only with an asymmetric (one-sided) cap and the table would be hard to interpret.
Stretch goal, left out.

Time: local = UTC+8; released tables use coarse bins (am/pm); TOD_BINS ([0,6),[6,12),[12,18),[18,24) local) only in fitted params; daytypes
types.DAYTYPES (weekend = Sat/Sun local).
"""
from __future__ import annotations

from geolife.transfer.types import DAYTYPES, LOCAL_UTC_OFFSET_H, TOD_BINS, TOD_COARSE, local_coarse_bin  # noqa: F401
from geolife.transfer.schema import INF, bin_keys, bin_of, fmt, mid  # noqa: F401  (re-exported)

# ---- caps (per user)
MAX_OUTINGS = 100
MAX_STOP_CELLS = 50
MAX_TRAV_CELLS = 1000

# ---- names
N_USERS = "n_users"
PLACE_TIERS = ("place_users_r6", "place_users_r7", "place_users_r8")
TRAV_TIERS = ("trav_users_r6", "trav_users_r7")
PLACE_RES = (6, 7, 8)
TRAV_RES = (6, 7)
# per-user max keys per level (presence tables); sensitivity = sqrt(m)
PLACE_MAX_KEYS = (12, 25, 50)
TRAV_MAX_KEYS = (20, 100)
# declared rho priorities, coarse -> fine (fixed before looking at any SNR; see measure_v2 caveats)
PLACE_PRIORITIES = (6.0, 3.0, 1.0)
TRAV_PRIORITIES = (3.0, 2.0)
# refine thresholds on NOISY values, in sigma units (child level released only under kept parents)
PLACE_TAU = (2.0, 2.0)
TRAV_TAU = (2.0,)

CAT_TIME = "cat_time_visits"
OUTING_STOPS = "outing_stops"
OUTING_DURATION = "outing_duration"
OUTING_LENGTH = "outing_length"
OUTING_MODE = "outing_mode"
OUTING_START = "outing_start"
STOP_DWELL = "stop_dwell"
STOP_HOP = "stop_hop"
OUTING_DETOUR = "outing_detour"
USER_OUTINGS = "user_outings_hist"
USER_PLACES = "user_places_hist"
USER_SPAN = "user_span_hist"
REVISIT = "revisit_hist"
CAT_TRANS = "cat_transitions"

STOPS_KEYS = tuple(str(i) for i in range(10)) + ("10+",)
MODES = ("walk", "bike", "bus", "car", "train", "unknown")

# ---- histogram bin edges. Chosen AFTER looking at the outings EDA (ledger caveat).
DURATION_MIN_EDGES = [0, 5, 10, 20, 40, 80, 160, 320, 640, INF]
LENGTH_KM_EDGES = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, INF]
DWELL_MIN_EDGES = [0, 10, 15, 20, 30, 45, 60, 90, 150, 300, INF]
HOP_KM_EDGES = [0, 0.5, 1, 2, 4, 8, 16, 32, INF]
DETOUR_EDGES = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, INF]
OUTINGS_EDGES = [0, 10, 15, 20, 30, 45, 70, 110, 170, 260, 400, INF]
PLACES_EDGES = [0, 1, 3, 5, 8, 13, 21, 34, 55, 89, 144, INF]
SPAN_DAYS_EDGES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, INF]
REVISIT_EDGES = [round(0.1 * i, 1) for i in range(10)] + [1.0001]
DETOUR_MIN_STRAIGHT_KM = 2.0

HIST_EDGES = {
    OUTING_DURATION: DURATION_MIN_EDGES, OUTING_LENGTH: LENGTH_KM_EDGES, STOP_DWELL: DWELL_MIN_EDGES,
    STOP_HOP: HOP_KM_EDGES, OUTING_DETOUR: DETOUR_EDGES, USER_OUTINGS: OUTINGS_EDGES,
    USER_PLACES: PLACES_EDGES, USER_SPAN: SPAN_DAYS_EDGES, REVISIT: REVISIT_EDGES,
}

# kind per table: "pmf" (L2<=1), "onehot" (L2=1) or "presence" (L2=sqrt(m))
KIND = {N_USERS: "onehot", CAT_TIME: "pmf", OUTING_STOPS: "pmf",
        OUTING_DURATION: "pmf", OUTING_LENGTH: "pmf", OUTING_MODE: "pmf", OUTING_START: "pmf",
        STOP_DWELL: "pmf", STOP_HOP: "pmf", OUTING_DETOUR: "pmf", USER_OUTINGS: "onehot",
        USER_PLACES: "onehot", USER_SPAN: "onehot", REVISIT: "onehot",
        CAT_TRANS: "pmf", **{n: "presence" for n in PLACE_TIERS + TRAV_TIERS}}

ALL_NAMES = (N_USERS,) + PLACE_TIERS + TRAV_TIERS + (
    CAT_TIME, OUTING_STOPS, OUTING_DURATION, OUTING_LENGTH, OUTING_MODE, OUTING_START,
    STOP_DWELL, STOP_HOP, OUTING_DETOUR, USER_OUTINGS, USER_PLACES, USER_SPAN, REVISIT,
    CAT_TRANS)


def max_keys_of(name: str) -> int | None:
    if name in PLACE_TIERS:
        return PLACE_MAX_KEYS[PLACE_TIERS.index(name)]
    if name in TRAV_TIERS:
        return TRAV_MAX_KEYS[TRAV_TIERS.index(name)]
    return None


def l2_bound(name: str) -> float:
    """Exact worst-case per-user L2 norm of the table's vector (the stated sensitivity)."""
    import math
    k = KIND[name]
    return math.sqrt(max_keys_of(name)) if k == "presence" else 1.0


def tod_of(t_utc: int) -> tuple[str, str]:
    """(tod bin, daytype) of a UTC timestamp in local time UTC+8."""
    lt = int(t_utc) + LOCAL_UTC_OFFSET_H * 3600
    hour = (lt // 3600) % 24
    wd = (lt // 86400 + 3) % 7          # 1970-01-01 was Thursday -> 0 = Monday
    return TOD_BINS[hour // 6], DAYTYPES[1 if wd >= 5 else 0]


def local_hour(t_utc: int) -> int:
    return ((int(t_utc) + LOCAL_UTC_OFFSET_H * 3600) // 3600) % 24


def coarse_of(t_utc: int) -> tuple[str, str]:
    """(coarse bin am/pm, daytype) of a UTC timestamp, local time UTC+8."""
    return local_coarse_bin(t_utc), tod_of(t_utc)[1]


def _fine_weights() -> dict[str, dict[str, float]]:
    """Within-coarse-bin split across the 4 fine bins: proportional to the number of local hours of
    the coarse bin that fall in each fine bin. ASSUMPTION, not measured: it treats visits/starts as
    uniform over the hours of the coarse bin, which they are not (the real day has peaks); the
    released data cannot say more. The 08h and 17-18h peaks are separated by the coarse edges but
    their shape inside a bin is lost."""
    hours = {c: [h for h in range(24) if local_coarse_bin(h * 3600 - LOCAL_UTC_OFFSET_H * 3600) == c]
             for c in TOD_COARSE}
    return {c: {t: sum(1 for h in hs if h // 6 == i) / len(hs) for i, t in enumerate(TOD_BINS)}
            for c, hs in hours.items()}


COARSE_TO_FINE = _fine_weights()   # am: night 1/9, morning 6/9, afternoon 2/9; pm: night 5/15, afternoon 4/15, evening 6/15

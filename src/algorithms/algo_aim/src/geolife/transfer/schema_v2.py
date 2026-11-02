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
  outing_stops            (n,) n in "0".."9","10+"     PMF     1               stops (kind=stop) per outing
  outing_duration         (lo_min, hi_min)             PMF     1               outing duration histogram
  outing_length           (lo_km, hi_km)               PMF     1               outing length histogram
  outing_mode             (mode,)                      PMF     1               walk|bike|bus|car|train|unknown
  outing_start            (tod3, daytype)              PMF     1               outing start, TOD_3 x weekday/weekend (3 x 2)
  stop_dwell              (lo_min, hi_min)             PMF     1               dwell of kind=stop visits
  stop_hop                (lo_km, hi_km)               PMF     1               distance between centres of consecutive visits in an outing
  outing_detour           (lo, hi)                     PMF     1               length / distance between first and last visit cell centres (only if that distance >= 2 km; proxy)
  user_outings_hist       (lo, hi)                     ONEHOT  1               number of outings per user (uncapped count)
  user_places_hist        (lo, hi)                     ONEHOT  1               number of distinct res-8 visit cells per user (uncapped)
  user_span_hist          (lo_days, hi_days)           ONEHOT  1               days between first and last outing of the user
  revisit_hist            (lo, hi)                     ONEHOT  1               share of the user's stops at a res-8 cell visited in an earlier outing

v2b (data-dependent redesign, see learn-v2.md section v2b). Dropped after a first run showed they were
pure noise: place_time_users, stop_transitions_r6, trav_users_r8, trav_users_r9 (their parameters fall
back to documented defaults); outing_start went from 24h x 2 to coarse 2 x 2 and cat_time_visits from
4 to 2 time bins. Their rho is re-spent on the remaining tables by the same weighted split.

v2c (decisions D19, D20): the released time bins move from the coarse am/pm (types.TOD_COARSE, kept
for back-compatibility) to the 3 bins types.TOD_3 (morning [05,11), midday [11,16), evening [16,05)
local) x weekday/weekend. fit_behaviour_v2 expands them to the 4 TOD_BINS with TOD3_TO_FINE below.
v2c also adds the sensor tables (SENSOR_NAMES, measured from the raw points by
inside/measure_sensors.py; see the sensor section at the bottom of this file).

v2d (decision D27, 2026-11-02): zero-signal budget dropped after the per-table SNR of the 3
committed dev seeds. No longer released: cat_time_visits (signal 2.0-3.1, at the noise edge; even
on the one seed above the 3.0 floor its per-category content stayed below the fit's internal
floors, so cat_mix / per-category profiles fell back to defaults on EVERY seed), cat_transitions
(signal 0.2-1.2, pure noise, and no parameter ever read it), sensor_step (signal 2.6-3.5 with
every per-mode slice below the per-mode floor on every seed, and the renderer never draws step
lengths - they follow from speed x dt). Their rho is re-spent on the remaining tables by the same
weighted split (the sensor_step share mostly flows to sensor_speed, the renderer's most important
draw). The name constants and bin edges below are kept so old artifacts stay readable. Like v2b,
this is a data-dependent design choice listed in the ledger caveats.

Not released (decision): co-visit degree histogram under a neighbour cap. Adding a user changes
the degree of every neighbour (up to the cap), and a symmetric cap depends on other users' data,
so the per-user vector is not a function of that user alone; the exact sensitivity would be
sqrt(1 + cap) only with an asymmetric (one-sided) cap and the table would be hard to interpret.
Stretch goal, left out.

Time: local = UTC+8; released tables use coarse bins (am/pm); TOD_BINS ([0,6),[6,12),[12,18),[18,24) local) only in fitted params; daytypes
types.DAYTYPES (weekend = Sat/Sun local).
"""
from __future__ import annotations

import math

from geolife.transfer.types import (DAYTYPES, LOCAL_UTC_OFFSET_H, TOD_3, TOD3_EDGES_H, TOD_BINS,  # noqa: F401
                                    TOD_COARSE, local_coarse_bin, local_tod3_bin)

# ---- bin helpers (moved here from the retired v1 transfer/schema.py)
INF = float("inf")


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

CAT_TIME = "cat_time_visits"        # NOT released since v2d (D27); constant kept for old artifacts
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
CAT_TRANS = "cat_transitions"       # NOT released since v2d (D27); constant kept for old artifacts

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

# kind per table: "pmf" (L2<=1), "onehot" (L2=1) or "presence" (L2=sqrt(m)).
# Released tables only (cat_time_visits / cat_transitions dropped in v2d, D27).
KIND = {N_USERS: "onehot", OUTING_STOPS: "pmf",
        OUTING_DURATION: "pmf", OUTING_LENGTH: "pmf", OUTING_MODE: "pmf", OUTING_START: "pmf",
        STOP_DWELL: "pmf", STOP_HOP: "pmf", OUTING_DETOUR: "pmf", USER_OUTINGS: "onehot",
        USER_PLACES: "onehot", USER_SPAN: "onehot", REVISIT: "onehot",
        **{n: "presence" for n in PLACE_TIERS + TRAV_TIERS}}

ALL_NAMES = (N_USERS,) + PLACE_TIERS + TRAV_TIERS + (
    OUTING_STOPS, OUTING_DURATION, OUTING_LENGTH, OUTING_MODE, OUTING_START,
    STOP_DWELL, STOP_HOP, OUTING_DETOUR, USER_OUTINGS, USER_PLACES, USER_SPAN, REVISIT)


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
    """(coarse bin am/pm, daytype) of a UTC timestamp, local time UTC+8 (v2b design, kept for
    back-compatibility; the released tables use tod3_of since v2c)."""
    return local_coarse_bin(t_utc), tod_of(t_utc)[1]


def tod3_of(t_utc: int) -> tuple[str, str]:
    """(TOD_3 bin, daytype) of a UTC timestamp, local time UTC+8 (released time design since v2c)."""
    return local_tod3_bin(t_utc), tod_of(t_utc)[1]


def _fine_weights(bins, bin_of_hour) -> dict[str, dict[str, float]]:
    """Within-released-bin split across the 4 fine bins: proportional to the number of local hours of
    the released bin that fall in each fine bin. ASSUMPTION, not measured: it treats visits/starts as
    uniform over the hours of the released bin, which they are not (the real day has peaks); the
    released data cannot say more. The 08h and 17-18h peaks are separated by the released edges but
    their shape inside a bin is lost."""
    hours = {c: [h for h in range(24) if bin_of_hour(h * 3600 - LOCAL_UTC_OFFSET_H * 3600) == c]
             for c in bins}
    return {c: {t: sum(1 for h in hs if h // 6 == i) / len(hs) for i, t in enumerate(TOD_BINS)}
            for c, hs in hours.items()}


COARSE_TO_FINE = _fine_weights(TOD_COARSE, local_coarse_bin)   # am: night 1/9, morning 6/9, afternoon 2/9; pm: night 5/15, afternoon 4/15, evening 6/15
# morning [05,11): night 1/6, morning 5/6; midday [11,16): morning 1/5, afternoon 4/5;
# evening [16,05): afternoon 2/13, evening 6/13, night 5/13.
TOD3_TO_FINE = _fine_weights(TOD_3, local_tod3_bin)


# ------------------------------------------------------------------ sensor tables (v2c, PLAN step 1)
# Raw-point statistics measured by inside/measure_sensors.py and rendered by outside/render_v2.py.
# All are user-level PMF (per-user counts L1-normalised, L2 <= 1) or ONEHOT (one bucket per user,
# L2 = 1) vectors over FIXED PUBLIC bin edges, so the exact sensitivity of every table is 1.
# Per-user contribution is additionally capped (deterministic uniform subsample of the user's own
# steps/points, seeded from the pseudonym): the cap does not change the sensitivity (the pmf is
# normalised either way) but bounds the work and makes the per-user rule explicit.

SENSOR_INTERVAL = "sensor_interval"              # (lo_s, hi_s)        PMF: dt of consecutive points, dt <= gap-min
SENSOR_INTERVAL_USER = "sensor_interval_user"    # (lo_s, hi_s)        ONEHOT: the user's MEDIAN sampling interval
SENSOR_GAP_LEN = "sensor_gap_len"                # (lo_s, hi_s)        PMF: dt in (gap-min, outing gap]
SENSOR_GAP_RATE = "sensor_gap_rate"              # (lo, hi)            ONEHOT: the user's gaps per logged hour
SENSOR_SPEED = "sensor_speed"                    # (mode, lo, hi)      PMF: step speed m/s by run mode (label + heuristic)
SENSOR_STEP = "sensor_step"                      # NOT released since v2d (D27); constant kept for old artifacts
SENSOR_HEADING = "sensor_heading"                # (lo_deg, hi_deg)    PMF: |heading change| of consecutive moving steps
SENSOR_STOP_JITTER = "sensor_stop_jitter"        # (lo_m, hi_m)        PMF: distance of in-stop points from the stop centre
SENSOR_PTS_OUTING = "sensor_points_per_outing"   # (lo, hi)            PMF: raw points per outing-like run

# sensor_heading and sensor_points_per_outing are not consumed by the renderer; they are kept as
# released DIAGNOSTICS (usable SNR on every dev seed) for the heading-change and duty-cycle realism
# panels; stated in the ledger caveats (D27).
SENSOR_NAMES = (SENSOR_INTERVAL, SENSOR_INTERVAL_USER, SENSOR_GAP_LEN, SENSOR_GAP_RATE,
                SENSOR_SPEED, SENSOR_HEADING, SENSOR_STOP_JITTER, SENSOR_PTS_OUTING)

# caps and thresholds (public; ledger caveat)
MAX_SENSOR_STEPS = 100_000       # per-user cap on steps/points sampled per statistic
SENSOR_GAP_MIN_S = 60.0          # dt above this is a signal-loss gap, not a sampling interval
SENSOR_MOVE_MIN_M = 5.0          # heading change only over steps at least this long
SENSOR_SPEED_MAX_DT_S = 60.0     # step-length statistics only over raw steps with dt <= this
SENSOR_SPEED_THIN_S = 30         # speed is measured on 30 s-thinned steps (raw 1-5 s steps are
SENSOR_SPEED_THIN_MAX_DT_S = 120.0   # GPS-noise dominated; same thinning as the tokenizer heuristic)

# fixed public bin edges (log-spaced where the quantity is heavy-tailed; ledger caveat: informed
# by the public EDA summary of the sensor regime, not re-tuned on any SNR)
INTERVAL_S_EDGES = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 8, 11, 16, 23, 32, 45, 60.0001]
GAP_S_EDGES = [60.0001, 90, 135, 200, 300, 450, 680, 1000, 1500, 2300, 3400, 5100, 7200.0001]
GAP_RATE_EDGES = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, INF]
SPEED_MS_EDGES = [0, 0.3, 0.7, 1.0, 1.4, 2, 3, 4.5, 7, 10, 15, 22, 32, 50.0001]
STEP_M_EDGES = [0, 1, 2, 4, 8, 15, 30, 60, 120, 250, 500, 1000.0001]
HEADING_DEG_EDGES = [0, 5, 10, 20, 30, 45, 60, 90, 120, 150, 180.0001]
JITTER_M_EDGES = [0, 2, 4, 7, 11, 17, 26, 40, 60, 95, 150, 240.0001]
PTS_PER_OUTING_EDGES = [0, 10, 30, 100, 300, 1000, 3000, 10000, INF]

SENSOR_HIST_EDGES = {
    SENSOR_INTERVAL: INTERVAL_S_EDGES, SENSOR_INTERVAL_USER: INTERVAL_S_EDGES,
    SENSOR_GAP_LEN: GAP_S_EDGES, SENSOR_GAP_RATE: GAP_RATE_EDGES,
    SENSOR_SPEED: SPEED_MS_EDGES,
    SENSOR_HEADING: HEADING_DEG_EDGES, SENSOR_STOP_JITTER: JITTER_M_EDGES,
    SENSOR_PTS_OUTING: PTS_PER_OUTING_EDGES,
}

SENSOR_KIND = {SENSOR_INTERVAL: "pmf", SENSOR_INTERVAL_USER: "onehot", SENSOR_GAP_LEN: "pmf",
               SENSOR_GAP_RATE: "onehot", SENSOR_SPEED: "pmf",
               SENSOR_HEADING: "pmf", SENSOR_STOP_JITTER: "pmf", SENSOR_PTS_OUTING: "pmf"}
KIND.update(SENSOR_KIND)        # l2_bound(name) = 1.0 for every sensor table

# declared rho priorities within the sensor group (fixed before any sensor SNR was computed; the
# renderer's most important draws get the larger share, scaled further by ln(vocab) in the split)
SENSOR_PRIORITIES = {SENSOR_INTERVAL: 1.5, SENSOR_INTERVAL_USER: 1.5, SENSOR_GAP_LEN: 1.5,
                     SENSOR_GAP_RATE: 1.0, SENSOR_SPEED: 2.0,
                     SENSOR_HEADING: 1.0, SENSOR_STOP_JITTER: 1.5, SENSOR_PTS_OUTING: 1.0}

# share of the total rho spent on the sensor group when sensor views are given (decision D20):
# the behaviour tables keep 80%, the sensor tables get 20%. Public reasoning: each sensor table
# aggregates millions of per-point events under a per-user cap and L1 normalisation, so its
# user-level pmf has high signal-to-noise already at a small budget.
SENSOR_RHO_FRAC = 0.2


def sensor_vocab(name: str, modes=MODES) -> list[tuple[str, ...]]:
    """Public vocabulary of a sensor table."""
    bins = bin_keys(SENSOR_HIST_EDGES[name])
    if name == SENSOR_SPEED:
        return [(m, lo, hi) for m in modes for lo, hi in bins]
    return [tuple(b) for b in bins]

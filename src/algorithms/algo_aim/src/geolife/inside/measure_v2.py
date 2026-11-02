"""Writer side (v2): list[UserOutings] -> DP measurements (schema_v2) + privacy ledger.

Per user: make_view() caps and summarises one user's outings using only that user's data (random
subsampling is seeded from the pseudonym, not from a shared stream). Tables are sums of per-user
vectors with the exact L2 bounds of schema_v2. Place and traversal tables are released
hierarchically: a child level has as vocabulary the H3 children of parents whose NOISY count passed
tau*sigma (post-processing of the released parent), so no true value ever selects keys.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from geolife.inside.dp import CountTable, ZcdpAccountant, gaussian_measure, split_budget
from geolife.transfer.types import Measurement, Outing, PrivacyLedger, UserOutings
from geolife.transfer import schema_v2 as S2


# ------------------------------------------------------------------ weighted rho split
# (moved here from the retired v1 inside/measure.py; the rule is unchanged)

def rho_weight(l2_sensitivity: float, per_key_clip: float, vocab_size: int, priority: float = 1.0) -> float:
    """Public weight of one table in the rho split.

    Rule (uses only public quantities: the table's exact L2 bound, its per-key clip, the size of
    its public vocabulary, and a declared priority; never any count):
        w = priority * (l2_sensitivity / per_key_clip)^2 * ln(max(vocab_size, 2))
    Derivation: sigma = l2 / sqrt(2 rho). A single user moves a key by at most `per_key_clip`;
    the detection threshold over V noisy keys is ~ sqrt(2 ln V) * sigma. Equalising
    per_key_clip / threshold across tables gives rho_i proportional to (l2/clip)^2 * ln V
    (the clip cancels; (l2/clip)^2 = max keys per user). `priority` is a declared design choice.
    """
    return float(priority) * (l2_sensitivity / per_key_clip) ** 2 * math.log(max(int(vocab_size), 2))


FLOOR_FRAC = 0.4   # share of rho split equally across tables so no table is starved


def weighted_rho_split(rho_total: float, weights: Sequence[float],
                       floor_frac: float = FLOOR_FRAC) -> list[float]:
    """rho_i = rho_total * (floor_frac / n + (1 - floor_frac) * w_i / sum(w)).

    Only the public weights enter. The equal-split floor keeps cheap-but-vital tables (n_users)
    from being starved: the weight rule is blind to how large the true signal is relative to the
    clip bound. The shares sum to rho_total (up to float rounding)."""
    n = len(weights)
    sh = split_budget(rho_total * (1 - floor_frac), weights)
    return [rho_total * floor_frac / n + x for x in sh]


# ------------------------------------------------------------------ public vocabulary

@dataclass
class VocabV2:
    """Public vocabulary: res-6 H3 cells covering the analysis region and the place categories."""
    r6_cells: list[str]
    categories: list[str] = field(default_factory=lambda: ["unknown"])

    def __post_init__(self):
        self.r6_cells = sorted(set(self.r6_cells))
        if "unknown" not in self.categories:
            self.categories = list(self.categories) + ["unknown"]

    @classmethod
    def from_bbox(cls, bbox, categories: Sequence[str] = ("unknown",)) -> "VocabV2":
        import h3
        lat0, lat1, lon0, lon1 = bbox
        poly = h3.LatLngPoly([(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)])
        return cls(sorted(h3.polygon_to_cells(poly, 6)), list(categories))


def _cell(c: str) -> str:
    return c[3:] if c.startswith("h3:") else c


@lru_cache(maxsize=None)
def _parent(cell: str, res: int) -> str:
    import h3
    return cell if h3.get_resolution(cell) <= res else h3.cell_to_parent(cell, res)


@lru_cache(maxsize=None)
def _centre(cell: str) -> tuple[float, float]:
    import h3
    return h3.cell_to_latlng(cell)


def _km(a: str, b: str) -> float:
    (la1, lo1), (la2, lo2) = _centre(a), _centre(b)
    p1, p2 = math.radians(la1), math.radians(la2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(h)))


# ------------------------------------------------------------------ per-user view

@dataclass
class UserView:
    """Everything one user contributes, computed from that user's data only."""
    user: str
    place: list[Counter]            # per place level: cell -> #outings with a visit there (after caps)
    trav: list[Counter]             # per trav level: cell -> #outings traversing it (after caps)
    pmf: dict[str, Counter]         # pmf/onehot table name -> raw counts (key tuple -> count)


def _bucket(x: float, edges) -> tuple[str, str] | None:
    b = S2.bin_of(x, edges)
    return None if b is None else S2.bin_keys(edges)[b]


def _user_rng(seed: int, user: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}:{user}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def make_view(u: UserOutings, vocab: VocabV2, seed: int = 0, caps: dict | None = None) -> UserView:
    caps = {"outings": S2.MAX_OUTINGS, "stop_cells": S2.MAX_STOP_CELLS,
            "trav_cells": S2.MAX_TRAV_CELLS, **(caps or {})}
    r6set = set(vocab.r6_cells)
    pm: dict[str, Counter] = {n: Counter() for n in S2.ALL_NAMES if S2.KIND[n] != "presence"}
    pm[S2.N_USERS][("users",)] = 1

    allo = list(u.outings)
    # ---- user-level histograms (one-hot; statistics are uncapped, sensitivity is 1 regardless)
    if allo:
        pm[S2.USER_OUTINGS][_bucket(len(allo), S2.OUTINGS_EDGES)] += 1
        span = (max(o.t_end for o in allo) - min(o.t_start for o in allo)) / 86400.0
        pm[S2.USER_SPAN][_bucket(span, S2.SPAN_DAYS_EDGES)] += 1
        cells_all = {_cell(v.cell) for o in allo for v in o.visits}
        if cells_all:
            pm[S2.USER_PLACES][_bucket(len(cells_all), S2.PLACES_EDGES)] += 1

    # ---- cap 1: outings (uniform subsample, per-user seed)
    if len(allo) > caps["outings"]:
        idx = _user_rng(seed, u.user).permutation(len(allo))[:caps["outings"]]
        allo = [allo[i] for i in sorted(idx)]
    allo.sort(key=lambda o: (o.t_start, o.t_end))

    # ---- cap 2: distinct res-8 visit cells
    vc = Counter(_cell(v.cell) for o in allo for v in o.visits if _parent(_cell(v.cell), 6) in r6set)
    keep_cells = {c for c, _ in sorted(vc.items(), key=lambda kv: (-kv[1], kv[0]))[:caps["stop_cells"]]}

    def vis(o: Outing):
        return [v for v in o.visits if _cell(v.cell) in keep_cells]

    place = [Counter() for _ in S2.PLACE_RES]
    seen: set[str] = set()
    n_stop = n_rev = 0
    for o in allo:
        vs = vis(o)
        cs = {_cell(v.cell) for v in vs}
        for k, r in enumerate(S2.PLACE_RES):
            for c in {_parent(c, r) for c in cs}:
                place[k][c] += 1
        # outing shape (per-outing events -> user pmf)
        dur_min = (o.t_end - o.t_start) / 60.0
        b = _bucket(dur_min, S2.DURATION_MIN_EDGES)
        if b:
            pm[S2.OUTING_DURATION][b] += 1
        b = _bucket(o.length_m / 1000.0, S2.LENGTH_KM_EDGES)
        if b:
            pm[S2.OUTING_LENGTH][b] += 1
        pm[S2.OUTING_MODE][(o.mode if o.mode in S2.MODES else "unknown",)] += 1
        pm[S2.OUTING_START][S2.tod3_of(o.t_start)] += 1
        stops = [v for v in vs if v.kind == "stop"]
        pm[S2.OUTING_STOPS][(S2.STOPS_KEYS[min(len(stops), 10)],)] += 1
        for v in stops:
            b = _bucket(v.dwell_s / 60.0, S2.DWELL_MIN_EDGES)
            if b:
                pm[S2.STOP_DWELL][b] += 1
            n_stop += 1
            n_rev += _cell(v.cell) in seen
        seen |= {_cell(v.cell) for v in stops}
        # visits: hop distances (the category x time tables were dropped in v2d, D27)
        prev = None
        for v in vs:
            c = _cell(v.cell)
            if prev is not None:
                b = _bucket(_km(prev, c), S2.HOP_KM_EDGES)
                if b:
                    pm[S2.STOP_HOP][b] += 1
            prev = c
        if len(vs) >= 2:
            a, z = _cell(vs[0].cell), _cell(vs[-1].cell)
            d = _km(a, z)
            if d >= S2.DETOUR_MIN_STRAIGHT_KM:
                ratio = max(1.0, (o.length_m / 1000.0) / d)
                b = _bucket(ratio, S2.DETOUR_EDGES)
                if b:
                    pm[S2.OUTING_DETOUR][b] += 1
    if n_stop:
        pm[S2.REVISIT][_bucket(n_rev / n_stop, S2.REVISIT_EDGES)] += 1

    # ---- cap 3: traversed res-8 cells (input cells are res 9; parents give res 8)
    c8 = Counter()
    for o in allo:
        for c in {_parent(_cell(x), 8) for x in o.cells if _parent(_cell(x), 6) in r6set}:
            c8[c] += 1
    keep8 = {c for c, _ in sorted(c8.items(), key=lambda kv: (-kv[1], kv[0]))[:caps["trav_cells"]]}
    trav = [Counter() for _ in S2.TRAV_RES]
    for o in allo:
        c9 = {_cell(x) for x in o.cells if _parent(_cell(x), 8) in keep8}
        for k, r in enumerate(S2.TRAV_RES):
            for c in {_parent(c, r) for c in c9}:
                trav[k][c] += 1
    return UserView(u.user, place, trav, pm)


# ------------------------------------------------------------------ tables

def presence_table(views: Sequence[UserView], counter_of: Callable[[UserView], Counter],
                   vocab: list[tuple[str, ...]], m: int) -> CountTable:
    """Distinct users per key. A user keeps its m most frequent keys inside the vocabulary
    (ties: key order). Exact L2 bound sqrt(m)."""
    idx = {k: i for i, k in enumerate(vocab)}
    vecs = []
    for v in views:
        cand = [(c, n) for c, n in counter_of(v).items() if (c,) in idx]
        top = sorted(cand, key=lambda cn: (-cn[1], cn[0]))[:m]
        vecs.append({idx[(c,)]: 1.0 for c, _ in top})
    return CountTable("presence", list(vocab), vecs, math.sqrt(m), 1.0, m)


def pmf_table(views: Sequence[UserView], name: str, vocab: list[tuple[str, ...]]) -> CountTable:
    """Sum of per-user pmfs (counts / sum of in-vocabulary counts). L2 bound 1."""
    idx = {k: i for i, k in enumerate(vocab)}
    vecs, oov = [], 0
    for v in views:
        cnt = {idx[k]: float(n) for k, n in v.pmf[name].items() if k in idx}
        oov += sum(1 for k in v.pmf[name] if k not in idx)
        tot = sum(cnt.values())
        vecs.append({i: x / tot for i, x in cnt.items()} if tot > 0 else {})
    return CountTable("transition", list(vocab), vecs, 1.0, 1.0, max(1, len(vocab)), oov)


def children_of(cell: str, res: int) -> list[str]:
    import h3
    return sorted(h3.cell_to_children(cell, res + 1))


def pmf_vocab(name: str, vocab: VocabV2) -> list[tuple[str, ...]]:
    tods, days = S2.TOD_3, S2.DAYTYPES
    if name == S2.N_USERS:
        return [("users",)]
    if name == S2.OUTING_STOPS:
        return [(k,) for k in S2.STOPS_KEYS]
    if name == S2.OUTING_MODE:
        return [(m,) for m in S2.MODES]
    if name == S2.OUTING_START:
        return [(t, d) for t in tods for d in days]
    return S2.bin_keys(S2.HIST_EDGES[name])


# ------------------------------------------------------------------ main entry

CAVEATS = [
    "Parameters informed by real data (non-DP, not accounted): outing gap 2 h, real-outing rule "
    "(>=5 min, >=500 m), minimum activity (>=10 outings on >=5 days), stop rule 200 m / 10 min, "
    "the place-snap radius (150 m, raised from 75 m after seeing the snap rate), the "
    "time bins (4 tod x weekday/weekend in the tokenizer, 3 tod bins x weekday/weekend in the "
    "released tables), the H3 hierarchy resolutions (place 6-7-8, traversal 6-7), "
    "and the per-user caps (100 outings, 50 res-8 stop cells, 1000 res-8 traversed cells). They were "
    "chosen from the outings EDA run on the real data.",
    "The analysis region (the Beijing core bbox that fixes the public H3 vocabulary and restricts "
    "which points/users contribute) was chosen by looking at where the real data is: a "
    "data-dependent public setting outside the DP guarantee (ledger-review finding, 2026-11-02).",
    "The tokenizer's mode heuristic (median-speed cuts walk<2 / bike<4 / bus<7 / car<15 m/s, label "
    "vote threshold 0.5) and the 70 m/s speed-spike point filter are public settings informed by "
    "the (non-DP) EDA; they shape outing_mode and the per-mode sensor tables "
    "(ledger-review finding, 2026-11-02).",
    "Histogram bin edges and the per-table per-user key caps (place m=12/25/50, traversal "
    "m=20/100) were chosen after looking at the EDA quantiles of the real data (not "
    "accounted).",
    "Declared design choices fixed before any SNR was computed: place-tier priorities "
    f"{list(S2.PLACE_PRIORITIES)}, traversal-tier priorities {list(S2.TRAV_PRIORITIES)}, refine "
    f"thresholds tau place {list(S2.PLACE_TAU)} / traversal {list(S2.TRAV_TAU)} (in sigma units of the "
    f"parent's noisy value) and a {int(FLOOR_FRAC * 100)}% equal-split floor. rho weights use public "
    "quantities only (table sensitivity^2, public vocabulary size, priority).",
    "Floating-point Gaussian sampling (numpy) is not hardened against floating-point attacks.",
    "Per-user processing (outing subsampling seeded from the user's pseudonym, top-m key truncation, "
    "L1 normalisation, stop detection and snapping upstream) is a deterministic function of that "
    "user's data alone, so the stated sensitivities hold; the tokenizer's behaviour is assumed to "
    "have the same property.",
    "Child tiers are released only under parents whose NOISY count passed tau*sigma (post-processing); "
    "when a tier's vocabulary is empty its planned rho is still charged (conservative).",
    "Tables are sums of per-user pmfs (user-weighted); user-level histograms are one-hot per user "
    "(L2 = 1), computed on uncapped per-user statistics.",
    "outing_detour is a proxy (route length / distance between first and last visit cell centres, "
    "only when that distance >= 2 km, quantised at about 0.5 km); outing_mode comes from labels "
    "available for a subset of users and is mostly 'unknown'.",
    "DATA-DEPENDENT DESIGN CHOICE (outside the formal DP guarantee): the table list (which tables "
    "were dropped: place_time_users, stop_transitions_r6, trav_users_r8, trav_users_r9 after the "
    "first real-data run; cat_time_visits, cat_transitions and sensor_step after the per-table "
    "signal-to-noise of the 3-seed dev run, decision D27) and the "
    "coarse time bins were chosen after looking at the "
    "signal-to-noise of a previous run on the real data. The released time design is now 3 "
    "time-of-day bins x weekday/weekend (morning [05,11) / midday [11,16) / evening [16,05) local, "
    "decision D19/D20): both the choice of 3 bins and their edges are public settings fixed after "
    "that previous run, from public reasoning about the day (the 08h and 17-18h peaks). The final "
    "mechanism's DP guarantee is therefore conditional on these data-dependent design choices, "
    "which are not accounted for.",
    "No category table is released (cat_time_visits and cat_transitions dropped in D27: at or "
    "below the usability edge on every dev seed, and the fitted category mix / per-category time "
    "profiles fell back to defaults on every seed even when the table passed the overall floor). "
    "The generator's place categories and visit-time-by-category profiles are therefore public "
    "defaults, not measured; the freed rho is re-spent on the remaining tables by the same "
    "weighted split.",
    "Within-released-bin split of time mass into the 4 fine bins is a fixed "
    "assumption, not measured (fit_v2 weights the overlapped local hours by the public HOUR_SHAPE "
    "prior; schema_v2.TOD3_TO_FINE documents the uniform variant).",
    "Generator-side post-processing settings (fit_v2 DEFAULTS priors and shrinkage strengths, the "
    "revisit/start/end draw design, mode length factors, AR noise time constants; decisions "
    "D21-D23) are public settings tuned against the real-vs-synthetic comparison: outside the "
    "formal DP guarantee, like every other public setting listed here.",
    "Co-visit degree histogram not released (sensitivity would depend on other users' data).",
]

SENSOR_CAVEATS = [
    "Sensor tables (sensor_*) are user-level pmf/one-hot histograms of raw-point statistics with "
    f"FIXED PUBLIC bin edges and a per-user CAPPED SAMPLE of steps/points (deterministic uniform "
    f"subsample of at most {S2.MAX_SENSOR_STEPS} steps per statistic, seeded from the user's "
    "pseudonym; a function of that user's data alone). The bin edges, the cap, the gap threshold "
    f"({int(S2.SENSOR_GAP_MIN_S)} s) and the run/mode heuristics were informed by the (non-DP) "
    "sensor EDA of the real data: public settings outside the DP guarantee.",
    "The split of the total rho between the behaviour tables (80%) and the sensor tables (20%), "
    "the declared sensor priorities and the equal-split floor within the sensor group use public "
    "quantities only (decision D20, fixed before any sensor SNR was computed).",
    "Any sensor table whose released mass is mostly noise is ignored by the generator, which then "
    "falls back to the hand-set defaults of outside/render_v2.SENSOR_DEFAULTS (sensor noise ~6 m, "
    "stop jitter ~25 m, ~5 gaps/h lognormal median 210 s, per-user sampling period from "
    "{1,2,3,4,5,13} s, speeds walk 1.3 / bike 3.5 / bus 5.0 / car 9.0 / train 19.0 m/s): hand-set "
    "values copied from the (non-DP) EDA of the real data, outside the DP guarantee.",
    "sensor_step is NOT released (dropped in D27: every per-mode slice was below the per-mode "
    "noise floor on every dev seed, and the renderer derives step lengths from speed x sampling "
    "interval, never from a step table); its rho is re-spent within the sensor group, mostly on "
    "sensor_speed. sensor_heading and sensor_points_per_outing are released as DIAGNOSTICS only "
    "(usable SNR on every dev seed): the renderer does not draw from them; they anchor the "
    "heading-change and duty-cycle panels of the real-vs-synthetic comparison (D27).",
]

NO_SENSOR_CAVEAT = (
    "Sensor statistics (sampling interval, gap rate, noise, speeds) are NOT measured in this "
    "artifact (no raw points were given to the learner); the generator uses the hand-set defaults "
    "of outside/render_v2.SENSOR_DEFAULTS (sensor noise ~6 m, stop jitter ~25 m, ~5 gaps/h, "
    "per-user sampling period from {1,2,3,4,5,13} s, speeds by mode), copied from the (non-DP) EDA "
    "of the real data.")


def build_measurements_v2(users: Sequence[UserOutings], vocab: VocabV2, rho_total: float,
                          rng: np.random.Generator, delta: float = 1e-6, seed: int = 0,
                          vocab_override: dict[str, list] | None = None,
                          caps: dict | None = None,
                          sensor_views: Sequence | None = None,
                          sensor_frac: float = S2.SENSOR_RHO_FRAC
                          ) -> tuple[list[Measurement], PrivacyLedger]:
    """Release the full v2 table set. `vocab_override {name: keys}` is an evaluation aid
    (noiseless re-run on the keys a noisy run released; use with a huge rho_total).

    `sensor_views` (inside/measure_sensors.SensorView, objects with `.pmf`) adds the sensor tables:
    the behaviour tables then get (1 - sensor_frac) of rho_total and the sensor group sensor_frac
    (decision D20: 80% / 20%), split within each group by the same public-weight rule."""
    views = [make_view(u, vocab, seed, caps) for u in users]
    n6 = len(vocab.r6_cells)
    rho_behaviour = rho_total * (1.0 - sensor_frac) if sensor_views is not None else rho_total
    rho_sensor = rho_total - rho_behaviour

    # ---- plan the rho split from public quantities
    plan: list[tuple[str, float]] = []
    for name in S2.ALL_NAMES:
        if S2.KIND[name] != "presence":
            plan.append((name, rho_weight(S2.l2_bound(name), 1.0, len(pmf_vocab(name, vocab)))))
    for names, pri, mk in ((S2.PLACE_TIERS, S2.PLACE_PRIORITIES, S2.PLACE_MAX_KEYS),
                           (S2.TRAV_TIERS, S2.TRAV_PRIORITIES, S2.TRAV_MAX_KEYS)):
        for k, nm in enumerate(names):
            plan.append((nm, rho_weight(math.sqrt(mk[k]), 1.0, n6 * 7 ** k, pri[k])))
    shares = weighted_rho_split(rho_behaviour, [w for _, w in plan])
    rho_of = dict(zip([n for n, _ in plan], shares))
    if sensor_views is not None:
        sw = [(n, rho_weight(1.0, 1.0, len(S2.sensor_vocab(n)), S2.SENSOR_PRIORITIES[n]))
              for n in S2.SENSOR_NAMES]
        rho_of.update(zip([n for n, _ in sw], weighted_rho_split(rho_sensor, [w for _, w in sw])))

    acc = ZcdpAccountant(rho_total * (1 + 1e-9))
    out: list[Measurement] = []

    def release(t: CountTable, name: str) -> Measurement:
        m = gaussian_measure(t, rho=rho_of[name], rng=rng, name=name)
        acc.spend(m)
        out.append(m)
        return m

    for name in S2.ALL_NAMES:
        if S2.KIND[name] == "presence":
            continue
        voc = (vocab_override[name] if vocab_override and name in vocab_override
               else pmf_vocab(name, vocab))
        release(pmf_table(views, name, [tuple(k) for k in voc]), name)

    for names, res, mk, taus, attr in (
            (S2.PLACE_TIERS, S2.PLACE_RES, S2.PLACE_MAX_KEYS, S2.PLACE_TAU, "place"),
            (S2.TRAV_TIERS, S2.TRAV_RES, S2.TRAV_MAX_KEYS, S2.TRAV_TAU, "trav")):
        voc = [(c,) for c in vocab.r6_cells]
        for k, nm in enumerate(names):
            if vocab_override and nm in vocab_override:
                voc = [tuple(x) for x in vocab_override[nm]]
            t = presence_table(views, lambda v, k=k, a=attr: getattr(v, a)[k], voc, mk[k])
            m = release(t, nm)
            if k < len(names) - 1:
                cut = taus[k] * m.sigma
                voc = [(c,) for key, val in zip(m.keys, m.values) if val >= cut
                       for c in children_of(key[0], res[k])]

    if sensor_views is not None:
        for name in S2.SENSOR_NAMES:
            voc = (vocab_override[name] if vocab_override and name in vocab_override
                   else S2.sensor_vocab(name))
            release(pmf_table(sensor_views, name, [tuple(k) for k in voc]), name)

    caps_ledger = {"max_outings_per_user": S2.MAX_OUTINGS, "max_stop_cells_per_user": S2.MAX_STOP_CELLS,
                   "max_trav_cells_per_user": S2.MAX_TRAV_CELLS,
                   **{f"max_keys_{n}": float(S2.max_keys_of(n)) for n in S2.PLACE_TIERS + S2.TRAV_TIERS}}
    caveats = list(CAVEATS)
    if sensor_views is not None:
        caps_ledger["max_sensor_steps_per_user"] = float(S2.MAX_SENSOR_STEPS)
        caveats += SENSOR_CAVEATS
    else:
        caveats.append(NO_SENSOR_CAVEAT)
    return out, acc.ledger(delta=delta, caps=caps_ledger, caveats=caveats)

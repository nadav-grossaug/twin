"""Writer side: UserRecords -> the DP measurement set that fit_behaviour consumes.

Roles are derived per user from that user's own data only (so the clipping bound stays exact).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from geolife.inside.dp import (CountTable, ZcdpAccountant, build_count_table, gaussian_measure, split_budget,
                  threshold_release)
from geolife.transfer.types import Measurement, PrivacyLedger, Trip, UserRecord
from geolife.transfer import schema as S


def assign_roles(r: UserRecord, tz_offset_s: int = 0) -> dict[str, str]:
    """place_id -> role. home = most night dwell; work = most weekday-daytime dwell elsewhere."""
    night: dict[str, float] = {}
    day: dict[str, float] = {}
    total: dict[str, float] = {}
    for s in r.stays:
        d = max(0, s.t_end - s.t_start)
        lt = s.t_start + tz_offset_s
        h = (lt // 3600) % 24
        wd = (lt // 86400 + 3) % 7  # 1970-01-01 was Thursday -> 0=Mon
        total[s.place_id] = total.get(s.place_id, 0) + d
        if h >= 20 or h < 8:
            night[s.place_id] = night.get(s.place_id, 0) + d
        elif wd < 5 and 9 <= h < 17:
            day[s.place_id] = day.get(s.place_id, 0) + d
    roles: dict[str, str] = {}
    if not total:
        return roles
    pool = night or total
    home = sorted(pool, key=lambda p: (-pool[p], p))[0]
    roles[home] = "home"
    cand = {p: v for p, v in day.items() if p != home}
    if cand:
        roles[sorted(cand, key=lambda p: (-cand[p], p))[0]] = "work"
    for p in total:
        roles.setdefault(p, "other")
    return roles


# ---------------------------------------------------------------- weighted rho split

def rho_weight(l2_sensitivity: float, per_key_clip: float, vocab_size: int, priority: float = 1.0) -> float:
    """Public weight of one table in the rho split.

    Rule (uses only public quantities: the table's exact L2 bound, its per-key clip, the size of
    its public vocabulary, and a declared priority; never any count):
        w = priority * (l2_sensitivity / per_key_clip)^2 * ln(max(vocab_size, 2))
    Derivation: sigma = l2 / sqrt(2 rho). A single user moves a key by at most `per_key_clip`;
    the detection threshold over V noisy keys is ~ sqrt(2 ln V) * sigma. Equalising
    per_key_clip / threshold across tables gives rho_i proportional to (l2/clip)^2 * ln V
    (the clip cancels; (l2/clip)^2 = max keys per user). `priority` is a declared design choice:
    the shared-anchor place tiers carry the D2 relation (see PLACE_PRIORITIES).
    """
    return float(priority) * (l2_sensitivity / per_key_clip) ** 2 * math.log(max(int(vocab_size), 2))


FLOOR_FRAC = 0.4   # share of rho split equally across tables so no table is starved


def weighted_rho_split(rho_total: float, weights: Sequence[float],
                       floor_frac: float = FLOOR_FRAC) -> list[float]:
    """rho_i = rho_total * (floor_frac / n + (1 - floor_frac) * w_i / sum(w)).

    Only the public weights enter. The equal-split floor keeps cheap-but-vital tables (n_users,
    role_users, return_rate) from being starved: the weight rule is blind to how large the true
    signal is relative to the clip bound. The shares sum to rho_total (up to float rounding)."""
    n = len(weights)
    sh = split_budget(rho_total * (1 - floor_frac), weights)
    return [rho_total * floor_frac / n + x for x in sh]


# Declared place-tier priorities, coarse -> fine: finer cells hold fewer users each (public
# reasoning), so the coarse tier and the shared-anchor histogram get the most budget.
PLACE_PRIORITIES = (6.0, 3.0, 1.0)
HIST_PRIORITY = 6.0


@dataclass
class HierarchySpec:
    """Fixed, public place hierarchy (fixed before the v1 SNR was looked at: H3 res 6 -> 7 -> 8).

    place_cell: public map, place id -> H3 cell at the finest H3 resolution (res[-1]) for ids that
    are not themselves "h3:<cell>" tokens (e.g. Overture places). tau: a child level is released only
    under parents whose NOISY count is >= tau * sigma (same logic as dp.hierarchical_release,
    chained over three levels with each level's parents re-used from the previous release).
    """
    place_cell: dict[str, str] = field(default_factory=dict)
    res: tuple[int, ...] = (6, 7, 8)
    max_keys: tuple[int, ...] = (6, 8, 8)
    tau: float = 2.0
    priorities: tuple[float, ...] = PLACE_PRIORITIES
    hist_priority: float = HIST_PRIORITY

    def tier_names(self) -> list[str]:
        return [f"place_role_users_r{r}" for r in self.res[:-1]] + [S.PLACE_ROLE_USERS]


class _FixedTable(CountTable):
    """Table whose true values are given directly (a deterministic function of per-user vectors
    with a separately proven L2 bound, e.g. a histogram of per-cell user counts)."""

    def __init__(self, vocab, values, l2_sensitivity):
        super().__init__("transition", list(vocab), [], float(l2_sensitivity), 1.0, 1)
        self._values = np.asarray(values, dtype=float)

    @property
    def true_values(self):
        return self._values


def _cell_chain(pid: str, spec: HierarchySpec) -> list[str] | None:
    """Node ids of `pid` at each level of the hierarchy (public map lookup only)."""
    import h3
    c = pid[3:] if pid.startswith("h3:") else spec.place_cell.get(pid)
    if c is None:
        return None
    chain = []
    for r in spec.res[:-1]:
        chain.append("h3:" + (h3.cell_to_parent(c, r) if h3.get_resolution(c) > r else c))
    chain.append(pid)
    return chain


def build_behaviour_measurements(
        records: Sequence[UserRecord], place_vocab: Sequence[tuple[str, str]], rho_total: float,
        rng: np.random.Generator, tz_offset_s: int = 0, max_places: int = 8, clip: float = 3.0,
        detour_fn: Callable[[Trip], float | None] | None = None,
        delta: float = 1e-6, caps: dict | None = None, hierarchy: HierarchySpec | None = None,
        split: str = "weighted", vocab_override: dict[str, list] | None = None
) -> tuple[list[Measurement], PrivacyLedger]:
    """Release all behaviour tables. `place_vocab` = public [(place_id, category)].

    split="weighted" (default): rho by `rho_weight` (public quantities only); "equal": legacy split.
    hierarchy=None: legacy flat place_role_users table. With a HierarchySpec: coarse tier released
    fully, finer tiers only under parents whose noisy count passes tau (hierarchical release), plus
    the shared-anchor histogram `work_users_hist`.
    vocab_override {name: keys}: evaluation aid (noiseless re-run on the keys a noisy run released).
    """
    cat = dict(place_vocab)
    roles_cache = {r.user: assign_roles(r, tz_offset_s) for r in records}
    tables = []

    def add(name, spec, vocab, kind="presence", max_keys=1, per_key_clip=None):
        t = build_count_table(records, spec, vocab, kind, max_keys=max_keys, per_key_clip=per_key_clip)
        tables.append((name, t))

    add(S.N_USERS, lambda r: {("users",): 1.0}, [("users",)])
    add(S.ROLE_USERS, lambda r: {(ro,): 1.0 for ro in set(roles_cache[r.user].values())},
        [(ro,) for ro in S.ROLES], max_keys=3)
    add(S.ANCHORS_OTHER, lambda r: {(str(min(S.MAX_OTHER_ANCHORS, sum(
        1 for v in roles_cache[r.user].values() if v == "other"))),): 1.0},
        [(str(n),) for n in range(S.MAX_OTHER_ANCHORS + 1)])
    if hierarchy is None:
        add(S.PLACE_ROLE_USERS, lambda r: {(p, cat.get(p, "unknown"), ro): 1.0
                                           for p, ro in roles_cache[r.user].items()},
            [(p, c, ro) for p, c in place_vocab for ro in S.ROLES], max_keys=max_places)

    def arrival(r):
        d: dict = {}
        for s in r.stays:
            ro = roles_cache[r.user].get(s.place_id)
            if ro:
                k = (ro, str(((s.t_start + tz_offset_s) // 3600) % 24))
                d[k] = d.get(k, 0) + 1.0
        return d
    add(S.ARRIVAL_HOUR, arrival, [(ro, str(h)) for ro in S.ROLES for h in range(24)],
        "transition", max_keys=24, per_key_clip=clip)

    def dwell(r):
        d: dict = {}
        for s in r.stays:
            ro = roles_cache[r.user].get(s.place_id)
            b = S.bin_of(s.t_end - s.t_start, S.DWELL_EDGES)
            if ro and b is not None:
                k = (ro,) + S.bin_keys(S.DWELL_EDGES)[b]
                d[k] = d.get(k, 0) + 1.0
        return d
    add(S.DWELL_HIST, dwell, [(ro,) + b for ro in S.ROLES for b in S.bin_keys(S.DWELL_EDGES)],
        "transition", max_keys=12, per_key_clip=clip)

    def rtrips(r):
        d: dict = {}
        m = roles_cache[r.user]
        for t in r.trips:
            if t.origin in m and t.dest in m:
                k = (m[t.origin], m[t.dest])
                d[k] = d.get(k, 0) + 1.0
        return d
    add(S.ROLE_TRIPS, rtrips, [(a, b) for a in S.ROLES for b in S.ROLES], "transition",
        max_keys=9, per_key_clip=clip)

    def tdist(r):
        d: dict = {}
        m = roles_cache[r.user]
        for t in r.trips:
            if t.origin not in m or t.dest not in m or t.length_m <= 0:
                continue
            kind = "commute" if {m[t.origin], m[t.dest]} == {"home", "work"} else "other"
            b = S.bin_of(t.length_m, S.DIST_EDGES)
            k = (kind,) + S.bin_keys(S.DIST_EDGES)[b]
            d[k] = d.get(k, 0) + 1.0
        return d
    add(S.TRIP_DIST, tdist, [(k,) + b for k in S.KINDS for b in S.bin_keys(S.DIST_EDGES)],
        "transition", max_keys=8, per_key_clip=clip)

    if detour_fn is not None:
        def det(r):
            d: dict = {}
            for t in r.trips:
                x = detour_fn(t)
                b = S.bin_of(x, S.DETOUR_EDGES) if x is not None else None
                if b is not None:
                    k = S.bin_keys(S.DETOUR_EDGES)[b]
                    d[k] = d.get(k, 0) + 1.0
            return d
        add(S.DETOUR, det, S.bin_keys(S.DETOUR_EDGES), "transition", max_keys=6, per_key_clip=clip)

    def ret(r):
        seen: set = set()
        new = ret_ = 0
        for s in sorted(r.stays, key=lambda s: s.t_start):
            if s.place_id in seen:
                ret_ += 1
            else:
                new += 1
                seen.add(s.place_id)
        return {("new",): float(new), ("return",): float(ret_)}
    add(S.RETURN_RATE, ret, [("new",), ("return",)], "transition", max_keys=2, per_key_clip=50.0)

    # ---- plan: (name, weight) for every table, including the hierarchy levels and the histogram
    plan: list[tuple[str, float]] = [
        (name, rho_weight(t.l2_sensitivity, t.per_key_clip, len(t.vocab))) for name, t in tables]
    chains = {}
    if hierarchy is not None:
        hs = hierarchy
        nl = len(hs.res)
        chains = {p: _cell_chain(p, hs) for p, _ in place_vocab}
        chains = {p: c for p, c in chains.items() if c is not None}
        nodes = [sorted({c[k] for c in chains.values()}) for k in range(nl)]
        for k, nm in enumerate(hs.tier_names()):
            plan.append((nm, rho_weight(math.sqrt(hs.max_keys[k]), 1.0,
                                        len(nodes[k]) * len(S.ROLES), hs.priorities[k])))
        plan.append((S.WORK_HIST, rho_weight(math.sqrt(2.0), 1.0, len(S.WORK_HIST_EDGES) - 1,
                                             hs.hist_priority)))
    names = [n for n, _ in plan]
    shares = (weighted_rho_split(rho_total, [w for _, w in plan]) if split == "weighted"
              else [rho_total / len(plan)] * len(plan))
    rho_of = dict(zip(names, shares))
    acc = ZcdpAccountant(rho_total * (1 + 1e-9))
    out: list[Measurement] = []

    def release(t: CountTable, name: str, l2: float | None = None) -> Measurement:
        m = gaussian_measure(t, l2_sensitivity=l2, rho=rho_of[name], rng=rng, name=name)
        acc.spend(m)
        out.append(m)
        return m

    for name, t in tables:
        release(t, name)

    if hierarchy is not None:
        hs = hierarchy
        nl = len(hs.res)
        kids: list[dict[str, list[str]]] = []   # public parent -> children maps
        for k in range(nl - 1):
            d: dict[str, set] = {}
            for c in chains.values():
                d.setdefault(c[k], set()).add(c[k + 1])
            kids.append({p: sorted(v) for p, v in d.items()})

        def cat_at(node, k):
            return cat.get(node, "unknown") if k == nl - 1 else "unknown"

        def spec_at(k):
            def f(r):
                d = {}
                for p, ro in roles_cache[r.user].items():
                    c = chains.get(p)
                    if c is not None:
                        d[(c[k], cat_at(c[k], k), ro)] = 1.0
                return d
            return f

        vocab_k = [(n, "unknown", ro) for n in nodes[0] for ro in S.ROLES]
        r6_true = r6_keys = None
        for k, nm in enumerate(hs.tier_names()):
            if vocab_override is not None:
                vocab_k = [tuple(x) for x in vocab_override.get(nm, [])]
            # an empty vocabulary is still "released" (zero keys) so its planned rho is charged:
            # the choice to skip is a function of the noisy parent output, and reserving is conservative
            t = build_count_table(records, spec_at(k), vocab_k, "presence", max_keys=hs.max_keys[k])
            m = release(t, nm)
            if k == 0:
                r6_true, r6_keys = t.true_values, list(t.vocab)
            if k < nl - 1:
                # child vocabulary = f(noisy parent release only): public given that output
                kept = threshold_release(m, hs.tau)
                vocab_k = [(c, cat_at(c, k + 1), key[2]) for key in kept.keys
                           for c in kids[k][key[0]]]
        if r6_true is not None:
            edges = S.WORK_HIST_EDGES
            hist = np.zeros(len(edges) - 1)
            for key, v in zip(r6_keys, r6_true):
                if key[2] == "work":
                    hist[S.bin_of(float(v), edges)] += 1
            # adding/removing one user moves one coarse-cell count by 1 (a user has one work place),
            # i.e. two histogram bins by 1 each => L2 sensitivity sqrt(2) (max_keys truncation can
            # only remove that user's contribution, never add another)
            release(_FixedTable(S.bin_keys(edges), hist, math.sqrt(2.0)), S.WORK_HIST,
                    l2=math.sqrt(2.0))
    return out, acc.ledger(delta=delta, caps=caps)


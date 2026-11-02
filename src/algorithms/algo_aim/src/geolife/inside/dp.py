"""User-level differential privacy core (numpy only, deterministic given a seeded Generator).

Adjacency: two datasets are neighbours if one is obtained from the other by ADDING or REMOVING
all records of a single user (user-level add/remove adjacency). Every sensitivity below is the
exact worst-case L2 change of the true table under that relation: removing a user subtracts
that user's (bounded) vector, so ||delta||_2 = ||v_user||_2 <= stated bound.

Pipeline: cap_user_records (contribution bounding, per-user only) -> build_count_table
(per-key clipping + max keys per user => exact L2 bound) -> gaussian_measure (zCDP rho =
sens^2 / (2 sigma^2)) -> post-processing (non-negativity, noisy-value thresholding)
-> ZcdpAccountant / PrivacyLedger.

Public vocabulary: keys of a table MUST be fixed independently of the private data (public map
ids, or keys chosen from previously released noisy output). Noise is added to EVERY vocabulary
key, including those with true count 0; otherwise the support would leak.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from geolife.transfer.types import Measurement, PrivacyLedger, UserRecord

Key = tuple[str, ...]
KeySpec = Callable[[UserRecord], dict[Key, float]]

# ------------------------------------------------------------------ contribution bounding


@dataclass
class CapReport:
    users: int = 0
    stays_before: int = 0
    stays_after: int = 0
    trips_before: int = 0
    trips_after: int = 0
    users_truncated: int = 0
    caps: dict[str, int] = field(default_factory=dict)

    @property
    def stays_dropped_frac(self) -> float:
        return 1 - self.stays_after / self.stays_before if self.stays_before else 0.0

    @property
    def trips_dropped_frac(self) -> float:
        return 1 - self.trips_after / self.trips_before if self.trips_before else 0.0


def cap_user_records(records: Sequence[UserRecord], max_stays: int, max_trips: int,
                     max_distinct_places: int,
                     rng: np.random.Generator) -> tuple[list[UserRecord], CapReport]:
    """Deterministic per-user contribution bounding; each user's output depends only on that user.

    1. keep the `max_distinct_places` places with largest total dwell (ties: place_id order);
       drop stays elsewhere and trips whose origin or dest is dropped;
    2. keep the `max_stays` longest-dwell stays (ties broken by a seeded random permutation of
       that user's own stays, which does not use other users' data);
    3. keep the `max_trips` longest-duration trips (same tie rule); output chronological.
    Inputs are not mutated. Returns (capped records, CapReport).
    """
    rep = CapReport(users=len(records), caps={"max_stays_per_user": max_stays,
                                              "max_trips_per_user": max_trips,
                                              "max_distinct_places_per_user": max_distinct_places})
    out: list[UserRecord] = []
    for r in records:
        rep.stays_before += len(r.stays)
        rep.trips_before += len(r.trips)
        dwell: dict[str, int] = {}
        for s in r.stays:
            dwell[s.place_id] = dwell.get(s.place_id, 0) + max(0, s.t_end - s.t_start)
        keep_places = set(sorted(dwell, key=lambda p: (-dwell[p], p))[:max_distinct_places])
        stays = [s for s in r.stays if s.place_id in keep_places]
        trips = [t for t in r.trips if t.origin in keep_places and t.dest in keep_places]
        stays = _top(stays, max_stays, lambda s: s.t_end - s.t_start, lambda s: s.t_start, rng)
        trips = _top(trips, max_trips, lambda t: t.t_end - t.t_start, lambda t: t.t_start, rng)
        if len(stays) < len(r.stays) or len(trips) < len(r.trips):
            rep.users_truncated += 1
        rep.stays_after += len(stays)
        rep.trips_after += len(trips)
        out.append(UserRecord(user=r.user, stays=stays, trips=trips))
    return out, rep


def _top(items: list, k: int, score: Callable, chrono: Callable, rng: np.random.Generator) -> list:
    if len(items) <= k:
        return sorted(items, key=chrono)
    perm = rng.permutation(len(items))
    order = sorted(range(len(items)), key=lambda i: (-score(items[i]), perm[i]))[:k]
    return sorted((items[i] for i in order), key=chrono)


# ------------------------------------------------------------------ count tables

Kind = Literal["presence", "dwell", "transition"]


@dataclass
class CountTable:
    """Per-user clipped sparse vectors over a public vocabulary, plus their sum.

    user_vectors[i] maps vocabulary index -> clipped value (zeros omitted).
    l2_sensitivity is the exact worst-case ||v_user||_2 given the clipping parameters:
        presence:   per-key value 1, at most m keys           -> sqrt(m)
        dwell:      per-key value <= c (seconds), <= m keys   -> c * sqrt(m)
        transition: per-key count  <= c, <= m keys            -> c * sqrt(m)
    (sum of at most m squares each <= c^2). `observed_max_norm` is the actual max norm (<= bound).
    """
    kind: str
    vocab: list[Key]
    user_vectors: list[dict[int, float]]
    l2_sensitivity: float
    per_key_clip: float
    max_keys: int
    dropped_out_of_vocab: int = 0

    @property
    def true_values(self) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for uv in self.user_vectors:
            for j, x in uv.items():
                v[j] += x
        return v

    @property
    def observed_max_norm(self) -> float:
        return max((math.sqrt(sum(x * x for x in uv.values())) for uv in self.user_vectors),
                   default=0.0)


def build_count_table(records: Sequence[UserRecord], keyspec: KeySpec, vocab: Iterable[Key],
                      kind: Kind = "presence", max_keys: int = 10,
                      per_key_clip: float | None = None) -> CountTable:
    """Build clipped per-user vectors.

    keyspec(record) -> {key: raw_value}; raw values are e.g. visit counts, dwell seconds or
    transition counts. Keys outside the PUBLIC `vocab` are discarded. Per user:
      - presence: every key with raw value > 0 becomes 1 (per_key_clip forced to 1);
      - dwell / transition: raw value clipped to [0, per_key_clip] (required for these kinds);
      - keep only the `max_keys` keys with the largest clipped value (ties: key order), a rule
        that uses only that user's data.
    """
    vocab = list(vocab)
    idx = {k: i for i, k in enumerate(vocab)}
    if len(idx) != len(vocab):
        raise ValueError("vocab has duplicate keys")
    if kind == "presence":
        clip = 1.0
    else:
        if per_key_clip is None or per_key_clip <= 0:
            raise ValueError(f"kind={kind} requires per_key_clip > 0")
        clip = float(per_key_clip)
    if max_keys < 1:
        raise ValueError("max_keys must be >= 1")
    vecs: list[dict[int, float]] = []
    oov = 0
    for r in records:
        cand: dict[Key, float] = {}
        for k, raw in keyspec(r).items():
            if k not in idx:
                oov += 1
                continue
            if kind == "presence":
                val = 1.0 if raw > 0 else 0.0
            else:
                val = min(max(float(raw), 0.0), clip)
            if val > 0:
                cand[k] = val
        top = sorted(cand, key=lambda k: (-cand[k], k))[:max_keys]
        vecs.append({idx[k]: cand[k] for k in top})
    return CountTable(kind, vocab, vecs, clip * math.sqrt(max_keys), clip, max_keys, oov)


# ---- common keyspecs ----

def presence_by_place(r: UserRecord) -> dict[Key, float]:
    return {(s.place_id,): 1.0 for s in r.stays}


def dwell_by_place(r: UserRecord) -> dict[Key, float]:
    d: dict[Key, float] = {}
    for s in r.stays:
        d[(s.place_id,)] = d.get((s.place_id,), 0.0) + max(0, s.t_end - s.t_start)
    return d


def transitions(r: UserRecord) -> dict[Key, float]:
    d: dict[Key, float] = {}
    for t in r.trips:
        d[(t.origin, t.dest)] = d.get((t.origin, t.dest), 0.0) + 1.0
    return d


# ------------------------------------------------------------------ Gaussian mechanism


def rho_of(l2_sensitivity: float, sigma: float) -> float:
    return l2_sensitivity ** 2 / (2 * sigma ** 2)


def gaussian_measure(table: CountTable, l2_sensitivity: float | None = None,
                     sigma: float | None = None, rho: float | None = None,
                     rng: np.random.Generator | None = None, name: str = "measure") -> Measurement:
    """Release table sums + N(0, sigma^2) on EVERY vocabulary key. zCDP rho = sens^2/(2 sigma^2).

    Give exactly one of sigma / rho (rho => sigma = sens / sqrt(2 rho)). `l2_sensitivity` defaults
    to the table's exact bound and may not be set below it (that would void the guarantee).
    """
    if rng is None:
        raise ValueError("rng required")
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("name must match [a-z0-9_]+")
    sens = table.l2_sensitivity if l2_sensitivity is None else float(l2_sensitivity)
    if sens < table.l2_sensitivity - 1e-12:
        raise ValueError("l2_sensitivity below the table's clipping bound")
    if (sigma is None) == (rho is None):
        raise ValueError("give exactly one of sigma or rho")
    if sigma is None:
        if rho <= 0:
            raise ValueError("rho must be > 0")
        sigma = sens / math.sqrt(2 * rho)
    elif sigma <= 0:
        raise ValueError("sigma must be > 0")
    noisy = table.true_values + rng.normal(0.0, sigma, size=len(table.vocab))
    return Measurement(name=name, keys=list(table.vocab), values=[float(x) for x in noisy],
                       sigma=float(sigma), l2_sensitivity=sens, rho=rho_of(sens, sigma),
                       notes=f"kind={table.kind}; per_key_clip={table.per_key_clip}; "
                             f"max_keys={table.max_keys}")


# ---- post-processing (free: functions of the noisy output only) ----

def project_nonneg(m: Measurement, name: str | None = None) -> Measurement:
    """Clamp negatives to 0. Post-processing; rho unchanged."""
    return Measurement(name or m.name, list(m.keys), [max(0.0, v) for v in m.values], m.sigma,
                       m.l2_sensitivity, m.rho, m.notes + "; nonneg")


def threshold_release(m: Measurement, tau: float, name: str | None = None) -> Measurement:
    """Drop keys whose NOISY value < tau*sigma. Post-processing, so no extra privacy cost.

    Thresholding on TRUE values (e.g. 'drop places with fewer than k real users') is NOT
    allowed: the surviving key set would itself leak membership. This function only ever
    reads m.values, which are already noisy.
    """
    cut = tau * m.sigma
    keep = [i for i, v in enumerate(m.values) if v >= cut]
    return Measurement(name or m.name, [m.keys[i] for i in keep], [m.values[i] for i in keep],
                       m.sigma, m.l2_sensitivity, m.rho, m.notes + f"; thresholded tau={tau}")


# ---- hierarchical release ----

def hierarchical_release(records: Sequence[UserRecord], parent_keyspec: KeySpec,
                         child_keyspec: KeySpec, children: dict[Key, list[Key]],
                         kind: Kind = "presence", parent_params: dict | None = None,
                         child_params: dict | None = None, tau: float = 3.0,
                         rho_parent: float = 0.1, rho_child: float = 0.1,
                         rng: np.random.Generator | None = None,
                         name: str = "hier") -> tuple[Measurement, Measurement, Measurement]:
    """Level 1: all parents (public vocab = children.keys()). Level 2: children only under parents
    whose NOISY count >= tau*sigma. Returns (parents_full, parents_kept, children_measurement).

    Privacy: the child vocabulary is a function of the released noisy parent output only, hence
    public given that output; total cost is rho_parent + rho_child (sequential composition;
    spend parents_full and children_measurement, NOT parents_kept, in the accountant).
    `children` maps parent key -> child keys and is public (e.g. H3 cell -> map places).
    parent_params / child_params are kwargs for build_count_table (max_keys, per_key_clip).
    """
    parents = list(children)
    pt = build_count_table(records, parent_keyspec, parents, kind, **(parent_params or {}))
    pm = gaussian_measure(pt, rho=rho_parent, rng=rng, name=f"{name}_l1")
    kept = threshold_release(pm, tau, name=f"{name}_l1_kept")
    kids = [c for p in kept.keys for c in children[p]]
    ct = build_count_table(records, child_keyspec, kids, kind, **(child_params or {}))
    cm = gaussian_measure(ct, rho=rho_child, rng=rng, name=f"{name}_l2")
    return pm, kept, cm


# ------------------------------------------------------------------ accountant


def zcdp_to_epsilon(rho: float, delta: float) -> float:
    """(eps, delta)-DP from rho-zCDP: eps = rho + 2 sqrt(rho ln(1/delta))."""
    if rho <= 0:
        return 0.0
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def split_budget(rho_total: float, weights: Sequence[float]) -> list[float]:
    w = np.asarray(weights, dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    return [float(rho_total * x / w.sum()) for x in w]


class ZcdpAccountant:
    """Sequential composition of zCDP: rho adds up. Adjacency: add/remove one user."""

    def __init__(self, rho_budget: float | None = None):
        self.rho_budget = rho_budget
        self.entries: list[dict] = []

    @property
    def rho_total(self) -> float:
        return float(sum(e["rho"] for e in self.entries))

    def spend(self, m: Measurement) -> None:
        if self.rho_budget is not None and self.rho_total + m.rho > self.rho_budget * (1 + 1e-9):
            raise ValueError(f"budget exceeded: {self.rho_total + m.rho} > {self.rho_budget}")
        self.entries.append({"name": m.name, "rho": m.rho, "sigma": m.sigma,
                             "l2_sensitivity": m.l2_sensitivity, "n_keys": len(m.keys),
                             "notes": m.notes})

    def to_epsilon(self, delta: float) -> float:
        return zcdp_to_epsilon(self.rho_total, delta)

    def ledger(self, delta: float = 1e-6, caps: dict[str, float] | None = None,
               caveats: list[str] | None = None) -> PrivacyLedger:
        base = ["Guarantee covers only the Gaussian measurements listed in entries.",
                "Public vocabulary (map ids, H3 cells) is assumed data-independent.",
                "Contribution capping is per-user deterministic, so it is covered by the "
                "clipping bound; stay-point/map-matching steps run per user and add no leakage "
                "beyond that bound if their parameters are fixed in advance.",
                "Hyperparameters (caps, sigma, tau) were not tuned on private data unless stated; "
                "any such tuning is NOT accounted.",
                "Floating-point Gaussian sampling is not hardened against floating-point attacks."]
        return PrivacyLedger(adjacency="user", rho_total=self.rho_total,
                             epsilon=self.to_epsilon(delta), delta=delta, caps=dict(caps or {}),
                             entries=[dict(e) for e in self.entries],
                             caveats=base + list(caveats or []))

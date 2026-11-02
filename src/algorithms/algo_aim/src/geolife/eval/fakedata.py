"""Fake UserRecords from a known toy 'source city' (for tests / smoke runs; no real data)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from geolife.transfer.types import Stay, Trip, UserRecord

BASE = 1704067200  # Monday 00:00 UTC


@dataclass
class FakeWorld:
    records: list[UserRecord]
    place_vocab: list[tuple[str, str]]
    xy: dict[str, tuple[float, float]]
    straight: dict[tuple[str, str], float]
    detour: dict[tuple[str, str, int], float]
    extent_m: float

    def detour_fn(self, t: Trip):
        return self.detour.get((t.origin, t.dest, t.t_start))


def make_fake_world(n_users: int = 400, n_days: int = 10, seed: int = 1, extent: float = 8000.0,
                    n_work: int = 20, n_other: int = 100, zipf_s: float = 1.0) -> FakeWorld:
    rng = np.random.default_rng(seed)
    xy: dict[str, tuple[float, float]] = {}
    vocab: list[tuple[str, str]] = []
    homes = [f"h{i:03d}" for i in range(n_users)]
    works = [f"w{i:02d}" for i in range(n_work)]
    others = [f"o{i:03d}" for i in range(n_other)]
    for p, c, core in ([(h, "residential", 0.3) for h in homes] + [(w, "office", 0.7) for w in works]
                       + [(o, rng.choice(["shop", "food", "leisure"]), 0.5) for o in others]):
        pt = rng.normal(extent / 2, extent / 6, 2) if rng.random() < core else rng.uniform(0, extent, 2)
        xy[p] = tuple(np.clip(pt, 0, extent))
        vocab.append((p, str(c)))
    wsize = np.arange(1, n_work + 1) ** -0.6
    ow = np.arange(1, n_other + 1) ** -zipf_s
    ow = ow[rng.permutation(n_other)]
    pmf = np.array([0, .2, .3, .3, .2])
    straight: dict = {}
    detour: dict = {}

    def trip(u_trips, a, b, t0, t1):
        d = float(np.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])) * 1.27
        straight[(a, b)] = max(d, 1.0)
        ratio = 1.0 + rng.exponential(0.15)
        detour[(a, b, t0)] = ratio
        u_trips.append(Trip(a, b, t0, t1, "car", (), max(d, 1.0) * ratio))

    recs = []
    for i in range(n_users):
        home = homes[i]
        work = works[int(rng.choice(n_work, p=wsize / wsize.sum()))] if rng.random() < 0.8 else None
        a = int(rng.choice(len(pmf), p=pmf))
        hx, hy = xy[home]
        d = np.maximum(np.hypot(*(np.array([xy[o] for o in others]) - [hx, hy]).T), 200.0)
        pr = ow * d ** -0.8
        mine = [others[j] for j in rng.choice(n_other, size=a, replace=False, p=pr / pr.sum())]
        stays: list[Stay] = []
        trips: list[Trip] = []

        def stay(p, c, t0, t1):
            stays.append(Stay(p, c, "c0", int(t0), int(t1)))
        cat = dict(vocab)
        for day in range(n_days):
            t = BASE + day * 86400
            seq = [(home, t, t + 8 * 3600)]
            weekday = day % 7 < 5
            if work and weekday:
                w0 = t + 9 * 3600 + rng.uniform(0, 1800)
                seq.append((work, w0, w0 + rng.uniform(7, 8.5) * 3600))
            tcur = seq[-1][2] + 1800
            if mine and rng.random() < (0.4 if weekday else 0.6):
                o = mine[min(len(mine) - 1, int(rng.geometric(0.5)) - 1)]
                tcur = max(tcur, t + 11 * 3600 if not weekday else t + 18 * 3600)
                seq.append((o, tcur, tcur + rng.uniform(0.5, 2) * 3600))
                tcur = seq[-1][2] + 1200
            seq.append((home, max(tcur, t + 20 * 3600 if weekday else tcur), t + 86400))
            for k, (p, a0, a1) in enumerate(seq):
                stay(p, cat[p], a0, a1)
                if k:
                    trip(trips, seq[k - 1][0], p, int(seq[k - 1][2]), int(a0))
        recs.append(UserRecord(user=f"x{i:03d}", stays=stays, trips=trips))
    return FakeWorld(recs, vocab, xy, straight, detour, extent)


@lru_cache(maxsize=None)
def cached_fake_artifact(rho_total: float = 40.0):
    """(artifact_dir, world, measurements, ledger); built once per process."""
    import tempfile
    d = tempfile.mkdtemp(prefix="fake_artifact_")
    w = make_fake_world()
    ms, led = save_fake_artifact(d, w, rho_total=rho_total, generation={"n_days": 4})
    return d, w, ms, led


def save_fake_artifact(path, world: FakeWorld, rho_total: float = 40.0, seed: int = 5,
                       generation: dict | None = None):
    """records -> DP measurements -> artifact on disk. Returns (measurements, ledger)."""
    from geolife.transfer.artifact import save_artifact
    from geolife.transfer.types import ArtifactManifest
    from geolife.inside.measure import build_behaviour_measurements
    rng = np.random.default_rng(seed)
    ms, led = build_behaviour_measurements(world.records, world.place_vocab, rho_total, rng,
                                           detour_fn=world.detour_fn)
    save_artifact(path, ArtifactManifest(created="test"),
                  {"categories": sorted({c for _, c in world.place_vocab})},
                  {}, ms, led, {}, None, {"extent_m": world.extent_m}, {}, generation or {})
    return ms, led

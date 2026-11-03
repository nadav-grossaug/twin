"""Fixture generator of UserOutings for the v2 learner tests (no real data)."""
import h3
import numpy as np

from geolife.transfer.types import Outing, UserOutings, Visit

BBOX = (39.85, 40.00, 116.25, 116.50)
CATS = ["food", "shop", "office", "leisure", "unknown"]
T0 = 1_230_000_000   # 2008-12-23 (UTC)


def make_users(n=40, seed=0, pool=150, max_outings=250):
    rng = np.random.default_rng(seed)
    lat0, lat1, lon0, lon1 = BBOX
    pool_cells = sorted({h3.latlng_to_cell(rng.uniform(lat0, lat1), rng.uniform(lon0, lon1), 8)
                         for _ in range(pool * 2)})[:pool]
    cat_of = {c: CATS[i % len(CATS)] for i, c in enumerate(pool_cells)}
    w = 1.0 / np.arange(1, len(pool_cells) + 1) ** 1.1
    w /= w.sum()
    users = []
    for u in range(n):
        n_out = int(min(max_outings, max(10, rng.lognormal(3.4, 0.9))))
        fav = rng.choice(len(pool_cells), size=min(len(pool_cells), 5 + u % 30), replace=False, p=w)
        outs = []
        t = T0 + int(rng.integers(0, 40)) * 86400
        for _ in range(n_out):
            t += int(rng.integers(3 * 3600, 3 * 86400))
            dur = int(rng.lognormal(7.8, 0.8))
            k = int(rng.choice([0, 1, 2, 3, 12], p=[.4, .25, .15, .15, .05]))
            cells = [pool_cells[i] for i in rng.choice(fav, size=k + 2)]
            vis = [Visit("h3:" + c, cat_of[c], c, t + 60 * j, 700 if 0 < j <= k else 0,
                         "stop" if 0 < j <= k else ("start" if j == 0 else "end"))
                   for j, c in enumerate(cells)]
            trav = set()
            for c in cells:
                c9 = h3.cell_to_center_child(c, 9)
                trav |= set(h3.grid_disk(c9, int(rng.integers(0, 4))))
            outs.append(Outing(t, t + dur, str(rng.choice(["walk", "bus", "unknown", "unknown"])),
                               float(rng.lognormal(8.8, 1.0)), tuple(vis), tuple(sorted(trav))))
        users.append(UserOutings(f"p{u:04d}", outs))
    return users

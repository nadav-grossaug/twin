"""A synthetic Manhattan-grid city implementing the MapProvider protocol (tests, toy runs, M harness)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from geolife.transfer.maps import MapPlace, Route


CATEGORIES = (("residential", 0.55), ("office", 0.12), ("shop", 0.12), ("food", 0.08),
              ("leisure", 0.06), ("school", 0.04), ("health", 0.03))


class GridCityProvider:
    """n x n Manhattan grid with jittered edge lengths, places snapped to nearby nodes."""

    def __init__(self, n: int = 14, spacing: float = 250.0, n_places: int = 900, seed: int = 0,
                 origin: tuple[float, float] = (39.9, 116.4)):
        rng = np.random.default_rng(seed)
        self.n, self.spacing, self.origin = n, spacing, origin
        k = n * n
        self.node_xy = np.array([(j * spacing, i * spacing) for i in range(n) for j in range(n)], float)
        rows, cols, w = [], [], []
        self._seg: dict[tuple[int, int], str] = {}
        for i in range(n):
            for j in range(n):
                a = i * n + j
                for b in ((a + 1) if j + 1 < n else None, (a + n) if i + 1 < n else None):
                    if b is None:
                        continue
                    length = spacing * (1 + 0.2 * rng.random())
                    rows += [a, b]; cols += [b, a]; w += [length, length]
                    self._seg[(a, b)] = self._seg[(b, a)] = f"seg{a}_{b}"
        g = coo_matrix((w, (rows, cols)), shape=(k, k)).tocsr()
        self._D, self._P = dijkstra(g, directed=False, return_predecessors=True)
        names, probs = zip(*CATEGORIES)
        cats = rng.choice(len(names), size=n_places, p=np.array(probs) / sum(probs))
        nodes = rng.integers(0, k, size=n_places)
        off = rng.uniform(-60, 60, size=(n_places, 2))
        pop = np.where(np.array(names)[cats] == "residential", 1.0,
                       1.0 + rng.pareto(1.2, size=n_places))
        self._places = [MapPlace(f"p{i:04d}", names[cats[i]], float(pop[i]),
                                 float(self.node_xy[nodes[i], 0] + off[i, 0]),
                                 float(self.node_xy[nodes[i], 1] + off[i, 1]))
                        for i in range(n_places)]
        self._node = {p.id: int(nodes[i]) for i, p in enumerate(self._places)}
        self._byid = {p.id: p for p in self._places}

    def places(self) -> Sequence[MapPlace]:
        return self._places

    def route(self, a: str, b: str) -> Route:
        pa, pb = self._byid[a], self._byid[b]
        na, nb = self._node[a], self._node[b]
        path = [nb]
        while path[-1] != na:
            path.append(int(self._P[na, path[-1]]))
        path.reverse()
        segs = tuple(self._seg[(path[i], path[i + 1])] for i in range(len(path) - 1))
        poly = np.vstack([[pa.x, pa.y], self.node_xy[path], [pb.x, pb.y]])
        length = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        return Route(segs, length, poly)

    def to_latlon(self, x, y):
        lat0, lon0 = self.origin
        x, y = np.asarray(x, float), np.asarray(y, float)
        lat = lat0 + y / 111_320.0
        lon = lon0 + x / (111_320.0 * np.cos(np.radians(lat0)))
        return lat, lon

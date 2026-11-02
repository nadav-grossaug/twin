"""MapProvider protocol (target-city map as seen by the generator): MapPlace, Route, MapProvider.

The synthetic grid city (outside/grid_city.py) and the Overture-backed provider (outside/map/provider.py) implement it. Coordinates handed to the
generator are local metric (x, y) in metres; `to_latlon` converts for GPS emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class MapPlace:
    id: str
    category: str
    weight: float          # popularity weight (>0)
    x: float
    y: float


@dataclass
class Route:
    segments: tuple[str, ...]
    length_m: float
    polyline: np.ndarray    # (n, 2) metric xy, from origin place to destination place


class MapProvider(Protocol):
    def places(self) -> Sequence[MapPlace]: ...

    def route(self, a: str, b: str) -> Route:
        """Shortest road path between two places (by id)."""
        ...

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised metric xy -> (lat, lon) WGS-84."""
        ...

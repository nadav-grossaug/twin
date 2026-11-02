"""artifact (dir or object) + MapProvider -> synthetic users. Convenience wrapper over L5-L7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geolife.transfer.artifact import Artifact, load_artifact
from geolife.outside.fit import fit_behaviour
from geolife.transfer.maps import MapProvider
from geolife.outside.population import Population, sample_population
from geolife.outside.render import RenderResult, render
from geolife.outside.teleport import TeleportResult, ground_places


@dataclass
class Generated:
    params: dict
    population: Population
    teleport: TeleportResult
    result: RenderResult


def generate(artifact: Artifact | str | Path, provider: MapProvider, n_users: int, seed: int,
             extent_m: float | None = None) -> Generated:
    art = artifact if isinstance(artifact, Artifact) else load_artifact(artifact)
    gen = art.generation or {}
    ext = extent_m or float(art.grounding.get("extent_m", gen.get("extent_m", 10000.0)))
    params = fit_behaviour(art.measurements, art.ledger, extent_m=ext)
    pop = sample_population(params, n_users, seed, gen)
    tp = ground_places(pop, provider, art.grounding, seed)
    res = render(pop, tp.mapping, provider, gen, seed)
    return Generated(params, pop, tp, res)

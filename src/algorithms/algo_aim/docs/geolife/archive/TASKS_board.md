# GeoLife implementation board

Orchestrator: main session, branch `feat/geolife-impl` (worktree `.claude/worktrees/geolife-orch`).
Agents work in their own worktrees on `worktree-agent-*` branches; orchestrator merges `--no-ff`.
Contract: `src/geolife/transfer/types.py` (orchestrator-owned). Env: `C:/Users/NaomiD/geo-venv` (light; numpy pandas scipy pytest pyarrow shapely networkx matplotlib).
Run tests: `C:/Users/NaomiD/geo-venv/Scripts/python.exe -m pytest tests/geolife` with `PYTHONPATH=src`.

| Wave | Task | Scope (files) | Status |
|---|---|---|---|
| 1 | P0 Overture/routing feasibility | `docs/geolife/p0-feasibility.md`, `src/geolife/transfer/map/ + src/geolife/outside/map/` | merged |
| 1 | P1 GeoLife loader + EDA | `src/geolife/inside/data.py`, `geolife/` | merged (data at agent-a04900ec477b059b7 worktree `geolife/inside/data/`, gitignored) |
| 1 | DP core | `src/geolife/inside/dp.py` | merged |
| 1 | Artifact I/O + inspector | `src/geolife/transfer/artifact.py`, `inspect.py` | merged |
| 2 | GEN: fit, population, teleport, render (toy map) | `src/geolife/outside/` | merged |
| 2 | TOK: stays, trips, matching, learner | `src/geolife/transfer/tokenize/`, `learn.py` | running |
| 2 | MAP: Overture MapProvider + place categories | `src/geolife/outside/map/provider.py` | running |
| 3 | Method M + transfer evaluation | needs wave 2 | pending |

Findings: stay-level overlap is weak (res 8: 63 of 1,236 cells have >=10 users; point-level is strong). Matcher on real graph recovers 19/27 segments with 33 extras. Contract gaps: ledger `entries` keys (name, rho, sigma, l2_sensitivity) undeclared; `Artifact` container lives in artifact.py.

## Decisions (user, 2026-11-01)
- **D1 place tier:** hierarchical release (district-scale H3 res 6 first, refine only under parents whose noisy count passes) + weighted rho split instead of equal split. Keep user-level, epsilon ~6. Choosing resolution after seeing SNR is unaccounted tuning: state it in the ledger caveats, or fix the hierarchy from public considerations.
- **D2 relation for M:** shared anchor (two users share a home/work-type place). Co-presence is out of scope for now.

## Wave 3 (running)
| Task | Scope | Status |
|---|---|---|
| LEARN2: hierarchical place tier + weighted budget | `learn.py`, `generate/measure.py`, `generate/fit.py` | running |
| M + transfer evaluation (synthetic only) | `src/geolife/eval/m/`, `geolife/reports/` | running |

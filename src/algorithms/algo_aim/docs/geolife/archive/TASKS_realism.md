# Realism round board (orchestrator: main session, branch `feat/geolife-realism`)

Plan: `PLAN.md` section 9. Agents work in their own worktrees (reset to `feat/geolife-realism` first), orchestrator merges `--no-ff`.
Tests: `PYTHONPATH=src:. C:/Users/NaomiD/geo-venv/Scripts/python.exe -m pytest -q tests/geolife`.

| Task | Plan step | Files | Status |
|---|---|---|---|
| TIME3: 3 time bins | 2 | `transfer/schema_v2`, `inside/measure_v2`, `outside/fit_v2`, ledger caveats, artifacts | running |
| SENSOR: DP sensor tables + render from them | 1 | new `inside/sensor.py`, `transfer/schema_sensor.py`, `outside/render_v2`, fit hook | running |
| VIZ: real vs synthetic comparison suite | 3 | `eval/viz.py`, `eval/compare.py`, notebook, stage 2 source-city option | running |
| then: steps 4-5, 6, 7, 8 | | | pending |

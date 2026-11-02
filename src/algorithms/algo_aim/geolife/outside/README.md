# geolife/outside

Never sees real data. `generate_synthetic.py` (STAGE 2) turns `../transfer/artifacts/*` into `synthetic/<variant>_seed<k>/`
(`traces.parquet`, `outings.json`, `truth.json`, `meta.json`; the truth is the generator's own planted structure).
Run it with only `src` on the path:
`PYTHONPATH=src python geolife/outside/generate_synthetic.py`.

`cache/` holds the **public** map (Overture parquet) used as the target city; inside may read it (places table for
tokenization), it is gitignored. `fetch_public_map.py` fills it (needs network and duckdb; the only script that does, and
the boundary test bars it from importing inside or eval). `reports/` has the generator write-up (`generator-v2.md`) and
the teleport demo (`teleport_demo.py`, `teleport-demo.png`). Code: `src/geolife/outside/`.

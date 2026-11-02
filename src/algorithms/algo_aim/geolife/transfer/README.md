# geolife/transfer

The only folder both sides read. `artifacts/<variant>_seed<k>/` holds what stage 1 (inside) writes and stage 2 (outside)
reads: `manifest.json` (SHA-256 of every file), `schema.json`, `preprocessing.json`, `privacy.json` (zCDP ledger with
caveats), `generation.json`, `grounding.json`, `diagnostics.json`, `model/behaviour/*.{parquet|csv,meta.json}` (the noisy
tables) and `model/behaviour_summaries.json`. Nothing here is a record, a coordinate of a real user or an un-noised count.

Seed-0 artifacts are committed, other seeds are regenerated. Files are byte-exact (`.gitattributes`: `-text`).
The code that defines the format is `src/geolife/transfer/` (`artifact.py`, `types.py`, `schema*.py`, `inspect.py`).
Look at one: `PYTHONPATH=src python -m geolife.transfer.artifact_cli inspect geolife/transfer/artifacts/<dir>`.

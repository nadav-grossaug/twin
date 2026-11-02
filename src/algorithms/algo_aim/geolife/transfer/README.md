# geolife/transfer

The only folder both sides read. `artifacts/<variant>_seed<k>/` holds what stage 1 (inside) writes and stage 2 (outside)
reads: `manifest.json` (SHA-256 of every file), `schema.json`, `preprocessing.json`, `privacy.json` (zCDP ledger with
caveats), `generation.json`, `grounding.json`, `diagnostics.json`, `model/behaviour/*.{parquet|csv,meta.json}` (the noisy
tables) and `model/behaviour_summaries.json`. Nothing here is a record, a coordinate of a real user or an un-noised count.

Seed-0 artifacts are committed, other seeds are regenerated. Files are byte-exact (`.gitattributes`: `-text`).
The code that defines the format is `src/geolife/transfer/` (`artifact.py`, `types.py`, `schema*.py`, `inspect.py`).
Look at one: `PYTHONPATH=src python -m geolife.transfer.artifact_cli inspect geolife/transfer/artifacts/<dir>`.

The committed `outings_v2_seed0-2` are a DEV set: three noise replicates of one fit. A real release publishes exactly
one seed (`fit_artifacts.py --release`); averaging k seeds cuts the noise std by sqrt(k) and honestly costs k * rho
(measured in `geolife/eval/reports/privacy-attacks.md`).

Authenticity (optional): `manifest.hmac.json` is an HMAC-SHA256 over `manifest.json` (which hashes every other file).
Sign inside (`artifact_cli sign <dir> --key F` or `fit_artifacts.py --sign-key F`), verify outside
(`artifact_cli verify <dir> --key F`, `generate_synthetic.py --key F`). Shared-key authenticity, NOT a public-key
signature: the key is generated inside the air gap and carried to the verifier over the same trusted channel that
moves this code; never store it next to the artifact. Unsigned artifacts stay valid.

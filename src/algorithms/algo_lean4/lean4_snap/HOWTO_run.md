# HOWTO run — lean4_snap (enclave social-graph encoder + verified synthetic graphs)

Input: `facebook_full_features.graphml` (SNAP "Social circles: Facebook";
regenerate from the published SNAP files with
`cd input_data && python3 snap_facebook_to_graphml.py`).

## Install

1. **Python** ≥ 3.12 with: `duckdb`, `dbt-core` + `dbt-duckdb`, `torch`,
   `torch-geometric`, `networkx`, `pandas`, `numpy`, `scipy`, `marimo`,
   `pytest` (`pip install -r ../../../requirements.txt` or equivalents).
2. **Lean store** — nothing to do: `make` bootstraps elan + the pinned
   mathlib/SampCert package store into `LEAN_HOME=/opt/data/lean` (shared
   across the lean4_* projects; installed once, reused by every build,
   including bundle rebuilds). Override with `make LEAN_HOME=...`.

## The six flows

```bash
make help               # this summary
make train              # the entire enclave flow:
                        #   GraphML ingest -> dbt (features, motifs, chi2 assoc,
                        #   DP inputs; 124+ models/tests) -> PyG minimodels ->
                        #   ml scores -> Lean build + SQL<->Lean checks +
                        #   eps-DP release (SampCert discrete Laplace,
                        #   385/100 <= 4/1 exact-fraction ledger) + verified-
                        #   sampler smoke -> distill genspec -> pytest ->
                        #   run manifest -> policy export -> ONE source-only
                        #   bundle in dist/bundle/
make synth              # synthetic GraphML -> synth/synth_graph.graphml +
                        #   plan.json + meta.json, then round-trip validation
                        #   (re-ingest through the production pipeline;
                        #   planted hub/broker/butterfly recall, degree GOF,
                        #   clustering band)          [N_NODES=4039]
make synth_from_bundle  # rebuild the generator FROM dist/bundle's Lean
                        #   sources (kernel re-checks every theorem), distill
                        #   the bundle's release, synth -> synth_bundle/
make notebook-enclave   # serve the enclave EDA marimo notebook
make notebook-synth     # serve the synth-vs-real scorecard notebook
make clean              # remove ALL generated data (keeps sources, code and
                        #   the shared Lean store)
```

Extras: `make certify` (kernel-checked certificate over the synth plan, `by
decide`), `make realism-audit`, `make notebook-html`, `make test`.

## Notes

- **Deployment is source-only.** `dist/bundle/` ships Lean sources +
  `dp_release.json` + stdlib-only `distill.py`; the consumer's kernel
  re-checks every theorem before anything runs. No binaries, no model
  weights, no raw data. The generator's only inputs are Lean sources and
  the ε-DP release.
- Dev-host export (no enclave attestation): `SNAP_ALLOW_DEV_RELEASE=1 make
  train` (the release is then marked `attested: false`).
- `FLOW_DEVICE=cpu` forces deterministic CPU training; `FLOW_RETRAIN=1`
  activates a retrain despite content-addressed caching.
- DP noise is intentionally non-reproducible (urandom-backed SampCert FFI);
  rerun `make distill` after any fresh release.
- Privacy unit: one person's derived node row (add/remove); the neighbor-
  induced node-DP gap is DECLARED (goal.md, Enclave Boundary) — read it
  before quoting the ε.

# HOWTO run — enclave stock-series encoder + synthetic generator

One-time setup:

    make setup              # python deps (dbt-duckdb, torch, marimo, ...)
    make lean-build         # Lean toolchain via elan; mathlib cache; SampCert pin

Full pipeline (order matters — distill AFTER lean, DP draws are
non-reproducible):

    make all                # dbt train ml lean distill test manifest

Step by step:

    make dbt                # TSV -> staging -> features -> motifs -> DP inputs
    make train              # 5 minimodels, content-addressed (FLOW_RETRAIN=1
                            #   to activate a retrain; FLOW_DEVICE=cpu for
                            #   the deterministic mode)
    make ml                 # materialize ml_series_scores with the ACTIVE model
    make lean               # build + checks + DP release + verified sampling
    make distill            # released facts -> lean/out/genspec.json + fidelity
    make test               # pytest (spec sync, crosscheck, conformance, gate,
                            #   distill provenance, synth, minimodel gates)
    make manifest           # fingerprints -> artifacts/runs/<id>/

Synthetic data:

    make synth              # Lean generator -> synth/stock_synth.tsv (+ plan,
                            #   meta); validate = re-ingest through PRODUCTION
                            #   dbt into synth/scratch.duckdb + Lean deciders +
                            #   planted-pattern recall (exit 0/2 soft/1 hard)
    make certify            # kernel-checked GenCert (by decide) -> certificate.json
    make realism-audit      # enclave-only: production models judge the synth

Export (policy-gated, default deny):

    FLOW_ALLOW_DEV_RELEASE=1 make export   # -> release/<id>/ with manifest;
                                           #   dev export is marked attested: false

Off-enclave generation (another directory / machine, same OS+arch):

    make bundle                            # -> dist/bundle/ (generate binary +
                                           #   patched FFI dylib + dp_release.json
                                           #   + distill.py + README + MANIFEST)
    # then, anywhere (python3 stdlib only; no repo, no dbt, no torch):
    cp -R dist/bundle /elsewhere && cd /elsewhere
    shasum -a 256 -c MANIFEST.sha256
    python3 python/distill.py              # release -> lean/out/genspec.json
    ./bin/generate lean/out/genspec.json 8 168 out

Notebooks (validate headless):

    make notebook           # enclave EDA (marimo)
    make notebook-synth     # synth-vs-real scorecard
    make notebook-html      # static exports -> artifacts/reports/

Key invariants:

- The concept spec lives in `dbt/dbt_project.yml vars:`, mirrored by
  `python/spec.py` and `lean/EnclaveSeries/Params.lean`, locked by
  `tests/test_spec_sync.py`. Bump `spec_version` on any change.
- The ε ledger is EXACTLY 400/100 = 4/1 (see `config/release_policy.yaml`);
  `lake exe release` refuses over-budget exports with exit 2.
- Privacy unit: one ticker. n = 8 — expect noisy aux histograms; the
  generator's structure comes from declared public priors anchored to the
  released marginals (see `python/distill.py`).
- One pipeline driver at a time: concurrent `make` runs race over the DuckDB
  write lock and generated dirs that `clean*` deletes mid-flight.

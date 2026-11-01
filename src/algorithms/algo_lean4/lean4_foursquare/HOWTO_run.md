# HOWTO run — enclave travel-sequence encoder + synthetic generator

One-time setup:

    make setup              # python deps (dbt-duckdb, torch, marimo, ...)

Lean toolchain policy: `LEAN_HOME=/opt/data/lean` always. The Makefile
bootstraps it automatically (`lean-store` target): elan toolchains in
`$LEAN_HOME/elan`, the mathlib olean cache in `$LEAN_HOME/cache`, the pinned
package store (mathlib, SampCert @ a2c53bb) in `$LEAN_HOME/packages`,
symlinked into `lean/.lake/packages`. Installed once, reused by every build —
including bundle rebuilds. Nothing depends on `~/.elan`.

## The six flows

    make clean              # remove ALL generated data (keeps sources + the
                            #   shared Lean store)

    make train              # the ENTIRE enclave flow, in order:
                            #   dbt        TSV -> staging -> features ->
                            #              motifs -> episodes -> DP inputs
                            #   models     4 minimodels, content-addressed,
                            #              gated vs frozen baselines
                            #              (FLOW_RETRAIN=1 to force retrain;
                            #              FLOW_DEVICE=cpu deterministic)
                            #   ml         materialize ml_travel_scores
                            #   lean       build + checks + ε-DP release +
                            #              verified-sampler smoke
                            #   distill    released facts -> genspec.json
                            #   test       the full pytest suite
                            #   manifest   fingerprints -> artifacts/runs/
                            #   export     policy-filtered release/<ts>/
                            #              (FLOW_ALLOW_DEV_RELEASE=1 on a
                            #              non-enclave dev host)
                            #   bundle     ONE source-only bundle -> dist/bundle

    make synth              # Lean generator -> synth/travel_synth.tsv
                            #   (+ plan.json ground truth, meta.json
                            #   provenance); then validate: re-ingest through
                            #   the PRODUCTION dbt into a scratch DB, planted
                            #   corridor + tour recall must be 1.0, Lean
                            #   deciders re-verify (exit 0 / 2 soft / 1 hard)
                            #   N_USERS=5000 to change the synthetic user count

    make synth_from_bundle  # proves the bundle is self-contained: rebuild the
                            #   generator FROM dist/bundle's Lean sources (the
                            #   kernel re-checks every theorem), distill the
                            #   bundle's dp_release.json, generate ->
                            #   synth_bundle/. The ONLY synthesis inputs are
                            #   Lean sources + released JSON — no binaries.

    make notebook-enclave   # serve the enclave EDA notebook (marimo)
    make notebook-synth     # serve the synth-vs-real scorecard notebook

Optional extras: `make certify` (kernel-checked certificate over the synth
plan), `make realism-audit` (enclave-side scoring of the synthetic data),
`make notebook-html` (headless exports into artifacts/reports/).

## Order matters

`train` runs distill AFTER the Lean release because DP draws are
intentionally non-reproducible (urandom-backed FFI): the genspec is
hash-keyed to the release file. Re-running `make synth` after a fresh
`make train` always re-distills against the new release.

One pipeline driver at a time: concurrent `make` runs race over the DuckDB
write lock and the generated dirs that `clean` deletes.

## Where things land

    artifacts/enclave.duckdb   the in-enclave warehouse (never exported)
    models_out/                content-addressed minimodels + registry
    lean/data/                 enclave-side exports for the Lean layer
    lean/out/                  dp_release.json, genspec.json, lean_results.json
    release/<ts>/              the policy-filtered export (what may leave)
    dist/bundle/               the single source-only deployment bundle
    synth/, synth_bundle/      synthetic TSV + plan + meta + validation

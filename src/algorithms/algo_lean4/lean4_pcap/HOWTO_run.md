# HOWTO run — enclave flow encoder + synthetic generator

One-time setup:

    make setup              # python deps (dbt-duckdb, torch, marimo, geoip2fast, ...)
    make repair-data        # once per NEW Joy export: fixes known Cisco Joy
                           #   JSON serializer bugs in flow.jsonl (idempotent)

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
                            #   dbt        flow.jsonl -> staging -> features ->
                            #              motifs -> assoc (chi2) -> sequences
                            #              -> DP inputs (all dbt tests green)
                            #   models     5 minimodels, content-addressed,
                            #              gated vs frozen baselines
                            #              (FLOW_RETRAIN=1 to force retrain;
                            #              FLOW_DEVICE=cpu deterministic)
                            #   ml         materialize ml_flow_scores
                            #   lean       build + checks + eps-DP release +
                            #              verified-sampler smoke
                            #   distill    released facts -> genspec.json
                            #   test       the full pytest suite
                            #   manifest   fingerprints -> artifacts/runs/
                            #   export     policy-filtered release/<ts>/
                            #              (FLOW_ALLOW_DEV_RELEASE=1 on a
                            #              non-enclave dev host)
                            #   bundle     ONE source-only bundle -> dist/bundle

    make synth              # Lean generator -> synth/flow_synth.jsonl (Joy
                            #   schema; + plan.json ground truth, meta.json
                            #   provenance); then validate: re-ingest through
                            #   the PRODUCTION dbt into a scratch DB, planted
                            #   beacons recalled at 1.0, Lean deciders
                            #   re-verify (exit 0 / 2 soft / 1 hard)
                            #   SYNTH_HOSTS=500 to change the synthetic host
                            #   count (default 10,000; max 10,000)

    make synth_from_bundle  # proves the bundle is self-contained: rebuild the
                            #   generator FROM dist/bundle's Lean sources (the
                            #   kernel re-checks every theorem), distill the
                            #   bundle's dp_release.json, generate ->
                            #   synth_bundle/. The ONLY synthesis inputs are
                            #   Lean sources + released JSON — no binaries.

    make notebook-enclave   # serve the enclave EDA notebook (marimo)
    make notebook-synth     # serve the synth-vs-real scorecard notebook

`make help` prints this summary. Optional extras: `make certify`
(kernel-checked certificate over the synth plan), `make realism-audit`
(enclave-side scoring of the synthetic data), `make notebook-html` (headless
exports into artifacts/reports/).

## Order matters

`train` runs distill AFTER the Lean release because DP draws are intentionally
non-reproducible (urandom-backed FFI): the genspec is hash-keyed to the
release file. Re-running `make synth` after a fresh `make train` always
re-distills against the new release.

One pipeline driver at a time: concurrent `make` runs race over the DuckDB
write lock and the generated dirs that `clean` deletes.

## Where things land

    artifacts/enclave.duckdb   the in-enclave warehouse (never exported)
    models_out/                content-addressed minimodels + registry
    lean/data/                 enclave-side exports for the Lean layer
    lean/out/                  dp_release.json, genspec.json, lean_results.json
    release/<ts>/              the policy-filtered export (what may leave)
    dist/bundle/               the single source-only deployment bundle
    synth/, synth_bundle/      synthetic Joy JSONL + plan + meta + validation
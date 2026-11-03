# HOWTO run — enclave booking-trip encoder + synthetic generator

One-time setup:

    make setup              # python deps (dbt-duckdb, torch, marimo, ...)

Lean toolchain policy: `LEAN_HOME=/opt/data/lean` always. The Makefile
bootstraps it automatically (`lean-store` target): elan toolchains in
`$LEAN_HOME/elan`, the mathlib olean cache in `$LEAN_HOME/cache`, the pinned
package store (mathlib, SampCert @ a2c53bb) in `$LEAN_HOME/packages`,
symlinked into `lean/.lake/packages`. Installed once, reused by every build —
including bundle rebuilds. Nothing depends on `~/.elan`.

Input data: `combined_set.csv` (Booking.com multi-destination trips,
1,545,502 reservations) in the project root; the pipeline reads it via the
`BOOKING_CSV` env var (default `../combined_set.csv` from `dbt/`).

## The six flows

    make help                # prints the high-level summary

    make clean              # remove ALL generated data (keeps sources + the
                            #   shared Lean store)

    make train              # the ENTIRE enclave flow, in order:
                            #   dbt        CSV -> staging -> features ->
                            #              trips/users -> motifs -> χ² ->
                            #              DP inputs
                            #   models     5 minimodels, content-addressed,
                            #              gated vs frozen baselines
                            #              (BOOKING_RETRAIN=1 to force
                            #              retrain; CPU-deterministic)
                            #   ml         materialize ml_trip_scores
                            #   lean       build + checks + ε-DP release +
                            #              verified-sampler smoke
                            #   distill    released facts -> genspec.json
                            #   test       the full pytest suite
                            #   manifest   fingerprints -> artifacts/runs/
                            #   export     policy-filtered release/<ts>/
                            #              (BOOKING_ALLOW_DEV_RELEASE=1 on a
                            #              non-enclave dev host — user-run)
                            #   bundle     ONE source-only bundle -> dist/bundle

    make synth              # Lean generator -> synth/booking_synth.csv
                            #   (+ plan.json ground truth, meta.json
                            #   provenance); then validate: re-ingest through
                            #   the PRODUCTION dbt into a scratch DB, planted
                            #   repeat-city + tour + longstay recall must be
                            #   1.0, Lean deciders re-verify
                            #   (exit 0 / 2 soft / 1 hard); then replicates:
                            #   REPLICATES=8 independent generations, the
                            #   repeat-city share and res_count TV must stay
                            #   in their public tolerance bands
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

`make export` refuses on a non-enclave dev host unless the user exports
`BOOKING_ALLOW_DEV_RELEASE=1` (the export is then marked `attested: false`).
The privacy budget is deliberately loose (ledger 1008/100 = 10.08 ε of the
14/1 per-export cap — the fidelity-over-privacy choice recorded in goal.md);
every re-release composes with prior ones, and since spec 1.1.0 the
append-only `config/epsilon_ledger.jsonl` tracks the CUMULATIVE spend of all
exports of the same data against `dp_lifetime_epsilon_max: 42/1` — an export
past the lifetime cap is refused.

One pipeline driver at a time: concurrent `make` runs race over the DuckDB
write lock and the generated dirs that `clean` deletes.

## Where things land

    artifacts/enclave.duckdb   the in-enclave warehouse (never exported)
    models_out/                content-addressed minimodels + registry
    lean/data/                 enclave-side exports for the Lean layer
    lean/out/                  dp_release.json, genspec.json, lean_results.json
    release/<ts>/              the policy-filtered export (what may leave)
    dist/bundle/               the single source-only deployment bundle
    synth/, synth_bundle/      synthetic CSV + plan + meta + validation

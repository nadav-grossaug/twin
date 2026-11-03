# HOWTO run — enclave GPS-trajectory encoder + synthetic generator (GeoLife)

One-time setup:

    make setup              # python deps (duckdb, dbt-duckdb, torch, marimo, ...)
                            #   = pip install -r requirements.txt (repo root)

Lean toolchain policy: `LEAN_HOME=/opt/data/lean` always. The Makefile
bootstraps it automatically (`lean-store` target): elan toolchains in
`$LEAN_HOME/elan`, the mathlib olean cache in `$LEAN_HOME/cache`, the pinned
package store (mathlib, SampCert @ a2c53bb) in `$LEAN_HOME/packages`,
symlinked into `lean/.lake/packages`. Installed once, reused by every build —
including bundle rebuilds. Nothing depends on `~/.elan`.

Env overrides: `GEOLIFE_TSV` (source points TSV, default geolife_data.tsv)
and `ENCLAVE_DB` (warehouse path, default artifacts/enclave.duckdb).

## The six flows

    make clean              # remove ALL generated data (keeps sources + the
                            #   shared Lean store)

    make train              # the ENTIRE enclave flow, in order:
                            #   dbt        TSV -> staging -> staypoints ->
                            #              anchors/visits/legs -> motifs ->
                            #              DP inputs (29 models + tests)
                            #   models     4 minimodels (NextRoleGRU,
                            #              DwellClassGRU, ModeLegMLP,
                            #              UserProfileAE), content-addressed,
                            #              gated vs frozen baselines
                            #              (FLOW_RETRAIN=1 to force retrain;
                            #              FLOW_DEVICE=cpu deterministic)
                            #   ml         materialize ml_geo_scores
                            #   lean       build + checks (the 0-disagreement
                            #              SQL<->Lean contract) + ε-DP release
                            #              (ε per export from
                            #              config/release_policy.yaml —
                            #              29/1 since release-3.0.0) +
                            #              verified-sampler smoke
                            #   distill    released facts -> genspec.json
                            #              (the LIVE lean/out release when it
                            #              differs from the latest export)
                            #   test       the full pytest suite
                            #   manifest   fingerprints -> artifacts/runs/
                            #   export     policy-filtered release/<ts>/
                            #              (GEO_ALLOW_DEV_RELEASE=1 on a
                            #              non-enclave dev host; REFUSED past
                            #              the lifetime ε cap — see below)
                            #   bundle     ONE source-only bundle -> dist/bundle

    make synth              # Lean generator -> synth/geolife_synth.tsv
                            #   (+ plan.json ground truth, meta.json
                            #   provenance). Generation is in a neutral frame;
                            #   the public Tokyo-metro `manipulate` step is
                            #   applied before validation. Then validate:
                            #   re-ingest through the PRODUCTION dbt into a
                            #   scratch DB, planted-motif recall must be 1.0,
                            #   Lean deciders re-verify (exit 0 / 2 soft /
                            #   1 hard). N_USERS=64 (the real cohort size)
                            #   is the default; N_USERS=500 to change it

    make synth_from_bundle  # proves the bundle is self-contained: rebuild the
                            #   generator FROM dist/bundle's Lean sources (the
                            #   kernel re-checks every EnclaveGeo theorem),
                            #   distill the bundle's dp_release.json, generate
                            #   -> synth_bundle/. The ONLY synthesis inputs
                            #   are Lean sources + released files — no binaries

    make notebook-enclave   # serve the enclave EDA notebook (marimo)
    make notebook-synth     # serve the synth-vs-real scorecard notebook

Optional extras: `make certify` (kernel-checked certificate over the synth
plan), `make realism-audit` (enclave-side scoring of the synthetic data),
`make notebook-html` (headless exports into artifacts/reports/),
`make help` (this summary, from the Makefile itself).

## Order matters

`train` runs distill AFTER the Lean release because DP draws are
intentionally non-reproducible (urandom-backed FFI): the genspec is
hash-keyed to the release file. Re-running `make synth` after a fresh
`make train` always re-distills against the new release — distill prefers
the live `lean/out/dp_release.json` over an older export (it prints a
warning; the genspec then stays in the enclave until `make export`).

**Every export spends ε on the same 64 users.** `config/epsilon_ledger.jsonl`
(append-only, git-tracked) records each export; `release.py` refuses an
export past `dp_lifetime_epsilon_max` in `config/release_policy.yaml`. As of
2026-11-03 the ledger is at 33/33 (4 + 29): a further export needs a
deliberate cap raise, or — only for an export that never left the enclave —
`python3 python/release.py --void <export_id> --reason "..."`.

After a spec change, activate retrained minimodels with `FLOW_RETRAIN=1`.

One pipeline driver at a time: concurrent `make` runs race over the DuckDB
write lock and the generated dirs that `clean` deletes. Stop runs by PID or
full project path — sibling `lean4_*` builds use the same process names.

## Where things land

    artifacts/enclave.duckdb   the in-enclave warehouse (never exported)
    models_out/                content-addressed minimodels + registry
    lean/data/                 enclave-side exports for the Lean layer
    lean/out/                  dp_release.json, genspec.json, samples.json,
                               lean_results.json
    release/<ts>/              the policy-filtered export (what may leave):
                               DP histograms + the suppressed CSVs
                               (noise_moments, kinematics, turn_hist, dist_mix,
                               route_shape, seg_hist, junction_hist)
                               + data_integrity + lean_claims + manifest
    config/epsilon_ledger.jsonl  lifetime ε ledger (commit it; never edit by hand)
    dist/bundle/               the single source-only deployment bundle
    synth/, synth_bundle/      synthetic TSV + plan + meta + validation

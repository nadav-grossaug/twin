# Enclave stock-series encoder — TSV -> dbt/DuckDB -> PyTorch -> Lean4+SampCert
SHELL := /bin/bash
export PATH := $(HOME)/.elan/bin:$(PATH)
PY ?= python3
DBT := cd dbt && DBT_PROFILES_DIR=. dbt
N_SAMPLES ?= 5000
SYNTH_TICKERS ?= 8
SYNTH_DAYS ?= 168
CERT_N ?= 120

.PHONY: all help setup dbt train ml distill export-lean lean-build lean-checks lean-release \
        lean-sample lean test manifest export synth certify realism-audit \
        notebook notebook-synth notebook-edit notebook-html \
        clean_dbt clean_db clean_models clean_lean clean_notebook clean_reports \
        clean_release clean_synth clean_py clean distclean

## Full pipeline: features/motifs -> minimodels -> scores -> formal layer -> tests -> manifest
all: dbt train ml lean distill test manifest

help:                 ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:                ## install Python deps (Lean toolchain comes via elan on first lake run)
	$(PY) -m pip install -r requirements.txt

dbt:                  ## parse the stock TSV, build features, stats, motifs + run tests
	mkdir -p artifacts
	$(DBT) build --exclude ml_series_scores

train:                ## fit minimodels into the content-addressed store (no-op if inputs unchanged; FLOW_RETRAIN=1 activates a retrain; FLOW_DEVICE=cpu for deterministic mode)
	$(PY) python/train_minimodels.py

ml:                   ## re-materialize inference scores with the ACTIVE model artifact
	$(DBT) build --select ml_series_scores

export-lean:          ## export DP counts + motif/assoc/session observations to lean/data/
	$(PY) python/export_lean_data.py

lean-build:           ## build the Lean formal layer (first run: fetches mathlib cache)
	cd lean && lake exe cache get && lake build

lean-checks: export-lean  ## evaluate the formal definitions (motifs, chi-square, sessions) on enclave data
	cd lean && lake exe checks

lean-release: export-lean ## DP release: SampCert discrete Laplace + exact-rational budget ledger
	cd lean && lake exe release

lean-sample:          ## draw from the released histogram with the verified sampler
	cd lean && lake exe sample $(N_SAMPLES)

lean: lean-build lean-checks lean-release lean-sample

distill:              ## distill released facts into the typed Lean genspec (+ enclave-only fidelity report)
	$(PY) python/distill.py

test:                 ## pytest: spec sync, Lean<->SQL crosscheck, SampCert conformance, release gate
	$(PY) -m pytest -q tests

manifest:             ## write the run manifest (fingerprints, versions, table hashes)
	$(PY) scripts/run_manifest.py

export:               ## policy-filtered release into release/<id>/ (requires attestation; dev override FLOW_ALLOW_DEV_RELEASE=1)
	$(PY) python/release.py

bundle:               ## self-contained off-enclave generation bundle from the latest release -> dist/bundle/
	bash scripts/make_bundle.sh

synth:                ## synthetic stock TSV -> synth/ (standalone Lean generator from the genspec)
	cd lean && lake build generate checks
	$(PY) python/distill.py
	$(PY) scripts/synth.py generate --tickers $(SYNTH_TICKERS) --days $(SYNTH_DAYS) --out synth
	$(PY) scripts/synth.py validate --synth synth || [ $$? = 2 ]

certify:              ## kernel-checked Lean certificate that planted synth structures satisfy the spec
	cd lean && lake build EnclaveSeries
	$(PY) scripts/synth.py certify --synth synth --n $(CERT_N)

realism-audit:        ## enclave-only: score synthetic series with the production models
	$(PY) scripts/realism_audit.py --synth synth

notebook:             ## serve the marimo visualization (read-only app mode)
	marimo run visualize_enclave_mo.py

notebook-synth:       ## serve the synthetic-vs-enclave comparison notebook
	marimo run visualize_synth_mo.py

notebook-edit:        ## open the marimo notebook for editing
	marimo edit visualize_enclave_mo.py

notebook-html:        ## static HTML export of both notebooks into artifacts/reports/
	mkdir -p artifacts/reports
	marimo export html --no-include-code visualize_enclave_mo.py -o artifacts/reports/visualize_enclave.html
	marimo export html --no-include-code visualize_synth_mo.py -o artifacts/reports/visualize_synth.html

# ---------------------------------------------------------------------------
# Cleaning. Source data (stock_normalized_train.tsv), code and Lean
# DEPENDENCIES are never touched except by `distclean`, which also drops the
# downloaded Lean packages (~5 GB re-download on the next lean-build).
# ---------------------------------------------------------------------------
clean_dbt:            ## dbt run state (target/, logs)
	rm -rf dbt/target dbt/logs dbt/dbt_packages

clean_db:             ## the DuckDB warehouse built by dbt
	rm -f artifacts/enclave.duckdb artifacts/enclave.duckdb.wal

clean_models:         ## trained model artifacts and the registry (files are read-only)
	@if [ -d models_out ]; then chmod -R u+w models_out; fi
	rm -rf models_out

clean_lean:           ## this project's Lean build outputs, exported data, results, certificates (keeps mathlib/SampCert packages)
	rm -rf lean/.lake/build lean/out lean/data lean/GenCert.lean

clean_reports:        ## run manifests, reports, notebook HTML, logs
	rm -rf artifacts/runs artifacts/reports artifacts/logs

clean_release:        ## exported policy-filtered releases
	rm -rf release

clean_synth:          ## synthetic data and its checks
	rm -rf synth

clean_notebook:       ## marimo caches
	rm -rf __marimo__ .marimo.toml

clean_py:             ## Python caches
	find . -path ./lean/.lake -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \) -print0 | xargs -0 rm -rf

clean: clean_dbt clean_db clean_models clean_lean clean_reports clean_release clean_synth clean_notebook clean_py ## everything generated (keeps source data + Lean deps)
	rm -rf artifacts

distclean: clean      ## clean + downloaded Lean packages (mathlib, SampCert)
	rm -rf lean/.lake

# Enclave mobility-trajectory encoder — TSV -> dbt/DuckDB -> PyTorch -> Lean4+SampCert
#
# Six user-facing flows (everything else is an internal step):
#   make clean              remove all generated data
#   make train              entire enclave flow: dbt -> minimodels -> ml ->
#                           lean (build/checks/release/sample) -> distill ->
#                           tests -> export -> ONE source-only bundle
#   make synth              synthetic TSV from the local working tree
#   make synth_from_bundle  rebuild generator from dist/bundle sources, synth there
#   make notebook-enclave   serve the enclave EDA notebook
#   make notebook-synth     serve the synth-vs-real comparison notebook
#
# Lean toolchain policy: LEAN_HOME=/opt/data/lean always — elan toolchains,
# the mathlib cache and the pinned package store live there, installed once
# and reused by every build (repo and bundle alike). No reliance on ~/.elan.
SHELL := /bin/bash
LEAN_HOME ?= /opt/data/lean
export LEAN_HOME
export ELAN_HOME := $(LEAN_HOME)/elan
export XDG_CACHE_HOME := $(LEAN_HOME)/cache
export PATH := $(ELAN_HOME)/bin:$(PATH)
PY ?= python3
DBT := cd dbt && DBT_PROFILES_DIR=. dbt
# point-level generation: 64 synthetic users ≈ 200k GPS points
N_USERS ?= 64
CERT_N ?= 120

.PHONY: help clean train synth synth_from_bundle notebook-enclave notebook-synth \
        lean-store setup dbt train-models ml export-lean lean-build lean-checks \
        lean-release lean-sample lean distill test manifest export bundle certify \
        realism-audit notebook-html \
        clean_dbt clean_db clean_models clean_lean clean_notebook clean_reports \
        clean_release clean_synth clean_bundle clean_py distclean

help:                 ## this summary of the high-level flows
	@echo "Enclave mobility-trajectory encoder + verified synthetic generator"
	@echo "(geolife_data.tsv -> dbt/DuckDB -> PyTorch -> Lean4+SampCert)"
	@echo
	@echo "The six flows:"
	@echo "  make clean              remove ALL generated data (keeps sources, code,"
	@echo "                          and the shared Lean store in LEAN_HOME)"
	@echo "  make train              entire enclave flow: dbt features/motifs/DP inputs"
	@echo "                          -> minimodels -> ml scores -> Lean build + checks"
	@echo "                          + eps-DP release + sampler -> distill -> tests ->"
	@echo "                          manifest -> export -> ONE source-only bundle"
	@echo "  make synth              synthetic TSV -> synth/ (Lean generator from the"
	@echo "                          local genspec) + round-trip validation  [N_USERS=$(N_USERS)]"
	@echo "  make synth_from_bundle  rebuild the generator FROM dist/bundle sources"
	@echo "                          (kernel re-checks every theorem), synth -> synth_bundle/"
	@echo "  make notebook-enclave   serve the enclave EDA marimo notebook"
	@echo "  make notebook-synth     serve the synth-vs-real scorecard notebook"
	@echo
	@echo "Extras:"
	@echo "  make certify            kernel-checked certificate over the synth plan"
	@echo "  make realism-audit      score synthetic data with the enclave's own models"
	@echo "  make notebook-html      headless HTML export of both notebooks"
	@echo "  make test               the pytest suite on its own"
	@echo
	@echo "Lean toolchain: LEAN_HOME=$(LEAN_HOME) (elan + mathlib cache + pinned"
	@echo "packages, shared and reused; bootstrapped automatically by 'lean-store')."
	@echo "Details: HOWTO_run.md"

# ---------------------------------------------------------------------------
# The six flows
# ---------------------------------------------------------------------------

train: dbt train-models ml lean distill test manifest export bundle ## entire enclave flow: features -> models -> formal layer -> release -> tests -> source-only bundle

synth: lean-store     ## synthetic GeoLife TSV -> synth/ (standalone Lean generator from the local genspec)
	cd lean && lake build generate checks
	$(PY) python/distill.py
	$(PY) scripts/synth.py generate --users $(N_USERS) --out synth
	$(PY) scripts/synth.py validate --synth synth || [ $$? = 2 ]

synth_from_bundle: lean-store ## build the generator FROM THE BUNDLE's Lean sources, distill its release, synth -> synth_bundle/
	@[ -d dist/bundle ] || { echo "ERROR: dist/bundle missing — run 'make train' first" >&2; exit 1; }
	cd dist/bundle && ./build_from_source.sh
	cd dist/bundle && $(PY) python/distill.py
	$(PY) scripts/synth.py generate --users $(N_USERS) --out synth_bundle --bundle dist/bundle
	$(PY) scripts/synth.py validate --synth synth_bundle || [ $$? = 2 ]

notebook-enclave:     ## serve the enclave EDA marimo notebook (read-only app mode)
	marimo run --no-sandbox visualize_enclave_mo.py

notebook-synth:       ## serve the synthetic-vs-enclave comparison notebook
	marimo run --no-sandbox visualize_synth_mo.py

clean: clean_dbt clean_db clean_models clean_lean clean_reports clean_release clean_synth clean_bundle clean_notebook clean_py ## everything generated (keeps source data, code, and the shared Lean store)
	rm -rf artifacts

# ---------------------------------------------------------------------------
# Lean store bootstrap (LEAN_HOME=/opt/data/lean)
# ---------------------------------------------------------------------------
lean-store:           ## ensure elan + shared package store exist in LEAN_HOME and are linked into lean/.lake
	@mkdir -p $(LEAN_HOME)/packages $(LEAN_HOME)/cache
	@if [ ! -x $(ELAN_HOME)/bin/elan ]; then \
	  echo "installing elan into $(ELAN_HOME)"; \
	  curl -sSf https://elan.lean-lang.org/elan-init.sh | ELAN_HOME=$(ELAN_HOME) sh -s -- -y --default-toolchain none; \
	fi
	@mkdir -p lean/.lake
	@if [ ! -e lean/.lake/packages ]; then ln -s $(LEAN_HOME)/packages lean/.lake/packages; \
	elif [ ! -L lean/.lake/packages ]; then echo "ERROR: lean/.lake/packages exists but is not a symlink to the shared store" >&2; exit 1; fi

# ---------------------------------------------------------------------------
# Internal steps (composed by the flows above)
# ---------------------------------------------------------------------------
setup:
	$(PY) -m pip install -r ../../../requirements.txt

dbt:
	mkdir -p artifacts
	$(DBT) seed
	$(DBT) build --exclude ml_geo_scores ml_leg_scores

train-models:
	$(PY) python/train_minimodels.py

ml:
	$(DBT) build --select ml_geo_scores ml_leg_scores

export-lean:
	$(PY) python/export_lean_data.py

lean-build: lean-store
	cd lean && lake exe cache get && lake build

lean-checks: export-lean
	cd lean && lake exe checks

lean-release: export-lean
	cd lean && lake exe release

lean-sample:
	cd lean && lake exe sample 5000

lean: lean-build lean-checks lean-release lean-sample

distill:
	$(PY) python/distill.py

test:
	$(PY) -m pytest -q tests

manifest:
	$(PY) scripts/run_manifest.py

export:
	$(PY) python/release.py

bundle:
	bash scripts/make_bundle.sh

certify:
	cd lean && lake build EnclaveGeo
	$(PY) scripts/synth.py certify --synth synth --n $(CERT_N)

realism-audit:
	$(PY) scripts/realism_audit.py --synth synth

notebook-html:
	mkdir -p artifacts/reports
	marimo export html --no-include-code visualize_enclave_mo.py -o artifacts/reports/visualize_enclave.html
	marimo export html --no-include-code visualize_synth_mo.py -o artifacts/reports/visualize_synth.html

# ---------------------------------------------------------------------------
# Cleaning. Source data (geolife_data.tsv), code and the SHARED Lean
# store in LEAN_HOME are never touched. `distclean` additionally drops the
# project's .lake build dir (still not the shared packages, which are only
# symlinked).
# ---------------------------------------------------------------------------
clean_dbt:
	rm -rf dbt/target dbt/logs dbt/dbt_packages

clean_db:
	rm -f artifacts/enclave.duckdb artifacts/enclave.duckdb.wal

clean_models:
	@if [ -d models_out ]; then chmod -R u+w models_out; fi
	rm -rf models_out

clean_lean:
	rm -rf lean/.lake/build lean/out lean/data lean/GenCert.lean

clean_reports:
	rm -rf artifacts/runs artifacts/reports artifacts/logs

clean_release:
	rm -rf release

clean_synth:
	rm -rf synth synth_bundle

clean_bundle:
	rm -rf dist

clean_notebook:
	rm -rf __marimo__ .marimo.toml

clean_py:
	find . -path ./lean/.lake -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \) -print0 | xargs -0 rm -rf

distclean: clean
	rm -rf lean/.lake

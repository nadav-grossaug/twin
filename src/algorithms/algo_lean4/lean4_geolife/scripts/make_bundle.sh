#!/bin/bash
# Build a self-contained OFF-ENCLAVE generation bundle from the latest
# policy-filtered release. Trust model: NO compiled binaries are shipped.
# The consumer rebuilds the generator from the audited Lean sources
# (build_from_source.sh), which makes the Lean kernel re-check every
# EnclaveGeo theorem on their own machine before anything runs.
#
#   bundle/
#     lean/                     Lean sources, verbatim from the repo:
#                               EnclaveGeo/ (all proofs), executable roots,
#                               lakefile.lean, lake-manifest.json (pinned deps),
#                               lean-toolchain
#     lean/out/dp_release.json  the ε-DP release (copied from release/<id>/)
#     noise_moments.csv         the four suppressed (category-2) release
#     kinematics.csv            artifacts distill.py post-processes into the
#     turn_hist.csv             genspec: GPS-noise moments, per-mode
#     dist_mix.csv + route_shape/seg_hist/junction_hist.csv — kinematics,
#                               turn/segment/junction histograms, circuity,
#                               distance mix
#     python/distill.py         release -> genspec (post-processing only)
#     python/spec.py            the concept-spec reader distill imports
#     dbt/dbt_project.yml       the public spec vars spec.py reads
#     dbt/seeds/mode_map.csv    the public raw-label -> mode-class seed
#     build_from_source.sh      elan/lake bootstrap (shared LEAN_HOME store if
#                               present): fetch pinned deps, re-check proofs,
#                               build the generator
#     README.md, MANIFEST.sha256, release_manifest.json
set -euo pipefail
cd "$(dirname "$0")/.."

REL_DIR=${1:-$(ls -d release/*/ 2>/dev/null | sort | tail -1)}
[ -n "$REL_DIR" ] || { echo "ERROR: no release/ export found — run 'make export' first" >&2; exit 1; }

OUT=dist/bundle
rm -rf "$OUT"
mkdir -p "$OUT/lean/out" "$OUT/lean/EnclaveGeo" "$OUT/python" "$OUT/dbt/seeds"

# Lean sources, verbatim (no bundle-specific lakefile — zero drift from the
# repo build): library + every executable root lakefile.lean declares, plus
# the pinned toolchain and dependency manifest.
cp lean/lakefile.lean lean/lake-manifest.json lean/lean-toolchain "$OUT/lean/"
cp lean/EnclaveGeo.lean lean/Generate.lean lean/Release.lean \
   lean/Sample.lean lean/Catsample.lean lean/Checks.lean "$OUT/lean/"
cp lean/EnclaveGeo/*.lean "$OUT/lean/EnclaveGeo/"

cp "$REL_DIR/dp_release.json" "$OUT/lean/out/"
cp "$REL_DIR/noise_moments.csv" "$REL_DIR/kinematics.csv" \
   "$REL_DIR/route_shape.csv" "$REL_DIR/seg_hist.csv" "$REL_DIR/junction_hist.csv" \
   "$REL_DIR/turn_hist.csv" "$REL_DIR/dist_mix.csv" "$OUT/"
cp "$REL_DIR/manifest.json" "$OUT/release_manifest.json"
# distill + exactly what it imports/reads in bundle mode: spec.py, which
# parses the PUBLIC vars block of dbt_project.yml and the mode_map seed
cp python/distill.py python/spec.py "$OUT/python/"
cp dbt/dbt_project.yml "$OUT/dbt/"
cp dbt/seeds/mode_map.csv "$OUT/dbt/seeds/"

cat > "$OUT/build_from_source.sh" <<'EOF'
#!/bin/bash
# Verify-and-build: fetches the pinned toolchain + dependencies, has the Lean
# kernel re-check every EnclaveGeo theorem, then links the generator.
# A shared Lean store (LEAN_HOME, default /opt/data/lean) is used when
# present — elan toolchains, the mathlib cache and the pinned package store
# live there, installed once and reused by every build. Without one, first
# run needs network (toolchain, pinned git deps, mathlib olean cache) and
# downloads several GB; SampCert always compiles locally (~tens of min).
set -euo pipefail
cd "$(dirname "$0")/lean"

LEAN_HOME=${LEAN_HOME:-/opt/data/lean}
if [ -d "$LEAN_HOME" ]; then
  # shared-store bootstrap: reuse the host's toolchains, cache and packages
  export ELAN_HOME="$LEAN_HOME/elan"
  export XDG_CACHE_HOME="$LEAN_HOME/cache"
  export PATH="$ELAN_HOME/bin:$PATH"
  mkdir -p "$LEAN_HOME/packages" .lake
  if [ ! -e .lake/packages ]; then
    ln -s "$LEAN_HOME/packages" .lake/packages
  elif [ ! -L .lake/packages ]; then
    echo "ERROR: .lake/packages exists but is not a symlink to the shared store" >&2
    exit 1
  fi
  echo "using shared Lean store at $LEAN_HOME"
fi

command -v elan >/dev/null 2>&1 || {
  echo "ERROR: elan not found. Install it first:" >&2
  echo "  curl https://elan.lean-lang.org/elan-init.sh -sSf | sh" >&2
  exit 1
}
echo "toolchain: $(cat lean-toolchain) (elan fetches it on first use)"

# Prebuilt mathlib oleans; best-effort — without it mathlib builds from
# source, which is correct but takes hours.
lake exe cache get || echo "WARN: mathlib cache unavailable — building mathlib from source (slow)" >&2

# Default targets = EnclaveGeo lib (all theorems kernel-checked) + the
# generator and the checks decider. A failing proof fails the build:
# nothing runs unverified.
lake build EnclaveGeo generate checks

echo
echo "OK: proofs re-checked; generator at lean/.lake/build/bin/generate"
EOF
chmod +x "$OUT/build_from_source.sh"

cat > "$OUT/README.md" <<'EOF'
# Off-enclave synthetic-trajectory bundle (GeoLife, source-verified)

No compiled binaries are shipped. The generator is built on your machine from
the Lean sources in ./lean, so the Lean kernel re-checks every EnclaveGeo
theorem (DP mechanism, sampler law, motif, motion and generator contracts)
before anything runs — the formal guarantee is verified at deploy, not
asserted. Contents are limited to source code, the policy-filtered ε-DP
release (daily-motif, role-transition, dwell, departure and mode histograms),
the four suppressed release artifacts (GPS-noise moments, per-mode
kinematics, turn histogram, distance mix) and post-processing tools; no raw
GPS points, staypoint or anchor locations, model weights or per-user
artifacts are included.

    shasum -a 256 -c MANIFEST.sha256   # integrity check (run before building)
    ./build_from_source.sh             # elan/lake: pinned deps, re-check proofs,
                                       #   build the generator
    python3 python/distill.py          # release -> lean/out/genspec.json
    ./lean/.lake/build/bin/generate lean/out/genspec.json 64 out
    # -> out/geolife_synth.tsv (+ out/plan.json ground truth)

Every random draw inside `generate` goes through the formally verified
sampler (theorem sampleCellPMF_law, P(i) = w_i/Σw, SampCert). The privacy
accounting is in release_manifest.json (exact-fraction ε ledger, total 4/1);
the genspec is post-processing of dp_release.json plus the suppressed
artifacts, so it inherits that accounting. Draws are intentionally
non-reproducible (urandom FFI). The generator emits a neutral geographic
frame; the enclave repo's scripts/synth.py applies the public Tokyo-metro
`manipulate` step afterwards. Declared gaps: see "assumptions" in the
enclave's synth meta — notably street/transit geometry (points are not
snapped to any road or rail network), altitude dynamics (altitude_ft is a
public constant, not modelled), home-day device-off behaviour (recording
gaps on stay-at-home days) and cross-user relations (users are sampled
independently).

Supply-chain surface: build_from_source.sh fetches the Lean toolchain via
elan and the dependencies pinned by lake-manifest.json (SampCert @ exact
commit, mathlib @ exact commit, transitive pins), or reuses a shared store
at LEAN_HOME=/opt/data/lean when present. Verify MANIFEST first; the pins
are part of the hashed contents.

Requires: elan, python3 (stdlib only — the shipped spec module reads the
public dbt vars and mode seed, nothing else), network + several GB disk on
first build without a shared store (the Lean toolchain bundles its own C
compiler). Any OS/arch Lean supports — the bundle is not tied to the build
host's platform.
EOF

# shasum ships with macOS; coreutils sha256sum is the usual Linux spelling.
if command -v shasum >/dev/null 2>&1; then SHA=(shasum -a 256); else SHA=(sha256sum); fi
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -exec "${SHA[@]}" {} \; | sort -k2 > MANIFEST.sha256 )
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1)) from $REL_DIR — source-only, no binaries"

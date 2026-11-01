#!/bin/bash
# Build a self-contained OFF-ENCLAVE generation bundle from the latest
# policy-filtered release: everything a consumer needs to distill the genspec
# and generate synthetic series in another directory / on another machine
# (same OS/arch), without the repo, the enclave DB, dbt or PyTorch.
#
#   bundle/
#     bin/generate            standalone Lean generator (verified sampler inside)
#     bin/libleanffi.dylib    SampCert /dev/urandom FFI (install name patched)
#     lean/out/dp_release.json  the ε-DP release (copied from release/<id>/)
#     python/distill.py       release -> genspec (post-processing only; stdlib)
#     README.md, MANIFEST.sha256
set -euo pipefail
cd "$(dirname "$0")/.."

REL_DIR=${1:-$(ls -d release/*/ 2>/dev/null | sort | tail -1)}
[ -n "$REL_DIR" ] || { echo "ERROR: no release/ export found — run 'make export' first" >&2; exit 1; }
BIN=lean/.lake/build/bin/generate
[ -f "$BIN" ] || { echo "ERROR: run 'make lean-build' first" >&2; exit 1; }
FFI=lean/.lake/packages/sampcert/.lake/build/lib/libleanffi.dylib

OUT=dist/bundle
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lean/out" "$OUT/python"
cp "$REL_DIR/dp_release.json" "$OUT/lean/out/"
cp "$REL_DIR/manifest.json" "$OUT/release_manifest.json"
cp python/distill.py "$OUT/python/"
cp "$BIN" "$OUT/bin/"
cp "$FFI" "$OUT/bin/"
# the binary records the FFI dylib by absolute build path; point it next to
# the executable instead
OLD=$(otool -L "$OUT/bin/generate" | awk '/libleanffi/ {print $1}')
install_name_tool -change "$OLD" @executable_path/libleanffi.dylib "$OUT/bin/generate"
codesign --force --sign - "$OUT/bin/generate" 2>/dev/null || true

cat > "$OUT/README.md" <<'EOF'
# Off-enclave synthetic-series bundle

Contents are limited to the policy-filtered ε-DP release and post-processing
tools; no raw data, model weights or per-ticker artifacts are included.

    python3 python/distill.py                     # release -> lean/out/genspec.json
    ./bin/generate lean/out/genspec.json 8 168 out
    # -> out/stock_synth.tsv (+ out/plan.json ground truth)

Every random draw inside `generate` goes through the formally verified
sampler (theorem sampleCellPMF_law, P(i) = w_i/Σw, SampCert). The privacy
accounting is in release_manifest.json (exact-fraction ε ledger); the genspec
is post-processing of dp_release.json, so it inherits that accounting.
Draws are intentionally non-reproducible (urandom FFI). Declared gaps:
see "assumptions" in the enclave's synth meta — notably exact cross-ticker
correlation, market holidays, absolute price levels, volume.
Requires: macOS (same arch as the build host), python3 (stdlib only).
EOF

( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -exec shasum -a 256 {} \; | sort -k2 > MANIFEST.sha256 )
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1)) from $REL_DIR"

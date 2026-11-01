import Lake
open Lake DSL

package «enclavetravel» where
  -- SampCert pins Lean v4.34.1 and mathlib v4.34.1; we use its toolchain
  -- for the whole formal layer (goal.md guideline 11).

require sampcert from git
  "https://github.com/leanprover/SampCert" @ "a2c53bb5ff790e729e252958060699b7eff898b9"

-- Link args for executables that call SampCert's /dev/urandom FFI.
-- macOS: the dylib is resolved through its recorded install name (an absolute
-- build path; scripts/make_bundle.sh repoints it for relocated bundles), so
-- the original args suffice. Linux: the binaries need an $ORIGIN-relative
-- rpath to run outside `lake exe` (pytest execs them directly), plus $ORIGIN
-- itself so a copy of libleanffi.so next to the binary also works (bundles).
def ffiLinkArgs : Array String :=
  let common := #["-L.lake/packages/sampcert/.lake/build/lib", "-lleanffi"]
  if System.Platform.isOSX then
    common
  else
    common ++ #["-Wl,-rpath,$ORIGIN/../../packages/sampcert/.lake/build/lib",
                "-Wl,-rpath,$ORIGIN"]

@[default_target]
lean_lib EnclaveTravel

-- Executables that draw randomness must link SampCert's /dev/urandom FFI.
-- All executables are default targets so a plain `lake build` always
-- produces every binary the Makefile and scripts/synth.py rely on.
@[default_target]
lean_exe release where
  root := `Release
  moreLinkArgs := ffiLinkArgs

@[default_target]
lean_exe sample where
  root := `Sample
  moreLinkArgs := ffiLinkArgs

-- Verified categorical sampler over stdin weights (used by scripts/synth.py).
@[default_target]
lean_exe catsample where
  root := `Catsample
  moreLinkArgs := ffiLinkArgs

-- Deterministic motif/stat checks on exported data; no randomness needed.
@[default_target]
lean_exe checks where
  root := `Checks

-- Standalone genspec-driven generator (the big rewrite): generation runs
-- entirely in Lean from the distilled, released specification.
@[default_target]
lean_exe generate where
  root := `Generate
  moreLinkArgs := ffiLinkArgs

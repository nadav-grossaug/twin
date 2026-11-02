import Lake
open Lake DSL

package «enclaveflows» where
  -- SampCert pins Lean v4.34.1 and mathlib v4.34.1; we use its toolchain
  -- for the whole formal layer (goal.md guideline 11).

require sampcert from git
  "https://github.com/leanprover/SampCert" @ "a2c53bb5ff790e729e252958060699b7eff898b9"

@[default_target]
lean_lib EnclaveFlows

-- Executables that draw randomness must link SampCert's /dev/urandom FFI.
-- All executables are default targets so a plain `lake build` always
-- produces every binary the Makefile and scripts/synth.py rely on.
@[default_target]
lean_exe release where
  root := `Release
  moreLinkArgs := #["-L.lake/packages/sampcert/.lake/build/lib", "-lleanffi"]

@[default_target]
lean_exe sample where
  root := `Sample
  moreLinkArgs := #["-L.lake/packages/sampcert/.lake/build/lib", "-lleanffi"]

-- Verified categorical sampler over stdin weights (used by scripts/synth.py).
@[default_target]
lean_exe catsample where
  root := `Catsample
  moreLinkArgs := #["-L.lake/packages/sampcert/.lake/build/lib", "-lleanffi"]

-- Deterministic motif/stat checks on exported data; no randomness needed.
@[default_target]
lean_exe checks where
  root := `Checks

-- Standalone genspec-driven generator (the big rewrite): generation runs
-- entirely in Lean from the distilled, released specification.
@[default_target]
lean_exe generate where
  root := `Generate
  moreLinkArgs := #["-L.lake/packages/sampcert/.lake/build/lib", "-lleanffi"]

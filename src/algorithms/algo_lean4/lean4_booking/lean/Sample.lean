/-
Verified sampling executable: draws from the RELEASED (noised, clamped)
res_count histogram via `sampleCellPMF`, whose exact law
`P(cell i) = wᵢ/Σw` is proved in EnclaveTrip/Sampler.lean
(sampleCellPMF_law). Reads only the DP output — post-processing, so the DP
guarantee is preserved. This is the entry point for the downstream
synthetic-data generator's GOF check.
-/
import EnclaveTrip.Sampler
import Lean.Data.Json

open Lean SLang PMF EnclaveTrip

def parseRelease (j : Json) : Except String (List (String × Nat)) := do
  let arr ← (← j.getObjVal? "cells").getArr?
  arr.toList.mapM (fun c => do
    let name ← (← c.getObjVal? "cell").getStr?
    let rel ← (← c.getObjVal? "noised_count").getInt?
    pure (name, rel.toNat))

def main (args : List String) : IO UInt32 := do
  let n := (args.head?.bind String.toNat?).getD 5000
  let txt ← IO.FS.readFile "out/dp_release.json"
  match Json.parse txt >>= parseRelease with
  | .error e => IO.eprintln s!"sample: bad release file: {e}"; return 1
  | .ok cells =>
    let ws := cells.map (·.2)
    if h : 0 < ws.sum then
      let dist := sampleCellPMF ws h
      let mut counts : Array Nat := Array.replicate cells.length 0
      for _ in [0:n] do
        let i ← run dist
        if hi : i < counts.size then
          counts := counts.set i (counts[i] + 1)
      let outCells := (cells.zipIdx.map (fun (c, i) => Json.mkObj
        [ ("cell", Json.str c.1)
        , ("weight", Json.num (JsonNumber.fromNat c.2))
        , ("draws", Json.num (JsonNumber.fromNat (counts[i]?.getD 0))) ])).toArray
      let out := Json.mkObj
        [ ("sampler", Json.str "UniformSample ∘ bucketOf (theorem sampleCellPMF_law)")
        , ("n_draws", Json.num (JsonNumber.fromNat n))
        , ("total_weight", Json.num (JsonNumber.fromNat ws.sum))
        , ("cells", Json.arr outCells) ]
      IO.FS.createDirAll "out"
      IO.FS.writeFile "out/samples.json" (out.pretty)
      IO.println s!"sample: drew {n} samples from {cells.length} cells -> out/samples.json"
      return 0
    else
      IO.eprintln "sample: released histogram is all zero; nothing to sample"
      return 1

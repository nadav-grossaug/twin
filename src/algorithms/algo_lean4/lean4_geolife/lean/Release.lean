/-
DP release executable: reads the clipped per-cell counts exported from the
enclave store, adds SampCert discrete Laplace noise — the SAME sampler
(`DiscreteLaplaceGenSamplePMF (Δ·ε₂) ε₁`) that the proved mechanism
`privClippedCell` applies to the query result — enforces the exact-rational
ε budget ledger (total 400/100 ≤ 4/1 for the travel release), and writes the
exportable histogram.

Trust note (goal.md guideline 10): this I/O wrapper and the SQL that computed
the counts are assumed, not proved; the mechanism's DP theorem is proved in
EnclaveGeo/DPRelease.lean. Clamping the noisy counts at 0 downstream is
post-processing. Randomness: SampCert FFI reading /dev/urandom.
-/
import SampCert
import EnclaveGeo.DPRelease
import Lean.Data.Json

open Lean SLang PMF

structure CellIn where
  name : String
  count : Int

/-- An auxiliary histogram (region, region_transition, gap_class, …): same
mechanism, its own clip and per-cell ε — every cell is charged in the ledger. -/
structure AuxHist where
  name : String
  clip : Nat
  e1 : Nat
  e2 : Nat
  cells : List CellIn

structure ReleaseIn where
  specVersion : String
  privacyUnit : String
  cells : List CellIn
  clip : Nat
  e1 : Nat
  e2 : Nat
  budgetNum : Nat
  budgetDen : Nat
  aux : List AuxHist

def toPNat (n : Nat) : Option (PNat) :=
  if h : 0 < n then some ⟨n, h⟩ else none

def parseCells (j : Json) : Except String (List CellIn) := do
  let arr ← (← j.getObjVal? "cells").getArr?
  arr.toList.mapM (fun c => do
    let name ← (← c.getObjVal? "cell").getStr?
    let count ← (← c.getObjVal? "clipped_count").getInt?
    pure { name, count : CellIn })

def parseAux (j : Json) : Except String (List AuxHist) :=
  match j.getObjVal? "aux" with
  | .error _ => .ok []        -- aux histograms are optional (legacy inputs)
  | .ok a => do
    let arr ← a.getArr?
    arr.toList.mapM (fun h => do
      let name ← (← h.getObjVal? "name").getStr?
      let clip ← (← h.getObjVal? "clip").getNat?
      let e1 ← (← h.getObjVal? "eps1").getNat?
      let e2 ← (← h.getObjVal? "eps2").getNat?
      let cells ← parseCells h
      pure { name, clip, e1, e2, cells : AuxHist })

def parseInput (j : Json) : Except String ReleaseIn := do
  let specVersion ← (← j.getObjVal? "spec_version").getStr?
  let privacyUnit ← (← j.getObjVal? "privacy_unit").getStr?
  let clip ← (← j.getObjVal? "clip").getNat?
  let e1 ← (← j.getObjVal? "eps1").getNat?
  let e2 ← (← j.getObjVal? "eps2").getNat?
  let budgetNum ← (← j.getObjVal? "budget_num").getNat?
  let budgetDen ← (← j.getObjVal? "budget_den").getNat?
  if budgetDen = 0 then
    throw "budget_den must be positive"
  let cells ← parseCells j
  let aux ← parseAux j
  pure { specVersion, privacyUnit, cells, clip, e1, e2, budgetNum, budgetDen, aux }

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/dp_counts.json"
  let parsed := Json.parse txt >>= parseInput
  match parsed with
  | .error e => IO.eprintln s!"release: bad input: {e}"; return 1
  | .ok inp =>
    let (some clipP, some e1P, some e2P) :=
        (toPNat inp.clip, toPNat inp.e1, toPNat inp.e2)
      | do IO.eprintln "release: clip/eps must be positive"; return 1
    -- budget ledger: the main histogram's k cells are each ε₁/ε₂-DP and every
    -- aux histogram's kᵢ cells are each ε₁ᵢ/ε₂ᵢ-DP (composed bound per
    -- histogram: histMech_DP). The total is summed as an EXACT fraction and
    -- compared against budget_num/budget_den — no floats in the ledger.
    let mainFrac : Nat × Nat := (inp.cells.length * inp.e1, inp.e2)
    let rawFrac : Nat × Nat := inp.aux.foldl
      (fun acc h => (acc.1 * h.e2 + h.cells.length * h.e1 * acc.2, acc.2 * h.e2))
      mainFrac
    -- reduce the exact fraction (the comparison below is exact either way)
    let g := Nat.gcd rawFrac.1 rawFrac.2
    let totalFrac := if g = 0 then rawFrac else (rawFrac.1 / g, rawFrac.2 / g)
    if totalFrac.1 * inp.budgetDen > inp.budgetNum * totalFrac.2 then
      IO.eprintln s!"release: REFUSED — total ε = {totalFrac.1}/{totalFrac.2} exceeds budget {inp.budgetNum}/{inp.budgetDen}"
      return 2
    IO.FS.createDirAll "out"
    let mut outCells : Array Json := #[]
    for c in inp.cells do
      -- the proved mechanism's noise step: discrete Laplace at scale (Δ·ε₂)/ε₁
      let noised : Int ← run <| DiscreteLaplaceGenSamplePMF (clipP * e2P) e1P c.count
      let released := max 0 noised  -- post-processing (clamp)
      outCells := outCells.push <| Json.mkObj
        [("cell", Json.str c.name),
         ("noised_count", Json.num (JsonNumber.fromInt released))]
    -- aux histograms: identical noise step with each histogram's clip and ε
    let mut outAux : Array Json := #[]
    for h in inp.aux do
      let (some hClipP, some hE1P, some hE2P) := (toPNat h.clip, toPNat h.e1, toPNat h.e2)
        | do IO.eprintln s!"release: aux {h.name}: clip/eps must be positive"; return 1
      let mut hCells : Array Json := #[]
      for c in h.cells do
        let noised : Int ← run <| DiscreteLaplaceGenSamplePMF (hClipP * hE2P) hE1P c.count
        let released := max 0 noised
        hCells := hCells.push <| Json.mkObj
          [("cell", Json.str c.name),
           ("noised_count", Json.num (JsonNumber.fromInt released))]
      outAux := outAux.push <| Json.mkObj
        [ ("name", Json.str h.name)
        , ("clip", Json.num (JsonNumber.fromNat h.clip))
        , ("eps1", Json.num (JsonNumber.fromNat h.e1))
        , ("eps2", Json.num (JsonNumber.fromNat h.e2))
        , ("cells", Json.arr hCells) ]
    let out := Json.mkObj
      [ ("mechanism", Json.str "sampcert_discrete_laplace")
      , ("spec_version", Json.str inp.specVersion)
      , ("privacy_unit", Json.str inp.privacyUnit)
      , ("clip", Json.num (JsonNumber.fromNat inp.clip))
      , ("eps1", Json.num (JsonNumber.fromNat inp.e1))
      , ("eps2", Json.num (JsonNumber.fromNat inp.e2))
      , ("budget", Json.mkObj
          [ ("spent_num", Json.num (JsonNumber.fromNat totalFrac.1))
          , ("spent_den", Json.num (JsonNumber.fromNat totalFrac.2))
          , ("total_num", Json.num (JsonNumber.fromNat inp.budgetNum))
          , ("total_den", Json.num (JsonNumber.fromNat inp.budgetDen))
          , ("n_cells", Json.num (JsonNumber.fromNat inp.cells.length))
          , ("n_aux_cells", Json.num (JsonNumber.fromNat
              (inp.aux.foldl (· + ·.cells.length) 0)))
          , ("within_budget", Json.bool true) ])
      , ("cells", Json.arr outCells)
      , ("aux", Json.arr outAux) ]
    IO.FS.writeFile "out/dp_release.json" (out.pretty)
    IO.println s!"release: wrote out/dp_release.json ({inp.cells.length} cells + {inp.aux.length} aux histograms, total ε = {totalFrac.1}/{totalFrac.2} ≤ {inp.budgetNum}/{inp.budgetDen})"
    return 0

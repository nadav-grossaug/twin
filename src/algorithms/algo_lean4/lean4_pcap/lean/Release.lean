/-
DP release executable: reads the clipped per-cell counts exported from the
enclave store, adds SampCert discrete Laplace noise — the SAME sampler
(`DiscreteLaplaceGenSamplePMF (Δ·ε₂) ε₁`) that the proved mechanism
`privClippedCell` applies to the query result — enforces the ε budget ledger,
and writes the exportable histogram.

Trust note (goal.md guideline 10): this I/O wrapper and the SQL that computed
the counts are assumed, not proved; the mechanism's DP theorem is proved in
EnclaveFlows/DPRelease.lean. Normalisation of the noisy counts downstream is
post-processing. Randomness: SampCert FFI reading /dev/urandom.
-/
import SampCert
import EnclaveFlows.DPRelease
import Lean.Data.Json

open Lean SLang PMF

structure CellIn where
  name : String
  count : Int

/-- An auxiliary histogram (e.g. TLS stack ranks, DNS name shape): same
mechanism, its own clip and per-cell ε — every cell is charged in the ledger. -/
structure AuxHist where
  name : String
  clip : Nat
  e1 : Nat
  e2 : Nat
  cells : List CellIn

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

def parseInput (j : Json) :
    Except String (List CellIn × Nat × Nat × Nat × Float × Option (Nat × Nat) × List AuxHist) := do
  let clip ← (← j.getObjVal? "clip").getNat?
  let e1 ← (← j.getObjVal? "eps1").getNat?
  let e2 ← (← j.getObjVal? "eps2").getNat?
  let budget := ((← (← j.getObjVal? "budget_total_eps").getNum?)).toFloat
  -- exact-rational budget (preferred when present)
  let budgetFrac := (do
    let n ← ((j.getObjVal? "budget_num").toOption.bind (Json.getNat? · |>.toOption))
    let d ← ((j.getObjVal? "budget_den").toOption.bind (Json.getNat? · |>.toOption))
    if d = 0 then none else some (n, d))
  let cells ← parseCells j
  let aux ← parseAux j
  pure (cells, clip, e1, e2, budget, budgetFrac, aux)

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/dp_counts.json"
  let parsed := Json.parse txt >>= parseInput
  match parsed with
  | .error e => IO.eprintln s!"release: bad input: {e}"; return 1
  | .ok (cells, clip, e1, e2, budget, budgetFrac, aux) =>
    let (some clipP, some e1P, some e2P) := (toPNat clip, toPNat e1, toPNat e2)
      | do IO.eprintln "release: clip/eps must be positive"; return 1
    -- budget ledger: the main histogram's k cells are each ε₁/ε₂-DP and every
    -- aux histogram's kᵢ cells are each ε₁ᵢ/ε₂ᵢ-DP (composed bound per
    -- histogram: histMech_DP). The total is summed as an exact fraction when
    -- the input carries budget_num/budget_den; float only as legacy fallback.
    let mainFrac : Nat × Nat := (cells.length * e1, e2)
    let totalFrac : Nat × Nat := aux.foldl
      (fun acc h => (acc.1 * h.e2 + h.cells.length * h.e1 * acc.2, acc.2 * h.e2))
      mainFrac
    let totalEps := totalFrac.1.toFloat / totalFrac.2.toFloat
    let overBudget : Bool := match budgetFrac with
      | some (bn, bd) => totalFrac.1 * bd > bn * totalFrac.2
      | none => totalEps > budget
    if overBudget then
      IO.eprintln s!"release: REFUSED — total ε = {totalFrac.1}/{totalFrac.2} = {totalEps} exceeds budget {budget}"
      return 2
    IO.FS.createDirAll "out"
    let mut outCells : Array Json := #[]
    for c in cells do
      -- the proved mechanism's noise step: discrete Laplace at scale (Δ·ε₂)/ε₁
      let noised : Int ← run <| DiscreteLaplaceGenSamplePMF (clipP * e2P) e1P c.count
      let released := max 0 noised  -- post-processing (clamp)
      outCells := outCells.push <| Json.mkObj
        [("cell", Json.str c.name), ("released", Json.num (JsonNumber.fromInt released))]
    -- aux histograms: identical noise step with each histogram's clip and ε
    let mut outAux : Array Json := #[]
    for h in aux do
      let (some hClipP, some hE1P, some hE2P) := (toPNat h.clip, toPNat h.e1, toPNat h.e2)
        | do IO.eprintln s!"release: aux {h.name}: clip/eps must be positive"; return 1
      let mut hCells : Array Json := #[]
      for c in h.cells do
        let noised : Int ← run <| DiscreteLaplaceGenSamplePMF (hClipP * hE2P) hE1P c.count
        let released := max 0 noised
        hCells := hCells.push <| Json.mkObj
          [("cell", Json.str c.name), ("released", Json.num (JsonNumber.fromInt released))]
      outAux := outAux.push <| Json.mkObj
        [ ("name", Json.str h.name)
        , ("clip", Json.num (JsonNumber.fromNat h.clip))
        , ("eps_per_cell", Json.mkObj [("num", Json.num (JsonNumber.fromNat h.e1)),
                                       ("den", Json.num (JsonNumber.fromNat h.e2))])
        , ("cells", Json.arr hCells) ]
    let out := Json.mkObj
      [ ("mechanism", Json.str "SampCert discrete Laplace (privClippedCell per cell; composed bound histMech_DP)")
      , ("privacy_unit", Json.str "one monitored internal host (add/remove-one Neighbour; flows attributed to their internal endpoint)")
      , ("clip", Json.num (JsonNumber.fromNat clip))
      , ("eps_per_cell", Json.mkObj [("num", Json.num (JsonNumber.fromNat e1)),
                                     ("den", Json.num (JsonNumber.fromNat e2))])
      , ("ledger", Json.mkObj
          [ ("n_cells", Json.num (JsonNumber.fromNat cells.length))
          , ("n_aux_cells", Json.num (JsonNumber.fromNat (aux.foldl (· + ·.cells.length) 0)))
          , ("total_eps_num", Json.num (JsonNumber.fromNat totalFrac.1))
          , ("total_eps_den", Json.num (JsonNumber.fromNat totalFrac.2))
          , ("budget", Json.str (toString budget))
          , ("within_budget", Json.bool true) ])
      , ("aux", Json.arr outAux)
      , ("cells", Json.arr outCells) ]
    IO.FS.writeFile "out/dp_release.json" (out.pretty)
    IO.println s!"release: wrote out/dp_release.json ({cells.length} cells + {aux.length} aux histograms, total ε = {totalFrac.1}/{totalFrac.2} = {totalEps} ≤ {budget})"
    return 0

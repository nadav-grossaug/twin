/-
The restricted generative specification (genspec) and its semantics for the
stock-series generator.

`python/distill.py` distills the minimodels' generative behavior into a TYPED
spec — distributions (DP-released histograms), dependencies (conditional
class-transition and dow-seasonality histograms, vol-state transitions = the
discretized GARCH diffusion, magnitude bins, a co-movement coupling fraction)
and constraints — never arbitrary code. This module gives the spec its Lean
types, JSON parsing, and executable sampling semantics on top of the VERIFIED
sampler (`sampleCellPMF`, law `sampleCellPMF_law`: P(i) = wᵢ/Σw).

Reusable structural properties proved here:
  * `stepTimes_sorted` — positive gaps ⇒ strictly increasing event times;
  * `mkDay_valid`      — the day constructor satisfies the class/magnitude
                          validity predicate by construction;
  * `genChain` is structurally recursive on fuel — bounded generation
                          terminates by construction.
-/
import EnclaveSeries.Sampler
import EnclaveSeries.Motifs
import Lean.Data.Json

namespace EnclaveSeries.Gen

open Lean SLang PMF EnclaveSeries

/-! ### Spec types -/

structure Cell where
  name : String
  w : Nat

structure Hist where
  cells : List Cell

def Hist.weights (h : Hist) : List Nat := h.cells.map (·.w)

structure CondHist where
  given : String
  hist : Hist

structure GenSpec where
  version : String
  classes : Hist                 -- marginal return-class mix (DP)
  transitions : List CondHist    -- class -> class rows (DP + public prior)
  dow : List CondHist            -- day-of-week -> class rows (DP, seasonality)
  volTrans : List CondHist       -- vol-state -> vol-state rows (DP, GARCH)
  magBins : Hist                 -- |log return| magnitude bins (DP)
  comoveLambdaPct : Nat          -- market-coupling probability (post-processed)

/-! ### JSON parsing (the spec is data, never code) -/

def parseCellList (j : Json) : Except String (List Cell) := do
  let arr ← j.getArr?
  arr.toList.mapM fun c => do
    pure { name := (← (← c.getObjVal? "name").getStr?),
           w := (← (← c.getObjVal? "w").getNat?) }

def parseHist (j : Json) (k : String) : Except String Hist := do
  pure ⟨← parseCellList (← (← j.getObjVal? k).getObjVal? "cells")⟩

def parseCond (j : Json) (k : String) : Except String (List CondHist) := do
  let arr ← (← j.getObjVal? k).getArr?
  arr.toList.mapM fun t => do
    pure ({ given := (← (← t.getObjVal? "given").getStr?),
            hist := ⟨← parseCellList (← t.getObjVal? "cells")⟩ } : CondHist)

def GenSpec.parse (j : Json) : Except String GenSpec := do
  pure { version := (← (← j.getObjVal? "genspec_version").getStr?),
         classes := ← parseHist j "classes",
         transitions := ← parseCond j "transitions",
         dow := ← parseCond j "dow",
         volTrans := ← parseCond j "vol_transitions",
         magBins := ← parseHist j "mag_bins",
         comoveLambdaPct := (← (← j.getObjVal? "comove_lambda_pct").getNat?) }

/-! ### Executable sampling semantics -/

/-- One verified draw: index `i` with probability `wᵢ/Σw` (`sampleCellPMF_law`).
An all-zero weight vector falls back to a uniform draw (declared in meta). -/
def draw (ws : List Nat) : IO Nat := do
  let ws' := if ws.sum = 0 then ws.map (fun _ => 1) else ws
  if h : 0 < ws'.sum then
    run (sampleCellPMF ws' h)
  else
    pure 0

def Hist.choose (h : Hist) : IO Cell := do
  let i ← draw h.weights
  pure (h.cells.getD i ⟨"", 0⟩)

/-- Conditional row lookup with marginal fallback (all-zero or missing rows
fall back to the marginal histogram — declared). -/
def rowOf (rows : List CondHist) (fallback : Hist) (prev : String) : Hist :=
  match rows.find? (·.given = prev) with
  | some c => if c.hist.weights.sum = 0 then fallback else c.hist
  | none => fallback

/-- Bounded state sequence: structurally recursive on fuel, so bounded
generation requests terminate by construction. -/
def genChain (rows : List CondHist) (fallback : Hist) :
    Nat → String → IO (List String)
  | 0, _ => pure []
  | n + 1, prev => do
    let c ← (rowOf rows fallback prev).choose
    let rest ← genChain rows fallback n c.name
    pure (c.name :: rest)

/-! ### Reusable structural properties -/

/-- Event times accumulated from gaps: `t0, t0+g₁, t0+g₁+g₂, …`. -/
def stepTimes (t0 : Nat) : List Nat → List Nat
  | [] => [t0]
  | g :: gs => t0 :: stepTimes (t0 + g) gs

theorem stepTimes_head (t0 : Nat) (gs : List Nat) :
    (stepTimes t0 gs).head? = some t0 := by
  cases gs <;> rfl

theorem le_of_mem_stepTimes : ∀ (gs : List Nat) (t0 x : Nat),
    x ∈ stepTimes t0 gs → t0 ≤ x
  | [], t0, x, hx => by
    simp only [stepTimes, List.mem_singleton] at hx
    omega
  | g :: gs, t0, x, hx => by
    simp only [stepTimes, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact Nat.le_refl _
    · have := le_of_mem_stepTimes gs (t0 + g) x hx
      omega

/-- **Event-ordering preservation**: strictly positive gaps produce strictly
increasing event times (pairwise, so every pair is ordered). -/
theorem stepTimes_sorted (t0 : Nat) (gs : List Nat) (hp : ∀ g ∈ gs, 0 < g) :
    (stepTimes t0 gs).Pairwise (· < ·) := by
  induction gs generalizing t0 with
  | nil =>
    exact List.Pairwise.cons (by intro a ha; simp at ha) List.Pairwise.nil
  | cons g gs ih =>
    have hg : 0 < g := hp g (by simp)
    simp only [stepTimes]
    refine List.Pairwise.cons ?_ (ih (t0 + g) (fun x hx => hp x (by simp [hx])))
    intro y hy
    have := le_of_mem_stepTimes gs (t0 + g) y hy
    omega

/-! ### Day-plan validity by construction -/

/-- The fields of a planned synthetic day the validity predicate speaks about. -/
structure DayPlan where
  retBp : Int
  cls : Nat
  volState : Nat

/-- Structural day validity: the planned class is exactly the public
taxonomy's class of the planned return, and the magnitude is bounded. -/
def ValidDay (flatBp bigBp maxAbsBp : ℤ) (d : DayPlan) : Prop :=
  d.cls = classOfRet flatBp bigBp d.retBp ∧ |d.retBp| ≤ maxAbsBp

/-- Smart constructor: the class is computed, never passed in. -/
def mkDay (flatBp bigBp : ℤ) (r : Int) (vs : Nat) : DayPlan :=
  { retBp := r, cls := classOfRet flatBp bigBp r, volState := vs }

/-- **Day validity by construction**: every day the generator renders through
`mkDay` satisfies the validity predicate. -/
theorem mkDay_valid (flatBp bigBp maxAbs : ℤ) (r : Int) (vs : Nat)
    (hr : |r| ≤ maxAbs) :
    ValidDay flatBp bigBp maxAbs (mkDay flatBp bigBp r vs) := ⟨rfl, hr⟩

end EnclaveSeries.Gen

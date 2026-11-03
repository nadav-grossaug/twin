/-
The restricted generative specification (genspec) and its semantics.

`python/distill.py` distills the minimodels' generative behavior into a TYPED
spec — distributions (DP-released histograms), dependencies (conditional
next-service histograms = the probabilistic state machine replacing the
NextServiceGRU teacher), temporal structure (FlowShape constants) and
constraints — never arbitrary code. This module gives the spec its Lean
types, JSON parsing, and executable sampling semantics on top of the VERIFIED
sampler (`sampleCellPMF`, law `sampleCellPMF_law`: P(i) = wᵢ/Σw).

Reusable properties proved here (structural correctness, separate from
statistical fidelity and privacy, which are accounted upstream):
  * `stepTimes_chain`  — positive gaps ⇒ strictly increasing event times
                          (event-ordering preservation);
  * `mkFlow_valid`     — the record constructor satisfies the structural
                          validity predicate by construction;
  * `genChain` is structurally recursive on fuel — bounded generation
                          terminates by construction.
-/
import EnclaveFlows.Sampler
import Lean.Data.Json

namespace EnclaveFlows.Gen

open Lean SLang PMF EnclaveFlows

/-! ### Generator constants (assumption-level, `derived_from_data: false`) -/

def epoch0 : Nat := 1216691467
def sessionsPerHostChoices : List Nat := [2, 3]
def sessionsPerHostWeights : List Nat := [3, 2]
def interSessionExtraMs : Nat := 20000
def hostStartMaxS : Nat := 2000

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
  services : Hist
  transitions : List CondHist
  tlsStacks : Hist
  dnsKind : Hist
  dnsLabels : Hist
  dnsLen : Hist
  qnameTopk : List Cell
  sniTopk : List Cell

/-! ### JSON parsing (the spec is data, never code) -/

def parseCellList (j : Json) : Except String (List Cell) := do
  let arr ← j.getArr?
  arr.toList.mapM fun c => do
    pure { name := (← (← c.getObjVal? "name").getStr?),
           w := (← (← c.getObjVal? "w").getNat?) }

def parseHist (j : Json) (k : String) : Except String Hist := do
  pure ⟨← parseCellList (← (← j.getObjVal? k).getObjVal? "cells")⟩

def GenSpec.parse (j : Json) : Except String GenSpec := do
  let trArr ← (← j.getObjVal? "transitions").getArr?
  let transitions ← trArr.toList.mapM fun t => do
    pure ({ given := (← (← t.getObjVal? "given").getStr?),
            hist := ⟨← parseCellList (← t.getObjVal? "cells")⟩ } : CondHist)
  pure { version := (← (← j.getObjVal? "genspec_version").getStr?),
         services := ← parseHist j "services",
         transitions := transitions,
         tlsStacks := ← parseHist j "tls_stacks",
         dnsKind := ← parseHist j "dns_kind",
         dnsLabels := ← parseHist j "dns_labels",
         dnsLen := ← parseHist j "dns_len",
         qnameTopk := ← parseCellList (← j.getObjVal? "qname_topk"),
         sniTopk := ← parseCellList (← j.getObjVal? "sni_topk") }

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

/-- Weighted choice over (choices, weights): the `weightedChoice` primitive. -/
def weightedChoice (choices weights : List Nat) : IO Nat := do
  let i ← draw weights
  pure (choices.getD i 0)

/-- Transition step of the distilled state machine: the next-state histogram
conditioned on the previous state; all-zero or missing rows fall back to the
marginal service histogram (declared). -/
def GenSpec.step (s : GenSpec) (prev : String) : Hist :=
  match s.transitions.find? (·.given = prev) with
  | some c => if c.hist.weights.sum = 0 then s.services else c.hist
  | none => s.services

/-- Bounded state sequence: structurally recursive on fuel, so bounded
generation requests terminate by construction. -/
def genChain (s : GenSpec) : Nat → String → IO (List String)
  | 0, _ => pure []
  | n + 1, prev => do
    let c ← (s.step prev).choose
    let rest ← genChain s n c.name
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

/-- The fields of a rendered Joy flow that the validity predicate speaks
about. -/
structure Flow where
  pr : Nat
  dp : Nat
  sp : Nat
  bytesOut : Nat
  bytesIn : Nat
  pktsOut : Nat
  pktsIn : Nat
  tMs : Nat
  durMs : Nat

/-- Structural record validity (mirrors tests/test_synth_columns.py and the
feat_flows ingestion assumptions). -/
def ValidFlow (f : Flow) : Prop :=
  (f.pr = 6 ∨ f.pr = 17)
  ∧ 0 < f.durMs
  ∧ f.pktsOut = 1 + f.bytesOut / 1400
  ∧ f.pktsIn = (if f.bytesIn = 0 then 0 else 1 + f.bytesIn / 1400)
  ∧ 49152 ≤ f.sp

/-- Smart constructor: derived fields are computed, never passed in. -/
def mkFlow (pr dp spSeed bo bi t dur : Nat) : Flow :=
  { pr := pr, dp := dp, sp := 49152 + spSeed % 16000,
    bytesOut := bo, bytesIn := bi,
    pktsOut := 1 + bo / 1400,
    pktsIn := if bi = 0 then 0 else 1 + bi / 1400,
    tMs := t, durMs := dur }

/-- **Record validity by construction**: every flow the generator renders
through `mkFlow` satisfies the validity predicate. -/
theorem mkFlow_valid (pr dp spSeed bo bi t dur : Nat)
    (hpr : pr = 6 ∨ pr = 17) (hdur : 0 < dur) :
    ValidFlow (mkFlow pr dp spSeed bo bi t dur) := by
  refine ⟨hpr, hdur, rfl, rfl, ?_⟩
  simp only [mkFlow]
  omega

end EnclaveFlows.Gen

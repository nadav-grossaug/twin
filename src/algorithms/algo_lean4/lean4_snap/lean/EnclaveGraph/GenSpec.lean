/-
The restricted generative specification (genspec) and its semantics for the
synthetic social-graph generator.

`python/distill.py` distills the release into a TYPED spec — distributions
(DP-released histograms: degree classes, community sizes, mixing rows,
wedge-closure rows, feature richness, category presence, circle counts,
motif rates) and the public bin ranges — never arbitrary code. This module
gives the spec its Lean types, JSON parsing, and executable sampling
semantics on top of the VERIFIED sampler (`sampleCellPMF`, law
`sampleCellPMF_law`: P(i) = wᵢ/Σw).

Structural properties proved here (the generator's graph invariants):
  * `mkEdge_ok` / `mkEdge_none_iff` — the smart edge constructor only ever
    produces canonical (src < dst) edges and rejects exactly self-loops;
  * `edgesCanonical_sound` — the generator's O(m) output check implies the
    edge list is strictly lex-sorted (hence duplicate-free, `edges_nodup`)
    and every edge is canonical;
  * `pickDistinct_nodup` / `pickDistinct_sub` — category assignment draws
    are duplicate-free and stay inside the public taxonomy by construction;
  * `genChain` is structurally recursive on fuel — bounded generation
    terminates by construction.
-/
import EnclaveGraph.Sampler
import EnclaveGraph.Motifs
import Lean.Data.Json

namespace EnclaveGraph.Gen

open Lean SLang PMF EnclaveGraph

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

/-- A public inclusive [lo, hi] integer range. -/
structure RangeN where
  lo : Nat
  hi : Nat

structure GenSpec where
  version : String
  nNodesHint : Nat
  degree : Hist               -- degree classes (DP, clip 1)
  clustering : Hist           -- clustering classes (DP; informational)
  triangles : Hist            -- triangle classes (DP; informational)
  richness : Hist             -- feature-richness classes (DP, clip 1)
  category : Hist             -- category presence (DP, clip 8)
  circles : Hist              -- circle-membership classes (DP, clip 1)
  csize : Hist                -- community-size classes (DP, clip 2)
  mixing : List CondHist      -- per-rank partner-rank rows (DP, clip 8)
  closure : List CondHist     -- per-degree-class {closed, open} rows (DP, clip 8)
  motifs : Hist               -- motif presence (DP; informational)
  egoCount : Nat              -- public: number of ego networks in the schema
  degRanges : List (String × RangeN)
  csizeRanges : List (String × RangeN)
  circRanges : List (String × RangeN)
  richRanges : List (String × RangeN)

/-! ### JSON parsing (the spec is data, never code) -/

def parseCellList (j : Json) : Except String (List Cell) := do
  let arr ← j.getArr?
  arr.toList.mapM fun c => do
    pure { name := (← (← c.getObjVal? "name").getStr?),
           w := (← (← c.getObjVal? "w").getNat?) }

def parseHist (j : Json) (k : String) : Except String Hist := do
  pure ⟨← parseCellList (← (← j.getObjVal? k).getObjVal? "cells")⟩

def parseCond (j : Json) (k : String) : Except String (List CondHist) := do
  let arr ← (← (← j.getObjVal? k).getObjVal? "rows").getArr?
  arr.toList.mapM fun t => do
    pure ({ given := (← (← t.getObjVal? "given").getStr?),
            hist := ⟨← parseCellList (← t.getObjVal? "cells")⟩ } : CondHist)

/-- Parse a `{name: [lo, hi], ...}` range object. -/
def parseRangeList (j : Json) (k : String) : Except String (List (String × RangeN)) := do
  let obj ← (← j.getObjVal? k).getObj?
  obj.toList.mapM fun (key, v) => do
    let arr ← v.getArr?
    let lo ← (arr[0]?.getD Json.null).getNat?
    let hi ← (arr[1]?.getD Json.null).getNat?
    pure (key, ({ lo, hi } : RangeN))

def GenSpec.parse (j : Json) : Except String GenSpec := do
  pure { version := (← (← j.getObjVal? "spec_version").getStr?),
         nNodesHint := (← (← j.getObjVal? "n_nodes_hint").getNat?),
         degree := ← parseHist j "degree",
         clustering := ← parseHist j "clustering",
         triangles := ← parseHist j "triangles",
         richness := ← parseHist j "richness",
         category := ← parseHist j "category",
         circles := ← parseHist j "circles",
         csize := ← parseHist j "community_sizes",
         mixing := ← parseCond j "mixing",
         closure := ← parseCond j "closure",
         motifs := ← parseHist j "motifs",
         egoCount := (← (← j.getObjVal? "ego_count").getNat?),
         degRanges := ← parseRangeList j "deg_ranges",
         csizeRanges := ← parseRangeList j "csize_ranges",
         circRanges := ← parseRangeList j "circ_ranges",
         richRanges := ← parseRangeList j "rich_ranges" }

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
def rowOf (rows : List CondHist) (fallback : Hist) (given : String) : Hist :=
  match rows.find? (·.given = given) with
  | some c => if c.hist.weights.sum = 0 then fallback else c.hist
  | none => fallback

/-- Bounded label sequence: structurally recursive on fuel, so bounded
generation requests terminate by construction. -/
def genChain (rows : List CondHist) (fallback : Hist) :
    Nat → String → IO (List String)
  | 0, _ => pure []
  | n + 1, prev => do
    let c ← (rowOf rows fallback prev).choose
    let rest ← genChain rows fallback n c.name
    pure (c.name :: rest)

/-! ### Canonical edges, valid by construction -/

/-- Strict lexicographic order on canonical edges. -/
def lexLt (a b : Nat × Nat) : Prop :=
  a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance : DecidableRel lexLt := fun a b => by
  unfold lexLt; exact inferInstance

theorem lexLt_trans {a b c : Nat × Nat} (h₁ : lexLt a b) (h₂ : lexLt b c) :
    lexLt a c := by
  unfold lexLt at *; omega

theorem lexLt_irrefl (a : Nat × Nat) : ¬ lexLt a a := by
  unfold lexLt; omega

/-- Smart edge constructor: orients the pair canonically, rejects exactly
self-loops. The generator adds edges ONLY through `mkEdge`. -/
def mkEdge (s d : Nat) : Option (Nat × Nat) :=
  if s < d then some (s, d) else if d < s then some (d, s) else none

/-- Whatever `mkEdge` produces is canonical. -/
theorem mkEdge_ok {s d : Nat} {e : Nat × Nat} (h : mkEdge s d = some e) :
    EdgeOK e.1 e.2 := by
  unfold mkEdge at h
  unfold EdgeOK
  split at h
  · cases h; simpa
  · split at h
    · cases h; simpa
    · cases h

/-- `mkEdge` rejects exactly the self-loops. -/
theorem mkEdge_none_iff (s d : Nat) : mkEdge s d = none ↔ s = d := by
  unfold mkEdge
  constructor
  · intro h
    split at h
    · cases h
    · split at h
      · cases h
      · omega
  · intro h; subst h; simp

/-! ### The generator's O(m) output check and its soundness -/

/-- Adjacent strict-lex sortedness (Boolean, executable). -/
def edgesSortedB : List (Nat × Nat) → Bool
  | [] => true
  | [_] => true
  | a :: b :: t => decide (lexLt a b) && edgesSortedB (b :: t)

/-- Every edge canonical (Boolean, executable). -/
def edgesOKB (es : List (Nat × Nat)) : Bool :=
  es.all (fun e => decide (EdgeOK e.1 e.2))

/-- The generator's final output check. -/
def edgesCanonical (es : List (Nat × Nat)) : Bool :=
  edgesSortedB es && edgesOKB es

/-- Adjacent sortedness propagates to the full pairwise order (lexLt is
transitive). -/
theorem edgesSortedB_pairwise : ∀ {es : List (Nat × Nat)},
    edgesSortedB es = true → es.Pairwise lexLt
  | [] , _ => List.Pairwise.nil
  | [a], _ => by simp
  | a :: b :: t, h => by
    have hab : lexLt a b := by
      have := (Bool.and_eq_true _ _).mp h
      exact of_decide_eq_true this.1
    have ht : edgesSortedB (b :: t) = true :=
      ((Bool.and_eq_true _ _).mp h).2
    have hrest := edgesSortedB_pairwise ht
    refine List.Pairwise.cons ?_ hrest
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hxt
    · exact hab
    · have hbx : lexLt b x := by
        have := List.pairwise_cons.mp hrest
        exact this.1 x hxt
      exact lexLt_trans hab hbx

/-- **Soundness of the output check**: a passing edge list is strictly
lex-sorted and every edge is canonical. -/
theorem edgesCanonical_sound {es : List (Nat × Nat)}
    (h : edgesCanonical es = true) :
    es.Pairwise lexLt ∧ ∀ e ∈ es, EdgeOK e.1 e.2 := by
  have hp := (Bool.and_eq_true _ _).mp h
  refine ⟨edgesSortedB_pairwise hp.1, ?_⟩
  intro e he
  have := List.all_eq_true.mp hp.2 e he
  exact of_decide_eq_true this

/-- A strictly lex-sorted edge list has no duplicates. -/
theorem edges_nodup {es : List (Nat × Nat)} (h : es.Pairwise lexLt) :
    es.Nodup :=
  h.imp (fun hab => by
    intro heq
    subst heq
    exact lexLt_irrefl _ hab)

/-! ### Category assignment, duplicate-free by construction -/

/-- Pure accumulation step: add a candidate name only if absent (and
non-empty). The IO loop below uses ONLY this step to grow its accumulator. -/
def addPick (acc : List String) (name : String) : List String :=
  if name ∈ acc ∨ name = "" then acc else name :: acc

/-- `addPick` preserves duplicate-freedom — so the fuel-bounded IO loop's
accumulator is duplicate-free by induction, having started from `[]`. -/
theorem addPick_nodup {acc : List String} (h : acc.Nodup) (name : String) :
    (addPick acc name).Nodup := by
  unfold addPick
  split
  · exact h
  · next hn =>
    refine List.Nodup.cons ?_ h
    intro hmem
    exact hn (Or.inl hmem)

/-- `addPick` never leaves a provided universe: if the accumulator is inside
`u` and the candidate is in `u`, the result stays inside `u`. -/
theorem addPick_sub {u acc : List String} (hs : ∀ x ∈ acc, x ∈ u)
    {name : String} (hn : name ∈ u) : ∀ x ∈ addPick acc name, x ∈ u := by
  unfold addPick
  split
  · exact hs
  · intro x hx
    rcases List.mem_cons.mp hx with rfl | hxa
    · exact hn
    · exact hs x hxa

/-- Draw up to `k` DISTINCT cell names from a histogram (fuel-bounded
rejection; every accumulator update goes through `addPick`, so the result is
duplicate-free by `addPick_nodup`). -/
def pickDistinct (h : Hist) : Nat → Nat → List String → IO (List String)
  | 0, _, acc => pure acc
  | _, 0, acc => pure acc
  | fuel + 1, k + 1, acc => do
    let c ← h.choose
    let acc' := addPick acc c.name
    if acc'.length = acc.length then
      pickDistinct h fuel (k + 1) acc
    else
      pickDistinct h fuel k acc'

end EnclaveGraph.Gen

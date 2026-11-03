/-
Public structural constants of the synthetic-graph generator and the checks
executable: the total class maps from raw per-node integers (degree,
triangles, richness, circles, community size) onto the PUBLIC cell grids,
the fixed cell orders every histogram uses, and the GraphML rendering
helpers (XML escaping, feature-key shape). The generator and the checks
executable import these definitions — one literal shared with SQL and
Python via the spec sync test. Kernel-checked anchors tie the grid sizes to
the ledger arithmetic.
-/
import Mathlib.Tactic
import EnclaveGraph.Params
import EnclaveGraph.Motifs

namespace EnclaveGraph.Shape

open EnclaveGraph

/-! ### Degree classes -/

def degreeClasses : List String :=
  ["deg_1", "deg_2", "deg_3_4", "deg_5_8", "deg_9_16", "deg_17_32",
   "deg_33_64", "deg_65p"]

/-- Total map degree → public degree class (degree 0 folds into the lowest
bin; the input graph is connected, so it never occurs — checked, not
assumed). -/
def degreeClass (d : Nat) : String :=
  if d < Params.degE2 then "deg_1"
  else if d < Params.degE3 then "deg_2"
  else if d < Params.degE5 then "deg_3_4"
  else if d < Params.degE9 then "deg_5_8"
  else if d < Params.degE17 then "deg_9_16"
  else if d < Params.degE33 then "deg_17_32"
  else if d < Params.degE65 then "deg_33_64"
  else "deg_65p"

theorem degreeClass_mem (d : Nat) : degreeClass d ∈ degreeClasses := by
  unfold degreeClass degreeClasses
  repeat' split
  all_goals simp

/-! ### Local clustering classes (exact, via cross-multiplication) -/

def ccClasses : List String :=
  ["cc_zero", "cc_0_20", "cc_20_40", "cc_40_60", "cc_60_80", "cc_80_99",
   "cc_one"]

/-- Total map (degree, triangles) → public clustering class. `cc = 2t/(d(d−1))`;
`t = 0` or no wedges → `cc_zero`; `2t = d(d−1)` → `cc_one`; otherwise the
k/5 edges by exact Nat comparison (`cc < k/5 ⟺ 5·2t < k·d(d−1)`). -/
def ccClass (d t : Nat) : String :=
  if t = 0 ∨ ccDen d = 0 then "cc_zero"
  else if ccDen d ≤ ccNum t then "cc_one"
  else if CcBelow 1 Params.ccDen d t then "cc_0_20"
  else if CcBelow 2 Params.ccDen d t then "cc_20_40"
  else if CcBelow 3 Params.ccDen d t then "cc_40_60"
  else if CcBelow 4 Params.ccDen d t then "cc_60_80"
  else "cc_80_99"

theorem ccClass_mem (d t : Nat) : ccClass d t ∈ ccClasses := by
  unfold ccClass ccClasses
  repeat' split
  all_goals simp

/-- The measured whole-graph anchor: a median node (degree 25, 161 triangles)
sits in `cc_40_60` (cc = 322/600 ≈ 0.537) — kernel-checked. -/
example : ccClass 25 161 = "cc_40_60" := by decide

/-! ### Triangle-count classes -/

def triClasses : List String :=
  ["tri_0", "tri_1_10", "tri_11_100", "tri_101_1000", "tri_1001_10000",
   "tri_10001p"]

def triClass (t : Nat) : String :=
  if t < Params.triE1 then "tri_0"
  else if t < Params.triE11 then "tri_1_10"
  else if t < Params.triE101 then "tri_11_100"
  else if t < Params.triE1001 then "tri_101_1000"
  else if t < Params.triE10001 then "tri_1001_10000"
  else "tri_10001p"

theorem triClass_mem (t : Nat) : triClass t ∈ triClasses := by
  unfold triClass triClasses
  repeat' split
  all_goals simp

/-! ### Feature-richness classes -/

def richClasses : List String :=
  ["rich_0", "rich_1", "rich_2", "rich_3", "rich_4", "rich_5", "rich_6",
   "rich_7", "rich_8p"]

def richClass (r : Nat) : String :=
  if r < Params.richE8 then s!"rich_{r}" else "rich_8p"

theorem richClass_mem (r : Nat) : richClass r ∈ richClasses := by
  unfold richClass richClasses
  split
  · next h =>
    unfold Params.richE8 at h
    interval_cases r <;> decide
  · decide

/-! ### Circle-membership classes -/

def circClasses : List String := ["circ_0", "circ_1", "circ_2", "circ_3", "circ_4p"]

def circClass (c : Nat) : String :=
  if c < Params.circE4 then s!"circ_{c}" else "circ_4p"

theorem circClass_mem (c : Nat) : circClass c ∈ circClasses := by
  unfold circClass circClasses
  split
  · next h =>
    unfold Params.circE4 at h
    interval_cases c <;> decide
  · decide

/-! ### Community-size classes -/

def csizeClasses : List String :=
  ["csize_lt25", "csize_25_50", "csize_51_100", "csize_101_200",
   "csize_201_400", "csize_gt400"]

def csizeClass (s : Nat) : String :=
  if s < Params.csizeE25 then "csize_lt25"
  else if s < Params.csizeE51 then "csize_25_50"
  else if s < Params.csizeE101 then "csize_51_100"
  else if s < Params.csizeE201 then "csize_101_200"
  else if s < Params.csizeE401 then "csize_201_400"
  else "csize_gt400"

theorem csizeClass_mem (s : Nat) : csizeClass s ∈ csizeClasses := by
  unfold csizeClass csizeClasses
  repeat' split
  all_goals simp

/-! ### Category, mixing, closure and motif grids (fixed cell orders) -/

def categoryCells : List String := Params.categories.map (fun c => s!"cat_{c}")

/-- Rank index in the fixed order c1 < c2 < … < c8 < rest (total; unknown
labels sort last so the pair cell is still well-defined). -/
def rankIndex (r : String) : Nat :=
  (Params.communityRanks.idxOf r)

/-- Ordered mixing cell `<ra>__<rb>` with ra ≤ rb in rank order. -/
def mixCell (a b : String) : String :=
  if rankIndex a ≤ rankIndex b then s!"{a}__{b}" else s!"{b}__{a}"

/-- The 45 mixing cells, row-major upper triangle over the 9 ranks. -/
def mixingCells : List String :=
  (Params.communityRanks.zipIdx.flatMap (fun (a, i) =>
    (Params.communityRanks.zipIdx.filterMap (fun (b, j) =>
      if i ≤ j then some s!"{a}__{b}" else none))))

example : mixingCells.length = 45 := by decide

/-- A `mixCell` of two known ranks is always one of the 45 public cells. -/
theorem mixCell_mem {a b : String} (ha : a ∈ Params.communityRanks)
    (hb : b ∈ Params.communityRanks) : mixCell a b ∈ mixingCells := by
  fin_cases ha <;> fin_cases hb <;> decide

/-- The 16 wedge-closure cells: `clo_<degree class>__{closed|open}`. -/
def closureCells : List String :=
  degreeClasses.flatMap (fun d => [s!"clo_{d}__closed", s!"clo_{d}__open"])

def closureCell (degClass : String) (closed : Bool) : String :=
  if closed then s!"clo_{degClass}__closed" else s!"clo_{degClass}__open"

example : closureCells.length = 16 := by decide

/-- `closureCell` of a public degree class lands in the public closure grid. -/
theorem closureCell_mem' {s : String} (hs : s ∈ degreeClasses) (b : Bool) :
    closureCell s b ∈ closureCells := by
  unfold closureCells
  refine List.mem_flatMap.mpr ⟨s, hs, ?_⟩
  cases b <;> simp [closureCell]

theorem closureCell_mem (d : Nat) (closed : Bool) :
    closureCell (degreeClass d) closed ∈ closureCells :=
  closureCell_mem' (degreeClass_mem d) closed

def motifCells : List String :=
  ["motif_hub", "motif_leaf", "motif_broker", "motif_bridge",
   "motif_butterfly", "motif_core"]

/-- Ledger anchor: the grid sizes behind the 385/100 budget arithmetic
(8+7+6+9+11+5+6+45+16+6 cells; the per-grid ε are in the release). -/
example :
    degreeClasses.length = 8 ∧ ccClasses.length = 7 ∧
    triClasses.length = 6 ∧ richClasses.length = 9 ∧
    categoryCells.length = 11 ∧ circClasses.length = 5 ∧
    csizeClasses.length = 6 ∧ mixingCells.length = 45 ∧
    closureCells.length = 16 ∧ motifCells.length = 6 := by decide

/-! ### GraphML rendering helpers (assumed I/O per goal.md guideline 10) -/

/-- Minimal XML attribute/text escaping. -/
def escapeXml (s : String) : String :=
  s.foldl (fun acc c =>
    acc ++ match c with
      | '&' => "&amp;"
      | '<' => "&lt;"
      | '>' => "&gt;"
      | '"' => "&quot;"
      | '\'' => "&apos;"
      | c => String.mk [c]) ""

example : escapeXml "a<b&c" = "a&lt;b&amp;c" := by native_decide

/-- Synthetic feature key in the public shape convention
`ego_<e>__f<i>_<category>_anonymized_feature_<n>` (a labelling device —
fresh ids, no data). -/
def featureKey (ego idx : Nat) (cat : String) (n : Nat) : String :=
  s!"ego_{ego}__f{idx}_{cat}_anonymized_feature_{n}"

example :
    featureKey 0 18 "education_year" 257
      = "ego_0__f18_education_year_anonymized_feature_257" := by native_decide

end EnclaveGraph.Shape

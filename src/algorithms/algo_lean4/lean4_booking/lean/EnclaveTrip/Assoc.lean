/-
Formal chi-square association analysis over ℚ.

The dbt mart mart_feature_assoc computes pairwise chi-square statistics in
floating point; this module defines the statistic EXACTLY over ℚ (expected
counts under independence from the table's own margins), with the basic
guarantees: the statistic is nonnegative, and it vanishes on exactly
independent tables. `lake exe checks` recomputes chi-square from exported
contingency cells with exact rational arithmetic and compares against SQL.
-/
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace EnclaveTrip

/-- A contingency table: rows of observed counts. -/
abbrev ContTable := List (List ℚ)

def rowSums (t : ContTable) : List ℚ := t.map List.sum

def colSums (t : ContTable) : List ℚ := t.transpose.map List.sum

def total (t : ContTable) : ℚ := (rowSums t).sum

/-- Expected count of cell (i, j) under independence of rows and columns
(0 when the table is empty, by ℚ's division convention). -/
def expected (t : ContTable) (i j : ℕ) : ℚ :=
  (rowSums t).getD i 0 * (colSums t).getD j 0 / total t

/-- One cell's chi-square contribution; 0 for non-positive expected counts
(degenerate margins), keeping the statistic total and executable. -/
def chiTerm (o e : ℚ) : ℚ := if e ≤ 0 then 0 else (o - e) ^ 2 / e

/-- The chi-square statistic of a contingency table. -/
def chiSq (t : ContTable) : ℚ :=
  (t.zipIdx.map (fun ri =>
    (ri.1.zipIdx.map (fun oj => chiTerm oj.1 (expected t ri.2 oj.2))).sum)).sum

theorem chiTerm_nonneg (o e : ℚ) : 0 ≤ chiTerm o e := by
  unfold chiTerm
  split
  · exact le_refl 0
  · next h => exact div_nonneg (sq_nonneg _) (le_of_lt (lt_of_not_ge h))

theorem chiTerm_self (e : ℚ) : chiTerm e e = 0 := by
  unfold chiTerm
  split
  · rfl
  · simp

/-- The chi-square statistic is nonnegative. -/
theorem chiSq_nonneg (t : ContTable) : 0 ≤ chiSq t := by
  unfold chiSq
  apply List.sum_nonneg
  intro x hx
  obtain ⟨ri, _, rfl⟩ := List.mem_map.mp hx
  apply List.sum_nonneg
  intro y hy
  obtain ⟨oj, _, rfl⟩ := List.mem_map.mp hy
  exact chiTerm_nonneg _ _

/-- Exact independence: every observed cell equals its expected count. -/
def IsIndependent (t : ContTable) : Prop :=
  ∀ ri ∈ t.zipIdx, ∀ oj ∈ ri.1.zipIdx, oj.1 = expected t ri.2 oj.2

/-- On an exactly independent table the statistic vanishes. -/
theorem chiSq_eq_zero_of_indep {t : ContTable} (h : IsIndependent t) :
    chiSq t = 0 := by
  unfold chiSq
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨ri, hri, rfl⟩ := List.mem_map.mp hx
  apply List.sum_eq_zero
  intro y hy
  obtain ⟨oj, hoj, rfl⟩ := List.mem_map.mp hy
  rw [h ri hri oj hoj]
  exact chiTerm_self _

/-- Decidable association claim: the statistic reaches threshold `τ`. -/
def AssocAtLeast (τ : ℚ) (t : ContTable) : Prop := τ ≤ chiSq t

instance (τ : ℚ) (t : ContTable) : Decidable (AssocAtLeast τ t) := by
  unfold AssocAtLeast; exact inferInstance

/-- Association claims are monotone in the threshold. -/
theorem assocAtLeast_mono {τ τ' : ℚ} {t : ContTable} (h : τ' ≤ τ)
    (ha : AssocAtLeast τ t) : AssocAtLeast τ' t :=
  le_trans h ha

end EnclaveTrip

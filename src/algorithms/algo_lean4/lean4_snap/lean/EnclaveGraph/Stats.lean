/-
The histogram-cell query and its sensitivity proof.

Privacy unit: one user = one list element, SampCert's add/remove-one
`Neighbour` relation (a user's whole travel history is one record). Each
user's contribution to a cell is clipped into [0, c], so the query has
sensitivity c — the hypothesis the discrete Laplace mechanism needs
(see DPRelease.lean).
-/
import SampCert.DifferentialPrivacy.Sensitivity

namespace EnclaveGraph

/-- Clip a user's per-cell contribution into `[0, c]`. -/
def clip (c : ℕ) (x : ℤ) : ℤ := max 0 (min x c)

/-- The cell query: sum of clipped per-user contributions. -/
def clippedSum (c : ℕ) (l : List ℤ) : ℤ := (l.map (clip c)).sum

theorem clip_nonneg (c : ℕ) (x : ℤ) : 0 ≤ clip c x := le_max_left _ _

theorem clip_le (c : ℕ) (x : ℤ) : clip c x ≤ c := by
  unfold clip; omega

/-- Adding or removing one user changes the clipped cell sum by at most `c`:
`clippedSum c` is `c`-sensitive in SampCert's sense. -/
theorem clippedSum_sensitivity (c : ℕ) : sensitivity (clippedSum c) c := by
  intro l₁ l₂ H
  cases H with
  | Addition h1 h2 =>
    rename_i a b n
    subst h1; subst h2
    have hlo := clip_nonneg c n
    have hhi := clip_le c n
    simp only [clippedSum, List.map_append, List.sum_append, List.map_cons,
      List.sum_cons, List.map_nil, List.sum_nil]
    omega
  | Deletion h1 h2 =>
    rename_i a n b
    subst h1; subst h2
    have hlo := clip_nonneg c n
    have hhi := clip_le c n
    simp only [clippedSum, List.map_append, List.sum_append, List.map_cons,
      List.sum_cons, List.map_nil, List.sum_nil]
    omega

end EnclaveGraph

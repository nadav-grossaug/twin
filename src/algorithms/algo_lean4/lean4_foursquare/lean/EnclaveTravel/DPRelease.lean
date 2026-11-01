/-
The differentially private release mechanism for one histogram cell, built on
SampCert's verified discrete Laplace mechanism, with its privacy theorem.

Proved: `privClippedCell c ε₁ ε₂` is pure (ε₁/ε₂)-DP w.r.t. add/remove-one
user (`Neighbour`), for ANY input list — this follows from
`clippedSum_sensitivity` and SampCert's `privNoisedQueryPure_DP`.

Assumed (goal.md guideline 10): the per-user contributions fed to the
mechanism were correctly computed by the dbt SQL; SampCert's /dev/urandom FFI;
the unverified I/O wrappers in Release.lean. Releasing k cells sequentially
costs k·(ε₁/ε₂) of budget; the executable enforces the budget ledger.

Interface defaults for the travel release come from the mirrored spec
(Params.lean): `dp_clip_total = 8` for the auxiliary histograms and
`dp_count_clip = 1` for the one-cell-per-user histograms (travel_count,
return_rate).
-/
import SampCert
import EnclaveTravel.Stats
import EnclaveTravel.Params

namespace EnclaveTravel

open SLang

/-- DP release of one histogram cell: discrete Laplace noise (scale ε₁/(c·ε₂))
added to the clipped per-user sum. `noncomputable` like SampCert's mechanism;
the executable runs the identical underlying sampler. -/
noncomputable def privClippedCell (c ε₁ ε₂ : ℕ+) : Mechanism ℤ ℤ :=
  privNoisedQueryPure (clippedSum c) c ε₁ ε₂

/-- The release mechanism for one cell is pure (ε₁/ε₂)-differentially private. -/
theorem privClippedCell_PureDP (c ε₁ ε₂ : ℕ+) :
    PureDP (privClippedCell c ε₁ ε₂) ((ε₁ : NNReal) / ε₂) :=
  privNoisedQueryPure_DP (clippedSum c) c ε₁ ε₂ ((ε₁ : NNReal) / ε₂) rfl
    (clippedSum_sensitivity c)

end EnclaveTravel

/-! ### Composed histogram mechanism

The whole released histogram as ONE mechanism with a proved total budget
(pattern shared with the GeoLife sibling encoder): per-user clipped counts via
SampCert's `exactBoundedSum`, noised per cell, composed over the public cell
list with `privCompose`/`privPostProcess`. `K` cells cost `K·ε₁/ε₂` — the
budget the executable's ledger enforces is now a theorem, not bookkeeping. -/

namespace EnclaveTravel

open SLang

/-- A user's contribution to cell `c` (cells absent from the vector count 0). -/
def cellOf (c : ℕ) (u : List ℕ) : ℕ := u.getD c 0

/-- Clipped per-cell count: each user contributes at most `C`. -/
def cellQuery (C : ℕ+) (c : ℕ) (l : List (List ℕ)) : ℤ :=
  exactBoundedSum C (l.map (cellOf c))

/-- `Neighbour` transports along `map`: one SampCert sensitivity theorem then
covers every per-cell projection of a user-contribution vector. -/
theorem neighbour_map {α β : Type} (f : α → β) {l₁ l₂ : List α}
    (h : Neighbour l₁ l₂) : Neighbour (l₁.map f) (l₂.map f) := by
  cases h with
  | @Addition a b n h1 h2 =>
    exact Neighbour.Addition (a := a.map f) (b := b.map f) (n := f n)
      (by rw [h1, List.map_append]) (by rw [h2]; simp)
  | @Deletion a n b h1 h2 =>
    exact Neighbour.Deletion (a := a.map f) (b := b.map f) (n := f n)
      (by rw [h1]; simp) (by rw [h2, List.map_append])

/-- Adding or removing one user changes a clipped cell count by at most `C`. -/
theorem cellQuery_sensitivity (C : ℕ+) (c : ℕ) : sensitivity (cellQuery C c) C :=
  fun _ _ h => exactBoundedSum_sensitivity C _ _ (neighbour_map (cellOf c) h)

/-- One noisy cell: discrete Laplace noise at scale `C·ε₂/ε₁`. -/
def cellMech (C ε₁ ε₂ : ℕ+) (c : ℕ) : Mechanism (List ℕ) ℤ :=
  privNoisedQueryPure (cellQuery C c) C ε₁ ε₂

theorem cellMech_DP (C ε₁ ε₂ : ℕ+) (c : ℕ) :
    PureDP (cellMech C ε₁ ε₂ c) ((ε₁ : NNReal) / ε₂) :=
  privNoisedQueryPure_DP _ C ε₁ ε₂ _ rfl (cellQuery_sensitivity C c)

instance : MeasurableSpace (List ℤ) := ⊤
instance : DiscreteMeasurableSpace (List ℤ) :=
  ⟨fun _ => MeasurableSpace.measurableSet_top⟩

/-- Noisy histogram over a (public, data-independent) list of cells. -/
def histMech (C ε₁ ε₂ : ℕ+) : List ℕ → Mechanism (List ℕ) (List ℤ)
  | [] => privConst []
  | c :: cs => privPostProcess
      (privCompose (cellMech C ε₁ ε₂ c) (histMech C ε₁ ε₂ cs))
      (fun p => p.1 :: p.2)

/-- **User-level privacy of the released histogram**: `K` cells cost `K·ε₁/ε₂`. -/
theorem histMech_DP (C ε₁ ε₂ : ℕ+) (cells : List ℕ) :
    PureDP (histMech C ε₁ ε₂ cells) (cells.length * ((ε₁ : NNReal) / ε₂)) := by
  induction cells with
  | nil =>
    simp only [histMech, List.length_nil, CharP.cast_eq_zero, zero_mul]
    exact (PureDPSystem (T := List ℕ)).const_prop rfl
  | cons c cs ih =>
    simp only [histMech]
    apply (PureDPSystem (T := List ℕ)).postprocess_prop
    apply DPSystem.compose_prop (dps := PureDPSystem) _ (cellMech_DP C ε₁ ε₂ c) ih
    simp [List.length_cons]; ring

end EnclaveTravel

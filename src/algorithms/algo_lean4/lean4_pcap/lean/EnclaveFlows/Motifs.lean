/-
Formal definitions of the flow motifs computed by the dbt marts, over ℚ so
every predicate is decidable and executable (no square roots: the CV threshold
is squared). `lake exe checks` evaluates these very definitions on data
exported from the enclave store — the "explicitly connected executable checks"
of goal.md.
-/
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace EnclaveFlows

/-! ### Beacon motif -/

/-- Public beacon thresholds. `maxCvSq` is the squared coefficient-of-variation
threshold (SQL uses cv ≤ 0.6, so `maxCvSq = 9/25`). Periods are milliseconds. -/
structure BeaconParams where
  minFlows : Nat
  maxCvSq : ℚ
  minPeriodMs : ℚ

/-- Mean of a list of rationals (`0` for `[]`, by ℚ's division-by-zero convention). -/
def qMean (l : List ℚ) : ℚ := l.sum / l.length

/-- Sum of squared deviations from the mean. -/
def qSsd (l : List ℚ) : ℚ := (l.map (fun x => (x - qMean l) ^ 2)).sum

/--
`IsBeacon p gaps`: the channel has at least `minFlows` flows (`gaps` are the
`n-1` inter-start gaps), a positive mean period of at least `minPeriodMs`, and
sample variance `qSsd/(n-1) ≤ maxCvSq · mean²` — i.e. `cv ≤ √maxCvSq`,
stated multiplicatively to stay in ℚ.
-/
def IsBeacon (p : BeaconParams) (gapsMs : List ℚ) : Prop :=
  p.minFlows ≤ gapsMs.length + 1 ∧
  0 < qMean gapsMs ∧
  p.minPeriodMs ≤ qMean gapsMs ∧
  qSsd gapsMs ≤ p.maxCvSq * (qMean gapsMs) ^ 2 * ((gapsMs.length - 1 : ℕ) : ℚ)

instance (p : BeaconParams) (gaps : List ℚ) : Decidable (IsBeacon p gaps) := by
  unfold IsBeacon; exact inferInstance

theorem qMean_replicate {n : ℕ} (hn : n ≠ 0) (g : ℚ) :
    qMean (List.replicate n g) = g := by
  have hn' : (n : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hn
  rw [qMean, List.sum_replicate, List.length_replicate, nsmul_eq_mul,
      mul_div_cancel_left₀ _ hn']

theorem qSsd_replicate {n : ℕ} (hn : n ≠ 0) (g : ℚ) :
    qSsd (List.replicate n g) = 0 := by
  simp [qSsd, List.map_replicate, qMean_replicate hn]

/-- A perfectly periodic channel is a beacon (sanity theorem: the definition
accepts the motif it was designed around). -/
theorem isBeacon_replicate (p : BeaconParams) {n : ℕ} {g : ℚ}
    (hn : p.minFlows ≤ n + 1) (hpos : 0 < n) (hg : 0 < g)
    (hp : p.minPeriodMs ≤ g) (hcv : 0 ≤ p.maxCvSq) :
    IsBeacon p (List.replicate n g) := by
  have hne : n ≠ 0 := hpos.ne'
  refine ⟨by simpa using hn, ?_, ?_, ?_⟩
  · rw [qMean_replicate hne]; exact hg
  · rw [qMean_replicate hne]; exact hp
  · rw [qSsd_replicate hne]
    exact mul_nonneg (mul_nonneg hcv (sq_nonneg _)) (Nat.cast_nonneg _)

/-- Relaxing the CV threshold preserves beacon-hood (monotonicity). -/
theorem isBeacon_mono_cv {p q : BeaconParams} {gaps : List ℚ}
    (hm : p.minFlows = q.minFlows) (hper : p.minPeriodMs = q.minPeriodMs)
    (hcv : p.maxCvSq ≤ q.maxCvSq) (h : IsBeacon p gaps) : IsBeacon q gaps := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨hm ▸ h1, h2, hper ▸ h3, le_trans h4 ?_⟩
  have hM : (0 : ℚ) ≤ (qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ) :=
    mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)
  calc p.maxCvSq * (qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ)
      = p.maxCvSq * ((qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ)) := by ring
    _ ≤ q.maxCvSq * ((qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ)) :=
        mul_le_mul_of_nonneg_right hcv hM
    _ = q.maxCvSq * (qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ) := by ring

/-! ### Fan-out motif -/

/-- `IsFanout k dsts`: the source contacted at least `k` distinct destinations. -/
def IsFanout {α : Type} [DecidableEq α] (k : Nat) (dsts : List α) : Prop :=
  k ≤ dsts.dedup.length

instance {α : Type} [DecidableEq α] (k : Nat) (dsts : List α) :
    Decidable (IsFanout k dsts) := by unfold IsFanout; exact inferInstance

/-- Lowering the threshold preserves fan-out (monotonicity). -/
theorem isFanout_mono {α : Type} [DecidableEq α] {k k' : ℕ} {dsts : List α}
    (hk : k' ≤ k) (h : IsFanout k dsts) : IsFanout k' dsts :=
  le_trans hk h

/-! ### Return-trip motif A → B → A -/

/-- A destination sequence contains a return trip `A → B → A` with `A ≠ B`. -/
inductive HasReturnTrip {α : Type} : List α → Prop where
  | here {a b : α} {rest : List α} (hne : a ≠ b) : HasReturnTrip (a :: b :: a :: rest)
  | there {x : α} {rest : List α} : HasReturnTrip rest → HasReturnTrip (x :: rest)

theorem not_hasReturnTrip_nil {α : Type} : ¬ HasReturnTrip ([] : List α) :=
  fun h => nomatch h

theorem not_hasReturnTrip_one {α : Type} (a : α) : ¬ HasReturnTrip [a] := by
  intro h
  cases h with
  | there h' => exact not_hasReturnTrip_nil h'

theorem not_hasReturnTrip_two {α : Type} (a b : α) : ¬ HasReturnTrip [a, b] := by
  intro h
  cases h with
  | there h' => exact not_hasReturnTrip_one b h'

/-- Inversion: a window either matches at the head or the motif lies in the tail. -/
theorem hasReturnTrip_cons_iff {α : Type} {a b c : α} {rest : List α} :
    HasReturnTrip (a :: b :: c :: rest) ↔
      (a = c ∧ a ≠ b) ∨ HasReturnTrip (b :: c :: rest) := by
  constructor
  · intro h
    cases h with
    | here hne => exact Or.inl ⟨rfl, hne⟩
    | there h' => exact Or.inr h'
  · rintro (⟨rfl, hne⟩ | h)
    · exact .here hne
    · exact .there h

instance decHasReturnTrip {α : Type} [DecidableEq α] :
    (l : List α) → Decidable (HasReturnTrip l)
  | [] => isFalse not_hasReturnTrip_nil
  | [a] => isFalse (not_hasReturnTrip_one a)
  | [a, b] => isFalse (not_hasReturnTrip_two a b)
  | a :: b :: c :: rest =>
    have tail := decHasReturnTrip (b :: c :: rest)
    decidable_of_iff _ hasReturnTrip_cons_iff.symm

/-- Sanity example: the canonical return trip is recognized. -/
example : HasReturnTrip ["A", "B", "A"] := .here (by decide)

/-- Sanity example: a one-way chain is not a return trip. -/
example : ¬ HasReturnTrip ["A", "B", "C"] := by decide

/-- Inversion both ways: a return trip is EXACTLY an `[a, b, a]` infix with
`a ≠ b` (soundness and completeness, via Mathlib's infix relation). -/
theorem hasReturnTrip_iff_infix {α : Type} {l : List α} :
    HasReturnTrip l ↔ ∃ a b, a ≠ b ∧ [a, b, a] <:+: l := by
  constructor
  · intro h
    induction h with
    | @here a b rest hne =>
      exact ⟨a, b, hne, [], rest, by simp⟩
    | @there x rest h' ih =>
      obtain ⟨a, b, hne, hinf⟩ := ih
      exact ⟨a, b, hne, List.infix_cons hinf⟩
  · rintro ⟨a, b, hne, s, t, rfl⟩
    induction s with
    | nil => exact .here hne
    | cons x xs ih => exact .there ih

end EnclaveFlows

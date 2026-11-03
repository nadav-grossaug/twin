/-
Formal temporal flow logic: booking trips.

A rendered trip is a list of checkin observations (epoch day + nights). The
dataset's timely flow logic: checkins are strictly ordered within a utrip,
and 99.1% of consecutive reservations are CONTIGUOUS — the next checkin
equals the previous checkout. `IsWellFormedTrip` states the ordered,
bounded-gap version over checkin days (the formal counterpart of
feat_trips.all_contiguous = false); `IsContiguousTrip` states the exact
contiguous version over (checkin, nights) pairs. Both are decidable and
executable on observations exported from the enclave (`lake exe checks`).
-/
import Mathlib.Data.List.Chain
import Mathlib.Tactic
import EnclaveTrip.Params

namespace EnclaveTrip

/-! ### Well-formed trips (ordered, bounded checkin gaps) -/

/-- Consecutive step of a trip: strictly ordered checkin days, with the
gap bounded by `maxGap`. -/
def TripStep (maxGap : Nat) (a b : Nat) : Prop := a < b ∧ b - a ≤ maxGap

instance (maxGap : Nat) : DecidableRel (TripStep maxGap) := fun _ _ => by
  unfold TripStep; exact inferInstance

/-- A well-formed trip at an explicit gap bound: every consecutive pair of
checkin days is a trip step. -/
def IsWellFormedTripAt (maxGap : Nat) (days : List Nat) : Prop :=
  List.IsChain (TripStep maxGap) days

instance (maxGap : Nat) (days : List Nat) :
    Decidable (IsWellFormedTripAt maxGap days) := by
  unfold IsWellFormedTripAt; exact inferInstance

/-- A well-formed trip at the public spec bound `seqbreak_gap_d`. -/
def IsWellFormedTrip (days : List Nat) : Prop :=
  IsWellFormedTripAt Params.seqbreakGapD days

instance (days : List Nat) : Decidable (IsWellFormedTrip days) := by
  unfold IsWellFormedTrip; exact inferInstance

/-- Dropping the first reservation preserves the well-formed-trip property. -/
theorem isWellFormedAt_tail {g : Nat} {d : Nat} {days : List Nat}
    (h : IsWellFormedTripAt g (d :: days)) : IsWellFormedTripAt g days :=
  List.IsChain.tail h

/-- Relaxing the gap bound preserves well-formedness (monotonicity). -/
theorem isWellFormedAt_mono {g g' : Nat} (hg : g ≤ g') {days : List Nat}
    (h : IsWellFormedTripAt g days) : IsWellFormedTripAt g' days :=
  h.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩)

/-- A well-formed trip is strictly chronologically ordered. -/
theorem isWellFormedAt_ordered {g : Nat} {days : List Nat}
    (h : IsWellFormedTripAt g days) : List.IsChain (· < ·) days :=
  h.imp (fun _ _ hab => hab.1)

/-! ### Contiguous trips (the 99.1% flow: checkout = next checkin)

Executed as a Bool predicate (`contigB`) so decidability is structural and
`by decide` works on exported literals; `IsContiguousTrip` is its Prop image. -/

/-- One rendered reservation: checkin epoch day + stay nights. -/
structure TripObs where
  day : Nat
  nights : Nat

/-- The contiguous-flow predicate over a trip's (checkin, nights) pairs:
every consecutive pair satisfies checkin' = checkin + nights (next checkin =
previous checkout, i.e. zero inter-leg gap) with strictly increasing days
(nights ≥ 1 makes the increase strict). -/
def contigB : List TripObs → Bool
  | [] => true
  | [_] => true
  | a :: b :: rest =>
      (a.day < b.day && decide (b.day = a.day + a.nights) && contigB (b :: rest))

/-- `IsContiguousTrip obs`: the Bool predicate holds. -/
def IsContiguousTrip (obs : List TripObs) : Prop := contigB obs = true

instance (obs : List TripObs) : Decidable (IsContiguousTrip obs) := by
  unfold IsContiguousTrip; infer_instance

/-- The canonical contiguous trip (one night per leg, zero gaps). -/
theorem isContiguousTrip_canonical :
    IsContiguousTrip [{ day := 0, nights := 1 }, { day := 1, nights := 2 },
                      { day := 3, nights := 1 }] := by decide

/-- Destructuring lemma: the Bool predicate unfolds to exactly the pairwise
condition plus the tail condition (right-associated). -/
theorem contigB_cons {a b : TripObs} {rest : List TripObs} :
    contigB (a :: b :: rest) = true ↔
      a.day < b.day ∧ b.day = a.day + a.nights ∧ contigB (b :: rest) = true := by
  simp only [contigB, Bool.and_eq_true, decide_eq_true_eq]
  simp [and_assoc]

/-- All later observations of a contiguous trip happen strictly after the
first one's checkin. -/
theorem contiguous_head_lt : ∀ (a : TripObs) (rest : List TripObs),
    contigB (a :: rest) = true → ∀ x ∈ rest, a.day < x.day
  | _, [], _, x, hx => by simp at hx
  | a, b :: rest', h, x, hx => by
      obtain ⟨h1, _, h3⟩ := contigB_cons.mp h
      rcases List.mem_cons.mp hx with rfl | hx
      · exact h1
      · have hlt := contiguous_head_lt b rest' h3 x hx
        omega

/-- **Strict ordering**: a contiguous trip has strictly increasing checkin
days (`List.Pairwise (· < ·)` over the day column). -/
theorem contiguous_sorted : ∀ (obs : List TripObs),
    contigB obs = true → (obs.map (·.day)).Pairwise (· < ·)
  | [], _ => List.Pairwise.nil
  | [a], _ =>
    List.Pairwise.cons (fun x hx => by simp at hx) List.Pairwise.nil
  | a :: b :: rest', h => by
    obtain ⟨h1, _, h3⟩ := contigB_cons.mp h
    have ih := contiguous_sorted (b :: rest') h3
    simp only [List.map_cons] at ⊢
    refine List.Pairwise.cons ?_ ih
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hx
    · exact h1
    · obtain ⟨o, ho, rfl⟩ := List.mem_map.mp hx
      exact contiguous_head_lt a (b :: rest') h o
        (List.mem_cons_of_mem _ ho)

end EnclaveTrip

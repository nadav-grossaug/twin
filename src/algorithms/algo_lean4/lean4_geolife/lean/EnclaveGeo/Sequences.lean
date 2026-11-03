/-
Formal temporal flow logic: visit sequences.

A visit sequence is a run of stay intervals `(arrive, leave)` that is
chronologically ordered (each stay begins after the previous one ends) with
every observation gap bounded by `maxGap` (`visit_gap_break_s`) — the formal
counterpart of feat_visits.seq_ok and mart_seq_transitions. A recording gap
larger than the bound BREAKS the sequence, so it never creates a false
transition. Stated with `List.IsChain`, so decidability and the structural
lemmas come cheap and the predicate is executable on exported visit
sequences (`lake exe checks`).
-/
import Mathlib.Data.List.Chain
import Mathlib.Tactic
import EnclaveGeo.Params

namespace EnclaveGeo

/-- A stay interval: arrival and departure epoch seconds. -/
abbrev StayIv := Nat × Nat

/-- Consecutive step of a visit sequence: the next stay begins after this one
ends, within the observation-gap bound. -/
def VisitStep (maxGap : Nat) (a b : StayIv) : Prop :=
  a.2 ≤ b.1 ∧ b.1 - a.2 ≤ maxGap

instance (maxGap : Nat) : DecidableRel (VisitStep maxGap) := fun _ _ => by
  unfold VisitStep; exact inferInstance

/-- A gap-chained visit sequence at an explicit gap bound. -/
def IsVisitSeqAt (maxGap : Nat) (vs : List StayIv) : Prop :=
  List.IsChain (VisitStep maxGap) vs

instance (maxGap : Nat) (vs : List StayIv) :
    Decidable (IsVisitSeqAt maxGap vs) := by
  unfold IsVisitSeqAt; exact inferInstance

/-- A visit sequence at the public spec bound `visit_gap_break_s`. -/
def IsVisitSeq (vs : List StayIv) : Prop :=
  IsVisitSeqAt Params.visitGapBreakS vs

instance (vs : List StayIv) : Decidable (IsVisitSeq vs) := by
  unfold IsVisitSeq; exact inferInstance

/-- Dropping the first visit preserves the sequence property. -/
theorem isVisitSeqAt_tail {g : Nat} {v : StayIv} {vs : List StayIv}
    (h : IsVisitSeqAt g (v :: vs)) : IsVisitSeqAt g vs :=
  List.IsChain.tail h

/-- Relaxing the gap bound preserves the sequence property (monotonicity). -/
theorem isVisitSeqAt_mono {g g' : Nat} (hg : g ≤ g') {vs : List StayIv}
    (h : IsVisitSeqAt g vs) : IsVisitSeqAt g' vs :=
  h.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩)

/-- A visit sequence is chronologically ordered: each arrival is not before
the previous departure. -/
theorem isVisitSeqAt_ordered {g : Nat} {vs : List StayIv}
    (h : IsVisitSeqAt g vs) : List.IsChain (fun a b => a.2 ≤ b.1) vs :=
  h.imp (fun _ _ hab => hab.1)

end EnclaveGeo

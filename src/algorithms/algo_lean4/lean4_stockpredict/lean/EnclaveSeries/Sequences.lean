/-
Formal temporal flow logic: timely sessions.

A "timely session" is a run of flow start times that is chronologically
ordered with every consecutive gap bounded by `maxGap` — the formal
counterpart of the SQL mart mart_seq_sessions (and the windowing rule of the
NextServiceGRU minimodel). Stated with `List.IsChain`, so decidability and the
structural lemmas come cheap and the predicate is executable on exported
sessions (`lake exe checks`).
-/
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace EnclaveSeries

/-- Consecutive step of a timely session: ordered, with a bounded gap. -/
def TimelyStep (maxGap : ℚ) (a b : ℚ) : Prop := a ≤ b ∧ b - a ≤ maxGap

instance (maxGap : ℚ) : DecidableRel (TimelyStep maxGap) := fun _ _ => by
  unfold TimelyStep; exact inferInstance

/-- A timely session: every consecutive pair of start times is a timely step. -/
def IsTimelySession (maxGap : ℚ) (times : List ℚ) : Prop :=
  List.IsChain (TimelyStep maxGap) times

instance (maxGap : ℚ) (times : List ℚ) :
    Decidable (IsTimelySession maxGap times) := by
  unfold IsTimelySession; exact inferInstance

/-- Dropping the first flow preserves the session property. -/
theorem isTimelySession_tail {g : ℚ} {t : ℚ} {ts : List ℚ}
    (h : IsTimelySession g (t :: ts)) : IsTimelySession g ts :=
  List.IsChain.tail h

/-- Relaxing the gap bound preserves the session property (monotonicity). -/
theorem isTimelySession_mono {g g' : ℚ} (hg : g ≤ g') {ts : List ℚ}
    (h : IsTimelySession g ts) : IsTimelySession g' ts :=
  h.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩)

/-- A timely session is chronologically ordered. -/
theorem isTimelySession_ordered {g : ℚ} {ts : List ℚ}
    (h : IsTimelySession g ts) : List.IsChain (· ≤ ·) ts :=
  h.imp (fun _ _ hab => hab.1)

end EnclaveSeries

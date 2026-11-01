/-
Formal temporal flow logic: travel episodes.

An "episode" is a run of transfer timestamps (epoch seconds) that is strictly
chronologically ordered with every consecutive gap bounded by `maxGap`
(`episode_gap_max_s`) — the formal counterpart of features.feat_episodes and
mart_episodes. Stated with `List.IsChain`, so decidability and the structural
lemmas come cheap and the predicate is executable on exported episodes
(`lake exe checks`).
-/
import Mathlib.Data.List.Chain
import Mathlib.Tactic
import EnclaveTravel.Params

namespace EnclaveTravel

/-- Consecutive step of an episode: strictly ordered, with a bounded gap. -/
def EpisodeStep (maxGap : Nat) (a b : Nat) : Prop := a < b ∧ b - a ≤ maxGap

instance (maxGap : Nat) : DecidableRel (EpisodeStep maxGap) := fun _ _ => by
  unfold EpisodeStep; exact inferInstance

/-- A gap-chained episode at an explicit gap bound: every consecutive pair of
transfer timestamps is an episode step. -/
def IsEpisodeAt (maxGap : Nat) (ts : List Nat) : Prop :=
  List.IsChain (EpisodeStep maxGap) ts

instance (maxGap : Nat) (ts : List Nat) : Decidable (IsEpisodeAt maxGap ts) := by
  unfold IsEpisodeAt; exact inferInstance

/-- A travel episode at the public spec bound `episode_gap_max_s`. -/
def IsEpisode (ts : List Nat) : Prop := IsEpisodeAt Params.episodeGapMaxS ts

instance (ts : List Nat) : Decidable (IsEpisode ts) := by
  unfold IsEpisode; exact inferInstance

/-- Dropping the first transfer preserves the episode property. -/
theorem isEpisodeAt_tail {g : Nat} {t : Nat} {ts : List Nat}
    (h : IsEpisodeAt g (t :: ts)) : IsEpisodeAt g ts :=
  List.IsChain.tail h

/-- Relaxing the gap bound preserves the episode property (monotonicity). -/
theorem isEpisodeAt_mono {g g' : Nat} (hg : g ≤ g') {ts : List Nat}
    (h : IsEpisodeAt g ts) : IsEpisodeAt g' ts :=
  h.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩)

/-- An episode is strictly chronologically ordered. -/
theorem isEpisodeAt_ordered {g : Nat} {ts : List Nat}
    (h : IsEpisodeAt g ts) : List.IsChain (· < ·) ts :=
  h.imp (fun _ _ hab => hab.1)

end EnclaveTravel

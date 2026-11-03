/-
Formal definitions of the mobility motifs computed by the dbt marts, over Nat
(epoch seconds, dwell seconds, day indices, anchor ids) and String (roles),
so every predicate is decidable and executable. `lake exe checks` evaluates
these very definitions on observations exported from the enclave store. All
thresholds come from the mirrored concept spec (Params.lean).

The classifier functions below ARE the spec of the SQL marts: dwell/departure/
displacement/speed/staypoints-per-day class assignment, the day-chain
splitting, and the daily-motif priority (tour > commute > errand_loop >
home_day > irregular). Only the TSV/JSON export wrappers are unverified.
-/
import Mathlib.Data.List.Chain
import Mathlib.Data.List.Dedup
import Mathlib.Tactic
import EnclaveGeo.Params

namespace EnclaveGeo

/-! ### Public classifier mirrors (SQL `case` ladders restated over Nat) -/

/-- Dwell class of a stay of `d` seconds (mirror of feat_visits.dwell_class). -/
def dwellClass (d : Nat) : String :=
  if d < Params.dwE1S then "dw_lt1h"
  else if d < Params.dwE2S then "dw_1_3h"
  else if d < Params.dwE3S then "dw_3_8h"
  else if d < Params.dwE4S then "dw_8_14h"
  else "dw_ge14h"

/-- Hour bin of a local hour (mirror of arr_bin / dep_bin / hour_bin). -/
def hourBin (h : Nat) : String :=
  if h ≤ Params.depNightMaxH then "dep_night"
  else if h ≤ Params.depMorningMaxH then "dep_morning"
  else if h ≤ Params.depAfternoonMaxH then "dep_afternoon"
  else "dep_evening"

/-- Trip-displacement class in meters (mirror of feat_visits.dist_class). -/
def distClass (m : Nat) : String :=
  if m < Params.distE1M then "dist_lt1k"
  else if m < Params.distE2M then "dist_1_3k"
  else if m < Params.distE3M then "dist_3_10k"
  else "dist_ge10k"

/-- Speed class in milli-m/s (mirror of feat_steps.speed_class). -/
def speedClass (mmps : Nat) : String :=
  if mmps < Params.spdStillMaxMmps then "spd_still"
  else if mmps < Params.spdWalkMaxMmps then "spd_walk"
  else if mmps < Params.spdCityMaxMmps then "spd_city"
  else if mmps < Params.spdFastMaxMmps then "spd_fast"
  else "spd_vfast"

/-- Staypoints-per-day bin (mirror of feat_user_days.spd_bin). -/
def spdBin (n : Nat) : String :=
  if n < Params.spdbinE2 then "spd_1"
  else if n < Params.spdbinE3 then "spd_2"
  else if n < Params.spdbinE4 then "spd_3"
  else if n < Params.spdbinE6 then "spd_4_5"
  else "spd_6p"

/-- Segment-length class of a junction-to-junction stretch, metres (mirror
of feat_route_segments.seg_class). -/
def segClass (m : Nat) : String :=
  if m < Params.segE1M then "seg_lt50"
  else if m < Params.segE2M then "seg_50_150"
  else if m < Params.segE3M then "seg_150_600"
  else "seg_ge600"

/-- Junction-angle class, absolute degrees (mirror of
feat_route_segments.ja_class; only |turn| ≥ junction_min_turn_deg rows
exist). -/
def jaClass (absDeg : Nat) : String :=
  if absDeg < Params.jaE1Deg then "ja_30_60"
  else if absDeg < Params.jaE2Deg then "ja_60_120"
  else "ja_120_180"

/-- Circuity class, permille of path/net displacement (mirror of
feat_route_legs.circ_class). -/
def circClass (pm : Nat) : String :=
  if pm < Params.circE1Pm then "circ_lt120"
  else if pm < Params.circE2Pm then "circ_120_150"
  else if pm < Params.circE3Pm then "circ_150_250"
  else "circ_ge250"

/-! ### Stays and visits -/

/-- A stay qualifies as a staypoint: at least the public minimum dwell. -/
def IsStay (dwell : Nat) : Prop := Params.spMinDwellS ≤ dwell

instance (dwell : Nat) : Decidable (IsStay dwell) := by
  unfold IsStay; exact inferInstance

/-- A long dwell (home-base signal). -/
def IsLongDwell (dwell : Nat) : Prop := Params.longdwellMinS ≤ dwell

instance (dwell : Nat) : Decidable (IsLongDwell dwell) := by
  unfold IsLongDwell; exact inferInstance

/-- One observed anchor visit, carrying its predecessor edge (the previous
visit's role and leave time — the data a transition is decided from). -/
structure VisitObs where
  role : String
  anchor : Nat
  arrive : Nat
  leave : Nat
  prevRole : Option String := none
  prevLeave : Option Nat := none
  deriving Repr, DecidableEq

/-- The predecessor edge is sequence-ok: the previous visit ended before this
one began, within the public observation-gap bound (mirror of seq_ok —
a recording gap never creates a false transition). -/
def transOk (v : VisitObs) : Bool :=
  match v.prevLeave with
  | none => false
  | some pl => pl ≤ v.arrive && v.arrive - pl ≤ Params.visitGapBreakS

/-- The transition this visit realizes, when its edge is sequence-ok
(mirror of feat_visits.trans_cell). -/
def transCell (v : VisitObs) : Option (String × String) :=
  match v.prevRole with
  | none => none
  | some pr => if transOk v then some (pr, v.role) else none

/-! ### Day chains and the daily-motif classifier -/

/-- Split a day's arrival-ordered visits into chains: a new chain starts
whenever the gap from the previous visit exceeds `tour_max_gap_s` (mirror of
feat_visits.day_chain_id; the day's first visit always starts a chain). -/
def chainsAux (cur : List VisitObs) : List VisitObs → List (List VisitObs)
  | [] => [cur.reverse]
  | b :: rest =>
    match cur with
    | [] => chainsAux [b] rest
    | a :: _ =>
      if a.leave ≤ b.arrive ∧ b.arrive - a.leave ≤ Params.tourMaxGapS then
        chainsAux (b :: cur) rest
      else
        cur.reverse :: chainsAux [b] rest

def dayChains (visits : List VisitObs) : List (List VisitObs) :=
  match visits with
  | [] => []
  | v :: rest => chainsAux [v] rest

/-- Distinct non-home anchors reached inside one chain. -/
def nonHomeAnchors (chain : List VisitObs) : Nat :=
  ((chain.filter (fun v => v.role ≠ "home")).map (·.anchor)).dedup.length

/-- Tour day: some day chain reaches `tour_min_anchors` distinct non-home
anchors (the rarest planted pattern). -/
def isTourDay (visits : List VisitObs) : Bool :=
  (dayChains visits).any (fun c => Params.tourMinAnchors ≤ nonHomeAnchors c)

/-- Commute day: the day realizes a home→work AND a work→home transition. -/
def isCommuteDay (visits : List VisitObs) : Bool :=
  visits.any (fun v => transCell v = some ("home", "work")) &&
  visits.any (fun v => transCell v = some ("work", "home"))

/-- Errand pair: an other-anchor stop under the dwell cap reached from home,
followed (next visit, same day list) by a sequence-ok return home. -/
def isErrandPair (a b : VisitObs) : Bool :=
  transCell a = some ("home", "other") &&
  a.leave - a.arrive < Params.errandMaxDwellS &&
  transCell b = some ("other", "home")

def hasErrandLoop : List VisitObs → Bool
  | a :: b :: rest => isErrandPair a b || hasErrandLoop (b :: rest)
  | _ => false

/-- Home day: every visit that day is at the home anchor. -/
def isHomeDay (visits : List VisitObs) : Bool :=
  !visits.isEmpty && visits.all (fun v => v.role = "home")

/-- The daily-motif classifier (mirror of feat_user_days.motif_class):
mutually exclusive classes with the public priority order
tour > commute > errand_loop > home_day > irregular. -/
def motifClassOfDay (visits : List VisitObs) : String :=
  if isTourDay visits then "tour"
  else if isCommuteDay visits then "commute"
  else if hasErrandLoop visits then "errand_loop"
  else if isHomeDay visits then "home_day"
  else "irregular"

/-- The classifier is total over the public class list (kernel-checked shape
guarantee: no day escapes the 5-cell DP grid). -/
theorem motifClassOfDay_mem (visits : List VisitObs) :
    motifClassOfDay visits ∈ Params.motifClasses := by
  unfold motifClassOfDay Params.motifClasses
  repeat' split
  all_goals simp

/-! ### Canonical daily patterns (sanity: each definition accepts the motif
it was designed around; all kernel-checked by `decide`) -/

private def vHome (arrive leave : Nat) (pr : Option String := none)
    (pl : Option Nat := none) : VisitObs :=
  { role := "home", anchor := 1, arrive, leave, prevRole := pr, prevLeave := pl }

/-- The canonical commute day H(night)→W(9-17)→H(evening). -/
theorem commute_canonical :
    motifClassOfDay
      [ { role := "work", anchor := 2, arrive := 32400, leave := 61200,
          prevRole := some "home", prevLeave := some 30600 },
        vHome 63000 86000 (some "work") (some 61200) ] = "commute" := by
  decide

/-- The canonical 3-stop tour classifies as a tour even when the day also
commutes (priority sanity). -/
theorem tour_priority_canonical :
    motifClassOfDay
      [ { role := "work", anchor := 2, arrive := 32400, leave := 36000,
          prevRole := some "home", prevLeave := some 30600 },
        { role := "other", anchor := 3, arrive := 37800, leave := 39600,
          prevRole := some "work", prevLeave := some 36000 },
        { role := "other", anchor := 4, arrive := 41400, leave := 43200,
          prevRole := some "other", prevLeave := some 39600 },
        vHome 45000 86000 (some "other") (some 43200) ] = "tour" := by
  decide

/-- The canonical errand loop H→O(<3h)→H. -/
theorem errand_canonical :
    motifClassOfDay
      [ { role := "other", anchor := 5, arrive := 36000, leave := 39600,
          prevRole := some "home", prevLeave := some 34200 },
        vHome 41400 86000 (some "other") (some 39600) ] = "errand_loop" := by
  decide

/-- A recording gap larger than the break bound never creates a transition:
the same commute-shaped day with a 7-hour hole classifies irregular. -/
theorem gap_breaks_commute :
    motifClassOfDay
      [ { role := "work", anchor := 2, arrive := 32400, leave := 61200,
          prevRole := some "home", prevLeave := some 5000 },
        vHome 87000 100000 (some "work") (some 61200) ] = "irregular" := by
  decide

/-! ### Revisit beacon -/

/-- Revisit beacon at explicit thresholds: some visit day anchors a window of
`window` days containing at least `minDays` distinct visit days. -/
def IsRevisitAt (minDays window : Nat) (days : List Nat) : Prop :=
  ∃ d ∈ days, minDays ≤ (days.dedup.filter (fun d' => d ≤ d' ∧ d' < d + window)).length

instance (minDays window : Nat) (days : List Nat) :
    Decidable (IsRevisitAt minDays window days) := by
  unfold IsRevisitAt; exact List.decidableBEx _ days

/-- A revisit beacon at the public spec thresholds. -/
def IsRevisit (days : List Nat) : Prop :=
  IsRevisitAt Params.revisitMinDays Params.revisitWindowDays days

instance (days : List Nat) : Decidable (IsRevisit days) := by
  unfold IsRevisit; exact inferInstance

/-- The canonical 5-weekday revisit is a beacon (sanity). -/
theorem isRevisit_canonical : IsRevisit [100, 101, 102, 103, 104] := by decide

/-- Lowering the day threshold preserves beacon-hood (monotonicity). -/
theorem isRevisitAt_mono {m m' w : Nat} {days : List Nat} (hm : m' ≤ m)
    (h : IsRevisitAt m w days) : IsRevisitAt m' w days := by
  obtain ⟨d, hd, hc⟩ := h
  exact ⟨d, hd, le_trans hm hc⟩

/-- Widening the window preserves beacon-hood (monotonicity). -/
theorem isRevisitAt_mono_window {m w w' : Nat} {days : List Nat} (hw : w ≤ w')
    (h : IsRevisitAt m w days) : IsRevisitAt m w' days := by
  obtain ⟨d, hd, hc⟩ := h
  refine ⟨d, hd, le_trans hc (List.Sublist.length_le ?_)⟩
  exact List.monotone_filter_right _ (by
    intro d' hd'
    simp only [decide_eq_true_eq] at *
    exact ⟨hd'.1, lt_of_lt_of_le hd'.2 (Nat.add_le_add_left hw d)⟩)

end EnclaveGeo

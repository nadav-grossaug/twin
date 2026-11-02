/-
The restricted generative specification (genspec) and its semantics for the
trajectory generator.

`python/distill.py` distills the release into a TYPED spec — distributions
(DP-released histograms: daily motifs, role-transition rows, dwell classes,
departure bins, mode mix) plus the category-2 suppressed summaries (turn-bin
histogram, displacement mix, per-mode kinematics, GPS-noise moments), the
public window and bin ranges — never arbitrary code. This module gives the
spec its Lean types, JSON parsing, and executable sampling semantics on top
of the VERIFIED sampler (`sampleCellPMF`, law `sampleCellPMF_law`:
P(i) = wᵢ/Σw).

Structural properties proved here (the generator's plan invariants, by
construction on the rendering function `mkStays`):
  * `mkStays_times_sorted` — positive gaps and dwells ⇒ strictly increasing
    arrive/leave times down the day (`List.Pairwise (· < ·)`);
  * `mkStays_visitSeq`     — gap-bounded plans render stay intervals that
    satisfy `IsVisitSeqAt maxGap` (Sequences.lean): the planted day IS a
    sequence-ok visit chain, so re-ingestion recovers its transitions;
  * `mkStays_dwell_ge`     — the smart constructor clamps every dwell to
    `genStayMinDwellS`, which clears the public staypoint threshold
    (`genStay_isStay` in Motion.lean);
  * `mkStays_le`           — every rendered time is bounded by the start plus
    the plan's total duration (the window-containment hypothesis);
  * `stepTimes_sorted`     — the reusable gap-accumulation lemma (rendered
    point timestamps inside trips and stays);
  * `genChain` is structurally recursive on fuel — bounded generation
    terminates by construction.
-/
import EnclaveGeo.Sampler
import EnclaveGeo.Motifs
import EnclaveGeo.Sequences
import EnclaveGeo.Motion
import Lean.Data.Json

namespace EnclaveGeo.Gen

open Lean SLang PMF EnclaveGeo

/-! ### Spec types -/

structure Cell where
  name : String
  w : Nat

structure Hist where
  cells : List Cell

def Hist.weights (h : Hist) : List Nat := h.cells.map (·.w)

structure CondHist where
  given : String
  hist : Hist

/-- A public inclusive [lo, hi] integer range. -/
structure RangeN where
  lo : Nat
  hi : Nat

/-- Per-mode released kinematic summary (category-2): cruise speed (mm/s)
and sampling interval (deciseconds). -/
structure ModeKin where
  mode : String
  speedMmps : Nat
  dtDs : Nat

/-- Released GPS-noise moments (category-2, tight staypoints): residual sd
(mm) and lag-1 autocorrelation (percent). -/
structure NoiseMoments where
  sdMm : Nat
  lag1Pct : Nat

/-- Per-mode released route shape (category-2): median circuity (permille of
path over net displacement) and junction density (junctions per km × 100). -/
structure RouteCirc where
  mode : String
  circPm : Nat
  juncPkmC : Nat := 300

structure GenSpec where
  version : String
  nUsersHint : Nat
  motif : Hist                  -- daily-motif classes (DP)
  transitions : List CondHist   -- role -> role rows (DP + priors)
  dwell : Hist                  -- dwell classes (DP)
  departure : Hist              -- departure-hour bins (DP)
  mode : Hist                   -- mode-class mix (DP)
  turn : Hist                   -- 45° turn bins (category-2)
  segHist : Hist                -- junction-to-junction segment classes (category-2)
  junctionHist : Hist           -- junction-angle classes (category-2)
  shareLeftPct : Nat            -- left-turn share at junctions (category-2)
  routeCirc : List RouteCirc    -- per-mode median circuity (category-2)
  distMix : Hist                -- displacement classes (category-2)
  kinematics : List ModeKin     -- per-mode summaries (category-2)
  noise : NoiseMoments          -- GPS-noise moments (category-2)
  windowStartS : Nat            -- public export window [start, end)
  windowEndS : Nat
  dwellRangesS : List (String × RangeN)
  depRangesH : List (String × RangeN)
  distRangesM : List (String × RangeN)

/-! ### JSON parsing (the spec is data, never code) -/

def parseCellList (j : Json) : Except String (List Cell) := do
  let arr ← j.getArr?
  arr.toList.mapM fun c => do
    pure { name := (← (← c.getObjVal? "name").getStr?),
           w := (← (← c.getObjVal? "w").getNat?) }

def parseHist (j : Json) (k : String) : Except String Hist := do
  pure ⟨← parseCellList (← (← j.getObjVal? k).getObjVal? "cells")⟩

def parseCond (j : Json) (k : String) : Except String (List CondHist) := do
  let arr ← (← (← j.getObjVal? k).getObjVal? "rows").getArr?
  arr.toList.mapM fun t => do
    pure ({ given := (← (← t.getObjVal? "given").getStr?),
            hist := ⟨← parseCellList (← t.getObjVal? "cells")⟩ } : CondHist)

def parseKin (j : Json) : Except String (List ModeKin) := do
  let arr ← (← j.getObjVal? "kinematics").getArr?
  arr.toList.mapM fun c => do
    pure { mode := (← (← c.getObjVal? "mode").getStr?),
           speedMmps := (← (← c.getObjVal? "speed_mmps").getNat?),
           dtDs := (← (← c.getObjVal? "dt_ds").getNat?) }

def parseNoise (j : Json) : Except String NoiseMoments := do
  let n ← j.getObjVal? "noise"
  pure { sdMm := (← (← n.getObjVal? "sd_mm").getNat?),
         lag1Pct := (← (← n.getObjVal? "lag1_pct").getNat?) }

def parseRoutes (j : Json) : Except String (Nat × List RouteCirc) := do
  let r ← j.getObjVal? "routes"
  let share ← (← r.getObjVal? "share_left_pct").getNat?
  let arr ← (← r.getObjVal? "circ_pm").getArr?
  let circs ← arr.toList.mapM fun c => do
    pure ({ mode := (← (← c.getObjVal? "mode").getStr?),
            circPm := (← (← c.getObjVal? "circ_pm").getNat?),
            juncPkmC := (((c.getObjVal? "junc_pkm_c").bind
              Json.getNat?).toOption.getD 300) } : RouteCirc)
  pure (share, circs)

/-- Parse a `{name: [lo, hi], ...}` range object. -/
def parseRangeList (j : Json) (k : String) : Except String (List (String × RangeN)) := do
  let obj ← (← j.getObjVal? k).getObj?
  obj.toList.mapM fun (key, v) => do
    let arr ← v.getArr?
    let lo ← (arr[0]?.getD Json.null).getNat?
    let hi ← (arr[1]?.getD Json.null).getNat?
    pure (key, ({ lo, hi } : RangeN))

def GenSpec.parse (j : Json) : Except String GenSpec := do
  let w ← j.getObjVal? "window"
  let (shareLeft, circs) ← parseRoutes j
  pure { version := (← (← j.getObjVal? "spec_version").getStr?),
         nUsersHint := (← (← j.getObjVal? "n_users_hint").getNat?),
         motif := ← parseHist j "motif",
         transitions := ← parseCond j "transitions",
         dwell := ← parseHist j "dwell",
         departure := ← parseHist j "departure",
         mode := ← parseHist j "mode",
         turn := ← parseHist j "turn",
         segHist := ← parseHist j "seg_hist",
         junctionHist := ← parseHist j "junction_hist",
         shareLeftPct := shareLeft,
         routeCirc := circs,
         distMix := ← parseHist j "dist_mix",
         kinematics := ← parseKin j,
         noise := ← parseNoise j,
         windowStartS := (← (← w.getObjVal? "start_s").getNat?),
         windowEndS := (← (← w.getObjVal? "end_s").getNat?),
         dwellRangesS := ← parseRangeList j "dwell_ranges_s",
         depRangesH := ← parseRangeList j "dep_ranges_h",
         distRangesM := ← parseRangeList j "dist_ranges_m" }

/-! ### Executable sampling semantics -/

/-- One verified draw: index `i` with probability `wᵢ/Σw` (`sampleCellPMF_law`).
An all-zero weight vector falls back to a uniform draw (declared in meta). -/
def draw (ws : List Nat) : IO Nat := do
  let ws' := if ws.sum = 0 then ws.map (fun _ => 1) else ws
  if h : 0 < ws'.sum then
    run (sampleCellPMF ws' h)
  else
    pure 0

def Hist.choose (h : Hist) : IO Cell := do
  let i ← draw h.weights
  pure (h.cells.getD i ⟨"", 0⟩)

/-- Conditional row lookup with marginal fallback (all-zero or missing rows
fall back to the marginal histogram — declared). -/
def rowOf (rows : List CondHist) (fallback : Hist) (prev : String) : Hist :=
  match rows.find? (·.given = prev) with
  | some c => if c.hist.weights.sum = 0 then fallback else c.hist
  | none => fallback

/-- Bounded state sequence: structurally recursive on fuel, so bounded
generation requests terminate by construction. -/
def genChain (rows : List CondHist) (fallback : Hist) :
    Nat → String → IO (List String)
  | 0, _ => pure []
  | n + 1, prev => do
    let c ← (rowOf rows fallback prev).choose
    let rest ← genChain rows fallback n c.name
    pure (c.name :: rest)

/-! ### Reusable structural properties -/

/-- Event times accumulated from gaps: `t0, t0+g₁, t0+g₁+g₂, …`. -/
def stepTimes (t0 : Nat) : List Nat → List Nat
  | [] => [t0]
  | g :: gs => t0 :: stepTimes (t0 + g) gs

theorem stepTimes_head (t0 : Nat) (gs : List Nat) :
    (stepTimes t0 gs).head? = some t0 := by
  cases gs <;> rfl

theorem le_of_mem_stepTimes : ∀ (gs : List Nat) (t0 x : Nat),
    x ∈ stepTimes t0 gs → t0 ≤ x
  | [], t0, x, hx => by
    simp only [stepTimes, List.mem_singleton] at hx
    omega
  | g :: gs, t0, x, hx => by
    simp only [stepTimes, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact Nat.le_refl _
    · have := le_of_mem_stepTimes gs (t0 + g) x hx
      omega

/-- **Event-ordering preservation**: strictly positive gaps produce strictly
increasing event times (pairwise, so every pair is ordered). -/
theorem stepTimes_sorted (t0 : Nat) (gs : List Nat) (hp : ∀ g ∈ gs, 0 < g) :
    (stepTimes t0 gs).Pairwise (· < ·) := by
  induction gs generalizing t0 with
  | nil =>
    exact List.Pairwise.cons (by intro a ha; simp at ha) List.Pairwise.nil
  | cons g gs ih =>
    have hg : 0 < g := hp g (by simp)
    simp only [stepTimes]
    refine List.Pairwise.cons ?_ (ih (t0 + g) (fun x hx => hp x (by simp [hx])))
    intro y hy
    have := le_of_mem_stepTimes gs (t0 + g) y hy
    omega

/-! ### Stay-plan rendering, valid by construction -/

/-- One planned stay of a user-day: anchor role and id, travel gap (s) since
the previous stay's departure, and the dwell (s) at the anchor. -/
structure StayPlan where
  role : String
  anchor : Nat
  gap : Nat
  dwell : Nat

/-- One rendered stay (the plan image of a visit; point rendering attaches
coordinates at I/O time from the user's local anchor frame). -/
structure Stay where
  role : String
  anchor : Nat
  arrive : Nat
  leave : Nat

/-- Smart stay constructor: the arrival is DERIVED from the previous
departure plus a positive travel gap, and the dwell is clamped up to the
generator's minimum (`genStayMinDwellS`), which clears the public staypoint
threshold — stays can never be planted too short or out of order. -/
def mkStay (tPrevLeave : Nat) (p : StayPlan) : Stay :=
  { role := p.role, anchor := p.anchor,
    arrive := tPrevLeave + max 1 p.gap,
    leave := tPrevLeave + max 1 p.gap + max Shape.genStayMinDwellS p.dwell }

/-- Render a day's stay chain from its plan: each stay starts where the
previous one ended (times are threaded, never passed in). -/
def mkStays (tPrevLeave : Nat) : List StayPlan → List Stay
  | [] => []
  | p :: rest =>
    let s := mkStay tPrevLeave p
    s :: mkStays s.leave rest

/-- Every rendered stay dwells at least the generator minimum — hence at
least the public staypoint threshold (`Shape.genStay_isStay`). -/
theorem mkStays_dwell_ge : ∀ (ps : List StayPlan) (t : Nat),
    ∀ s ∈ mkStays t ps, Shape.genStayMinDwellS ≤ s.leave - s.arrive
  | [], _, s, hs => by simp [mkStays] at hs
  | p :: rest, t, s, hs => by
    simp only [mkStays, List.mem_cons] at hs
    rcases hs with rfl | hs
    · simp only [mkStay]; omega
    · exact mkStays_dwell_ge rest _ s hs

/-- Every rendered stay is a staypoint by the public spec. -/
theorem mkStays_isStay (ps : List StayPlan) (t : Nat) :
    ∀ s ∈ mkStays t ps, IsStay (s.leave - s.arrive) := fun s hs =>
  le_trans Shape.genStay_isStay (mkStays_dwell_ge ps t s hs)

/-- Arrivals strictly follow the previous leave (auxiliary). -/
theorem mkStays_ge : ∀ (ps : List StayPlan) (t : Nat),
    ∀ s ∈ mkStays t ps, t < s.arrive ∧ s.arrive < s.leave
  | [], _, s, hs => by simp [mkStays] at hs
  | p :: rest, t, s, hs => by
    simp only [mkStays, List.mem_cons] at hs
    rcases hs with rfl | hs
    · constructor
      · simp only [mkStay]; omega
      · simp only [mkStay]
        have : 0 < Shape.genStayMinDwellS := by decide
        omega
    · have h := mkStays_ge rest (mkStay t p).leave s hs
      have ht : t < (mkStay t p).leave := by
        simp only [mkStay]
        have : 0 < Shape.genStayMinDwellS := by decide
        omega
      exact ⟨lt_trans ht h.1, h.2⟩

/-- **Per-day time ordering**: the interleaved arrive/leave times of a
rendered day are strictly increasing. -/
theorem mkStays_times_sorted (ps : List StayPlan) (t : Nat) :
    ((mkStays t ps).flatMap (fun s => [s.arrive, s.leave])).Pairwise (· < ·) := by
  induction ps generalizing t with
  | nil => simp [mkStays]
  | cons p rest ih =>
    simp only [mkStays, List.flatMap_cons]
    have harr : (mkStay t p).arrive < (mkStay t p).leave := (by
      simp only [mkStay]
      have : 0 < Shape.genStayMinDwellS := by decide
      omega)
    refine List.pairwise_append.mpr ⟨?_, ih (mkStay t p).leave, ?_⟩
    · exact List.pairwise_pair.mpr harr
    · intro a ha b hb
      obtain ⟨s, hs, hsb⟩ : ∃ s ∈ mkStays (mkStay t p).leave rest,
          b = s.arrive ∨ b = s.leave := by
        simp only [List.mem_flatMap] at hb
        obtain ⟨s, hs, hbs⟩ := hb
        exact ⟨s, hs, by simpa using hbs⟩
      have hgt := mkStays_ge rest (mkStay t p).leave s hs
      have hab : a = (mkStay t p).arrive ∨ a = (mkStay t p).leave := by
        simpa using ha
      rcases hab with rfl | rfl <;> rcases hsb with rfl | rfl <;> omega

/-- **Sequence-ok by construction**: if every planned gap is within `maxGap`
(and positive), the rendered stay intervals form a sequence-ok visit chain
(`IsVisitSeqAt` from Sequences.lean) — re-ingestion recovers every planted
transition. -/
theorem mkStays_visitSeq : ∀ (ps : List StayPlan) (t : Nat) (maxGap : Nat),
    (∀ p ∈ ps, 0 < p.gap ∧ p.gap ≤ maxGap) →
    IsVisitSeqAt maxGap ((mkStays t ps).map (fun s => (s.arrive, s.leave)))
  | [], _, _, _ => List.IsChain.nil
  | [p], t, _, _ => by
    simp only [mkStays, List.map_cons, List.map_nil]
    exact List.isChain_singleton _
  | p :: q :: rest, t, maxGap, hp => by
    have hq := mkStays_visitSeq (q :: rest) (mkStay t p).leave maxGap
      (fun x hx => hp x (by simp [hx]))
    simp only [mkStays, List.map_cons] at hq ⊢
    refine List.IsChain.cons hq ?_
    intro y hy
    simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hy
    subst hy
    have hqg := (hp q (by simp)).1
    have hqle := (hp q (by simp)).2
    have hmax : max 1 q.gap = q.gap := Nat.max_eq_right hqg
    refine ⟨?_, ?_⟩ <;> simp only [mkStay, hmax] <;> omega

/-- Every rendered time is bounded by the start plus the plan's total
duration (each stay costs at most `max 1 gap + max minDwell dwell`). -/
def planCost (p : StayPlan) : Nat :=
  max 1 p.gap + max Shape.genStayMinDwellS p.dwell

theorem mkStays_le : ∀ (ps : List StayPlan) (t : Nat),
    ∀ s ∈ mkStays t ps, s.leave ≤ t + (ps.map planCost).sum
  | [], _, s, hs => by simp [mkStays] at hs
  | p :: rest, t, s, hs => by
    simp only [mkStays, List.mem_cons] at hs
    simp only [List.map_cons, List.sum_cons]
    rcases hs with rfl | hs
    · simp only [mkStay, planCost]; omega
    · have := mkStays_le rest (mkStay t p).leave s hs
      simp only [mkStay, planCost] at *
      omega

end EnclaveGeo.Gen

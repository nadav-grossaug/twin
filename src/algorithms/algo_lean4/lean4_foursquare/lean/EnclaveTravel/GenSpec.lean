/-
The restricted generative specification (genspec) and its semantics for the
travel-transfer generator.

`python/distill.py` distills the release into a TYPED spec — distributions
(DP-released histograms: travel_count, region_mix, per-region transition
rows, gap/duration/distance classes, months, episode lengths, return-rate
bands), the released top corridors, the public window and bin ranges —
never arbitrary code. This module gives the spec its Lean types, JSON
parsing, and executable sampling semantics on top of the VERIFIED sampler
(`sampleCellPMF`, law `sampleCellPMF_law`: P(i) = wᵢ/Σw).

Structural properties proved here (the generator's row invariants, by
construction on the rendering function `mkUserHistory`):
  * `mkUserHistory_times_sorted` — positive gaps ⇒ strictly increasing
    per-user timestamps (`List.Pairwise (· < ·)`);
  * `mkUserHistory_prev_ne_next` — adjacent-distinct hop plans render rows
    with prev_country ≠ next_country; `mkHops` builds only such plans;
  * `mkUserHistory_durChained`   — row i+1's prev_duration equals row i's
    next_duration: the renderer derives the prev side from the previous hop,
    never takes it as input (durations are `Nat`, hence ≥ 0 by type);
  * `mkUserHistory_ts_in_window` — all timestamps in [wStart, wEnd) when the
    start time and total gap are in range; `hopsWithin` clamps any hop plan
    into range (`hopsWithin_sum_lt`);
  * `stepTimes_sorted`           — the reusable gap-accumulation lemma;
  * `genChain` is structurally recursive on fuel — bounded generation
    terminates by construction.
-/
import EnclaveTravel.Sampler
import EnclaveTravel.Motifs
import Lean.Data.Json

namespace EnclaveTravel.Gen

open Lean SLang PMF EnclaveTravel

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

/-- A released top corridor: country pair plus its public region pair. -/
structure Corridor where
  a : String
  b : String
  regionA : String
  regionB : String

/-- A public inclusive [lo, hi] integer range. -/
structure RangeN where
  lo : Nat
  hi : Nat

structure GenSpec where
  version : String
  nUsersHint : Nat
  travelCount : Hist            -- travels-per-user bins (DP, clip 1)
  regionMix : Hist              -- arrival-region mix (DP)
  transitions : List CondHist   -- region -> region rows (DP + return-boost prior)
  gapClass : Hist               -- inter-transfer gap classes (DP)
  durationClass : Hist          -- stay-duration classes (DP)
  distanceClass : Hist          -- distance classes (DP; informational)
  month : Hist                  -- arrival months (DP; informational)
  episodeLen : Hist             -- episode-length bins (DP; informational)
  returnRate : Hist             -- return-rate bands (DP; informational)
  corridors : List Corridor     -- released top corridors
  windowStartS : Nat            -- public export window [start, end)
  windowEndS : Nat
  tcBinRanges : List (String × RangeN)
  gapRangesS : List (String × RangeN)
  durRangesS : List (String × RangeN)

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

def parseCorridors (j : Json) : Except String (List Corridor) := do
  let arr ← (← j.getObjVal? "corridors").getArr?
  arr.toList.mapM fun c => do
    pure { a := (← (← c.getObjVal? "a").getStr?),
           b := (← (← c.getObjVal? "b").getStr?),
           regionA := (← (← c.getObjVal? "region_a").getStr?),
           regionB := (← (← c.getObjVal? "region_b").getStr?) }

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
  pure { version := (← (← j.getObjVal? "spec_version").getStr?),
         nUsersHint := (← (← j.getObjVal? "n_users_hint").getNat?),
         travelCount := ← parseHist j "travel_count",
         regionMix := ← parseHist j "region_mix",
         transitions := ← parseCond j "transitions",
         gapClass := ← parseHist j "gap_class",
         durationClass := ← parseHist j "duration_class",
         distanceClass := ← parseHist j "distance_class",
         month := ← parseHist j "month",
         episodeLen := ← parseHist j "episode_len",
         returnRate := ← parseHist j "return_rate",
         corridors := ← parseCorridors j,
         windowStartS := (← (← w.getObjVal? "start_s").getNat?),
         windowEndS := (← (← w.getObjVal? "end_s").getNat?),
         tcBinRanges := ← parseRangeList j "tc_bin_ranges",
         gapRangesS := ← parseRangeList j "gap_ranges_s",
         durRangesS := ← parseRangeList j "dur_ranges_s" }

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

/-! ### Travel-row rendering, valid by construction -/

/-- One planned hop of a user's history: destination country, gap (s) since
the previous transfer, and the stay duration (s) at the destination. -/
structure Hop where
  country : String
  gap : Nat
  stay : Nat

/-- One rendered transfer row (the Lean image of a country_transfer.tsv row;
coordinates and venue ids are attached at I/O time from public tables).
Durations are `Nat`, so durations ≥ 0 holds by type. -/
structure TravelRow where
  ts : Nat
  prevCountry : String
  nextCountry : String
  prevDuration : Nat
  nextDuration : Nat

/-- Smart row constructor: the prev-side fields are the PREVIOUS hop's
next-side state, passed down the fold — they can never be set freely. -/
def mkTravelRow (prev : String) (prevDur t : Nat) (h : Hop) : TravelRow :=
  { ts := t + h.gap, prevCountry := prev, nextCountry := h.country,
    prevDuration := prevDur, nextDuration := h.stay }

/-- Render a user's history: row i+1's prev fields are DERIVED from row i's
next fields (never passed in), so duration chaining and country continuity
are structural. -/
def mkUserHistory (prev : String) (prevDur t : Nat) : List Hop → List TravelRow
  | [] => []
  | h :: rest =>
      mkTravelRow prev prevDur t h ::
        mkUserHistory h.country h.stay (t + h.gap) rest

/-- Every rendered timestamp is at least the start time. -/
theorem mkUserHistory_ts_ge : ∀ (hops : List Hop) (prev : String) (d t : Nat),
    ∀ r ∈ mkUserHistory prev d t hops, t ≤ r.ts
  | [], _, _, _, r, hr => by simp [mkUserHistory] at hr
  | h :: rest, prev, d, t, r, hr => by
    simp only [mkUserHistory, List.mem_cons] at hr
    rcases hr with rfl | hr
    · simp only [mkTravelRow]; omega
    · have := mkUserHistory_ts_ge rest h.country h.stay (t + h.gap) r hr
      omega

/-- With positive gaps, every rendered timestamp is strictly past the start. -/
theorem mkUserHistory_ts_lt : ∀ (hops : List Hop) (prev : String) (d t : Nat),
    (∀ h ∈ hops, 0 < h.gap) →
    ∀ r ∈ mkUserHistory prev d t hops, t < r.ts
  | [], _, _, _, _, r, hr => by simp [mkUserHistory] at hr
  | h :: rest, prev, d, t, hp, r, hr => by
    have hg : 0 < h.gap := hp h (by simp)
    simp only [mkUserHistory, List.mem_cons] at hr
    rcases hr with rfl | hr
    · simp only [mkTravelRow]; omega
    · have := mkUserHistory_ts_lt rest h.country h.stay (t + h.gap)
        (fun x hx => hp x (by simp [hx])) r hr
      omega

/-- **Per-user timestamp ordering**: positive gaps render a strictly
increasing timestamp column (`List.Pairwise (· < ·)`). -/
theorem mkUserHistory_times_sorted (prev : String) (d t : Nat) (hops : List Hop)
    (hp : ∀ h ∈ hops, 0 < h.gap) :
    ((mkUserHistory prev d t hops).map (·.ts)).Pairwise (· < ·) := by
  induction hops generalizing prev d t with
  | nil => simp [mkUserHistory]
  | cons h rest ih =>
    simp only [mkUserHistory, List.map_cons]
    refine List.Pairwise.cons ?_
      (ih h.country h.stay (t + h.gap) (fun x hx => hp x (by simp [hx])))
    intro y hy
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hy
    have := mkUserHistory_ts_lt rest h.country h.stay (t + h.gap)
      (fun x hx => hp x (by simp [hx])) r hr
    simp only [mkTravelRow]
    omega

/-- Adjacent distinctness of a hop plan: each hop's destination differs from
where the user currently is. -/
def DistinctFrom : String → List Hop → Prop
  | _, [] => True
  | p, h :: rest => p ≠ h.country ∧ DistinctFrom h.country rest

/-- **No self-transfers**: an adjacent-distinct hop plan renders rows with
prev_country ≠ next_country, every row. -/
theorem mkUserHistory_prev_ne_next : ∀ (hops : List Hop) (prev : String) (d t : Nat),
    DistinctFrom prev hops →
    ∀ r ∈ mkUserHistory prev d t hops, r.prevCountry ≠ r.nextCountry
  | [], _, _, _, _, r, hr => by simp [mkUserHistory] at hr
  | h :: rest, prev, d, t, hd, r, hr => by
    simp only [mkUserHistory, List.mem_cons] at hr
    rcases hr with rfl | hr
    · exact hd.1
    · exact mkUserHistory_prev_ne_next rest h.country h.stay (t + h.gap) hd.2 r hr

/-- Duration chaining: row i+1's prev_duration equals row i's next_duration
(stated structurally over adjacent pairs). -/
def DurChained : List TravelRow → Prop
  | r :: r' :: rest => r'.prevDuration = r.nextDuration ∧ DurChained (r' :: rest)
  | _ => True

/-- **Duration chaining by construction**: the renderer passes row i's
next-side state down as row i+1's prev side, so chaining always holds. -/
theorem mkUserHistory_durChained : ∀ (hops : List Hop) (prev : String) (d t : Nat),
    DurChained (mkUserHistory prev d t hops)
  | [], _, _, _ => trivial
  | [_], _, _, _ => trivial
  | h₁ :: h₂ :: rest, _, _, t =>
    ⟨rfl, mkUserHistory_durChained (h₂ :: rest) h₁.country h₁.stay (t + h₁.gap)⟩

/-- Every rendered timestamp is bounded by start + total gap. -/
theorem mkUserHistory_ts_le : ∀ (hops : List Hop) (prev : String) (d t : Nat),
    ∀ r ∈ mkUserHistory prev d t hops, r.ts ≤ t + (hops.map (·.gap)).sum
  | [], _, _, _, r, hr => by simp [mkUserHistory] at hr
  | h :: rest, prev, d, t, r, hr => by
    simp only [mkUserHistory, List.mem_cons] at hr
    simp only [List.map_cons, List.sum_cons]
    rcases hr with rfl | hr
    · simp only [mkTravelRow]; omega
    · have := mkUserHistory_ts_le rest h.country h.stay (t + h.gap) r hr
      omega

/-- **Window containment**: start at/after `wStart` with total gap ending
before `wEnd` puts every rendered timestamp in [wStart, wEnd). -/
theorem mkUserHistory_ts_in_window {wStart wEnd : Nat} (hops : List Hop)
    (prev : String) (d t : Nat) (hstart : wStart ≤ t)
    (hend : t + (hops.map (·.gap)).sum < wEnd) :
    ∀ r ∈ mkUserHistory prev d t hops, wStart ≤ r.ts ∧ r.ts < wEnd := by
  intro r hr
  have h1 := mkUserHistory_ts_ge hops prev d t r hr
  have h2 := mkUserHistory_ts_le hops prev d t r hr
  omega

/-! ### Smart hop-plan constructors (generator side)

The generator builds hop plans ONLY through `mkHops` (destinations forced
off the current location, gaps forced positive) and `hopsWithin` (window
clamp), so the hypotheses of the rendering theorems hold by construction. -/

/-- Choose a destination ≠ `prev`: keep the candidate when it already moves,
else fall back to the fixed public pair US/FR. -/
def pickNe (prev cand : String) : String :=
  if cand = prev then (if prev = "US" then "FR" else "US") else cand

theorem pickNe_ne (prev cand : String) : prev ≠ pickNe prev cand := by
  unfold pickNe
  split
  · split
    · next hus => rw [hus]; decide
    · next hus => exact hus
  · next hc => exact fun he => hc he.symm

/-- Build a hop plan from (candidate country, gap, stay) triples:
destinations via `pickNe`, gaps forced positive via `max 1`. -/
def mkHops : String → List (String × Nat × Nat) → List Hop
  | _, [] => []
  | prev, (c, g, s) :: rest =>
      ⟨pickNe prev c, max 1 g, s⟩ :: mkHops (pickNe prev c) rest

theorem mkHops_distinct : ∀ (prev : String) (xs : List (String × Nat × Nat)),
    DistinctFrom prev (mkHops prev xs)
  | _, [] => trivial
  | prev, (c, _, _) :: rest =>
    ⟨pickNe_ne prev c, mkHops_distinct (pickNe prev c) rest⟩

theorem mkHops_gap_pos : ∀ (prev : String) (xs : List (String × Nat × Nat)),
    ∀ h ∈ mkHops prev xs, 0 < h.gap
  | _, [], h, hh => by simp [mkHops] at hh
  | prev, (c, g, s) :: rest, h, hh => by
    simp only [mkHops, List.mem_cons] at hh
    rcases hh with rfl | hh
    · exact Nat.lt_of_lt_of_le Nat.one_pos (Nat.le_max_left 1 g)
    · exact mkHops_gap_pos (pickNe prev c) rest h hh

/-- Keep hops while the running timestamp stays strictly below `tEnd` — the
generator's window clamp (drops the tail, keeps a prefix). -/
def hopsWithin (tEnd : Nat) : Nat → List Hop → List Hop
  | _, [] => []
  | t, h :: rest =>
      if t + h.gap < tEnd then h :: hopsWithin tEnd (t + h.gap) rest else []

/-- The clamp works: starting below `tEnd`, the clamped plan's total gap
stays below `tEnd` (so `mkUserHistory_ts_in_window` applies). -/
theorem hopsWithin_sum_lt (tEnd : Nat) : ∀ (t : Nat) (hops : List Hop),
    t < tEnd → t + ((hopsWithin tEnd t hops).map (·.gap)).sum < tEnd
  | t, [], ht => by simpa [hopsWithin] using ht
  | t, h :: rest, ht => by
    simp only [hopsWithin]
    split
    · next hlt =>
      simp only [List.map_cons, List.sum_cons]
      have := hopsWithin_sum_lt tEnd (t + h.gap) rest hlt
      omega
    · simpa using ht

/-- Clamped hops are among the original hops (gap positivity transports). -/
theorem hopsWithin_subset (tEnd : Nat) : ∀ (t : Nat) (hops : List Hop),
    ∀ h ∈ hopsWithin tEnd t hops, h ∈ hops
  | _, [], h, hh => by simp [hopsWithin] at hh
  | t, h₀ :: rest, h, hh => by
    simp only [hopsWithin] at hh
    split at hh
    · simp only [List.mem_cons] at hh ⊢
      rcases hh with rfl | hh
      · exact Or.inl rfl
      · exact Or.inr (hopsWithin_subset tEnd (t + h₀.gap) rest h hh)
    · simp at hh

/-- Clamping preserves adjacent distinctness (it keeps a prefix). -/
theorem hopsWithin_distinctFrom (tEnd : Nat) : ∀ (t : Nat) (prev : String)
    (hops : List Hop), DistinctFrom prev hops →
    DistinctFrom prev (hopsWithin tEnd t hops)
  | _, _, [], _ => trivial
  | t, prev, h :: rest, hd => by
    simp only [hopsWithin]
    split
    · exact ⟨hd.1, hopsWithin_distinctFrom tEnd (t + h.gap) h.country rest hd.2⟩
    · trivial

end EnclaveTravel.Gen

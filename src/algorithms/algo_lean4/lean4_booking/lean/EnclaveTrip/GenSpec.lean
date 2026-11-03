/-
The restricted generative specification (genspec) and its semantics for the
booking-trip generator.

`python/distill.py` distills the release into a TYPED spec — distributions
(DP-released histograms: res_count, country_mix, per-tier country-transition
rows, nights, gap, month_window, device, affiliate, booker, return_rate,
repeat_struct, trips_per_user, device_persist, affiliate_persist), the
released top city-popularity list, the released top city transitions
(corridors), the suppressed gap moments, the public window and bin ranges —
never arbitrary code. This module gives the spec its Lean types, JSON
parsing, and executable sampling semantics on top of the VERIFIED sampler
(`sampleCellPMF`, law `sampleCellPMF_law`: P(i) = wᵢ/Σw).

Structural properties proved here (the generator's row invariants, by
construction on the rendering function `mkLegs`):
  * `mkLegs_checkins_sorted`  — stays of at least one night ⇒ strictly
    increasing per-trip checkin days (`List.Pairwise (· < ·)`);
  * `mkLegs_contiguous`       — a zero-gap leg plan renders the exact
    contiguous flow (checkin_{i+1} = checkin_i + nights_i, i.e. the next
    checkin equals the previous checkout — the dataset's 99.1% flow);
  * `mkLegs_in_window`        — rendered checkins/checkout stay in
    [wStart, wEnd] when the start day is in range and the plan's total span
    fits the window;
  * `mkLegs_no_overlap`       — a plan whose total span fits the window
    renders rows where every stay closes at or before the next checkin and
    the last checkout stays at or before the window end (no overlapping
    presence), with `legsWithin_span_le` carrying the clamp's span bound;
  * `mkLeg_gap_range` / `mkLeg_nights_range` — the public ranges
    [0, 33] / [1, 30] hold on every smart-constructed leg.
-/
import EnclaveTrip.Sampler
import EnclaveTrip.Motifs
import Lean.Data.Json

namespace EnclaveTrip.Gen

open Lean SLang PMF EnclaveTrip

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

/-- One entry of the released city-popularity list: city id, its hotel
country, its public tier, and the released weight. -/
structure City where
  cityId : Nat
  country : String
  ct : String
  w : Nat

/-- A released top city transition (corridor analog): ordered city pair plus
its public tier and suppression-passing counts. -/
structure CityCorridor where
  aCity : Nat
  bCity : Nat
  ct : String
  nUsers : Nat
  nTrans : Nat

/-- A suppressed gap-moment row (informational for the generator). -/
structure GapMoment where
  gapClass : String
  meanDays : Float
  sdDays : Float

/-- A public inclusive [lo, hi] integer range. -/
structure RangeN where
  lo : Nat
  hi : Nat

structure GenSpec where
  version : String
  nUsersHint : Nat
  resCount : Hist              -- reservations-per-user bins (DP, clip 1)
  countryMix : Hist            -- hotel-country tier mix (DP)
  transitions : List CondHist   -- tier -> tier rows (DP + return-boost prior)
  nights : Hist                -- stay-nights bins (DP)
  gap : Hist                    -- inter-leg gap bins (DP)
  monthWindow : Hist           -- first-trip start month-of-window (DP, clip 1;
                                -- v1.2.0: replaced month-of-year — the real
                                -- calendar is an observation ramp)
  device : Hist                -- device classes (DP)
  affiliate : Hist             -- affiliate tiers (DP)
  booker : Hist                -- booker countries (DP, clip 1)
  returnRate : Hist            -- repeat-city trip-share bands (DP, clip 1)
  repeatStruct : Hist         -- repeat-trip anatomy (DP, clip 1; v1.2.0:
                                -- r_ext1/r_ext2/r_ext3p consecutive same-city
                                -- extensions, r_ret1/r_ret2p non-adjacent
                                -- revisits, r_both)
  tripsPerUser : Hist         -- trips-per-user bins tp_1/tp_2/tp_3p (DP,
                                -- clip 1; v1.3.0 — the generator renders
                                -- multi-trip users from this)
  devicePersist : Hist        -- device-classes-per-trip dv_1/dv_2/dv_3 (DP,
                                -- clip 1; v1.3.0 — per-trip device plans;
                                -- 70.7% of real trips are single-device)
  affPersist : Hist           -- affiliate-tiers-per-trip ap_1..ap_4p (DP,
                                -- clip 1; v1.3.0 — per-trip affiliate
                                -- plans; 75.4% of legs keep the prior tier)
  cities : List City            -- released top city popularity
  corridors : List CityCorridor -- released top city transitions
  gapMoments : List GapMoment   -- suppressed gap moments (informational)
  windowStartD : Nat            -- public export window [start, end)
  windowEndD : Nat
  rcBinRanges : List (String × RangeN)
  gapRangesD : List (String × RangeN)
  nightsRanges : List (String × RangeN)

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

def parseCities (j : Json) : Except String (List City) := do
  let arr ← (← j.getObjVal? "cities").getArr?
  arr.toList.mapM fun c => do
    pure { cityId := (← (← c.getObjVal? "city_id").getNat?),
           country := (← (← c.getObjVal? "country").getStr?),
           ct := (← (← c.getObjVal? "ct").getStr?),
           w := (← (← c.getObjVal? "w").getNat?) }

def parseCorridors (j : Json) : Except String (List CityCorridor) := do
  let arr ← (← j.getObjVal? "corridors").getArr?
  arr.toList.mapM fun c => do
    pure { aCity := (← (← c.getObjVal? "a_city").getNat?),
           bCity := (← (← c.getObjVal? "b_city").getNat?),
           ct := (← (← c.getObjVal? "ct").getStr?),
           nUsers := (← (← c.getObjVal? "n_users").getNat?),
           nTrans := (← (← c.getObjVal? "n_trans").getNat?) }

def parseGapMoments (j : Json) : Except String (List GapMoment) := do
  match j.getObjVal? "gap_moments" with
  | .error _ => .ok []        -- informational; optional in legacy inputs
  | .ok a => do
    let arr ← a.getArr?
    arr.toList.mapM fun c => do
      pure { gapClass := (← (← c.getObjVal? "gap_class").getStr?),
             meanDays := JsonNumber.toFloat
               (← (← c.getObjVal? "mean_days").getNum?),
             sdDays := JsonNumber.toFloat
               (← (← c.getObjVal? "sd_days").getNum?) }

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
         resCount := ← parseHist j "res_count",
         countryMix := ← parseHist j "country_mix",
         transitions := ← parseCond j "transitions",
         nights := ← parseHist j "nights",
         gap := ← parseHist j "gap",
         monthWindow := ← parseHist j "month_window",
         device := ← parseHist j "device",
         affiliate := ← parseHist j "affiliate",
         booker := ← parseHist j "booker",
         returnRate := ← parseHist j "return_rate",
         repeatStruct := ← parseHist j "repeat_struct",
         tripsPerUser := ← parseHist j "trips_per_user",
         devicePersist := ← parseHist j "device_persist",
         affPersist := ← parseHist j "affiliate_persist",
         cities := ← parseCities j,
         corridors := ← parseCorridors j,
         gapMoments := ← parseGapMoments j,
         windowStartD := (← (← w.getObjVal? "start_day").getNat?),
         windowEndD := (← (← w.getObjVal? "end_day").getNat?),
         rcBinRanges := ← parseRangeList j "rc_bin_ranges",
         gapRangesD := ← parseRangeList j "gap_ranges_d",
         nightsRanges := ← parseRangeList j "nights_ranges" }

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

/-- Bounded tier walk: structurally recursive on fuel, so bounded
generation requests terminate by construction. -/
def genChain (rows : List CondHist) (fallback : Hist) :
    Nat → String → IO (List String)
  | 0, _ => pure []
  | n + 1, prev => do
    let c ← (rowOf rows fallback prev).choose
    let rest ← genChain rows fallback n c.name
    pure (c.name :: rest)

/-! ### Trip-row rendering, valid by construction

A `Leg` is one planned reservation: destination city/country/tier, the gap
(days) from the PREVIOUS checkout to this checkin (0 on contiguous legs),
the stay nights, and the per-leg device / affiliate labels. `mkLegs` renders
a leg list into rows, deriving checkin i+1 from checkout i (checkin_i +
nights_i) plus the next leg's gap — the renderer never takes a row's checkin
as free input. -/

structure Leg where
  city : Nat
  country : String
  ct : String
  device : String
  aff : String
  gap : Nat
  nights : Nat

/-- One rendered reservation row (the Lean image of a combined_set.csv row;
user / date / utrip fields attach at I/O time). -/
structure LegRow where
  city : Nat
  country : String
  ct : String
  device : String
  aff : String
  checkin : Nat
  nights : Nat

/-- Smart leg constructor: the gap is clamped into the public range
[0, 33] and the nights into [1, 30], so the range hypotheses of the
rendering theorems hold by construction. -/
def mkLeg (city : Nat) (country ct device aff : String) (gap nights : Nat) : Leg :=
  { city, country, ct, device, aff
    gap := min gap Params.gapMax
    nights := max 1 (min nights Params.nightsMax) }

/-- The public range law of the smart constructor: gaps land in [0, 33]. -/
theorem mkLeg_gap_range (city : Nat) (country ct device aff : String)
    (gap nights : Nat) :
    (mkLeg city country ct device aff gap nights).gap ≤ Params.gapMax := by
  unfold mkLeg; exact Nat.min_le_right _ _

/-- The public range law of the smart constructor: nights land in [1, 30]. -/
theorem mkLeg_nights_range (city : Nat) (country ct device aff : String)
    (gap nights : Nat) :
    1 ≤ (mkLeg city country ct device aff gap nights).nights ∧
      (mkLeg city country ct device aff gap nights).nights ≤ Params.nightsMax := by
  unfold mkLeg
  constructor
  · exact Nat.le_max_left _ _
  · show (max 1 (min nights Params.nightsMax)) ≤ Params.nightsMax
    have hM : Params.nightsMax = 30 := rfl
    omega

/-- Render a trip. `start` is the first leg's anchor: row i+1's checkin is
DERIVED from row i's checkout (checkin + nights) plus the next leg's gap —
never passed in — so the contiguous flow and strict ordering are structural. -/
def mkLegs (start : Nat) : List Leg → List LegRow
  | [] => []
  | l :: rest =>
      ⟨l.city, l.country, l.ct, l.device, l.aff, start + l.gap, l.nights⟩ ::
        mkLegs (start + l.gap + l.nights) rest

@[simp] theorem mkLegs_nil (start : Nat) : mkLegs start [] = [] := rfl

@[simp] theorem mkLegs_cons (start : Nat) (l : Leg) (rest : List Leg) :
    mkLegs start (l :: rest) =
      ⟨l.city, l.country, l.ct, l.device, l.aff, start + l.gap, l.nights⟩ ::
        mkLegs (start + l.gap + l.nights) rest := rfl

/-- Every rendered checkin is at least the start day. -/
theorem mkLegs_checkin_ge : ∀ (legs : List Leg) (start : Nat),
    ∀ r ∈ mkLegs start legs, start ≤ r.checkin
  | [], _, r, hr => by simp at hr
  | l :: rest, start, r, hr => by
    simp only [mkLegs_cons, List.mem_cons] at hr
    rcases hr with rfl | hr
    · show start ≤ start + l.gap
      omega
    · have := mkLegs_checkin_ge rest (start + l.gap + l.nights) r hr
      omega

/-- Every rendered checkout (checkin + nights) is bounded by the start day
plus the plan's total span (sum of all gap + nights). -/
theorem mkLegs_checkout_le : ∀ (legs : List Leg) (start : Nat),
    ∀ r ∈ mkLegs start legs,
      r.checkin + r.nights ≤ start + (legs.map (fun l => l.gap + l.nights)).sum
  | [], _, r, hr => by simp at hr
  | l :: rest, start, r, hr => by
    simp only [mkLegs_cons, List.mem_cons, List.map_cons, List.sum_cons] at hr ⊢
    rcases hr with rfl | hr
    · show (start + l.gap) + l.nights ≤
          start + ((l.gap + l.nights) + (rest.map (fun l => l.gap + l.nights)).sum)
      omega
    · have ih := mkLegs_checkout_le rest (start + l.gap + l.nights) r hr
      omega

/-- With a positive base offset and stays of at least one night, every row
rendered from `start + base` is strictly after `start`. -/
theorem mkLegs_checkin_lt : ∀ (rest : List Leg) (start base : Nat),
    1 ≤ base → (∀ l ∈ rest, 1 ≤ l.nights) →
    ∀ r ∈ mkLegs (start + base) rest, start < r.checkin
  | [], _, _, _, _, r, hr => by simp at hr
  | l :: rest', start, base, hb, hp, r, hr => by
    have h1 : 1 ≤ l.nights := hp l (by simp)
    simp only [mkLegs_cons, List.mem_cons] at hr
    rcases hr with rfl | hr
    · show start < start + base + l.gap
      omega
    · have hrec := mkLegs_checkin_lt rest' (start + base + l.gap) l.nights
        (by omega) (fun x hx => hp x (by simp [hx])) r hr
      omega

/-- **Per-trip checkin ordering**: stays of at least one night render a
strictly increasing checkin column (`List.Pairwise (· < ·)`). -/
theorem mkLegs_checkins_sorted (legs : List Leg) (start : Nat)
    (hp : ∀ l ∈ legs, 1 ≤ l.nights) :
    ((mkLegs start legs).map (·.checkin)).Pairwise (· < ·) := by
  induction legs generalizing start with
  | nil => simp
  | cons l rest ih =>
    have h1 : 1 ≤ l.nights := hp l (by simp)
    simp only [mkLegs_cons, List.map_cons]
    refine List.Pairwise.cons ?_
      (ih (start + l.gap + l.nights) (fun x hx => hp x (by simp [hx])))
    intro y hy
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp hy
    exact mkLegs_checkin_lt rest (start + l.gap) l.nights h1
      (fun x hx => hp x (by simp [hx])) r hr

/-- Zero-gap plans: every non-first leg has gap 0 (the exact contiguous
flow; the first leg's gap just anchors the start day). -/
def ContigGaps : List Leg → Prop
  | [] => True
  | [_] => True
  | l :: l' :: rest => l'.gap = 0 ∧ ContigGaps (l' :: rest)

/-- The rendered contiguous-flow property: every adjacent pair satisfies
checkin' = checkin + nights (next checkin = previous checkout). -/
def Contiguous : List LegRow → Prop
  | [] => True
  | [_] => True
  | r :: r' :: rest => r'.checkin = r.checkin + r.nights ∧ Contiguous (r' :: rest)

/-- **Contiguity by construction**: a zero-gap leg plan renders the exact
contiguous flow of the real dataset (checkout = next checkin on every
adjacent pair). -/
theorem mkLegs_contiguous : ∀ (legs : List Leg) (start : Nat),
    ContigGaps legs → Contiguous (mkLegs start legs)
  | [], _, _ => trivial
  | [_], _, _ => trivial
  | l :: l' :: rest, start, hd => by
    have h0 : l'.gap = 0 := hd.1
    have ih := mkLegs_contiguous (l' :: rest) (start + l.gap + l.nights) hd.2
    refine ⟨?_, ih⟩
    show (start + l.gap + l.nights + l'.gap) = (start + l.gap) + l.nights
    omega

/-- No overlapping presence: each stay closes at or before the next
checkin, and the last stay closes at or before the window end. -/
def NoOverlap (wEnd : Nat) : List LegRow → Prop
  | [] => True
  | [r] => r.checkin + r.nights ≤ wEnd
  | r :: r' :: rest => r.checkin + r.nights ≤ r'.checkin ∧ NoOverlap wEnd (r' :: rest)

/-- **No overlapping presence by construction**: a plan whose total span
fits the window renders rows where every stay interval closes before the
next checkin — a user is never booked in two places at once — and the final
checkout stays at or before the window end. -/
theorem mkLegs_no_overlap (wEnd : Nat) : ∀ (start : Nat) (legs : List Leg),
    start + (legs.map (fun l => l.gap + l.nights)).sum ≤ wEnd →
    NoOverlap wEnd (mkLegs start legs)
  | _, [], _ => trivial
  | start, [l], hw => by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil] at hw
    show (start + l.gap) + l.nights ≤ wEnd
    omega
  | start, l :: l' :: rest, hw => by
    have hw' : (start + l.gap + l.nights) +
        ((l' :: rest).map (fun l => l.gap + l.nights)).sum ≤ wEnd := by
      simp only [List.map_cons, List.sum_cons] at hw ⊢
      omega
    have ih := mkLegs_no_overlap wEnd (start + l.gap + l.nights) (l' :: rest) hw'
    refine ⟨?_, ih⟩
    show (start + l.gap) + l.nights ≤ (start + l.gap + l.nights) + l'.gap
    omega

/-- **Window containment**: with the start day in range and the plan's
total span fitting the window, every rendered checkin lands in
[wStart, wEnd] with its checkout at or before wEnd. -/
theorem mkLegs_in_window (wStart wEnd start : Nat) (legs : List Leg)
    (hstart : wStart ≤ start)
    (hspan : start + (legs.map (fun l => l.gap + l.nights)).sum ≤ wEnd) :
    ∀ r ∈ mkLegs start legs, wStart ≤ r.checkin ∧ r.checkin + r.nights ≤ wEnd := by
  intro r hr
  have hge := mkLegs_checkin_ge legs start r hr
  have hle := mkLegs_checkout_le legs start r hr
  omega

/-! ### Window clamp (prefix cut)

The generator's window clamp: keep legs while the running checkout stays at
or before `wEnd` — drops the tail, keeps a prefix. Only the span and subset
lemmas are needed; the rendering theorems above then apply to the kept list. -/

/-- Keep legs while the running checkout (start + gap + nights) stays at or
before `tEnd`. -/
def legsWithin (tEnd : Nat) : Nat → List Leg → List Leg
  | _, [] => []
  | t, l :: rest =>
      if t + l.gap + l.nights ≤ tEnd
      then l :: legsWithin tEnd (t + l.gap + l.nights) rest
      else []

/-- The clamp works: starting at or before `tEnd`, the kept plan's total
span still ends at or before `tEnd` (so `mkLegs_in_window` and
`mkLegs_no_overlap` apply to the kept list). -/
theorem legsWithin_span_le : ∀ (tEnd t : Nat) (legs : List Leg),
    t ≤ tEnd →
    t + ((legsWithin tEnd t legs).map (fun l => l.gap + l.nights)).sum ≤ tEnd
  | _, _, [], h => by simp only [legsWithin, List.map_nil, List.sum_nil]; omega
  | tEnd, t, l :: rest, h => by
    simp only [legsWithin]
    split
    · next hlt =>
      simp only [List.map_cons, List.sum_cons]
      have ih := legsWithin_span_le tEnd (t + l.gap + l.nights) rest hlt
      omega
    · simpa using h

/-- The clamp keeps a prefix: every kept leg is an original leg (so
per-leg range properties transport to the kept list). -/
theorem legsWithin_subset : ∀ (tEnd t : Nat) (legs : List Leg),
    ∀ l ∈ legsWithin tEnd t legs, l ∈ legs
  | _, _, [], _, hl => by simp [legsWithin] at hl
  | tEnd, t, l₀ :: rest, l, hl => by
    simp only [legsWithin] at hl
    split at hl
    · simp only [List.mem_cons] at hl ⊢
      rcases hl with rfl | hl
      · exact Or.inl rfl
      · exact Or.inr (legsWithin_subset tEnd (t + l₀.gap + l₀.nights) rest l hl)
    · simp at hl

end EnclaveTrip.Gen

/-
Formal definitions of the booking trip motifs computed by the dbt marts, over
Nat (epoch days, nights, gap lists) and String (countries, city labels), so
every predicate is decidable and executable. `lake exe checks` evaluates
these very definitions on observations exported from the enclave store. All
thresholds come from the mirrored concept spec (Params.lean); the `...At`
variants take the threshold explicitly so monotonicity can be stated and the
executable can honor exported parameters.

Motif set (booking multi-destination trips):
  * HasRepeatCity — a trip's city list contains a city twice (38.1% of real
    trips revisit a city within the same trip; the core return motif);
  * IsCorridor    — one user repeats the same ordered country pair with
    regular inter-repeat gaps (the commuter-corridor analog across a
    user's reservations);
  * IsTour        — a trip of at least tour_min_legs legs, every consecutive
    checkin gap bounded, visiting at least tour_min_countries distinct
    hotel countries;
  * IsLongStay / AnyLongStay — a stay of at least longstay_min_nights nights.
-/
import Mathlib.Data.List.Chain
import Mathlib.Data.List.Dedup
import Mathlib.Tactic
import EnclaveTrip.Params

namespace EnclaveTrip

/-! ### Return motif (repeat city within one trip) -/

/-- `HasRepeatCity cities`: the trip's city list contains some city at least
twice — equivalently the deduplicated list is strictly shorter. -/
def HasRepeatCity (cities : List String) : Prop :=
  cities.dedup.length < cities.length

instance (cities : List String) : Decidable (HasRepeatCity cities) := by
  unfold HasRepeatCity; exact inferInstance

/-- The canonical A → B → A trip revisits a city (sanity: the definition
accepts the motif it was designed around). -/
theorem hasRepeatCity_canonical : HasRepeatCity ["A", "B", "A"] := by decide

/-- Re-visiting any city already on the trip creates a repeat: prepending a
city that is already in the list satisfies `HasRepeatCity`. -/
theorem hasRepeatCity_cons_self {l : List String} {x : String}
    (h : x ∈ l) : HasRepeatCity (x :: l) := by
  have hd : (x :: l).dedup = l.dedup := List.dedup_cons_of_mem h
  have hle : l.dedup.length ≤ l.length := l.dedup_sublist.length_le
  unfold HasRepeatCity
  rw [hd, List.length_cons]
  omega

/-- A list with no repeats does not satisfy the predicate (distinction, so
the decidable export is meaningful on both sides). -/
theorem not_hasRepeatCity_distinct : ¬ HasRepeatCity ["A", "B", "C"] := by
  decide

/-! ### Corridor motif (repeated ordered country pair with regular gaps)

Gap regularity is the SQL condition `cv ≤ num/den` (`cv = sd/mean` of the
inter-repeat gaps, population variance), restated EXACTLY over Nat.
With `k = gaps.length`, `s1 = Σ gaps`, `s2 = Σ gaps²` and `mean = s1/k`:

    var = s2/k − (s1/k)²  and  cv² = var/mean² = (k·s2 − s1²)/s1² = k·s2/s1² − 1.

So, for `s1 > 0`,

    cv ≤ num/den  ⟺  cv² ≤ num²/den²
                  ⟺  den²·(k·s2 − s1²) ≤ num²·s1²
                  ⟺  den²·k·s2 ≤ (den² + num²)·s1².

With the public bound 4/5 this is `25·k·s2 ≤ 41·s1²` — pure Nat arithmetic,
no subtraction, no division, hence decidable and exactly the SQL float test
(up to the float rounding `lake exe checks` is there to catch). -/

/-- Gap regularity of a corridor: positive total gap and
`den²·k·s2 ≤ (den² + num²)·s1²` (⟺ CV ≤ num/den, see the algebra above). -/
def GapsRegular (gaps : List Nat) : Prop :=
  0 < gaps.sum ∧
    Params.corridorCvMaxDen * Params.corridorCvMaxDen * gaps.length *
        (gaps.map (fun g => g * g)).sum ≤
      (Params.corridorCvMaxDen * Params.corridorCvMaxDen +
        Params.corridorCvMaxNum * Params.corridorCvMaxNum) *
        (gaps.sum * gaps.sum)

instance (gaps : List Nat) : Decidable (GapsRegular gaps) := by
  unfold GapsRegular; exact inferInstance

/-- Corridor at an explicit repeat threshold: `n` repeats of the same
country pair whose inter-repeat gap list is regular. -/
def IsCorridorAt (minRep : Nat) (n : Nat) (gaps : List Nat) : Prop :=
  minRep ≤ n ∧ GapsRegular gaps

instance (minRep n : Nat) (gaps : List Nat) :
    Decidable (IsCorridorAt minRep n gaps) := by
  unfold IsCorridorAt; exact inferInstance

/-- A corridor at the public spec threshold `corridor_min_repeats`. -/
def IsCorridor (n : Nat) (gaps : List Nat) : Prop :=
  IsCorridorAt Params.corridorMinRepeats n gaps

instance (n : Nat) (gaps : List Nat) : Decidable (IsCorridor n gaps) := by
  unfold IsCorridor; exact inferInstance

/-- Constant positive gaps are regular: `s1 = m·g`, `s2 = m·g²`, `k = m`, so
`den²·k·s2 = den²·s1² ≤ (den² + num²)·s1²` — CV = 0. -/
theorem gapsRegular_replicate {m g : Nat} (hm : 0 < m) (hg : 0 < g) :
    GapsRegular (List.replicate m g) := by
  unfold GapsRegular
  rw [List.length_replicate, List.map_replicate, List.sum_replicate,
    List.sum_replicate, smul_eq_mul, smul_eq_mul]
  refine ⟨Nat.mul_pos hm hg, ?_⟩
  calc Params.corridorCvMaxDen * Params.corridorCvMaxDen * m * (m * (g * g))
      = (Params.corridorCvMaxDen * Params.corridorCvMaxDen) *
          (m * g * (m * g)) := by ring
    _ ≤ (Params.corridorCvMaxDen * Params.corridorCvMaxDen +
          Params.corridorCvMaxNum * Params.corridorCvMaxNum) *
          (m * g * (m * g)) :=
        Nat.mul_le_mul_right _ (Nat.le_add_right _ _)

/-- A corridor with constant positive gaps and enough repeats is a
corridor (sanity: the definition accepts the motif it was designed around). -/
theorem isCorridor_const {n m g : Nat} (hn : Params.corridorMinRepeats ≤ n)
    (hm : 0 < m) (hg : 0 < g) : IsCorridor n (List.replicate m g) :=
  ⟨hn, gapsRegular_replicate hm hg⟩

/-- Lowering the repeat threshold preserves corridor-hood (monotonicity). -/
theorem isCorridorAt_mono {r r' n : Nat} {gaps : List Nat} (hr : r' ≤ r)
    (h : IsCorridorAt r n gaps) : IsCorridorAt r' n gaps :=
  ⟨le_trans hr h.1, h.2⟩

/-! ### Tour motif (multi-country trip with bounded inter-leg gaps) -/

/-- Consecutive step of a tour over GAP-NORMALIZED days: non-decreasing
(the SQL gap days are `next checkin − prev checkout`, so 0 = contiguous
legs — the dominant real case) with the step bounded by `maxGap`.
(`lake exe checks` normalizes exported checkin days by the per-leg nights
before judging; overlap truncates to 0 and is caught separately by the
trip well-formedness check.) -/
def TourStep (maxGap : Nat) (a b : Nat) : Prop := a ≤ b ∧ b - a ≤ maxGap

instance (maxGap : Nat) : DecidableRel (TourStep maxGap) := fun _ _ => by
  unfold TourStep; exact inferInstance

/-- Tour at an explicit inter-leg-gap bound: at least `tour_min_legs` legs,
every consecutive gap bounded, visiting at least `tour_min_countries`
distinct hotel countries. `days` is the gap-normalized day list. -/
def IsTourAt (maxGap : Nat) (days : List Nat) (countries : List String) : Prop :=
  Params.tourMinLegs ≤ days.length ∧
    List.IsChain (TourStep maxGap) days ∧
    Params.tourMinCountries ≤ countries.dedup.length

instance (maxGap : Nat) (days : List Nat) (countries : List String) :
    Decidable (IsTourAt maxGap days countries) := by
  unfold IsTourAt; exact inferInstance

/-- A tour at the public spec bound `tour_max_gap_d`. -/
def IsTour (days : List Nat) (countries : List String) : Prop :=
  IsTourAt Params.tourMaxGapD days countries

instance (days : List Nat) (countries : List String) :
    Decidable (IsTour days countries) := by
  unfold IsTour; exact inferInstance

/-- The canonical 4-leg, 3-country, contiguous trip is a tour (sanity). -/
theorem isTour_canonical :
    IsTour [0, 2, 4, 6] ["Gondal", "Elbonia", "Kasnia"] := by decide

/-- Widening the checkin-gap bound preserves tour-hood (monotonicity). -/
theorem isTour_mono {g g' : Nat} (hg : g ≤ g') {days : List Nat}
    {countries : List String} (h : IsTourAt g days countries) :
    IsTourAt g' days countries :=
  ⟨h.1, h.2.1.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩), h.2.2⟩

/-! ### Long-stay motif -/

/-- Long stay at an explicit threshold: the stay reaches `minNights` nights. -/
def IsLongStayAt (minNights d : Nat) : Prop := minNights ≤ d

instance (minNights d : Nat) : Decidable (IsLongStayAt minNights d) := by
  unfold IsLongStayAt; exact inferInstance

/-- A long stay at the public spec threshold `longstay_min_nights`. -/
def IsLongStay (d : Nat) : Prop := IsLongStayAt Params.longstayMinNights d

instance (d : Nat) : Decidable (IsLongStay d) := by
  unfold IsLongStay; exact inferInstance

/-- `AnyLongStay nights`: some stay in the trip reaches the public
long-stay threshold (the list-level form used by the trips mart). -/
def anyLongStayB (nights : List Nat) : Bool :=
  nights.any (fun n => decide (Params.longstayMinNights ≤ n))

def AnyLongStay (nights : List Nat) : Prop := anyLongStayB nights = true

instance (nights : List Nat) : Decidable (AnyLongStay nights) := by
  unfold AnyLongStay; infer_instance

/-- The canonical 8-night stay is a long stay (sanity). -/
theorem isLongStay_canonical : IsLongStay 8 := by decide

/-- The canonical trip containing an 8-night stay satisfies AnyLongStay. -/
theorem anyLongStay_canonical : AnyLongStay [1, 8, 1] := by decide

/-- `AnyLongStay` holds exactly when some listed stay is a long stay. -/
theorem anyLongStay_iff (nights : List Nat) :
    AnyLongStay nights ↔ ∃ n ∈ nights, IsLongStayAt Params.longstayMinNights n := by
  unfold AnyLongStay anyLongStayB IsLongStayAt
  simp

/-- Lowering the duration threshold preserves long-stay-hood (monotonicity). -/
theorem isLongStay_mono {m m' d : Nat} (hm : m' ≤ m)
    (h : IsLongStayAt m d) : IsLongStayAt m' d :=
  le_trans hm h

/-! ### Booking-channel constancy motifs (spec 1.3.0)

Real trips are booked on ONE device class 70.7% of the time and stick to
few affiliate tiers (47.5% exactly one; 75.4% of legs keep the previous
leg's tier). The generator draws a per-trip DEVICE PLAN and AFFILIATE PLAN
from the dv_*/ap_* releases; these predicates are what the plans must
satisfy on their constant cells, re-decided by `lake exe checks` against
the SQL flags (is_devconst / is_affconst in feat_trips). -/

/-- `IsConstantRun classes`: every entry of the per-leg class list equals
every other (a one-class trip; vacuously true for 0/1-leg lists). -/
def IsConstantRun (classes : List String) : Prop :=
  classes.all (fun c => classes.all (fun d => c == d))

instance (classes : List String) : Decidable (IsConstantRun classes) := by
  unfold IsConstantRun; exact inferInstance

/-- The canonical single-device trip is constant (sanity). -/
theorem isConstantRun_canonical : IsConstantRun ["mobile", "mobile"] := by
  decide

/-- A mixed trip is not constant (distinction). -/
theorem not_isConstantRun_mixed : ¬ IsConstantRun ["mobile", "desktop"] := by
  decide

/-! ### Single-country trip motif (spec 1.4.0)

72.2% of real trips stay in ONE hotel country (the mean is 1.41 countries
per trip); the unconstrained tier walk rendered 2.26 — synthetic
globe-trotters. The generator now draws a per-trip COUNTRY BUDGET from the
countries_per_trip release and locks the walk to the visited set; this
predicate is what the budget must produce on cpt_1 trips, re-decided by
`lake exe checks` against the SQL flag (n_countries = 1 in feat_trips). -/

/-- `IsSingleCountryTrip countries`: the trip's hotel-country list has at
most one distinct entry. -/
def IsSingleCountryTrip (countries : List String) : Prop :=
  countries.dedup.length ≤ 1

instance (countries : List String) :
    Decidable (IsSingleCountryTrip countries) := by
  unfold IsSingleCountryTrip; exact inferInstance

/-- The canonical single-country trip (sanity). -/
theorem isSingleCountryTrip_canonical :
    IsSingleCountryTrip ["Gondal", "Gondal", "Gondal"] := by decide

/-- A two-country trip is not single-country (distinction). -/
theorem not_isSingleCountryTrip_two :
    ¬ IsSingleCountryTrip ["Gondal", "Kangan"] := by decide

end EnclaveTrip

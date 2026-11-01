/-
Formal definitions of the travel motifs computed by the dbt marts, over Nat
(epoch seconds, durations, gap lists) and String (ISO countries), so every
predicate is decidable and executable. `lake exe checks` evaluates these very
definitions on observations exported from the enclave store. All thresholds
come from the mirrored concept spec (Params.lean); the `...At` variants take
the threshold explicitly so monotonicity can be stated and the executable can
honor exported parameters.
-/
import Mathlib.Data.List.Chain
import Mathlib.Data.List.Dedup
import Mathlib.Tactic
import EnclaveTravel.Params

namespace EnclaveTravel

/-! ### Return-pair motif -/

/-- `IsReturnPair a b c d`: two consecutive transfers `a→b`, `c→d` of one
user form an immediate return trip — the second departs where the first
arrived (`c = b`) and arrives back where it started (`d = a`), with a real
move (`a ≠ b`). -/
def IsReturnPair (a b c d : String) : Prop := a ≠ b ∧ c = b ∧ d = a

instance (a b c d : String) : Decidable (IsReturnPair a b c d) := by
  unfold IsReturnPair; exact inferInstance

/-- The canonical consecutive pair `a→b`, `b→a` is a return pair (sanity:
the definition accepts the motif it was designed around). -/
theorem isReturnPair_canonical {a b : String} (h : a ≠ b) :
    IsReturnPair a b b a := ⟨h, rfl, rfl⟩

/-! ### Corridor motif (repeated commute with regular gaps)

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

/-- A commuter corridor with constant positive gaps and enough repeats is a
corridor (sanity: the definition accepts the motif it was designed around). -/
theorem isCorridor_const {n m g : Nat} (hn : Params.corridorMinRepeats ≤ n)
    (hm : 0 < m) (hg : 0 < g) : IsCorridor n (List.replicate m g) :=
  ⟨hn, gapsRegular_replicate hm hg⟩

/-- Lowering the repeat threshold preserves corridor-hood (monotonicity). -/
theorem isCorridorAt_mono {r r' n : Nat} {gaps : List Nat} (hr : r' ≤ r)
    (h : IsCorridorAt r n gaps) : IsCorridorAt r' n gaps :=
  ⟨le_trans hr h.1, h.2⟩

/-! ### Tour motif (multi-country trip with bounded hop gaps) -/

/-- Consecutive step of a tour: strictly increasing timestamps with the hop
gap bounded by `maxGap`. -/
def TourStep (maxGap : Nat) (a b : Nat) : Prop := a < b ∧ b - a ≤ maxGap

instance (maxGap : Nat) : DecidableRel (TourStep maxGap) := fun _ _ => by
  unfold TourStep; exact inferInstance

/-- Tour at an explicit hop-gap bound: at least `tour_min_hops` transfers,
every consecutive gap bounded, visiting at least `tour_min_countries`
distinct countries. -/
def IsTourAt (maxGap : Nat) (ts : List Nat) (countries : List String) : Prop :=
  Params.tourMinHops ≤ ts.length ∧
    List.IsChain (TourStep maxGap) ts ∧
    Params.tourMinCountries ≤ countries.dedup.length

instance (maxGap : Nat) (ts : List Nat) (countries : List String) :
    Decidable (IsTourAt maxGap ts countries) := by
  unfold IsTourAt; exact inferInstance

/-- A tour at the public spec bound `tour_max_gap_s`. -/
def IsTour (ts : List Nat) (countries : List String) : Prop :=
  IsTourAt Params.tourMaxGapS ts countries

instance (ts : List Nat) (countries : List String) :
    Decidable (IsTour ts countries) := by
  unfold IsTour; exact inferInstance

/-- The canonical 3-hop, 3-country, day-apart trip is a tour (sanity). -/
theorem isTour_canonical :
    IsTour [0, 86400, 172800] ["US", "FR", "JP"] := by decide

/-- Widening the hop-gap bound preserves tour-hood (monotonicity). -/
theorem isTour_mono {g g' : Nat} (hg : g ≤ g') {ts : List Nat}
    {countries : List String} (h : IsTourAt g ts countries) :
    IsTourAt g' ts countries :=
  ⟨h.1, h.2.1.imp (fun _ _ hab => ⟨hab.1, le_trans hab.2 hg⟩), h.2.2⟩

/-! ### Burst motif (many transfers in a short window) -/

/-- Burst at an explicit window: at least `burst_min_transfers` sorted
timestamps spanning at most `window` seconds (last − first). -/
def IsBurstAt (window : Nat) (ts : List Nat) : Prop :=
  Params.burstMinTransfers ≤ ts.length ∧
    List.IsChain (· ≤ ·) ts ∧
    ts.getLastD 0 - ts.headD 0 ≤ window

instance (window : Nat) (ts : List Nat) : Decidable (IsBurstAt window ts) := by
  unfold IsBurstAt; exact inferInstance

/-- A burst at the public spec window `burst_window_s`. -/
def IsBurst (ts : List Nat) : Prop := IsBurstAt Params.burstWindowS ts

instance (ts : List Nat) : Decidable (IsBurst ts) := by
  unfold IsBurst; exact inferInstance

/-- The canonical tight triple is a burst (sanity). -/
theorem isBurst_canonical : IsBurst [0, 1, 2] := by decide

/-- Widening the window preserves burst-hood (monotonicity). -/
theorem isBurst_mono {w w' : Nat} (hw : w ≤ w') {ts : List Nat}
    (h : IsBurstAt w ts) : IsBurstAt w' ts :=
  ⟨h.1, h.2.1, le_trans h.2.2 hw⟩

/-! ### Long-stay motif -/

/-- Long stay at an explicit threshold: the stay duration reaches `minS`. -/
def IsLongStayAt (minS d : Nat) : Prop := minS ≤ d

instance (minS d : Nat) : Decidable (IsLongStayAt minS d) := by
  unfold IsLongStayAt; exact inferInstance

/-- A long stay at the public spec threshold `longstay_min_s`. -/
def IsLongStay (d : Nat) : Prop := IsLongStayAt Params.longstayMinS d

instance (d : Nat) : Decidable (IsLongStay d) := by
  unfold IsLongStay; exact inferInstance

/-- Lowering the duration threshold preserves long-stay-hood (monotonicity). -/
theorem isLongStay_mono {m m' d : Nat} (hm : m' ≤ m)
    (h : IsLongStayAt m d) : IsLongStayAt m' d :=
  le_trans hm h

end EnclaveTravel

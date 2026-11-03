/-
Generator shape parameters and their linking theorems.

`scripts/synth.py` regex-reads the `def`s below (Lean is the single source of
the generator's distributions — the simulation cannot drift from the file the
theorems are about). The values are ASSUMPTIONS about rendered traffic, not
released facts. The linking theorems prove that a generator honouring these
parameters emits traffic the encoder MUST accept: planned beacon channels
satisfy `IsBeacon`, planned sessions satisfy `IsTimelySession` — "synthetic
data passes the spec" is a theorem, not an empirical hope.
-/
import EnclaveFlows.Params
import EnclaveFlows.Motifs
import EnclaveFlows.Sequences

namespace EnclaveFlows.FlowShape

open EnclaveFlows

/-! ### Generator parameters (assumptions; regex-read by synth.py) -/

/-- Services the generator renders (a subset of the public taxonomy). -/
def genServices : List String := ["http", "https", "dns", "ntp", "chat", "ephemeral", "other"]

/-- Beacon channel period choices (ms) and the jitter bound (ms). -/
def beaconPeriodMsChoices : List Nat := [27000, 40000, 60000]
def beaconJitterMs : Nat := 1500
/-- Flows emitted per planned beacon channel. -/
def beaconFlowsMin : Nat := 12

/-- Session rendering: intra-session step bound and inter-session silence. -/
def sessionStepMsMax : Nat := 4000
def interSessionGapMsMin : Nat := 8000
def sessionFlowsChoices : List Nat := [2, 3, 4, 6, 9]
def sessionFlowsWeights : List Nat := [30, 25, 20, 15, 10]

/-- Per-flow size/duration distributions (weights for the verified sampler). -/
def bytesOutChoices : List Nat := [80, 260, 600, 1400, 4000]
def bytesOutWeights : List Nat := [20, 35, 25, 15, 5]
def bytesInChoices : List Nat := [0, 400, 2000, 12000, 40000]
def bytesInWeights : List Nat := [25, 25, 25, 15, 10]
def flowDurationMsChoices : List Nat := [20, 90, 400, 2000, 9000]
def flowDurationMsWeights : List Nat := [25, 30, 25, 15, 5]

/-- The encoder's beacon parameters, built from the mirrored spec. -/
def specBeaconParams : BeaconParams :=
  { minFlows := Params.beaconMinFlows
  , maxCvSq := ((Params.beaconMaxCvNum : ℚ) ^ 2) / ((Params.beaconMaxCvDen : ℚ) ^ 2)
  , minPeriodMs := (Params.beaconMinPeriodMs : ℚ) }

/-! ### Linking theorems: parameters ⇒ encoder acceptance -/

/-- Session steps stay strictly inside the session-gap threshold. -/
theorem sessionStep_lt_chainGap : sessionStepMsMax < Params.chainMaxGapMs := by decide

/-- Inter-session silence exceeds the session-gap threshold (a real break). -/
theorem interSessionGap_breaks : Params.chainMaxGapMs < interSessionGapMsMin := by decide

/-- The generator emits at least the spec's minimum flows per beacon. -/
theorem beaconFlows_ge_min : Params.beaconMinFlows ≤ beaconFlowsMin := by decide

/-- Every period choice stays above the spec minimum even at full jitter. -/
theorem beaconPeriod_ge_min :
    ∀ p ∈ beaconPeriodMsChoices,
      Params.beaconMinPeriodMs ≤ p - beaconJitterMs := by decide

/-- Mean of a list bounded in `[lo, hi]` lies in `[lo, hi]`. -/
theorem qMean_mem_of_bounded {gaps : List ℚ} {lo hi : ℚ} (hne : gaps ≠ [])
    (hb : ∀ g ∈ gaps, lo ≤ g ∧ g ≤ hi) :
    lo ≤ qMean gaps ∧ qMean gaps ≤ hi := by
  have hlen : 0 < (gaps.length : ℚ) := by
    have := List.length_pos_iff.mpr hne
    exact_mod_cast this
  have hup : gaps.sum ≤ gaps.length • hi :=
    List.sum_le_length_nsmul gaps hi (fun g hg => (hb g hg).2)
  have hlo : gaps.length • lo ≤ gaps.sum :=
    List.length_nsmul_le_sum gaps lo (fun g hg => (hb g hg).1)
  rw [nsmul_eq_mul] at hup hlo
  constructor
  · rw [qMean, le_div_iff₀ hlen]
    linarith
  · rw [qMean, div_le_iff₀ hlen]
    linarith

/--
**Generator contract for beacons** (generic form): gaps confined to
`[p − j, p + j]` with enough flows form a beacon, provided the public
side-condition `4·j²·n ≤ maxCvSq·(p−j)²·(n−1)` holds.
-/
theorem isBeacon_of_bounded_gaps (P : BeaconParams) (p j : ℚ) (gaps : List ℚ)
    (hn : P.minFlows ≤ gaps.length + 1) (hne : gaps ≠ [])
    (hj : 0 ≤ j) (hpj : 0 < p - j) (hper : P.minPeriodMs ≤ p - j)
    (hc : 0 ≤ P.maxCvSq)
    (hcv : 4 * j ^ 2 * (gaps.length : ℚ)
            ≤ P.maxCvSq * (p - j) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ))
    (hb : ∀ g ∈ gaps, p - j ≤ g ∧ g ≤ p + j) :
    IsBeacon P gaps := by
  obtain ⟨hm_lo, hm_hi⟩ := qMean_mem_of_bounded hne hb
  have hmean_pos : 0 < qMean gaps := lt_of_lt_of_le hpj hm_lo
  refine ⟨hn, hmean_pos, le_trans hper hm_lo, ?_⟩
  -- each squared deviation is at most (2j)²
  have hdev : ∀ x ∈ gaps.map (fun x => (x - qMean gaps) ^ 2), x ≤ (2 * j) ^ 2 := by
    intro x hx
    obtain ⟨g, hg, rfl⟩ := List.mem_map.mp hx
    have h1 := (hb g hg).1
    have h2 := (hb g hg).2
    exact sq_le_sq' (by linarith) (by linarith)
  have hssd : qSsd gaps ≤ (gaps.length : ℚ) * (2 * j) ^ 2 := by
    have := List.sum_le_length_nsmul _ _ hdev
    rwa [nsmul_eq_mul, List.length_map] at this
  have hsq : (p - j) ^ 2 ≤ (qMean gaps) ^ 2 :=
    pow_le_pow_left₀ (le_of_lt hpj) hm_lo 2
  calc qSsd gaps
      ≤ (gaps.length : ℚ) * (2 * j) ^ 2 := hssd
    _ = 4 * j ^ 2 * (gaps.length : ℚ) := by ring
    _ ≤ P.maxCvSq * (p - j) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ) := hcv
    _ ≤ P.maxCvSq * (qMean gaps) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ) := by
        have hcast : (0 : ℚ) ≤ ((gaps.length - 1 : ℕ) : ℚ) := Nat.cast_nonneg _
        have := mul_le_mul_of_nonneg_left hsq hc
        exact mul_le_mul_of_nonneg_right this hcast

/-- The concrete numeric side-condition for every period choice:
`8·j² ≤ maxCvSq·(p−j)²` (the factor 2 absorbs `n/(n−1) ≤ 2` for `n ≥ 2`). -/
theorem gen_beacon_side_ok :
    ∀ p ∈ beaconPeriodMsChoices,
      8 * ((beaconJitterMs : ℚ)) ^ 2
        ≤ specBeaconParams.maxCvSq * ((p : ℚ) - (beaconJitterMs : ℚ)) ^ 2 := by
  intro p hp
  fin_cases hp <;> norm_num [specBeaconParams, beaconJitterMs,
    Params.beaconMaxCvNum, Params.beaconMaxCvDen]

/--
**Generator contract, concrete**: any channel the generator renders with a
period from `beaconPeriodMsChoices`, jitter within `beaconJitterMs`, and at
least `beaconMinFlows` flows IS a beacon under the spec parameters.
-/
theorem genBeacon_isBeacon (p : ℕ) (hp : p ∈ beaconPeriodMsChoices)
    (gaps : List ℚ)
    (hn : Params.beaconMinFlows ≤ gaps.length + 1) (hne : gaps ≠ [])
    (hb : ∀ g ∈ gaps, (p : ℚ) - (beaconJitterMs : ℚ) ≤ g
                      ∧ g ≤ (p : ℚ) + (beaconJitterMs : ℚ)) :
    IsBeacon specBeaconParams gaps := by
  have hside := gen_beacon_side_ok p hp
  have hjq : (0 : ℚ) ≤ (beaconJitterMs : ℚ) := Nat.cast_nonneg _
  have hle : beaconJitterMs ≤ p := by fin_cases hp <;> decide
  have h1 : (Params.beaconMinPeriodMs : ℚ) ≤ (p : ℚ) - (beaconJitterMs : ℚ) := by
    have h0 : (Params.beaconMinPeriodMs : ℚ) ≤ ((p - beaconJitterMs : ℕ) : ℚ) := by
      exact_mod_cast beaconPeriod_ge_min p hp
    have hsub : ((p - beaconJitterMs : ℕ) : ℚ) = (p : ℚ) - (beaconJitterMs : ℚ) := by
      push_cast [hle]; ring
    linarith [h0, hsub.le, hsub.ge]
  have hpj : (0 : ℚ) < (p : ℚ) - (beaconJitterMs : ℚ) := by
    have hpos : (0 : ℚ) < (Params.beaconMinPeriodMs : ℚ) := by
      norm_num [Params.beaconMinPeriodMs]
    linarith
  have hper : specBeaconParams.minPeriodMs ≤ (p : ℚ) - (beaconJitterMs : ℚ) := h1
  have hc : (0 : ℚ) ≤ specBeaconParams.maxCvSq := by
    show (0 : ℚ) ≤ ((Params.beaconMaxCvNum : ℚ) ^ 2) / ((Params.beaconMaxCvDen : ℚ) ^ 2)
    positivity
  -- length ≥ 2 (minFlows = 8), so n ≤ 2(n−1)
  have hlen2 : 2 ≤ gaps.length := by
    have : Params.beaconMinFlows = 8 := rfl
    omega
  have hnat : gaps.length ≤ 2 * (gaps.length - 1) := by omega
  have hcast : (gaps.length : ℚ) ≤ 2 * ((gaps.length - 1 : ℕ) : ℚ) := by
    exact_mod_cast hnat
  refine isBeacon_of_bounded_gaps specBeaconParams _ _ gaps hn hne hjq hpj hper hc ?_ hb
  have hj2 : (0 : ℚ) ≤ 4 * ((beaconJitterMs : ℚ)) ^ 2 := by positivity
  calc 4 * ((beaconJitterMs : ℚ)) ^ 2 * (gaps.length : ℚ)
      ≤ 4 * ((beaconJitterMs : ℚ)) ^ 2 * (2 * ((gaps.length - 1 : ℕ) : ℚ)) :=
        mul_le_mul_of_nonneg_left hcast hj2
    _ = 8 * ((beaconJitterMs : ℚ)) ^ 2 * ((gaps.length - 1 : ℕ) : ℚ) := by ring
    _ ≤ specBeaconParams.maxCvSq * ((p : ℚ) - (beaconJitterMs : ℚ)) ^ 2
          * ((gaps.length - 1 : ℕ) : ℚ) :=
        mul_le_mul_of_nonneg_right hside (Nat.cast_nonneg _)

/-! ### Sessions: cumulative bounded steps are timely -/

/-- Start times from an initial time and a list of step gaps. -/
def cumTimes (t0 : ℚ) : List ℚ → List ℚ
  | [] => [t0]
  | s :: rest => t0 :: cumTimes (t0 + s) rest

/-- **Generator contract for sessions**: nonnegative steps bounded by `g`
produce a timely session at gap bound `g`. -/
theorem isTimely_cumTimes (g t0 : ℚ) (steps : List ℚ)
    (h : ∀ s ∈ steps, 0 ≤ s ∧ s ≤ g) :
    IsTimelySession g (cumTimes t0 steps) := by
  induction steps generalizing t0 with
  | nil => exact .singleton _
  | cons s rest ih =>
    have hs := h s List.mem_cons_self
    have hrest : ∀ x ∈ rest, 0 ≤ x ∧ x ≤ g :=
      fun x hx => h x (List.mem_cons_of_mem _ hx)
    have hstep : TimelyStep g t0 (t0 + s) :=
      show t0 ≤ t0 + s ∧ t0 + s - t0 ≤ g from ⟨by linarith [hs.1], by linarith [hs.2]⟩
    cases rest with
    | nil =>
      exact .cons_cons hstep (.singleton _)
    | cons s2 r2 =>
      have htail := ih (t0 + s) hrest
      simp only [cumTimes] at htail ⊢
      exact .cons_cons hstep htail

/-- The generator's session steps satisfy the spec gap bound (ms scale). -/
theorem genSession_isTimely (t0 : ℚ) (steps : List ℚ)
    (h : ∀ s ∈ steps, 0 ≤ s ∧ s ≤ (sessionStepMsMax : ℚ)) :
    IsTimelySession (Params.chainMaxGapMs : ℚ) (cumTimes t0 steps) := by
  have hmono : (sessionStepMsMax : ℚ) ≤ (Params.chainMaxGapMs : ℚ) := by
    exact_mod_cast le_of_lt sessionStep_lt_chainGap
  exact isTimely_cumTimes _ _ _
    (fun s hs => ⟨(h s hs).1, le_trans (h s hs).2 hmono⟩)

end EnclaveFlows.FlowShape

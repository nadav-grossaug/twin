/-
Formal definitions of the stock price motifs computed by the dbt marts, over
ℤ (log returns in exact basis points) and ℚ (prices), so every predicate is
decidable and executable. `lake exe checks` evaluates these very definitions
on data exported from the enclave store.
-/
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

namespace EnclaveSeries

/-! ### Return sign and the public 5-class taxonomy -/

/-- Sign of a daily log return (bp) with a flat band of width `flatBp`. -/
def retSign (flatBp : ℤ) (r : ℤ) : ℤ :=
  if flatBp < r then 1 else if r < -flatBp then -1 else 0

/-- Public 5-class return taxonomy: 0=crash 1=down 2=flat 3=up 4=surge. -/
def classOfRet (flatBp bigBp : ℤ) (r : ℤ) : Nat :=
  if r ≤ -bigBp then 0
  else if r < -flatBp then 1
  else if r ≤ flatBp then 2
  else if r < bigBp then 3
  else 4

/-- A zero-sign (flat-band) return is the flat class. -/
theorem classOfRet_flat {flatBp bigBp r : ℤ} (hb : flatBp < bigBp)
    (h : retSign flatBp r = 0) : classOfRet flatBp bigBp r = 2 := by
  unfold retSign at h
  unfold classOfRet
  split_ifs at h ⊢ <;> omega

/-- Positive-sign returns land in the up/surge classes (for the sensible
threshold shape 0 ≤ flat < big). -/
theorem classOfRet_pos {flatBp bigBp r : ℤ} (h0 : 0 ≤ flatBp)
    (hb : flatBp < bigBp) (h : retSign flatBp r = 1) :
    classOfRet flatBp bigBp r = 3 ∨ classOfRet flatBp bigBp r = 4 := by
  unfold retSign at h
  unfold classOfRet
  split_ifs at h ⊢ <;> omega

/-! ### Run motif (momentum streak) -/

/-- `IsRun k flatBp rs`: at least `k` daily returns, all with the same
nonzero sign (a same-direction streak). -/
def IsRun (k : ℕ) (flatBp : ℤ) (rs : List ℤ) : Prop :=
  k ≤ rs.length ∧
    ((∀ r ∈ rs, retSign flatBp r = 1) ∨ (∀ r ∈ rs, retSign flatBp r = -1))

instance (k : ℕ) (flatBp : ℤ) (rs : List ℤ) : Decidable (IsRun k flatBp rs) := by
  unfold IsRun; exact inferInstance

/-- Lowering the length threshold preserves run-hood (monotonicity). -/
theorem isRun_mono {k k' : ℕ} {flatBp : ℤ} {rs : List ℤ}
    (hk : k' ≤ k) (h : IsRun k flatBp rs) : IsRun k' flatBp rs :=
  ⟨le_trans hk h.1, h.2⟩

/-- A constant above-band streak is a run (sanity: the definition accepts the
motif it was designed around). -/
theorem isRun_replicate {k n : ℕ} {flatBp g : ℤ} (hk : k ≤ n) (hg : flatBp < g) :
    IsRun k flatBp (List.replicate n g) := by
  refine ⟨by simpa using hk, Or.inl ?_⟩
  intro r hr
  rw [List.eq_of_mem_replicate hr, retSign, if_pos hg]

/-! ### Shock-day motif -/

/-- `IsShock s r`: the day's |log return| reaches the shock threshold. -/
def IsShock (s : ℤ) (r : ℤ) : Prop := s ≤ |r|

instance (s r : ℤ) : Decidable (IsShock s r) := by
  unfold IsShock; exact inferInstance

/-- Some day in the sequence is a shock day. -/
def HasShock (s : ℤ) (rs : List ℤ) : Prop := ∃ r ∈ rs, IsShock s r

instance (s : ℤ) (rs : List ℤ) : Decidable (HasShock s rs) := by
  unfold HasShock; exact inferInstance

/-- Lowering the shock threshold preserves shock-hood (monotonicity). -/
theorem isShock_mono {s s' r : ℤ} (hs : s' ≤ s) (h : IsShock s r) :
    IsShock s' r := le_trans hs h

/-! ### Drawdown motif -/

/-- All (earlier, later) price pairs of a series. -/
def ddPairs : List ℚ → List (ℚ × ℚ)
  | [] => []
  | p :: rest => rest.map (fun q => (p, q)) ++ ddPairs rest

/-- `HasDrawdown thr ps`: some later price sits at least `thr` (a fraction)
below some earlier price — a peak-to-trough decline of ≥ thr. -/
def HasDrawdown (thr : ℚ) (ps : List ℚ) : Prop :=
  ∃ pq ∈ ddPairs ps, pq.2 ≤ (1 - thr) * pq.1

instance (thr : ℚ) (ps : List ℚ) : Decidable (HasDrawdown thr ps) := by
  unfold HasDrawdown; exact inferInstance

/-- The canonical peak-then-trough pair is a drawdown (sanity theorem). -/
theorem hasDrawdown_pair (thr p : ℚ) :
    HasDrawdown thr [p, (1 - thr) * p] := by
  refine ⟨(p, (1 - thr) * p), ?_, le_refl _⟩
  simp [ddPairs]

/-- Relaxing the threshold preserves drawdown-hood (monotonicity; prices are
nonnegative). -/
theorem hasDrawdown_mono {thr thr' : ℚ} {ps : List ℚ}
    (hle : thr' ≤ thr) (hpos : ∀ pq ∈ ddPairs ps, 0 ≤ pq.1)
    (h : HasDrawdown thr ps) : HasDrawdown thr' ps := by
  obtain ⟨pq, hmem, hpq⟩ := h
  exact ⟨pq, hmem,
    le_trans hpq (mul_le_mul_of_nonneg_right (by linarith) (hpos pq hmem))⟩

/-! ### V-recovery motif -/

/-- `IsVRecovery drop level w peak ps`: inside the exported window `ps`
(prices after the peak day) there is a trough ≤ (1−drop)·peak followed within
`w` steps by a recovery ≥ level·peak. -/
def IsVRecovery (drop level : ℚ) (w : ℕ) (peak : ℚ) (ps : List ℚ) : Prop :=
  ∃ jq ∈ ps.zipIdx, jq.1 ≤ (1 - drop) * peak ∧
    ∃ kq ∈ ps.zipIdx, jq.2 < kq.2 ∧ kq.2 ≤ jq.2 + w ∧ level * peak ≤ kq.1

instance (drop level : ℚ) (w : ℕ) (peak : ℚ) (ps : List ℚ) :
    Decidable (IsVRecovery drop level w peak ps) := by
  unfold IsVRecovery; exact inferInstance

/-- Widening the recovery window preserves the motif (monotonicity). -/
theorem isVRecovery_mono_window {drop level : ℚ} {w w' : ℕ} {peak : ℚ}
    {ps : List ℚ} (hw : w ≤ w') (h : IsVRecovery drop level w peak ps) :
    IsVRecovery drop level w' peak ps := by
  obtain ⟨jq, hj, hdrop, kq, hk, hjk, hkw, hrec⟩ := h
  exact ⟨jq, hj, hdrop, kq, hk, hjk, le_trans hkw (by omega), hrec⟩

/-- The canonical trough-then-recovery window is a V-recovery (sanity). -/
theorem isVRecovery_canonical (drop level p : ℚ) {w : ℕ} (hw : 1 ≤ w) :
    IsVRecovery drop level w p [(1 - drop) * p, level * p] := by
  have hz : ([(1 - drop) * p, level * p].zipIdx)
      = [((1 - drop) * p, 0), (level * p, 1)] := rfl
  refine ⟨((1 - drop) * p, 0), ?_, le_refl _,
          (level * p, 1), ?_, ?_, ?_, le_refl _⟩
  · rw [hz]; simp
  · rw [hz]; simp
  · omega
  · omega

/-! ### Volatility-burst motif -/

/-- Mean of a list of rationals (`0` for `[]`, by ℚ's division convention). -/
def qMean (l : List ℚ) : ℚ := l.sum / l.length

/-- Sum of squared deviations from the mean. -/
def qSsd (l : List ℚ) : ℚ := (l.map (fun x => (x - qMean l) ^ 2)).sum

/-- `IsVolBurst thr rs`: the window's sample variance reaches `thr²` —
i.e. sample sd ≥ thr, stated multiplicatively to stay in ℚ. -/
def IsVolBurst (thr : ℚ) (rs : List ℚ) : Prop :=
  2 ≤ rs.length ∧ thr ^ 2 * ((rs.length - 1 : ℕ) : ℚ) ≤ qSsd rs

instance (thr : ℚ) (rs : List ℚ) : Decidable (IsVolBurst thr rs) := by
  unfold IsVolBurst; exact inferInstance

/-- Lowering the sd threshold preserves burst-hood (monotonicity, for
nonnegative thresholds). -/
theorem isVolBurst_mono {thr thr' : ℚ} {rs : List ℚ}
    (h0 : 0 ≤ thr') (hle : thr' ≤ thr) (h : IsVolBurst thr rs) :
    IsVolBurst thr' rs := by
  refine ⟨h.1, le_trans ?_ h.2⟩
  have hsq : thr' ^ 2 ≤ thr ^ 2 := by nlinarith
  exact mul_le_mul_of_nonneg_right hsq (Nat.cast_nonneg _)

/-- A ±g alternation with `thr² ≤ 2g²` is a volatility burst (sanity). -/
theorem isVolBurst_pair {thr g : ℚ} (h : thr ^ 2 ≤ 2 * g ^ 2) :
    IsVolBurst thr [g, -g] := by
  have hm : qMean [g, -g] = 0 := by
    simp [qMean]
  have hs : qSsd [g, -g] = 2 * g ^ 2 := by
    simp [qSsd, hm]
    ring
  refine ⟨by simp, ?_⟩
  rw [hs]
  simpa using h

end EnclaveSeries

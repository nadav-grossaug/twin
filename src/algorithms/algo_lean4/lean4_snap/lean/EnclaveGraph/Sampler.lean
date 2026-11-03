/-
Verified sampler for drawing from a released (noised, clamped) histogram:
a SampCert `UniformSample` over the total weight, bucketed into cells.
This is the entry point for the downstream synthetic-data generator.

Proved here: the exact law of the sampler — the probability of cell `i`
equals `#{r < total | bucketOf ws r = i} / total` (preimage-count form), and
that count equals the cell's weight `ws[i]` (so P(cell i) = wᵢ / Σw).
Normalization is carried by construction in the `SPMF` monad.
-/
import SampCert
import Mathlib.Tactic

namespace EnclaveGraph

open SLang

/-- Index of the bucket containing position `r` in the cumulative layout of `ws`. -/
def bucketOf (ws : List ℕ) (r : ℕ) : ℕ :=
  match ws with
  | [] => 0
  | w :: rest => if r < w then 0 else bucketOf rest (r - w) + 1

/-- The released-histogram sampler: uniform position, mapped to its bucket.
Built with the `SPMF` combinators, so normalization holds by construction and
`PMF.run` can execute it against the /dev/urandom FFI. -/
def sampleCellPMF (ws : List ℕ) (h : 0 < ws.sum) : SPMF ℕ :=
  SPMF_bind (UniformSample_PMF ⟨ws.sum, h⟩) (fun r => SPMF_pure (bucketOf ws r))

/-- Law of a uniform draw post-processed by an arbitrary bucket map: the
probability of hitting `i` is the size of its preimage over the total. -/
theorem uniformBucket_apply (n : ℕ+) (f : ℕ → ℕ) (i : ℕ) :
    probBind (UniformSample n) (fun r => probPure (f r)) i
      = (((Finset.range (n : ℕ)).filter (fun r => f r = i)).card : ENNReal)
        / ((n : ℕ) : ENNReal) := by
  simp only [probBind, probPure]
  -- probPure is defined under `open Classical`; normalize its ite instance
  -- to the canonical Nat decidable equality so later steps unify.
  have hite : ∀ r : ℕ,
      (@ite ENNReal (i = f r) (Classical.propDecidable (i = f r)) (1 : ENNReal) 0)
        = @ite ENNReal (i = f r) (instDecidableEqNat i (f r)) (1 : ENNReal) 0 := by
    intro r
    rw [Subsingleton.elim (Classical.propDecidable (i = f r)) (instDecidableEqNat i (f r))]
  simp only [hite]
  rw [tsum_eq_sum (s := Finset.range (n : ℕ))
      (by
        intro r hr
        rw [UniformSample_apply']
        simp only [Finset.mem_range, not_lt] at hr
        simp [Nat.not_lt.mpr hr])]
  have hterm : ∀ r ∈ Finset.range (n : ℕ),
      (UniformSample n r * if i = f r then 1 else 0)
        = if f r = i then (((n : ℕ) : ENNReal))⁻¹ else 0 := by
    intro r hr
    rw [UniformSample_apply']
    simp only [Finset.mem_range] at hr
    rw [if_pos hr]
    by_cases hb : f r = i
    · simp [hb, one_div]
    · have hb' : ¬ (i = f r) := fun hh => hb hh.symm
      simp [hb, hb']
  refine (Finset.sum_congr rfl hterm).trans ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]

/-- Exact law, preimage-count form: the sampler hits cell `i` with probability
`#{r ∈ [0, Σw) | bucketOf ws r = i} / Σw`. -/
theorem sampleCellPMF_apply (ws : List ℕ) (h : 0 < ws.sum) (i : ℕ) :
    (sampleCellPMF ws h : SLang ℕ) i
      = (((Finset.range ws.sum).filter (fun r => bucketOf ws r = i)).card : ENNReal)
        / (ws.sum : ENNReal) :=
  uniformBucket_apply ⟨ws.sum, h⟩ (bucketOf ws) i

/-- Members of the succ-cell preimage lie in the shifted tail preimage. -/
theorem mem_succ_preimage {w : ℕ} {rest : List ℕ} {i a : ℕ}
    (ha : a ∈ (Finset.range ((w :: rest).sum)).filter
            (fun r => bucketOf (w :: rest) r = i + 1)) :
    w ≤ a ∧ a - w < rest.sum ∧ bucketOf rest (a - w) = i := by
  simp only [Finset.mem_filter, Finset.mem_range, List.sum_cons] at ha
  obtain ⟨hlt, hb⟩ := ha
  by_cases hw : a < w
  · rw [bucketOf, if_pos hw] at hb
    omega
  · rw [bucketOf, if_neg hw] at hb
    exact ⟨by omega, by omega, by omega⟩

/-- The preimage count of cell `i` is exactly its weight `ws[i]`. -/
theorem card_bucketOf : ∀ (ws : List ℕ) (i : ℕ) (hi : i < ws.length),
    ((Finset.range ws.sum).filter (fun r => bucketOf ws r = i)).card = ws[i]'hi
  | [], i, hi => absurd hi (by simp)
  | w :: rest, 0, _ => by
    have hset : (Finset.range ((w :: rest).sum)).filter
        (fun r => bucketOf (w :: rest) r = 0) = Finset.range w := by
      ext r
      constructor
      · intro hr
        have hb : bucketOf (w :: rest) r = 0 := (Finset.mem_filter.mp hr).2
        rw [Finset.mem_range]
        by_contra hw
        rw [bucketOf, if_neg hw] at hb
        omega
      · intro hr
        rw [Finset.mem_range] at hr
        refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr ?_, ?_⟩
        · simp only [List.sum_cons]; omega
        · rw [bucketOf, if_pos hr]
    rw [hset, Finset.card_range]
    simp
  | w :: rest, i + 1, hi => by
    have hi' : i < rest.length := by simpa using hi
    simp only [List.getElem_cons_succ]
    rw [← card_bucketOf rest i hi']
    refine Finset.card_bij' (fun a _ => a - w) (fun b _ => b + w) ?_ ?_ ?_ ?_
    · intro a ha
      obtain ⟨hwa, hlt, hb⟩ := mem_succ_preimage ha
      simp only [Finset.mem_filter, Finset.mem_range]
      exact ⟨hlt, hb⟩
    · intro b hb
      simp only [Finset.mem_filter, Finset.mem_range] at hb
      obtain ⟨hlt, hb'⟩ := hb
      simp only [Finset.mem_filter, Finset.mem_range, List.sum_cons]
      refine ⟨by omega, ?_⟩
      rw [bucketOf, if_neg (by omega : ¬ b + w < w), Nat.add_sub_cancel, hb']
    · intro a ha
      have := (mem_succ_preimage ha).1
      omega
    · intro b _
      omega

/-- Exact law, weight form: `P(cell i) = ws[i] / Σ ws`. -/
theorem sampleCellPMF_law (ws : List ℕ) (h : 0 < ws.sum) (i : ℕ) (hi : i < ws.length) :
    (sampleCellPMF ws h : SLang ℕ) i = (ws[i]'hi : ENNReal) / (ws.sum : ENNReal) := by
  rw [sampleCellPMF_apply, card_bucketOf ws i hi]

end EnclaveGraph

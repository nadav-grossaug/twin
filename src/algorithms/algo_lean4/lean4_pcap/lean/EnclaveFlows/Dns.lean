/-
Formal DNS name-structure definitions, mirroring dbt's feat_dns /
mart_dns_domains.

A query name is modeled as its LABEL LIST (what the DNS wire format actually
carries — length-prefixed labels), not as a dotted string: `nameLen` is the
presentation-form length this model induces, so every theorem is arithmetic
over our own definition and needs no string-parsing lemmas.

* `IsWellFormedQName` — the RFC 1035 shape: at least one label, every label
  1–63 bytes, presentation form ≤ 253 bytes. Decidable; the proved bound
  `wellFormed_labels_le` caps any well-formed name at 127 labels.
* `IsSubdomainFan` — the DNS-tunneling/DGA/CDN motif of
  mart_dns_domains.is_subdomain_fan: one registered domain queried under at
  least `k` distinct names. Definitionally the generic `IsFanout` over
  names, so monotonicity is inherited.

`lake exe checks` re-decides both on every distinct name / domain exported
from the enclave store and compares against the SQL verdicts.
-/
import EnclaveFlows.Motifs
import Mathlib.Tactic

namespace EnclaveFlows

/-- A DNS query name as its label list (wire form). -/
abbrev QName := List String

/-- Presentation-form length: label bytes plus the dots between labels. -/
def nameLen (q : QName) : Nat :=
  (q.map String.length).sum + (q.length - 1)

/-- RFC 1035 shape: nonempty, labels 1–63 bytes, total ≤ 253 bytes
(mirrors feat_dns.is_wellformed; bounds from Params). -/
def IsWellFormedQName (maxLabel maxName : Nat) (q : QName) : Prop :=
  q ≠ [] ∧ nameLen q ≤ maxName ∧ ∀ l ∈ q, 1 ≤ l.length ∧ l.length ≤ maxLabel

instance (maxLabel maxName : Nat) (q : QName) :
    Decidable (IsWellFormedQName maxLabel maxName q) := by
  unfold IsWellFormedQName; exact inferInstance

/-- Every label contributes at least one byte to the name. -/
theorem length_le_sum_of_labels {q : QName}
    (h : ∀ l ∈ q, 1 ≤ l.length) : q.length ≤ (q.map String.length).sum := by
  calc q.length = (q.map String.length).length • 1 := by simp
    _ ≤ (q.map String.length).sum :=
        List.length_nsmul_le_sum (q.map String.length) 1 (fun x hx => by
          obtain ⟨l, hl, rfl⟩ := List.mem_map.mp hx
          exact h l hl)

/-- STRUCTURE BOUND: a well-formed name has at most ⌊(maxName+1)/2⌋ labels
(each label costs ≥ 1 byte plus a separating dot) — with the RFC 1035
`maxName = 253` this caps any legal name at 127 labels. -/
theorem wellFormed_labels_le {maxLabel maxName : Nat} {q : QName}
    (h : IsWellFormedQName maxLabel maxName q) :
    q.length ≤ (maxName + 1) / 2 := by
  obtain ⟨hne, hlen, hlab⟩ := h
  have hpos : 1 ≤ q.length := List.length_pos_of_ne_nil hne
  have hsum : q.length ≤ (q.map String.length).sum :=
    length_le_sum_of_labels (fun l hl => (hlab l hl).1)
  -- 2·n − 1 ≤ sum + (n − 1) = nameLen ≤ maxName
  have h2 : 2 * q.length - 1 ≤ maxName := by
    have : 2 * q.length - 1 ≤ nameLen q := by
      unfold nameLen; omega
    omega
  omega

/-- The subdomain-fan motif: at least `k` distinct names queried under one
registered domain (mirrors mart_dns_domains.is_subdomain_fan). It is the
generic fan-out predicate over names. -/
abbrev IsSubdomainFan (k : Nat) (names : List String) : Prop :=
  IsFanout k names

/-- Monotonicity is inherited from the generic fan-out motif. -/
theorem isSubdomainFan_mono {k k' : ℕ} {names : List String}
    (hk : k' ≤ k) (h : IsSubdomainFan k names) : IsSubdomainFan k' names :=
  isFanout_mono hk h

/-- A fan needs at least `k` names in the list (dedup never grows a list). -/
theorem isSubdomainFan_le_length {k : ℕ} {names : List String}
    (h : IsSubdomainFan k names) : k ≤ names.length :=
  le_trans h (List.Sublist.length_le (List.dedup_sublist names))

end EnclaveFlows

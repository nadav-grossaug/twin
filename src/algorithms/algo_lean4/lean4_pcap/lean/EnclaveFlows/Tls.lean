/-
Formal TLS packet-structure definitions, mirroring dbt's feat_tls /
mart_tls_fingerprints / mart_tls_records.

Two objects are formalized:

* the TLS RECORD SEQUENCE of a session (joy's `srlt`): direction, content
  type (20 ChangeCipherSpec, 21 alert, 22 handshake, 23 application data),
  size and inter-arrival time of every record — the packet structure that
  survives encryption. `IsCanonicalHandshake` is the decidable shape
  predicate: the session opens with a client-sent handshake record and no
  direction carries application data before its ChangeCipherSpec.

* the JA3-style CLIENT-STACK FINGERPRINT (hello version + offered cipher
  suites + extension names). `IsRareStack` is the rare-fingerprint motif of
  mart_tls_fingerprints.

`lake exe checks` re-decides both on every TLS flow exported from the
enclave store and compares against the SQL verdicts.
-/
import Mathlib.Data.List.Count
import Mathlib.Data.List.Dedup
import Mathlib.Tactic

namespace EnclaveFlows

/-- One TLS record as joy's `srlt` reports it. `dirOut = true` is
client→server (`>`); `tp` is the TLS record content type. -/
structure TlsRec where
  dirOut : Bool
  tp : Nat
  b : Nat
  iptMs : Nat
deriving DecidableEq, Repr

/-! ### Canonical handshake (record-structure motif) -/

/--
`NoEarlyData seenOut seenIn l`: scanning `l` left to right, an
application-data record (tp 23) may appear in a direction only after a
ChangeCipherSpec (tp 20) was seen in that direction; the Booleans carry the
"key change seen" state per direction. Structural recursion keeps every
proof a plain induction and gives an executable decision procedure.
-/
def NoEarlyData : Bool → Bool → List TlsRec → Prop
  | _, _, [] => True
  | seenOut, seenIn, r :: rest =>
      (r.tp = 23 → (if r.dirOut then seenOut else seenIn) = true) ∧
      NoEarlyData (seenOut || (r.tp == 20 && r.dirOut))
                  (seenIn  || (r.tp == 20 && !r.dirOut)) rest

instance instDecNoEarlyData :
    ∀ (seenOut seenIn : Bool) (l : List TlsRec), Decidable (NoEarlyData seenOut seenIn l)
  | _, _, [] => isTrue trivial
  | seenOut, seenIn, r :: rest =>
    haveI := instDecNoEarlyData (seenOut || (r.tp == 20 && r.dirOut))
                                (seenIn  || (r.tp == 20 && !r.dirOut)) rest
    inferInstanceAs (Decidable (_ ∧ _))

/-- No key change has been seen yet at the start of a session. -/
def KeyChangeBeforeData (l : List TlsRec) : Prop := NoEarlyData false false l

instance (l : List TlsRec) : Decidable (KeyChangeBeforeData l) :=
  instDecNoEarlyData false false l

/-- The session opens with a client-sent handshake record (the ClientHello). -/
def StartsClientHello : List TlsRec → Prop
  | [] => False
  | r :: _ => r.dirOut = true ∧ r.tp = 22

instance : ∀ (l : List TlsRec), Decidable (StartsClientHello l)
  | [] => isFalse id
  | _ :: _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Canonical TLS session shape (mirrors feat_tls.is_canonical_hs). -/
def IsCanonicalHandshake (l : List TlsRec) : Prop :=
  StartsClientHello l ∧ KeyChangeBeforeData l

instance (l : List TlsRec) : Decidable (IsCanonicalHandshake l) := by
  unfold IsCanonicalHandshake; exact inferInstance

/-- Extraction: a canonical session literally begins with a client handshake
record. -/
theorem startsClientHello_head {l : List TlsRec} (h : StartsClientHello l) :
    ∃ r rest, l = r :: rest ∧ r.dirOut = true ∧ r.tp = 22 := by
  cases l with
  | nil => exact absurd h id
  | cons r rest => exact ⟨r, rest, rfl, h.1, h.2⟩

/-- A record sequence respecting the key-change order never OPENS with
application data. -/
theorem keyChangeBeforeData_head_not_data {r : TlsRec} {rest : List TlsRec}
    (h : KeyChangeBeforeData (r :: rest)) : r.tp ≠ 23 := by
  intro h23
  obtain ⟨h1, -⟩ := h
  have := h1 h23
  cases hd : r.dirOut <;> rw [hd] at this <;> simp at this

/-- Scanning soundness: under `NoEarlyData`, an application-data record in a
direction whose "key seen" flag starts false coexists with a ChangeCipherSpec
record in the same direction. -/
theorem noEarlyData_ccs_of_data {l : List TlsRec} :
    ∀ {so si : Bool}, NoEarlyData so si l →
      ∀ r ∈ l, r.tp = 23 →
        (r.dirOut = true → so = false) → (r.dirOut = false → si = false) →
        ∃ q ∈ l, q.tp = 20 ∧ q.dirOut = r.dirOut := by
  induction l with
  | nil => intro _ _ _ r hr; cases hr
  | cons a rest ih =>
    intro so si h r hr h23 hdt hdf
    obtain ⟨h1, h2⟩ := h
    rcases List.mem_cons.mp hr with rfl | hr
    · -- the head itself cannot be early application data
      have h1' := h1 h23
      cases hd : r.dirOut
      · rw [hd, if_neg (by simp), hdf hd] at h1'; cases h1'
      · rw [hd, if_pos rfl, hdt hd] at h1'; cases h1'
    · cases hd : r.dirOut
      · -- r flows server→client: track the `seenIn` accumulator
        cases hb : (a.tp == 20 && !a.dirOut)
        · have ⟨q, hq, hq20, hqd⟩ := ih h2 r hr h23
            (fun ht => absurd (hd ▸ ht) (by simp))
            (fun _ => by
              rw [hdf hd, hb]; rfl)
          exact ⟨q, List.mem_cons_of_mem a hq, hq20, hqd.trans hd⟩
        · -- a is the witness: a ChangeCipherSpec in r's direction
          simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hb
          exact ⟨a, List.mem_cons_self .., hb.1, hb.2⟩
      · -- r flows client→server: track the `seenOut` accumulator
        cases hb : (a.tp == 20 && a.dirOut)
        · have ⟨q, hq, hq20, hqd⟩ := ih h2 r hr h23
            (fun _ => by
              rw [hdt hd, hb]; rfl)
            (fun hf => absurd (hd ▸ hf) (by simp))
          exact ⟨q, List.mem_cons_of_mem a hq, hq20, hqd.trans hd⟩
        · simp only [Bool.and_eq_true, beq_iff_eq] at hb
          exact ⟨a, List.mem_cons_self .., hb.1, hb.2⟩

/-- MOTIF SOUNDNESS: in a canonical TLS session, encrypted application data
only flows where a key change happened — every application-data record is
accompanied by a ChangeCipherSpec in its direction. -/
theorem canonicalHandshake_data_has_ccs {l : List TlsRec}
    (h : IsCanonicalHandshake l) {r : TlsRec} (hr : r ∈ l) (h23 : r.tp = 23) :
    ∃ q ∈ l, q.tp = 20 ∧ q.dirOut = r.dirOut :=
  noEarlyData_ccs_of_data h.2 r hr h23 (fun _ => rfl) (fun _ => rfl)

/-! ### Client-stack fingerprint (JA3-style) -/

/-- Hello version, offered cipher suites (offer order) and extension names
(sorted) — exactly the triple hashed by feat_tls.fingerprint. -/
structure Fingerprint where
  version : Nat
  cs : List String
  exts : List String
deriving DecidableEq, Repr

/-- The fingerprint histogram: each distinct stack with its flow count. -/
def fpCounts (flows : List Fingerprint) : List (Fingerprint × Nat) :=
  flows.dedup.map (fun f => (f, flows.count f))

/-- The histogram counts PARTITION the TLS flows: Σ counts = N. -/
theorem fpCounts_partition (flows : List Fingerprint) :
    ((fpCounts flows).map Prod.snd).sum = flows.length := by
  rw [fpCounts, List.map_map]
  exact flows.sum_map_count_dedup_eq_length

/-- `IsRareStack k fp flows`: the stack occurs, on at most `k` flows
(mirrors mart_tls_fingerprints.is_rare_stack). -/
def IsRareStack (k : Nat) (fp : Fingerprint) (flows : List Fingerprint) : Prop :=
  fp ∈ flows ∧ flows.count fp ≤ k

instance (k : Nat) (fp : Fingerprint) (flows : List Fingerprint) :
    Decidable (IsRareStack k fp flows) := by
  unfold IsRareStack; exact inferInstance

/-- Rare-stack verdicts are monotone in the rarity threshold. -/
theorem isRareStack_mono {k k' : Nat} {fp : Fingerprint} {flows : List Fingerprint}
    (hk : k ≤ k') (h : IsRareStack k fp flows) : IsRareStack k' fp flows :=
  ⟨h.1, h.2.trans hk⟩

/-- A rare stack still occurs: its count is positive. -/
theorem IsRareStack.count_pos {k : Nat} {fp : Fingerprint} {flows : List Fingerprint}
    (h : IsRareStack k fp flows) : 0 < flows.count fp :=
  List.count_pos_iff.mpr h.1

end EnclaveFlows

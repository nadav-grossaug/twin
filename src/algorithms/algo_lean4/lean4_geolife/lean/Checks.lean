/-
Executable checks connecting the formal motif definitions to enclave data:
re-decides every exported user-day's daily-motif class with the Lean
classifier `motifClassOfDay` (vs the SQL mart), evaluates the DECIDABLE
predicates IsRevisit / IsStay / IsLongDwell and the classifier mirrors
(dwell / hour / displacement / speed / staypoints-per-day ladders) on
exported observations, re-checks maximal sequence-ok visit runs against
`IsVisitSeq`, recomputes the chi-square association statistics EXACTLY over
ℚ from exported contingency cells (vs SQL's floating-point values), and
records the proof-status ledger of the formal layer. Exit code: 0 when every
compared verdict agrees with SQL, 1 otherwise.
-/
import EnclaveGeo.Motifs
import EnclaveGeo.Assoc
import EnclaveGeo.Sequences
import EnclaveGeo.Motion
import EnclaveGeo.GenSpec
import EnclaveGeo.Params
import Lean.Data.Json

open Lean EnclaveGeo

def claim (name statement status : String) : Json :=
  Json.mkObj [("name", Json.str name), ("statement", Json.str statement),
              ("status", Json.str status)]

def claims : Json := Json.arr #[
  claim "clippedSum_sensitivity"
    "Adding/removing one user changes a clipped cell sum by ≤ clip (sensitivity = clip)" "proved",
  claim "privClippedCell_PureDP"
    "The per-cell release mechanism is pure (ε₁/ε₂)-DP (via SampCert privNoisedQueryPure_DP)" "proved",
  claim "cellQuery_sensitivity"
    "The per-cell clipped histogram query has sensitivity = clip (via exactBoundedSum + neighbour_map)" "proved",
  claim "histMech_DP"
    "The composed K-cell histogram mechanism is pure (K·ε₁/ε₂)-DP (privCompose/privPostProcess induction)" "proved",
  claim "sampleCellPMF_law"
    "The release sampler draws cell i with probability wᵢ/Σw exactly (via SampCert UniformSample_apply')" "proved",
  claim "motifClassOfDay_mem"
    "The daily-motif classifier is total over the 5 public classes (no day escapes the DP grid)" "proved",
  claim "commute_canonical"
    "The canonical H→W→H day classifies as commute" "proved",
  claim "tour_priority_canonical"
    "A 3-anchor chain day classifies as tour even when it also commutes (priority order)" "proved",
  claim "errand_canonical"
    "The canonical H→O(<3h)→H day classifies as errand_loop" "proved",
  claim "gap_breaks_commute"
    "A recording gap beyond visit_gap_break_s never creates a transition (the commute-shaped day with a hole classifies irregular)" "proved",
  claim "isRevisit_canonical"
    "The canonical 5-weekday anchor revisit satisfies IsRevisit" "proved",
  claim "isRevisitAt_mono"
    "IsRevisit is monotone in the day threshold" "proved",
  claim "isRevisitAt_mono_window"
    "IsRevisit is monotone in the window" "proved",
  claim "isVisitSeqAt_mono"
    "Visit sequences are preserved under relaxing the gap bound" "proved",
  claim "isVisitSeqAt_ordered"
    "A visit sequence is chronologically ordered" "proved",
  claim "mkStays_times_sorted"
    "Generator contract: rendered arrive/leave times are strictly increasing down the day (List.Pairwise)" "proved",
  claim "mkStays_isStay"
    "Generator contract: every planted stay clears the public staypoint dwell threshold" "proved",
  claim "mkStays_visitSeq"
    "Generator contract: gap-bounded day plans render sequence-ok visit chains (IsVisitSeqAt)" "proved",
  claim "mkStays_le"
    "Generator contract: rendered times are bounded by start + plan cost (window containment)" "proved",
  claim "stepTimes_sorted"
    "Generator contract: positive gaps produce strictly increasing event times" "proved",
  claim "wobble_box_fits"
    "Generator contract: stay wobble (2·cap) fits inside the staypoint detection radius" "proved",
  claim "genStay_isStay"
    "Generator contract: the generator's minimum stay dwell clears the staypoint threshold" "proved",
  claim "trip_interval_le_gap"
    "Generator contract: the trip-duration cap never breaks a visit sequence" "proved",
  claim "genTrip_chains"
    "Generator contract: the trip-duration cap respects the day-chain (tour) bound" "proved",
  claim "render_speed_within_cap"
    "Generator contract: every rendered cruise speed is inside the public plausibility cap" "proved",
  claim "modeRep_roundtrip"
    "Generator contract: every rendered raw mode label maps back to its class through the public mode map" "proved",
  claim "excl_tgt_disjoint"
    "The Beijing exclusion box and the Tokyo target box are disjoint (the manipulate contract is satisfiable)" "proved",
  claim "jaRange_is_junction"
    "Generator contract: every rendered junction angle clears the public junction threshold and never degenerates to a reversal" "proved",
  claim "segRange_resolvable"
    "Generator contract: every rendered route segment survives the public route-resampling grid" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "sql_counts_correct"
    "The dbt SQL/Python nodes computed staypoints/visits/user-days correctly from the raw TSV" "assumed",
  claim "entropy_source"
    "SampCert FFI randomness (/dev/urandom) is a fair entropy source" "assumed",
  claim "io_wrappers"
    "JSON/TSV I/O wrappers in the executables are correct" "assumed"]

def getNatD (j : Json) (k : String) (d : Nat) : Nat :=
  ((j.getObjVal? k).bind Json.getNat?).toOption.getD d

def ratToFloat (q : ℚ) : Float :=
  Float.ofInt q.num / Float.ofNat q.den

def readJson? (path : System.FilePath) : IO (Option Json) := do
  try
    let txt ← IO.FS.readFile path
    match Json.parse txt with
    | .ok j => pure (some j)
    | .error _ => pure none
  catch _ =>
    pure none

def natListOf (j : Json) (k : String) : List Nat :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun x => (x.getInt?.toOption.getD 0).toNat)

def boolOf (j : Json) (k : String) : Bool :=
  ((j.getObjVal? k).bind Json.getBool?).toOption.getD false

def strOf (j : Json) (k : String) : String :=
  ((j.getObjVal? k).bind Json.getStr?).toOption.getD ""

def natOf (j : Json) (k : String) : Nat :=
  (((j.getObjVal? k).bind Json.getInt?).toOption.getD 0).toNat

def visitOf (j : Json) : VisitObs :=
  { role := strOf j "role",
    anchor := natOf j "anchor",
    arrive := natOf j "arrive_s",
    leave := natOf j "leave_s",
    prevRole := ((j.getObjVal? "prev_role").bind Json.getStr?).toOption,
    prevLeave := (((j.getObjVal? "prev_leave_s").bind Json.getInt?).toOption.map
      Int.toNat) }

/-- One compared classifier ladder: Lean recomputes the class from the raw
value and compares against SQL's. -/
def ladder (items : Array Json) (rawKey sqlKey : String) (f : Nat → String) :
    Nat × Nat × Array Json := Id.run do
  let mut agree := 0
  let mut disagree := 0
  let mut out : Array Json := #[]
  for b in items do
    let leanC := f (natOf b rawKey)
    let sqlC := strOf b sqlKey
    if leanC == sqlC then agree := agree + 1 else disagree := disagree + 1
    out := out.push <| Json.mkObj
      [ (rawKey, (b.getObjVal? rawKey).toOption.getD Json.null)
      , ("lean_class", Json.str leanC)
      , ("sql_class", Json.str sqlC)
      , ("agree", Json.bool (leanC == sqlC)) ]
  return (agree, disagree, out)

def arrOf (j : Json) (k : String) : Array Json :=
  ((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/motifs.json"
  match Json.parse txt with
  | .error e => IO.eprintln s!"checks: bad input: {e}"; return 1
  | .ok j =>
    let params := (j.getObjVal? "params").toOption.getD Json.null
    let mut nAgree := 0
    let mut nDisagree := 0

    -- daily motifs: the Lean classifier re-decides every exported user-day
    let mut dayResults : Array Json := #[]
    for d in arrOf j "days" do
      let visits := (arrOf d "visits").toList.map visitOf
      let leanM := motifClassOfDay visits
      let sqlM := strOf d "sql_motif"
      if leanM == sqlM then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      dayResults := dayResults.push <| Json.mkObj
        [ ("user", (d.getObjVal? "user").toOption.getD Json.null)
        , ("day_idx", (d.getObjVal? "day_idx").toOption.getD Json.null)
        , ("lean_motif", Json.str leanM)
        , ("sql_motif", Json.str sqlM)
        , ("agree", Json.bool (leanM == sqlM)) ]

    -- revisit beacons
    let mut revisitResults : Array Json := #[]
    for b in arrOf j "revisits" do
      let days := natListOf b "days"
      let leanR := decide (IsRevisit days)
      let sqlR := boolOf b "sql_is_revisit"
      if leanR == sqlR then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      revisitResults := revisitResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("anchor", (b.getObjVal? "anchor").toOption.getD Json.null)
        , ("n_days", Json.num (JsonNumber.fromNat days.length))
        , ("lean_is_revisit", Json.bool leanR)
        , ("sql_is_revisit", Json.bool sqlR)
        , ("agree", Json.bool (leanR == sqlR)) ]

    -- classifier ladders
    let (a1, d1, dwellResults) := ladder (arrOf j "dwells") "dwell_s" "sql_class" dwellClass
    let (a2, d2, hourResults) := ladder (arrOf j "hours") "hour" "sql_bin" hourBin
    let (a3, d3, distResults) := ladder (arrOf j "dists") "disp_m" "sql_class" distClass
    let (a4, d4, speedResults) := ladder (arrOf j "speeds") "mmps" "sql_class" speedClass
    let (a5, d5, spdResults) := ladder (arrOf j "spd_bins") "n" "sql_bin" spdBin
    let (a6, d6, segResults) := ladder (arrOf j "segs") "seg_m" "sql_class" segClass
    let (a7, d7, jaResults) := ladder (arrOf j "jas") "abs_deg" "sql_class" jaClass
    let (a8, d8, circResults) := ladder (arrOf j "circs") "circ_pm" "sql_class" circClass
    nAgree := nAgree + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    nDisagree := nDisagree + d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8

    -- long dwells
    let mut longResults : Array Json := #[]
    for b in arrOf j "longdwells" do
      let leanL := decide (IsLongDwell (natOf b "dwell_s"))
      let sqlL := boolOf b "sql_is_longdwell"
      if leanL == sqlL then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      longResults := longResults.push <| Json.mkObj
        [ ("dwell_s", (b.getObjVal? "dwell_s").toOption.getD Json.null)
        , ("lean_is_longdwell", Json.bool leanL)
        , ("sql_is_longdwell", Json.bool sqlL)
        , ("agree", Json.bool (leanL == sqlL)) ]

    -- maximal sequence-ok runs must satisfy IsVisitSeq (SQL cut them at the
    -- same gap bound, so every run must verify)
    let mut seqResults : Array Json := #[]
    if let some sj ← readJson? "data/seqs.json" then
      let mg := getNatD sj "visit_gap_break_s" Params.visitGapBreakS
      for s in arrOf sj "runs" do
        let ivs := (arrOf s "intervals").toList.map (fun iv =>
          match iv.getArr?.toOption.getD #[] with
          | #[a, l] => (((a.getInt?.toOption.getD 0).toNat),
                        ((l.getInt?.toOption.getD 0).toNat))
          | _ => (0, 0))
        let leanS := decide (IsVisitSeqAt mg ivs)
        if leanS then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        seqResults := seqResults.push <| Json.mkObj
          [ ("user", (s.getObjVal? "user").toOption.getD Json.null)
          , ("n_visits", Json.num (JsonNumber.fromNat ivs.length))
          , ("lean_is_visit_seq", Json.bool leanS)
          , ("agree", Json.bool leanS) ]

    -- exact chi-square recomputation on exported contingency tables
    let mut assocResults : Array Json := #[]
    if let some aj ← readJson? "data/assoc.json" then
      for pa in arrOf aj "pairs" do
        let rowsJ := arrOf pa "observed"
        let table : ContTable := rowsJ.toList.map (fun row =>
          (row.getArr?.toOption.getD #[]).toList.map
            (fun c => ((c.getInt?.toOption.getD 0 : Int) : ℚ)))
        let leanF := ratToFloat (chiSq table)
        let sqlF := ((((pa.getObjVal? "sql_chi2").bind Json.getNum?).toOption.map
          (fun n => n.toFloat))).getD 0.0
        let tol := 1e-6 * max 1.0 (Float.abs leanF)
        let agree := Float.abs (leanF - sqlF) ≤ tol
        if agree then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        assocResults := assocResults.push <| Json.mkObj
          [ ("feat_a", (pa.getObjVal? "feat_a").toOption.getD (Json.str "?"))
          , ("feat_b", (pa.getObjVal? "feat_b").toOption.getD (Json.str "?"))
          , ("lean_chi2", Json.str (toString leanF))
          , ("sql_chi2", Json.str (toString sqlF))
          , ("agree", Json.bool agree) ]

    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("params", params)
      , ("motif_days", Json.arr dayResults)
      , ("revisits", Json.arr revisitResults)
      , ("dwell_classes", Json.arr dwellResults)
      , ("hour_bins", Json.arr hourResults)
      , ("dist_classes", Json.arr distResults)
      , ("speed_classes", Json.arr speedResults)
      , ("spd_bins", Json.arr spdResults)
      , ("seg_classes", Json.arr segResults)
      , ("ja_classes", Json.arr jaResults)
      , ("circ_classes", Json.arr circResults)
      , ("longdwells", Json.arr longResults)
      , ("visit_seqs", Json.arr seqResults)
      , ("assoc", Json.arr assocResults)
      , ("agreement", Json.mkObj
          [ ("n_checked", Json.num (JsonNumber.fromNat (nAgree + nDisagree)))
          , ("n_agree", Json.num (JsonNumber.fromNat nAgree))
          , ("n_disagree", Json.num (JsonNumber.fromNat nDisagree)) ]) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {dayResults.size} motif-day, {revisitResults.size} revisit, {dwellResults.size}+{hourResults.size}+{distResults.size}+{speedResults.size}+{spdResults.size}+{segResults.size}+{jaResults.size}+{circResults.size} class, {longResults.size} longdwell, {seqResults.size} seq, {assocResults.size} chi-square checks -> out/lean_results.json ({nDisagree} disagreements)"
    return (if nDisagree = 0 then 0 else 1)

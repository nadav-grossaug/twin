/-
Executable checks connecting the formal motif definitions to enclave data:
evaluates the DECIDABLE Lean predicates HasRepeatCity / IsCorridor /
IsTour / AnyLongStay / IsLongStay / IsWellFormedTrip / IsContiguousTrip on
observations exported from the dbt marts, recomputes the chi-square
association statistics EXACTLY over ℚ from exported contingency cells (vs
SQL's floating-point values), and records the proof-status ledger of the
formal layer. Exit code: 0 when every compared verdict agrees with SQL,
1 otherwise.
-/
import EnclaveTrip.Motifs
import EnclaveTrip.Assoc
import EnclaveTrip.Sequences
import EnclaveTrip.TripShape
import EnclaveTrip.GenSpec
import EnclaveTrip.Params
import Lean.Data.Json

open Lean EnclaveTrip

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
  claim "hasRepeatCity_canonical"
    "The canonical A→B→A trip satisfies HasRepeatCity (a revisited city within one trip)" "proved",
  claim "hasRepeatCity_cons_self"
    "Re-visiting any city already on the trip creates a repeat (prepending an in-list city)" "proved",
  claim "isCorridor_const"
    "A corridor with constant positive gaps and ≥ corridor_min_repeats repeats satisfies IsCorridor (CV = 0)" "proved",
  claim "gapsRegular_replicate"
    "Constant positive gap lists satisfy the exact rational CV bound den²·k·s2 ≤ (den²+num²)·s1²" "proved",
  claim "isCorridorAt_mono"
    "IsCorridor is monotone in the repeat threshold" "proved",
  claim "isTour_canonical"
    "The canonical 4-leg, 3-country, contiguous trip satisfies IsTour" "proved",
  claim "isTour_mono"
    "IsTour is monotone in the checkin-gap bound" "proved",
  claim "isLongStay_mono"
    "IsLongStay is monotone in the nights threshold" "proved",
  claim "anyLongStay_canonical"
    "A trip containing an 8-night stay satisfies AnyLongStay" "proved",
  claim "isWellFormedAt_mono"
    "Well-formed trips are preserved under relaxing the checkin-gap bound" "proved",
  claim "isWellFormedAt_ordered"
    "A well-formed trip is strictly chronologically ordered" "proved",
  claim "contiguous_sorted"
    "Generator contract: a contiguous trip has strictly increasing checkin days (List.Pairwise)" "proved",
  claim "isContiguousTrip_canonical"
    "The canonical one-night-per-leg contiguous trip satisfies IsContiguousTrip" "proved",
  claim "mkLegs_checkins_sorted"
    "Generator contract: stays of ≥ 1 night render strictly increasing per-trip checkin days (List.Pairwise)" "proved",
  claim "mkLegs_contiguous"
    "Generator contract: a zero-gap leg plan renders the exact contiguous flow (next checkin = previous checkout)" "proved",
  claim "mkLegs_in_window"
    "Generator contract: rendered checkins stay in the public window when start + total span fits" "proved",
  claim "mkLegs_no_overlap"
    "Generator contract: a stay never extends past the next checkin; the last checkout stays ≤ window end" "proved",
  claim "mkLeg_gap_range"
    "Generator contract: every smart-constructed leg's gap lands in the public range [0, 33]" "proved",
  claim "mkLeg_nights_range"
    "Generator contract: every smart-constructed leg's nights land in the public range [1, 30]" "proved",
  claim "legsWithin_span_le"
    "Generator contract: the window clamp keeps the plan's total span at or before the window end" "proved",
  claim "legsWithin_subset"
    "Generator contract: the window clamp keeps a prefix of the original leg plan" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "sql_counts_correct"
    "The dbt SQL computed per-user reservations/gaps/trip shapes correctly from the raw CSV" "assumed",
  claim "entropy_source"
    "SampCert FFI randomness (/dev/urandom) is a fair entropy source" "assumed",
  claim "io_wrappers"
    "JSON/CSV I/O wrappers in the executables are correct" "assumed"]

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

def intListOf (j : Json) (k : String) : List Int :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun x => x.getInt?.toOption.getD 0)

def natListOf (j : Json) (k : String) : List Nat :=
  (intListOf j k).map Int.toNat

/-- First non-empty Nat-list among the given keys (tolerant reader: the
exporter's key naming may vary between spec revisions). -/
def natListAny (j : Json) : List String → List Nat
  | [] => []
  | k :: rest =>
      match natListOf j k with
      | [] => natListAny j rest
      | xs => xs

/-- Gap-normalize a trip's checkin days by its per-leg nights so that
consecutive differences equal the SQL inter-leg gaps
(`gap_i = checkin[i+1] − (checkin[i] + nights[i])`, the checkout-to-checkin
distance). Overlapping legs truncate to 0 — overlaps are caught separately
by the trip well-formedness check. -/
def gapNormFrom (acc : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | [d], _ => [acc]
  | d1 :: d2 :: ds, n1 :: ns =>
      acc :: gapNormFrom (acc + (d2 - (d1 + n1))) (d2 :: ds) ns
  | d1 :: d2 :: ds, [] =>
      acc :: gapNormFrom (acc + (d2 - d1)) (d2 :: ds) []

def strListOf (j : Json) (k : String) : List String :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun x => x.getStr?.toOption.getD "")

/-- First non-empty String-list among the given keys (tolerant reader). -/
def strListAny (j : Json) : List String → List String
  | [] => []
  | k :: rest =>
      match strListOf j k with
      | [] => strListAny j rest
      | xs => xs

def boolOf (j : Json) (k : String) : Bool :=
  ((j.getObjVal? k).bind Json.getBool?).toOption.getD false

def strOf (j : Json) (k : String) : String :=
  ((j.getObjVal? k).bind Json.getStr?).toOption.getD ""

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/motifs.json"
  match Json.parse txt with
  | .error e => IO.eprintln s!"checks: bad input: {e}"; return 1
  | .ok j =>
    let params := (j.getObjVal? "params").toOption.getD Json.null
    -- defaults come from the mirrored concept spec (Params.lean); the CV
    -- bound (corridor_cv_max_num/den) is baked into GapsRegular from Params
    -- and only echoed here — tests/test_spec_sync.py keeps them in lockstep.
    let minRep := getNatD params "corridor_min_repeats" Params.corridorMinRepeats
    let tourGap := getNatD params "tour_max_gap_d" Params.tourMaxGapD
    let longMin := getNatD params "longstay_min_nights" Params.longstayMinNights
    let mut nAgree := 0
    let mut nDisagree := 0

    -- returns: a trip's city list with a revisit + the SQL verdict
    let mut returnResults : Array Json := #[]
    for b in ((j.getObjVal? "returns").bind Json.getArr?).toOption.getD #[] do
      let cities := strListOf b "cities"
      let leanR := decide (HasRepeatCity cities)
      let sqlR := boolOf b "sql_is_return"
      if leanR == sqlR then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      returnResults := returnResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_cities", Json.num (JsonNumber.fromNat cities.length))
        , ("lean_is_return", Json.bool leanR)
        , ("sql_is_return", Json.bool sqlR)
        , ("agree", Json.bool (leanR == sqlR)) ]

    -- corridors: inter-repeat gap list (n_repeats = gaps + 1) + SQL verdict
    let mut corridorResults : Array Json := #[]
    for b in ((j.getObjVal? "corridors").bind Json.getArr?).toOption.getD #[] do
      let gaps := natListAny b ["gaps_d", "gaps_s"]
      let nRep := getNatD b "n_repeats" (gaps.length + 1)
      let leanC := decide (IsCorridorAt minRep nRep gaps)
      let sqlC := boolOf b "sql_is_corridor"
      if leanC == sqlC then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      corridorResults := corridorResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_repeats", Json.num (JsonNumber.fromNat nRep))
        , ("lean_is_corridor", Json.bool leanC)
        , ("sql_is_corridor", Json.bool sqlC)
        , ("agree", Json.bool (leanC == sqlC)) ]

    -- tours: gap-normalized checkin days (via the per-leg nights) + visited
    -- hotel countries + SQL verdict — the chain steps are then exactly the
    -- SQL gap days, so the decider judges the same predicate SQL judged
    let mut tourResults : Array Json := #[]
    for b in ((j.getObjVal? "tours").bind Json.getArr?).toOption.getD #[] do
      let days := natListAny b ["days", "ts_s", "days_s"]
      let nights := natListAny b ["nights", "nights_list", "nights_d"]
      let t := if nights.isEmpty then days else gapNormFrom 0 days nights
      let cs := strListAny b ["countries", "country_list"]
      let leanT := decide (IsTourAt tourGap t cs)
      let sqlT := boolOf b "sql_is_tour"
      if leanT == sqlT then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      tourResults := tourResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_legs", Json.num (JsonNumber.fromNat days.length))
        , ("n_countries", Json.num (JsonNumber.fromNat cs.length))
        , ("lean_is_tour", Json.bool leanT)
        , ("sql_is_tour", Json.bool sqlT)
        , ("agree", Json.bool (leanT == sqlT)) ]

    -- long stays: a trip's nights list, a scalar nights value, or one
    -- duration, each + the SQL verdict
    let mut longstayResults : Array Json := #[]
    for b in ((j.getObjVal? "longstays").bind Json.getArr?).toOption.getD #[] do
      let nights := natListAny b ["nights_list", "nights_d"]
      let d :=
        if nights.isEmpty then
          getNatD b "nights" (getNatD b "duration_d" (getNatD b "duration_s" 0))
        else
          nights.foldl max 0
      let leanL := decide (IsLongStayAt longMin d)
      let sqlL := boolOf b "sql_is_longstay"
      if leanL == sqlL then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      longstayResults := longstayResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("lean_is_longstay", Json.bool leanL)
        , ("sql_is_longstay", Json.bool sqlL)
        , ("agree", Json.bool (leanL == sqlL)) ]

    -- exact chi-square recomputation on exported contingency tables
    let mut assocResults : Array Json := #[]
    if let some aj ← readJson? "data/assoc.json" then
      let pairsA := ((aj.getObjVal? "pairs").bind Json.getArr?).toOption.getD #[]
      for pa in pairsA do
        let rowsJ := ((pa.getObjVal? "observed").bind Json.getArr?).toOption.getD #[]
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

    -- trips: the temporal-flow check — Lean re-decides the contiguous-flow
    -- predicate on (checkin day, nights) observations and compares against
    -- the SQL flag; the well-formed (ordered, bounded-gap) verdict is
    -- recorded lean-only, as the exported cut need not match it.
    let mut tripResults : Array Json := #[]
    if let some tj ← readJson? "data/trips.json" then
      let mg := getNatD tj "seqbreak_gap_d" Params.seqbreakGapD
      for t in ((tj.getObjVal? "trips").bind Json.getArr?).toOption.getD #[] do
        let days := natListAny t ["days", "days_d"]
        let nights := natListAny t ["nights", "nights_list"]
        let obs := (days.zip nights).map (fun p => ⟨p.1, p.2⟩)
        let leanContig := decide (IsContiguousTrip obs) && obs.length == days.length
        let sqlContig := boolOf t "sql_all_contiguous"
        if leanContig == sqlContig then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        -- 1.3.0 booking-channel constancy motifs: re-decide the per-trip
        -- device / affiliate class plans against the SQL flags
        let dvs := strListAny t ["devices", "device_list"]
        let afs := strListAny t ["affs", "aff_list"]
        let cos := strListAny t ["countries", "country_list"]
        let leanDev := decide (IsConstantRun dvs) && dvs.length == days.length
        let leanAff := decide (IsConstantRun afs) && afs.length == days.length
        let leanSingle := decide (IsSingleCountryTrip cos)
        let sqlDev := boolOf t "sql_is_devconst"
        let sqlAff := boolOf t "sql_is_affconst"
        let sqlSingle := boolOf t "sql_is_single_country"
        if leanDev == sqlDev then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        if leanAff == sqlAff then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        if leanSingle == sqlSingle then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        tripResults := tripResults.push <| Json.mkObj
          [ ("user", (t.getObjVal? "user").toOption.getD Json.null)
          , ("utrip", (t.getObjVal? "utrip").toOption.getD Json.null)
          , ("n_legs", Json.num (JsonNumber.fromNat days.length))
          , ("lean_is_contiguous", Json.bool leanContig)
          , ("sql_all_contiguous", Json.bool sqlContig)
          , ("lean_is_wellformed", Json.bool (decide (IsWellFormedTripAt mg days)))
          , ("lean_is_devconst", Json.bool leanDev)
          , ("sql_is_devconst", Json.bool sqlDev)
          , ("devconst_agree", Json.bool (leanDev == sqlDev))
          , ("lean_is_affconst", Json.bool leanAff)
          , ("sql_is_affconst", Json.bool sqlAff)
          , ("affconst_agree", Json.bool (leanAff == sqlAff))
          , ("lean_is_single_country", Json.bool leanSingle)
          , ("sql_is_single_country", Json.bool sqlSingle)
          , ("single_country_agree", Json.bool (leanSingle == sqlSingle))
          , ("agree", Json.bool (leanContig == sqlContig)) ]

    let agreement := Json.mkObj
      [ ("n_checked", Json.num (JsonNumber.fromNat (nAgree + nDisagree)))
      , ("n_agree", Json.num (JsonNumber.fromNat nAgree))
      , ("n_disagree", Json.num (JsonNumber.fromNat nDisagree)) ]
    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("params", params)
      , ("returns", Json.arr returnResults)
      , ("corridors", Json.arr corridorResults)
      , ("tours", Json.arr tourResults)
      , ("longstays", Json.arr longstayResults)
      , ("assoc", Json.arr assocResults)
      , ("trips", Json.arr tripResults)
      , ("agreement", agreement)
      , ("agreement_counts", agreement) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {returnResults.size} return, {corridorResults.size} corridor, {tourResults.size} tour, {longstayResults.size} longstay, {assocResults.size} chi-square, {tripResults.size} trip checks -> out/lean_results.json ({nDisagree} disagreements)"
    return (if nDisagree = 0 then 0 else 1)

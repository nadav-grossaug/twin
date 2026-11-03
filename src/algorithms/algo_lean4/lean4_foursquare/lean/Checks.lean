/-
Executable checks connecting the formal motif definitions to enclave data:
evaluates the DECIDABLE Lean predicates IsReturnPair / IsCorridor / IsTour /
IsBurst / IsLongStay / IsEpisode on observations exported from the dbt marts,
recomputes the chi-square association statistics EXACTLY over ℚ from exported
contingency cells (vs SQL's floating-point values), and records the
proof-status ledger of the formal layer. Exit code: 0 when every compared
verdict agrees with SQL, 1 otherwise.
-/
import EnclaveTravel.Motifs
import EnclaveTravel.Assoc
import EnclaveTravel.Sequences
import EnclaveTravel.TravelShape
import EnclaveTravel.GenSpec
import EnclaveTravel.Params
import Lean.Data.Json

open Lean EnclaveTravel

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
  claim "isReturnPair_canonical"
    "The canonical consecutive pair a→b, b→a satisfies IsReturnPair" "proved",
  claim "isCorridor_const"
    "A commuter corridor with constant positive gaps and ≥ corridor_min_repeats repeats satisfies IsCorridor (CV = 0)" "proved",
  claim "gapsRegular_replicate"
    "Constant positive gap lists satisfy the exact rational CV bound den²·k·s2 ≤ (den²+num²)·s1²" "proved",
  claim "isCorridorAt_mono"
    "IsCorridor is monotone in the repeat threshold" "proved",
  claim "isTour_canonical"
    "The canonical 3-hop, 3-country, day-apart trip satisfies IsTour" "proved",
  claim "isTour_mono"
    "IsTour is monotone in the hop-gap bound" "proved",
  claim "isBurst_canonical"
    "The canonical tight timestamp triple satisfies IsBurst" "proved",
  claim "isBurst_mono"
    "IsBurst is monotone in the window" "proved",
  claim "isLongStay_mono"
    "IsLongStay is monotone in the duration threshold" "proved",
  claim "isEpisodeAt_mono"
    "Episodes are preserved under relaxing the gap bound" "proved",
  claim "isEpisodeAt_ordered"
    "An episode is strictly chronologically ordered" "proved",
  claim "mkUserHistory_times_sorted"
    "Generator contract: positive gaps render strictly increasing per-user timestamps (List.Pairwise)" "proved",
  claim "mkUserHistory_prev_ne_next"
    "Generator contract: adjacent-distinct hop plans render prev_country ≠ next_country on every row" "proved",
  claim "mkUserHistory_durChained"
    "Generator contract: row i+1's prev_duration equals row i's next_duration, by construction of the renderer" "proved",
  claim "mkUserHistory_ts_in_window"
    "Generator contract: rendered timestamps lie in [window_start_s, window_end_s) (with hopsWithin_sum_lt clamping)" "proved",
  claim "mkUserHistory_no_overlap"
    "Generator contract: a user's stay interval never extends past their next transfer (no simultaneous presence in two countries)" "proved",
  claim "stepTimes_sorted"
    "Generator contract: positive gaps produce strictly increasing event times" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "sql_counts_correct"
    "The dbt SQL computed per-user transfers/gaps/episodes correctly from the raw TSV" "assumed",
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

def intListOf (j : Json) (k : String) : List Int :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun x => x.getInt?.toOption.getD 0)

def natListOf (j : Json) (k : String) : List Nat :=
  (intListOf j k).map Int.toNat

def strListOf (j : Json) (k : String) : List String :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun x => x.getStr?.toOption.getD "")

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
    let tourGap := getNatD params "tour_max_gap_s" Params.tourMaxGapS
    let burstWin := getNatD params "burst_window_s" Params.burstWindowS
    let longMin := getNatD params "longstay_min_s" Params.longstayMinS
    let mut nAgree := 0
    let mut nDisagree := 0

    -- return pairs: the two consecutive transfers a→b, c→d + the SQL verdict
    let mut returnResults : Array Json := #[]
    for b in ((j.getObjVal? "returns").bind Json.getArr?).toOption.getD #[] do
      let leanR := decide (IsReturnPair (strOf b "a") (strOf b "b")
        (strOf b "c") (strOf b "d"))
      let sqlR := boolOf b "sql_is_return"
      if leanR == sqlR then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      returnResults := returnResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("lean_is_return", Json.bool leanR)
        , ("sql_is_return", Json.bool sqlR)
        , ("agree", Json.bool (leanR == sqlR)) ]

    -- corridors: inter-repeat gap list (n_repeats = gaps + 1) + SQL verdict
    let mut corridorResults : Array Json := #[]
    for b in ((j.getObjVal? "corridors").bind Json.getArr?).toOption.getD #[] do
      let gaps := natListOf b "gaps_s"
      let leanC := decide (IsCorridorAt minRep (gaps.length + 1) gaps)
      let sqlC := boolOf b "sql_is_corridor"
      if leanC == sqlC then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      corridorResults := corridorResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_repeats", Json.num (JsonNumber.fromNat (gaps.length + 1)))
        , ("lean_is_corridor", Json.bool leanC)
        , ("sql_is_corridor", Json.bool sqlC)
        , ("agree", Json.bool (leanC == sqlC)) ]

    -- tours: hop timestamps + visited countries + SQL verdict
    let mut tourResults : Array Json := #[]
    for b in ((j.getObjVal? "tours").bind Json.getArr?).toOption.getD #[] do
      let ts := natListOf b "ts_s"
      let cs := strListOf b "countries"
      let leanT := decide (IsTourAt tourGap ts cs)
      let sqlT := boolOf b "sql_is_tour"
      if leanT == sqlT then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      tourResults := tourResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_hops", Json.num (JsonNumber.fromNat ts.length))
        , ("lean_is_tour", Json.bool leanT)
        , ("sql_is_tour", Json.bool sqlT)
        , ("agree", Json.bool (leanT == sqlT)) ]

    -- bursts: timestamp window + SQL verdict
    let mut burstResults : Array Json := #[]
    for b in ((j.getObjVal? "bursts").bind Json.getArr?).toOption.getD #[] do
      let ts := natListOf b "ts_s"
      let leanB := decide (IsBurstAt burstWin ts)
      let sqlB := boolOf b "sql_is_burst"
      if leanB == sqlB then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      burstResults := burstResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("n_transfers", Json.num (JsonNumber.fromNat ts.length))
        , ("lean_is_burst", Json.bool leanB)
        , ("sql_is_burst", Json.bool sqlB)
        , ("agree", Json.bool (leanB == sqlB)) ]

    -- long stays: one duration + SQL verdict
    let mut longstayResults : Array Json := #[]
    for b in ((j.getObjVal? "longstays").bind Json.getArr?).toOption.getD #[] do
      let d := (((b.getObjVal? "duration_s").bind Json.getInt?).toOption.getD 0).toNat
      let leanL := decide (IsLongStayAt longMin d)
      let sqlL := boolOf b "sql_is_longstay"
      if leanL == sqlL then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
      longstayResults := longstayResults.push <| Json.mkObj
        [ ("user", (b.getObjVal? "user").toOption.getD Json.null)
        , ("duration_s", Json.num (JsonNumber.fromNat d))
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

    -- episodes: IsEpisode over transfer timestamps (lean-only verdict; the
    -- exported episodes were cut by SQL at the same gap bound)
    let mut episodeResults : Array Json := #[]
    if let some sj ← readJson? "data/episodes.json" then
      let mg := getNatD sj "episode_gap_max_s" Params.episodeGapMaxS
      for s in ((sj.getObjVal? "episodes").bind Json.getArr?).toOption.getD #[] do
        let ts := natListOf s "ts_s"
        episodeResults := episodeResults.push <| Json.mkObj
          [ ("user", (s.getObjVal? "user").toOption.getD Json.null)
          , ("n_hops", Json.num (JsonNumber.fromNat ts.length))
          , ("lean_is_episode", Json.bool (decide (IsEpisodeAt mg ts))) ]

    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("params", params)
      , ("returns", Json.arr returnResults)
      , ("corridors", Json.arr corridorResults)
      , ("tours", Json.arr tourResults)
      , ("bursts", Json.arr burstResults)
      , ("longstays", Json.arr longstayResults)
      , ("assoc", Json.arr assocResults)
      , ("episodes", Json.arr episodeResults)
      , ("agreement", Json.mkObj
          [ ("n_checked", Json.num (JsonNumber.fromNat (nAgree + nDisagree)))
          , ("n_agree", Json.num (JsonNumber.fromNat nAgree))
          , ("n_disagree", Json.num (JsonNumber.fromNat nDisagree)) ]) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {returnResults.size} return, {corridorResults.size} corridor, {tourResults.size} tour, {burstResults.size} burst, {longstayResults.size} longstay, {assocResults.size} chi-square, {episodeResults.size} episode checks -> out/lean_results.json ({nDisagree} disagreements)"
    return (if nDisagree = 0 then 0 else 1)

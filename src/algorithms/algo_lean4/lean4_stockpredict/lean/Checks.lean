/-
Executable checks connecting the formal motif definitions to enclave data:
evaluates the DECIDABLE Lean predicates IsRun / HasShock / HasDrawdown /
IsVRecovery / IsVolBurst / IsTimelySession on observations exported from the
dbt marts, recomputes the chi-square association statistics EXACTLY over ℚ
from exported contingency cells (vs SQL's floating-point values), and records
the proof-status ledger of the formal layer.
-/
import EnclaveSeries.Motifs
import EnclaveSeries.Assoc
import EnclaveSeries.Sequences
import EnclaveSeries.SeriesShape
import EnclaveSeries.Params
import Lean.Data.Json

open Lean EnclaveSeries

def claim (name statement status : String) : Json :=
  Json.mkObj [("name", Json.str name), ("statement", Json.str statement),
              ("status", Json.str status)]

def claims : Json := Json.arr #[
  claim "clippedSum_sensitivity"
    "Adding/removing one ticker changes a clipped cell sum by ≤ clip (sensitivity = clip)" "proved",
  claim "privClippedCell_PureDP"
    "The per-cell release mechanism is pure (ε₁/ε₂)-DP (via SampCert privNoisedQueryPure_DP)" "proved",
  claim "cellQuery_sensitivity"
    "The per-cell clipped histogram query has sensitivity = clip (via exactBoundedSum + neighbour_map)" "proved",
  claim "histMech_DP"
    "The composed K-cell histogram mechanism is pure (K·ε₁/ε₂)-DP (privCompose/privPostProcess induction)" "proved",
  claim "sampleCellPMF_law"
    "The release sampler draws cell i with probability wᵢ/Σw exactly (via SampCert UniformSample_apply')" "proved",
  claim "isRun_replicate"
    "A constant above-band streak satisfies IsRun" "proved",
  claim "isRun_mono"
    "IsRun is monotone in the length threshold" "proved",
  claim "isShock_mono"
    "IsShock is monotone in the shock threshold" "proved",
  claim "hasDrawdown_pair"
    "The canonical peak-then-trough pair satisfies HasDrawdown" "proved",
  claim "hasDrawdown_mono"
    "HasDrawdown is monotone in the threshold on nonnegative prices" "proved",
  claim "isVRecovery_canonical"
    "The canonical trough-then-recovery window satisfies IsVRecovery" "proved",
  claim "isVRecovery_mono_window"
    "IsVRecovery is monotone in the recovery window" "proved",
  claim "isVolBurst_pair"
    "A ±g alternation with thr² ≤ 2g² satisfies IsVolBurst" "proved",
  claim "isVolBurst_mono"
    "IsVolBurst is monotone in the sd threshold" "proved",
  claim "classOfRet_flat"
    "A flat-band return is classified flat (sign/class coherence)" "proved",
  claim "genRet_class_contract"
    "Generator contract: magnitudes drawn from a class's public bins classify back to that class (genRet_class_*)" "proved",
  claim "mkDay_valid"
    "Generator contract: every day rendered through mkDay satisfies the class/magnitude validity predicate" "proved",
  claim "stepTimes_sorted"
    "Generator contract: positive gaps produce strictly increasing event times" "proved",
  claim "tradingDayNum_strictMono"
    "Generator contract: the weekday calendar maps trading-day indexes to strictly increasing dates" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "isTimelySession_mono"
    "Timely sessions are preserved under relaxing the gap bound" "proved",
  claim "isTimelySession_ordered"
    "A timely session is chronologically ordered" "proved",
  claim "sql_counts_correct"
    "The dbt SQL computed per-ticker counts/returns correctly from raw prices" "assumed",
  claim "entropy_source"
    "SampCert FFI randomness (/dev/urandom) is a fair entropy source" "assumed",
  claim "io_wrappers"
    "JSON I/O wrappers in the executables are correct" "assumed"]

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

def boolOf (j : Json) (k : String) : Bool :=
  ((j.getObjVal? k).bind Json.getBool?).toOption.getD false

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/motifs.json"
  match Json.parse txt with
  | .error e => IO.eprintln s!"checks: bad input: {e}"; return 1
  | .ok j =>
    let params := (j.getObjVal? "params").toOption.getD Json.null
    -- defaults come from the mirrored concept spec (Params.lean)
    let flatBp : Int := getNatD params "ret_flat_bp" Params.retFlatBp
    let shockBp : Int := getNatD params "shock_bp" Params.shockBp
    let runMin := getNatD params "run_min_days" Params.runMinDays
    let volMinBp := getNatD params "volburst_min_bp" Params.volburstMinBp
    let ddNum := getNatD params "drawdown_num" Params.drawdownNum
    let ddDen := getNatD params "drawdown_den" Params.drawdownDen
    let vdNum := getNatD params "vrecov_drop_num" Params.vrecovDropNum
    let vdDen := getNatD params "vrecov_drop_den" Params.vrecovDropDen
    let vlNum := getNatD params "vrecov_level_num" Params.vrecovLevelNum
    let vlDen := getNatD params "vrecov_level_den" Params.vrecovLevelDen
    let vWin := getNatD params "vrecov_max_days" Params.vrecovMaxDays
    let ddThr : ℚ := (ddNum : ℚ) / (ddDen : ℚ)
    let vDrop : ℚ := (vdNum : ℚ) / (vdDen : ℚ)
    let vLevel : ℚ := (vlNum : ℚ) / (vlDen : ℚ)

    -- run segments: list of returns + the SQL verdict
    let mut runResults : Array Json := #[]
    for b in ((j.getObjVal? "runs").bind Json.getArr?).toOption.getD #[] do
      let rets : List Int := intListOf b "rets_bp"
      let leanR := decide (IsRun runMin flatBp rets)
      let sqlR := boolOf b "sql_is_run"
      runResults := runResults.push <| Json.mkObj
        [ ("ticker", (b.getObjVal? "ticker").toOption.getD (Json.str "?"))
        , ("len", Json.num (JsonNumber.fromNat rets.length))
        , ("lean_is_run", Json.bool leanR)
        , ("sql_is_run", Json.bool sqlR)
        , ("agree", Json.bool (leanR == sqlR)) ]

    -- shock days
    let mut shockResults : Array Json := #[]
    for b in ((j.getObjVal? "shocks").bind Json.getArr?).toOption.getD #[] do
      let r : Int := ((b.getObjVal? "ret_bp").bind Json.getInt?).toOption.getD 0
      let leanS := decide (IsShock shockBp r)
      let sqlS := boolOf b "sql_is_shock"
      shockResults := shockResults.push <| Json.mkObj
        [ ("ticker", (b.getObjVal? "ticker").toOption.getD (Json.str "?"))
        , ("d", (b.getObjVal? "d").toOption.getD Json.null)
        , ("ret_bp", Json.num (JsonNumber.fromInt r))
        , ("lean_is_shock", Json.bool leanS)
        , ("sql_is_shock", Json.bool sqlS)
        , ("agree", Json.bool (leanS == sqlS)) ]

    -- drawdowns: full price series in micro units (scale-invariant predicate)
    let mut ddResults : Array Json := #[]
    for b in ((j.getObjVal? "drawdowns").bind Json.getArr?).toOption.getD #[] do
      let ps : List ℚ := (intListOf b "prices_micro").map (fun m => (m : ℚ))
      let leanD := decide (HasDrawdown ddThr ps)
      let sqlD := boolOf b "sql_has_drawdown"
      ddResults := ddResults.push <| Json.mkObj
        [ ("ticker", (b.getObjVal? "ticker").toOption.getD (Json.str "?"))
        , ("n_days", Json.num (JsonNumber.fromNat ps.length))
        , ("lean_has_drawdown", Json.bool leanD)
        , ("sql_has_drawdown", Json.bool sqlD)
        , ("agree", Json.bool (leanD == sqlD)) ]

    -- V-recoveries: peak + post-peak window in micro units
    let mut vrResults : Array Json := #[]
    for b in ((j.getObjVal? "vrecoveries").bind Json.getArr?).toOption.getD #[] do
      let peak : ℚ := (((b.getObjVal? "peak_micro").bind Json.getInt?).toOption.getD 0 : Int)
      let ws : List ℚ := (intListOf b "window_micro").map (fun m => (m : ℚ))
      let leanV := decide (IsVRecovery vDrop vLevel vWin peak ws)
      let sqlV := boolOf b "sql_is_vrecovery"
      vrResults := vrResults.push <| Json.mkObj
        [ ("ticker", (b.getObjVal? "ticker").toOption.getD (Json.str "?"))
        , ("window_len", Json.num (JsonNumber.fromNat ws.length))
        , ("lean_is_vrecovery", Json.bool leanV)
        , ("sql_is_vrecovery", Json.bool sqlV)
        , ("agree", Json.bool (leanV == sqlV)) ]

    -- volatility bursts: the rolling window's returns
    let mut vbResults : Array Json := #[]
    for b in ((j.getObjVal? "volbursts").bind Json.getArr?).toOption.getD #[] do
      let rets : List ℚ := (intListOf b "window_rets_bp").map (fun m => (m : ℚ))
      let leanB := decide (IsVolBurst (volMinBp : ℚ) rets)
      let sqlB := boolOf b "sql_is_volburst"
      vbResults := vbResults.push <| Json.mkObj
        [ ("ticker", (b.getObjVal? "ticker").toOption.getD (Json.str "?"))
        , ("d", (b.getObjVal? "d").toOption.getD Json.null)
        , ("lean_is_volburst", Json.bool leanB)
        , ("sql_is_volburst", Json.bool sqlB)
        , ("agree", Json.bool (leanB == sqlB)) ]

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
        assocResults := assocResults.push <| Json.mkObj
          [ ("feat_a", (pa.getObjVal? "feat_a").toOption.getD (Json.str "?"))
          , ("feat_b", (pa.getObjVal? "feat_b").toOption.getD (Json.str "?"))
          , ("lean_chi2", Json.str (toString leanF))
          , ("sql_chi2", Json.str (toString sqlF))
          , ("agree", Json.bool (Float.abs (leanF - sqlF) ≤ tol)) ]

    -- contiguous trading runs: IsTimelySession over day numbers, gap ≤ 4 days
    let mut sessionResults : Array Json := #[]
    if let some sj ← readJson? "data/sessions.json" then
      let mg := getNatD sj "max_gap_days" Params.gapMaxDays
      for s in ((sj.getObjVal? "sessions").bind Json.getArr?).toOption.getD #[] do
        let times : List ℚ := (intListOf s "day_nums").map (fun t => (t : ℚ))
        sessionResults := sessionResults.push <| Json.mkObj
          [ ("ticker", (s.getObjVal? "ticker").toOption.getD (Json.str "?"))
          , ("n_days", Json.num (JsonNumber.fromNat times.length))
          , ("lean_is_timely_session",
             Json.bool (decide (IsTimelySession (mg : ℚ) times))) ]

    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("params", params)
      , ("runs", Json.arr runResults)
      , ("shocks", Json.arr shockResults)
      , ("drawdowns", Json.arr ddResults)
      , ("vrecoveries", Json.arr vrResults)
      , ("volbursts", Json.arr vbResults)
      , ("assoc", Json.arr assocResults)
      , ("sessions", Json.arr sessionResults) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {runResults.size} run, {shockResults.size} shock, {ddResults.size} drawdown, {vrResults.size} v-recovery, {vbResults.size} vol-burst, {assocResults.size} chi-square, {sessionResults.size} session checks -> out/lean_results.json"
    return 0

/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → stock TSV.

    lake exe generate <genspec.json> <tickers> <days> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the SeriesShape constants that the linking theorems are about. Every random
draw runs the verified sampler (`sampleCellPMF_law`); each day is built
through `mkDay` (class/magnitude validity by construction, `mkDay_valid`;
bin→class contract `genRet_class_*`); dates follow the weekday calendar
(`tradingDayNum_strictMono`). Writes stock_synth.tsv (source TSV schema) and
plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.

Generative structure (all public priors declared derived_from_data: false):
  class chain   = DP transitions ⊙ DP dow-seasonality rows (elementwise,
                  +1 smoothing), market-coupled with probability λ from the
                  DP co-movement release (inter-stock collinearity);
  vol chain     = DP vol-state transitions (discretized GARCH diffusion);
  |log return|  = DP magnitude bins ∩ class-compatible bins, boosted by the
                  public vol-state table, uniform inside the bin;
  price path    = 100 · exp(Σ log returns)  (random walk in log space).
-/
import EnclaveSeries.GenSpec
import EnclaveSeries.SeriesShape
import EnclaveSeries.Params

open Lean EnclaveSeries EnclaveSeries.Gen EnclaveSeries.Shape

def classNames : List String := ["crash", "down", "flat", "up", "surge"]
def volNames : List String := ["calm", "normal", "elevated", "burst"]
def dowNames : List String := ["mon", "tue", "wed", "thu", "fri"]

def wOf (h : Hist) (name : String) : Nat :=
  match h.cells.find? (·.name = name) with
  | some c => c.w
  | none => 0

def pad2 (n : Nat) : String := if n < 10 then s!"0{n}" else s!"{n}"

def pad4 (n : Nat) : String :=
  if n < 10 then s!"000{n}"
  else if n < 100 then s!"00{n}"
  else if n < 1000 then s!"0{n}"
  else s!"{n}"

def dateStr (n : Nat) : String :=
  let (y, m, d) := civilFromDays (tradingDayNum n)
  s!"{y}-{pad2 m}-{pad2 d}"

/-- Price from cumulative log return (bp): 100 · exp(cum/10000), rendered
with 4 decimals (positive by construction; rounding guarded away from 0). -/
def priceStr (cumBp : Int) : String :=
  let p : Float := 100.0 * Float.exp (Float.ofInt cumBp / 10000.0)
  let micro : Nat := max ((p * 10000.0 + 0.5).toUInt64.toNat) 1
  s!"{micro / 10000}.{pad4 (micro % 10000)}"

def signOfClass (ci : Nat) : Int :=
  if ci ≤ 1 then -1 else if 3 ≤ ci then 1 else 0

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)
def ji (i : Int) : Json := Json.num (JsonNumber.fromInt i)

def uniformVol : Hist := ⟨volNames.map (fun n => ⟨n, 1⟩)⟩

def main (args : List String) : IO UInt32 := do
  let [specPath, tickersStr, daysStr, outDir] := args
    | do IO.eprintln "usage: generate <genspec.json> <tickers> <days> <out_dir>"; return 1
  let some nTickers := tickersStr.toNat?
    | do IO.eprintln "generate: tickers must be a natural"; return 1
  let some nDays := daysStr.toNat?
    | do IO.eprintln "generate: days must be a natural"; return 1
  if nTickers = 0 ∨ nTickers > 20 then
    IO.eprintln "generate: tickers must be in 1..20"; return 1
  if nDays < 30 ∨ nDays > 400 then
    IO.eprintln "generate: days must be in 30..400"; return 1
  let txt ← IO.FS.readFile specPath
  let spec ← match Json.parse txt >>= GenSpec.parse with
    | .error e => do IO.eprintln s!"generate: bad genspec: {e}"; return 1
    | .ok s => pure s

  let flat : Int := (Params.retFlatBp : Int)
  let big : Int := (Params.retBigBp : Int)
  let lam := min spec.comoveLambdaPct 100
  let plantStart := min 60 (nDays / 2)

  -- shared market mode chain (collinearity backbone; public persistence prior)
  let mut market : Array Int := #[1]
  for _ in [1:nDays] do
    let keep ← draw [marketKeepW, marketSwitchW]
    let prev := market.getD (market.size - 1) 1
    market := market.push (if keep = 0 then prev else -prev)

  let mut tsv : Array String := #["stockname\tdate-yyyy-mm-dd\tstockprice"]
  let mut planTickers : Array Json := #[]
  let mut planted : Array Json := #[]

  for i in [1:nTickers + 1] do
    let name := s!"SYN{pad2 i}"
    let mut cum : Int := 0
    let mut prevC := ""          -- unknown prev → marginal fallback (declared)
    let mut prevV := "normal"
    let mut classes : Array String := #[]
    let mut vols : Array String := #[]
    let mut rets : Array Json := #[]
    tsv := tsv.push s!"{name}\t{dateStr 0}\t100.0000"
    for t in [1:nDays] do
      -- class chain: DP transitions ⊙ DP dow rows (+1 smoothing), then
      -- market coupling with probability λ (mirror the sign to the market)
      let transRow := rowOf spec.transitions spec.classes prevC
      let dowRow := rowOf spec.dow spec.classes (dowNames.getD (dowOf t) "mon")
      let ws := classNames.map (fun n => (wOf transRow n + 1) * (wOf dowRow n + 1))
      let mut ci ← draw ws
      let couple ← draw [lam, 100 - lam]
      let mkt := market.getD t 1
      if couple = 0 ∧ signOfClass ci ≠ 0 ∧ signOfClass ci ≠ mkt then
        ci := 4 - ci
      -- vol chain: discretized GARCH diffusion over the public state grid
      let volRow := rowOf spec.volTrans uniformVol prevV
      let vCell ← volRow.choose
      let si := (volNames.findIdx? (· = vCell.name)).getD 1
      -- magnitude: DP bins ∩ class-compatible bins, vol-state boosted
      let allowed := classBins ci
      let boost := volBoost.getD si []
      let mws := allowed.map (fun b =>
        (wOf spec.magBins s!"mag_{b}" + 1) * (boost.getD b 0 + 1))
      let bi ← draw mws
      let b := allowed.getD bi 0
      let lo := magLoBp.getD b 0
      let hi := magHiBp.getD b 50
      let off ← draw (List.replicate (hi - lo + 1) 1)
      let sgn : Int ←
        if ci ≤ 1 then pure (-1)
        else if 3 ≤ ci then pure 1
        else do pure (if (← draw [1, 1]) = 0 then 1 else -1)
      let mut retBp : Int := sgn * Int.ofNat (lo + off)
      -- planted round-trip pattern: a down run with a shock day inside a
      -- drawdown, on the last synthetic ticker (plan.json ground truth)
      if i = nTickers ∧ plantStart ≤ t ∧ t < plantStart + 10 then
        retBp := if t = plantStart + 3 then -1300 else -250
      let day := mkDay flat big retBp si
      cum := cum + day.retBp
      prevC := classNames.getD day.cls "flat"
      prevV := vCell.name
      classes := classes.push prevC
      vols := vols.push prevV
      rets := rets.push (ji day.retBp)
      tsv := tsv.push s!"{name}\t{dateStr t}\t{priceStr cum}"
    planTickers := planTickers.push <| Json.mkObj
      [ ("ticker", Json.str name)
      , ("n_days", jn nDays)
      , ("classes", Json.arr (classes.map Json.str))
      , ("vol_states", Json.arr (vols.map Json.str))
      , ("rets_bp", Json.arr rets) ]
    if i = nTickers then
      planted := planted.push <| Json.mkObj
        [ ("ticker", Json.str name)
        , ("kind", Json.str "shock_in_drawdown_down_run")
        , ("start_idx", jn plantStart)
        , ("shock_idx", jn (plantStart + 3))
        , ("run_days", jn 10)
        , ("shock_bp", ji (-1300))
        , ("step_bp", ji (-250)) ]

  IO.FS.createDirAll outDir
  IO.FS.writeFile s!"{outDir}/stock_synth.tsv"
    (String.intercalate "\n" tsv.toList ++ "\n")
  let plan := Json.mkObj
    [ ("tickers", Json.arr planTickers)
    , ("planted", Json.arr planted)
    , ("market_signs", Json.arr (market.map ji))
    , ("comove_lambda_pct", jn lam)
    , ("n_tickers", jn nTickers)
    , ("n_days", jn nDays)
    , ("anchor", Json.str (dateStr 0))
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {nTickers} tickers x {nDays} days -> {outDir}/stock_synth.tsv"
  return 0

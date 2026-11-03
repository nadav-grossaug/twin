/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → travel TSV.

    lake exe generate <genspec.json> <n_users> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the TravelShape constants (public centroids, calendar, region reps). Every
random draw runs the verified sampler (`sampleCellPMF_law`); each user's
rows are built through `mkHops`/`hopsWithin`/`capStays`/`mkUserHistory`, so
the source invariants hold by construction and by theorem:
  per-user timestamps strictly increasing   (mkUserHistory_times_sorted),
  prev_country ≠ next_country on every row  (mkUserHistory_prev_ne_next),
  durations ≥ 0                             (Nat, by type),
  next_duration(i) = prev_duration(i+1)     (mkUserHistory_durChained),
  timestamps in [window_start, window_end)  (mkUserHistory_ts_in_window +
                                             hopsWithin_sum_lt + capStays_gaps),
  no overlapping presence: ts_i + next_duration_i ≤ ts_{i+1}, last stay ends
  before window_end                         (mkUserHistory_no_overlap, stays
                                             capped by capStays).
Writes travel_synth.tsv (source TSV schema, 12 columns + header) and
plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.

Generative structure (public priors declared derived_from_data: false):
  travels/user  = DP travel_count bins, uniform inside the public bin range;
  return-first  = per user, a return probability p = midpoint of the DP
                  return_rate band; each hop after the first goes BACK to
                  the country the user came from with probability p
                  (sampleCellPMF over [p, 100-p]), region implied;
  region walk   = otherwise (new hop) DP region transitions (rows), start
                  from the DP region mix;
  countries     = the released country-popularity list of the destination
                  region, weighted by released arrivals w_in, excluding the
                  current country; fallback: any released country of the
                  region, then the public region rep. Departures need no
                  separate draw: the chain makes every arrival the next
                  departure, which with returns + w_in weighting reproduces
                  the released in/out near-balance;
  times         = day-level scaffold (uniform start day + cumulative gap
                  days from the DP gap classes); each transfer's final ts =
                  day start + (DP hod hour)×3600 + grid-uniform sub-hour
                  offset in [0, 3600); a normalization step forces
                  ts_{i+1} ≥ ts_i + 1 (gap := max 1 (ts_raw − ts_prev));
  stays         = DP duration classes, grid-uniform inside the public range,
                  then capped at the gap to the next transfer (capStays) so
                  a user is never in two countries at once;
  coordinates   = public country centroids; venue ids = counter in 24-hex.
PLANTED ground truth: user 1 = commuter corridor (6 repeats of corridor[0],
constant gaps, CV = 0); user 2 = tour (3 hops, 2d apart, 3+ countries).
Planted users obey the same invariants: their whole-day gaps keep the drawn
start time-of-day (hod + sub-hour) on every transfer, and capStays caps
their stays like everyone else's (planted stays already fit, so the plan
shapes are unchanged).
-/
import EnclaveTravel.GenSpec
import EnclaveTravel.TravelShape
import EnclaveTravel.Params

open Lean EnclaveTravel EnclaveTravel.Gen EnclaveTravel.Shape

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)

def tsvHeader : String :=
  "userid\tdt\tprev_country\tprev_last_venue_id\tprev_last_lat\tprev_last_long\tprev_duration\tnext_country\tnext_first_venue_id\tnext_first_lat\tnext_first_long\tnext_duration"

def rangeOf (rs : List (String × RangeN)) (k : String) (dflt : RangeN) : RangeN :=
  ((rs.find? (·.1 == k)).map (·.2)).getD dflt

/-- Grid-uniform draw in [lo, hi] via the verified sampler: uniform over at
most 4097 evenly spaced grid points (declared "uniform-ish within the
released range"; exact uniform for spans ≤ 4096). -/
def drawRange (r : RangeN) : IO Nat := do
  let span := r.hi - r.lo
  if span = 0 then
    pure r.lo
  else
    let buckets := min span 4096
    let i ← draw (List.replicate (buckets + 1) 1)
    pure (r.lo + span * i / buckets)

/-- Midpoint return probability (percent) of a released return-rate band
(declared rule: band → its midpoint). -/
def bandMidpoint (name : String) : Nat :=
  if name = "rr_0_19" then 10
  else if name = "rr_20_39" then 30
  else if name = "rr_40_59" then 50
  else if name = "rr_60_79" then 70
  else if name = "rr_80_100" then 90
  else 50

/-- The released country-popularity entries of one region. -/
def countriesIn (cl : List Country) (r : String) : List Country :=
  cl.filter (fun c => c.region == r)

/-- Draw a destination country in region `r` from the released
country-popularity list, weighted by released arrivals (w_in), EXCLUDING
the current country `cur`; fallback: any released country of the region,
then the public region representative (declared prior). -/
def drawCountryIn (cl : List Country) (r cur : String) : IO String := do
  let inR := countriesIn cl r
  let pool := inR.filter (fun c => c.c != cur)
  let pool := if pool.isEmpty then inR else pool
  if pool.isEmpty then
    pure (regionRep r)
  else
    let i ← draw (pool.map (·.wIn))
    pure ((pool.getD i ⟨regionRep r, r, 0, 0⟩).c)

/-- Time-of-day in [0, 86400): the hour from the DP hod histogram (cells
h00..h23, index = hour) + a grid-uniform sub-hour offset in [0, 3600)
(declared public prior). -/
def drawTimeOfDay (hod : Hist) : IO Nat := do
  let i ← draw hod.weights
  let sub ← drawRange ⟨0, 3599⟩
  pure (min i 23 * 3600 + sub)

/-- Deterministic venue id from the counter: scrambled by a fixed odd
constant (invertible mod 2⁹⁶, so ids stay distinct) to render as plausible
24-hex ids — a labelling device, carries no data. -/
def venueId (n : Nat) : String :=
  hex24 ((n + 1) * 0x9e3779b97f4a7c15f39cc0605cedc835 % (2 ^ 96))

/-- Planted commuter-corridor plan: alternate a→b / b→a with constant gaps
(11 rows ⇒ 6 a→b repeats; inter-repeat gaps constant ⇒ CV = 0). -/
def corridorTriples (cb ca : String) : List (String × Nat × Nat) :=
  (List.range 11).map (fun i =>
    (if i % 2 == 0 then cb else ca, 259200, 86400))

/-- Planted tour plan: 3 hops, 2 days apart, across 3 distinct countries. -/
def tourTriples : List (String × Nat × Nat) :=
  [("US", 172800, 86400), ("FR", 172800, 86400), ("JP", 172800, 86400)]

def main (args : List String) : IO UInt32 := do
  let [specPath, usersStr, outDir] := args
    | do IO.eprintln "usage: generate <genspec.json> <n_users> <out_dir>"; return 1
  let some nUsers := usersStr.toNat?
    | do IO.eprintln "generate: n_users must be a natural"; return 1
  if nUsers < 3 ∨ nUsers > 200000 then
    IO.eprintln "generate: n_users must be in 3..200000"; return 1
  let txt ← IO.FS.readFile specPath
  let spec ← match Json.parse txt >>= GenSpec.parse with
    | .error e => do IO.eprintln s!"generate: bad genspec: {e}"; return 1
    | .ok s => pure s
  if spec.windowEndS ≤ spec.windowStartS then
    IO.eprintln "generate: genspec window is empty"; return 1

  -- planted corridor pair = top released corridor (public), with a fixed
  -- public fallback if the spec carries none
  let (ca, cb) := match spec.corridors.head? with
    | some c => if c.a == c.b then ("MY", "SG") else (c.a, c.b)
    | none => ("MY", "SG")

  let mut tsv : Array String := #[tsvHeader]
  let mut userCounts : Array Json := #[]
  let mut venueCtr : Nat := 0
  let mut planCorridor := Json.null
  let mut planTour := Json.null

  for u in [1:nUsers + 1] do
    -- plan: (start country, start duration, start time, hop triples)
    let (start, startDur, t0, triples) ←
      if u = 1 then do
        -- planted corridor: whole-day gaps keep the drawn time-of-day on
        -- every transfer; gaps stay constant (CV = 0)
        let tod ← drawTimeOfDay spec.hod
        pure (ca, 86400, spec.windowStartS + tod, corridorTriples cb ca)
      else if u = 2 then do
        let tod ← drawTimeOfDay spec.hod
        pure ("GB", 86400, spec.windowStartS + tod, tourTriples)
      else do
        -- travels per user: DP tc bin -> uniform inside the public bin range
        let tcCell ← spec.travelCount.choose
        let nRaw ← drawRange (rangeOf spec.tcBinRanges tcCell.name ⟨1, 1⟩)
        let nTravels := max 1 (min nRaw 200)
        -- per-user return probability: midpoint of the DP return-rate band
        let rrCell ← spec.returnRate.choose
        let p := bandMidpoint rrCell.name
        -- start region from the DP region mix; start country from the
        -- released country-popularity list (w_in weights)
        let rCell ← spec.regionMix.choose
        let startRegion := if rCell.name = "" then "EU" else rCell.name
        let startC ← drawCountryIn spec.countries startRegion ""
        -- day-level scaffold: uniform start day + DP hod time-of-day
        let offD ← draw (List.replicate 181 1)
        let tod0 ← drawTimeOfDay spec.hod
        let t0 := min (spec.windowStartS + offD * 86400 + tod0)
          (spec.windowEndS - 1)
        let mut day := offD
        let mut prevTs := t0
        let mut cur := startC
        let mut curRegion := startRegion
        let mut cameFrom := ""
        let mut cameFromRegion := startRegion
        let mut triples : Array (String × Nat × Nat) := #[]
        for _ in [0:nTravels] do
          -- RETURN-FIRST: with probability p go back where the user came
          -- from (region implied); otherwise walk the DP region transitions
          -- and draw the destination from the released w_in weights
          let retIdx ← draw [p, 100 - p]
          let (destC, destRegion) ←
            if retIdx == 0 && cameFrom != "" && cameFrom != cur then
              pure (cameFrom, cameFromRegion)
            else do
              let row := rowOf spec.transitions spec.regionMix curRegion
              let nCell ← row.choose
              let nextRegion := if nCell.name = "" then curRegion else nCell.name
              let c ← drawCountryIn spec.countries nextRegion cur
              pure (c, nextRegion)
          -- ≠-forced destination, exactly as the renderer will force it
          -- (pickNe is idempotent, so mkHops re-applies it as a no-op)
          let dest := pickNe cur destC
          -- gap: DP gap class -> whole days onto the scaffold; final ts =
          -- day start + hod hour + sub-hour; normalization keeps ts strict
          let gCell ← spec.gapClass.choose
          let gapS ← drawRange (rangeOf spec.gapRangesS gCell.name ⟨3600, 172800⟩)
          day := day + gapS / 86400
          let tod ← drawTimeOfDay spec.hod
          let tsRaw := spec.windowStartS + day * 86400 + tod
          let gap := max 1 (tsRaw - prevTs)
          prevTs := prevTs + gap
          -- stay from the DP duration classes (capStays caps it later)
          let dCell ← spec.durationClass.choose
          let stay ← drawRange (rangeOf spec.durRangesS dCell.name ⟨3600, 86400⟩)
          triples := triples.push (dest, gap, stay)
          cameFrom := cur
          cameFromRegion := curRegion
          cur := dest
          curRegion := destRegion
        let sdCell ← spec.durationClass.choose
        let startDur ← drawRange (rangeOf spec.durRangesS sdCell.name ⟨3600, 86400⟩)
        pure (startC, startDur, t0, triples.toList)
    -- smart constructors: ≠-forced destinations, positive gaps, window
    -- clamp, stays capped at the gap to the next transfer (hypotheses of
    -- the rendering theorems hold by construction; capStays preserves the
    -- gaps/countries the other theorems depend on)
    let hops := capStays spec.windowEndS t0
      (hopsWithin spec.windowEndS t0 (mkHops start triples))
    let rows := mkUserHistory start startDur t0 hops
    for r in rows do
      let pv := venueId venueCtr
      let nv := venueId (venueCtr + 1)
      venueCtr := venueCtr + 2
      let (plat, plon) := centroidOf r.prevCountry
      let (nlat, nlon) := centroidOf r.nextCountry
      tsv := tsv.push <| s!"{u}\t{dtStr r.ts}\t{r.prevCountry}\t{pv}\t" ++
        s!"{coordStr plat}\t{coordStr plon}\t{r.prevDuration}\t" ++
        s!"{r.nextCountry}\t{nv}\t{coordStr nlat}\t{coordStr nlon}\t{r.nextDuration}"
    userCounts := userCounts.push <| Json.mkObj
      [("user", jn u), ("n_travels", jn rows.length)]
    if u = 1 then
      let abTs := (rows.filter (fun r => r.nextCountry == cb)).map (·.ts)
      let gaps := (abTs.zip abTs.tail).map (fun p => p.2 - p.1)
      planCorridor := Json.mkObj
        [ ("user", jn 1), ("a", Json.str ca), ("b", Json.str cb)
        , ("n_repeats", jn abTs.length)
        , ("gaps_s", Json.arr (gaps.map jn).toArray)
        , ("ts_s", Json.arr (abTs.map jn).toArray) ]
    if u = 2 then
      planTour := Json.mkObj
        [ ("user", jn 2), ("start", Json.str "GB")
        , ("countries", Json.arr (rows.map (fun r => Json.str r.nextCountry)).toArray)
        , ("ts_s", Json.arr (rows.map (fun r => jn r.ts)).toArray) ]

  IO.FS.createDirAll outDir
  IO.FS.writeFile s!"{outDir}/travel_synth.tsv"
    (String.intercalate "\n" tsv.toList ++ "\n")
  let plan := Json.mkObj
    [ ("corridor", planCorridor)
    , ("tour", planTour)
    , ("users", Json.arr userCounts)
    , ("n_users", jn nUsers)
    , ("spec_version", Json.str spec.version)
    , ("window", Json.mkObj [("start_s", jn spec.windowStartS),
                             ("end_s", jn spec.windowEndS)])
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {nUsers} users, {tsv.size - 1} transfers -> {outDir}/travel_synth.tsv"
  return 0

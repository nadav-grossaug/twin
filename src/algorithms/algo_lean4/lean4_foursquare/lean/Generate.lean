/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → travel TSV.

    lake exe generate <genspec.json> <n_users> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the TravelShape constants (public centroids, calendar, region reps). Every
random draw runs the verified sampler (`sampleCellPMF_law`); each user's
rows are built through `mkHops`/`hopsWithin`/`mkUserHistory`, so the source
invariants hold by construction and by theorem:
  per-user timestamps strictly increasing   (mkUserHistory_times_sorted),
  prev_country ≠ next_country on every row  (mkUserHistory_prev_ne_next),
  durations ≥ 0                             (Nat, by type),
  next_duration(i) = prev_duration(i+1)     (mkUserHistory_durChained),
  timestamps in [window_start, window_end)  (mkUserHistory_ts_in_window +
                                             hopsWithin_sum_lt).
Writes travel_synth.tsv (source TSV schema, 12 columns + header) and
plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.

Generative structure (public priors declared derived_from_data: false):
  travels/user  = DP travel_count bins, uniform inside the public bin range;
  region walk   = DP region transitions (rows), start from DP region mix;
  countries     = released corridors filtered by region pair, else the
                  region's corridor country, else the public region rep;
  gaps          = DP gap classes, grid-uniform inside the public range;
  stays         = DP duration classes, grid-uniform inside the public range;
  coordinates   = public country centroids; venue ids = counter in 24-hex.
PLANTED ground truth: user 1 = commuter corridor (6 repeats of corridor[0],
constant gaps, CV = 0); user 2 = tour (3 hops, 2d apart, 3+ countries).
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

/-- Destination country for a region hop from the released corridors. -/
def corridorNext (cs : List Corridor) (ra rb : String) : Option String :=
  (cs.find? (fun c => c.regionA == ra && c.regionB == rb)).map (·.b)

/-- A country inside region `r`: first corridor country in that region, else
the public region representative (declared prior). -/
def countryInRegion (cs : List Corridor) (r : String) : String :=
  match cs.find? (fun c => c.regionA == r) with
  | some c => c.a
  | none =>
    match cs.find? (fun c => c.regionB == r) with
    | some c => c.b
    | none => regionRep r

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
      if u = 1 then
        pure (ca, 86400, spec.windowStartS, corridorTriples cb ca)
      else if u = 2 then
        pure ("GB", 86400, spec.windowStartS, tourTriples)
      else do
        -- travels per user: DP tc bin -> uniform inside the public bin range
        let tcCell ← spec.travelCount.choose
        let nRaw ← drawRange (rangeOf spec.tcBinRanges tcCell.name ⟨1, 1⟩)
        let nTravels := max 1 (min nRaw 200)
        -- start region from the DP region mix
        let rCell ← spec.regionMix.choose
        let startRegion := if rCell.name = "" then "EU" else rCell.name
        let startC := countryInRegion spec.corridors startRegion
        let mut prevRegion := startRegion
        let mut triples : Array (String × Nat × Nat) := #[]
        for _ in [0:nTravels] do
          -- region walk via the DP transition rows (marginal fallback)
          let row := rowOf spec.transitions spec.regionMix prevRegion
          let nCell ← row.choose
          let nextRegion := if nCell.name = "" then prevRegion else nCell.name
          let cand := (corridorNext spec.corridors prevRegion nextRegion).getD
            (countryInRegion spec.corridors nextRegion)
          -- gap and stay from the DP class mixes + public ranges
          let gCell ← spec.gapClass.choose
          let gap ← drawRange (rangeOf spec.gapRangesS gCell.name ⟨3600, 172800⟩)
          let dCell ← spec.durationClass.choose
          let stay ← drawRange (rangeOf spec.durRangesS dCell.name ⟨3600, 86400⟩)
          triples := triples.push (cand, gap, stay)
          prevRegion := nextRegion
        -- start time: uniform day offset into the window (public prior)
        let offD ← draw (List.replicate 181 1)
        let t0 := min (spec.windowStartS + offD * 86400) (spec.windowEndS - 1)
        let sdCell ← spec.durationClass.choose
        let startDur ← drawRange (rangeOf spec.durRangesS sdCell.name ⟨3600, 86400⟩)
        pure (startC, startDur, t0, triples.toList)
    -- smart constructors: ≠-forced destinations, positive gaps, window clamp
    -- (hypotheses of the rendering theorems hold by construction)
    let hops := hopsWithin spec.windowEndS t0 (mkHops start triples)
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

/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → booking CSV.

    lake exe generate <genspec.json> <n_users> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the TripShape constants (calendar, month anchors, synthetic city ids). Every
random draw runs the verified sampler (`sampleCellPMF_law`); each user's
rows are built through `mkLeg`/`legsWithin`/`mkLegs`, so the source
invariants hold by construction and by theorem:
  per-trip checkins strictly increasing    (mkLegs_checkins_sorted,
                                            nights ≥ 1 by mkLeg_nights_range),
  checkout_i = checkin_i + nights_i        (structural, by mkLegs),
  checkin_{i+1} = checkout_i + gap_i       (structural, by mkLegs),
  gaps in [0, 33], nights in [1, 30]       (mkLeg_gap_range/mkLeg_nights_range),
  zero-gap plans render the exact contiguous flow — next checkin = previous
  checkout, the dataset's 99.1% flow         (mkLegs_contiguous + ContigGaps),
  checkins in the public window, no stay past the window end and no
  overlapping presence                      (mkLegs_in_window /
                                            mkLegs_no_overlap + legsWithin_span_le,
                                            driven via legsWithin).
Writes booking_synth.csv (source CSV schema, 9 columns + header) and
plan.json (the latent plan for the round-trip check). No PyTorch, no
Python, no neural weights.

Generative structure (public priors declared derived_from_data: false):
  reservations/user = DP res_count bins (per-USER total), DECAY-weighted
                     inside the public bin range (within_bin_decay: real
                     within-bin mass decays ~0.7/step; spec 1.2.0);
  trips/user       = DP trips_per_user cells tp_1/tp_2/tp_3p (spec 1.3.0;
                     real 91.1/7.4/1.5%); legs split near-equally
                     (measured: both trips >= 4 legs in 99.6% of 2-trip
                     users); inter-trip gap uniform [1, 94] (median-
                     matched to the measured 47 days); utrip '<user>_<k>';
  start month     = DP month-of-WINDOW cells (spec 1.2.0: the real
                    observation ramp, not a flattened month-of-year
                    marginal), day-of-month uniform in 1..28 (uniform_dow
                    prior: dow is NOT released — measured near-uniform);
  booker country  = DP booker cells, drawn once per user (one cell per user);
  country walk    = DP country_mix start + DP country_transition rows with
                    the return_boost diagonal prior (row r: diagonal
                    += max(1, row sum)); country-tier -> city via the
                    released city-popularity list weighted by w, excluding
                    EVERY city the trip already visited (spec 1.1.0), so
                    the repeat-city motif comes only from the drawn repeat
                    structure; synthetic id fallback;
  repeat          = per trip, a repeat decision at the DP return_rate band
                    midpoint (38.1% of real trips revisit a city); GIVEN a
                    repeat trip, the anatomy comes from the DP repeat_struct
                    cells (spec 1.2.0): a CHAIN of consecutive same-city
                    stay-extension hops at the current city (real: 9.01%
                    of legs are same-city-consecutive; the pre-1.2.0 single
                    return hop rendered 1.66%) and/or hops back to the
                    previous city;
  nights / gaps   = DP nights / gap cells, DECAY-weighted inside the public
                    ranges (within_bin_decay: ~0.45 / ~0.5 per step),
                    clamped by mkLeg;
  device/affiliate = per-trip CLASS PLANS from the DP device_persist /
                    affiliate_persist cells (spec 1.3.0): k distinct
                    classes over drawn segments — real trips are 70.7%
                    single-device and keep the prior affiliate tier on
                    75.4% of legs; the per-leg marginals hide this.
PLANTED ground truth: user 1 = repeat-city trip (A,B,A,B); user 2 = 4-leg
3-country contiguous tour; user 3 = trip containing an 8-night long stay.
Planted users obey the same invariants as everyone else.
-/
import EnclaveTrip.GenSpec
import EnclaveTrip.TripShape
import EnclaveTrip.Params
import Lean.Data.Json

open Lean EnclaveTrip EnclaveTrip.Gen EnclaveTrip.Shape

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)

def csvHeader : String :=
  "user_id,checkin,checkout,city_id,device_class,affiliate_id,booker_country,hotel_country,utrip_id"

/-- RFC 4180 quoting for one CSV field: a cell is wrapped in double quotes
(with inner quotes doubled) exactly when it contains a comma, a quote or a
newline — the anonymized country names include values like
"Merania, Kingdom of". -/
def csvCell (s : String) : String :=
  if s.any (fun c => c == ',' || c == '"' || c == '\n') then
    "\"" ++ s.replace "\"" "\"\"" ++ "\""
  else
    s

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

/-- Integer sub-bin weights decaying geometrically at ratio num/den per step
(declared structural prior `within_bin_decay`, spec 1.2.0: measured real
within-bin mass decays toward the upper edge — per-step ratios rc ~0.7,
nights ~0.45, gaps ~0.5 — e.g. rc_9_12 real shares 9:10:11:12 =
2.67:1.63:1.03:0.66%). -/
def decayWeights (span num den : Nat) : List Nat :=
  (List.range span).map (fun i => max 1 (4096 * num ^ i / den ^ i))

/-- Decay-weighted draw in [lo, hi]: unit sub-bins weighted by
`decayWeights`, still through the verified sampler. -/
def drawRangeDecay (r : RangeN) (num den : Nat) : IO Nat := do
  let span := r.hi - r.lo
  if span = 0 then
    pure r.lo
  else
    let i ← draw (decayWeights span num den)
    pure (r.lo + i)

/-- The per-family decay ratios of the declared within_bin_decay prior. -/
def rcDecay : Nat × Nat := (7, 10)      -- measured ~0.7 per step
def nightsDecay : Nat × Nat := (9, 20)  -- measured ~0.45
def gapDecay : Nat × Nat := (1, 2)      -- measured ~0.5

/-- Midpoint return probability (percent) of a released return-rate band
(declared rule: band → its midpoint). -/
def bandMidpoint (name : String) : Nat :=
  if name = "rr_0_19" then 10
  else if name = "rr_20_39" then 30
  else if name = "rr_40_59" then 50
  else if name = "rr_60_79" then 70
  else if name = "rr_80_100" then 90
  else 50

/-- Month number of a released month cell (default July on malformed). -/
def monthNum (name : String) : Nat :=
  match name with
  | "m01" => 1 | "m02" => 2 | "m03" => 3 | "m04" => 4 | "m05" => 5 | "m06" => 6
  | "m07" => 7 | "m08" => 8 | "m09" => 9 | "m10" => 10 | "m11" => 11 | "m12" => 12
  | _ => 7

/-- Month-of-window index of a released mw cell (mw00 = 2015-12, the
window start; default mid-window). -/
def mwNum (name : String) : Nat :=
  match name with
  | "mw00" => 0 | "mw01" => 1 | "mw02" => 2 | "mw03" => 3 | "mw04" => 4
  | "mw05" => 5 | "mw06" => 6 | "mw07" => 7 | "mw08" => 8 | "mw09" => 9
  | "mw10" => 10 | "mw11" => 11 | "mw12" => 12 | "mw13" => 13 | "mw14" => 14
  | _ => 7

/-- Repeat-trip anatomy from the released structure cells (v1.2.0):
(ext chain length, ret hop count). r_ext* renders a CHAIN of k
consecutive same-city stay-extension hops at the current city (real
repeat trips are dominated by A,A,A patterns — 9.01% of real legs are
same-city-consecutive); r_ret* renders k hops back to the previous city;
r_both renders one of each (measured both-anatomy: 1.44 consec / 1.10
revisit). Unknown cells degrade to a single return hop. -/
def repeatAnatomy (name : String) : IO (Nat × Nat) := do
  if name = "r_ext1" then pure (1, 0)
  else if name = "r_ext2" then pure (2, 0)
  else if name = "r_ext3p" then do
    let k ← drawRange ⟨3, 5⟩
    pure (k, 0)
  else if name = "r_ret1" then pure (0, 1)
  else if name = "r_ret2p" then do
    let k ← drawRange ⟨2, 3⟩
    pure (0, k)
  else if name = "r_both" then pure (1, 1)
  else pure (0, 1)

/-- `k` distinct hop positions drawn uniformly from [1, hi], EXCLUDING the
extension-chain window [extPos, extPos + extK) (positions of the return
hops; fewer than k if the trip is short). All draws through the verified
sampler. -/
def drawPositionsFiltered (hi k extPos extK : Nat) : IO (List Nat) := do
  let free := ((List.range hi).map (· + 1)).filter
    (fun i => !(extPos ≤ i && i < extPos + extK))
  let mut out : List Nat := []
  let mut avail := free
  for _ in [0:k] do
    if !avail.isEmpty then
      let i ← draw (avail.map (fun _ => 1))
      out := (avail.getD i 0) :: out
      avail := avail.erase (avail.getD i 0)
  pure out

/-- The per-trip CLASS PLAN (spec 1.3.0): `k` distinct classes drawn from a
released histogram (device classes or affiliate tiers), laid over `legs`
legs in `k` segments whose lengths come from drawn ascending cut points —
rendering the strong within-trip channel persistence the per-leg marginals
hide (70.7% of real trips are single-device; 75.4% of legs keep the
previous leg's affiliate tier). -/
def classPlan (h : Hist) (k legs : Nat) : IO (List String) := do
  if k ≤ 1 || legs ≤ 1 then
    let c ← h.choose
    pure (List.replicate (max 1 legs) c.name)
  else
    -- k distinct classes (bounded re-draw on collision; accept a dup only
    -- if the histogram is exhausted)
    let mut picked : List String := []
    for _ in [0:k] do
      let mut c ← h.choose
      for _ in [0:16] do
        if picked.contains c.name then
          c ← h.choose
      picked := picked ++ [c.name]
    -- ascending cut points: cut_i in [prev+1, legs-(k-1-i)], so all k
    -- segments have length >= 1 and sum to legs
    let mut cuts : List Nat := []
    let mut lo := 1
    for i in [0:k - 1] do
      let hi := legs - (k - 1 - i)
      let cpt ← drawRange ⟨lo, hi⟩
      cuts := cuts ++ [cpt]
      lo := cpt + 1
    -- expand segments into the per-leg plan
    let mut plan : List String := []
    let mut prev := 0
    let mut idx := 0
    for cpt in cuts do
      for _ in [prev:cpt] do
        plan := plan ++ [picked.getD idx ""]
      prev := cpt
      idx := idx + 1
    for _ in [prev:legs] do
      plan := plan ++ [picked.getD idx ""]
    pure plan

/-- The tier of a planted destination's country label (the 15 named tiers
are single-country by the public seed; the suppressed tail is the
"Otherland" placeholder). -/
def tierOfCountry (co : String) : String :=
  if co = "Otherland" then "ct_other" else co

/-- The country label of a tier (declared placeholder for the suppressed
tail: the 180-country remainder of ct_other collapses to "Otherland"). -/
def countryOfTier (ct : String) : String :=
  if ct = "ct_other" then "Otherland" else ct

/-- Synthetic affiliate id for the suppressed tail: the aff_other tier
covers ~3,600 real affiliate ids whose individual identities are NOT
released; the generator renders ONE synthetic id per (user, plan-segment)
— stable within a booking segment (real within-tail legs repeat the same
id on 68.7% of consecutive pairs) and distinct across segments and users
(declared labelling prior; carries no data). -/
def affIdSynth (n : Nat) : Nat := 100000 + (n * 2654435761) % 90000

/-- Affiliate id of a plan cell: a9924-style tiers carry their released
id; "s<id>" is the per-segment synthetic tail id (affIdSynth); a bare
aff_other falls back to 359 (a released cell). -/
def affIdOf (name : String) : Nat :=
  if name = "aff_other" then 359
  else if name.startsWith "s" then
    match (name.drop 1).toNat? with
      | some i => i
      | none => 359
  else match (name.drop 1).toNat? with
    | some i => i
    | none => 359

/-- The released city-popularity entries of one tier. -/
def citiesIn (cl : List City) (ct : String) : List City :=
  cl.filter (fun c => c.ct == ct)

/-- Draw a destination (city id, hotel country) in tier `ct` from the
released city-popularity list, weighted by released popularity `w`,
EXCLUDING every city the trip has already VISITED (spec 1.1.0: with only
the current city excluded, incidental redraws of released popular cities
pushed the trip-level repeat-city share to 0.455 vs the released ~0.402 —
the repeat motif is now carried ONLY by the drawn return hop); fallback: a
synthetic id with the tier's country label (declared prior). -/
def drawCityIn (cl : List City) (ct : String) (visited : List Nat)
    (synthId : Nat) : IO (Nat × String) := do
  let inCt := citiesIn cl ct
  let pool := inCt.filter (fun c => !visited.contains c.cityId)
  if pool.isEmpty then
    pure (synthId, countryOfTier ct)
  else
    let i ← draw (pool.map (·.w))
    let c := pool.getD i ⟨synthId, countryOfTier ct, ct, 0⟩
    pure (c.cityId, c.country)

/-- Planted repeat-city nights: A,B,A,B with nights 2,2,1,1 (A revisited —
the core return motif). -/
def plantedReturnNights : List Nat := [2, 2, 1, 1]

/-- Planted tour nights: 4 one-night legs over 3 distinct countries. -/
def plantedTourNights : List Nat := [1, 1, 1, 1]

/-- Planted long-stay nights: an 8-night stay followed by three short legs. -/
def plantedLongstayNights : List Nat := [8, 1, 1, 1]

/-- The distinct-country prefix of the released city list: at most `n`
cities of pairwise-distinct countries (for the planted motifs). -/
def distinctCountries : List City → Nat → List City
  | _, 0 => []
  | [], _ => []
  | c :: rest, n + 1 =>
      c :: distinctCountries (rest.filter (fun x => x.country != c.country)) n

/-- Pick `n` city/country pairs for planted motifs, preferring released
cities of distinct countries; synthetic fallback with tier labels. -/
def plantedPicks (cl : List City) (n : Nat) (synth : Nat → Nat) : List (Nat × String) :=
  let picked := (distinctCountries cl n).map (fun c => (c.cityId, c.country))
  if picked.length ≥ n then picked
  else
    (List.range n).map (fun i =>
      (synth (i + 1), countryOfTier (Params.ctCells.getD i "ct_other")))

/-- Build the planted leg list from city/country picks, a night pattern and
a tier-country pattern, drawing device/affiliate per leg (gaps 0: the
contiguous flow). -/
def plantedLegs (spec : GenSpec) (picks : List (Nat × String))
    (tiers : List Nat) (nightsPat : List Nat) : IO (List Leg) := do
  let n := tiers.length
  let mut ls : Array Leg := #[]
  for i in [0:n] do
    let pc := picks.getD (tiers.getD i 0)
      (0, countryOfTier (Params.ctCells.getD 0 "ct_other"))
    let dvc ← spec.device.choose
    let aff ← spec.affiliate.choose
    ls := ls.push (mkLeg pc.1 pc.2 (tierOfCountry pc.2) dvc.name aff.name 0
      (nightsPat.getD i 1))
  pure ls.toList

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
  if spec.windowEndD ≤ spec.windowStartD then
    IO.eprintln "generate: genspec window is empty"; return 1

  let mut tsv : Array String := #[csvHeader]
  let mut userPlan : Array Json := #[]
  let mut cityCtr : Nat := 0
  let mut planReturn := Json.null
  let mut planTour := Json.null
  let mut planLongstay := Json.null

  for u in [1:nUsers + 1] do
    -- per-user draws (planted users draw theirs too: they follow the same
    -- release, only the trip shape is fixed)
    let bookerCell ← spec.booker.choose
    let booker := if bookerCell.name = "" then "Gondal" else bookerCell.name

    -- plan: (start day, leg list) PER TRIP — v1.3.0 renders 1..k trips per
    -- user from the trips_per_user release (planted users: exactly one).
    -- First leg: gap 0 (the start anchor); later legs carry their drawn gap.
    let tripPlans : List (Nat × List Leg) ←
      if u = 1 then do
        -- planted repeat-city trip: A,B,A,B over 4 legs, contiguous
        let picks := plantedPicks spec.cities 2 (fun i => cityId (2 * u + i))
        let legs ← plantedLegs spec picks [0, 1, 0, 1] plantedReturnNights
        pure [(monthStartDay 1 + 10, legs)]
      else if u = 2 then do
        -- planted tour: 4 legs over 3 distinct countries, one night each,
        -- contiguous (checkin gaps of 1 day ≤ tour_max_gap_d)
        let picks := plantedPicks spec.cities 3 (fun i => cityId (2 * u + i))
        let legs ← plantedLegs spec picks [0, 1, 2, 0] plantedTourNights
        pure [(monthStartDay 1 + 20, legs)]
      else if u = 3 then do
        -- planted long-stay trip: an 8-night stay + three short legs
        let picks := plantedPicks spec.cities 4 (fun i => cityId (2 * u + i))
        let legs ← plantedLegs spec picks [0, 1, 2, 3] plantedLongstayNights
        pure [(monthStartDay 1 + 30, legs)]
      else do
        -- reservations per user (per-USER total; the rc release is
        -- unchanged) -> decay-weighted inside the public bin range
        let rcCell ← spec.resCount.choose
        let nRaw ← drawRangeDecay (rangeOf spec.rcBinRanges rcCell.name ⟨1, 3⟩)
                    rcDecay.1 rcDecay.2
        let nRes := max 1 (min nRaw Params.rcMax)
        -- trips per user (v1.3.0): the tp release renders MULTI-TRIP users
        -- (real 91.1% / 7.4% / 1.5% for 1 / 2 / 3+; the pre-1.3.0
        -- single_trip prior rendered exactly 1.00)
        let tpCell ← spec.tripsPerUser.choose
        let kTrips ←
          if tpCell.name = "tp_2" then pure 2
          else if tpCell.name = "tp_3p" then
            drawRangeDecay ⟨3, Params.tpDrawHi⟩ rcDecay.1 rcDecay.2
          else pure 1
        let nTrips := min kTrips nRes
        -- balanced split prior (measured: 2-trip users' legs are
        -- near-equal, both >= 4 legs in 99.6% of cases)
        let base := nRes / nTrips
        let rem := nRes - base * nTrips
        let parts : List Nat :=
          (List.range nTrips).map (fun i => base + (if i < rem then 1 else 0))
        let mut outTrips : List (Nat × List Leg) := []
        let mut prevEnd := spec.windowStartD
        for j in [0:nTrips] do
          let nResT := parts.getD j 1
          -- start placement: the FIRST trip from the DP month-of-window
          -- cell (the real observation ramp); later trips follow the
          -- previous one plus a drawn inter-trip gap (declared prior:
          -- uniform [1, intertrip_gap_max_d], median-matched to the
          -- measured 47 days, IQR 4-117)
          let sd ←
            if j == 0 then do
              let mwCell ← spec.monthWindow.choose
              let dayOff ← drawRange ⟨1, 28⟩
              pure (monthWindowStartDay (mwNum mwCell.name) + (dayOff - 1))
            else do
              let g ← drawRange ⟨1, Params.intertripGapMaxD⟩
              pure (prevEnd + 1 + g)
          -- skip a trip that would start past the window end (the legs
          -- would clamp to nothing; better to leave the user shorter)
          if sd < spec.windowEndD then
            -- per-trip DEVICE plan (v1.3.0): dv cells -> k distinct
            -- classes over segments; 70.7% of real trips are single-device
            -- (iid per-leg draws would render ~12%)
            let dvCell ← spec.devicePersist.choose
            let kDev0 : Nat := if dvCell.name = "dv_2" then 2
                              else if dvCell.name = "dv_3" then 3
                              else 1
            let kDev := min kDev0 nResT
            let devPlan ← classPlan spec.device kDev nResT
            -- per-trip AFFILIATE plan: ap cells -> up to 4 distinct tiers
            -- (75.4% of real legs keep the previous leg's tier)
            let apCell ← spec.affPersist.choose
            let kAff0 : Nat := if apCell.name = "ap_2" then 2
                              else if apCell.name = "ap_3" then 3
                              else if apCell.name = "ap_4p" then 4
                              else 1
            let kAff := min kAff0 nResT
            let affPlan ← classPlan spec.affiliate kAff nResT
            -- start tier from the DP country mix; start city from the
            -- released popularity list of that tier
            let ctCell ← spec.countryMix.choose
            let startCt := if ctCell.name = "" then "Gondal" else ctCell.name
            cityCtr := cityCtr + 1
            let sc ← drawCityIn spec.cities startCt [] (cityId cityCtr)
            -- every city this trip has already visited
            let mut visited : List Nat := [sc.1]
            -- per-trip repeat decision: midpoint of the DP return-rate
            -- band; GIVEN a repeat trip, the anatomy from the DP
            -- repeat_struct cells (ext chains / return hops)
            let rrCell ← spec.returnRate.choose
            let p := bandMidpoint rrCell.name
            let repeatDraw ← draw [p, 100 - p]
            let (extK0, retK0) ←
              if repeatDraw == 0 && nResT > 1 then do
                let sCell ← spec.repeatStruct.choose
                repeatAnatomy sCell.name
              else
                pure (0, 0)
            let extK := min extK0 (nResT - 1)
            let retK := min retK0 (nResT - 1 - extK)
            let extPos ←
              if extK > 0 then do
                let i ← draw (List.replicate (nResT - extK) 1)
                pure (i + 1)
              else
                pure 0
            let retPoss ←
              if retK > 0 then
                drawPositionsFiltered (nResT - 1) retK extPos extK
              else
                pure []
            -- tier walk with second-order repeat structure
            let mut curCity := sc.1
            let mut curCountry := sc.2
            let mut curCt := startCt
            -- the "came from" of the first hop is the trip's ORIGIN: a
            -- hop-1 return renders a consecutive same-city stay (A,A)
            let mut cameCity := sc.1
            let mut cameCountry := sc.2
            let mut cameCt := startCt
            let mut ls : Array Leg := #[]
            -- per-segment synthetic tail ids: affSeg increments whenever the
            -- affiliate plan's TIER changes (a new booking channel segment)
            let mut affSeg : Nat := 1
            let mut prevAffCell := affPlan.getD 0 "aff_other"
            let n0Cell ← spec.nights.choose
            let n0 ← drawRangeDecay (rangeOf spec.nightsRanges n0Cell.name ⟨1, 1⟩)
                       nightsDecay.1 nightsDecay.2
            ls := ls.push (mkLeg curCity curCountry curCt
                             (devPlan.getD 0 "desktop")
                             (if prevAffCell = "aff_other"
                              then s!"s{affIdSynth (u * 131 + affSeg)}"
                              else prevAffCell)
                             0 n0)
            for i in [1:nResT] do
              let dest ←
                if extPos ≤ i && i < extPos + extK then
                  pure (curCity, curCountry, curCt)
                else if retPoss.contains i then
                  pure (cameCity, cameCountry, cameCt)
                else do
                  let row := rowOf spec.transitions spec.countryMix curCt
                  let nCell ← row.choose
                  let nCt := if nCell.name = "" then curCt else nCell.name
                  cityCtr := cityCtr + 1
                  let dc ← drawCityIn spec.cities nCt visited (cityId cityCtr)
                  pure (dc.1, dc.2, nCt)
              -- gap: DP gap class -> decay-weighted inside the public range
              let gCell ← spec.gap.choose
              let gap ← drawRangeDecay (rangeOf spec.gapRangesD gCell.name ⟨0, 0⟩)
                          gapDecay.1 gapDecay.2
              -- nights: DP nights class -> decay-weighted inside the range
              let nCell ← spec.nights.choose
              let nights ← drawRangeDecay (rangeOf spec.nightsRanges nCell.name ⟨1, 1⟩)
                             nightsDecay.1 nightsDecay.2
              -- device / affiliate from the per-trip PLAN (channel
              -- persistence; the marginals alone over-render switching);
              -- aff_other segments carry per-segment synthetic tail ids
              let dvc := devPlan.getD i "desktop"
              let affCell := affPlan.getD i "aff_other"
              if affCell != prevAffCell then
                affSeg := affSeg + 1
                prevAffCell := affCell
              let aff := if affCell = "aff_other"
                         then s!"s{affIdSynth (u * 131 + affSeg)}"
                         else affCell
              ls := ls.push (mkLeg dest.1 dest.2.1 dest.2.2 dvc aff gap nights)
              visited := dest.1 :: visited
              cameCity := curCity
              cameCountry := curCountry
              cameCt := curCt
              curCity := dest.1
              curCountry := dest.2.1
              curCt := dest.2.2
            outTrips := outTrips ++ [(sd, ls.toList)]
            let span := (ls.map (fun l => l.gap + l.nights)).foldl (· + ·) 0
            prevEnd := sd + span
        pure outTrips

    -- smart-constructed legs + window clamp + rendering PER TRIP: the
    -- hypotheses of every rendering theorem hold by construction per trip
    -- (nights ≥ 1 by mkLeg, span by legsWithin_span_le, hence
    -- mkLegs_in_window / mkLegs_no_overlap); the utrip suffix is the
    -- 1-based trip index (v1.3.0 multi-trip users)
    let mut totalRows : Nat := 0
    let mut firstTripRows : List LegRow := []
    for j in [0:tripPlans.length] do
      let (startDay, legs) := tripPlans.getD j (0, [])
      let kept := legsWithin spec.windowEndD startDay legs
      let rows := mkLegs startDay kept
      if j == 0 then
        firstTripRows := rows
      totalRows := totalRows + rows.length
      for r in rows do
        -- country names in the released lists may contain commas (e.g.
        -- "Merania, Kingdom of"): every string field goes through
        -- csvCell, which quotes exactly when RFC 4180 requires it, so
        -- the re-ingestion parser never splits a field.
        tsv := tsv.push <| s!"{u},{dateStr r.checkin},{dateStr (r.checkin + r.nights)}," ++
          s!"{r.city},{csvCell r.device},{affIdOf r.aff},{csvCell booker},{csvCell r.country},{u}_{j + 1}"
    let rows := firstTripRows
    userPlan := userPlan.push <| Json.mkObj
      [("user", jn u), ("n_res", jn totalRows)]
    if u = 1 then
      planReturn := Json.mkObj
        [ ("user", jn 1)
        , ("cities", Json.arr ((rows.map (fun r => jn r.city)).toArray))
        , ("days", Json.arr ((rows.map (fun r => jn r.checkin)).toArray))
        , ("nights", Json.arr ((rows.map (fun r => jn r.nights)).toArray)) ]
    if u = 2 then
      planTour := Json.mkObj
        [ ("user", jn 2)
        , ("countries", Json.arr ((rows.map (fun r => Json.str r.country)).toArray))
        , ("days", Json.arr ((rows.map (fun r => jn r.checkin)).toArray))
        , ("nights", Json.arr ((rows.map (fun r => jn r.nights)).toArray)) ]
    if u = 3 then
      planLongstay := Json.mkObj
        [ ("user", jn 3)
        , ("nights", Json.arr ((rows.map (fun r => jn r.nights)).toArray))
        , ("days", Json.arr ((rows.map (fun r => jn r.checkin)).toArray)) ]

  IO.FS.createDirAll outDir
  IO.FS.writeFile s!"{outDir}/booking_synth.csv"
    (String.intercalate "\n" tsv.toList ++ "\n")
  let plan := Json.mkObj
    [ ("planted", Json.mkObj
        [ ("return", planReturn)
        , ("tour", planTour)
        , ("longstay", planLongstay) ])
    , ("users", Json.arr userPlan)
    , ("n_users", jn nUsers)
    , ("spec_version", Json.str spec.version)
    , ("window", Json.mkObj [("start_day", jn spec.windowStartD),
                             ("end_day", jn spec.windowEndD)])
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {nUsers} users, {tsv.size - 1} reservations -> {outDir}/booking_synth.csv"
  return 0

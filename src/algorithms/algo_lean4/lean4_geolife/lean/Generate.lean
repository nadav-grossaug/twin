/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → GeoLife TSV.

    lake exe generate <genspec.json> <n_users> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the Motion constants (public physics, calendar, id shapes). Every random
draw runs the verified sampler (`sampleCellPMF_law`); each user-day's stay
chain is built through `mkStay`/`mkStays`, so the plan invariants hold by
construction and by theorem:
  arrive/leave strictly increasing per day   (mkStays_times_sorted),
  every planted stay clears the staypoint
  dwell threshold                            (mkStays_isStay),
  planted days are sequence-ok visit chains  (mkStays_visitSeq +
                                              trip_interval_le_gap),
  stay wobble fits the detection box         (wobble_box_fits),
  rendered speeds within the public cap      (render_speed_within_cap),
  point timestamps strictly increasing       (stepTimes_sorted pattern).
Writes geolife_synth.tsv (source TSV schema, 10 columns + header) and
plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.

COORDINATES are a NEUTRAL local frame (degrees around 0°N 0°E, isotropic by
construction — see Motion.lean): the mandatory `scripts/synth.py manipulate`
step relocates each user onto Tokyo public geography. No source geography is
ever input, so none can leak.

Generative structure (public priors declared derived_from_data: false):
  daily motif   = DP motif histogram; visit chains per motif template;
  role walk     = DP role transitions (irregular days);
  dwells        = DP dwell classes, grid-uniform inside the public range;
  departures    = DP departure bins, uniform inside the public hour range;
  modes/speeds  = DP mode mix + released kinematics (category-2);
  paths         = ROAD-LIKE trip polylines from the released route shape
                  (category-2): junction-to-junction segment lengths from
                  the released segment histogram, junction angles from the
                  released junction-angle histogram (in-class jitter keeps
                  them off the 0/±90/180 lattice), left/right from the
                  released share, path-length budget from the released
                  per-mode circuity — then the shape is rotated/scaled to
                  land exactly on the target anchor. Within segments the
                  AR(1) GPS noise breaks exact collinearity;
  GPS noise     = AR(1) per axis matched to the released noise moments
                  (category-2), wobble clamped into the detection box.
PLANTED ground truth: user 1 = commuter (commute day every day); user 2 =
revisit beacon (same other anchor daily) + a 3-anchor tour on day 1.
-/
import EnclaveGeo.GenSpec
import EnclaveGeo.Motion
import EnclaveGeo.Params

open Lean EnclaveGeo EnclaveGeo.Gen EnclaveGeo.Shape

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)

def tsvHeader : String :=
  "user_id\tsequence_uuid\ttrajectory_id\tdatetime\tlatitude\tlongitude\taltitude_ft\tlabel_start\tlabel_end\tmode"

def pad3 (n : Nat) : String :=
  if n < 10 then s!"00{n}" else if n < 100 then s!"0{n}" else s!"{n}"

def rangeOf (rs : List (String × RangeN)) (k : String) (dflt : RangeN) : RangeN :=
  ((rs.find? (·.1 == k)).map (·.2)).getD dflt

/-- Grid-uniform draw in [lo, hi] via the verified sampler. -/
def drawRange (r : RangeN) : IO Nat := do
  let span := r.hi - r.lo
  if span = 0 then
    pure r.lo
  else
    let buckets := min span 4096
    let i ← draw (List.replicate (buckets + 1) 1)
    pure (r.lo + span * i / buckets)

/-- Uniform Float in [0, 1) via the verified sampler (1/1024 grid). -/
def drawU : IO Float := do
  let i ← draw (List.replicate 1024 1)
  pure (Float.ofNat i / 1024.0)

/-- Approximately standard-normal innovation (Irwin–Hall of 4 uniforms),
every uniform through the verified sampler. -/
def drawGauss : IO Float := do
  let a ← drawU; let b ← drawU; let c ← drawU; let d ← drawU
  pure ((a + b + c + d - 2.0) * 1.732)

def fToInt (f : Float) : Int :=
  if f < 0 then -(Int.ofNat ((-f).toUInt64.toNat)) else Int.ofNat f.toUInt64.toNat

/-- One TSV point row in the neutral frame (x, y in mm; t epoch seconds). -/
def pointRow (uid : Nat) (seqId trajId : String) (t : Nat) (x y : Int)
    (labelA labelB : Nat) (mode : String) : String :=
  s!"{pad3 uid}\t{seqId}\t{trajId}\t{dtStr t}\t{deg6Str (mmToMicrodeg y)}\t{deg6Str (mmToMicrodeg x)}\t-777\t{dtStr labelA}\t{dtStr labelB}\t{mode}"

structure RenderState where
  tsv : Array String := #[]
  seqCtr : Nat := 0
  trajId : String := ""
  nPoints : Nat := 0

/-- Released (or fallback) kinematics of a mode class. -/
def kinOf (spec : GenSpec) (m : String) : Nat × Nat :=
  match spec.kinematics.find? (·.mode == m) with
  | some k => (min (max k.speedMmps 500) (renderSpeedCapMmps m), max 10 k.dtDs)
  | none => (modeCruiseMmps m, 10 * genSampleDtS)

/-- Released (or fallback) route shape of a mode: (median circuity permille,
junctions per km × 100). -/
def circOf (spec : GenSpec) (m : String) : Nat × Nat :=
  match spec.routeCirc.find? (·.mode == m) with
  | some r => (r.circPm, max 10 r.juncPkmC)
  | none => (defaultCircPm, 300)

/-- Build a road-like trip route from `p0` to `p1`: segment lengths drawn
from the released segment histogram (public class ranges, `segRangeOf`)
scaled to the mode's released junction DENSITY (a car turns ~0.8×/km, a
walker ~11×/km), junction angles from the released junction-angle histogram
(public ranges `jaRangeOf` + in-class jitter — real turns by
`jaRange_is_junction`, never on the 90° lattice), turn direction from the
released left share with the cumulative deviation clamped to ±75° (streets
alternate: routes progress forward instead of meandering), and the
path-length budget from the released per-mode circuity. The raw shape is
then rotated and scaled so its endpoint lands exactly on `p1` (an affine
map preserves the turn-angle structure). A shape that curls up (endpoint
< 15% of its length) falls back to a gentle arc — declared. Returns the
polyline (mm) and its total length. -/
def buildRoute (spec : GenSpec) (mode : String) (p0 p1 : Int × Int) :
    IO (Array (Float × Float) × Float) := do
  let x0 := Float.ofInt p0.1
  let y0 := Float.ofInt p0.2
  let dx := Float.ofInt (p1.1 - p0.1)
  let dy := Float.ofInt (p1.2 - p0.2)
  let d := Float.sqrt (dx * dx + dy * dy)
  if d < 50000.0 then   -- < 50 m: a single micro-hop
    return (#[(x0, y0), (x0 + dx, y0 + dy)], max d 1.0)
  let (circPm, juncPkmC) := circOf spec mode
  let jit ← drawU
  let budget := d * (Float.ofNat circPm / 1000.0) * (0.85 + 0.3 * jit)
  -- segment-length scale: mean inter-junction distance = 1 km / density
  -- (the released histogram carries the SHAPE of the distribution, the
  -- released per-mode density carries its scale; ~200 m = the pooled mean)
  let segScale := min 8.0 (max 0.2 ((100000.0 / Float.ofNat juncPkmC) / 200.0))
  -- raw shape from the released route statistics
  let mut vx : Array (Float × Float) := #[(0.0, 0.0)]
  let mut dev := 0.0   -- cumulative deviation from the base heading
  let mut sx := 0.0
  let mut sy := 0.0
  let mut cum := 0.0
  for _ in [0:maxTripSegments] do
    if cum < budget then
      let sc ← spec.segHist.choose
      let (lo, hi) := segRangeOf sc.name
      let lenM ← drawRange ⟨lo, hi⟩
      let len := min 2000000.0 (max 25000.0 (Float.ofNat (lenM * 1000) * segScale))
      sx := sx + len * Float.cos dev
      sy := sy + len * Float.sin dev
      vx := vx.push (sx, sy)
      cum := cum + len
      let jc ← spec.junctionHist.choose
      let (alo, ahi) := jaRangeOf jc.name
      let aDeg ← drawRange ⟨alo, ahi⟩
      let jj ← drawU
      let s ← draw [spec.shareLeftPct, 100 - spec.shareLeftPct]
      let sign := if s = 0 then 1.0 else -1.0
      let turn := (Float.ofNat aDeg + jj - 0.5) * 3.14159265 / 180.0
      -- clamp the deviation: when a turn would swing past ±75°, take it the
      -- other way (street grids alternate; trips progress toward the target)
      dev := if (dev + sign * turn).abs > 1.309 then dev - sign * turn
             else dev + sign * turn
  let e := Float.sqrt (sx * sx + sy * sy)
  -- the endpoint must carry ≥ 40% of the drawn path (realized circuity
  -- ≤ 2.5, the top public circuity class edge) — keeps every user's frame
  -- extent inside the manipulate target bbox
  if e < 1.0 ∨ e < 0.4 * cum then
    -- curled-up walk: gentle arc through a perpendicular midpoint (declared)
    let bow ← drawU
    let off := d * (0.08 + 0.15 * bow)
    let mx := x0 + dx / 2.0 - dy / d * off
    let my := y0 + dy / 2.0 + dx / d * off
    let l2 := 2.0 * Float.sqrt (d * d / 4.0 + off * off)
    return (#[(x0, y0), (mx, my), (x0 + dx, y0 + dy)], l2)
  -- affine fit: rotate the shape-end vector onto the target vector, scale
  let k := d / e
  let cosR := (sx * dx + sy * dy) / (e * d)
  let sinR := (sx * dy - sy * dx) / (e * d)
  let mut out : Array (Float × Float) := #[]
  for (ax, ay) in vx do
    out := out.push (x0 + k * (cosR * ax - sinR * ay),
                     y0 + k * (sinR * ax + cosR * ay))
  out := out.set! (out.size - 1) (x0 + dx, y0 + dy)
  return (out, cum * k)

/-- Render one trip by walking its route polyline over [t0, t1] at the
implied constant pace, with AR(1) GPS noise per axis. MOVING noise is the
released static sd attenuated 8× (declared public prior: receivers filter
during motion — the measured 25 m-resampled real tracks turn a median 2.2°,
which a static-sd jitter would swamp); stay wobble keeps the full released
moments. The residual noise still breaks exact collinearity inside segments.
Points are emitted strictly before `t1` — the arrival instant belongs to the
stay. -/
def renderTripAlong (spec : GenSpec) (route : Array (Float × Float))
    (t0 t1 : Nat) : IO (Array (Nat × Int × Int)) := do
  let dur := t1 - t0
  if dur < 2 ∨ route.size < 2 then
    let (x, y) := route[0]!
    return #[(t0, fToInt x, fToInt y)]
  let dt := max genSampleDtS ((dur + 119) / 120)
  let mut cums : Array Float := #[0.0]
  for i in [1:route.size] do
    let (ax, ay) := route[i - 1]!
    let (bx, qy) := route[i]!
    cums := cums.push (cums[i - 1]! +
      Float.sqrt ((bx - ax) * (bx - ax) + (qy - ay) * (qy - ay)))
  let total := max cums[cums.size - 1]! 1.0
  let sd := Float.ofNat spec.noise.sdMm / 8.0
  let rho := Float.ofNat spec.noise.lag1Pct / 100.0
  let innovScale := sd * Float.sqrt (max 0.05 (1.0 - rho * rho))
  let mut nx := 0.0
  let mut ny := 0.0
  let mut seg := 1
  let mut out : Array (Nat × Int × Int) := #[]
  let n := dur / dt
  for i in [0:n + 1] do
    let t := t0 + i * dt
    if t < t1 then
      let s := total * Float.ofNat (t - t0) / Float.ofNat dur
      while seg < route.size - 1 ∧ cums[seg]! < s do
        seg := seg + 1
      let (ax, ay) := route[seg - 1]!
      let (bx, qy) := route[seg]!
      let c0 := cums[seg - 1]!
      let cl := max (cums[seg]! - c0) 1.0
      let f := min 1.0 (max 0.0 ((s - c0) / cl))
      out := out.push (t, fToInt (ax + f * (bx - ax) + nx),
                       fToInt (ay + f * (qy - ay) + ny))
      let g1 ← drawGauss
      let g2 ← drawGauss
      nx := rho * nx + innovScale * g1
      ny := rho * ny + innovScale * g2
  -- snap the LAST recorded fix onto the route endpoint (+ noise): the
  -- emission grid stops short of the arrival instant, and an un-snapped
  -- final fix up to path/120 away from the anchor smears the implied-stay
  -- clusters on re-ingestion (13 "anchors" where the plan has 2)
  if h : out.size > 0 then
    let (tl, _, _) := out[out.size - 1]
    let (ex, ey) := route[route.size - 1]!
    out := out.set! (out.size - 1) (tl, fToInt (ex + nx), fToInt (ey + ny))
  return out

/-- Render stay wobble at an anchor: pure AR(1) around the anchor, clamped
into the detection box (`wobble_box_fits`). -/
def renderWobble (spec : GenSpec) (p : Int × Int) (t0 t1 : Nat) :
    IO (Array (Nat × Int × Int)) := do
  let sd := Float.ofNat spec.noise.sdMm
  let rho := Float.ofNat spec.noise.lag1Pct / 100.0
  let innovScale := sd * Float.sqrt (max 0.05 (1.0 - rho * rho))
  let cap := Float.ofNat wobbleCapMm
  let clampW (v : Float) : Float := max (-cap) (min cap v)
  let n := (t1 - t0) / genStayDtS
  let mut wx := 0.0
  let mut wy := 0.0
  let mut out : Array (Nat × Int × Int) := #[]
  for i in [0:min n 120 + 1] do
    let t := t0 + i * genStayDtS
    if t < t1 then
      out := out.push (t, fToInt (Float.ofInt p.1 + wx), fToInt (Float.ofInt p.2 + wy))
    let g1 ← drawGauss
    let g2 ← drawGauss
    wx := clampW (rho * wx + innovScale * g1)
    wy := clampW (rho * wy + innovScale * g2)
  return out

/-- A user's anchor frame: home at the origin, the others placed by released
displacement classes at golden-angle bearings (no lattice directions). -/
def mkAnchors (spec : GenSpec) (nOther : Nat) : IO (Array (Int × Int)) := do
  let mut anchors : Array (Int × Int) := #[(0, 0)]   -- 0 = home
  for i in [0:nOther + 1] do                          -- 1 = work, 2.. = other
    let dCell ← spec.distMix.choose
    let distM ← drawRange (rangeOf spec.distRangesM
      (if dCell.name = "" then "dist_1_3k" else dCell.name) ⟨800, 4000⟩)
    let j ← drawU
    let ang := 2.39996 * Float.ofNat (i + 1) + (j - 0.5) * 0.9
    let x := fToInt (Float.ofNat (distM * 1000) * Float.cos ang)
    let y := fToInt (Float.ofNat (distM * 1000) * Float.sin ang)
    anchors := anchors.push (x, y)
  return anchors

/-- Draw a dwell (s) for a class drawn from the DP dwell histogram, clamped
into [lo, hi]. -/
def drawDwell (spec : GenSpec) (lo hi : Nat) : IO Nat := do
  let c ← spec.dwell.choose
  let d ← drawRange (rangeOf spec.dwellRangesS
    (if c.name = "" then "dw_1_3h" else c.name) ⟨1800, 7200⟩)
  pure (min hi (max lo d))

/-- Build a day's stay plan for a motif class. Anchor slots: 0 = home,
1 = work, 2+ = other. Gaps are TRIP DURATIONS, filled in at render time from
distance/speed; here they carry the planned trip cap so `mkStays_visitSeq`'s
hypothesis holds by construction. -/
def dayPlanOf (spec : GenSpec) (motif : String) (nAnchors : Nat)
    (beaconAnchor : Option Nat) : IO (List (String × Nat)) := do
  -- (role, anchor-slot) chain, home-return at the end where the motif says so
  match motif with
  | "commute" => pure [("work", 1), ("home", 0)]
  | "errand_loop" =>
    let o := beaconAnchor.getD 2
    pure [("other", o), ("home", 0)]
  | "tour" => do
    let a := beaconAnchor.getD 2
    let b := if a + 1 < nAnchors then a + 1 else 2
    let c := if a + 2 < nAnchors then a + 2 else 3
    pure [("other", a), ("other", b), ("other", c), ("home", 0)]
  | "home_day" => pure []
  | _ => do   -- irregular: role walk via the DP transition rows
    let roles ← genChain spec.transitions spec.motif 3 "home"
    let mut plan : List (String × Nat) := []
    let mut slot := 2
    for r in roles do
      match r with
      | "home" => plan := plan ++ [("home", 0)]
      | "work" => plan := plan ++ [("work", 1)]
      | _ =>
        plan := plan ++ [("other", min slot (nAnchors - 1))]
        slot := slot + 1
    pure (if plan.isEmpty then [("other", 2), ("home", 0)] else plan)

def main (args : List String) : IO UInt32 := do
  let [specPath, usersStr, outDir] := args
    | do IO.eprintln "usage: generate <genspec.json> <n_users> <out_dir>"; return 1
  let some nUsers := usersStr.toNat?
    | do IO.eprintln "generate: n_users must be a natural"; return 1
  if nUsers < 3 ∨ nUsers > 5000 then
    IO.eprintln "generate: n_users must be in 3..5000"; return 1
  let txt ← IO.FS.readFile specPath
  let spec ← match Json.parse txt >>= GenSpec.parse with
    | .error e => do IO.eprintln s!"generate: bad genspec: {e}"; return 1
    | .ok s => pure s
  if spec.windowEndS ≤ spec.windowStartS + 20 * 86400 then
    IO.eprintln "generate: genspec window too small"; return 1

  let nDays := 14                      -- public prior: a two-week slice
  let nOther := 6                      -- anchor slots per user (public prior)
  let spanDays := (spec.windowEndS - spec.windowStartS) / 86400 - (nDays + 1)

  let mut tsv : Array String := #[tsvHeader]
  let mut seqCtr : Nat := 0
  let mut planUsers : Array Json := #[]

  for u in [1:nUsers + 1] do
    let anchors ← mkAnchors spec nOther
    -- user day-0 offset into the window (uniform; public prior)
    let offD ← draw (List.replicate (min spanDays 365) 1)
    let day0 := spec.windowStartS / 86400 + offD
    let mut dayMotifs : Array Json := #[]
    for d in [0:nDays] do
      -- planted users: 1 = commuter, 2 = revisit beacon (+ tour on day 1)
      let motif ←
        if u = 1 then pure "commute"
        else if u = 2 then pure (if d = 1 then "tour" else "errand_loop")
        else do pure (← spec.motif.choose).name
      let motif := if motif = "" then "irregular" else motif
      let plan ← dayPlanOf spec motif anchors.size (if u = 2 then some 2 else none)
      dayMotifs := dayMotifs.push (Json.str motif)
      if plan.isEmpty then
        continue   -- home day: device off, the day is unobserved (declared)
      -- departure from home: DP departure bin -> hour -> second of day
      let depCell ← spec.departure.choose
      let depR := rangeOf spec.depRangesH
        (if depCell.name = "" then "dep_morning" else depCell.name) ⟨7, 10⟩
      let depH ← drawRange depR
      let depMin ← drawRange ⟨0, 59⟩
      let mut tCur := day0 * 86400 + d * 86400 + depH * 3600 + depMin * 60
      let mut pCur : Int × Int := anchors[0]!   -- leave from home
      -- walk the plan: trip -> stay(wobble) -> trip ... (one trajectory per
      -- recorded burst; long stays switch the device off)
      let mut stayPlans : Array (String × Nat × Nat × Nat) := #[]  -- role, slot, trip_s, dwell_s
      let mut routes : Array (Array (Float × Float) × String) := #[]
      for (i, (role, slot)) in plan.zipIdx.map (fun p => (p.2, p.1)) do
        let target := anchors[slot]!
        let modeCell ← spec.mode.choose
        let mode := if modeCell.name = "" then "walk" else modeCell.name
        -- the road-like route is built FIRST; the trip duration follows the
        -- actual path length, never the straight-line distance
        let (route, lenMm) ← buildRoute spec mode pCur target
        let (speedMmps, _) := kinOf spec mode
        let tripRaw := (fToInt (lenMm / Float.ofNat speedMmps)).toNat
        -- clamped trips still respect the public speed cap (96 < 120 m/s)
        let minDur := (fToInt (lenMm / 96000.0)).toNat
        let trip := min genTripMaxS (max (max 180 minDur) tripRaw)
        let isLast := i + 1 = plan.length
        let dwell ←
          if role = "home" then pure (10 * 3600)
          else if motif = "tour" ∧ ¬ isLast then drawDwell spec 1800 (Params.tourMaxGapS - trip - 60)
          else if motif = "errand_loop" ∧ role = "other" then drawDwell spec 1500 (Params.errandMaxDwellS - 1)
          else if role = "work" then drawDwell spec (6 * 3600) (12 * 3600)
          else drawDwell spec 1500 (3 * 3600)
        stayPlans := stayPlans.push (role, slot, trip, dwell)
        routes := routes.push (route, mode)
        pCur := target
      -- render: the smart constructor threads the times (theorems' shape)
      let stays := mkStays tCur
        (stayPlans.toList.map (fun (_, _, g, dw) =>
          { role := "", anchor := 0, gap := g, dwell := dw : StayPlan }))
      let mut pFrom : Int × Int := anchors[0]!
      let mut tLeave := tCur
      let mut trajStart := tCur
      let mut trajPts : Array (Nat × Int × Int × String × Nat × Nat) := #[]  -- t x y mode labelA labelB
      for (i, s) in stays.zipIdx.map (fun p => (p.2, p.1)) do
        let (role, slot, _, dwellPlanned) := stayPlans[i]!
        let target := anchors[slot]!
        let (route, mode) := routes[i]!
        let pts ← renderTripAlong spec route tLeave s.arrive
        for (t, x, y) in pts do
          trajPts := trajPts.push (t, x, y, modeRep mode, tLeave, s.arrive)
        -- short stays stay recorded (wobble); long stays end the trajectory
        if s.leave - s.arrive ≤ 3600 then
          let w ← renderWobble spec target s.arrive s.leave
          for (t, x, y) in w do
            trajPts := trajPts.push (t, x, y, "walk", s.arrive, s.leave)
        else
          -- flush the trajectory recorded so far
          if trajPts.size > 0 then
            let tid := tidStr trajStart
            let sid := uuidStr seqCtr
            seqCtr := seqCtr + 1
            for (t, x, y, m, la, lb) in trajPts do
              tsv := tsv.push (pointRow u sid tid t x y la lb m)
            trajPts := #[]
          trajStart := s.leave
        let _ := dwellPlanned
        let _ := role
        pFrom := target
        tLeave := s.leave
      if trajPts.size > 0 then
        let tid := tidStr trajStart
        let sid := uuidStr seqCtr
        seqCtr := seqCtr + 1
        for (t, x, y, m, la, lb) in trajPts do
          tsv := tsv.push (pointRow u sid tid t x y la lb m)
    planUsers := planUsers.push <| Json.mkObj
      [ ("user", Json.str (pad3 u)), ("day0", jn day0)
      , ("motifs", Json.arr dayMotifs) ]

  IO.FS.createDirAll outDir
  IO.FS.writeFile s!"{outDir}/geolife_synth.tsv"
    (String.intercalate "\n" tsv.toList ++ "\n")
  let plan := Json.mkObj
    [ ("planted", Json.mkObj
        [ ("commute_user", Json.str (pad3 1))
        , ("beacon_user", Json.str (pad3 2))
        , ("beacon_min_days", jn Params.revisitMinDays)
        , ("tour_user", Json.str (pad3 2)) ])
    , ("users", Json.arr planUsers)
    , ("n_users", jn nUsers)
    , ("n_days", jn nDays)
    , ("frame", Json.str "neutral-local (0N 0E isotropic); manipulate step required")
    , ("spec_version", Json.str spec.version)
    , ("window", Json.mkObj [("start_s", jn spec.windowStartS),
                             ("end_s", jn spec.windowEndS)])
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {nUsers} users, {tsv.size - 1} points -> {outDir}/geolife_synth.tsv"
  return 0

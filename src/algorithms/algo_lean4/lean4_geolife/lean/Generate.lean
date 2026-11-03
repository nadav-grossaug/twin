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

/-- The uniform 1/1024 grid, hoisted: `drawU` runs twice per rendered
coordinate axis, so the weight list is allocated once, not per draw. -/
def uGrid1024 : List Nat := List.replicate 1024 1

/-- Uniform Float in [0, 1) via the verified sampler (1/1024 grid). -/
def drawU : IO Float := do
  let i ← draw uGrid1024
  pure (Float.ofNat i / 1024.0)

/-- Approximately standard-normal innovation (Irwin–Hall of 4 uniforms),
every uniform through the verified sampler. -/
def drawGauss : IO Float := do
  let a ← drawU; let b ← drawU; let c ← drawU; let d ← drawU
  pure ((a + b + c + d - 2.0) * 1.732)

def fToInt (f : Float) : Int :=
  if f < 0 then -(Int.ofNat ((-f).toUInt64.toNat)) else Int.ofNat f.toUInt64.toNat

/-- One TSV point row in the neutral frame (x, y in mm). `t`, `labelA`,
`labelB` are Beijing-LOCAL epoch seconds (the frame every released hour bin
lives in); the row is written in the source convention, UTC, so production
staging's `source_utc_offset_s` shift maps it back exactly. -/
def pointRow (uid : Nat) (seqId trajId : String) (t : Nat) (x y : Int)
    (labelA labelB : Nat) (mode : String) : String :=
  let off := Params.sourceUtcOffsetS
  s!"{pad3 uid}\t{seqId}\t{trajId}\t{dtStr (t - off)}\t{deg6Str (mmToMicrodeg y)}\t{deg6Str (mmToMicrodeg x)}\t-777\t{dtStr (labelA - off)}\t{dtStr (labelB - off)}\t{mode}"

structure RenderState where
  tsv : Array String := #[]
  seqCtr : Nat := 0
  trajId : String := ""
  nPoints : Nat := 0

/-- TSV rows of one recorded trajectory. A new `sequence_uuid` starts at
every label-window change (mode, label_start, label_end), as in the source,
where each sequence is exactly ONE labelled segment (0 of 2,493 real
sequences span two windows). Sharing one id across a trajectory merged
same-mode trips separated by a recorded stay into one out-and-back "leg"
in production `feat_legs`, collapsing re-detected straightness
(bike/bus/rail 0.43 vs released 0.73). Returns the rows and the next free
sequence counter. -/
def trajRows (u : Nat) (tid : String)
    (pts : Array (Nat × Int × Int × String × Nat × Nat)) (ctr : Nat) :
    Array String × Nat := Id.run do
  let mut out : Array String := #[]
  let mut c := ctr
  let mut key : Option (String × Nat × Nat) := none
  let mut sid := ""
  for (t, x, y, m, la, lb) in pts do
    if key != some (m, la, lb) then
      sid := uuidStr c
      c := c + 1
      key := some (m, la, lb)
    out := out.push (pointRow u sid tid t x y la lb m)
  return (out, c)

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

/-- Trip mode CONDITIONED ON TRIP LENGTH: weight of mode m = released mode
weight × a log-normal plausibility (σ = 1 in log-metres, a declared public
prior) of this trip's expected path (straight-line distance × the mode's
released circuity) around the mode's RELEASED median leg path (kinematics
`median_path_m`, category-2). Drawing the mode independently of distance
produced 20 km "walks" capped at the 2 h trip limit and 2-minute "rail"
hops: 65% of synthetic points were walk vs 28% released, rail 4% vs 17%.
Modes without a released path keep their marginal weight. One verified draw
(weights scaled ×1000 to integers). -/
def chooseMode (spec : GenSpec) (dMm : Float) : IO String := do
  let ws := spec.mode.cells.map fun c =>
    match spec.kinematics.find? (·.mode == c.name) with
    | some k =>
      if k.pathM = 0 then c.w * 1000
      else
        let pathM := max 1.0 (dMm / 1000.0 * Float.ofNat (circOf spec c.name).1 / 1000.0)
        let z := Float.log pathM - Float.log (Float.ofNat k.pathM)
        (fToInt (Float.ofNat (c.w * 1000) * Float.exp (-0.5 * z * z))).toNat
    | none => c.w * 1000
  let i ← draw ws
  let name := (spec.mode.cells.getD i ⟨"walk", 0⟩).name
  pure (if name = "" then "walk" else name)

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
stay. `dtS` is the sampling cadence in seconds: the RELEASED per-mode median
sampling interval (genspec kinematics `dt_ds`), so synthetic tracks carry the
source logger's point density instead of a fixed points-per-trip budget. -/
def renderTripAlong (spec : GenSpec) (route : Array (Float × Float))
    (t0 t1 : Nat) (dtS : Nat) : IO (Array (Nat × Int × Int)) := do
  let dur := t1 - t0
  if dur < 2 ∨ route.size < 2 then
    let (x, y) := route[0]!
    return #[(t0, fToInt x, fToInt y)]
  let dt := max 1 dtS
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
into the detection box (`wobble_box_fits`). `dtS` = cadence in seconds
(`genStayDtS`, a sparse public prior; recorded stays are ≤ 1 h, so the
point count is bounded by 3600/dtS). -/
def renderWobble (spec : GenSpec) (p : Int × Int) (t0 t1 : Nat) (dtS : Nat) :
    IO (Array (Nat × Int × Int)) := do
  let sd := Float.ofNat spec.noise.sdMm
  let rho := Float.ofNat spec.noise.lag1Pct / 100.0
  let innovScale := sd * Float.sqrt (max 0.05 (1.0 - rho * rho))
  let cap := Float.ofNat wobbleCapMm
  let clampW (v : Float) : Float := max (-cap) (min cap v)
  let dt := max 1 dtS
  let n := (t1 - t0) / dt
  let mut wx := 0.0
  let mut wy := 0.0
  let mut out : Array (Nat × Int × Int) := #[]
  for i in [0:n + 1] do
    let t := t0 + i * dt
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

/-- Every pair of the given anchor slots is at least `minMm` apart. -/
def slotsSeparated (anchors : Array (Int × Int)) (slots : List Nat)
    (minMm : Float) : Bool :=
  slots.all fun i => slots.all fun j =>
    i == j ||
      (let (ax, ay) := anchors[i]!
       let (bx, by_) := anchors[j]!
       let dx := Float.ofInt (ax - bx)
       let dy := Float.ofInt (ay - by_)
       Float.sqrt (dx * dx + dy * dy) ≥ minMm)

/-- Anchor frame of the PLANTED tour user: redraw (bounded fuel) until home
and the three tour stops are pairwise ≥ 4 anchor cells apart, so the
production anchor grid can never fold a tour stop into home or into another
stop (an unseparated draw, e.g. a 200 m stop beside home, re-ingests as a
2-stop day and the planted tour is not recalled), AND every stop is within
5 km of home, so (with car legs, see `main`) every tour leg
is far below the `tour_max_gap_s` chaining bound: a leg at the 2 h trip cap
left zero slack, and re-detected stay boundaries pushed one gap to 7,520 s,
breaking the planted tour chain. Ground-truth users only; ordinary users
keep the released displacement mix untouched. -/
def mkPlantedTourAnchors (spec : GenSpec) (nOther : Nat) :
    IO (Array (Int × Int)) := do
  let minMm := Float.ofNat (4 * Params.anchorCellM * 1000)
  let reachMm := 5000000.0
  let near (a : Array (Int × Int)) : Bool :=
    [2, 3, 4].all fun i =>
      let (x, y) := a[i]!
      Float.sqrt (Float.ofInt x * Float.ofInt x + Float.ofInt y * Float.ofInt y) ≤ reachMm
  let mut a ← mkAnchors spec nOther
  for _ in [0:2000] do
    if slotsSeparated a [0, 2, 3, 4] minMm ∧ near a then
      return a
    a ← mkAnchors spec nOther
  return a

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
  -- per-user recorded-days slice: heavy-tailed PUBLIC prior
  -- (`activeDayClasses`, Motion.lean); planted users keep a fixed slice
  if spec.windowEndS ≤ spec.windowStartS + (maxActiveDays + 7) * 86400 then
    IO.eprintln "generate: genspec window too small"; return 1

  let nOther := 6                      -- anchor slots per user (public prior)
  -- offsets bounded by the LONGEST slice, so every user fits the window
  let spanDays := (spec.windowEndS - spec.windowStartS) / 86400 - (maxActiveDays + 2)

  let mut tsv : Array String := #[tsvHeader]
  let mut seqCtr : Nat := 0
  let mut planUsers : Array Json := #[]

  for u in [1:nUsers + 1] do
    let nDays ←
      if u ≤ 2 then pure plantedActiveDays
      else do
        let i ← draw (activeDayClasses.map (·.2.2))
        let (lo, hi, _) := activeDayClasses.getD i (17, 32, 1)
        drawRange ⟨lo, hi⟩
    let anchors ←
      if u = 2 then mkPlantedTourAnchors spec nOther else mkAnchors spec nOther
    -- user day-0 offset into the window (uniform; public prior). Day 0 is
    -- the window's SECOND local day, so a 00:xx local departure (16:xx UTC
    -- the day before) still falls inside the window in the UTC output too.
    let offD ← draw (List.replicate (min spanDays 365) 1)
    let day0 := spec.windowStartS / 86400 + 1 + offD
    let mut dayMotifs : Array Json := #[]
    -- a user's schedule can spill past midnight; the next day's departure
    -- must not start before it ends, or two trajectories share a start second
    let mut tPrevEnd : Nat := 0
    -- every cohort member recorded at least once (that is why they are in
    -- the data); a short slice drawn as all home days would render a user
    -- with zero points, so the last day is forced observable if needed
    let mut observed := false
    for d in [0:nDays] do
      -- planted users: 1 = commuter, 2 = revisit beacon (+ tour on day 1)
      let motif ←
        if u = 1 then pure "commute"
        else if u = 2 then pure (if d = 1 then "tour" else "errand_loop")
        else do pure (← spec.motif.choose).name
      let motif := if motif = "" then "irregular" else motif
      let motif := if motif = "home_day" ∧ d + 1 = nDays ∧ ¬ observed
        then "irregular" else motif
      let plan ← dayPlanOf spec motif anchors.size (if u = 2 then some 2 else none)
      dayMotifs := dayMotifs.push (Json.str motif)
      if plan.isEmpty then
        continue   -- home day: device off, the day is unobserved (declared)
      observed := true
      -- departure from home: DP departure bin -> hour -> second of day
      let depCell ← spec.departure.choose
      -- planted ground-truth users leave in a fixed PUBLIC morning window
      -- (plantedDepartureH): their recall tests the encoder, so it must not
      -- hinge on DP noise — a noised departure mix with morning = 0 put the
      -- commuter's 6–12 h work stays outside the public work hours and the
      -- planted commute days were never detected
      let depR : RangeN := if u ≤ 2 then ⟨plantedDepartureH.1, plantedDepartureH.2⟩
        else rangeOf spec.depRangesH
          (if depCell.name = "" then "dep_morning" else depCell.name) ⟨7, 10⟩
      let depH ← drawRange depR
      let depMin ← drawRange ⟨0, 59⟩
      let mut tCur := day0 * 86400 + d * 86400 + depH * 3600 + depMin * 60
      if tCur ≤ tPrevEnd then
        tCur := tPrevEnd + 60
      let mut pCur : Int × Int := anchors[0]!   -- leave from home
      -- walk the plan: trip -> stay(wobble) -> trip ... (one trajectory per
      -- recorded burst; long stays switch the device off)
      let mut stayPlans : Array (String × Nat × Nat × Nat) := #[]  -- role, slot, trip_s, dwell_s
      let mut routes : Array (Array (Float × Float) × String) := #[]
      for (i, (role, slot)) in plan.zipIdx.map (fun p => (p.2, p.1)) do
        let target := anchors[slot]!
        -- the planted tour drives (fast legs = slack under tour_max_gap_s);
        -- everyone else: mode conditioned on this trip's length
        let dMm := Float.sqrt (Float.ofInt (target.1 - pCur.1) * Float.ofInt (target.1 - pCur.1)
                             + Float.ofInt (target.2 - pCur.2) * Float.ofInt (target.2 - pCur.2))
        let mode ← if u = 2 ∧ motif = "tour" then pure "car" else chooseMode spec dMm
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
          ((kinOf spec mode).2 / 10)
        for (t, x, y) in pts do
          trajPts := trajPts.push (t, x, y, modeRep mode, tLeave, s.arrive)
        -- short stays stay recorded (wobble); long stays end the trajectory
        if s.leave - s.arrive ≤ 3600 then
          let w ← renderWobble spec target s.arrive s.leave genStayDtS
          for (t, x, y) in w do
            trajPts := trajPts.push (t, x, y, "walk", s.arrive, s.leave)
        else
          -- flush the trajectory recorded so far
          if trajPts.size > 0 then
            let tid := tidStr (trajStart - Params.sourceUtcOffsetS)
            let (rows, c') := trajRows u tid trajPts seqCtr
            tsv := tsv ++ rows
            seqCtr := c'
            trajPts := #[]
          trajStart := s.leave
        let _ := dwellPlanned
        let _ := role
        pFrom := target
        tLeave := s.leave
      if trajPts.size > 0 then
        let tid := tidStr (trajStart - Params.sourceUtcOffsetS)
        let (rows, c') := trajRows u tid trajPts seqCtr
        tsv := tsv ++ rows
        seqCtr := c'
      tPrevEnd := tLeave
    planUsers := planUsers.push <| Json.mkObj
      [ ("user", Json.str (pad3 u)), ("day0", jn day0), ("n_days", jn nDays)
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
    , ("n_days_prior", Json.str "per-user: activeDayClasses (Motion.lean, public prior); planted users plantedActiveDays")
    , ("frame", Json.str "neutral-local (0N 0E isotropic); manipulate step required")
    , ("spec_version", Json.str spec.version)
    , ("window", Json.mkObj [("start_s", jn spec.windowStartS),
                             ("end_s", jn spec.windowEndS)])
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {nUsers} users, {tsv.size - 1} points -> {outDir}/geolife_synth.tsv"
  return 0

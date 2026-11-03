/-
Public structural constants of the synthetic-trajectory generator: the civil
calendar arithmetic (epoch seconds → GeoLife timestamp / trajectory-id
strings), the local-frame coordinate rendering, id rendering (uuid-shaped
sequence ids), per-mode-class rendering constants (public physics, declared
`derived_from_data: false`), and the LINKING THEOREMS that tie the
generator's parameters to the encoder's acceptance thresholds:

  * `wobble_box_fits`     — staypoint wobble is capped at `wobbleCapMm`, and
                            2·cap ≤ sp_radius, so every pair of stay points
                            lies within the detection radius: a planted stay
                            RE-FIRES Li-et-al staypoint detection;
  * `genStay_isStay`      — the generator's minimum stay dwell clears the
                            public staypoint threshold;
  * `trip_interval_le_gap`— the generator's trip-duration cap never exceeds
                            the visit-sequence break bound: a planted trip
                            never splits a visit sequence;
  * `genTrip_chains`      — the trip cap also respects the day-chain bound
                            (`tour_max_gap_s`): planted tours stay chained;
  * `render_speed_within_cap` — every mode's rendered cruise speed is inside
                            the public plausibility cap (no implausible-speed
                            quality flags on synthetic steps);
  * `modeRep_roundtrip`   — the rendered raw mode label of each class maps
                            back to that class through the public mode map.

The generator regex-reads nothing — it imports these definitions; the
kernel-checked anchors below tie the calendar to the public export window.
-/
import EnclaveGeo.Params

namespace EnclaveGeo.Shape

open EnclaveGeo

/-! ### Civil calendar (Howard Hinnant's algorithm, as in the prior builds) -/

/-- Days since 1970-01-01 for a civil date (all dates used here are far past
1970, so operands stay positive). -/
def daysFromCivil (y : Int) (m d : Nat) : Int :=
  let y' : Int := if m ≤ 2 then y - 1 else y
  let era : Int := (if 0 ≤ y' then y' else y' - 399) / 400
  let yoe : Int := y' - era * 400
  let mp : Int := (Int.ofNat m + 9) % 12
  let doy : Int := (153 * mp + 2) / 5 + Int.ofNat d - 1
  let doe : Int := yoe * 365 + yoe / 4 - yoe / 100 + doy
  era * 146097 + doe - 719468

/-- Civil date (y, m, d) from days since 1970-01-01. -/
def civilFromDays (z0 : Int) : Int × Nat × Nat :=
  let z : Int := z0 + 719468
  let era : Int := (if 0 ≤ z then z else z - 146096) / 146097
  let doe : Int := z - era * 146097
  let yoe : Int := (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
  let y : Int := yoe + era * 400
  let doy : Int := doe - (365 * yoe + yoe / 4 - yoe / 100)
  let mp : Int := (5 * doy + 2) / 153
  let d : Int := doy - (153 * mp + 2) / 5 + 1
  let m : Int := if mp < 10 then mp + 3 else mp - 9
  (if m ≤ 2 then y + 1 else y, m.toNat, d.toNat)

def pad2 (n : Nat) : String := if n < 10 then s!"0{n}" else s!"{n}"

/-- Render epoch seconds as 'YYYY-MM-DD HH:MM:SS' (the GeoLife timestamp
convention — local naive time). An assumed I/O wrapper per goal.md guideline
10; the window anchors below are kernel-checked round-trips of the same
arithmetic. -/
def dtStr (s : Nat) : String :=
  let sod := s % 86400
  let (y, m, d) := civilFromDays (Int.ofNat (s / 86400))
  s!"{y}-{pad2 m}-{pad2 d} {pad2 (sod / 3600)}:{pad2 (sod % 3600 / 60)}:{pad2 (sod % 60)}"

/-- Render epoch seconds as a GeoLife trajectory id ('yyyymmddhhmmss'). -/
def tidStr (s : Nat) : String :=
  let sod := s % 86400
  let (y, m, d) := civilFromDays (Int.ofNat (s / 86400))
  s!"{y}{pad2 m}{pad2 d}{pad2 (sod / 3600)}{pad2 (sod % 3600 / 60)}{pad2 (sod % 60)}"

/-- The calendar arithmetic round-trips on the window-start anchor
(2007-04-12T00:00:00, kernel-checked). -/
example : Params.windowStartS / 86400 = 13615 := by decide
example : civilFromDays 13615 = (2007, 4, 12) := by decide
example : daysFromCivil 2007 4 12 = 13615 := by decide

/-- ... and on the (exclusive) window-end anchor 2012-01-01T00:00:00. -/
example : Params.windowEndS / 86400 = 15340 := by decide
example : civilFromDays 15340 = (2012, 1, 1) := by decide

/-! ### Local-frame coordinates

The generator emits a NEUTRAL local frame: meters east/north of the frame
origin (0°N 0°E), rendered as degrees at the equatorial scale 1° = 111320 m
on both axes (cos 0 = 1, so the frame is isotropic by construction). The
mandatory `manipulate` step relocates each user's frame onto Tokyo public
geography; no source geography can appear because none is ever input. -/

/-- Millimeters → microdegrees in the neutral frame (1° = 111320 m). -/
def mmToMicrodeg (mm : Int) : Int := mm * 1000000 / 111320000

/-- Render microdegrees as a decimal-degree string with 6 decimals. -/
def deg6Str (micro : Int) : String :=
  let a := micro.natAbs
  let sgn := if micro < 0 then "-" else ""
  let frac := a % 1000000
  let fs := toString frac
  let pad := String.mk (List.replicate (6 - fs.length) '0')
  s!"{sgn}{a / 1000000}.{pad}{fs}"

example : deg6Str (mmToMicrodeg 111320000) = "1.000000" := by decide
example : deg6Str (-500) = "-0.000500" := by decide

/-! ### Id rendering -/

def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat (48 + n) else Char.ofNat (87 + n % 16)

def hexN (k n : Nat) : String :=
  String.mk ((List.range k).map (fun i => hexDigit ((n >>> (4 * (k - 1 - i))) % 16)))

/-- Deterministic uuid-SHAPED sequence id from a counter (matches the source
`8-4-4-4-12` hex convention; scrambled by a fixed odd constant so ids stay
distinct — a labelling device, carries no data). -/
def uuidStr (n : Nat) : String :=
  let v := (n + 1) * 0x9e3779b97f4a7c15f39cc0605cedc835 % (2 ^ 128)
  s!"{hexN 8 (v >>> 96)}-{hexN 4 ((v >>> 80) % 65536)}-{hexN 4 ((v >>> 64) % 65536)}-{hexN 4 ((v >>> 48) % 65536)}-{hexN 12 (v % (2 ^ 48))}"

example : (uuidStr 0).length = 36 := by decide

/-! ### Rendering constants (public physics; declared priors) -/

/-- Per-mode-class cruise speed for rendering, mm/s (public physics prior;
the released kinematics summaries override these when present). -/
def modeCruiseMmps (c : String) : Nat :=
  match c with
  | "walk" => 1300 | "bike" => 3500 | "bus" => 5000
  | "car" => 9000 | "rail" => 18000 | _ => 25000

/-- Hard per-step rendered speed cap, mm/s: twice cruise, clamped into the
public plausibility cap. -/
def renderSpeedCapMmps (c : String) : Nat :=
  min (2 * modeCruiseMmps c) (Params.maxSpeedMps * 1000)

/-- Representative raw GeoLife label rendered for each mode class. -/
def modeRep (c : String) : String :=
  match c with
  | "walk" => "walk" | "bike" => "bike" | "bus" => "bus"
  | "car" => "car" | "rail" => "train" | _ => "airplane"

/-- Base sampling interval of rendered points, seconds. -/
def genSampleDtS : Nat := 3
/-- Stay-wobble sampling interval, seconds (public prior). Recorded stays are
rendered SPARSELY: the labelled source records movement, and full-rate (2 s)
wobble labelled "walk" made 71% of synthetic points walk vs 29% real —
the released per-mode point counts (kinematics n_points) are the check. -/
def genStayDtS : Nat := 30
/-- The generator's minimum planted stay dwell, seconds. -/
def genStayMinDwellS : Nat := 1500
/-- The generator's trip-duration cap, seconds. -/
def genTripMaxS : Nat := 7200
/-- Recorded-days classes per synthetic user, `(lo, hi, weight)`: a
DECLARED PUBLIC PRIOR (derived_from_data: false — no per-user activity
statistic is released). GPS-logging cohorts are heavy-tailed: most
participants log a few days or weeks, a few log for months. Log-spaced
classes with decreasing weight give median ≈ 12 days, p90 ≈ 96 and mean
≈ 25 (the mean keeps the corpus-volume parity the fixed 24-day slice had).
A user's day count is a class draw, then uniform within the class. -/
def activeDayClasses : List (Nat × Nat × Nat) :=
  [(2, 4, 20), (5, 8, 20), (9, 16, 20), (17, 32, 15), (33, 64, 15), (65, 128, 10)]
/-- Longest recorded slice any user can draw (window-fit bound). -/
def maxActiveDays : Nat := 128
/-- Planted ground-truth users keep a fixed slice: the revisit beacon needs
≥ `revisitMinDays` visits inside a `revisitWindowDays` window. -/
def plantedActiveDays : Nat := 24
/-- Planted users' departure-hour window `(lo, hi)` (public prior, a morning
commute): leaving 7–8 h, a ≤ 2 h trip and a ≥ 6 h work dwell keep the work
stay inside the public work window, so planted commute recall cannot depend
on the (noisy) released departure mix. -/
def plantedDepartureH : Nat × Nat := (7, 8)
/-- Staypoint wobble cap: planted stay points never leave this radius (mm). -/
def wobbleCapMm : Nat := 50000
/-- A forced turn at least every `maxCollinearRun` rendered steps. -/
def maxCollinearRun : Nat := 5
/-- Minimum forced-turn magnitude, centidegrees (so no rendered leg is a
straight line, and jitter keeps turns off the 0/±90/180 lattice). -/
def minTurnCd : Nat := 200

/-! ### Route-shape rendering (road-like trips from released statistics) -/

/-- Public range of a segment-length class, metres (the open top bin is
capped for rendering; declared). -/
def segRangeOf (c : String) : Nat × Nat :=
  match c with
  | "seg_lt50" => (Params.routeResampleM, Params.segE1M - 1)
  | "seg_50_150" => (Params.segE1M, Params.segE2M - 1)
  | "seg_150_600" => (Params.segE2M, Params.segE3M - 1)
  | _ => (Params.segE3M, 1500)

/-- Public range of a junction-angle class, absolute degrees (the top bin is
capped at 175° — a rendered junction never doubles exactly back). -/
def jaRangeOf (c : String) : Nat × Nat :=
  match c with
  | "ja_30_60" => (Params.junctionMinTurnDeg, Params.jaE1Deg - 1)
  | "ja_60_120" => (Params.jaE1Deg, Params.jaE2Deg - 1)
  | _ => (Params.jaE2Deg, 175)

/-- Fuel bound on segments per rendered trip polyline (public). -/
def maxTripSegments : Nat := 48
/-- Default circuity budget when a mode has no released value (permille). -/
def defaultCircPm : Nat := 1400

/-- Every rendered junction angle is a REAL turn (at least the public
junction threshold) and never a degenerate reversal: the encoder's junction
detector re-fires on every planted junction. -/
theorem jaRange_is_junction :
    (["ja_30_60", "ja_60_120", "ja_120_180"].all
      (fun c => decide (Params.junctionMinTurnDeg ≤ (jaRangeOf c).1
                        ∧ (jaRangeOf c).2 < 180))) = true := by decide

/-- Every rendered segment survives the route resampling grid (segment
lengths never fall below `route_resample_m`), so re-detection sees the same
segment structure. -/
theorem segRange_resolvable :
    (["seg_lt50", "seg_50_150", "seg_150_600", "seg_ge600"].all
      (fun c => decide (Params.routeResampleM ≤ (segRangeOf c).1
                        ∧ (segRangeOf c).1 ≤ (segRangeOf c).2))) = true := by
  decide

/-! ### Linking theorems (generator parameters ⇒ encoder acceptance) -/

/-- Planted stays clear the public staypoint dwell threshold. -/
theorem genStay_isStay : Params.spMinDwellS ≤ genStayMinDwellS := by decide

/-- Stay wobble fits the detection box: any two points of a planted stay are
within `2·wobbleCap ≤ sp_radius` of each other (and of the first point), so
Li-et-al detection re-fires on the synthetic stream. -/
theorem wobble_box_fits : 2 * wobbleCapMm ≤ Params.spRadiusM * 1000 := by decide

/-- A planted trip never breaks a visit sequence. -/
theorem trip_interval_le_gap : genTripMaxS ≤ Params.visitGapBreakS := by decide

/-- Every active-day class is a non-empty range of ≥ 1 day inside the
window-fit bound, with positive weight (so every class is drawable). -/
theorem activeDayClasses_valid :
    activeDayClasses.all (fun c => decide (1 ≤ c.1 ∧ c.1 ≤ c.2.1
      ∧ c.2.1 ≤ maxActiveDays ∧ 0 < c.2.2)) = true := by decide

/-- The planted users' slice covers a full revisit-beacon window. -/
theorem planted_slice_covers_beacon :
    Params.revisitWindowDays ≤ plantedActiveDays
      ∧ plantedActiveDays ≤ maxActiveDays := by decide

/-- A planted commuter leaving inside `plantedDepartureH` arrives at work
(trip ≤ `genTripMaxS`) no later than the public work-window start + 2 h,
so a 6 h+ work dwell overlaps the work window by ≥ 4 h. -/
theorem planted_departure_reaches_work_window :
    plantedDepartureH.1 ≤ plantedDepartureH.2
      ∧ plantedDepartureH.2 * 3600 + 3600 + genTripMaxS
          ≤ (Params.workStartH + 2) * 3600 := by decide

/-- A planted tour-chain trip respects the day-chain bound. -/
theorem genTrip_chains : genTripMaxS ≤ Params.tourMaxGapS := by decide

/-- Every rendered cruise speed is inside the public plausibility cap. -/
theorem render_speed_within_cap :
    Params.modeClasses.all
      (fun c => renderSpeedCapMmps c ≤ Params.maxSpeedMps * 1000) = true := by
  decide

/-- Every rendered raw label maps back to its class through the public map. -/
theorem modeRep_roundtrip :
    Params.modeClasses.all
      (fun c => Params.modeClassOf (modeRep c) = c) = true := by
  decide

/-- The wobble cap stays inside the tight-noise acceptance bound scale (the
released tight residual p95 is measured in meters; cap/1000 ≤ radius/2). -/
theorem wobble_le_half_radius : wobbleCapMm ≤ Params.spRadiusM * 1000 / 2 := by
  decide

end EnclaveGeo.Shape

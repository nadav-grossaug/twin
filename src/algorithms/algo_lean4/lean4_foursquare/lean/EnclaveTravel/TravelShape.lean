/-
Public structural constants of the synthetic-travel generator: the civil
calendar arithmetic (epoch seconds → 'YYYY-MM-DD HH:MM:SS'), the public
country-centroid table (approximate geography, one entry per region_map
country — common-knowledge data, declared `derived_from_data: false`), the
per-region representative countries (declared prior), and venue-id rendering.
The generator regex-reads nothing — it imports these definitions; the
kernel-checked anchors below tie the calendar to the public export window.
-/
import EnclaveTravel.Params

namespace EnclaveTravel.Shape

open EnclaveTravel

/-! ### Civil calendar (Howard Hinnant's algorithm, as in the stock twin) -/

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

/-- Render epoch seconds as 'YYYY-MM-DD HH:MM:SS' (UTC). An assumed I/O
wrapper per goal.md guideline 10; the window anchors below are
kernel-checked round-trips of the same arithmetic. -/
def dtStr (s : Nat) : String :=
  let sod := s % 86400
  let (y, m, d) := civilFromDays (Int.ofNat (s / 86400))
  s!"{y}-{pad2 m}-{pad2 d} {pad2 (sod / 3600)}:{pad2 (sod % 3600 / 60)}:{pad2 (sod % 60)}"

/-- The calendar arithmetic round-trips on the window-start anchor
(2012-04-04T00:00:00Z, kernel-checked). -/
example : Params.windowStartS / 86400 = 15434 := by decide
example : civilFromDays 15434 = (2012, 4, 4) := by decide
example : daysFromCivil 2012 4 4 = 15434 := by decide

/-- ... and on the (exclusive) window-end anchor 2013-09-17T00:00:00Z. -/
example : Params.windowEndS / 86400 = 15965 := by decide
example : civilFromDays 15965 = (2013, 9, 17) := by decide

/-! ### Public country centroids (approximate, centidegrees)

One entry per region_map country: (lat, lon) in 1/100 degree. Public
geography, declared `derived_from_data: false` — the synthetic coordinates
carry no information about any user. -/

/-- Approximate centroid of a country, (lat, lon) in centidegrees. -/
def centroidOf (c : String) : Int × Int :=
  match c with
  | "US" => (3980, -9860) | "CA" => (5610, -10630)
  | "AR" => (-3840, -6360) | "BR" => (-1420, -5190) | "CL" => (-3570, -7150)
  | "CO" => (460, -7430) | "CR" => (970, -8380) | "DO" => (1870, -7020)
  | "EC" => (-180, -7820) | "JM" => (1810, -7730) | "MQ" => (1460, -6100)
  | "MX" => (2360, -10260) | "PA" => (850, -8080) | "PE" => (-920, -7500)
  | "PR" => (1820, -6640) | "PY" => (-2340, -5840) | "SV" => (1380, -8890)
  | "TT" => (1070, -6120) | "UY" => (-3250, -5580) | "VE" => (640, -6660)
  | "AT" => (4750, 1460) | "BE" => (5050, 450) | "BG" => (4270, 2550)
  | "BY" => (5370, 2790) | "CH" => (4680, 820) | "CY" => (3510, 3340)
  | "CZ" => (4980, 1550) | "DE" => (5120, 1040) | "DK" => (5630, 950)
  | "EE" => (5860, 2500) | "ES" => (4050, -370) | "FI" => (6190, 2570)
  | "FR" => (4620, 220) | "GB" => (5400, -290) | "GR" => (3910, 2180)
  | "HU" => (4720, 1950) | "IE" => (5340, -820) | "IT" => (4190, 1260)
  | "LV" => (5690, 2460) | "NL" => (5210, 530) | "PL" => (5190, 1910)
  | "PT" => (3940, -820) | "RO" => (4590, 2500) | "RU" => (6150, 10530)
  | "SE" => (6010, 1860) | "UA" => (4840, 3120)
  | "AE" => (2340, 5380) | "BH" => (2600, 5060) | "EG" => (2680, 3080)
  | "IL" => (3100, 3490) | "JO" => (3060, 3620) | "KW" => (2930, 4750)
  | "LB" => (3390, 3590) | "MA" => (3180, -710) | "OM" => (2150, 5590)
  | "QA" => (2540, 5120) | "SA" => (2390, 4510) | "TN" => (3390, 950)
  | "TR" => (3900, 3520)
  | "GH" => (790, -100) | "KE" => (20, 3790) | "ZA" => (-3060, 2290)
  | "AZ" => (4010, 4760) | "IN" => (2060, 7900) | "KZ" => (4800, 6690)
  | "LK" => (790, 8080)
  | "CN" => (3590, 10420) | "ID" => (-80, 11390) | "JP" => (3620, 13830)
  | "KR" => (3590, 12780) | "MY" => (420, 10200) | "PH" => (1290, 12180)
  | "SG" => (135, 10380) | "TH" => (1590, 10100) | "VN" => (1410, 10830)
  | "AU" => (-2530, 13380) | "NZ" => (-4090, 17490)
  | _ => (0, 0)

/-- Render a centidegree coordinate as a decimal string (e.g. `-98.60`). -/
def coordStr (v : Int) : String :=
  let a := v.natAbs
  let sgn := if v < 0 then "-" else ""
  s!"{sgn}{a / 100}.{pad2 (a % 100)}"

/-! ### Per-region representative countries (declared public prior) -/

/-- The public fallback country of a region (used when no released corridor
provides one — a declared prior, not data). -/
def regionRep (r : String) : String :=
  match r with
  | "NA" => "US" | "LA" => "BR" | "EU" => "FR" | "ME" => "TR"
  | "AF" => "ZA" | "SA" => "IN" | "EA" => "JP" | "OC" => "AU"
  | _ => "US"

/-- Every representative maps back to its region (kernel-checked coherence
with the mirrored region_map). -/
example : Params.regions.all (fun r => Params.regionOf (regionRep r) = r) = true := by
  decide

/-! ### Venue ids -/

def hexDigit (n : Nat) : Char :=
  if n < 10 then Char.ofNat (48 + n) else Char.ofNat (87 + n % 16)

/-- Deterministic 24-hex venue id from a counter (matches `[0-9a-f]{24}`;
a labelling device, carries no data). -/
def hex24 (n : Nat) : String :=
  String.mk ((List.range 24).map (fun i => hexDigit ((n >>> (4 * (23 - i))) % 16)))

example : hex24 0xdeadbeef = "0000000000000000deadbeef" := by decide
example : (hex24 0).length = 24 := by decide

end EnclaveTravel.Shape

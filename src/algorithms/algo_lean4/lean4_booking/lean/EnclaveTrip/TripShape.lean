/-
Public structural constants of the synthetic-trip generator: the civil
calendar arithmetic (epoch days → 'YYYY-MM-DD'), the 2016 month anchors
(the generator's start-month placement prior: every checkin month maps to
calendar year 2016, which lies wholly inside the public window), the
synthetic city-id scheme (a labelling device carrying no data), and the
kernel-checked window anchors. The generator regex-reads nothing — it
imports these definitions.
-/
import EnclaveTrip.Params

namespace EnclaveTrip.Shape

open EnclaveTrip

/-! ### Civil calendar (Howard Hinnant's algorithm) - days only -/

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

/-- Render an epoch day as 'YYYY-MM-DD' (UTC). An assumed I/O wrapper per
goal.md guideline 10; the window anchors below are kernel-checked
round-trips of the same arithmetic. -/
def dateStr (d : Nat) : String :=
  let (y, m, dd) := civilFromDays (Int.ofNat d)
  s!"{y}-{pad2 m}-{pad2 dd}"

/-- The calendar arithmetic round-trips on the window-start anchor
(2015-12-01, kernel-checked). -/
example : daysFromCivil 2015 12 1 = 16770 := by decide
example : civilFromDays 16770 = (2015, 12, 1) := by decide

/-- ... and on the (exclusive) window-end anchor 2017-03-01. -/
example : daysFromCivil 2017 3 1 = 17226 := by decide
example : civilFromDays 17226 = (2017, 3, 1) := by decide

/-! ### Month anchors (declared prior: months map to year 2016)

2016 is the bulk year of the public window and spans it fully, so the
drawn checkin-month marginal maps to real dates without leaving the
window. -/

/-- Epoch day of the 1st of month `m` (1..12) of 2016. -/
def monthStartDay (m : Nat) : Nat := (daysFromCivil 2016 m 1).toNat

example : monthStartDay 1 = 16801 := by decide
example : monthStartDay 12 = 17136 := by decide
example : monthStartDay 1 ≥ Params.windowStartDay := by decide
example : monthStartDay 12 + 27 < Params.windowEndDay := by decide

/-! ### Month-of-window anchors (v1.2.0: the released calendar)

The month-of-window release (15 cells mw00..mw14) carries the REAL
observation ramp — the window's own calendar months 2015-12 .. 2017-02 —
so trip starts are placed on the true shape instead of a flattened
month-of-year marginal. Every cell's month starts inside the window and
its day-of-month range (1..28) fits before the window end. -/

/-- Epoch day of the 1st of month-of-window cell `i`
(mw00 = 2015-12-01 = window start; mw14 = 2017-02-01). -/
def monthWindowStartDay (i : Nat) : Nat :=
  let mi := 11 + i                    -- months since 2015-01
  (daysFromCivil (2015 + mi / 12) (mi % 12 + 1) 1).toNat

example : monthWindowStartDay 0 = 16770 := by decide
example : monthWindowStartDay 1 = 16801 := by decide
example : monthWindowStartDay 14 = 17198 := by decide
example : monthWindowStartDay 14 + 27 < Params.windowEndDay := by decide

/-! ### Synthetic city ids

Real city ids are 5-digit-ish anonymized integers; synthetic ids are a
counter scrambled by a fixed odd constant into the same range. A labelling
device: ids carry no data (they exist so released ids never collide with
synthetic ones outside the released list by accident of ordering). -/

/-- Deterministic synthetic city id from a counter: scrambled into the
realistic 10000..99999 range (declared labelling device, carries no data). -/
def cityId (n : Nat) : Nat := 10000 + (n * 2654435761) % 90000

example : 10000 ≤ cityId 0 ∧ cityId 0 ≤ 99999 := by decide
example : 10000 ≤ cityId 12345 ∧ cityId 12345 ≤ 99999 := by decide

end EnclaveTrip.Shape

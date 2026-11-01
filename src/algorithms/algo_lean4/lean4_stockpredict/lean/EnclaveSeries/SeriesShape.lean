/-
Public structural constants of the synthetic-series generator, plus the
trading calendar. Everything here is declared `derived_from_data: false`:
fixed magnitude-bin edges nested in the public class taxonomy, the vol-state
boost table (the discretized GARCH diffusion prior), market-coupling priors,
and the weekday calendar arithmetic. The generator regex-reads nothing — it
imports these definitions; the linking theorems (`genRet_class_*`) prove the
bin → class contract the per-column tests rely on.
-/
import EnclaveSeries.Motifs

namespace EnclaveSeries.Shape

open EnclaveSeries

/-! ### Magnitude bins (bp), nested inside the 5-class taxonomy -/

/-- Inclusive lower bin edges (|log return| in bp). -/
def magLoBp : List Nat := [0, 51, 151, 300, 600, 1200]
/-- Inclusive upper bin edges. -/
def magHiBp : List Nat := [50, 150, 299, 599, 1199, 3000]

/-- Bins compatible with each class (0=crash 1=down 2=flat 3=up 4=surge):
flat draws from mag_0; down/up from mag_1-2; crash/surge from mag_3-5. -/
def classBins : Nat → List Nat
  | 0 => [3, 4, 5]
  | 1 => [1, 2]
  | 2 => [0]
  | 3 => [1, 2]
  | _ => [3, 4, 5]

/-- Vol-state boost over the 6 magnitude bins (rows: calm, normal, elevated,
burst) — the public prior that makes magnitudes track the sampled vol state
(discretized GARCH diffusion). -/
def volBoost : List (List Nat) :=
  [ [6, 4, 2, 1, 0, 0]
  , [4, 4, 3, 2, 1, 0]
  , [2, 3, 4, 3, 2, 1]
  , [1, 2, 3, 4, 4, 3] ]

/-- Market-mode persistence prior (keep : switch), public. -/
def marketKeepW : Nat := 7
def marketSwitchW : Nat := 3

/-! ### Bin → class contract (validity of generated returns by construction) -/

theorem genRet_class_surge (m : ℤ) (h1 : 300 ≤ m) :
    classOfRet 50 300 m = 4 := by
  unfold classOfRet; split_ifs <;> omega

theorem genRet_class_crash (m : ℤ) (h1 : 300 ≤ m) :
    classOfRet 50 300 (-m) = 0 := by
  unfold classOfRet; split_ifs <;> omega

theorem genRet_class_up (m : ℤ) (h1 : 51 ≤ m) (h2 : m ≤ 299) :
    classOfRet 50 300 m = 3 := by
  unfold classOfRet; split_ifs <;> omega

theorem genRet_class_down (m : ℤ) (h1 : 51 ≤ m) (h2 : m ≤ 299) :
    classOfRet 50 300 (-m) = 1 := by
  unfold classOfRet; split_ifs <;> omega

theorem genRet_class_flat (m : ℤ) (h1 : -50 ≤ m) (h2 : m ≤ 50) :
    classOfRet 50 300 m = 2 := by
  unfold classOfRet; split_ifs <;> omega

/-! ### Trading calendar (weekdays from the public anchor Monday) -/

/-- Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm;
all dates used here are far past 1970, so operands stay positive). -/
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

/-- Public anchor: Monday 2025-10-06 (the dataset's first trading day,
common-knowledge metadata). -/
def anchorDays : Int := daysFromCivil 2025 10 6

/-- Trading day `n` (0-based) maps to anchor + full weeks + weekday offset:
weekends are skipped by construction; holidays are declared not modelled. -/
def tradingDayNum (n : Nat) : Int := anchorDays + 7 * (n / 5) + (n % 5)

/-- Day-of-week index of trading day `n` (0=Mon .. 4=Fri). -/
def dowOf (n : Nat) : Nat := n % 5

/-- The calendar arithmetic round-trips on the anchor (kernel-checked). -/
example : civilFromDays (daysFromCivil 2025 10 6) = (2025, 10, 6) := by decide

/-- The anchor is a Monday: 1970-01-01 was a Thursday, so weekday =
(days + 4) % 7 with 1 = Monday. -/
example : (daysFromCivil 2025 10 6 + 4) % 7 = 1 := by decide

/-- Generated trading-day numbers are strictly increasing. -/
theorem tradingDayNum_strictMono {a b : Nat} (h : a < b) :
    tradingDayNum a < tradingDayNum b := by
  unfold tradingDayNum
  omega

end EnclaveSeries.Shape

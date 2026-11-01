/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 1.0.0).

These numbers are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (with the unit scalings named there). Everything else in this library
references these definitions, so one numeric literal is shared by the
proofs, the executables, Python and SQL.
-/

namespace EnclaveSeries.Params

def specVersion : String := "1.0.0"

/-- ret_flat_bp (|log return| ≤ flat ⇒ 'flat' class; basis points) -/
def retFlatBp : Nat := 50
/-- ret_big_bp (|log return| ≥ big ⇒ 'crash'/'surge' class) -/
def retBigBp : Nat := 300
/-- shock_bp -/
def shockBp : Nat := 1000
/-- run_min_days -/
def runMinDays : Nat := 5
/-- drawdown_num / drawdown_den (0.20 peak → trough) -/
def drawdownNum : Nat := 1
def drawdownDen : Nat := 5
/-- vrecov_drop_num / vrecov_drop_den (0.10 drop from the peak) -/
def vrecovDropNum : Nat := 1
def vrecovDropDen : Nat := 10
/-- vrecov_level_num / vrecov_level_den (recovery to 0.95 of the peak) -/
def vrecovLevelNum : Nat := 19
def vrecovLevelDen : Nat := 20
/-- vrecov_max_days -/
def vrecovMaxDays : Nat := 20
/-- volburst_window -/
def volburstWindow : Nat := 5
/-- volburst_min_bp (5-day sample sd of daily log returns, bp) -/
def volburstMinBp : Nat := 400
/-- gap_max_days (calendar gap breaking a ticker's sequence) -/
def gapMaxDays : Nat := 4
/-- vol_calm_max_bp / vol_normal_max_bp / vol_elevated_max_bp (vol-state grid) -/
def volCalmMaxBp : Nat := 150
def volNormalMaxBp : Nat := 300
def volElevatedMaxBp : Nat := 450
/-- dp_clip -/
def dpClip : Nat := 4
/-- dp_clip_total (class-mix L1 contribution budget per ticker) -/
def dpClipTotal : Nat := 128
/-- dp_comove_clip (each ticker contributes 1 to one co-movement share bin) -/
def dpComoveClip : Nat := 1
/-- comove_lo_pct / comove_hi_pct (public co-movement bin edges) -/
def comoveLoPct : Nat := 40
def comoveHiPct : Nat := 70

end EnclaveSeries.Params

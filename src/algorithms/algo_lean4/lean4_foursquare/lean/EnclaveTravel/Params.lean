/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 2.0.0) and
the ISO->region map in dbt/seeds/region_map.csv.

These values are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (and each `| "CC" => "RR"` match arm equals the seed row). Everything
else in this library references these definitions, so one literal is shared
by the proofs, the executables, Python and SQL.
-/

namespace EnclaveTravel.Params

def specVersion : String := "2.0.0"

/-- gap_hop_max_s (< 2d -> gap_hop) -/
def gapHopMaxS : Nat := 172800
/-- gap_trip_max_s (< 7d -> gap_trip) -/
def gapTripMaxS : Nat := 604800
/-- gap_monthly_max_s (< 30d -> gap_monthly) -/
def gapMonthlyMaxS : Nat := 2592000
/-- gap_quarterly_max_s (< 90d -> gap_quarterly, else gap_rare) -/
def gapQuarterlyMaxS : Nat := 7776000
/-- dur_day_max_s -/
def durDayMaxS : Nat := 86400
/-- dur_week_max_s -/
def durWeekMaxS : Nat := 604800
/-- dur_month_max_s -/
def durMonthMaxS : Nat := 2592000
/-- dur_quarter_max_s -/
def durQuarterMaxS : Nat := 7776000
/-- dist_short_max_km -/
def distShortMaxKm : Nat := 500
/-- dist_mid_max_km -/
def distMidMaxKm : Nat := 2000
/-- dist_long_max_km -/
def distLongMaxKm : Nat := 7000
/-- tc_e2 .. tc_e33 (travels-per-user bin lower edges) -/
def tcE2 : Nat := 2
def tcE3 : Nat := 3
def tcE5 : Nat := 5
def tcE9 : Nat := 9
def tcE17 : Nat := 17
def tcE33 : Nat := 33
/-- episode_gap_max_s (gaps <= 14d chain one journey) -/
def episodeGapMaxS : Nat := 1209600
/-- ep_e4 / ep_e6 (episode-length bin lower edges) -/
def epE4 : Nat := 4
def epE6 : Nat := 6
/-- corridor_min_repeats -/
def corridorMinRepeats : Nat := 5
/-- corridor_cv_max_num / corridor_cv_max_den (gap CV <= 4/5) -/
def corridorCvMaxNum : Nat := 4
def corridorCvMaxDen : Nat := 5
/-- tour_min_hops -/
def tourMinHops : Nat := 3
/-- tour_max_gap_s -/
def tourMaxGapS : Nat := 604800
/-- tour_min_countries -/
def tourMinCountries : Nat := 3
/-- burst_min_transfers -/
def burstMinTransfers : Nat := 3
/-- burst_window_s -/
def burstWindowS : Nat := 1209600
/-- longstay_min_s -/
def longstayMinS : Nat := 7776000
/-- seqbreak_gap_s (gaps > 90d break sequences) -/
def seqbreakGapS : Nat := 7776000
/-- rr_band_width_pct (per-user return-rate bands, uniform width %) -/
def rrBandWidthPct : Nat := 20
/-- topk_corridor_min_users -/
def topkCorridorMinUsers : Nat := 50
/-- topk_corridor_min_transfers -/
def topkCorridorMinTransfers : Nat := 100
/-- dp_clip_total (per-user L1 budget per histogram) -/
def dpClipTotal : Nat := 8
/-- dp_count_clip (one cell per user histograms) -/
def dpCountClip : Nat := 1
/-- window_start_s (2012-04-04T00:00:00Z) -/
def windowStartS : Nat := 1333497600
/-- window_end_s (2013-09-17T00:00:00Z, exclusive) -/
def windowEndS : Nat := 1379376000

/-- The 8 public world regions (the DP grid axis; fixed order). -/
def regions : List String := ["NA", "LA", "EU", "ME", "AF", "SA", "EA", "OC"]

/-- Mirror of dbt/seeds/region_map.csv: ISO 3166 country -> public region.
    Unknown codes map to "XX" (never a release cell; checked, not assumed). -/
def regionOf (c : String) : String :=
  match c with
  | "US" => "NA" | "CA" => "NA"
  | "AR" => "LA" | "BR" => "LA" | "CL" => "LA" | "CO" => "LA" | "CR" => "LA"
  | "DO" => "LA" | "EC" => "LA" | "JM" => "LA" | "MQ" => "LA" | "MX" => "LA"
  | "PA" => "LA" | "PE" => "LA" | "PR" => "LA" | "PY" => "LA" | "SV" => "LA"
  | "TT" => "LA" | "UY" => "LA" | "VE" => "LA"
  | "AT" => "EU" | "BE" => "EU" | "BG" => "EU" | "BY" => "EU" | "CH" => "EU"
  | "CY" => "EU" | "CZ" => "EU" | "DE" => "EU" | "DK" => "EU" | "EE" => "EU"
  | "ES" => "EU" | "FI" => "EU" | "FR" => "EU" | "GB" => "EU" | "GR" => "EU"
  | "HU" => "EU" | "IE" => "EU" | "IT" => "EU" | "LV" => "EU" | "NL" => "EU"
  | "PL" => "EU" | "PT" => "EU" | "RO" => "EU" | "RU" => "EU" | "SE" => "EU"
  | "UA" => "EU"
  | "AE" => "ME" | "BH" => "ME" | "EG" => "ME" | "IL" => "ME" | "JO" => "ME"
  | "KW" => "ME" | "LB" => "ME" | "MA" => "ME" | "OM" => "ME" | "QA" => "ME"
  | "SA" => "ME" | "TN" => "ME" | "TR" => "ME"
  | "GH" => "AF" | "KE" => "AF" | "ZA" => "AF"
  | "AZ" => "SA" | "IN" => "SA" | "KZ" => "SA" | "LK" => "SA"
  | "CN" => "EA" | "ID" => "EA" | "JP" => "EA" | "KR" => "EA" | "MY" => "EA"
  | "PH" => "EA" | "SG" => "EA" | "TH" => "EA" | "VN" => "EA"
  | "AU" => "OC" | "NZ" => "OC"
  | _ => "XX"

end EnclaveTravel.Params

/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 1.0.0) and
the tier maps in dbt/seeds/country_tiers.csv + dbt/seeds/affiliate_tiers.csv.

These values are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (and each tier-list membership equals the seed rows). Everything
else in this library references these definitions, so one literal is shared
by the proofs, the executables, Python and SQL.

Time unit: DAYS (epoch day numbers, 1970-01-01 = 0). The public window is
[2015-12-01, 2017-03-01) — day 16770 .. 17226 exclusive.
-/

namespace EnclaveTrip.Params

def specVersion : String := "1.3.0"

/-- rc_e4 .. rc_e13 (reservations-per-user bin lower edges) + rc_max -/
def rcE4 : Nat := 4
def rcE5 : Nat := 5
def rcE6 : Nat := 6
def rcE7 : Nat := 7
def rcE9 : Nat := 9
def rcE13 : Nat := 13
/-- rc_max (public cap of the rc_13p bin range) -/
def rcMax : Nat := 64
/-- rc_13p_draw_hi: upper edge of the rc_13p DRAW range (~p99 of the
measured per-user reservation counts). Uniform draws over [13, rcMax] put
the synthetic within-bin mean at 38.5 vs the real ~17-20, inflating the
records/user mean; the bin still COVERS 13+ (membership uncapped).
spec 1.1.0. -/
def rcDrawHi : Nat := 24
/-- tp_draw_hi: upper edge of the tp_3p (trips-per-user) draw range
(spec 1.3.0; measured max 27 trips, p99 ~ 6). -/
def tpDrawHi : Nat := 6
/-- intertrip_gap_max_d: declared prior for the days between a user's
consecutive trips — uniform over [1, intertrip_gap_max_d], median-matched
to the measured 47 days (IQR 4-117). spec 1.3.0. -/
def intertripGapMaxD : Nat := 94
/-- nights_e4 (stays of <= 3 nights are the exact bins n_1..n_3) -/
def nightsE4 : Nat := 4
/-- nights_e7 (>= 7 nights -> n_7p) -/
def nightsE7 : Nat := 7
/-- nights_max (public cap of the n_7p bin range) -/
def nightsMax : Nat := 30
/-- gap_e2 (gap 0 -> g_0, 1 -> g_1, 2-3 -> g_2_3) -/
def gapE2 : Nat := 2
/-- gap_e4 (>= 4 -> g_4p) -/
def gapE4 : Nat := 4
/-- gap_max (public cap of the g_4p bin range) -/
def gapMax : Nat := 33
/-- corridor_min_repeats -/
def corridorMinRepeats : Nat := 3
/-- corridor_cv_max_num / corridor_cv_max_den (gap CV <= 4/5) -/
def corridorCvMaxNum : Nat := 4
def corridorCvMaxDen : Nat := 5
/-- tour_min_countries -/
def tourMinCountries : Nat := 3
/-- tour_min_legs -/
def tourMinLegs : Nat := 4
/-- tour_max_gap_d (consecutive checkin-day gaps within a tour) -/
def tourMaxGapD : Nat := 7
/-- longstay_min_nights (a stay of >= 7 nights is a long stay) -/
def longstayMinNights : Nat := 7
/-- seqbreak_gap_d (consecutive checkin gaps > 14d are not well-formed
trip steps; the majority flow is contiguous, gaps of 0) -/
def seqbreakGapD : Nat := 14
/-- rr_band_width_pct (per-user repeat-city trip-share bands, %) -/
def rrBandWidthPct : Nat := 20
/-- topk_city_min_users / topk_city_min_res (suppression for topk_cities) -/
def topkCityMinUsers : Nat := 50
def topkCityMinRes : Nat := 100
/-- topk_trans_min_users / topk_trans_min_trans (suppression for
topk_city_transitions) -/
def topkTransMinUsers : Nat := 50
def topkTransMinTrans : Nat := 100
/-- dp_clip_total (per-user L1 budget per histogram) -/
def dpClipTotal : Nat := 8
/-- dp_count_clip (one cell per user histograms) -/
def dpCountClip : Nat := 1
/-- window_start_day (2015-12-01) -/
def windowStartDay : Nat := 16770
/-- window_end_day (2017-03-01, exclusive) -/
def windowEndDay : Nat := 17226

/-- The 7 public reservations-per-user bins (the MAIN DP histogram's grid,
in the fixed order the release iterates). -/
def rcBins : List String :=
  ["rc_1_3", "rc_4", "rc_5", "rc_6", "rc_7_8", "rc_9_12", "rc_13p"]

/-- The 16 public hotel-country tiers: the 15 named countries of the
country_tiers seed (themselves as tiers) plus "ct_other". Fixed order. -/
def ctCells : List String := ["Gondal", "Cobra Island", "Fook Island", "Glubbdubdrib", "Elbonia", "Borginia", "Kangan", "Bozatta", "Alvonia", "Holy Britannian Empire", "Axphain", "Santa Prisca", "Patusan", "Rolisica", "Kasnia", "ct_other"]

/-- The 5 public booker-country cells (each user falls in exactly one). -/
def bookerCells : List String :=
  ["Gondal", "The Devilfire Empire", "Elbonia", "Tcherkistan", "Bartovia"]

/-- The 3 public device-class cells. -/
def deviceCells : List String := ["desktop", "mobile", "tablet"]

/-- The 9 public affiliate tiers: the 8 named affiliates of the
affiliate_tiers seed plus "aff_other". Fixed order. -/
def affCells : List String := ["a9924", "a359", "a384", "a9452", "a4541", "a8436", "a10332", "a3631", "aff_other"]

/-- The 5 public stay-nights bins. -/
def nightsCells : List String := ["n_1", "n_2", "n_3", "n_4_6", "n_7p"]

/-- The 4 public inter-leg gap bins. -/
def gapCells : List String := ["g_0", "g_1", "g_2_3", "g_4p"]

/-- The 12 public checkin-month cells. -/
def monthCells : List String :=
  ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
   "m11", "m12"]

/-- The 7 public checkin day-of-week cells. -/
def dowCells : List String := ["d0", "d1", "d2", "d3", "d4", "d5", "d6"]

/-- The 5 public repeat-city trip-share bands. -/
def rrBands : List String :=
  ["rr_0_19", "rr_20_39", "rr_40_59", "rr_60_79", "rr_80_100"]

/-- The public tier of a hotel country: itself when named in the seed,
"ct_other" otherwise. -/
def ctOf (c : String) : String :=
  if ctCells.contains c && c != "ct_other" then c else "ct_other"

/-- Every tier maps to itself under `ctOf` (kernel-checked coherence with
the seed: the 15 named countries are their own tiers, and the suppressed
tail maps to "ct_other"). -/
example : ctCells.all (fun c => ctOf c == c) = true := by
  decide

end EnclaveTrip.Params

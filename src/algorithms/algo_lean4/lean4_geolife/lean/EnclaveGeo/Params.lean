/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 4.1.0) and
the raw-label->mode-class map in dbt/seeds/mode_map.csv.

These values are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (and each `| "raw" => "class"` match arm equals the seed row). Everything
else in this library references these definitions, so one literal is shared
by the proofs, the executables, Python and SQL.
-/

namespace EnclaveGeo.Params

def specVersion : String := "4.1.0"

/-- source_utc_offset_s (source timestamps are UTC; staging shifts them to
Beijing local time by this offset, the generator shifts back on output) -/
def sourceUtcOffsetS : Nat := 28800

/-- sp_radius_m (staypoint radius, meters) -/
def spRadiusM : Nat := 200
/-- sp_min_dwell_s (staypoint minimum dwell) -/
def spMinDwellS : Nat := 1200
/-- sp_resample_s (minimum spacing of kept points) -/
def spResampleS : Nat := 10
/-- sp_stream_break_s (recording gap that breaks a point stream) -/
def spStreamBreakS : Nat := 1800
/-- implied_stay_max_gap_s (inter-trajectory gap that still counts as a stay) -/
def impliedStayMaxGapS : Nat := 604800
/-- noise_tight_max_m (residual p95 bound for noise-moment staypoints) -/
def noiseTightMaxM : Nat := 50
/-- anchor_cell_m (anchor grid cell size, meters) -/
def anchorCellM : Nat := 300
/-- anchor_work_min_s (minimum work-hours dwell for a work anchor) -/
def anchorWorkMinS : Nat := 36000
/-- night_anchor_hour (the sleep hour the home test contains) -/
def nightAnchorHour : Nat := 3
/-- work_start_h / work_end_h (weekday work window, local hours) -/
def workStartH : Nat := 9
def workEndH : Nat := 17
/-- visit_gap_break_s (observation gap that breaks a visit sequence) -/
def visitGapBreakS : Nat := 21600
/-- dw_e1_s .. dw_e4_s (dwell-class upper edges, seconds) -/
def dwE1S : Nat := 3600
def dwE2S : Nat := 10800
def dwE3S : Nat := 28800
def dwE4S : Nat := 50400
/-- dep_night_max_h / dep_morning_max_h / dep_afternoon_max_h (hour bins) -/
def depNightMaxH : Nat := 5
def depMorningMaxH : Nat := 11
def depAfternoonMaxH : Nat := 16
/-- spd_still_max_mmps .. spd_fast_max_mmps (speed-class edges, milli-m/s) -/
def spdStillMaxMmps : Nat := 500
def spdWalkMaxMmps : Nat := 2000
def spdCityMaxMmps : Nat := 8000
def spdFastMaxMmps : Nat := 20000
/-- max_speed_mps (physical plausibility cap, public physics) -/
def maxSpeedMps : Nat := 120
/-- min_move_m (minimum displacement that counts as movement) -/
def minMoveM : Nat := 1
/-- dist_e1_m .. dist_e3_m (trip-displacement class edges, meters) -/
def distE1M : Nat := 1000
def distE2M : Nat := 3000
def distE3M : Nat := 10000
/-- spdbin_e2 .. spdbin_e6 (staypoints-per-day bin lower edges) -/
def spdbinE2 : Nat := 2
def spdbinE3 : Nat := 3
def spdbinE4 : Nat := 4
def spdbinE6 : Nat := 6
/-- route_resample_m (route points kept every this many metres of path) -/
def routeResampleM : Nat := 25
/-- junction_min_turn_deg (a junction turn on the resampled stream) -/
def junctionMinTurnDeg : Nat := 30
/-- ja_e1_deg / ja_e2_deg (junction-angle class edges, abs degrees) -/
def jaE1Deg : Nat := 60
def jaE2Deg : Nat := 120
/-- seg_e1_m .. seg_e3_m (junction-to-junction segment class edges, metres) -/
def segE1M : Nat := 50
def segE2M : Nat := 150
def segE3M : Nat := 600
/-- circ_e1_pm .. circ_e3_pm (circuity class edges, permille of path/net) -/
def circE1Pm : Nat := 1200
def circE2Pm : Nat := 1500
def circE3Pm : Nat := 2500
/-- route_min_net_m (circuity defined only for legs moving at least this) -/
def routeMinNetM : Nat := 200
/-- errand_max_dwell_s (errand stop dwell cap) -/
def errandMaxDwellS : Nat := 10800
/-- tour_min_anchors (distinct non-home anchors in one day chain) -/
def tourMinAnchors : Nat := 3
/-- tour_max_gap_s (visit gap that keeps a day chain alive) -/
def tourMaxGapS : Nat := 7200
/-- revisit_min_days / revisit_window_days (revisit-beacon thresholds) -/
def revisitMinDays : Nat := 5
def revisitWindowDays : Nat := 14
/-- longdwell_min_s (long-dwell / home-base signal) -/
def longdwellMinS : Nat := 50400
/-- dp_clip_days / dp_clip_visits / dp_clip_legs (per-user L1 budgets) -/
def dpClipDays : Nat := 10
def dpClipVisits : Nat := 10
def dpClipLegs : Nat := 10
/-- window_start_s (2007-04-12T00:00:00 local-naive) -/
def windowStartS : Nat := 1176336000
/-- window_end_s (2012-01-01T00:00:00 local-naive, exclusive) -/
def windowEndS : Nat := 1325376000
/-- suppress_min_users / suppress_min_points (category-2 suppression) -/
def suppressMinUsers : Nat := 5
def suppressMinPoints : Nat := 500
/-- excl_* (Beijing exclusion bbox, centidegrees — the synthetic output must
contain NO point inside this box) -/
def exclLatMinCd : Nat := 3940
def exclLatMaxCd : Nat := 4110
def exclLonMinCd : Nat := 11540
def exclLonMaxCd : Nat := 11750
/-- tgt_* (Tokyo target bbox, centidegrees — every synthetic point must land
inside this box after the manipulate step) -/
def tgtLatMinCd : Nat := 3530
def tgtLatMaxCd : Nat := 3610
def tgtLonMinCd : Nat := 13920
def tgtLonMaxCd : Nat := 14030

/-- The 3 public anchor roles (the transition grid axis; fixed order). -/
def roles : List String := ["home", "work", "other"]

/-- The 5 public daily-motif classes (fixed order). -/
def motifClasses : List String :=
  ["home_day", "commute", "errand_loop", "tour", "irregular"]

/-- The 6 public mode classes (fixed order). -/
def modeClasses : List String := ["walk", "bike", "bus", "car", "rail", "other"]

/-- Mirror of dbt/seeds/mode_map.csv: raw GeoLife label -> public mode class.
    Unknown labels map to "XX" (never a release cell; checked, not assumed). -/
def modeClassOf (m : String) : String :=
  match m with
  | "walk" => "walk" | "run" => "walk"
  | "bike" => "bike"
  | "bus" => "bus"
  | "car" => "car" | "taxi" => "car" | "motorcycle" => "car"
  | "train" => "rail" | "subway" => "rail"
  | "airplane" => "other" | "boat" => "other"
  | _ => "XX"

/-- The Beijing exclusion box and the Tokyo target box are disjoint (so the
manipulate contract is satisfiable; kernel-checked). -/
theorem excl_tgt_disjoint : tgtLatMaxCd < exclLatMinCd := by decide

end EnclaveGeo.Params

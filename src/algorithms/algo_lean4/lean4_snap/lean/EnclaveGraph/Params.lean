/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 1.0.0) and
the category taxonomy in dbt/seeds/category_map.csv.

These values are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (and each `| "raw" => "cat"` match arm equals the seed row). Everything
else in this library references these definitions, so one literal is shared
by the proofs, the executables, Python and SQL.
-/

namespace EnclaveGraph.Params

def specVersion : String := "1.0.0"

/-- deg_e2 .. deg_e65 (degree bin lower edges: 1,2,3-4,5-8,9-16,17-32,33-64,65+) -/
def degE2 : Nat := 2
def degE3 : Nat := 3
def degE5 : Nat := 5
def degE9 : Nat := 9
def degE17 : Nat := 17
def degE33 : Nat := 33
def degE65 : Nat := 65
/-- cc_den (clustering class edges are k/cc_den for k=1..4; exact ℚ) -/
def ccDen : Nat := 5
/-- tri_e1 .. tri_e10001 (triangle-count bin lower edges) -/
def triE1 : Nat := 1
def triE11 : Nat := 11
def triE101 : Nat := 101
def triE1001 : Nat := 1001
def triE10001 : Nat := 10001
/-- rich_e8 (feature-richness bins 0..7 singletons, then 8+) -/
def richE8 : Nat := 8
/-- circ_e4 (circle-membership bins 0,1,2,3,4+) -/
def circE4 : Nat := 4
/-- csize_e25 .. csize_e401 (community-size bin lower edges) -/
def csizeE25 : Nat := 25
def csizeE51 : Nat := 51
def csizeE101 : Nat := 101
def csizeE201 : Nat := 201
def csizeE401 : Nat := 401
/-- n_community_ranks (rank labels c1..c8 by size, then `rest`) -/
def nCommunityRanks : Nat := 8
/-- hub_degree_min -/
def hubDegreeMin : Nat := 150
/-- leaf_degree -/
def leafDegree : Nat := 1
/-- broker_degree_min -/
def brokerDegreeMin : Nat := 17
/-- broker_cc_max_num / broker_cc_max_den (local clustering < 1/5) -/
def brokerCcMaxNum : Nat := 1
def brokerCcMaxDen : Nat := 5
/-- bridge_communities_min -/
def bridgeCommunitiesMin : Nat := 3
/-- butterfly_circles_min (circles within ONE ego network) -/
def butterflyCirclesMin : Nat := 3
/-- core_degree_min -/
def coreDegreeMin : Nat := 65
/-- core_cc_min_num / core_cc_min_den (local clustering >= 3/5) -/
def coreCcMinNum : Nat := 3
def coreCcMinDen : Nat := 5
/-- suppress_min_nodes (non-DP suppression threshold) -/
def suppressMinNodes : Nat := 50
/-- dp_clip_total (per-node L1 budget on edge/wedge/category grids) -/
def dpClipTotal : Nat := 8
/-- dp_count_clip (one cell per node histograms) -/
def dpCountClip : Nat := 1
/-- dp_csize_clip (community-size histogram is NODE-weighted: each derived
    row contributes its community's size bin once — row-level sensitivity 1;
    a community-count histogram over 15 communities was pure noise) -/
def dpCsizeClip : Nat := 1
/-- dp_clip_motifs (motif-presence vector clip) -/
def dpClipMotifs : Nat := 4
/-- louvain_seed (ingest determinism; not a privacy parameter) -/
def louvainSeed : Nat := 7
/-- synth_closure_tol_pct (|synth closure − released| ≤ tol/100) -/
def synthClosureTolPct : Nat := 15

/-- The 11 public feature categories (the DP grid axis; fixed order). -/
def categories : List String :=
  ["birthday", "education", "first_name", "gender", "hometown", "languages",
   "last_name", "locale", "location", "work", "other"]

/-- Mirror of dbt/seeds/category_map.csv: raw SNAP featname category ->
    public category. Unknown raw categories map to "other" (total). -/
def categoryOf (c : String) : String :=
  match c with
  | "birthday" => "birthday"
  | "education" => "education"
  | "first_name" => "first_name"
  | "gender" => "gender"
  | "hometown" => "hometown"
  | "languages" => "languages"
  | "last_name" => "last_name"
  | "locale" => "locale"
  | "location" => "location"
  | "work" => "work"
  | "middle_name" => "other"
  | "name" => "other"
  | "political" => "other"
  | "religion" => "other"
  | _ => "other"

/-- The 9 community rank labels (fixed order; the mixing-grid axis). -/
def communityRanks : List String :=
  ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "rest"]

end EnclaveGraph.Params

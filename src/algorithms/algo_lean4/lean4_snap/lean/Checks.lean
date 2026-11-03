/-
Executable checks connecting the formal graph definitions to enclave data:
evaluates the DECIDABLE Lean predicates IsHub / IsLeaf / IsBroker / IsBridge
/ IsButterfly / IsCore and the total class maps (degree, clustering,
triangle, richness, circle) on per-node observations exported from the dbt
marts, recomputes the chi-square association statistics EXACTLY over ℚ from
exported contingency cells (vs SQL's floating-point values), and records the
proof-status ledger of the formal layer. Exit code: 0 when every compared
verdict agrees with SQL, 1 otherwise.
-/
import EnclaveGraph.Motifs
import EnclaveGraph.Assoc
import EnclaveGraph.GraphShape
import EnclaveGraph.GenSpec
import EnclaveGraph.Params
import Lean.Data.Json

open Lean EnclaveGraph EnclaveGraph.Shape

def claim (name statement status : String) : Json :=
  Json.mkObj [("name", Json.str name), ("statement", Json.str statement),
              ("status", Json.str status)]

def claims : Json := Json.arr #[
  claim "clippedSum_sensitivity"
    "Adding/removing one node's derived row changes a clipped cell sum by ≤ clip (sensitivity = clip)" "proved",
  claim "privClippedCell_PureDP"
    "The per-cell release mechanism is pure (ε₁/ε₂)-DP (via SampCert privNoisedQueryPure_DP)" "proved",
  claim "cellQuery_sensitivity"
    "The per-cell clipped histogram query has sensitivity = clip (via exactBoundedSum + neighbour_map)" "proved",
  claim "histMech_DP"
    "The composed K-cell histogram mechanism is pure (K·ε₁/ε₂)-DP (privCompose/privPostProcess induction)" "proved",
  claim "sampleCellPMF_law"
    "The release sampler draws cell i with probability wᵢ/Σw exactly (via SampCert UniformSample_apply')" "proved",
  claim "isHub_canonical"
    "The measured maximum-degree node (1045) satisfies IsHub" "proved",
  claim "isHubAt_mono"
    "IsHub is monotone in the degree threshold" "proved",
  claim "isHub_not_isLeaf"
    "No node is both a hub and a leaf (the public thresholds keep the classes disjoint)" "proved",
  claim "isBroker_canonical"
    "A degree-17 star center with no closed wedges satisfies IsBroker" "proved",
  claim "isBroker_not_isCore"
    "No node is both a broker and a core (broker_cc_max 1/5 < core_cc_min 3/5)" "proved",
  claim "ccBelow_not_atLeast"
    "The exact rational clustering comparisons cc<b and cc≥b are mutually exclusive" "proved",
  claim "isCore_canonical"
    "A 66-clique member (degree 65, all wedges closed) satisfies IsCore" "proved",
  claim "isBridgeAt_mono"
    "IsBridge is monotone in the community-count threshold" "proved",
  claim "isButterflyAt_mono"
    "IsButterfly is monotone in the circle-count threshold" "proved",
  claim "class_maps_total"
    "Every class map lands in its public grid (degreeClass_mem, ccClass_mem, triClass_mem, richClass_mem, circClass_mem, csizeClass_mem, mixCell_mem, closureCell_mem)" "proved",
  claim "grid_sizes_anchor"
    "The public grids have exactly 8+7+6+9+11+5+6+45+16+6 cells — the 385/100 ledger arithmetic (kernel-checked)" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "mkEdge_ok"
    "Generator contract: the smart edge constructor only produces canonical (src < dst) edges" "proved",
  claim "mkEdge_none_iff"
    "Generator contract: the smart edge constructor rejects exactly self-loops" "proved",
  claim "edgesCanonical_sound"
    "Generator contract: a passing output check implies a strictly lex-sorted, all-canonical edge list" "proved",
  claim "edges_nodup"
    "Generator contract: a strictly lex-sorted edge list has no duplicate edges" "proved",
  claim "addPick_nodup"
    "Generator contract: category assignment accumulators stay duplicate-free" "proved",
  claim "sql_counts_correct"
    "The dbt SQL computed degrees/triangles/communities/clipped counts correctly from the raw GraphML" "assumed",
  claim "row_level_privacy_unit"
    "DP is w.r.t. add/remove of ONE derived node row; neighbor-induced changes to other rows (full node-DP) are NOT covered — declared, see goal.md" "assumed",
  claim "entropy_source"
    "SampCert FFI randomness (/dev/urandom) is a fair entropy source" "assumed",
  claim "io_wrappers"
    "JSON I/O wrappers in the executables are correct" "assumed"]

def getNatD (j : Json) (k : String) (d : Nat) : Nat :=
  ((j.getObjVal? k).bind Json.getNat?).toOption.getD d

def ratToFloat (q : ℚ) : Float :=
  Float.ofInt q.num / Float.ofNat q.den

def readJson? (path : System.FilePath) : IO (Option Json) := do
  try
    let txt ← IO.FS.readFile path
    match Json.parse txt with
    | .ok j => pure (some j)
    | .error _ => pure none
  catch _ =>
    pure none

def boolOf (j : Json) (k : String) : Bool :=
  ((j.getObjVal? k).bind Json.getBool?).toOption.getD false

def strOf (j : Json) (k : String) : String :=
  ((j.getObjVal? k).bind Json.getStr?).toOption.getD ""

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/nodes.json"
  match Json.parse txt with
  | .error e => IO.eprintln s!"checks: bad input: {e}"; return 1
  | .ok j =>
    let mut nAgree := 0
    let mut nDisagree := 0
    let mut motifChecked := 0
    let mut classChecked := 0
    let mut disagreeSamples : Array Json := #[]

    for b in ((j.getObjVal? "nodes").bind Json.getArr?).toOption.getD #[] do
      let deg := getNatD b "degree" 0
      let tri := getNatD b "tri_count" 0
      let nbC := getNatD b "neighbor_communities" 0
      let maxCirc := getNatD b "max_circles_one_ego" 0
      let nCirc := getNatD b "n_circles" 0
      let rich := getNatD b "richness" 0
      -- motif agreement (6 predicates per node)
      let verdicts : List (String × Bool × Bool) :=
        [ ("is_hub", decide (IsHub deg), boolOf b "is_hub")
        , ("is_leaf", decide (IsLeaf deg), boolOf b "is_leaf")
        , ("is_broker", decide (IsBroker deg tri), boolOf b "is_broker")
        , ("is_bridge", decide (IsBridge nbC), boolOf b "is_bridge")
        , ("is_butterfly", decide (IsButterfly maxCirc), boolOf b "is_butterfly")
        , ("is_core", decide (IsCore deg tri), boolOf b "is_core") ]
      -- class-map agreement (5 total maps per node)
      let classes : List (String × String × String) :=
        [ ("degree_class", degreeClass deg, strOf b "degree_class")
        , ("cc_class", ccClass deg tri, strOf b "cc_class")
        , ("tri_class", triClass tri, strOf b "tri_class")
        , ("richness_class", richClass rich, strOf b "richness_class")
        , ("circle_class", circClass nCirc, strOf b "circle_class") ]
      for (name, leanV, sqlV) in verdicts do
        motifChecked := motifChecked + 1
        if leanV == sqlV then nAgree := nAgree + 1
        else
          nDisagree := nDisagree + 1
          if disagreeSamples.size < 20 then
            disagreeSamples := disagreeSamples.push <| Json.mkObj
              [ ("node", (b.getObjVal? "node_id").toOption.getD Json.null)
              , ("check", Json.str name)
              , ("lean", Json.bool leanV), ("sql", Json.bool sqlV) ]
      for (name, leanC, sqlC) in classes do
        classChecked := classChecked + 1
        if leanC == sqlC then nAgree := nAgree + 1
        else
          nDisagree := nDisagree + 1
          if disagreeSamples.size < 20 then
            disagreeSamples := disagreeSamples.push <| Json.mkObj
              [ ("node", (b.getObjVal? "node_id").toOption.getD Json.null)
              , ("check", Json.str name)
              , ("lean", Json.str leanC), ("sql", Json.str sqlC) ]

    -- exact chi-square recomputation on exported contingency tables
    let mut assocResults : Array Json := #[]
    if let some aj ← readJson? "data/assoc.json" then
      let pairsA := ((aj.getObjVal? "pairs").bind Json.getArr?).toOption.getD #[]
      for pa in pairsA do
        let rowsJ := ((pa.getObjVal? "observed").bind Json.getArr?).toOption.getD #[]
        let table : ContTable := rowsJ.toList.map (fun row =>
          (row.getArr?.toOption.getD #[]).toList.map
            (fun c => ((c.getInt?.toOption.getD 0 : Int) : ℚ)))
        let leanF := ratToFloat (chiSq table)
        let sqlF := ((((pa.getObjVal? "sql_chi2").bind Json.getNum?).toOption.map
          (fun n => n.toFloat))).getD 0.0
        let tol := 1e-6 * max 1.0 (Float.abs leanF)
        let agree := Float.abs (leanF - sqlF) ≤ tol
        if agree then nAgree := nAgree + 1 else nDisagree := nDisagree + 1
        assocResults := assocResults.push <| Json.mkObj
          [ ("grid", (pa.getObjVal? "grid").toOption.getD (Json.str "?"))
          , ("lean_chi2", Json.str (toString leanF))
          , ("sql_chi2", Json.str (toString sqlF))
          , ("agree", Json.bool agree) ]

    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("motif_checks", Json.num (JsonNumber.fromNat motifChecked))
      , ("class_checks", Json.num (JsonNumber.fromNat classChecked))
      , ("assoc", Json.arr assocResults)
      , ("disagreement_samples", Json.arr disagreeSamples)
      , ("agreement", Json.mkObj
          [ ("n_checked", Json.num (JsonNumber.fromNat (nAgree + nDisagree)))
          , ("n_agree", Json.num (JsonNumber.fromNat nAgree))
          , ("n_disagree", Json.num (JsonNumber.fromNat nDisagree)) ]) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {motifChecked} motif, {classChecked} class, {assocResults.size} chi-square checks -> out/lean_results.json ({nDisagree} disagreements)"
    return (if nDisagree = 0 then 0 else 1)

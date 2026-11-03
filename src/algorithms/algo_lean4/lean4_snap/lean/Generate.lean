/-
Standalone Lean generator: the external synthesizer is
    approved genspec + generation request → Lean sampler → synthetic GraphML.

    lake exe generate <genspec.json> <n_nodes> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the GraphShape constants (public class grids, feature-key shape). Every
random draw runs the verified sampler (`sampleCellPMF_law`); edges are
created ONLY through the smart constructor `mkEdge` (canonical by
`mkEdge_ok`), the final edge list must pass `edgesCanonical` (sorted,
duplicate-free, no self-loops — `edgesCanonical_sound`, `edges_nodup`) or
the generator refuses to write output, and category assignments go through
`addPick` (duplicate-free by `addPick_nodup`).
Writes synth_graph.graphml (source GraphML schema: sparse value-1 feature
keys, per-ego circles strings, is_dataset_ego markers, canonical undirected
edges) and plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.

Generative structure (public priors declared derived_from_data: false):
  communities   = DP community-size bins, uniform inside the public range;
  degree target = DP degree bins, uniform inside the public range, stub
                  budget pre-shrunk by the expected closure contribution
                  (computed from RELEASED closure weights — post-processing);
  edge partners = DP mixing rows (partner community rank | own rank), node
                  uniform inside the partner community;
  triangles     = closure pass: per node, `closureTrials` sampled neighbor
                  pairs closed with the RELEASED per-degree-class probability;
  features      = DP richness bins → distinct categories from the DP
                  category mix; instance ids uniform in a fixed public pool;
  circles       = DP circle bins, count inside the public range, labels
                  scoped to the community's pseudo-ego.
PLANTED ground truth (self-verified with the DECIDABLE motif predicates
before the plan is written): node 0 = hub (degree ≥ 150), node 1 = social
butterfly (≥ 3 circles in one ego), node 2 = broker (degree ≥ 17, cc < 1/5,
grown with zero-common-neighbor edges so its triangle count is invariant).
-/
import EnclaveGraph.GenSpec
import EnclaveGraph.GraphShape
import EnclaveGraph.Params
import Std.Data.TreeSet
import Std.Data.HashSet
import Lean.Data.Json

open Lean EnclaveGraph EnclaveGraph.Gen EnclaveGraph.Shape

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)

/-- Public priors (declared, not data): the cross/mixing-stub share of a
node's target degree (1/8), the per-endpoint degree cap (2× target), the
block-size cap, and one light closure pass (trial cap per node). -/
def stubShareDen : Nat := 8
def degCapFactor : Nat := 2
def blockSizeCap : Nat := 600
def closureTrialCap : Nat := 16

/-- Public: the node-count window the fidelity tolerances were validated on
(measured 2026-11-03: 18/18 runs at 500..1000 within the degree tolerance;
replicate closure verdict passes through ~10000). Outside it the generator
still runs and verifies its planted structures, but warns. -/
def validatedMinNodes : Nat := 500
def validatedMaxNodes : Nat := 10000

/-- Public prior: feature-instance pool size per (ego-scope, category). -/
def featurePool : Nat := 16

/-- Public prior (declared affil_window): affiliation group members are drawn
from a window of `factor x group size` inside the community; the factor depends
on the size bin. Measured (5 runs per setting, 9 categories): a tight window
shares the near-clique blocks and OVERSHOOTS the released density (factor 2:
mean |error| 0.24, mid bins +0.3..+0.5); a wide one leaves the degree-capped fill
no room and UNDERSHOOTS (factor 32: small bins -0.35..-0.48). The crossover
differs by size: small groups want ~3, mid-size ~16, large ~8 (mean |error|
0.058, was 0.095-0.119 with a single factor 8). -/
def affilWindowSmall : Nat := 3   -- g_2_4
def affilWindowMid : Nat := 16    -- g_5_8, g_9_16
def affilWindowLarge : Nat := 8   -- g_17p

def affilWindow (bin : String) : Nat :=
  if bin == "g_2_4" then affilWindowSmall
  else if bin == "g_17p" then affilWindowLarge
  else affilWindowMid

/-- Public prior: the density fill prefers pairs that close wedges. A missing
pair's weight is 1 + affilClosureBoost * min(common neighbors, affilClosureCap),
normalised so the expected number of added edges is unchanged (declared
affil_closure_bias). -/
def affilClosureBoost : Nat := 16
def affilClosureCap : Nat := 3

def rangeOf (rs : List (String × RangeN)) (k : String) (dflt : RangeN) : RangeN :=
  ((rs.find? (·.1 == k)).map (·.2)).getD dflt

/-- Affiliation group-size range for a (category, bin): the per-category
override `<category>/<bin>` (the open bin's cap differs by category: locale and
gender groups average ~213 members, work groups ~33), else the shared bin range. -/
def affilRange (a : AffilSpec) (cat bin : String) : RangeN :=
  rangeOf a.groupRanges s!"{cat}/{bin}" (rangeOf a.groupRanges bin ⟨2, 4⟩)

/-- Grid-uniform draw in [lo, hi] via the verified sampler: uniform over at
most 4097 evenly spaced grid points (declared "uniform-ish within the
released range"; exact uniform for spans ≤ 4096). -/
def drawRange (r : RangeN) : IO Nat := do
  let span := r.hi - r.lo
  if span = 0 then
    pure r.lo
  else
    let buckets := min span 4096
    let i ← draw (List.replicate (buckets + 1) 1)
    pure (r.lo + span * i / buckets)

/-- Uniform draw in [0, k) via the verified sampler. -/
def drawBelow (k : Nat) : IO Nat := do
  if k ≤ 1 then pure 0 else draw (List.replicate k 1)

/-- Canonical-edge ordering for the RBTree edge set. -/
def edgeCmp (a b : Nat × Nat) : Ordering :=
  match compare a.1 b.1 with
  | .eq => compare a.2 b.2
  | o => o

abbrev EdgeSet := Std.TreeSet (Nat × Nat) edgeCmp

/-- Released closure probability (closed_w, open_w) for a degree class. -/
def closureWeights (spec : GenSpec) (degClass : String) : Nat × Nat :=
  let row := rowOf spec.closure ⟨[⟨"closed", 1⟩, ⟨"open", 1⟩]⟩ degClass
  let c := ((row.cells.find? (·.name == "closed")).map (·.w)).getD 1
  let o := ((row.cells.find? (·.name == "open")).map (·.w)).getD 1
  (c, o)

structure GState where
  edges : EdgeSet := Std.TreeSet.empty
  adj : Array (List Nat) := #[]
  nEdges : Nat := 0

/-- Insert via the smart constructor; duplicate-free by the RBTree key. -/
def GState.addEdge (st : GState) (a b : Nat) : GState :=
  match mkEdge a b with
  | none => st
  | some e =>
    if st.edges.contains e then st
    else
      { edges := st.edges.insert e
        adj := (st.adj.modify e.1 (e.2 :: ·)).modify e.2 (e.1 :: ·)
        nEdges := st.nEdges + 1 }

def GState.degree (st : GState) (v : Nat) : Nat := (st.adj.getD v []).length

/-- Triangles at `v`: adjacent pairs among its neighbors (exact, used for
the planted-broker self-check). -/
def GState.triAt (st : GState) (v : Nat) : Nat :=
  let ns := st.adj.getD v []
  (ns.foldl (fun acc u =>
    acc + (ns.filter (fun w => decide (u < w) &&
      (match mkEdge u w with
       | some e => st.edges.contains e
       | none => false))).length) 0)

/-- No common neighbor with `v` (adding such a partner keeps `triAt v`
invariant — the broker-planting monotonicity). -/
def GState.noCommonNeighbor (st : GState) (v u : Nat) : Bool :=
  let nv := st.adj.getD v []
  (st.adj.getD u []).all (fun w => ¬ nv.contains w)

def main (args : List String) : IO UInt32 := do
  let [specPath, nStr, outDir] := args
    | do IO.eprintln "usage: generate <genspec.json> <n_nodes> <out_dir>"; return 1
  let some nNodes := nStr.toNat?
    | do IO.eprintln "generate: n_nodes must be a natural"; return 1
  if nNodes < 200 ∨ nNodes > 100000 then
    IO.eprintln "generate: n_nodes must be in 200..100000"; return 1
  if nNodes < validatedMinNodes then
    IO.eprintln s!"generate: warning: n={nNodes} is below the validated fidelity range ({validatedMinNodes}..{validatedMaxNodes}); planted structures are still verified, but degree-mix and group-count tolerances are not met by construction at this size"
  if nNodes > validatedMaxNodes then
    IO.eprintln s!"generate: warning: n={nNodes} is above the validated fidelity range ({validatedMinNodes}..{validatedMaxNodes}): the rank-labelled mixing rows extrapolate (per-node closure margin ~0.02, still in band in every measured run); 100000 nodes needs ~3 GB and ~5 min"
  let txt ← IO.FS.readFile specPath
  let spec ← match Json.parse txt >>= GenSpec.parse with
    | .error e => do IO.eprintln s!"generate: bad genspec: {e}"; return 1
    | .ok s => pure s

  -- 1. community plan: sizes from the DP histogram until n nodes (fuel),
  -- sorted descending → rank labels c1.. / rest, contiguous id ranges.
  let mut sizes : Array Nat := #[]
  let mut totalN : Nat := 0
  for _ in [0:800] do
    if totalN < nNodes then
      let cell ← spec.csize.choose
      let s ← drawRange (rangeOf spec.csizeRanges cell.name ⟨19, 100⟩)
      let s := max 8 s
      sizes := sizes.push s
      totalN := totalN + s
  if totalN < nNodes then
    IO.eprintln "generate: community plan fuel exhausted"; return 1
  -- trim overshoot from the last community (keep ≥ 8)
  let overshoot := totalN - nNodes
  let last := sizes.back!
  if last > overshoot + 8 then
    sizes := sizes.set! (sizes.size - 1) (last - overshoot)
  else
    sizes := sizes.pop
  let sizesSorted := sizes.qsort (· > ·)
  let n : Nat := sizesSorted.foldl (· + ·) 0
  -- contiguous ranges + rank labels
  let mut commLo : Array Nat := #[]
  let mut acc := 0
  for s in sizesSorted do
    commLo := commLo.push acc
    acc := acc + s
  let rankOfComm := fun (ci : Nat) =>
    if ci < Params.nCommunityRanks then s!"c{ci + 1}" else "rest"
  let mut commOfNode : Array Nat := Array.replicate n 0
  for ci in [0:sizesSorted.size] do
    let lo := commLo[ci]!
    for v in [lo : lo + sizesSorted[ci]!] do
      commOfNode := commOfNode.set! v ci
  -- nodes of a rank label (for partner draws): list of community indices per rank
  let commsOfRank := fun (r : String) =>
    (List.range sizesSorted.size).filter (fun ci => rankOfComm ci == r)

  -- 2. degree targets, sorted descending WITHIN each community (declared
  -- block-layout prior): near-clique blocks then get members of similar
  -- target degree, so block wiring reproduces both the degree histogram and
  -- the released closure level.
  let mut target : Array Nat := Array.replicate n 1
  for v in [0:n] do
    let cell ← spec.degree.choose
    let d ← drawRange (rangeOf spec.degRanges cell.name ⟨1, 8⟩)
    target := target.set! v (min (max 1 d) (n - 1))
  for ci in [0:sizesSorted.size] do
    let lo := commLo[ci]!
    let sz := sizesSorted[ci]!
    let sorted := (target.extract lo (lo + sz)).qsort (· > ·)
    for k in [0:sz] do
      target := target.set! (lo + k) sorted[k]!

  -- 3a. near-clique blocks within communities: block size ≈ d/p + 1 where p
  -- is the RELEASED closure probability for the head's degree class; every
  -- in-block pair is wired by a verified draw at that probability. Degree ≈
  -- p·(s−1) ≈ d and local clustering ≈ p, by construction.
  let mut st : GState := { adj := Array.replicate n [] }
  for ci in [0:sizesSorted.size] do
    let lo := commLo[ci]!
    let sz := sizesSorted[ci]!
    let mut pos := 0
    while pos < sz do
      let d := target[lo + pos]!
      let (cw, ow) := closureWeights spec (degreeClass d)
      let cw' := max 1 cw
      -- block density q = min(1, 5/3 · released closure p): the densification
      -- factor (declared prior) compensates the dilution from deficit stubs;
      -- block size s0 = d/q + 1 keeps each member's degree ≈ its target
      let s0 := min (min (sz - pos) blockSizeCap)
        (max 2 (d * (cw' + ow) * 3 / (cw' * 5) + 1))
      -- homogeneity cap: stop the block where member targets halve, so
      -- min-target wiring keeps the block near the released density
      let mut sGrow := 1
      while sGrow < s0 ∧ target[lo + pos + sGrow]! * 2 ≥ d do
        sGrow := sGrow + 1
      -- a lone leftover node forms a singleton "block" (the pair loop is
      -- empty); clamping to the remaining size keeps every index in range
      let s := min (max 2 sGrow) (sz - pos)
      for a in [0:s] do
        for b in [a+1:s] do
          -- node 2 is reserved for the planted broker: it joins no block
          -- (its sparse start keeps the zero-common-neighbor growth cheap)
          if lo + pos + a ≠ 2 ∧ lo + pos + b ≠ 2 then
            -- pair probability min(tₐ,t_b)/(s−1): each member keeps its OWN
            -- degree target; homogeneous blocks sit at the released closure
            -- density p (block size d/p + 1 makes t/(s−1) ≈ p there)
            let pm := min (min target[lo + pos + a]! target[lo + pos + b]!) (s - 1)
            let pick ← draw [pm, s - 1 - pm]
            if pick == 0 then
              st := st.addEdge (lo + pos + a) (lo + pos + b)
      pos := pos + s

  -- 3b. affiliation key-groups (education/work — the shared-attribute
  -- cliques): a group is a uniform DISTINCT sample of nodes inside ONE
  -- community (declared prior affil_group_sample — the key_community_mlp
  -- teacher verdict that affiliations align with structural communities).
  -- Group planting stops when the member budget (released member-weighted
  -- total, scaled to n) is spent; sizes bin-drawn from the RELEASED
  -- affil_sizes histogram; internal pairs are DEFICIT-FILLED up to the
  -- RELEASED integer-percent density — existing block edges count toward
  -- the target, the broker node 2 never joins a group, and the LATER stub
  -- pass adapts to each node's remaining deficit, so the degree histogram
  -- absorbs the group edges instead of inflating.
  let mut affil : Array (List (String × Nat)) := Array.replicate n []
  let mut affilGroups : Nat := 0
  let mut affilMembers : Nat := 0
  let mut affilEdges : Nat := 0
  match spec.affiliation with
  | none => pure ()
  | some aspec =>
    for cat in aspec.categories do
      let sizeHist := ((aspec.sizeRows.find? (·.given == cat)).map (·.hist)).getD ⟨[]⟩
      let densHist := ((aspec.densityRows.find? (·.given == cat)).map (·.hist)).getD ⟨[]⟩
      -- the size row holds GROUP-COUNT weights (distill divided the
      -- member-weighted release by the public bin midpoints); the member
      -- budget is recovered as sum(w * midpoint)
      let mid := fun (nm : String) =>
        let r := affilRange aspec cat nm
        max 1 ((r.lo + r.hi) / 2)
      let memberTotal := sizeHist.cells.foldl (fun acc c => acc + c.w * mid c.name) 0
      if memberTotal > 0 then
        -- member budget scaled to the synthetic node count (post-processing)
        let mut remaining := memberTotal * n / max 1 spec.nNodesHint
        for _ in [0 : 2 * n] do
          if remaining ≥ 2 then
            let cell ← sizeHist.choose
            let pct := min (max 1 aspec.densityDen)
              (((densHist.cells.find? (·.name == cell.name)).map (·.w)).getD 0)
            if cell.name ≠ "" ∧ pct > 0 then
              let sDrawn ← drawRange (affilRange aspec cat cell.name)
              let v ← drawBelow n
              let ci := commOfNode[v]!
              let lo := commLo[ci]!
              let sz := sizesSorted[ci]!
              -- members come from a WINDOW of affilWindow(bin) x the group
              -- size inside the community (declared prior affil_window): the
              -- community's nodes are laid out block-by-block, so nearby
              -- nodes already share block edges that count toward the
              -- density target, keeping closure instead of adding random
              -- edges. Distinct sample, fuel-bounded; broker node 2 excluded.
              let want := min sDrawn sz
              let wsz := min sz (affilWindow cell.name * want)
              let off0 ← drawBelow (sz - wsz + 1)
              let mut mem : List Nat := []
              for _ in [0 : 8 * want] do
                if mem.length < want then
                  let off ← drawBelow wsz
                  let u := lo + off0 + off
                  if u ≠ 2 ∧ ¬ mem.contains u then
                    mem := u :: mem
              let members := mem.toArray
              if members.size ≥ 2 then
                let gid := affilGroups
                affilGroups := affilGroups + 1
                affilMembers := affilMembers + members.size
                remaining := remaining - members.size
                for u in members do
                  affil := affil.set! u ((cat, gid) :: affil[u]!)
                -- deficit fill to the released density: count existing
                -- in-group edges, wire missing pairs at need/missing
                let mut existing := 0
                let mut missing : Array (Nat × Nat) := #[]
                for a in [0 : members.size] do
                  for b in [a + 1 : members.size] do
                    match mkEdge members[a]! members[b]! with
                    | some e =>
                      if st.edges.contains e then existing := existing + 1
                      else missing := missing.push (a, b)
                    | none => pure ()
                let totalPairs := existing + missing.size
                let den := max 1 aspec.densityDen
                -- STOCHASTIC rounding (verified sampler): round the target edge count
                -- up with probability equal to its fractional part. Deterministic
                -- rounding is biased on tiny groups - a size-2 group has one pair, so
                -- any released density >= 0.5 became a full pair (density 1.0) and
                -- the 2-4 bin overshot by ~0.16 at a released 0.62.
                let numE := pct * totalPairs
                let remE := numE % den
                let bump ← if remE == 0 then pure 1 else draw [remE, den - remE]
                let targetE := numE / den + (if bump == 0 ∧ remE > 0 then 1 else 0)
                let need := min (targetE - existing) missing.size
                if need > 0 then
                  -- closure-biased selection (prior affil_closure_bias): pairs
                  -- sharing neighbours are likelier, so the same number of
                  -- added edges closes wedges instead of opening new ones
                  let nbrSets : Array (Std.HashSet Nat) := members.map (fun u =>
                    (st.adj.getD u []).foldl (fun acc x => acc.insert x)
                      ({} : Std.HashSet Nat))
                  let wts : Array Nat := missing.map (fun (ia, ib) =>
                    let cn := ((st.adj.getD members[ia]! []).filter
                      (fun x => (nbrSets.getD ib {}).contains x)).length
                    1 + affilClosureBoost * min affilClosureCap cn)
                  let sumW := wts.foldl (· + ·) 0
                  for k in [0 : missing.size] do
                    let (ia, ib) := missing[k]!
                    let a := members[ia]!
                    let b := members[ib]!
                    if st.degree a < target[a]! + max 2 (target[a]! / 4) ∧
                        st.degree b < target[b]! + max 2 (target[b]! / 4) then
                      let num := need * wts[k]!
                      let pick ← if num ≥ sumW then pure 0 else draw [num, sumW - num]
                      if pick == 0 then
                        let before := st.nEdges
                        st := st.addEdge a b
                        if st.nEdges > before then
                          affilEdges := affilEdges + 1

  -- 3c. cross/mixing stubs: partner rank from the DP mixing row, node
  -- uniform in the destination (own community for the diagonal). Stub count
  -- adapts to the node's remaining degree deficit after block + affiliation
  -- wiring.
  let marginalMix : Hist :=
    ⟨Params.communityRanks.map (fun r => ⟨r, 1⟩)⟩
  for v in [0:n] do
    let myCi := commOfNode[v]!
    let myRank := rankOfComm myCi
    let row := rowOf spec.mixing marginalMix myRank
    -- Nodes in the open degree bin (target >= degE65) initiate their FULL
    -- remaining deficit; everyone else initiates half and expects the other half
    -- to arrive as other nodes' stubs. Partners are drawn uniformly, so almost
    -- nobody picks a hub and the half-deficit left hubs at about half their target
    -- (declared prior hub_full_deficit; measured over 8 runs each, before -> after
    -- (real): share of nodes with degree >= 100 0.078 -> 0.105 (0.122), transitivity
    -- 0.584 -> 0.531 (0.519), modularity 0.860 -> 0.840 (0.835), core share
    -- 0.137 -> 0.102 (0.091); edge count unchanged).
    let deficit := if st.degree v < target[v]! then
        (if target[v]! ≥ Params.degE65 then target[v]! - st.degree v
         else max 1 ((target[v]! - st.degree v) / 2))
      else 0
    for _ in [0:deficit] do
      let cell ← row.choose
      let destRank := if cell.name == "" then myRank else cell.name
      let cands := commsOfRank destRank
      if h : cands.length > 0 then
        let ciIdx ← drawBelow cands.length
        let ci := cands.getD ciIdx 0
        let u ← drawBelow sizesSorted[ci]!
        let uid := commLo[ci]! + u
        -- receiver cap: stubs never push a node far past its own target
        -- (protects the low-degree bins from inflow)
        if st.degree uid < target[uid]! + max 2 (target[uid]! / 4) then
          st := st.addEdge v uid

  -- 4. one light triadic-closure pass (released per-degree-class
  -- probabilities; per-endpoint degree caps protect the degree histogram)
  let mut closureAttempts := 0
  let mut closureClosed := 0
  for v in [0:n] do
    let ns := (st.adj.getD v []).toArray
    if ns.size ≥ 2 ∧ v ≠ 2 then
      let (cw, ow) := closureWeights spec (degreeClass ns.size)
      let trials := min ns.size closureTrialCap
      for _ in [0:trials] do
        let i ← drawBelow ns.size
        let j ← drawBelow ns.size
        if i ≠ j then
          let u := ns[i]!
          let w := ns[j]!
          if st.degree u < target[u]! + max 2 (target[u]! / 4) ∧
              st.degree w < target[w]! + max 2 (target[w]! / 4) then
            closureAttempts := closureAttempts + 1
            let pick ← draw [cw, ow]
            if pick == 0 then
              let before := st.nEdges
              st := st.addEdge u w
              if st.nEdges > before then
                closureClosed := closureClosed + 1

  -- 5. planting (after closure, so nothing disturbs the planted structure)
  -- hub: node 0 reaches degree ≥ hub_degree_min
  for u in [1:Params.hubDegreeMin + 6] do
    if u < n then
      st := st.addEdge 0 u
  -- broker: node 2 grows with zero-common-neighbor partners until the
  -- DECIDABLE predicate accepts (triAt 2 is invariant under such additions)
  let mut cand := 3
  let mut fuel := n
  while fuel > 0 ∧
      ¬ (decide (IsBroker (st.degree 2) (st.triAt 2))) do
    if cand < n ∧ cand ≠ 0 ∧ st.noCommonNeighbor 2 cand then
      st := st.addEdge 2 cand
    cand := if cand + 1 < n then cand + 1 else 3
    fuel := fuel - 1
  -- phase 2 (small/dense graphs only; never runs when phase 1 succeeded):
  -- zero-common-neighbor partners ran out, but IsBroker only needs a LOW
  -- triangle/wedge ratio, so greedily add the partner with the FEWEST common
  -- neighbors until the decidable predicate accepts.
  let mut fuel2 := n
  while fuel2 > 0 ∧ ¬ (decide (IsBroker (st.degree 2) (st.triAt 2))) do
    let n2 := st.adj.getD 2 []
    let mut best : Option (Nat × Nat) := none
    for c in [3:n] do
      if c ≠ 0 ∧ ¬ n2.contains c then
        let cn := ((st.adj.getD c []).filter (fun x => n2.contains x)).length
        match best with
        | some (bcn, _) => if cn < bcn then best := some (cn, c)
        | none => best := some (cn, c)
    match best with
    | some (_, c) => st := st.addEdge 2 c
    | none => fuel2 := 1
    fuel2 := fuel2 - 1
  if ¬ (decide (IsBroker (st.degree 2) (st.triAt 2))) then
    IO.eprintln "generate: broker planting failed (fuel)"; return 1

  -- 6. per-node attributes: richness → distinct categories (a node's
  -- affiliation categories are counted first); circle counts
  let mut cats : Array (List String) := Array.replicate n []
  let mut circles : Array Nat := Array.replicate n 0
  for v in [0:n] do
    let rCell ← spec.richness.choose
    let k ← drawRange (rangeOf spec.richRanges rCell.name ⟨0, 0⟩)
    let affCats := (affil[v]!.map (·.1)).eraseDups
    cats := cats.set! v
      (← pickDistinct spec.category 64 (k - min k affCats.length) affCats)
    let cCell ← spec.circles.choose
    circles := circles.set! v (← drawRange (rangeOf spec.circRanges cCell.name ⟨0, 0⟩))
  -- butterfly: node 1 carries ≥ butterfly_circles_min circles (one ego scope)
  circles := circles.set! 1 (max circles[1]! Params.butterflyCirclesMin)
  if ¬ (decide (IsButterfly circles[1]!)) then
    IO.eprintln "generate: butterfly planting failed"; return 1
  if ¬ (decide (IsHub (st.degree 0))) then
    IO.eprintln "generate: hub planting failed"; return 1

  -- 7. ego marking: the first node of each of the first ego_count communities
  let nEgos := min spec.egoCount sizesSorted.size
  let egoOfComm := fun (ci : Nat) => commLo[min ci (nEgos - 1)]!
  let egoIds := (List.range nEgos).map (fun ci => commLo[ci]!)

  -- 8. final canonicality check (the proved checker) before any output
  let edgeList := st.edges.toList
  if ¬ edgesCanonical edgeList then
    IO.eprintln "generate: edge list failed the canonicality check"; return 1

  -- 9. render GraphML
  let catIdx := fun (c : String) => (Params.categories.idxOf c)
  -- feature instance per (node, category): group keys carry the group id
  -- offset past the public pool (so same-group nodes SHARE a key and never
  -- collide with a pool draw); non-group AFFILIATION features get a
  -- node-unique instance (no phantom key-groups — released key-group
  -- structure comes ONLY from the planted groups); other categories
  -- uniform in the public pool
  let affilCatsAll := ((spec.affiliation.map (·.categories)).getD [])
  let mut feats : Array (List (Nat × String × Nat)) := Array.replicate n []
  for v in [0:n] do
    let e := egoOfComm commOfNode[v]!
    let affCats := (affil[v]!.map (·.1)).eraseDups
    let mut fs : List (Nat × String × Nat) := []
    for (c, gid) in affil[v]! do
      fs := (e, c, featurePool + gid) :: fs
    for c in cats[v]! do
      if ¬ affCats.contains c then
        if affilCatsAll.contains c then
          fs := (e, c, featurePool + affilGroups + v) :: fs
        else
          let inst ← drawBelow featurePool
          fs := (e, c, inst) :: fs
    feats := feats.set! v fs
  -- collect the key set (sorted for a deterministic header)
  let mut keySet : Std.TreeSet (Nat × Nat × Nat) (fun a b =>
    match compare a.1 b.1 with
    | .eq => (match compare a.2.1 b.2.1 with
              | .eq => compare a.2.2 b.2.2
              | o => o)
    | o => o) := Std.TreeSet.empty
  for v in [0:n] do
    for (e, c, inst) in feats[v]! do
      keySet := keySet.insert (e, catIdx c, inst)
  let featKeyName := fun (e ci inst : Nat) =>
    Shape.featureKey e (ci * featurePool + inst)
      (Params.categories.getD ci "other") inst
  let featKeyId := fun (e ci inst : Nat) =>
    s!"d{e}_{ci}_{inst}_{Params.categories.getD ci "other"}"
  let mut out : Array String := #[]
  out := out.push "<?xml version='1.0' encoding='utf-8'?>"
  out := out.push "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">"
  out := out.push "<key id=\"d0_is_ego\" for=\"node\" attr.name=\"is_dataset_ego\" attr.type=\"boolean\" />"
  out := out.push "<key id=\"d1_ego_id\" for=\"node\" attr.name=\"ego_network_id\" attr.type=\"string\" />"
  for e in egoIds do
    out := out.push s!"<key id=\"dcirc_{e}_circles\" for=\"node\" attr.name=\"ego_{e}__circles\" attr.type=\"string\" />"
  for (e, ci, inst) in keySet.toList do
    out := out.push s!"<key id=\"{featKeyId e ci inst}\" for=\"node\" attr.name=\"{featKeyName e ci inst}\" attr.type=\"long\" />"
  out := out.push "<graph edgedefault=\"undirected\">"
  for v in [0:n] do
    let e := egoOfComm commOfNode[v]!
    let mut row := s!"<node id=\"{v}\">"
    if egoIds.contains v then
      row := row ++ "<data key=\"d0_is_ego\">true</data>"
      row := row ++ s!"<data key=\"d1_ego_id\">{v}</data>"
    if circles[v]! > 0 then
      let labels := (List.range circles[v]!).map (fun j => s!"circle{j}")
      row := row ++ s!"<data key=\"dcirc_{e}_circles\">{String.intercalate "|" labels}</data>"
    for (fe, c, inst) in feats[v]! do
      row := row ++ s!"<data key=\"{featKeyId fe (catIdx c) inst}\">1</data>"
    row := row ++ "</node>"
    out := out.push row
  for (a, b) in edgeList do
    out := out.push s!"<edge source=\"{a}\" target=\"{b}\" />"
  out := out.push "</graph>"
  out := out.push "</graphml>"

  IO.FS.createDirAll outDir
  IO.FS.writeFile s!"{outDir}/synth_graph.graphml"
    (String.intercalate "\n" out.toList ++ "\n")
  let plan := Json.mkObj
    [ ("n_nodes", jn n)
    , ("n_edges", jn st.nEdges)
    , ("communities", Json.arr (sizesSorted.map jn))
    , ("ego_ids", Json.arr (egoIds.map jn).toArray)
    , ("planted", Json.mkObj
        [ ("hub", Json.mkObj [("node", jn 0), ("degree", jn (st.degree 0))])
        , ("broker", Json.mkObj
            [ ("node", jn 2), ("degree", jn (st.degree 2))
            , ("triangles", jn (st.triAt 2)) ])
        , ("butterfly", Json.mkObj
            [("node", jn 1), ("circles", jn circles[1]!)]) ])
    , ("closure", Json.mkObj [("attempts", jn closureAttempts),
                              ("closed", jn closureClosed)])
    , ("affiliation", Json.mkObj
        [ ("groups", jn affilGroups)
        , ("members", jn affilMembers)
        , ("edges_added", jn affilEdges)
        , ("modelled", Json.bool spec.affiliation.isSome) ])
    , ("spec_version", Json.str spec.version)
    , ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {n} nodes, {st.nEdges} edges -> {outDir}/synth_graph.graphml"
  return 0

/-
Formal definitions of the graph motifs computed by the dbt marts, over Nat
(degrees, triangle counts, community counts, circle counts), so every
predicate is decidable and executable. `lake exe checks` evaluates these very
definitions on per-node observations exported from the enclave store. All
thresholds come from the mirrored concept spec (Params.lean); the `...At`
variants take the threshold explicitly so monotonicity can be stated and the
executable can honor exported parameters.

Local clustering is kept EXACT: for a node of degree `d` with `t` triangles,
`cc = 2t / (d(d−1))`. Comparisons against a public rational bound `num/den`
are restated by cross-multiplication over Nat — no floats, no division:

    cc < num/den  ⟺  den·2t < num·d(d−1)      (for d(d−1) > 0)
    cc ≥ num/den  ⟺  num·d(d−1) ≤ den·2t
-/
import Mathlib.Tactic
import EnclaveGraph.Params

namespace EnclaveGraph

/-! ### Exact local-clustering comparisons -/

/-- `2t` — the numerator of the local clustering coefficient. -/
def ccNum (tri : Nat) : Nat := 2 * tri

/-- `d(d−1)` — the denominator of the local clustering coefficient
(also `2 × wedges centered at the node`). -/
def ccDen (deg : Nat) : Nat := deg * (deg - 1)

/-- `cc < num/den`, by cross-multiplication (`den·2t < num·d(d−1)`).
False whenever `d(d−1) = 0` (a node with no wedges has cc = 0 by the SQL
convention; the motif using `CcBelow` also requires degree ≥ 17, so the
denominator is positive on every accepted node). -/
def CcBelow (num den deg tri : Nat) : Prop :=
  den * ccNum tri < num * ccDen deg

instance (num den deg tri : Nat) : Decidable (CcBelow num den deg tri) := by
  unfold CcBelow; exact inferInstance

/-- `cc ≥ num/den`, by cross-multiplication (`num·d(d−1) ≤ den·2t`). -/
def CcAtLeast (num den deg tri : Nat) : Prop :=
  num * ccDen deg ≤ den * ccNum tri

instance (num den deg tri : Nat) : Decidable (CcAtLeast num den deg tri) := by
  unfold CcAtLeast; exact inferInstance

/-- The two comparisons against the same bound are mutually exclusive. -/
theorem ccBelow_not_atLeast {num den deg tri : Nat}
    (h : CcBelow num den deg tri) : ¬ CcAtLeast num den deg tri := by
  unfold CcBelow at h; unfold CcAtLeast; omega

/-! ### Hub motif (high-degree re-identification class) -/

/-- Hub at an explicit degree threshold. -/
def IsHubAt (minDeg deg : Nat) : Prop := minDeg ≤ deg

instance (minDeg deg : Nat) : Decidable (IsHubAt minDeg deg) := by
  unfold IsHubAt; exact inferInstance

/-- A hub at the public spec threshold `hub_degree_min`. -/
def IsHub (deg : Nat) : Prop := IsHubAt Params.hubDegreeMin deg

instance (deg : Nat) : Decidable (IsHub deg) := by
  unfold IsHub; exact inferInstance

/-- The measured maximum-degree node (1045) is a hub (sanity). -/
theorem isHub_canonical : IsHub 1045 := by decide

/-- Lowering the degree threshold preserves hub-hood (monotonicity). -/
theorem isHubAt_mono {m m' deg : Nat} (hm : m' ≤ m) (h : IsHubAt m deg) :
    IsHubAt m' deg := le_trans hm h

/-! ### Leaf motif (periphery) -/

/-- A leaf: degree exactly `leaf_degree` (= 1). -/
def IsLeaf (deg : Nat) : Prop := deg = Params.leafDegree

instance (deg : Nat) : Decidable (IsLeaf deg) := by
  unfold IsLeaf; exact inferInstance

/-- A hub is never a leaf (the thresholds keep the classes disjoint). -/
theorem isHub_not_isLeaf {deg : Nat} (h : IsHub deg) : ¬ IsLeaf deg := by
  unfold IsHub IsHubAt Params.hubDegreeMin at h
  unfold IsLeaf Params.leafDegree
  omega

/-! ### Broker motif (structural-hole spanner: connected but unclosed) -/

/-- Broker at explicit thresholds: degree ≥ `minDeg` and local clustering
strictly below `num/den`. -/
def IsBrokerAt (minDeg num den deg tri : Nat) : Prop :=
  minDeg ≤ deg ∧ CcBelow num den deg tri

instance (minDeg num den deg tri : Nat) :
    Decidable (IsBrokerAt minDeg num den deg tri) := by
  unfold IsBrokerAt; exact inferInstance

/-- A broker at the public spec thresholds (degree ≥ 17, cc < 1/5). -/
def IsBroker (deg tri : Nat) : Prop :=
  IsBrokerAt Params.brokerDegreeMin Params.brokerCcMaxNum
    Params.brokerCcMaxDen deg tri

instance (deg tri : Nat) : Decidable (IsBroker deg tri) := by
  unfold IsBroker; exact inferInstance

/-- The canonical star center (degree 17, no closed wedges) is a broker
(sanity: the definition accepts the motif it was designed around). -/
theorem isBroker_canonical : IsBroker 17 0 := by decide

/-- Raising the clustering bound preserves broker-hood (monotonicity). -/
theorem isBrokerAt_mono_cc {minDeg num num' den deg tri : Nat}
    (hn : num ≤ num') (h : IsBrokerAt minDeg num den deg tri) :
    IsBrokerAt minDeg num' den deg tri :=
  ⟨h.1, lt_of_lt_of_le h.2 (Nat.mul_le_mul_right _ hn)⟩

/-! ### Core motif (strongly embedded member) -/

/-- Core at explicit thresholds: degree ≥ `minDeg` and local clustering at
least `num/den`. -/
def IsCoreAt (minDeg num den deg tri : Nat) : Prop :=
  minDeg ≤ deg ∧ CcAtLeast num den deg tri

instance (minDeg num den deg tri : Nat) :
    Decidable (IsCoreAt minDeg num den deg tri) := by
  unfold IsCoreAt; exact inferInstance

/-- A core at the public spec thresholds (degree ≥ 65, cc ≥ 3/5). -/
def IsCore (deg tri : Nat) : Prop :=
  IsCoreAt Params.coreDegreeMin Params.coreCcMinNum Params.coreCcMinDen deg tri

instance (deg tri : Nat) : Decidable (IsCore deg tri) := by
  unfold IsCore; exact inferInstance

/-- A 66-clique member (degree 65, every wedge closed: tri = d(d−1)/2 = 2080)
is a core (sanity). -/
theorem isCore_canonical : IsCore 65 2080 := by decide

/-- **Broker and core are mutually exclusive** — the public bounds satisfy
`broker_cc_max (1/5) < core_cc_min (3/5)`, so no node is both. -/
theorem isBroker_not_isCore {deg tri : Nat} (h : IsBroker deg tri) :
    ¬ IsCore deg tri := by
  rintro ⟨-, hcc⟩
  have hb := h.2
  unfold CcBelow ccNum ccDen Params.brokerCcMaxNum Params.brokerCcMaxDen at hb
  unfold CcAtLeast ccNum ccDen Params.coreCcMinNum Params.coreCcMinDen at hcc
  omega

/-! ### Bridge motif (neighbors in many communities) -/

/-- Bridge at an explicit community-count threshold. -/
def IsBridgeAt (minC k : Nat) : Prop := minC ≤ k

instance (minC k : Nat) : Decidable (IsBridgeAt minC k) := by
  unfold IsBridgeAt; exact inferInstance

/-- A bridge at the public spec threshold `bridge_communities_min`:
the node's neighbors span at least 3 distinct communities. -/
def IsBridge (k : Nat) : Prop := IsBridgeAt Params.bridgeCommunitiesMin k

instance (k : Nat) : Decidable (IsBridge k) := by
  unfold IsBridge; exact inferInstance

/-- Lowering the threshold preserves bridge-hood (monotonicity). -/
theorem isBridgeAt_mono {m m' k : Nat} (hm : m' ≤ m) (h : IsBridgeAt m k) :
    IsBridgeAt m' k := le_trans hm h

/-! ### Social-butterfly motif (many circles within one ego network) -/

/-- Butterfly at an explicit circle-count threshold: `m` is the node's
maximum circle count within any single ego network. -/
def IsButterflyAt (minCirc m : Nat) : Prop := minCirc ≤ m

instance (minCirc m : Nat) : Decidable (IsButterflyAt minCirc m) := by
  unfold IsButterflyAt; exact inferInstance

/-- A butterfly at the public spec threshold `butterfly_circles_min`. -/
def IsButterfly (m : Nat) : Prop := IsButterflyAt Params.butterflyCirclesMin m

instance (m : Nat) : Decidable (IsButterfly m) := by
  unfold IsButterfly; exact inferInstance

/-- Lowering the threshold preserves butterfly-hood (monotonicity). -/
theorem isButterflyAt_mono {m m' k : Nat} (hm : m' ≤ m)
    (h : IsButterflyAt m k) : IsButterflyAt m' k := le_trans hm h

/-! ### Well-formed edges (the generator's validity unit) -/

/-- Canonical undirected edge: strictly ordered endpoints (hence no
self-loop). -/
def EdgeOK (s d : Nat) : Prop := s < d

instance (s d : Nat) : Decidable (EdgeOK s d) := by
  unfold EdgeOK; exact inferInstance

/-- A canonical edge has no self-loop. -/
theorem edgeOK_no_selfloop {s d : Nat} (h : EdgeOK s d) : s ≠ d :=
  Nat.ne_of_lt h

end EnclaveGraph

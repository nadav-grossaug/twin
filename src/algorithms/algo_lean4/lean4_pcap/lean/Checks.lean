/-
Executable checks connecting the formal motif definitions to enclave data:
evaluates the DECIDABLE Lean predicates IsBeacon / IsFanout / HasReturnTrip /
IsTimelySession on observations exported from the dbt marts, recomputes the
chi-square association statistics EXACTLY over ℚ from exported contingency
cells (vs SQL's floating-point values), and records the proof-status ledger
of the formal layer.
-/
import EnclaveFlows.Motifs
import EnclaveFlows.Assoc
import EnclaveFlows.Sequences
import EnclaveFlows.Tls
import EnclaveFlows.Dns
import EnclaveFlows.Params
import Lean.Data.Json

open Lean EnclaveFlows

def claim (name statement status : String) : Json :=
  Json.mkObj [("name", Json.str name), ("statement", Json.str statement),
              ("status", Json.str status)]

def claims : Json := Json.arr #[
  claim "clippedSum_sensitivity"
    "Adding/removing one user changes a clipped cell sum by ≤ clip (sensitivity = clip)" "proved",
  claim "privClippedCell_PureDP"
    "The per-cell release mechanism is pure (ε₁/ε₂)-DP (via SampCert privNoisedQueryPure_DP)" "proved",
  claim "sampleCellPMF_law"
    "The release sampler draws cell i with probability wᵢ/Σw exactly (via SampCert UniformSample_apply')" "proved",
  claim "isBeacon_replicate"
    "A perfectly periodic channel satisfies IsBeacon" "proved",
  claim "isBeacon_mono_cv"
    "IsBeacon is monotone in the CV threshold" "proved",
  claim "isFanout_mono"
    "IsFanout is monotone in the destination threshold" "proved",
  claim "hasReturnTrip_cons_iff"
    "Window inversion for the A→B→A return-trip motif (basis of decidability)" "proved",
  claim "chiSq_nonneg"
    "The chi-square statistic of any contingency table is nonnegative" "proved",
  claim "chiSq_eq_zero_of_indep"
    "Chi-square vanishes on exactly independent tables" "proved",
  claim "assocAtLeast_mono"
    "Association-at-threshold claims are monotone in the threshold" "proved",
  claim "isTimelySession_mono"
    "Timely sessions are preserved under relaxing the gap bound" "proved",
  claim "isTimelySession_ordered"
    "A timely session is chronologically ordered" "proved",
  claim "cellQuery_sensitivity"
    "The per-cell clipped histogram query has sensitivity = clip (via exactBoundedSum + neighbour_map)" "proved",
  claim "histMech_DP"
    "The composed K-cell histogram mechanism is pure (K·ε₁/ε₂)-DP (privCompose/privPostProcess induction)" "proved",
  claim "hasReturnTrip_iff_infix"
    "Return-trip extractor is sound AND complete: HasReturnTrip l ↔ ∃ a≠b, [a,b,a] infix of l" "proved",
  claim "genBeacon_isBeacon"
    "Generator contract: channels rendered from FlowShape period/jitter params satisfy IsBeacon" "proved",
  claim "isTimely_cumTimes"
    "Generator contract: cumulative bounded steps form a timely session" "proved",
  claim "canonicalHandshake_data_has_ccs"
    "In a canonical TLS session every application-data record is accompanied by a ChangeCipherSpec in its direction (soundness of the record scan)" "proved",
  claim "keyChangeBeforeData_head_not_data"
    "A canonical TLS record sequence never opens with application data" "proved",
  claim "fpCounts_partition"
    "The client-stack fingerprint histogram partitions the TLS flows (Σ counts = N)" "proved",
  claim "isRareStack_mono"
    "Rare-stack verdicts are monotone in the rarity threshold" "proved",
  claim "wellFormed_labels_le"
    "An RFC 1035 well-formed query name has at most ⌊(maxName+1)/2⌋ labels (127 for the standard 253-byte bound)" "proved",
  claim "isSubdomainFan_mono"
    "Subdomain-fan verdicts are monotone in the fan threshold (inherited from IsFanout)" "proved",
  claim "isSubdomainFan_le_length"
    "A k-fan requires at least k queried names" "proved",
  claim "sql_counts_correct"
    "The dbt SQL computed per-user counts/gaps correctly from raw flows" "assumed",
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

def tlsRecOf (j : Json) : TlsRec :=
  let a := j.getArr?.toOption.getD #[]
  let nat (i : Nat) : Nat := (a[i]?.map (fun x => x.getNat?.toOption.getD 0)).getD 0
  { dirOut := nat 0 == 1, tp := nat 1, b := nat 2, iptMs := nat 3 }

def fpOf (j : Json) : Fingerprint :=
  let strs (k : String) : List String :=
    (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
      (fun s => s.getStr?.toOption.getD "")
  { version := getNatD j "v" 0, cs := strs "cs", exts := strs "ext" }

def strListOf (j : Json) (k : String) : List String :=
  (((j.getObjVal? k).bind Json.getArr?).toOption.getD #[]).toList.map
    (fun s => s.getStr?.toOption.getD "")

def main : IO UInt32 := do
  let txt ← IO.FS.readFile "data/motifs.json"
  match Json.parse txt with
  | .error e => IO.eprintln s!"checks: bad input: {e}"; return 1
  | .ok j =>
    let params := (j.getObjVal? "params").toOption.getD Json.null
    -- defaults come from the mirrored concept spec (Params.lean)
    let minFlows := getNatD params "beacon_min_flows" Params.beaconMinFlows
    let cvNum := getNatD params "beacon_max_cv_num" Params.beaconMaxCvNum
    let cvDen := getNatD params "beacon_max_cv_den" Params.beaconMaxCvDen
    let minPeriodMs := getNatD params "beacon_min_period_ms" Params.beaconMinPeriodMs
    let fanoutMin := getNatD params "fanout_min_dsts" Params.fanoutMinDsts
    let p : BeaconParams :=
      { minFlows := minFlows
      , maxCvSq := ((cvNum : ℚ) ^ 2) / ((cvDen : ℚ) ^ 2)
      , minPeriodMs := (minPeriodMs : ℚ) }

    let mut beaconResults : Array Json := #[]
    let beacons := ((j.getObjVal? "beacons").bind Json.getArr?).toOption.getD #[]
    for b in beacons do
      let gapsJ := ((b.getObjVal? "gaps_ms").bind Json.getArr?).toOption.getD #[]
      let gaps : List ℚ := gapsJ.toList.map (fun g => ((g.getInt?.toOption.getD 0 : Int) : ℚ))
      let leanB := decide (IsBeacon p gaps)
      let sqlB := ((b.getObjVal? "sql_is_beacon").bind Json.getBool?).toOption.getD false
      beaconResults := beaconResults.push <| Json.mkObj
        [ ("sa", (b.getObjVal? "sa").toOption.getD (Json.str "?"))
        , ("da", (b.getObjVal? "da").toOption.getD (Json.str "?"))
        , ("dp", (b.getObjVal? "dp").toOption.getD Json.null)
        , ("n_gaps", Json.num (JsonNumber.fromNat gaps.length))
        , ("lean_is_beacon", Json.bool leanB)
        , ("sql_is_beacon", Json.bool sqlB)
        , ("agree", Json.bool (leanB == sqlB)) ]

    let mut fanoutResults : Array Json := #[]
    let fans := ((j.getObjVal? "fanout").bind Json.getArr?).toOption.getD #[]
    for f in fans do
      let dsts : List String := (((f.getObjVal? "dsts").bind Json.getArr?).toOption.getD #[]).toList.map
        (fun d => d.getStr?.toOption.getD "")
      let leanF := decide (IsFanout fanoutMin dsts)
      let sqlF := ((f.getObjVal? "sql_is_fanout").bind Json.getBool?).toOption.getD false
      fanoutResults := fanoutResults.push <| Json.mkObj
        [ ("sa", (f.getObjVal? "sa").toOption.getD (Json.str "?"))
        , ("n_distinct_dst", Json.num (JsonNumber.fromNat dsts.length))
        , ("lean_is_fanout", Json.bool leanF)
        , ("sql_is_fanout", Json.bool sqlF)
        , ("agree", Json.bool (leanF == sqlF)) ]

    let mut returnResults : Array Json := #[]
    let rets := ((j.getObjVal? "returns").bind Json.getArr?).toOption.getD #[]
    for r in rets do
      let seq : List String := (((r.getObjVal? "dst_seq").bind Json.getArr?).toOption.getD #[]).toList.map
        (fun d => d.getStr?.toOption.getD "")
      let leanR := decide (HasReturnTrip seq)
      returnResults := returnResults.push <| Json.mkObj
        [ ("sa", (r.getObjVal? "sa").toOption.getD (Json.str "?"))
        , ("seq_len", Json.num (JsonNumber.fromNat seq.length))
        , ("lean_has_return_trip", Json.bool leanR) ]

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
        assocResults := assocResults.push <| Json.mkObj
          [ ("feat_a", (pa.getObjVal? "feat_a").toOption.getD (Json.str "?"))
          , ("feat_b", (pa.getObjVal? "feat_b").toOption.getD (Json.str "?"))
          , ("lean_chi2", Json.str (toString leanF))
          , ("sql_chi2", Json.str (toString sqlF))
          , ("agree", Json.bool (Float.abs (leanF - sqlF) ≤ tol)) ]

    -- timely-session verification (List.Chain' on exported start times)
    let mut sessionResults : Array Json := #[]
    if let some sj ← readJson? "data/sessions.json" then
      let mg := getNatD sj "max_gap_ms" (Params.chainMaxGapMs + 1)
      let ss := ((sj.getObjVal? "sessions").bind Json.getArr?).toOption.getD #[]
      for s in ss do
        let times : List ℚ :=
          (((s.getObjVal? "times_ms").bind Json.getArr?).toOption.getD #[]).toList.map
            (fun t => ((t.getInt?.toOption.getD 0 : Int) : ℚ))
        sessionResults := sessionResults.push <| Json.mkObj
          [ ("sa", (s.getObjVal? "sa").toOption.getD (Json.str "?"))
          , ("n_flows", Json.num (JsonNumber.fromNat times.length))
          , ("lean_is_timely_session",
             Json.bool (decide (IsTimelySession (mg : ℚ) times))) ]

    -- TLS: canonical-handshake per flow + rare-stack per fingerprint
    let mut tlsCanonical : Array Json := #[]
    let mut tlsRare : Array Json := #[]
    let mut tlsPartitionOk := true
    if let some tj ← readJson? "data/tls.json" then
      let tparams := (tj.getObjVal? "params").toOption.getD Json.null
      let rareMax := getNatD tparams "tls_rare_max_flows" Params.tlsRareMaxFlows
      let flowsJ := ((tj.getObjVal? "flows").bind Json.getArr?).toOption.getD #[]
      let allFps : List Fingerprint := flowsJ.toList.map
        (fun f => fpOf ((f.getObjVal? "fp").toOption.getD Json.null))
      -- runtime instance of the proved fpCounts_partition theorem
      tlsPartitionOk := ((fpCounts allFps).map Prod.snd).sum == allFps.length
      for f in flowsJ do
        let recs : List TlsRec :=
          (((f.getObjVal? "recs").bind Json.getArr?).toOption.getD #[]).toList.map tlsRecOf
        let leanC := decide (IsCanonicalHandshake recs)
        let sqlC := ((f.getObjVal? "sql_is_canonical").bind Json.getBool?).toOption.getD false
        tlsCanonical := tlsCanonical.push <| Json.mkObj
          [ ("flow_id", (f.getObjVal? "flow_id").toOption.getD Json.null)
          , ("n_records", Json.num (JsonNumber.fromNat recs.length))
          , ("lean_is_canonical", Json.bool leanC)
          , ("sql_is_canonical", Json.bool sqlC)
          , ("agree", Json.bool (leanC == sqlC)) ]
      let statsJ := ((tj.getObjVal? "fp_stats").bind Json.getArr?).toOption.getD #[]
      for s in statsJ do
        let hash := (((s.getObjVal? "fp_hash").bind Json.getStr?).toOption.getD "?")
        let fpJ? := flowsJ.find? (fun f =>
          (((f.getObjVal? "fp_hash").bind Json.getStr?).toOption.getD "") == hash)
        let entry : Json :=
          match fpJ? with
          | none =>
            Json.mkObj [ ("fp_hash", Json.str hash), ("agree", Json.bool false)
                       , ("error", Json.str "fingerprint hash not found among exported flows") ]
          | some fj =>
            let fp := fpOf ((fj.getObjVal? "fp").toOption.getD Json.null)
            let leanCount := allFps.count fp
            let leanRare := decide (IsRareStack rareMax fp allFps)
            let sqlCount := getNatD s "n_flows" 0
            let sqlRare := ((s.getObjVal? "sql_is_rare").bind Json.getBool?).toOption.getD false
            Json.mkObj
              [ ("fp_hash", Json.str hash)
              , ("lean_count", Json.num (JsonNumber.fromNat leanCount))
              , ("sql_count", Json.num (JsonNumber.fromNat sqlCount))
              , ("lean_is_rare", Json.bool leanRare)
              , ("sql_is_rare", Json.bool sqlRare)
              , ("agree", Json.bool (leanCount == sqlCount && leanRare == sqlRare)) ]
        tlsRare := tlsRare.push entry

    -- DNS: RFC 1035 well-formedness per distinct name + subdomain fans
    let mut dnsNames : Array Json := #[]
    let mut dnsFans : Array Json := #[]
    if let some dj ← readJson? "data/dns.json" then
      let dparams := (dj.getObjVal? "params").toOption.getD Json.null
      let maxLabel := getNatD dparams "dns_max_label_len" Params.dnsMaxLabelLen
      let maxName := getNatD dparams "dns_max_name_len" Params.dnsMaxNameLen
      let fanMin := getNatD dparams "dns_subfan_min_names" Params.dnsSubfanMinNames
      let namesJ := ((dj.getObjVal? "qnames").bind Json.getArr?).toOption.getD #[]
      for nj in namesJ do
        let labels : QName := strListOf nj "labels"
        let leanW := decide (IsWellFormedQName maxLabel maxName labels)
        let sqlW := ((nj.getObjVal? "sql_is_wellformed").bind Json.getBool?).toOption.getD false
        dnsNames := dnsNames.push <| Json.mkObj
          [ ("qname", (nj.getObjVal? "qname").toOption.getD (Json.str "?"))
          , ("n_labels", Json.num (JsonNumber.fromNat labels.length))
          , ("lean_is_wellformed", Json.bool leanW)
          , ("sql_is_wellformed", Json.bool sqlW)
          , ("agree", Json.bool (leanW == sqlW)) ]
      let fansJ := ((dj.getObjVal? "fans").bind Json.getArr?).toOption.getD #[]
      for fj in fansJ do
        let names := strListOf fj "names"
        let leanF := decide (IsSubdomainFan fanMin names)
        let sqlF := ((fj.getObjVal? "sql_is_fan").bind Json.getBool?).toOption.getD false
        dnsFans := dnsFans.push <| Json.mkObj
          [ ("registered_domain", (fj.getObjVal? "registered_domain").toOption.getD (Json.str "?"))
          , ("n_distinct_names", Json.num (JsonNumber.fromNat names.dedup.length))
          , ("lean_is_fan", Json.bool leanF)
          , ("sql_is_fan", Json.bool sqlF)
          , ("agree", Json.bool (leanF == sqlF)) ]

    IO.FS.createDirAll "out"
    let out := Json.mkObj
      [ ("claims", claims)
      , ("params", params)
      , ("beacons", Json.arr beaconResults)
      , ("fanout", Json.arr fanoutResults)
      , ("returns", Json.arr returnResults)
      , ("assoc", Json.arr assocResults)
      , ("sessions", Json.arr sessionResults)
      , ("tls_canonical", Json.arr tlsCanonical)
      , ("tls_rare", Json.arr tlsRare)
      , ("tls_partition_ok", Json.bool tlsPartitionOk)
      , ("dns_names", Json.arr dnsNames)
      , ("dns_fans", Json.arr dnsFans) ]
    IO.FS.writeFile "out/lean_results.json" (out.pretty)
    IO.println s!"checks: {beaconResults.size} beacon, {fanoutResults.size} fanout, {returnResults.size} return-trip, {assocResults.size} chi-square, {sessionResults.size} session, {tlsCanonical.size} tls-canonical, {tlsRare.size} tls-rare, {dnsNames.size} dns-name, {dnsFans.size} dns-fan checks -> out/lean_results.json"
    return 0

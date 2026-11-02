/-
Standalone Lean generator: the external synthesizer is now
    approved genspec + generation request → Lean sampler → Joy JSONL.

    lake exe generate <genspec.json> <hosts> <out_dir>

Reads ONLY the released genspec (distilled, typed — see GenSpec.lean) plus
the FlowShape/GenSpec constants that the linking theorems are about. Every
random draw runs the verified sampler (`sampleCellPMF_law`); records are
built through `mkFlow` (validity by construction, `mkFlow_valid`); session
times follow `stepTimes` (ordering, `stepTimes_chain`); chains are fuel-
bounded (`genChain`). Writes flow_synth.jsonl (Joy schema, exact-decimal
timestamps) and plan.json (the latent plan for the round-trip check).
No PyTorch, no Python, no neural weights.
-/
import EnclaveFlows.GenSpec
import EnclaveFlows.FlowShape

open Lean EnclaveFlows EnclaveFlows.Gen EnclaveFlows.FlowShape

def svcPort : String → Nat × Option Nat
  | "http" => (6, some 80)
  | "https" => (6, some 443)
  | "dns" => (17, some 53)
  | "ntp" => (17, some 123)
  | "chat" => (6, some 5222)
  | "ephemeral" => (6, none)
  | _ => (6, some 8888)

/-- Six synthetic client stacks (cs, scs, extensions) — public contents, the
MIX comes from the released tls_stacks histogram. -/
def tlsStackDefs : List (List String × String × List String) :=
  [ (["c02b", "c02f", "009e"], "c02f", ["server_name", "supported_groups"]),
    (["c02c", "c030", "009f"], "c030", ["server_name", "session_ticket"]),
    (["002f", "0035", "000a"], "0035", ["server_name"]),
    (["1301", "1302", "1303"], "1301", ["key_share", "server_name", "supported_versions"]),
    (["c013", "c014", "0039"], "c013", ["ec_point_formats", "server_name"]),
    (["00ff", "0005"], "0005", []) ]

/-- Canonical-handshake record skeleton: client hello first; per direction the
first CCS precedes the first appdata (IsCanonicalHandshake by construction). -/
def tlsSrlt : List (Nat × String × Nat × Nat) :=  -- (b, dir, ipt, tp)
  [ (517, ">", 0, 22), (3579, "<", 35, 22), (126, ">", 18, 22),
    (6, ">", 1, 20), (45, ">", 1, 22), (6, "<", 28, 20), (45, "<", 1, 22),
    (388, ">", 52, 23), (1448, "<", 41, 23) ]

def dnsAlphabet : String := "abcdefghijklmnopqrstuvwxyz0123456789"

def jn (n : Nat) : Json := Json.num (JsonNumber.fromNat n)

/-- Exact-decimal epoch seconds from milliseconds: mantissa·10⁻³. -/
def jtime (tMs : Nat) : Json :=
  Json.num ⟨Int.ofNat (epoch0 * 1000 + tMs), 3⟩

def tlsJson (stackIdx : Nat) (sni : String) : Json :=
  let (cs, scs, exts) := tlsStackDefs.getD stackIdx (["00ff"], "00ff", [])
  Json.mkObj
    [ ("c_version", jn 3), ("s_version", jn 3),
      ("cs", Json.arr (cs.map Json.str).toArray), ("scs", Json.str scs),
      ("sni", Json.arr #[Json.str sni]),
      ("c_extensions", Json.arr (exts.map (fun k => Json.mkObj [(k, Json.str "")])).toArray),
      ("srlt", Json.arr (tlsSrlt.map (fun (b, d, ipt, tp) => Json.mkObj
        [("b", jn b), ("dir", Json.str d), ("ipt", jn ipt), ("tp", jn tp)])).toArray) ]

def synthLabel (len : Nat) : IO String := do
  let mut s := ""
  for _ in [0:len] do
    let i ← draw (List.replicate dnsAlphabet.length 1)
    s := s.push (dnsAlphabet.toList.getD i 'x')
  pure s

def dnsJson (spec : GenSpec) (da : String) : IO Json := do
  let kind ← spec.dnsKind.choose
  let qn ←
    if kind.name = "kind_reverse_lookup" then
      pure (String.intercalate "." (da.splitOn ".").reverse ++ ".in-addr.arpa")
    else
      let ord := spec.qnameTopk.filter (fun c => !c.name.startsWith "_")
      let sd := spec.qnameTopk.filter (fun c => c.name.startsWith "_")
      let pool := if kind.name = "kind_service_discovery" then sd else ord
      if pool.isEmpty then
        -- gibberish shape-fallback under .test (declared)
        let nl ← weightedChoice [2, 3, 4, 5] (spec.dnsLabels.weights)
        let lenIdx ← draw spec.dnsLen.weights
        let long := lenIdx = 1
        let mut labels : List String := []
        for i in [0:nl - 1] do
          let ll ← if long ∧ i = 0 then do
              let x ← draw (List.replicate 9 1); pure (12 + x)
            else do
              let x ← draw (List.replicate 5 1); pure (3 + x)
          labels := labels ++ [← synthLabel ll]
        pure (String.intercalate "." (labels ++ ["test"]))
      else
        let i ← draw (pool.map (·.w))
        pure ((pool.getD i ⟨"x.test", 1⟩).name)
  pure (Json.arr #[Json.mkObj [("qn", Json.str qn), ("rc", jn 0)]])

/-- Render one Joy record. All derived fields go through `mkFlow`
(`mkFlow_valid`). Returns (json, newFlowCtr, newEphCtr). -/
def emit (spec : GenSpec) (flowCtr ephCtr : Nat) (sa da svc : String)
    (tMs : Nat) : IO (Json × Nat × Nat) := do
  let (pr, dpOpt) := svcPort svc
  let (dp, eph') := match dpOpt with
    | some d => (d, ephCtr)
    | none => (50000 + ephCtr % 15000, ephCtr + 1)
  let bo ← weightedChoice bytesOutChoices bytesOutWeights
  let bi ← weightedChoice bytesInChoices bytesInWeights
  let dur ← weightedChoice flowDurationMsChoices flowDurationMsWeights
  let f := mkFlow pr dp flowCtr bo bi tMs (max dur 1)
  let tls ← if svc = "https" then do
      let i ← draw spec.tlsStacks.weights
      let sni ← if spec.sniTopk.isEmpty then pure "svc.synth.test" else do
          let j ← draw (spec.sniTopk.map (·.w))
          pure ((spec.sniTopk.getD j ⟨"svc.synth.test", 1⟩).name)
      pure (tlsJson i sni)
    else pure Json.null
  let dns ← if svc = "dns" then dnsJson spec da else pure Json.null
  let rec' := Json.mkObj
    [ ("sa", Json.str sa), ("da", Json.str da),
      ("pr", jn f.pr), ("sp", jn f.sp), ("dp", jn f.dp),
      ("bytes_out", jn f.bytesOut), ("num_pkts_out", jn f.pktsOut),
      ("bytes_in", jn f.bytesIn), ("num_pkts_in", jn f.pktsIn),
      ("time_start", jtime f.tMs), ("time_end", jtime (f.tMs + f.durMs)),
      ("packets", Json.null), ("byte_dist", Json.null), ("entropy", Json.null),
      ("dns", dns), ("tls", tls),
      ("ip", Json.mkObj [("out", Json.mkObj [("ttl", jn 64)]),
                         ("in", Json.mkObj [("ttl", jn 57)])]),
      ("expire_type", Json.str "i") ]
  pure (rec', flowCtr + 1, eph')

def main (args : List String) : IO UInt32 := do
  let [specPath, hostsStr, outDir] := args
    | do IO.eprintln "usage: generate <genspec.json> <hosts> <out_dir>"; return 1
  let some nHosts := hostsStr.toNat?
    | do IO.eprintln "generate: hosts must be a natural"; return 1
  if nHosts = 0 ∨ nHosts > 50 then
    IO.eprintln "generate: hosts must be in 1..50"; return 1
  let txt ← IO.FS.readFile specPath
  let spec ← match Json.parse txt >>= GenSpec.parse with
    | .error e => do IO.eprintln s!"generate: bad genspec: {e}"; return 1
    | .ok s => pure s

  let mut flows : Array (Nat × Json) := #[]
  let mut flowCtr := 0
  let mut ephCtr := 0
  let mut nBeaconHosts := 0
  let mut planHosts : Array (String × Json) := #[]
  let mut planBeacons : Array Json := #[]

  for i in [1:nHosts + 1] do
    let sa := s!"10.99.0.{i}"
    let daPool := (List.range 6).map (fun j => s!"198.51.100.{(i * 7 + j) % 200}")
    let beaconDa := s!"198.51.100.{200 + i % 55}"
    let mut svcCounts : List (String × Nat) := []
    let mut sessionStarts : Array Nat := #[]
    let mut firstServices : Array String := #[]
    let mut beaconJson := Json.null
    let u ← draw (List.replicate hostStartMaxS 1)
    let mut tMs := u * 1000

    let mut hasBeacon : Bool := (← draw [1, 1]) == 1
    if i = nHosts ∧ nBeaconHosts = 0 then
      hasBeacon := true
    if hasBeacon then
      nBeaconHosts := nBeaconHosts + 1
      let p ← weightedChoice beaconPeriodMsChoices
                (List.replicate beaconPeriodMsChoices.length 1)
      let mut gaps : Array Nat := #[]
      for _ in [0:beaconFlowsMin - 1] do
        -- gap = period - jitter + U[0, 2*jitter]  (genBeacon_isBeacon)
        let j ← draw (List.replicate (2 * beaconJitterMs + 1) 1)
        gaps := gaps.push (p - beaconJitterMs + j)
      let mut tb := tMs
      let mut times : Array Nat := #[tb]
      for g in gaps do
        tb := tb + g
        times := times.push tb
      for t in times do
        let (r, fc, ec) ← emit spec flowCtr ephCtr sa beaconDa "https" t
        flows := flows.push (t, r)
        flowCtr := fc; ephCtr := ec
        svcCounts := match svcCounts.find? (·.1 = "https") with
          | some (_, n) => svcCounts.filter (·.1 ≠ "https") ++ [("https", n + 1)]
          | none => svcCounts ++ [("https", 1)]
      beaconJson := Json.mkObj
        [ ("sa", Json.str sa), ("da", Json.str beaconDa), ("dp", jn 443),
          ("period_ms", jn p), ("n_flows", jn times.size),
          ("planned_gaps_ms", Json.arr (gaps.map jn)) ]
      planBeacons := planBeacons.push beaconJson
      let extra ← draw (List.replicate interSessionExtraMs 1)
      tMs := tb + interSessionGapMsMin + extra

    let nSessIdx ← draw sessionsPerHostWeights
    let nSessions := sessionsPerHostChoices.getD nSessIdx 2
    for _ in [0:nSessions] do
      let nf ← weightedChoice sessionFlowsChoices sessionFlowsWeights
      sessionStarts := sessionStarts.push tMs
      let mut st := tMs
      let mut prev := ""
      for k in [0:nf] do
        if k > 0 then
          let step ← draw (List.replicate sessionStepMsMax 1)
          st := st + 1 + step
        -- distilled state machine: first flow ~ services, then ~ step prev
        let c ← (if prev = "" then spec.services else spec.step prev).choose
        if prev = "" then
          firstServices := firstServices.push c.name
        prev := c.name
        let daIdx ← draw (List.replicate 6 1)
        let (r, fc, ec) ← emit spec flowCtr ephCtr sa (daPool.getD daIdx "198.51.100.0") c.name st
        flows := flows.push (st, r)
        flowCtr := fc; ephCtr := ec
        svcCounts := match svcCounts.find? (·.1 = c.name) with
          | some (_, n) => svcCounts.filter (·.1 ≠ c.name) ++ [(c.name, n + 1)]
          | none => svcCounts ++ [(c.name, 1)]
      let extra ← draw (List.replicate interSessionExtraMs 1)
      tMs := st + interSessionGapMsMin + extra

    planHosts := planHosts.push (sa, Json.mkObj
      [ ("service_counts", Json.mkObj (svcCounts.map (fun (k, v) => (k, jn v)))),
        ("n_sessions", jn nSessions),
        ("session_starts_ms", Json.arr (sessionStarts.map jn)),
        ("session_first_services", Json.arr (firstServices.map Json.str)),
        ("beacon", beaconJson) ])

  let sorted := flows.qsort (fun a b => a.1 < b.1)
  IO.FS.createDirAll outDir
  let header := Json.mkObj
    [ ("version", Json.str "4.5.0"), ("synthetic", jn 1), ("bidir", jn 1),
      ("generator", Json.str "lake exe generate (Lean; genspec-driven)"),
      ("end-config", jn 1) ]
  let lines := #[header.compress] ++ sorted.map (fun (_, r) => r.compress)
  IO.FS.writeFile s!"{outDir}/flow_synth.jsonl"
    (String.intercalate "\n" lines.toList ++ "\n")
  let plan := Json.mkObj
    [ ("hosts", Json.mkObj planHosts.toList),
      ("beacons", Json.arr planBeacons),
      ("n_hosts", jn nHosts), ("n_beacon_hosts", jn nBeaconHosts),
      ("n_flows", jn sorted.size),
      ("generator", Json.str "lean") ]
  IO.FS.writeFile s!"{outDir}/plan.json" (plan.pretty)
  IO.println s!"generate: {sorted.size} flows ({nHosts} hosts, {nBeaconHosts} beacon channels) -> {outDir}/flow_synth.jsonl"
  return 0

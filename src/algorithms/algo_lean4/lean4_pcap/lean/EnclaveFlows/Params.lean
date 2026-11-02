/-
Mirror of the concept spec in dbt/dbt_project.yml `vars:` (spec 1.0.0).

These numbers are *mirrored*, not imported: tests/test_spec_sync.py
regex-parses the `def`s below and checks each equals the corresponding dbt
var (with the unit scalings named there). Everything else in this library
references these definitions, so one numeric literal is shared by the
proofs, the executables, Python and SQL.
-/

namespace EnclaveFlows.Params

def specVersion : String := "1.3.0"

/-- beacon_min_flows -/
def beaconMinFlows : Nat := 8
/-- beacon_max_cv = cvNum/cvDen (0.6) -/
def beaconMaxCvNum : Nat := 3
def beaconMaxCvDen : Nat := 5
/-- beacon_min_period_s × 1000 -/
def beaconMinPeriodMs : Nat := 1000
/-- fanout_min_dsts -/
def fanoutMinDsts : Nat := 20
/-- return_max_gap_s -/
def returnMaxGapS : Nat := 300
/-- session_gap_s -/
def sessionGapS : Nat := 600
/-- dp_clip -/
def dpClip : Nat := 4
/-- dp_clip_total (service-mix L1 contribution budget per monitored host) -/
def dpClipTotal : Nat := 128
/-- entropy_high × 100 -/
def entropyHighCentibits : Nat := 600
/-- chain_max_gap_s × 1000 -/
def chainMaxGapMs : Nat := 5000
/-- tls_rare_max_flows -/
def tlsRareMaxFlows : Nat := 3
/-- dns_max_label_len (RFC 1035) -/
def dnsMaxLabelLen : Nat := 63
/-- dns_max_name_len (RFC 1035) -/
def dnsMaxNameLen : Nat := 253
/-- dns_subfan_min_names -/
def dnsSubfanMinNames : Nat := 20

end EnclaveFlows.Params

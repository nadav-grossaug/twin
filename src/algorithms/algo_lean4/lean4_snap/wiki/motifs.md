---
title: Motif taxonomy
type: concept
updated: 2026-10-26
sources: [../dbt/models/marts, ../lean/EnclaveFlows/Motifs.lean]
---

# Motif taxonomy

A motif is a recurring interaction shape over flows — the network analogue of
the GPS return-trip motifs in the original encoder goal. Each motif has three
coordinated encodings (same thresholds, declared public):

1. a **SQL detector** (dbt mart, [[dbt-pipeline]]),
2. a **formal definition** in Lean, decidable over ℚ ([[lean-formal-layer]]),
3. an **executable check** (`lake exe checks`) running definition 2 on the
   data that detector 1 found, recording agreement.

| motif | SQL mart | Lean predicate | page |
|---|---|---|---|
| periodic re-contact | `mart_motif_beacon` | `IsBeacon` | [[beacons]] |
| destination/port fan-out | `mart_motif_fanout` | `IsFanout` | [[fanout-scan]] |
| A→B→A return trip | `mart_motif_return` | `HasReturnTrip` | [[return-trips]] |

Thresholds live in `dbt/dbt_project.yml` vars and are mirrored in
`python/export_lean_data.py`; they are data-independent by design
(the DP grid argument in [[dp-release]] relies on that discipline).

---
title: Return-trip motif (A→B→A)
type: concept
updated: 2026-10-26
sources: [../dbt/models/marts/mart_motif_return.sql, ../lean/EnclaveFlows/Motifs.lean]
---

# Return-trip motif (A→B→A)

Carried over from the GPS origin of the encoder: in a host's time-ordered
destination sequence, visiting A, then B ≠ A, then A again within bounded gaps
(≤ 300 s per hop; a ≥ 600 s silence breaks the sequence — censoring per
goal.md guideline 5).

In [[dataset-nitroba]] this captures browsing "hub-and-spoke" behavior:
a page host (hub) → asset CDN (spoke) → back to the hub. The gateway's
request/response chatter with 192.168.1.64 also matches (20+ occurrences
within 2 s windows).

Lean encodes it as the inductive `HasReturnTrip` with window inversion lemma
`hasReturnTrip_cons_iff` giving decidability; `lake exe checks` evaluates it
on exported destination sequences ([[lean-formal-layer]], [[motifs]]).

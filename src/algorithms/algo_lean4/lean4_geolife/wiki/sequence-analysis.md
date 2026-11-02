---
title: Sequence analysis (temporal flow logic)
type: concept
updated: 2026-10-26
sources: [../dbt/models/marts/mart_seq_transitions.sql, ../lean/EnclaveFlows/Sequences.lean]
---

# Sequence analysis — temporal flow logic

Three SQL views of time-ordered per-host flow sequences, one formal predicate,
one learned model:

- **`mart_seq_transitions`** — service bigram matrix P(next | current) with
  lift vs the marginal. Finding: traffic is strongly *self-transitioning*
  (bursts): ssdp→ssdp lift 22.8, https→https 16.3, dns→dns 6.2 at
  P=0.97 — runs of the same service dominate, which is exactly why the
  repeat-last baseline is strong for [[minimodels]]' NextServiceGRU.
- **`mart_seq_chains`** — resolve→connect chains (DNS followed ≤ 5 s by
  HTTP/HTTPS). Finding: the suspect host 192.168.15.4 has **zero**
  DNS-preceded connections (its resolver is inside the dorm NAT — a sensor
  blind spot worth teaching); the Mac 192.168.1.64 shows the expected chain
  with median 0.75 s latency ([[hosts-of-interest]]).
- **`mart_seq_sessions`** — maximal runs with consecutive start gaps ≤ 5 s:
  1,066 sessions, mean 5.1 flows, max 416.
- **Lean**: `IsTimelySession` (`List.IsChain` of ordered, gap-bounded steps),
  decidable, with monotonicity and chronology theorems; `lake exe checks`
  verified **12/12** exported SQL sessions satisfy it ([[lean-formal-layer]]).
- **Learned**: NextServiceGRU predicts the next service at 0.971 val accuracy
  vs 0.951 repeat-last and 0.854 marginal ([[minimodels]]).

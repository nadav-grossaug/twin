---
title: DNS name structure and domain population
type: concept
updated: 2026-10-27
sources: [../dbt/models/features/feat_dns.sql, ../dbt/models/marts/mart_dns_domains.sql, ../lean/EnclaveFlows/Dns.lean]
---

# DNS name structure and domain population

815 flows carry joy's `dns` block, but the export is query-heavy: 1,488
query records (`qn`) over **550 distinct names**; responses are nearly
absent (all rcode 0, 4 answer records with `a`/`ttl`). So the structure
worth preserving is the **qname wire shape** and the per-domain population
(`feat_dns`, one row per query record):

- labels, label/name lengths, digit fraction, service-discovery (`_`-prefix)
  and reverse-lookup (`*.arpa`) flags, plus `is_wellformed` — the RFC 1035
  shape (labels 1–63 bytes, name ≤ 253; spec 1.2.0 vars). All 1,488 queries
  are well-formed. The 9- and 34-label spikes are reverse lookups
  (`in-addr.arpa` / `ip6.arpa`), not anomalies.
- `registered_domain` = last two labels; `mart_dns_domains` aggregates per
  domain. **Subdomain-fan motif** (`is_subdomain_fan`, ≥ 20 distinct names,
  `dns_subfan_min_names`): the DNS tunneling/DGA/CDN shape. 6 fans here —
  the top one, `dlr.com` (68 names), is a **search-domain artifact**:
  DNS-SD queries (`_pop3._tcp.…`) re-suffixed with the local domain — a
  labeled false positive, like the gateway in [[fanout-scan]].

Formal mirror (`EnclaveFlows/Dns.lean`): names modeled as label lists (what
the wire carries), `nameLen` defined over them, so proofs are pure
arithmetic. `IsWellFormedQName` decidable; theorem `wellFormed_labels_le`
caps legal names at ⌊(253+1)/2⌋ = 127 labels. `IsSubdomainFan` is the
generic `IsFanout` over names (monotonicity inherited; `…_le_length` bounds
the fan by the query count). `lake exe checks`: **550/550 name verdicts and
12/12 fan verdicts agree with SQL** ([[lean-formal-layer]]).

χ² wiring: `dns_query_shape` (no_dns / single / multi / service_discovery /
reverse_lookup) in `feat_flow_cats` — Cramér's V 0.72 with dst_loc, 0.68
with fanout_source ([[feature-associations]]).

ML: **DnsCharLM** ([[minimodels]]) — character-level next-char LM over
qnames; per-name bits/char is the name-shape anomaly (the DGA-detector
shape). Val 2.355 bits/char vs 4.698 unigram baseline → **adopted**.
Flow-level `dns_name_score` (max over the flow's names) + `dns_top_qname`
in `ml_flow_scores`; the notebook lists the most unusual names (long DNS-SD
and reverse-lookup names top it — correctly, they are the rarest shapes).

Synthetic data (since 2026-10-30): every dns-service flow carries one
`{qn, rc:0}` query. The kind mix follows the DP-released `dns_shape`
histogram (ε 9/25 of the 4/1 ledger); ordinary and DNS-SD names are drawn
from the **common-knowledge top-k qname histogram** (`mart_dns_topk` →
policy artifact `dns_topk`: suppression-thresholded — ≥2 hosts, ≥10
queries, no `*.arpa` or single-label names — released OUTSIDE the ε ledger
as public metadata, like the service taxonomy); reverse lookups render from
the synthetic `da`, and a gibberish `.test` shape-fallback covers runs with
no released top-k ([[synthetic-generation]], [[dp-release]]).

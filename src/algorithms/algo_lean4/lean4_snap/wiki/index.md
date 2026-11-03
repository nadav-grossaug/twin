# Index — enclave flow encoder wiki

Read this first when answering a query; drill into pages from here.
Conventions: [SCHEMA.md](SCHEMA.md). History: [log.md](log.md).

## Entities
- [[dataset-snap]] — **the CURRENT target**: `facebook_full_features.graphml`, 4,039 people / 88,234 friendships, 10 ego networks, 2,280 ego-scoped anonymized features, circles; avg clustering 0.6055 reproduced; no temporal fields; the triangle heavy tail is the clipping risk.
- [[dataset-foursquare]] — the previous target: `country_transfer.tsv`, 84,964 international transfers by 29,640 users across 77 countries (2012-04-04 → 2013-09-16, Foursquare TIST2015 derived); measured profile, thresholds, the heavy-tail clipping risk and the 72.1% return-trip structure. Implemented in `../lean4_foursquare` (the ops substrate this repo ports).
- [[dataset-stock]] — the previous target: `stock_normalized_train.tsv`, 8 tickers × 168 trading days (2025-10-06 → 2026-06-05), base-100 normalized; profile, thresholds and the n=8 privacy risk. Implemented in `../lean4_stockpredict` (the code base this repo ports).
- **Previous capture (flows)**: ~407k Joy flows (replaced 2026-10-28; CIC-IDS-style, July 2017; 60,796 TLS flows / 33 stacks, 267k DNS queries). No dedicated page yet — EDA lives in the notebook; `make repair-data` fixes Joy's `},,{` bug per new export.
- [[dataset-nitroba]] — the ORIGINAL capture (5,397 flows, 2008, Nitroba case); historical context for [[findings]] and [[hosts-of-interest]].
- [[hosts-of-interest]] — Nitroba-era: 192.168.15.4 (dominant suspect host), 192.168.1.64 (Mac), 192.168.1.254 (gateway).

## Concepts
- [[cisco-joy-features]] — the Joy per-flow feature schema and its quirks.
- [[motifs]] — motif taxonomy hub: [[beacons]], [[fanout-scan]], [[return-trips]].
- [[feature-associations]] — χ²/Cramér's V co-occurrence between features and motifs.
- [[affiliation-communities]] — key-groups (people sharing one anonymized feature instance): measured cohesion vs the random-pair baseline, the four affiliation minimodels; enclave-only pending a ledger review.
- [[sequence-analysis]] — transitions, resolve→connect chains, timely sessions.
- [[tls-fingerprinting]] — TLS packet structure, JA3-style client stacks, canonical-handshake and rare-stack motifs.
- [[dns-structure]] — DNS qname wire shape, RFC 1035 well-formedness, subdomain-fan motif, char-LM name scores.
- [[dp-release]] — the differentially private export: privacy unit, proportional L1 clipping, the 389/100 ≤ 4/1 ledger (service + tls_stack + dns_shape + svc_transition histograms).
- [[verified-sampling]] — drawing from released histograms with a proved law.
- [[enclave-boundary]] — what stays inside, what may leave, and why.
- [[data-quality]] — known artifacts (negative durations, entropy > 8, zero-byte flows).

## Components
- [[architecture]] — the five layers, trust boundaries, the three release categories; read before changing structure.
- `../METHOD.md` — dataset-agnostic replication guide: goal, plan, techniques and
  implementation notes for rebuilding this flow on a different sensitive dataset.
- [[dbt-pipeline]] — staging → features → marts in DuckDB.
- [[minimodels]] — the fifteen PyTorch models and their current metrics.
- [[lean-formal-layer]] — what is proved, what is assumed, where.
- [[synthetic-generation]] — standalone Lean generator (`lake exe generate`) from the distilled genspec; proved contracts, kernel certificates, TLS/DNS structural rendering.
- [[distillation]] — teacher minimodels → typed genspec: the mapping table, privacy accounting, fidelity report.

## Syntheses
- [[practices]] — techniques & working rules with the incidents that earned them; read before repeating a mistake.
- [[findings]] — current best answers: top beacons, anomalies, false positives (Nitroba-era; re-derive on the new capture before citing).
- [[migration-plan]] — prioritized adoption plan from [[geolife-reference]] (all phases implemented).
- [[migration-plan-stock]] — the plan: flows pipeline → [[dataset-stock]] (METHOD §12 substitution map; **implemented 2026-10-31**).
- [[migration-plan-foursquare]] — the foursquare plan: stock/flows pipeline → [[dataset-foursquare]] (substitution map, six-target ops contract, `LEAN_HOME` policy, source-only bundle; implemented 2026-11-01).
- [[migration-plan-snap]] — **the ACTIVE plan**: foursquare pipeline → [[dataset-snap]] social graph (travel→graph substitution map, five PyG learning targets, row-level DP unit with declared node-DP gap; in progress 2026-11-02).

## Components (stock build)
- [[stock-pipeline]] — the as-built arch_03 stock pipeline: layer map, final ε ledger (400/100 exactly), honest model gates, validation + certificate results.

## Components (travel build)
- [[travel-pipeline]] — the as-built lean4_foursquare pipeline: layer map, 400/100 ledger, honest fidelity negative, planted recall 1.0, source-only bundle + `LEAN_HOME` ops substrate, declared gaps.

## Components (snap graph build)
- [[snap-pipeline]] — **the as-built lean4_snap pipeline**: block-clique generator, 385/100 ledger, triad_vgae honest negative, planted recall 1.0, validate PASS, gotchas earned.
- [[snap-contract]] — the frozen cross-layer interfaces of the snap port (spec vars, grids, mart shapes, genspec fields).

## References
- [[geolife-reference]] — the sibling GeoLife implementation (torch-lean-arch 01): technique inventory.

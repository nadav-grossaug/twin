# Index — merged enclave wiki (all lean4_* builds)

Read this first when answering a query; drill into pages from here.
This wiki is the aggregate of the six sibling builds (snap, geolife, booking, foursquare, pcap, stockpredict); each build's own CURRENT target is marked below. Conventions: [SCHEMA.md](SCHEMA.md). History: [log.md](log.md).

## Entities
- [[dataset-snap]] — **snap build's target**: `facebook_full_features.graphml`, 4,039 people / 88,234 friendships, 10 ego networks, 2,280 ego-scoped anonymized features, circles; avg clustering 0.6055 reproduced; no temporal fields; the triangle heavy tail is the clipping risk.
- [[dataset-geolife]] — the geolife build's target: `geolife_data.tsv`, 5,440,384 GPS points by 64 users in 4,464 trajectories (2007-04-12 → 2011-12-31, Microsoft GeoLife labelled subset, 83.8% Beijing); **timestamps are UTC** (staging shifts +8 h); measured profile, the n=64 small-n DP regime and the 734-vs-20 trajectories/user heavy tail.
- [[dataset-booking]] — **booking build's target**: `combined_set.csv`, 1,545,502 Booking.com multi-destination reservations (WSDM/SIGIR'21 challenge), 259,159 users, 288,348 utrips, 39,901 cities, 2015-12-31 → 2017-02-27; measured profile: 99.11% contiguous legs, median 5 legs/trip, 38.07% repeat-city trips, the next-destination learning target.
- [[dataset-foursquare]] — the foursquare build's target: `country_transfer.tsv`, 84,964 international transfers by 29,640 users across 77 countries (2012-04-04 → 2013-09-16, Foursquare TIST2015 derived); measured profile, thresholds, the heavy-tail clipping risk and the 72.1% return-trip structure. Implemented in `../lean4_foursquare` (the ops substrate this repo ports).
- [[dataset-stock]] — the stockpredict build's target: `stock_normalized_train.tsv`, 8 tickers × 168 trading days (2025-10-06 → 2026-06-05), base-100 normalized; profile, thresholds and the n=8 privacy risk. Implemented in `../lean4_stockpredict` (the code base this repo ports).
- [[dataset-cicids]] — **pcap build's target**: `flow.jsonl`, 406,938 Cisco Joy flows from a CIC-IDS-style cyber-defense training capture (Thursday 2017-07-06, ≈8 h; 15 internal hosts, 2,433 external one-flow sources; 60,796 TLS / 33 stacks, 333,965 DNS rows); measured profile and the small-n privacy regime.
- **Earlier capture (flows)**: ~407k Joy flows (replaced 2026-10-28; CIC-IDS-style, July 2017; 60,796 TLS flows / 33 stacks, 267k DNS queries). No dedicated page yet — EDA lives in the notebook; `make repair-data` fixes Joy's `},,{` bug per new export.
- [[dataset-nitroba]] — the ORIGINAL capture (5,397 flows, 2008, Nitroba case); historical context for [[findings]] and [[hosts-of-interest]].
- [[hosts-of-interest]] — Nitroba-era: 192.168.15.4 (dominant suspect host), 192.168.1.64 (Mac), 192.168.1.254 (gateway).

## Concepts
- [[cisco-joy-features]] — the Joy per-flow feature schema and its quirks.
- [[motifs]] — motif taxonomy hub: [[beacons]], [[fanout-scan]], [[return-trips]].
- [[feature-associations]] — χ²/Cramér's V co-occurrence between features and motifs.
- [[affiliation-communities]] — key-groups (people sharing one anonymized feature instance): cohesion vs the random-pair baseline, **edge coverage by category** (locale 0.76, education 0.68, gender 0.55 …) and the nesting of groups, the four affiliation minimodels, and the two ledger reviews that released their size histogram + cohesion table (key-level marts stay denied).
- [[sequence-analysis]] — transitions, resolve→connect chains, timely sessions.
- [[tls-fingerprinting]] — TLS packet structure, JA3-style client stacks, canonical-handshake and rare-stack motifs.
- [[dns-structure]] — DNS qname wire shape, RFC 1035 well-formedness, subdomain-fan motif, char-LM name scores.
- [[dp-release]] — the differentially private export: privacy unit, proportional L1 clipping, the flow-era 389/100 ≤ 4/1 ledger; **GeoLife section: release-3.0.0 (ε 29/export, owner's fidelity choice) and the lifetime ε ledger (33/33 spent)**. Snap build: 1369/100 ≤ 14/1 fidelity-first ledger ([[snap-pipeline]]).
- [[verified-sampling]] — drawing from released histograms with a proved law.
- [[enclave-boundary]] — what stays inside, what may leave, and why.
- [[data-quality]] — known artifacts (negative durations, entropy > 8, zero-byte flows).

## Components
- [[architecture]] — the five layers, trust boundaries, the three release categories; read before changing structure.
- `../METHOD.md` — dataset-agnostic replication guide: goal, plan, techniques and
  implementation notes for rebuilding this flow on a different sensitive dataset.
- [[dbt-pipeline]] — staging → features → marts in DuckDB.
- [[minimodels]] — the PyTorch models and their current metrics: fifteen flow-build models and the nine snap graph models (merged page, one section per build).
- [[lean-formal-layer]] — what is proved, what is assumed, where.
- [[synthetic-generation]] — standalone Lean generator (`lake exe generate`) from the distilled genspec; proved contracts, kernel certificates, TLS/DNS structural rendering.
- [[distillation]] — teacher minimodels → typed genspec: the mapping tables (flow and snap builds), release selection, privacy accounting, fidelity report.

## Syntheses
- [[practices]] — techniques & working rules with the incidents that earned them (incl. the 2026-11-03 GeoLife rules: verify time zones from data, lifetime ε ledger, small-n SNR check, never distill the latest export blindly, slack for planted truth); snap-build rules (release what survives its noise, validators must report what they skip, test the documented range, silent build failures, A/B sample sizes); read before repeating a mistake.
- [[findings]] — current best answers: top beacons, anomalies, false positives (Nitroba-era; re-derive on the new capture before citing).
- [[migration-plan]] — prioritized adoption plan from [[geolife-reference]] (all phases implemented).
- [[migration-plan-stock]] — the plan: flows pipeline → [[dataset-stock]] (METHOD §12 substitution map; **implemented 2026-10-31**).
- [[migration-plan-foursquare]] — the plan: stock/flows pipeline → [[dataset-foursquare]] (substitution map, six-target ops contract, `LEAN_HOME` policy, source-only bundle; implemented 2026-11-02).
- [[migration-plan-geolife]] — the geolife plan: travel/flows pipeline → [[dataset-geolife]] (substitution map; staypoints/anchors/daily motifs, χ² associations, GPS-noise moments, correlated-heading paths, Tokyo juxtaposition; implemented 2026-11-02, fidelity pass 2026-11-03).
- [[migration-plan-pcap]] — the pcap plan: arch_02 flow pipeline → [[dataset-cicids]] under the six-flow ops contract (`LEAN_HOME` store, ONE source-only bundle, `synth_from_bundle`; **implemented 2026-11-02**).
- [[migration-plan-snap]] — the snap plan: foursquare pipeline → [[dataset-snap]] social graph (travel→graph substitution map, five PyG learning targets, row-level DP unit with declared node-DP gap; in progress 2026-11-02).
- [[migration-plan-booking]] — **the booking plan (this build)**: foursquare pipeline → [[dataset-booking]] (substitution map, contract-first; in progress 2026-11-03).

## Components (booking build)
- [[booking-pipeline]] — the as-built lean4_booking pipeline: layer map, ε ledger 1008/100 ≤ 14/1, results and open issues.
- [[booking-contract]] — the frozen cross-layer interfaces of the booking port (grids, mart shapes, JSON schemas, generator I/O, planted motifs).

## Components (stock build)
- [[stock-pipeline]] — the as-built arch_03 stock pipeline: layer map, final ε ledger (400/100 exactly), honest model gates, validation + certificate results.

## Components (travel build)
- [[travel-pipeline]] — the as-built lean4_foursquare pipeline: layer map, 400/100 ledger, honest fidelity negative, planted recall 1.0, source-only bundle + `LEAN_HOME` ops substrate, declared gaps.

## Components (flow build)
- [[flow-pipeline]] — **the as-built lean4_pcap pipeline**: domain layer from arch_02 (unchanged data, unchanged calibration) on the six-flow ops substrate; ε ledger 389/100 ≤ 4/1; validation + certificate + bundle results.

## Components (GeoLife build)
- [[geo-pipeline]] — **the as-built lean4_geolife pipeline** (spec 4.1.0, release-3.0.0): UTC→local staging, implied-stay layer, 0 SQL↔Lean disagreements, ε 29/export + lifetime ledger, Lean-only generator (released cadence, length-conditioned mode, heavy-tailed per-user days, planted users), validate's volume/time checks, latest results and **ranked open issues**.

## Components (snap graph build)
- [[snap-pipeline]] — **the as-built lean4_snap pipeline**: block-clique generator + nine-category affiliation key-groups, 1369/100 ≤ 14/1 fidelity-first ledger, triad_vgae honest negative, planted recall 1.0, validate PASS, gotchas earned.
- [[snap-contract]] — the frozen cross-layer interfaces of the snap port (spec vars, grids, mart shapes, genspec fields).

## References
- [[geolife-reference]] — the sibling GeoLife implementation (torch-lean-arch 01): technique inventory.

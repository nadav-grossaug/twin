# Log (append-only)

## [2026-10-26] ingest | EDA of flow.jsonl
Full exploratory analysis of the Joy export (one subagent pass). Created
[[dataset-nitroba]], [[cisco-joy-features]], [[hosts-of-interest]],
[[data-quality]], [[motifs]] and the motif pages. Key surprises: 33% zero-byte
TCP flows; the gateway is a textbook port-scan false positive; cleanest beacon
is a Google server-side keepalive at 27.2 s (CV 0.029).

## [2026-10-26] ingest | dbt pipeline built and tested
18/18 dbt models+tests green. Created [[dbt-pipeline]], [[dp-release]],
[[enclave-boundary]]. mart_dns needed `from_json` because DuckDB types joy's
`dns` as JSON, not a list.

## [2026-10-26] ingest | minimodels v20261026_205804 trained
MPS device. Autoencoder val MSE 0.45; byte-dist classifier val acc 0.945 vs
0.521 majority. Created [[minimodels]]. Top anomaly = the external one-way
push flagged independently by EDA ([[findings]]).

## [2026-10-26] ingest | Lean formal layer compiled against SampCert
SampCert @ a2c53bb (Lean v4.34.1). Sensitivity, DP theorem, sampler law
proved; motif predicates decidable. Created [[lean-formal-layer]],
[[verified-sampling]]. Updated [[dp-release]] with ledger semantics.

## [2026-10-26] update | goal.md rewritten for the flow domain; notebook shipped
goal.md adapted from GPS trajectories to Joy network flows with an acceptance
status table. visualize_enclave_mo.py validated end-to-end (headless export:
12 charts, MapLibre country map, verified-sampler χ² consistent). HOWTO_run.md
and Makefile targets (incl. clean_dbt / clean) in place. Lean executables ran:
25/25 beacon, 15/15 fanout SQL↔Lean agreement; DP release within budget.

## [2026-10-26] ingest | association + sequence analysis across all three layers
χ² co-occurrence analysis (features × motifs) and temporal flow logic added:
dbt marts (feat_flow_cats, mart_assoc_cells, mart_feature_assoc,
mart_seq_transitions, mart_seq_chains, mart_seq_sessions — 24/24 green),
Lean Assoc.lean (exact ℚ chiSq, nonneg + independence theorems) and
Sequences.lean (IsTimelySession via List.IsChain), NextServiceGRU minimodel
v20261026_213039 (0.971 val acc vs 0.951 repeat-last). Checks: 6/6 χ²
SQL↔Lean exact agreement, 12/12 timely sessions. Created
[[feature-associations]], [[sequence-analysis]]; updated [[minimodels]],
[[lean-formal-layer]], notebook.

## [2026-10-26] ingest | review of the GeoLife sibling implementation
Three-subagent review of ~/tmp/test_torch_lean_arch_01 (same torch-lean-arch
on Microsoft GeoLife, carried through to synthetic generation). Created
[[geolife-reference]] (technique inventory: spec-sync tests,
content-addressed immutable artifacts, policy-as-data release gate with
attestation, composed histMech_DP budget theorem, GenCert kernel-checked data
certificates, DP-release-only synthesis with tiered validate, realism audit,
robustness/reproduce harnesses, remote bundle) and [[migration-plan]]
(5 phases, 20 items; Makefile gaps listed — notably our clean_lean deletes
lean/.lake where theirs preserves deps and offers distclean).

## [2026-10-27] update | migration plan IMPLEMENTED (all five phases)
Phase 1: spec single-sourced (dbt vars → python/spec.py → Params.lean,
test-locked); pytest suite (make test). Phase 2: content-addressed immutable
artifacts + registry (FLOW_RETRAIN gating, verified idempotent); run
manifests; reproduce PASSED (frozen rebuild hash-identical, CPU retrain
bit-stable); robustness 5/5 cases (one declared tolerance: thin_half beacon
precision 0.75 — cv re-estimation under subsampling). Phase 3: release gate
(config/release_policy.yaml, SimulatedTEE attestation, EgressGuard,
exact-fraction ε ledger, dev export verified); histMech_DP composed budget
theorem proved; exact-rational budget in Release.lean; safe_logging. Phase 4:
[[synthetic-generation]] — FlowShape.lean contracts proved, synth
generate/validate/certify end-to-end (beacon recall 1.0, TV 0.096,
kernel-only certificate 2.4 s), realism audit. Phase 5: eval adoption gates,
right-censoring flag, hasReturnTrip_iff_infix completeness, q_data_integrity
ledger, dbt unit test sharing the Lean fixture. Final: 26/26 dbt, 30/30
pytest, 17 proved claims, notebook validated. The synth round-trip audit
FOUND AND FIXED a real encoder bug (least() NULL entropy — [[data-quality]]).
Deferred: review_sample, remote-bundle.

## [2026-10-27] update | fix: `make synth` could miss the catsample binary
`lake build` only built the library (exes weren't default targets), so a
clean_lean + `make lean` cycle left catsample/checks unbuilt and `make synth`
failed. All four executables are now `@[default_target]` in lakefile.lean, and
the synth/certify targets build their own prerequisites. Verified: make synth
(20 hosts) exit 0; make certify proved (10 beacons, 45 sessions, kernel-only,
2.5 s); 30/30 pytest.

## [2026-10-27] ingest | per-column synth tests + visualize_synth_mo.py
Every synthetic column now has an explicit test of its fill quality
(tests/test_synth_columns.py, 13 tests): χ² GOF against the DECLARED
distribution for sampled columns (FlowShape bytes/duration weights, the
DP-released service mix), uniformity of session steps on (0,4000]ms and of
beacon jitter on [p−j,p+j] (the genBeacon_isBeacon bound), and structural
invariants (pkts derived from bytes, port/protocol map, address plan,
constant/null fields). New notebook **visualize_synth_mo.py**
(`make notebook-synth`): synthetic EDA through the production pipeline,
declared-vs-observed GOF charts, planned-vs-detected beacons, and a
**column-by-column enclave-vs-synth table** (summaries, TV distance,
NULL-rate change, fill source per column) with an interactive drilldown,
plus the realism-audit section. Headless-validated; full suite **43 passed**
(certificate regenerated after the entropy-fix revalidation — validate
correctly invalidates stale certificates). Updated [[synthetic-generation]],
HOWTO, Makefile.

## [2026-10-27] ingest | TLS packet structure + fingerprinting across all layers
Spec bumped to **1.1.0** (new var `tls_rare_max_flows: 3`). New feature model
`feat_tls` (JA3-style client-stack fingerprint + full srlt record-structure
features), marts `mart_tls_fingerprints` (8 stacks, 6 rare) and
`mart_tls_records` (8 handshake shapes; 111/112 canonical — flow 852 opens
with appdata), `tls_version`/`tls_stack` wired into the χ² substrate
(service_class × tls_version V≈0.81; `mart_feature_assoc` gained a `having`
guard against single-valued features on synthetic data). Formal layer: new
`EnclaveFlows/Tls.lean` — decidable `IsCanonicalHandshake` (accumulator scan
`NoEarlyData`) with soundness theorem `canonicalHandshake_data_has_ccs`,
`IsRareStack`, `fpCounts_partition`; 21 proved claims total; `lake exe
checks` re-decides on all 112 flows: full SQL↔Lean agreement. Fourth
minimodel **TlsRecordGRU** (stack from record shape, no payload): val 0.739 <
0.787 majority → `adopted: false` recorded honestly; `pred_tls_stack` in
`ml_flow_scores`. New DuckDB gotcha: `array_to_string([])` is NULL (fixed
with coalesce). 51/51 pytest (8 new in tests/test_tls.py incl. Python↔SQL
fingerprint recompute); synth round-trip, DP release (ε 3.4 ≤ 4.0), certify
(9 beacons, 47 sessions kernel-proved) re-verified. New page
[[tls-fingerprinting]]; updated [[minimodels]], [[cisco-joy-features]],
index, README, goal.md.

## [2026-10-27] ingest | DNS name structure across all layers + notebook TLS/DNS sections
Spec bumped to **1.2.0** (vars `dns_max_label_len: 63`, `dns_max_name_len:
253`, `dns_subfan_min_names: 20`). New `feat_dns` (1,488 query records, 550
names, qname wire-shape features) and `mart_dns_domains` (159 domains, 6
**subdomain fans** — top fan `dlr.com` is a search-domain DNS-SD artifact, a
labeled false positive); `dns_query_shape` in the χ² substrate (V 0.72 with
dst_loc). Formal layer: `EnclaveFlows/Dns.lean` models names as label lists
— `IsWellFormedQName` (RFC 1035, decidable) with proved bound
`wellFormed_labels_le` (≤ 127 labels), `IsSubdomainFan` = `IsFanout` over
names; **24 proved claims**; checks agree 550/550 names, 12/12 fans. Fifth
minimodel **DnsCharLM** (char-level qname LM): val 2.355 bits/char vs 4.698
unigram → **adopted**; `dns_name_score`/`dns_top_qname` in `ml_flow_scores`.
Notebooks: visualize_enclave_mo.py gained **TLS fingerprint/record-shape
sections** (stack population, record walk per top shape) and **DNS
sections** (domain fans, label distribution by name kind, char-LM
most-unusual names), five-model tiles and TLS/DNS rows in the Lean checks
accordion; visualize_synth_mo.py gained the **TLS & DNS declared-gaps**
section (real→synth 112→0 / 1,488→0, why the generator doesn't render them).
Both headless-export clean. **58/58 pytest** (7 new in tests/test_dns.py);
synth round-trip + kernel certificate re-verified (15 beacons, 31 sessions).
New page [[dns-structure]]; updated [[minimodels]], [[cisco-joy-features]],
index, README, goal.md.

## [2026-10-28] ingest | New larger pcap export — Joy `},,{` malformed-JSON repair
flow.jsonl replaced by a much larger Joy export (406,939 lines, CIC-IDS-style
192.168.10.0/24 capture, July 2017 timestamps). `make` aborted in `stg_flows`:
DuckDB `read_json` hit malformed JSON. Root cause: Joy's serializer writes an
**empty element into the server-cert "X509v3 extensions" array** (`},,{`) on
2 lines (157463, 345707 — both GlobalSign-cert HTTPS flows); Python
`json.loads` consumers (pytest raw fixtures, exporters) would have failed on
the same lines. Fix: new `scripts/fix_joy_jsonl.py` + `make repair-data` —
in-place, byte-minimal, verified repair (`},,{` → `},{` only on lines that
fail to parse; refuses anything unknown). Recorded in [[data-quality]].

## [2026-10-28] models | DnsCharLM scaled to the large capture (count-weighted dedup)
On the new capture the old per-record DnsCharLM training wedged: 213,737
train query records at batch 64 = ~3,340 GRU steps/epoch over 65 timesteps
on MPS (hours for 60 epochs; looked hung after epoch 0). Fix in
train_minimodels.py: train on **distinct names weighted by query count**
(9,209 sequences, batch 256 — ~90× fewer steps; the expected gradient equals
the per-record objective), via a weighted masked-NLL loss and a new optional
`sample_weights` in `train_loop`. The frozen EVAL definition is untouched —
validation stays record-level bits/char vs unigram. New active artifact
**036101d45d95**: AE val MSE 0.17/p95 0.79; ByteDist 0.951 vs 0.536 (6
classes); NextService 0.885 vs 0.882 repeat-last (thin margin); TlsRecordGRU
**0.838 vs 0.450 over 24 stacks → adopted** (the Nitroba honest-negative
flips with more data); DnsCharLM **2.448 vs 4.605 bits/char → adopted**.
[[minimodels]] updated; adopted flags are data-dependent — read meta.json.

## [2026-10-30] release | Synthetic TLS/DNS structure via two aux DP histograms
Closed the declared tls/dns-null gap now that the large capture makes the
aggregates dense. Two NEW DP releases through the same SampCert gate:
`mart_dp_tls` (TLS flows per client-stack RANK — stack_1..5/other; the
rank→fingerprint map stays in-enclave, no hash or cipher list leaves) and
`mart_dp_dns` (query records per public shape cell: kind / label-count band /
length band — no qname string reaches the mechanism). Release.lean now parses
optional `aux` histograms with their own clip/ε and sums the ledger as an
exact fraction: **17/5 + 6/25 + 9/25 = 4/1 — the budget, saturated exactly**
(policy release-1.1.0 lists all three). synth.py renders: every https flow a
`tls` block (six synthetic stacks over a fixed canonical-handshake srlt
skeleton → `IsCanonicalHandshake` by construction; mix = released rank
histogram via catsample) and every dns flow one `{qn, rc:0}` query whose
shape follows the released dns_shape cells (chars uniform a-z0-9, TLD .test,
reverse lookups from the synthetic da). Cipher/name REALISM stays a declared
gap — only shape crosses the gate. tests/test_synth_columns.py: nulls test
now covers packets/byte_dist/entropy only + two new tests (canonical srlt,
well-formed qnames, declared mix sources). Synth notebook section renamed
"rendered from the DP release". See [[synthetic-generation]], [[dp-release]].

## [2026-10-30] release | Rich synthetic qnames from a common-knowledge top-k artifact
Follow-up to the aux-histogram release: synthetic `dns.qn` was gibberish-only.
New `mart_dns_topk` (top-32 names queried by ≥2 hosts ≥10 times; `*.arpa` and
single-label names excluded — they encode internal IPs / local hostnames) is
released as a **suppression-thresholded policy artifact** (`dns_topk` in
release_policy.yaml, same category as flow_stats), NOT through the ε ledger:
popular public domains are common-knowledge metadata, like the service
taxonomy. export_lean_data.py stages it as lean/out/dns_topk.json next to the
DP release; a formal export carries dns_topk.csv. synth.py draws ordinary and
DNS-SD qnames ~ the released query counts via catsample (kind mix still from
the DP dns_shape histogram; reverse lookups from the synthetic da; gibberish
`.test` shape-fallback when no top-k is staged, declared in meta). Per-column
test asserts name provenance (top-k ∪ .test ∪ .in-addr.arpa).

## [2026-10-30] release | Synthetic TLS SNI from a common-knowledge top-k artifact
Same pattern as dns_topk: new `mart_tls_sni_topk` (top-32 SNI hostnames seen
from ≥2 hosts in ≥10 TLS flows; single-label names excluded) released as the
suppression-thresholded policy artifact `tls_sni_topk`, outside the ε ledger.
export stages lean/out/tls_sni_topk.json; synth.py draws each https flow's
`sni` ~ released flow counts via catsample (`svc.synth.test` fallback), with
`load_topk` generalized over both histograms. Per-column test asserts sni
provenance (top-k ∪ fallback). Fingerprints unaffected — sni is not part of
the JA3-style fp_string.

## [2026-10-30] update | Wiki consolidation: architecture + practices pages, staleness pass
Two new pages: [[architecture]] (the five layers dbt→PyTorch→Lean→release
gate→generator, trust boundaries proved-vs-assumed, the THREE release
categories — ε-DP histograms / suppression aggregates / common-knowledge
metadata — and the one-spec-three-mirrors sync discipline) and [[practices]]
(techniques & working rules with the incidents that earned them: repair-not-
drop, dedup+weight training, adoption gates with data-dependent flags,
decidable mirrors + agreement counts, exact-rational ledgers, budget
saturation as a feature, structural exclusion over thresholds, round-trip
audits, one-pipeline-driver-at-a-time). Staleness pass: index.md now leads
with the CURRENT ~407k-flow capture and marks [[dataset-nitroba]],
[[hosts-of-interest]] and [[findings]] as Nitroba-era; [[dbt-pipeline]]
rewritten to the full model list (feat_tls/feat_dns, assoc/seq marts, DP
input marts, top-k marts, q_data_integrity); [[lean-formal-layer]] gained
histMech_DP, the TLS/DNS theorems, FlowShape contracts, GenCert and the
27-claim ledger.

## [2026-10-30] ingest | THE BIG REWRITE: generation entirely in Lean from a distilled genspec
Two-stage architecture: private learning in the enclave, generation from an
executable Lean spec outside it. New: `mart_dp_transitions` (7×7 DP histogram,
the semi-Markov distillation of NextServiceGRU) with ε REBUDGET 17/10 + 6/25 +
9/25 + 49/50 = **82/25 ≤ 4/1** (policy release-1.2.0); `python/distill.py`
(`make distill`) → typed `lean/out/genspec.json` (post-processing of released
facts only; provenance test-asserted weight-by-weight) + enclave-only
distill_report.json; `EnclaveFlows/GenSpec.lean` (spec types/parser, verified
draw via sampleCellPMF, theorems stepTimes_sorted / mkFlow_valid / fuel-bounded
genChain); `lake exe generate` (Generate.lean) renders Joy JSONL + plan.json
ENTIRELY in Lean — exact-decimal timestamps, canonical TLS skeleton, topk/
shape DNS names; scripts/synth.py generate() is now a provenance-metadata
adapter (Pool/catsample Python sampling path REMOVED; catsample exe retained
for GenCert). Fidelity: raw DP rows too noisy (student 0.469) → public
self-transition prior blended in → student = repeat-last 0.881 vs teacher
0.885, accepted (declared criterion; [[distillation]]). Tests: new
tests/test_distill.py (provenance/shape/fidelity); service-mix GOF moved to
session-FIRST flows (the distilled law); TV tolerance 0.20→0.35 declared.
New page [[distillation]]; [[architecture]], [[practices]], [[dp-release]],
[[synthetic-generation]], index updated.

## [2026-10-30] models | Ten additional minimodels (fifteen total, artifact 96a949312e59)
train_minimodels.py grew from five to FIFTEEN models, same discipline: frozen
EVAL_DEFINITIONS entry, declared non-ML baseline, time-ordered 80/20 split,
one content-addressed artifact per run (new active version **96a949312e59**;
the original five reproduce their 036101d45d95 metrics exactly under the
shared seed). New: **ByteDistAE** (256→32→256 payload-shape anomaly, ungated,
val MSE 1.27e-05), **VolumeServiceMLP** (service from volume/timing numerics
only, 0.893 vs 0.443 majority), **BytesInMLP** (log bytes_in, MAE 0.439 vs
1.784 per-service train median), **DurationBandMLP** (5 public bands, 0.791
vs 0.451), **PairGapGRU** (next channel gap, MAE 1.093 vs 1.659 repeat-last),
**SessionLenMLP** (flow-count bucket from the FIRST flow, 0.469 vs 0.407),
**UnansweredMLP** (0.993 vs 0.992 — adopted by a whisker, margin ≈ the
positive rate), **TlsSetupBandMLP** (handshake-setup band from fingerprint +
record volume, 0.601 vs 0.496), **DnsKindClassifier** (query kind from name
shape, 0.999 vs 0.988 — the flags are shape rules, so near-perfect is
expected), **BeaconChannelMLP** (0.977 vs 0.956 — a consistency probe of the
is_beacon threshold rule, not a discovery). All ten gates PASSED on this
capture; the flags stay data-dependent — read meta.json. ml_flow_scores adds
four flow-level columns (`bd_recon_score`, `pred_service_vol`+confidence,
`bytes_in_abs_err`, `pred_unanswered_conf`); channel/session/TLS/DNS-record
models evaluate in meta only. New tests/test_minimodels_extra.py (5 tests:
entries, gate consistency, blobs, materialized scores) — suite 66 passed /
2 skipped. Notebook minimodels cell now shows the five tiles + a ten-row
adoption-verdict table. Updated [[minimodels]], [[architecture]], index.

## [2026-10-30] update | notebook charts: aggregates only (442 MB cell fix)

`make notebook` hit marimo's output cap: 442,608,500 bytes from one cell.
Cause: `alt.data_transformers.disable_max_rows()` + charts binning per-flow
dataframes client-side — fine at Nitroba scale, explosive at 407k flows.
Fixed four cells (enclave distributions, anomaly histogram, DNS name-shape;
synth headline comparisons) to bin in SQL/numpy and chart only counts, and
replaced disable_max_rows with `enable("default", max_rows=20_000)` in both
notebooks so oversized embeds fail loudly. Headless exports: enclave 3.0 MB,
synth 0.9 MB, zero tracebacks/truncations. Updated [[practices]].

## [2026-10-30] update | src+dst country maps; synth similarity overview

Enclave notebook geo section now enriches BOTH directions (external `sa`
sources, 2,433 IPs/8.8k inbound flows; external `da` destinations, 8,448
IPs/214k flows) with side-by-side tables and a two-layer MapLibre map
(orange=source, blue=destination, per-country circle pairs, legend).
Synth notebook gains: (a) a similarity-overview scorecard — per-column TV
bar chart colored by fill source (target/assumption/derived/structural) with
stat tiles for service-mix TV, median TV, share ≤0.2 and the distillation
fidelity (student vs repeat-last vs teacher GRU); (b) the same geography map
(enclave side only — synth renders documentation ranges 10.99.0.0/24 →
198.51.100.0/24; declared fields_not_modelled gap). Headless exports clean:
enclave 3.0 MB, synth 1.0 MB.

## [2026-10-31] update | DP service-mix recalibration (proportional L1, internal unit)

The large pcap inverted the released service mix: privacy unit "one sa" made
2,433 external one-flow sources outvote 15 monitored hosts (ephemeral 3,588 of
3,591 clipped), and uniform clip 4 saturated every major class — released
dns=0, http=0 while never-seen cells got pure noise. Redesign (policy
release-1.3.0, spec 1.3.0): privacy unit = monitored internal host with each
flow attributed to its internal endpoint; per-host contribution = PROPORTIONAL
L1 allocation of dp_clip_total=128 over the public 7-class genServices grid
(largest-remainder, mart_dp_counts); transitions scoped to internal sa.
Ledger rebudget: 7@2/5 + 6@1/25 + 9@1/25 + 49@1/100 = 389/100 <= 4/1.
Pre-noise mix now https 51 / http 31 / dns 14 / eph 2.2% — the released
histogram estimates the PER-HOST AVERAGE mix (user-level DP weighs the 15
hosts equally; the raw flow mix is dominated by the DNS server 10.3 + gateway
10.1, and flow-weighted budgets would collapse SNR to ~1 — measured, declared,
not pursued). Synth after redistill: DNS flows 1.1%→9.5%, service mix no
longer inverted; TLS ~50% tracks the released https share (real flow-level
TLS 14.9% — residual, documented gap: per-host vs flow semantics + the
generator rendering TLS on every https flow). Suite 67 passed / 1 skipped.
Updated [[dp-release]], [[architecture]], [[distillation]], [[practices]].

## [2026-10-31] ingest | Stock dataset profiled; goal.md rewritten for the stock migration

New raw source `stock_normalized_train.tsv`: 1,344 rows = 8 tickers
(CHKP, CRWD, FTNT, NVDA, OKTA, PANW, PLTR, ZS) × 168 trading days,
2025-10-06 → 2026-06-05, base-100 normalized, weekday calendar, clean parse
(no repairs, no dups, no holes). Shock days up to +30.1% (OKTA) / −31.5%
(ZS). Wrote [[dataset-stock]] (entity, measured profile) and
[[migration-plan-stock]] (synthesis: METHOD §12 substitution map — ticker =
protected entity, 5-class return grid as the public taxonomy, motifs
runs/shock/drawdown/V-recovery/vol-burst, weekly shapes, rank labels; draft
ε ledger 39/10 ≤ 4/1). Rewrote `../goal.md` from the flows framing to the
stock plan (all acceptance rows ◻ planned; phase 0 ✅). Flagged the dominant
risk: only n = 8 protected entities ⇒ Phase 6 pre-noise SNR simulation is a
hard gate before committing the ledger ([[dp-release]] lessons applied).
Updated index.md and SCHEMA.md raw sources.

## [2026-10-31] ingest | Stock pipeline implemented end-to-end (dbt + torch + Lean + DP + synth + notebooks)

Full build in one session, scaffold APFS-cloned from arch_02 (incl. 8.7 GB
.lake — Mathlib/SampCert cache made `lake build` a 7,817-job incremental).
Lean lib renamed EnclaveFlows→EnclaveSeries; mechanism files (DPRelease,
Sampler, Stats, Sequences, Assoc) reused VERBATIM — they are dataset-agnostic
by design, which validated METHOD §12's "keep unchanged" list. As-built map
and numbers: [[stock-pipeline]]. Suite 51 passed; `make synth` validate
exit 0 (planted shock recall 1.0, class-mix TV 0.146, 0 Lean disagreements);
certificate kernel-checked; policy export green (4 files, dev-attested).

Incidents earned (the n=8 calibration war, as goal.md predicted):
1. **Pair-count co-movement drowned**: 3 bins over 28 pairs with sensitivity
   n−1=7 at ε 1/25 → noise scale 175, λ post-processed to 0. Redesign:
   per-ticker MEAN same-sign share, one count in one bin (clip 1), ε 1/5 →
   λ = 70%. Rule: never release pair-level counts over protected entities;
   release per-entity summaries (sensitivity 1).
2. **Ledger arithmetic slip**: 402/100 refused by `lake exe release` exit 2
   (the refusal path works!); mag bins trimmed to 1/150 → EXACTLY 400/100.
3. **Noisy aux rows need anchors**: generator marginal drifted (TV 0.305 vs
   genspec) because DP dow rows (scale 400) dominated the class blend and
   noise-shaped transition rows pulled the stationary distribution away.
   Fixes, both declared post-processing: dow rows rescaled to total 10;
   transition rows anchored to the released marginal (2× row mass) +
   diagonal persistence prior (1×) → TV 0.146, fidelity accepted=True.
4. **Honest negatives on 1,336 rows**: NextClassGRU 0.18 < repeat-last 0.29,
   SeasonalityMLP 0.28 < majority 0.33 — gates kept them out; the DP
   histograms are the generative source anyway. GARCH is real signal:
   6/8 tickers beat constant-vol NLL, persistence 0.92–0.999; surge>surge
   lift 1.92 in the transition mart confirms clustering in-class-space.
5. Planted-pattern index convention: plan rets index 0 = day 1 (day 0 has no
   return); certify derives segment as rets[start_idx-1 : +run_days].
Deferred: robustness harness, reproduce check (acceptance rows ◻).
Updated [[stock-pipeline]] (new), [[migration-plan-stock]], [[index]],
`../goal.md` (final ledger + ✅ statuses), `../HOWTO_run.md`, [[practices]].

## [2026-11-01] update | Off-enclave bundle (`make bundle`) + draw-robust distill prior

Generated dirs had been cleaned externally; full rebuild from sources
reproduced the SAME minimodel artifact version 1772b883c4c8 (content
addressing works) and a fresh DP release (ε = 400/100 again; λ now 46% —
non-reproducible draws, as declared). The fresh draw exposed a gate
fragility: the 1× diagonal persistence prior let per-draw transition noise
push the distilled student below repeat-last − 0.02 (accepted=False).
Raised to 2× row mass (declared public prior): the student's argmax is now
repeat-last across draws while DP rows still shape off-diagonal sampling
mass; provenance test updated; suite 43 passed / 8 skipped (synth wiped,
regenerate with `make synth`).

New: `make bundle` (scripts/make_bundle.sh) — self-contained off-enclave
generation bundle from the latest release/<id>/: `bin/generate` with the
SampCert FFI dylib install-name patched to @executable_path (the binary
records an absolute build path — otool showed it pointing into the arch_02
clone), `dp_release.json`, `distill.py` (duckdb import made lazy: the
fidelity report is enclave-only, so consumers need python3 stdlib ONLY),
README + sha256 MANIFEST. Verified in a fresh directory: manifest check →
distill (accepted=None, no enclave DB — correct) → generate 4×80 and 8×168.
This closes the reference repo's deferred remote-bundle item.

## [2026-11-01] update | Bundle goes source-only: binary trust eliminated

`make bundle` no longer ships compiled binaries. Rationale: MANIFEST.sha256
proved the shipped `bin/generate` was intact, not that it corresponded to
the verified sources — an opaque-binary trust gap. The bundle now carries
the Lean sources verbatim (EnclaveSeries/ proofs, all executable roots,
lakefile.lean, pinned lake-manifest.json, lean-toolchain) plus a new
`build_from_source.sh`: elan toolchain → `lake exe cache get` (best-effort
mathlib oleans) → `lake build`, so the Lean kernel re-checks every theorem
on the consumer's machine before anything runs; a failing proof fails the
build. Binary path moves to `lean/.lake/build/bin/generate` (built in
place, so no install-name patching; macOS dylib surgery deleted). Bundle
shrinks to ~148 KB and is no longer OS/arch-bound; cost: consumer needs
elan + network + several-GB first build (SampCert compiles locally).
Lean sources are code + public policy constants only, so the enclave
boundary is unchanged. Updated `../README.md`, `../HOWTO_run.md`.

## [2026-11-01] ingest | Foursquare country-transfer profiling; goal.md rewritten for the travel migration
Profiled `country_transfer.tsv` (84,964 transfers, 29,640 users, 77
countries, window 2012-04-04 → 2013-09-16): heavy-tailed travels/user
(median 2, p99 14, max 195), 72.1% consecutive-pair return rate, MY↔SG
corridor dominance, gap mean 40.7 d / sd 65.8 d, clean parse (no NULLs, no
self-transfers). Created [[dataset-foursquare]] and
[[migration-plan-foursquare]]; goal.md fully rewritten (protected entity =
user; 8-region public map; public bins for counts/gaps/durations/distances;
motifs: return, commuter corridor, tour, burst, long-stay; planned ledger
400/100 exactly with dow as a 0ε uniform prior). New ops briefs recorded:
six make flows only, `LEAN_HOME=/opt/data/lean` shared toolchain/package
store, single source-only bundle, synth consumes Lean sources + released
JSON only.

## [2026-11-02] ingest | travel pipeline implemented end-to-end; all six ops flows green
Full port from lean4_stockpredict landed (dbt 112/112; 65 pytest; `lake
build` clean, no sorry/native_decide). Created [[travel-pipeline]]. Results:
checks 154/154 SQL↔Lean agreements + 12/12 episodes; ε ledger spent exactly
400/100; all four minimodels adopted (NextRegionGRU 75.7% vs 72.0%
return-to-previous); distill fidelity HONEST NEGATIVE (first-order student
56.5% — the predicted second-order-returns failure; generator compensates
via released return_rate bands); synth validate exit 0 (corridor + tour
recall 1.0 + 1.0, travel-count TV 0.012); certificate kernel-checked;
`synth_from_bundle` rebuilt 7,817 jobs from the 188K source-only bundle and
validated. Incidents earned: (1) corridor "~761" was the repeat-count-only
candidate set — CV ≤ 0.8 keeps 398, the contract formula stood; (2) the
dbt ml node and the trainer drifted on artifact names mid-port (gap_ae.pt vs
profile_ae.pt) — content-addressed artifacts caught it at `make train`;
(3) the Lean release and Python distill briefly disagreed on the released
cell key (released vs noised_count) — canonicalized on noised_count;
(4) realism audit: synth return rate 0.20 vs real 0.82 (first-order walk
rarely routes back, so the second-order return draw under-fires) — top
candidate fix is a return-first generator, same released bands, no new ε.
Ops: LEAN_HOME=/opt/data/lean store (elan + mathlib cache + pinned packages)
seeded from the stock build; bundle's build_from_source.sh reuses it.

## [2026-11-02] plan | GeoLife migration opened: data profiled, goal.md rewritten
Fourth instantiation begins (flows -> stock -> travel -> GeoLife), closing
the loop to the method's reference implementation ([[geolife-reference]]).
Full-file DuckDB profile of `geolife_data.tsv`: 5,440,384 points, 64 users,
4,464 trajectories, 2007-2011, 100% mode-labelled (11 labels -> 6 public
classes), median sampling 2 s / step 4.9 m, 83.8% of points in the Beijing
bbox; quality: 1 invalid coord (lat 400), 159,651 duplicate
(user,traj,timestamp) groups, altitude -777 sentinel. Created
[[dataset-geolife]] + [[migration-plan-geolife]]; goal.md rewritten
(protected entity = user, n = 64 small-n regime; staypoint/anchor/daily-
motif backbone; chi2 association grids; gap-aware sequences; GPS-noise
moments as category-2 release; correlated-heading generator — no straight
lines / 90-degree turns; mandatory Tokyo manipulate step with Beijing-bbox
exclusion; planned ledger 400/100 exactly). Ops briefs carried over: six
make flows + help, LEAN_HOME=/opt/data/lean shared store, ONE source-only
bundle. Port base: ../lean4_foursquare; domain logic from the GeoLife
reference.

## [2026-11-02] ingest | GeoLife core pipeline landed: dbt + minimodels + Lean layer green, checks 0/16975 disagreements
dbt layer (27 models + 91 tests): stg_points (4.87M kept of 5.44M raw; 574,708
exact dup rows removed, 1 invalid coord), feat_steps/legs, feat_staypoints
python node, feat_anchors/visits/user_days, motif/assoc/seq/noise/kinematics
marts, 5 DP input marts (proportional-L1 clip macro). Minimodels all ADOPTED
on honest gates: NextRoleGRU 0.697 vs return-to-previous 0.672, DwellClassGRU
0.564 vs majority 0.480, ModeLegMLP 0.801 vs speed-rule 0.677, UserProfileAE
scoring-only. Lean: EnclaveGeo lib + all 5 executables build on the shared
LEAN_HOME store; `lake exe release` spent exactly 400/100 = 4/1; checks
re-decide 2,582 user-day motifs + 1,955 revisit anchors + 10,000 class-ladder
cells + 432 visit runs + 6 chi-squares with **0 disagreements**.
Incidents earned:
(1) **The labelled subset records movement, not dwells** — in-trajectory
staypoints alone gave 1 work anchor and 0 commute days; added IMPLIED
inter-trajectory stays (traj end -> next start at the same place = the dwell,
departure-censored when the places differ): 37/64 users with work anchors,
30 commute days, dwell long-tail restored. See [[dataset-geolife]].
(2) **Negative gaps**: a detected staypoint can overlap the implied stay at
the trajectory boundary; SQL's `gap <= break` accepted negative gaps, Lean's
Nat ordering refused — 31 disagreements. Fix: overlap is NOT a transition
(seq_ok needs gap in [0, break]); day chains also break on negative gaps;
errand pairs now require the seq-ok same-day return edge. Checks went to 0.
(3) **Ledger rebalance at n=64**: SNR = share x n x eps_cell is independent
of the clip, so the planned 7-histogram ledger drowned everything; committed
ledger concentrates on the backbone (motif 150/100, transitions 135/100,
dwell 65/100, departure 20/100, mode 30/100 = exactly 400/100, clips 10) and
drops staypoints/day + displacement to priors/category-2 ([[dp-release]]).
(4) GPS noise is strongly autocorrelated (staypoint residual lag-1 ~0.78-0.9,
tight-subset sd 23 m / MAD 14 m) — the generator's AR(1) noise model and its
released moments are justified by measurement ([[motifs]], noise section).

## [2026-11-02] ingest | GeoLife build complete: all six ops flows green end-to-end
Clean-slate `make train` green (dbt 29 models + 91 tests; 4/4 minimodels
adopted; Lean checks 16,975/16,975; release spent exactly 4/1; distill with
honest fidelity negative 0.39 vs 0.65 baseline; pytest; manifest 30 tables;
policy export 8 files; ONE 220K source-only bundle). `make synth` n=64:
308,130 points -> Tokyo manipulate (12 public districts, deterministic per
user) -> validate PASS exit 0: planted recall 1.0 (commuter 001, tour +
revisit beacon 002), mode TV 0.004, motif TV 0.30, collinear 2e-4, no 90-deg
atom (1.4e-3), noise sd 30.3 vs released 23.2 (in band), 0 points in the
Beijing exclusion box, re-ingested Lean crosscheck 0 disagreements across all
10 families. `make synth_from_bundle`: build_from_source re-checked all
theorems (7,806 jobs) off the shared LEAN_HOME store, bundle-mode distill
(no db, fidelity skipped by design), 316,104 points, validate PASS.
`make certify`: GenCert PROVED (kernel decide, 4 theorems). pytest final:
110 passed, 0 skipped. Both marimo notebooks export headless, 0 tracebacks.
Created [[geo-pipeline]] (as-built). Incident worth keeping: Generate.lean
initially emitted duplicate timestamps at trip/stay segment joins (8/75
trajectories) — fixed with strict t < t1 emission guards; the per-column
monotonicity test now locks it.

## [2026-11-02] feature | MapLibre trajectory maps in both notebooks
Vendored the reference implementation's MapLibre GL anywidget
(python/maplibre_widget.py: tracks layer colored by RAW mode labels, popups,
basemap dropdown incl. a zero-network "none" style). Enclave notebook: one
LineString per labelled leg, SQL bucket-downsampled to <= 150k coordinates
(converged at 106 s -> 9,793 legs / 149,509 coords of 4.87M points), default
view = p08-p92 Beijing core, ENCLAVE-ONLY framing (shows individual
movement). Synth notebook: per-sequence lines from the Tokyo-placed
synthetic TSV (33 s -> 1,924 sequences / 147,928 coords), guarded by the
"run make synth" placeholder. Both export headless with 0 tracebacks; the
synth map was exercised against the real 308k-point synth output (geography
badges 2 PASS). DuckDB quirk earned: `minutes` is a reserved word — alias
renamed.

## [2026-11-02] feature | route-shape layer: road-like synthetic trajectories
User observation: synthetic trips followed straight vectors (the steered
heading walk collapsed to the target bearing; synth straightness ~1.0 vs
real 0.45-0.77). Fix = measure, release, regenerate:
MEASURED (new dbt layer): legs distance-resampled to 25 m; junction turn =
|turn| >= 30 deg (8.9% of resampled steps); junction angles 53/30/17% over
30-60/60-120/120-180 deg (left/right 50.7/49.3); junction-to-junction
segments median 50 m (p90 586 m); circuity medians walk 1.99 / car 1.49 /
bus+bike 1.37 / rail 1.27; junction density walk 11.1/km ... rail 0.2/km.
New public bins (spec v4): seg_e* 50/150/600 m, ja_e* 60/120 deg, circ_e*
1200/1500/2500 permille; feat_route_points/segments/legs +
mart_route_shape/seg_hist/junction_hist (category-2, suppression);
7th chi2 pair mode x circ_class (Cramer's V 0.275).
GENERATOR: trips are now waypoint POLYLINES — segment lengths from the
released seg histogram scaled by the mode's released junction density,
junction angles from the released angle histogram (in-class jitter, never
exactly 90), left/right from the released share with cumulative deviation
clamped to +-75 deg (forward progress), path budget from released per-mode
circuity, affine-fit onto the anchor pair. New linking theorems:
jaRange_is_junction, segRange_resolvable. Lean checks now re-decide 3 more
class ladders (seg/ja/circ) — still 0 disagreements (8 ladders).
Calibration incidents: (1) unconstrained heading walk MEANDERED (synth
straightness 0.30-0.40, junction share 29%) — fixed with per-mode segment
scaling + deviation clamp; (2) applying the released STATIC staypoint noise
(sd 23 m) to moving points inflated path length and faked junctions — moving
noise now sd/8 (declared prior: real 25 m-resampled tracks turn a median
2.2 deg; receivers filter in motion), stay wobble keeps full moments. After
both: synth straightness walk 0.51 / bike 0.69 / bus 0.77 / car 0.74 vs
real 0.45 / 0.72 / 0.73 / 0.67 — matched. Also: `by` is a reserved word in
Lean (tuple pattern (bx, by) fails with "Invalid pattern").

## [2026-11-02] ingest | route-shape layer verified end-to-end; two generator bugs fixed
Full chain green after the road-like rewrite: pytest 130 passed (route
minimodels RouteShapeGRU adopted by +0.7pp over marginal — honest near-tie —
and LegShapeAE scoring; ml_leg_scores node 9,825 legs), fresh release with
route_shape/seg_hist/junction_hist CSVs, genspec carries seg/junction
histograms + per-mode circuity & junction density, make synth: 279,876
points, validate recall 1.0, straightness bike 0.71 / car 0.71 / walk 0.61
(released 0.73/0.67/0.50), circuity synth 1.34-1.56, 0 hard failures;
make synth_from_bundle (240K bundle, kernel re-checked): 266,840 points,
recall 1.0, 0 hard failures; both notebooks (incl. MapLibre maps, now
showing bending streets) export clean.
Incidents earned:
(5) **Stale frame reuse**: synth.py's manipulate only moved the fresh
generator output into geolife_synth_frame.tsv when no frame existed — a
re-run on a dirty synth/ silently transformed the PREVIOUS run's frame and
discarded the new generation (caught by the generate-vs-manipulate point
counts disagreeing, 280,563 vs 261,823). cmd_generate now always replaces
the frame.
(6) **Un-snapped trip endpoints**: the polyline walker emitted points
strictly before the arrival instant, so a trajectory's last recorded fix
fell up to path/120 (~300 m) short of the anchor at a DIFFERENT spot each
day — re-ingestion smeared user 001's two anchors into thirteen and commute
recall died (bundle validate caught it). The final fix now snaps onto the
route endpoint (+noise). Corollary cap: the curl-up fallback threshold rose
to e >= 0.4*cum (realized circuity <= 2.5, the top public class edge) so
every user frame provably fits a Tokyo placement; this trims walk's extreme
1.99 median circuity to ~1.56 in synth — declared tradeoff.

## [2026-11-02] update | incident 7: trajectory-id collision across overlapping days (HARD FAIL)

`make synth` hard-failed: "1 out-of-order timestamps within a (user, trajectory)".
User 013 had two trajectories sharing tid `20070716140700`: the previous day's
schedule spilled past midnight (long-stay `leave` landed at 2007-07-16 14:07:00
exactly) and the next day's departure draw independently produced 14:07:00.
`tidStr` is second-resolution, so the two trajectories collided and their
interleaved rows read as a time regression. Seed-dependent, ~1 in 250k points.

Fix (Generate.lean): per-user `tPrevEnd` tracks the last stay `leave` across
days; a day whose departure `tCur ≤ tPrevEnd` is pushed to `tPrevEnd + 60`.
Days now never overlap, so per-user trajectory start seconds strictly increase
and tids are unique by construction. Verified: `make synth` validate PASS
(order_violations 0, duplicate_timestamps 0, recall 1.0), certify PROVED,
bundle rebuilt, `make synth_from_bundle` validate PASS (recall 1.0), pytest 130.

## [2026-11-03] update | synthetic volume was ~5% of the source (FIXED)

64 synthetic users rendered ~250–310k points against the source's 5.44M
(≈ 5%). Three generator choices, compounding, none data-derived:
(1) `renderTripAlong` sampled every `max(3, dur/120)` s — a fixed ~120
points per trip regardless of length (a 1 h trip got one fix per 30 s,
real GeoLife logs every 1–5 s, median 2 s); (2) `renderWobble` sampled
stays every 30 s and capped them at 121 points; (3) a 14-day slice per user
while real users carry hundreds of recorded days (median 20 trajectories,
p90 170).

Fix (Generate.lean): trip AND stay cadence now come from the RELEASED
per-mode median sampling interval (genspec `kinematics.dt_ds`, already
category-2 released, previously only used as a fallback field) — 2 s for
walk/bike/bus/car, 1 s rail, 6 s other; stays use the walk cadence, and the
121-point stay cap is gone (recorded stays are ≤ 1 h, so ≤ 1,800 points).
The per-user slice is 24 days (public prior, sized for volume parity).
`genStayDtS` deleted from Motion.lean (dead); `drawU`'s 1,024-cell weight
list hoisted to a constant (it runs 4× per noise draw). Result: 6.46M points
for 64 users (≈ 1.19× source; ~100k/user), ~20 min generate on one core.
Validate surfaced a latent flake: the planted tour user's third stop can be
drawn 200 m from home (`dist_lt1k` starts at 200 m < `anchor_cell_m` 300),
folding into the home anchor → 2-stop day → tour not recalled. The planted
tour user (002 only) now redraws its frame until home + the three tour
stops are pairwise ≥ 4 anchor cells apart; ordinary users keep the released
displacement mix untouched.

## [2026-11-03] update | source timestamps are UTC, the pipeline read them as local (FIXED, spec 4.1.0)

The TSV's `datetime` / label columns are UTC, not Beijing local as
[[dataset-geolife]] claimed. Evidence: labelled trajectories and label
windows split at `16:00:00` / end at `15:59:59` every day (= Beijing
midnight), and the point-hour profile is near zero at 16–21 h (= 00–05
Beijing) and busy 00–13 h (= 08–21 Beijing). Nothing converted, so every
time-of-day concept was shifted 8 h: home = staypoint containing 03:00
"local" was really 11:00 Beijing (office hours — home/work anchors likely
swapped), work window 09–17 was really 17–01, departure/night bins shifted,
day boundaries at 08:00 Beijing, weekday flags off near midnight. Every
time-dependent DP histogram (motif, transitions, dwell, departure) and the
synth built from it inherited this.

Fix: new public spec var `source_utc_offset_s = 28800` (UTC+8, China has no
DST since 1991; international trips also read at +8 — declared), mirrored
in Params.lean (`sourceUtcOffsetS`); `stg_points` shifts `datetime` and
`ts_s` once, so every downstream model is local by construction. The Lean
generator still plans in local time (the released bins are local) and
writes rows / labels / trajectory ids in UTC like the source; user day 0
moved one day into the window so UTC output never precedes it. The synth
notebook and `test_points_inside_public_window` apply the same shift.
spec_version 4.0.0 → 4.1.0. Requires a full rebuild and RE-RELEASE: the
DP release is recomputed on the corrected features — a second ε = 4 spend
on the same 64 users, which release.py's per-release ledger does NOT
accumulate; the 2026-11-02 release (release/20261102T125506Z) is superseded
and must not be exported alongside the new one.

## [2026-11-03] update | lifetime ε ledger across exports (FIXED)

The release gate checked each export against `dp_total_epsilon_max` but
kept no memory across exports: re-exporting the same 64 users (e.g. the
UTC re-release above) silently spent a second ε = 4. Pure DP composes over
releases, so the real guarantee was ε = 8 with nothing recording it.

Fix (`python/release.py`, policy `dp_lifetime_epsilon_max: "4/1"`): an
append-only ledger `config/epsilon_ledger.jsonl` (outside every `make
clean` target, git-auditable; ε amounts and hashes only) keyed by the
source TSV's sha256. `export()` reconciles it with every
`release/*/manifest.json` first (so no export can go unrecorded; manifests
without a fingerprint are attributed to the current dataset —
over-counting is the safe direction), refuses an export that would exceed
the lifetime cap, and appends the spend BEFORE writing any file. Deleting
an export directory never refunds. An export that provably never left the
enclave can be voided with `python3 python/release.py --void <id> --reason
"..."` (append-only audited entry). The live ledger was seeded with
`20261102T125506Z` (4/1), so the next export is REFUSED until the owner
either voids that export or raises the cap — both deliberate privacy
decisions. 6 new tests in test_release_gate.py.

## [2026-11-03] update | validate now guards time-of-day and volume (FIXED)

The 5%-volume and shifted-hour regressions passed validate because it
never looked at either axis. `synth.py validate` now (soft, released
references only): departure-bin mix of re-detected visits vs the released
DP departure weights (TV ≤ 0.35), and per-mode median sampling interval
(re-ingested `mart_kinematics`) within [0.5×, 2×] of the released cadence
(kinematics `dt_ds`) — on the pre-fix synth it flags every mode (walk 35 s
vs 2 s). It also REPORTS per-user point spread (CV) and the local
hour-of-day profile; those have no released reference, so the enclave-only
`realism_audit.py` compares them with the real cohort (total-volume ratio,
points/trajectories-per-user spread, hour-of-day TV). 2 new tests.

Finding (not fixed, design-level): the released DEPARTURE histogram is
mostly noise. ε_cell = 5/100 with a per-user L1 clip of 10 gives a
discrete-Laplace scale of 200 per cell against ~640 clipped user-mass over
4 bins; the new release came out morning 0 / night 0, the old one night
395 / evening 0. Neither reflects the data (real uncensored departures:
night 3%, morning/afternoon/evening ~1/3 each). Any time-of-day fidelity
needs either more ε on departure (displacing another histogram) or a
coarser grid.

## [2026-11-03] update | distill silently used the previous export (FIXED)

`make train` runs `lake exe release` then `distill` BEFORE `make export`,
and distill preferred the latest `release/<id>/` export. So every rebuild
distilled the PREVIOUS release (provenance-skip even reported "already
keyed" to it) while the fresh release sat unused in lean/out — the UTC
rebuild kept generating from the pre-fix release with no error, and
test_distill's fixtures skipped instead of failing. validate's route-shape
band and test_distill's `_released_rows` had the same latest-export trap.

Fix: `distill._select_release()` — explicit `$RELEASE_DIR` wins; inside the
enclave (warehouse present) a live `lean/out/dp_release.json` that differs
from the latest export (or with no export) IS the current release: distill
it, with category-2 inputs from the policy-filtered marts (what release.py
will export), and warn loudly that the genspec is pre-export; otherwise
(bundle: no warehouse) the latest export as before. The bundle still copies
only from `release/<id>/`, so an unexported release never ships.
genspec_meta records `release_source`; `synth.py` validate reads route
shape from the release genspec_meta names (falling back to the
policy-filtered mart); 4 new selection tests.

## [2026-11-03] update | ε reallocation for fidelity: release-3.0.0 (OWNER DECISION)

Owner: "improve synth data fidelity even if privacy bounds are worse off;
raise dp_lifetime_epsilon_max". At n = 64 the SNR of a DP cell is
≈ share × n × ε_cell, independent of the clip; release-2.0.0 (ε_cell
0.05–0.30, total 4) put the discrete-Laplace scale at 33–200 per cell
against ~640 clipped user-mass, and departure / mode came out mostly noise
(a release with morning = 0 departures broke both the hour-of-day profile
and the planted commuter). release-3.0.0: ε_cell = 1 in every histogram →
noise scale = one user's clip (10), per-export ε = 29 (5+9+5+4+6 cells).
Lifetime cap 33 = the 2026-11-02 export (4) + one release-3.0.0 export;
any further re-release needs another deliberate raise. This is a WEAK
formal guarantee, chosen knowingly. `export_lean_data.py` now reads every
ε and the budget from the policy (its hand-kept mirror is gone; sync test
added); the 4/1-pinned tests assert agreement with the policy instead.

Also (same day): planted users (001 commuter, 002 beacon/tour) depart in
a fixed public window `plantedDepartureH` = 07–08 h (theorem
`planted_departure_reaches_work_window`) so planted recall never hinges on
DP noise; non-planted users draw recorded days from the heavy-tailed public
prior `activeDayClasses` (median ~12, p90 ~96, mean ~25 days; theorems
`activeDayClasses_valid`, `planted_slice_covers_beacon`).

## [2026-11-03] ingest | release-3.0.0 run results + generator fixes

Export 20261103T130405Z (ε 29, lifetime 33/33 — the dataset's budget is now
fully spent). Released vs true clipped counts (TV): departure 0.027, dwell
0.028, motif 0.057, transitions 0.069, mode 0.079 (release-2.0.0 departure
was mostly noise). First synth on it: planted recall 1.0, certify PROVED,
hour-of-day TV 0.29 (from 0.40), per-user cv 1.24 (real 1.92; was 0.39),
1.19× real volume. It also exposed three generator bugs, fixed:

1. **One sequence_uuid per trajectory.** Real GeoLife: each sequence is
   exactly one labelled window (0/2,493 span two); synth: 30% did, so
   production feat_legs merged same-mode trips around a recorded stay into
   one out-and-back leg (bike/bus/rail straightness 0.43 vs released 0.73).
   `trajRows` starts a new sequence id at every label-window change.
2. **A cohort member with zero points** (63/64 users): a 2–4 day slice
   drawn as all home days rendered nothing. The last day is forced
   observable when nothing was observed.
3. **Walk point share 71% vs released 28%**: full-rate (2 s) stay wobble
   labelled walk. Stays are rendered at `genStayDtS` = 30 s again (trips
   keep the released cadence). New validate soft check: per-mode POINT share
   vs released kinematics n_points (TV ≤ 0.35; 0.43 on the buggy run).

Also: test_production_release_within_budget compared spent ε to a float 4.0;
now exact against the policy.

Follow-up (same day): the next run fixed straightness (bike/bus 0.70/0.72
vs released ~0.73) and 64/64 users, but (a) the planted tour broke again —
a tour leg at the 2 h trip cap equals `tour_max_gap_s` exactly, and coarser
re-detected stay boundaries pushed one gap to 7,520 s; the planted tour now
uses car legs with stops 1.2–5 km from home (huge slack) — and (b) walk was
still 65% of points: the trip MODE was drawn independently of trip length
(20 km "walks" at the cap, 2-minute "rail"). `chooseMode` now weights each
mode by released weight × log-normal(σ = 1, declared) plausibility of the
trip's path around the mode's released median path (`kinematics.median_path_m`,
now carried in the genspec as `path_m`). 8-user trial: per-mode point-share
TV vs release 0.38 → 0.17 (rail still under-represented: few anchor pairs
are rail-distance apart). The heavy-tail plan test threshold was relaxed
from p90 ≥ 4× median to 2.5× (the prior's expected ratio is ~5–8; n≈60).

## [2026-11-03] ingest | release-3.0.0 synth run: planted recall 1.0, pytest 144/144

After the planted-tour slack fix and length-conditioned mode choice:
`make synth` 2,885,038 points, validate soft tolerance (exit 2) — planted
recall 1.0; mode mix TV 0.046, motif 0.138, departure 0.212, per-mode
point share 0.224; bike/bus/car straightness 0.66/0.71/0.69 vs released
0.73/0.73/0.67 (fixed), but walk/other 0.04 vs 0.50/0.96 (soft). certify
PROVED; pytest 144/144; bundle 256K from export 20261103T130405Z. Realism
audit: volume 0.59× real (regressed from 1.31× — padded walk points gone,
day prior not recalibrated), per-user cv 1.16 (real 1.92), hour TV 0.27.
Open issues ranked in [[geo-pipeline]].

## [2026-11-03] lint | wiki + METHOD + HOWTO brought up to date

geo-pipeline rewritten to the current state (spec 4.1.0, release-3.0.0,
lifetime ledger, generator rules, validate checks, latest results, ranked
open issues); practices gained eight GeoLife rules; GeoLife sections added
to dp-release and distillation, pointer on synthetic-generation; dataset,
migration-plan and architecture annotated; index lines updated. Log headers
normalized to the SCHEMA `## [date] op | title` format (11 entries). METHOD:
verify time zones from data (§4), small-n SNR measurement + lifetime ledger
+ single-sourced ε (§6), distill-the-current-release (§8), render structure
and volume + planted-truth slack (§9), every fidelity axis checked against a
released reference (§10), ops notes (§11), checklist (§13). HOWTO_run:
ε from the policy, live-release distill, ledger/void, FLOW_RETRAIN, PID-
scoped kills, export contents. goal.md left as the original task framing.

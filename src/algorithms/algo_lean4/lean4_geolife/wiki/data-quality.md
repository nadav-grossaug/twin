---
title: Data quality findings
type: concept
updated: 2026-10-26
sources: [../flow.jsonl, ../dbt/models/features/feat_flows.sql]
---

# Data quality findings

Checked and handled in [[dbt-pipeline]] `feat_flows` (goal.md guideline 1:
define conventions before fitting models). Each fix leaves a quality flag.

- **Negative durations** — 647 flows, min −0.545 s, mostly −0.01 s on
  gateway RST pairs. Joy timestamp artifact on single-packet flows.
  → clamped to 0, flagged `qf_negative_duration`.
- **entropy > 8** — 14 flows (max 56.78); 8 bits/byte is the theoretical max,
  so these are Joy computation bugs. → capped at 8, flagged `qf_invalid_entropy`.
- **Zero-byte flows** — 1,801 (33%), all TCP: SYN/RST scaffolding and NAT
  reflection. Kept (they carry the [[fanout-scan]] false-positive signal),
  flagged `is_zero_byte`.
- **Unanswered flows** — 34% have no inbound packets (`is_unanswered`);
  treat as censored, not as errors (guideline 5).
- **Missing byte_dist/entropy** — 33% have all-zero byte_dist (no payload);
  [[minimodels]] classifier trains only on the 2,848 payload-bearing flows.
- **Joy `},,{` malformed-JSON bug** (found 2026-10-28 when a new, larger
  pcap export replaced flow.jsonl): Joy sometimes writes an **empty element
  into the server-cert "X509v3 extensions" array** — literally `},,{` — which
  makes the whole line invalid JSON and aborts DuckDB `read_json`, the pytest
  raw fixtures and the exporters (2 of 406,939 lines, both GlobalSign-cert
  HTTPS flows). → `scripts/fix_joy_jsonl.py` (`make repair-data`) repairs the
  file in place: it touches only lines that fail `json.loads`, applies the
  single known `},,{` → `},{` fix, verifies the line re-parses, and refuses
  to guess at anything else. Run it once per new Joy export.
- **DuckDB `least()` ignores NULLs** (found 2026-10-27 by the
  [[synthetic-generation]] round-trip audit): `least(entropy_raw, 8.0)`
  silently turned all 1,801 NULL-entropy flows into entropy = 8.0, polluting
  `has_entropy`, the entropy band, and model inputs. Fixed with an explicit
  NULL guard in feat_flows; models retrained. Lesson: cross-checking the
  pipeline against its own synthetic data finds encoder bugs, not just
  generator bugs.

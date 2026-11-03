---
title: Stock price training dataset
type: entity
updated: 2026-10-31
sources: [../stock_normalized_train.tsv]
---

# Stock price training dataset — the CURRENT target

`stock_normalized_train.tsv`: the raw source for the stock migration of the
pipeline (see [[migration-plan-stock]] and `../goal.md`). Tab-separated,
header `stockname  date-yyyy-mm-dd  stockprice`.

## Measured profile (2026-10-31)

- **1,344 data rows = 8 tickers × 168 trading days each**, 2025-10-06 →
  2026-06-05 (242 calendar days).
- Tickers: CHKP, CRWD, FTNT, NVDA, OKTA, PANW, PLTR, ZS — a
  cybersecurity-heavy basket plus NVDA. **Portfolio membership is the
  sensitive fact**: tickers leave the enclave only as rank labels.
- **Normalization**: every series starts at exactly 100.0 on 2025-10-06
  (base-100; absolute price levels are not present in the data — a declared
  public construction fact, and a `fields_not_modelled` gap for synth).
- **Calendar**: weekdays only (Mon–Fri counts 32/35/35/32/34); max calendar
  gap 4 days (long weekends); **no duplicate dates, no missing business days,
  no unparseable prices** — no verified repairer needed (recorded as a
  measured fact; keep the pattern in reserve).
- Per-ticker daily-return stats (simple %, population sd):

| ticker | last | min | max | mean ret | sd | max up | max down |
|---|---|---|---|---|---|---|---|
| CHKP | 67.6 | 56.0 | 101.7 | −0.20 | 2.52 | +6.2 | −19.6 |
| CRWD | 135.3 | 70.6 | 157.7 | +0.23 | 3.06 | +8.9 | −9.9 |
| FTNT | 167.5 | 87.1 | 173.3 | +0.34 | 2.68 | +20.0 | −6.3 |
| NVDA | 110.5 | 89.0 | 127.1 | +0.09 | 2.39 | +7.9 | −6.2 |
| OKTA | 126.7 | 67.2 | 149.2 | +0.22 | 3.98 | +30.1 | −10.9 |
| PANW | 128.0 | 66.6 | 141.4 | +0.18 | 2.64 | +9.3 | −7.4 |
| PLTR | 75.5 | 71.3 | 115.4 | −0.11 | 3.37 | +9.2 | −11.6 |
| ZS | 42.8 | 38.7 | 110.2 | −0.40 | 4.37 | +11.4 | −31.5 |

- **Shock days exist** (|return| ≥ 10%): OKTA +30.1%, FTNT +20.0%,
  ZS −31.5%, CHKP −19.6% — earnings-style jumps. The shock motif is the
  rarest pattern and the planted round-trip target.
- Daily-return sd range 2.4–4.4% → the public return-class edges
  (±0.5%, ±3%) and motif thresholds in `../goal.md` were picked from these
  measured distributions (METHOD phase 0 discipline).

## Privacy framing

- Protected entity = **one ticker** (whole series). **Only n = 8 entities** —
  the dominant calibration risk: METHOD §6's SNR estimate
  `share × n × ε / k` ≈ 1.3 on a 5-cell grid at ε = 4. The Phase 6 pre-noise
  SNR simulation is a hard gate before any ledger commit
  ([[migration-plan-stock]] phase 6, [[dp-release]] for the lessons).
- Equal row counts (168 each) mean per-entity-average and row-weighted
  distributions coincide on THIS file — verify again on any future data.
- Cross-ticker correlation is pairwise over protected entities: not released
  initially, declared gap.

The file name says *train*: an external test split exists elsewhere; treat
series ends as censored.

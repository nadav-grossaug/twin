-- Data-integrity ledger: one named counter per parsing/exclusion/quality
-- event, across pipeline stages, for an at-a-glance audit trail.
-- Counts only — safe to eyeball, still in-enclave.
select 'prices_parsed' as counter, count(*)::bigint as n, 'stg_prices' as stage
from {{ ref('stg_prices') }}
union all
select 'nonpositive_prices', sum(case when qf_nonpositive_price then 1 else 0 end)::bigint, 'stg_prices'
from {{ ref('stg_prices') }}
union all
select 'weekend_rows', sum(case when qf_weekend then 1 else 0 end)::bigint, 'stg_prices'
from {{ ref('stg_prices') }}
union all
select 'return_days', count(ret_bp)::bigint, 'feat_returns'
from {{ ref('feat_returns') }}
union all
select 'gap_breaks', sum(case when qf_gap_break then 1 else 0 end)::bigint, 'feat_returns'
from {{ ref('feat_returns') }}
union all
select 'shock_days', sum(case when is_shock then 1 else 0 end)::bigint, 'feat_returns'
from {{ ref('feat_returns') }}
union all
select 'volburst_days', sum(case when is_volburst then 1 else 0 end)::bigint, 'feat_returns'
from {{ ref('feat_returns') }}
union all
select 'run_segments', count(*)::bigint, 'mart_motif_runs'
from {{ ref('mart_motif_runs') }} where is_run
union all
select 'drawdown_tickers', count(*)::bigint, 'mart_motif_drawdown'
from {{ ref('mart_motif_drawdown') }} where has_drawdown
union all
select 'vrecovery_troughs', count(*)::bigint, 'mart_motif_vrecovery'
from {{ ref('mart_motif_vrecovery') }} where is_vrecovery
union all
select 'trading_sessions', count(*)::bigint, 'mart_seq_sessions'
from {{ ref('mart_seq_sessions') }}

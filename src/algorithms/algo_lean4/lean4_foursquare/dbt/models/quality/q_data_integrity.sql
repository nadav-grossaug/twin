-- Data-integrity ledger: one named counter per parsing/exclusion/quality
-- event, across pipeline stages, for an at-a-glance audit trail.
-- Counts only — safe to eyeball, still in-enclave.
select 'transfers_parsed' as counter, count(*)::bigint as n, 'stg_transfers' as stage
from {{ ref('stg_transfers') }}
union all
select 'zero_prev_duration', sum(case when qf_zero_prev_duration then 1 else 0 end)::bigint, 'stg_transfers'
from {{ ref('stg_transfers') }}
union all
select 'zero_next_duration', sum(case when qf_zero_next_duration then 1 else 0 end)::bigint, 'stg_transfers'
from {{ ref('stg_transfers') }}
union all
-- rows whose prev or next country is off the public region map ('XX'):
-- counted here, filtered out of every DP input downstream
select 'unknown_region_rows', sum(case when qf_unknown_region then 1 else 0 end)::bigint, 'stg_transfers'
from {{ ref('stg_transfers') }}
union all
select 'self_transfers', sum(case when prev_country = next_country then 1 else 0 end)::bigint, 'stg_transfers'
from {{ ref('stg_transfers') }}
union all
select 'outside_window_rows',
       sum(case when ts_s < {{ var('window_start_s') }}
                  or ts_s >= {{ var('window_end_s') }} then 1 else 0 end)::bigint,
       'stg_transfers'
from {{ ref('stg_transfers') }}
union all
select 'dup_user_dt_rows', coalesce(sum(cnt - 1), 0)::bigint, 'stg_transfers'
from (
    select count(*) as cnt
    from {{ ref('stg_transfers') }}
    group by userid, dt
    having count(*) > 1
)
union all
select 'gap_pairs', count(gap_s)::bigint, 'feat_transfers'
from {{ ref('feat_transfers') }}
union all
select 'seq_breaks', sum(case when gap_s is not null and not seq_ok then 1 else 0 end)::bigint, 'feat_transfers'
from {{ ref('feat_transfers') }}
union all
select 'return_pairs', sum(case when is_return then 1 else 0 end)::bigint, 'feat_transfers'
from {{ ref('feat_transfers') }}
union all
select 'longstay_rows', sum(case when next_duration_s >= {{ var('longstay_min_s') }} then 1 else 0 end)::bigint, 'feat_transfers'
from {{ ref('feat_transfers') }}
union all
select 'episodes', count(*)::bigint, 'feat_episodes'
from {{ ref('feat_episodes') }}
union all
select 'tour_episodes', sum(case when has_tour then 1 else 0 end)::bigint, 'feat_episodes'
from {{ ref('feat_episodes') }}
union all
select 'corridor_motifs', count(*)::bigint, 'mart_motif_corridors'
from {{ ref('mart_motif_corridors') }} where is_corridor
union all
select 'tour_windows', count(*)::bigint, 'mart_motif_tours'
from {{ ref('mart_motif_tours') }} where is_tour
union all
select 'burst_windows', count(*)::bigint, 'mart_motif_bursts'
from {{ ref('mart_motif_bursts') }} where is_burst

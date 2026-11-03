-- Data-integrity ledger (goal.md acceptance row "Data integrity"): one named
-- counter per parsing/exclusion/quality event, across pipeline stages, for an
-- at-a-glance audit trail. Counts only — safe to eyeball, still in-enclave.
select 'flows_parsed' as counter, count(*)::bigint as n, 'stg_flows (joy header excluded)' as stage
from {{ ref('stg_flows') }}
union all
select 'zero_byte_flows', sum(case when is_zero_byte then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'unanswered_flows', sum(case when is_unanswered then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'negative_duration_clamped', sum(case when qf_negative_duration then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'invalid_entropy_capped', sum(case when qf_invalid_entropy then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'right_censored_active_timeout', sum(case when is_right_censored then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'flows_with_dns', sum(case when has_dns then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'flows_with_tls', sum(case when has_tls then 1 else 0 end)::bigint, 'feat_flows'
from {{ ref('feat_flows') }}
union all
select 'beacon_channels', count(*)::bigint, 'mart_motif_beacon'
from {{ ref('mart_motif_beacon') }} where is_beacon
union all
select 'fanout_sources', count(*)::bigint, 'mart_motif_fanout'
from {{ ref('mart_motif_fanout') }} where is_fanout
union all
select 'timely_sessions', count(*)::bigint, 'mart_seq_sessions'
from {{ ref('mart_seq_sessions') }}

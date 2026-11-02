-- Data-integrity ledger: one named counter per parsing/exclusion/quality
-- event, across pipeline stages, for an at-a-glance audit trail.
-- Counts only — safe to eyeball, still in-enclave.
with raw_n as (
    select count(*)::bigint as n
    from read_csv('{{ var("points_path") }}', delim = '\t', header = true,
                  all_varchar = true)
)

select 'raw_rows' as counter, n, 'source' as stage from raw_n
union all
select 'points_kept', count(*)::bigint, 'stg_points'
from {{ ref('stg_points') }}
union all
-- exact duplicate (user, trajectory, timestamp) rows reduced to one
select 'dup_rows_removed', coalesce(sum(dup_n - 1), 0)::bigint, 'stg_points'
from {{ ref('stg_points') }}
union all
-- invalid/missing coordinate or key rows dropped by the verified repair
select 'invalid_rows_dropped',
       (select n from raw_n)
         - count(*)::bigint
         - coalesce(sum(dup_n - 1), 0)::bigint,
       'stg_points'
from {{ ref('stg_points') }}
union all
select 'alt_missing_rows', sum(case when qf_alt_missing then 1 else 0 end)::bigint, 'stg_points'
from {{ ref('stg_points') }}
union all
select 'unknown_mode_rows', sum(case when qf_unknown_mode then 1 else 0 end)::bigint, 'stg_points'
from {{ ref('stg_points') }}
union all
select 'outside_window_rows', sum(case when qf_outside_window then 1 else 0 end)::bigint, 'stg_points'
from {{ ref('stg_points') }}
union all
select 'implausible_speed_steps', sum(case when qf_implausible_speed then 1 else 0 end)::bigint, 'feat_steps'
from {{ ref('feat_steps') }}
union all
select 'legs', count(*)::bigint, 'feat_legs'
from {{ ref('feat_legs') }}
union all
select 'route_points', count(*)::bigint, 'feat_route_points'
from {{ ref('feat_route_points') }}
union all
select 'route_junctions', sum(case when is_junction then 1 else 0 end)::bigint, 'feat_route_points'
from {{ ref('feat_route_points') }}
union all
select 'route_segments', count(*)::bigint, 'feat_route_segments'
from {{ ref('feat_route_segments') }}
union all
select 'staypoints', count(*)::bigint, 'feat_staypoints'
from {{ ref('feat_staypoints') }}
union all
select 'censored_staypoints',
       sum(case when arr_censored or dep_censored then 1 else 0 end)::bigint,
       'feat_staypoints'
from {{ ref('feat_staypoints') }}
union all
select 'anchors', count(*)::bigint, 'feat_anchors'
from {{ ref('feat_anchors') }}
union all
select 'users_with_work_anchor', count(distinct case when role = 'work' then user_id end)::bigint, 'feat_anchors'
from {{ ref('feat_anchors') }}
union all
select 'visits', count(*)::bigint, 'feat_visits'
from {{ ref('feat_visits') }}
union all
select 'seq_transitions', sum(case when seq_ok then 1 else 0 end)::bigint, 'feat_visits'
from {{ ref('feat_visits') }}
union all
select 'seq_breaks',
       sum(case when gap_s is not null and not seq_ok then 1 else 0 end)::bigint,
       'feat_visits'
from {{ ref('feat_visits') }}
union all
select 'user_days', count(*)::bigint, 'feat_user_days'
from {{ ref('feat_user_days') }}
union all
select 'tour_days', sum(case when motif_class = 'tour' then 1 else 0 end)::bigint, 'feat_user_days'
from {{ ref('feat_user_days') }}
union all
select 'commute_days', sum(case when motif_class = 'commute' then 1 else 0 end)::bigint, 'feat_user_days'
from {{ ref('feat_user_days') }}
union all
select 'revisit_anchors', sum(case when is_revisit then 1 else 0 end)::bigint, 'mart_motif_revisit'
from {{ ref('mart_motif_revisit') }}

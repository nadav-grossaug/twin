-- Temporal flow logic III: maximal timely sessions per host.
-- A session is a maximal run of flows from one host where consecutive start
-- gaps are <= chain_max_gap_s — the SQL counterpart of the Lean predicate
-- `IsTimelySession` (List.Chain'), re-verified by `lake exe checks` on
-- exported sessions.
with seq as (
    select
        sa,
        flow_id,
        time_start,
        service_class,
        case
            when time_start - lag(time_start) over w > {{ var('chain_max_gap_s') }}
                 or lag(time_start) over w is null
            then 1 else 0
        end as session_start
    from {{ ref('feat_flows') }}
    window w as (partition by sa order by time_start, flow_id)
),

sessions as (
    select
        *,
        sum(session_start) over (partition by sa order by time_start, flow_id) as session_id
    from seq
)

select
    sa,
    session_id,
    count(*)                        as n_flows,
    min(time_start)                 as session_start_t,
    max(time_start)                 as session_end_t,
    max(time_start) - min(time_start) as duration_s,
    count(distinct service_class)   as n_services,
    list(time_start order by time_start, flow_id) as start_times
from sessions
group by sa, session_id
order by n_flows desc

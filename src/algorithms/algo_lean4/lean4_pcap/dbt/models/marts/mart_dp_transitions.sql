-- Input to the DP release of the service-TRANSITION mix (NOT exportable):
-- the distilled replacement for the NextServiceGRU teacher (see
-- python/distill.py). Privacy unit: one MONITORED (internal) source host (sa),
-- matching mart_dp_counts; external responders' inbound bursts are not
-- session behaviour of a protected user. Cells are the PUBLIC 7x7 grid over
-- the generator service classes ("prev>next"); a pair counts when two flows
-- of the same host are adjacent in time within the session gap. Per-host
-- contribution per cell clipped at dp_clip.
with f as (
    select
        sa,
        time_start,
        case when service_class in ('http','https','dns','ntp','chat','ephemeral')
             then service_class else 'other' end as svc
    from {{ ref('feat_flows') }}
    where src_internal
),

pairs as (
    select
        sa,
        lag(svc) over (partition by sa order by time_start) as prev_svc,
        lag(time_start) over (partition by sa order by time_start) as prev_t,
        svc,
        time_start
    from f
),

grid as (
    select a.g || '>' || b.g as cell
    from (select unnest(['http','https','dns','ntp','chat','ephemeral','other']) as g) a
    cross join (select unnest(['http','https','dns','ntp','chat','ephemeral','other']) as g) b
),

per_user as (
    select
        sa,
        prev_svc || '>' || svc as cell,
        least(count(*), {{ var('dp_clip') }}) as clipped_n
    from pairs
    where prev_svc is not null
      and time_start - prev_t <= {{ var('session_gap_s') }}
    group by sa, cell
)

select
    g.cell,
    coalesce(sum(p.clipped_n), 0)::bigint  as clipped_count,
    count(p.sa)                            as n_contributing_users,
    {{ var('dp_clip') }}                   as clip
from grid g
left join per_user p on p.cell = g.cell
group by g.cell
order by g.cell

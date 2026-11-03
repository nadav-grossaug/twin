-- DNS query summary from joy's dns sub-records (present on ~15% of flows).
with queries as (
    select
        sa,
        da,
        time_start,
        unnest(from_json(dns, '[{"qn":"VARCHAR"}]')) as q
    from {{ ref('stg_flows') }}
    where dns is not null
)

select
    lower(q.qn)         as query_name,
    count(*)            as n_queries,
    count(distinct sa)  as n_clients,
    min(time_start)     as first_seen,
    max(time_start)     as last_seen
from queries
where q.qn is not null
group by lower(q.qn)
order by n_queries desc

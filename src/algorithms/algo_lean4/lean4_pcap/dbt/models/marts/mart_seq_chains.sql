-- Temporal flow logic II: resolve -> connect chains.
-- The classic timely sequence in traffic: a DNS flow followed within
-- chain_max_gap_s by an HTTP/HTTPS connection from the same host.
-- One summary row per host plus coverage statistics; gaps measured between
-- flow starts (asof semantics: the latest DNS at or before the connection).
with dns as (
    select sa, time_start from {{ ref('feat_flows') }} where service_class = 'dns'
),

conn as (
    select sa, da, service_class, time_start
    from {{ ref('feat_flows') }}
    where service_class in ('http', 'https')
),

matched as (
    select
        c.sa,
        c.service_class,
        c.time_start,
        max(d.time_start) as dns_time
    from conn c
    left join dns d
        on d.sa = c.sa
        and d.time_start <= c.time_start
        and c.time_start - d.time_start <= {{ var('chain_max_gap_s') }}
    group by c.sa, c.service_class, c.time_start
)

select
    sa,
    count(*)                                         as n_connections,
    sum(case when dns_time is not null then 1 else 0 end) as n_dns_preceded,
    avg(case when dns_time is not null then 1.0 else 0.0 end) as dns_preceded_frac,
    median(time_start - dns_time)                    as median_latency_s,
    quantile_cont(time_start - dns_time, 0.95)       as p95_latency_s
from matched
group by sa
having count(*) >= 5
order by n_connections desc

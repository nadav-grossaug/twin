-- Fan-out motif: one source contacting many distinct destinations (host scan
-- shape) or many distinct ports (port scan shape). The Nitroba gateway
-- (192.168.1.254) is a deliberate false positive of the port-count heuristic:
-- its "scan" is NAT/RST reflection to ephemeral client ports. The
-- zero_byte_frac column is what separates it from real browsing fan-out.
select
    sa,
    count(distinct da)                                   as n_distinct_dst,
    count(distinct dp)                                   as n_distinct_dport,
    count(*)                                             as n_flows,
    sum(bytes_total)                                     as bytes_total,
    avg(case when is_zero_byte then 1.0 else 0.0 end)    as zero_byte_frac,
    avg(case when is_unanswered then 1.0 else 0.0 end)   as unanswered_frac,
    min(time_start)                                      as first_seen,
    max(time_start)                                      as last_seen,
    (count(distinct da) >= {{ var('fanout_min_dsts') }}) as is_fanout,
    (
        count(distinct dp) >= {{ var('fanout_min_dsts') }}
        and avg(case when is_zero_byte then 1.0 else 0.0 end) > 0.5
    )                                                    as is_portscan_shape
from {{ ref('feat_flows') }}
group by sa
order by n_distinct_dst desc

-- Aggregate statistics per service class, plus a 'TOTAL' rollup row.
-- These are in-enclave stats; only the DP release mart may leave the boundary.
select
    coalesce(service_class, 'TOTAL')         as service_class,
    count(*)                                 as n_flows,
    count(distinct sa)                       as n_src,
    count(distinct da)                       as n_dst,
    sum(bytes_total)                         as bytes_total,
    sum(bytes_out)                           as bytes_out,
    sum(bytes_in)                            as bytes_in,
    sum(pkts_total)                          as pkts_total,
    min(duration_s)                          as min_duration_s,
    median(duration_s)                       as median_duration_s,
    avg(duration_s)                          as mean_duration_s,
    quantile_cont(duration_s, 0.95)          as p95_duration_s,
    max(duration_s)                          as max_duration_s,
    median(bytes_total)                      as median_bytes,
    quantile_cont(bytes_total, 0.95)         as p95_bytes,
    avg(entropy)                             as mean_entropy,
    sum(case when is_zero_byte then 1 else 0 end)      as n_zero_byte,
    sum(case when is_unanswered then 1 else 0 end)     as n_unanswered,
    sum(case when qf_negative_duration then 1 else 0 end) as n_qf_negative_duration,
    sum(case when qf_invalid_entropy then 1 else 0 end)   as n_qf_invalid_entropy
from {{ ref('feat_flows') }}
group by rollup (service_class)

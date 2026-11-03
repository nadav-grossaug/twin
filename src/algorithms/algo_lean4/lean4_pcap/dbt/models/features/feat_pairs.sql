-- Per-(sa, da, dp) repetition features: the substrate for beacon detection.
-- Gaps are inter-START times; cv_gap is the coefficient of variation
-- (low cv + many repeats = periodic candidate).
with ordered as (
    select
        sa,
        da,
        dp,
        pr,
        service_class,
        time_start,
        bytes_total,
        time_start - lag(time_start) over (
            partition by sa, da, dp order by time_start
        ) as gap_s
    from {{ ref('feat_flows') }}
)

select
    sa,
    da,
    dp,
    any_value(pr)                        as pr,
    any_value(service_class)             as service_class,
    count(*)                             as n_flows,
    min(time_start)                      as first_seen,
    max(time_start)                      as last_seen,
    sum(bytes_total)                     as bytes_total,
    avg(gap_s)                           as mean_gap_s,
    median(gap_s)                        as median_gap_s,
    stddev_samp(gap_s)                   as sd_gap_s,
    case when avg(gap_s) > 0
         then stddev_samp(gap_s) / avg(gap_s) end as cv_gap
from ordered
group by sa, da, dp

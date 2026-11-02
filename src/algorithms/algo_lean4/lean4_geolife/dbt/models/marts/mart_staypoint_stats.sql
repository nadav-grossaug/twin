-- Dwell structure per anchor role: visit counts, dwell moments and class
-- shares. ENCLAVE-ONLY ledger behind the dwell releases.
select
    role,
    count(*)                                   as n_visits,
    count(distinct user_id)                    as n_users,
    median(dwell_s)                            as median_dwell_s,
    quantile_cont(dwell_s, 0.95)               as p95_dwell_s,
    avg(dwell_s)                               as mean_dwell_s,
    stddev_samp(dwell_s)                       as sd_dwell_s,
    sum(case when is_longdwell then 1 else 0 end) as n_longdwell
from {{ ref('feat_visits') }}
where not arr_censored and not dep_censored
group by role
order by n_visits desc

-- Seasonality analysis (ENCLAVE-ONLY view; the DP counterpart is mart_dp_dow):
-- mean/median daily log return and class mix by day of week and by month.
with r as (
    select * from {{ ref('feat_returns') }} where ret_bp is not null
)

select
    'dow' as dimension,
    dow_name as bucket,
    count(*) as n_days,
    avg(ret_bp) as mean_bp,
    median(ret_bp) as median_bp,
    stddev_samp(ret_bp) as sd_bp,
    sum(case when ret_sign = 1 then 1 else 0 end)::double / count(*) as up_share
from r
group by dow_name

union all

select
    'month',
    'm' || lpad(month::varchar, 2, '0'),
    count(*),
    avg(ret_bp),
    median(ret_bp),
    stddev_samp(ret_bp),
    sum(case when ret_sign = 1 then 1 else 0 end)::double / count(*)
from r
group by month
order by dimension, bucket

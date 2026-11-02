-- GPS-noise statistical moments, measured where ground truth is known (the
-- device is stationary inside a staypoint): residuals vs the staypoint
-- centroid. Long format; suppression to suppress_min_users/points happens
-- in python/release.py (category-2: NO formal guarantee, declared).
with sp as (
    select * from {{ ref('feat_staypoints') }}
    where origin = 'sp' and n_points >= 10
),

tight as (
    -- tight staypoints: wander bounded well inside the detection radius, so
    -- residuals measure receiver noise rather than slow drift
    select * from sp where resid_p95_m <= {{ var('noise_tight_max_m') }}
)

select metric, value, n_staypoints, n_users from (
    select 'resid_sd_m_median' as metric, median(resid_sd_m) as value,
           count(*) as n_staypoints, count(distinct user_id) as n_users from sp
    union all
    select 'resid_sd_m_mean', avg(resid_sd_m), count(*), count(distinct user_id) from sp
    union all
    select 'resid_mad_m_median', median(resid_mad_m), count(*), count(distinct user_id) from sp
    union all
    select 'resid_p95_m_median', median(resid_p95_m), count(*), count(distinct user_id) from sp
    union all
    select 'resid_lag1_median', median(resid_lag1), count(*), count(distinct user_id) from sp
    union all
    select 'resid_lag1_mean', avg(resid_lag1), count(*), count(distinct user_id) from sp
    union all
    select 'tight_resid_sd_m_median', median(resid_sd_m), count(*), count(distinct user_id) from tight
    union all
    select 'tight_resid_mad_m_median', median(resid_mad_m), count(*), count(distinct user_id) from tight
    union all
    select 'tight_resid_p95_m_median', median(resid_p95_m), count(*), count(distinct user_id) from tight
    union all
    select 'tight_resid_lag1_median', median(resid_lag1), count(*), count(distinct user_id) from tight
)

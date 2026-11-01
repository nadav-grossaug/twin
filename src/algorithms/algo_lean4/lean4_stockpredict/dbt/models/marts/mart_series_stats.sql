-- Per-ticker distribution statistics of daily log returns: statistical
-- moments (mean, sd, skewness, kurtosis), extremes, lag-1 autocorrelation
-- (random-walk check: ~0 under the null), and class shares.
-- ENCLAVE-ONLY (per-ticker rows are identifying); DP releases go through
-- mart_dp_* instead.
with r as (
    select
        ticker,
        d,
        ret_bp,
        ret_class,
        lag(ret_bp) over (partition by ticker order by d) as prev_ret_bp
    from {{ ref('feat_returns') }}
    where ret_bp is not null
)

select
    ticker,
    count(*)                                      as n_days,
    avg(ret_bp)                                   as mean_bp,
    stddev_samp(ret_bp)                           as sd_bp,
    skewness(ret_bp)                              as skew,
    kurtosis(ret_bp)                              as kurtosis_excess,
    min(ret_bp)                                   as min_bp,
    max(ret_bp)                                   as max_bp,
    median(ret_bp)                                as median_bp,
    quantile_cont(ret_bp, 0.05)                   as p05_bp,
    quantile_cont(ret_bp, 0.95)                   as p95_bp,
    corr(ret_bp, prev_ret_bp)                     as autocorr1,
    -- volatility clustering signal: autocorrelation of |r| (GARCH footprint)
    corr(abs(ret_bp), abs(prev_ret_bp))           as abs_autocorr1,
    sum(case when ret_class = 'crash' then 1 else 0 end) as n_crash,
    sum(case when ret_class = 'down'  then 1 else 0 end) as n_down,
    sum(case when ret_class = 'flat'  then 1 else 0 end) as n_flat,
    sum(case when ret_class = 'up'    then 1 else 0 end) as n_up,
    sum(case when ret_class = 'surge' then 1 else 0 end) as n_surge
from r
group by ticker
order by ticker

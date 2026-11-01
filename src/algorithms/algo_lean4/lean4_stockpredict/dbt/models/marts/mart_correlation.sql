-- Inter-stock collinearity (ENCLAVE-ONLY: pairwise rows over protected
-- entities never leave; the DP counterpart is mart_dp_comove): pairwise
-- Pearson correlation of daily log returns and the same-sign-day share.
with r as (
    select ticker, d, log_ret, ret_sign
    from {{ ref('feat_returns') }}
    where log_ret is not null
)

select
    a.ticker                                       as ticker_a,
    b.ticker                                       as ticker_b,
    count(*)                                       as n_days,
    corr(a.log_ret, b.log_ret)                     as pearson_r,
    avg(case when a.ret_sign = b.ret_sign
             and a.ret_sign != 0 then 1.0
             when a.ret_sign != 0 or b.ret_sign != 0 then 0.0
        end)                                       as same_sign_share
from r a
join r b
  on a.d = b.d and a.ticker < b.ticker
group by 1, 2
order by pearson_r desc

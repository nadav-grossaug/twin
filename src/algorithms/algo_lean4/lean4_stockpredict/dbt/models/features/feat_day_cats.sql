-- Long-format categorical view of each return day: one row per (day, feature).
-- The substrate for the chi-square association analysis (mart_feature_assoc):
-- return class vs seasonality (dow, month), vs vol state (the GARCH signal),
-- vs the previous day's class (transition association), vs motif flags.
with base as (
    select
        obs_id,
        ret_class,
        coalesce(vol_state, 'warmup')                             as vol_state,
        dow_name,
        'm' || lpad(month::varchar, 2, '0')                       as month_bucket,
        case when is_shock then 'shock' else 'no_shock' end       as shock_day,
        case when in_drawdown then 'drawdown' else 'normal' end   as dd_state,
        coalesce(lag(ret_class) over (partition by ticker order by d),
                 'none')                                          as prev_class,
        case when ret_sign = 1 then 'pos'
             when ret_sign = -1 then 'neg' else 'zero' end        as sign_band
    from {{ ref('feat_returns') }}
    where ret_class is not null
)

unpivot base
on ret_class, vol_state, dow_name, month_bucket, shock_day, dd_state,
   prev_class, sign_band
into name feature value value
